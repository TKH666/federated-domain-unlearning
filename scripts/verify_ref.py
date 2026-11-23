#!/usr/bin/env python
"""Domain-level unlearning verification (Sec. V-B, Algorithm 1 of the paper).

A proxy validation model (U-Net) perturbs the representative samples of the forgetting domain D_f so that the
original global model predicts the anchor class y_T, whereas reference models that never saw D_f do not:

  loss = CE(f_w(T(z)), y_T) + beta / |R| * sum_r -log(1 - p_r(y_T | T(z)))        (reference models R)
         + spec_beta * mean_x -log(1 - p_w(y_T | T(x)))                           (images x of the other domains)

The score of a model is the fraction of representative samples (true label != y_T) that it assigns to y_T
under the perturbation. Final setting of the paper: --ref fl --ref_init shared --beta 20 --spec_beta 20
--holdout_seed 8888 --hold_max 5 --max_chunks 30 --keep_best (see reproduce/queues/06_verifier.q).

Reference models (--ref):
  fl     : snapshots of a short FL training of the retained clients (--ref_rounds), started from the initial
           global model w(0) with --ref_init shared; the held-out model (--holdout_seed) is not used for
           training and checks the calibration (--hold_max); more seeds are added from --seed_pool if needed.
  local  : one model per retained client trained on its own data only.
  domain : no reference model; the perturbation of a D_f sample is transplanted onto retained-domain samples.
  none   : proxy trained against the original model only (ablation without reference models).
The evaluated Retrain model is never used to build the signal; it is only scored.

Output: <run_dir>/verify/<tag>/<target>/{proxy.pt, scores.json, log.txt}
"""
import argparse
import copy
import glob
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import Timer, add_cfg_args, cfg_from_args, cpu_sd, log, save_json, set_seed  # noqa: E402
from fdu.data import DOMAINS, FedData  # noqa: E402
from fdu.fl import accuracy, avg_local_update  # noqa: E402
from fdu.forget import select  # noqa: E402
from fdu.models import build_model, logits_of  # noqa: E402
from fdu.ours import clip_image, marker_scores, new_proxy, recalibrate_bn  # noqa: E402
from fdu.unlearn import fl_rounds  # noqa: E402


def frozen(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def get_refs(kind, cfg, data, t, a, lg, ref_seed):
    """local: one model per retained client on its own data. fl: snapshots of ONE short FL run over the
    retained clients (ref_seed), taken at the rounds listed in --ref_rounds (e.g. 5,10,20).
    --ref_init fresh: new random init (seed ref_seed); shared: the FL initial model theta_0 that the original
    training started from (full/weight_init.pt; the server broadcast it, so the verifier knows it). ref_seed then
    only changes data order / sampling."""
    sh = "_sh" if a.ref_init == "shared" else ""
    w0 = torch.load(os.path.join(cfg.run_dir(), "full", "weight_init.pt")) if sh else None
    keep = [c for c in range(data.num_users) if c != t]
    rdir = os.path.join(cfg.run_dir(), "verify", "refs", DOMAINS[cfg.dataset][t])
    os.makedirs(rdir, exist_ok=True)
    if kind == "local":
        paths = [os.path.join(rdir, f"local_ep{a.ref_ep}_s{ref_seed}{sh}.pt")]
    else:
        rounds = sorted(int(r) for r in str(a.ref_rounds).split(","))
        paths = [os.path.join(rdir, f"fl_r{r}_s{ref_seed}{sh}.pt") for r in rounds]
    if not all(os.path.exists(pth) for pth in paths):
        tm = Timer()
        loaders = data.train_loaders(seed=ref_seed)
        if kind == "local":
            sds = []
            for c in keep:
                set_seed(ref_seed + c)
                net = build_model(cfg.dataset, cfg.device)
                if w0 is not None:
                    net.load_state_dict(w0)
                avg_local_update(net, loaders[c], a.ref_ep, cfg.lr, cfg.momentum, cfg.device)
                sds.append(cpu_sd(net.state_dict()))
            torch.save(sds, paths[0])
        else:
            set_seed(ref_seed)
            net = build_model(cfg.dataset, cfg.device)
            if w0 is not None:
                net.load_state_dict(w0)
            protos = None
            for r in range(1, max(rounds) + 1):
                net, protos = fl_rounds(net, loaders, keep, 1, cfg.local_ep, cfg, protos=protos)
                if r in rounds:
                    torch.save([cpu_sd(net.state_dict())], paths[rounds.index(r)])
                    lg(f"  fl reference snapshot at round {r} ({tm():.0f}s)")
        lg(f"trained reference model(s) [{kind}] in {tm():.0f}s")
    refs = []
    te = data.test_loaders()
    for pth in paths:
        for sd in torch.load(pth):
            net = build_model(cfg.dataset, cfg.device)
            net.load_state_dict(sd)
            refs.append(frozen(net))
            lg(f"  ref {os.path.basename(pth)} test acc {[round(accuracy(net, ld, cfg.device), 1) for ld in te]}")
    return refs


@torch.no_grad()
def recalibrate_bn_mixed(model, loader, spec_it, anchor, device):
    """recalibrate_bn for a generator trained on mixed [D_f ; retained] batches: running statistics from the same
    mixture, so that eval mode matches training (the same statistics are then used for every input domain)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for x, _, _ in loader:
        xs, ys, _ = next(spec_it)
        xs = xs[ys != anchor]
        model(torch.cat([x, xs]).to(device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


def train_proxy_ref(orig, refs, retained_it, proxy, loader, eps, anchor, beta, device, a, clip, lg):
    opt = torch.optim.Adam(proxy.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    hist, acc, racc, chunks, epochs = [], 0.0, 100.0, 0, 0
    best, best_state, best_chunk = -1.0, None, 0
    # --amp: fp16 autocast + loss scaling for proxy training only (engineering; scoring stays fp32).
    # Disabled (default): ac() is a no-op, scaler.scale(l) is l, scaler.step(opt) is opt.step().
    amp = getattr(a, "amp", False)
    ac = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=amp)  # noqa: E731
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    bw = lambda l, **k: scaler.scale(l).backward(**k)  # noqa: E731
    while (acc <= a.threshold or racc > a.ref_max) and chunks < a.max_chunks:
        for _ in range(a.epochs_per_chunk):
            proxy.train()
            c_o = c_r = n = n_r = c_s = n_s = 0
            for x, _, _ in loader:
                x = x.to(device)
                xs = None
                if getattr(a, "spec_it", None) is not None:
                    xs, ys, _ = next(a.spec_it)
                    xs, ys = xs.to(device), ys.to(device)
                    xs = xs[ys != anchor]
                    if xs.size(0) == 0:
                        xs = None
                if xs is not None:
                    # ONE generator forward on [D_f batch ; retained-domain batch]: in train mode BatchNorm uses
                    # batch statistics, and a separate retained-only batch would let the generator recognise the
                    # domain from them (spec term satisfied in training, violated in eval mode; observed on Caltech)
                    with ac():
                        dd = proxy(torch.cat([x, xs])) * eps
                    d, ds = dd[:x.size(0)], dd[x.size(0):]
                else:
                    with ac():
                        d = proxy(x) * eps
                xt = clip_image(x + d, *clip)
                tgt = torch.full((x.size(0),), anchor, dtype=torch.long, device=device)
                opt.zero_grad()
                # backward one model at a time (retain the proxy graph only): peak memory = proxy + 1 model
                n_terms = 1 + len(refs) + (retained_it is not None) + (xs is not None)
                with ac():
                    out = logits_of(orig(xt))
                    l_o = ce(out, tgt)
                bw(l_o, retain_graph=n_terms > 1)
                c_o += (out.argmax(1) == tgt).sum().item()
                n += x.size(0)
                for i, R in enumerate(refs):
                    with ac():
                        o = logits_of(R(xt))
                    pr = F.softmax(o.float(), 1)[:, anchor]
                    lr_ = beta * (-torch.log1p(-pr.clamp(max=1 - 1e-6))).mean() / len(refs)
                    bw(lr_, retain_graph=(i < len(refs) - 1) or retained_it is not None or xs is not None)
                    c_r += (o.argmax(1) == anchor).sum().item()
                    n_r += x.size(0)
                if retained_it is not None:
                    xr, yr, _ = next(retained_it)
                    xr, yr = xr.to(device), yr.to(device)
                    b = min(xr.size(0), d.size(0))
                    with ac():
                        outr = logits_of(orig(clip_image(xr[:b] + d[:b], *clip)))
                    bw(beta * ce(outr.float(), yr[:b]), retain_graph=xs is not None)
                    keep = yr[:b] != anchor
                    c_r += (outr.argmax(1)[keep] == anchor).sum().item()
                    n_r += keep.sum().item()
                if xs is not None:
                    # domain specificity: the generator applied to RETAINED-domain images must not make the
                    # original model predict y_T (the marker may only use what Orig learned from D_f)
                    with ac():
                        os_ = logits_of(orig(clip_image(xs + ds, *clip)))
                    ps = F.softmax(os_.float(), 1)[:, anchor]
                    bw(a.spec_beta * (-torch.log1p(-ps.clamp(max=1 - 1e-6))).mean())
                    c_s += (os_.argmax(1) == anchor).sum().item()
                    n_s += xs.size(0)
                scaler.step(opt)
                scaler.update()
            acc, racc = 100.0 * c_o / n, (100.0 * c_r / n_r if n_r else 0.0)
            epochs += 1
            hist.append((acc, racc))
        chunks += 1
        lg(f"proxy chunk {chunks}: marker(orig, running) {acc:.2f}  marker(reference, running) {racc:.2f}"
           + (f"  orig flips on retained-domain images (running) {100.0 * c_s / max(n_s, 1):.2f}"
              if getattr(a, "spec_it", None) is not None else ""))
        if a.keep_best and racc <= a.ref_max and acc > best:
            best, best_chunk = acc, chunks
            best_state = {k: v.detach().cpu().clone() for k, v in proxy.state_dict().items()}
    if a.keep_best and best_state is not None and not (acc > a.threshold and racc <= a.ref_max) \
            and best_chunk != chunks:
        # threshold not reached within the budget: use the chunk with the highest running Orig marker among the
        # chunks whose reference marker <= ref_max (selection uses only Orig and the references, never Retrain)
        proxy.load_state_dict(best_state)
        lg(f"keep best: restored chunk {best_chunk} (marker(orig, running) {best:.2f})")
    return proxy, hist, epochs


def train_proxy_protect(orig, refs, proxy, loader, eps, anchor, beta, device, a, clip, lg):
    """refs -> y_T (generic, transferable attack); orig keeps the true label (needs D_f-specific features)."""
    opt = torch.optim.Adam(proxy.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    hist, keep_o, racc, chunks, epochs = [], 0.0, 0.0, 0, 0
    while (keep_o < a.orig_keep or racc < a.threshold) and chunks < a.max_chunks:
        for _ in range(a.epochs_per_chunk):
            proxy.train()
            c_o = c_r = n = 0
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                m = y != anchor
                if m.sum() == 0:
                    continue
                x, y = x[m], y[m]
                xt = clip_image(x + proxy(x) * eps, *clip)
                tgt = torch.full((x.size(0),), anchor, dtype=torch.long, device=device)
                out = logits_of(orig(xt))
                loss = ce(out, y)
                lr_ = 0.0
                for R in refs:
                    o = logits_of(R(xt))
                    lr_ = lr_ + ce(o, tgt)
                    c_r += (o.argmax(1) == tgt).sum().item()
                loss = loss + beta * lr_ / len(refs)
                c_o += (out.argmax(1) == y).sum().item()
                n += x.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()
            keep_o, racc = 100.0 * c_o / n, 100.0 * c_r / (n * len(refs))
            epochs += 1
            hist.append((keep_o, racc))
        chunks += 1
        lg(f"proxy chunk {chunks}: orig acc under T (running) {keep_o:.2f}  refs marker (running) {racc:.2f}")
    return proxy, hist, epochs


@torch.no_grad()
def marked_vec(net, proxy, loader, eps, anchor, device, clip):
    """bool vector over the non-anchor samples of an UNSHUFFLED loader: net predicts y_T on T(z)."""
    net.eval()
    proxy.eval()
    out = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        x = x[y != anchor]
        if x.size(0) == 0:
            continue
        out.append(logits_of(net(clip_image(x + proxy(x) * eps, *clip))).argmax(1) == anchor)
    return torch.cat(out).cpu()


def consensus(net, proxies, loader, eps, anchor, device, clip):
    """K-proxy consensus marker: a sample counts only if EVERY proxy makes net predict y_T on it."""
    vs = torch.stack([marked_vec(net, pr, loader, eps, anchor, device, clip) for pr in proxies])
    return {"marker": 100.0 * vs.all(0).float().mean().item(),
            "marker_each": [100.0 * v.float().mean().item() for v in vs],
            "marker_major": 100.0 * (vs.float().mean(0) > 0.5).float().mean().item()}


@torch.no_grad()
def perturbed_acc(net, proxy, loader, eps, anchor, device, clip):
    net.eval()
    proxy.eval()
    c = n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        m = y != anchor
        x, y = x[m], y[m]
        if x.size(0) == 0:
            continue
        c += (logits_of(net(clip_image(x + proxy(x) * eps, *clip))).argmax(1) == y).sum().item()
        n += x.size(0)
    return 100.0 * c / max(n, 1)


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--ref", default="local", choices=["local", "fl", "local+fl", "domain", "none"])
    p.add_argument("--objective", default="flip", choices=["flip", "protect"],
                   help="flip: orig->y_T, refs away from y_T. protect: refs->y_T, orig keeps the true label "
                        "(score = accuracy under T(z); a model without D_f knowledge behaves like the refs)")
    p.add_argument("--orig_keep", type=float, default=90.0, help="protect: stop once orig acc under T >= this")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ref_ep", type=int, default=20)
    p.add_argument("--ref_rounds", default="5", help="fl: comma list of snapshot rounds, e.g. 5,10,20")
    p.add_argument("--ref_seed", default="7777", help="comma list: one reference run per seed")
    p.add_argument("--ref_init", default="fresh", choices=["fresh", "shared"],
                   help="reference models start from a fresh init or from the original FL initial model theta_0")
    p.add_argument("--ref_max", type=float, default=5.0, help="stop only once reference marker <= this")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--select", default="top", choices=["top", "bottom", "random", "all"],
                   help="E5 ablation: which samples of D_f build the marker (top = most forgetting events, paper)")
    p.add_argument("--anchor", type=int, default=9)
    p.add_argument("--threshold", type=float, default=95.0)
    p.add_argument("--epochs_per_chunk", type=int, default=20)
    p.add_argument("--max_chunks", type=int, default=10)
    p.add_argument("--holdout_seed", default="", help="extra FL reference(s) (comma list of seeds, rounds = max of "
                   "--ref_rounds) NOT used in proxy training; scored as hold<i> = self-check of the calibration "
                   "without Retrain (used to choose beta / eps)")
    p.add_argument("--hold_max", type=float, default=-1.0, help=">=0: add reference seeds from --seed_pool (one "
                   "at a time, proxy re-trained) until every held-out reference has marker <= hold_max")
    p.add_argument("--seed_pool", default="1111,2222,3333,4444,5555")
    p.add_argument("--proxy_bs", type=int, default=0, help="proxy-training batch size (0 = local_bs)")
    p.add_argument("--n_proxy", type=int, default=1, help="K>1: train K proxies (different init / data order; "
                   "with --proxy_refs split also different reference seeds) and score the CONSENSUS marker "
                   "(a sample counts only if all K proxies flip it to y_T)")
    p.add_argument("--proxy_refs", default="all", choices=["all", "split", "loo"],
                   help="split: proxy k is calibrated only on the references of reference seed k (round robin); "
                   "loo: proxy k on all reference seeds except seed k (leave one seed out)")
    p.add_argument("--vtag", default="")
    p.add_argument("--spec_beta", type=float, default=0.0, help=">0: add spec_beta * -log(1 - p_orig(y_T | T(z_r))) "
                   "on retained-domain images z_r (the noise must be specific to D_f, not an Orig-wide fingerprint)")
    p.add_argument("--init_proxy", default="", help="warm start: proxy state dict to start from (path relative to "
                   "<run_dir>/verify/ or absolute), e.g. the unconstrained proxy, then train with --spec_beta (two-stage)")
    p.add_argument("--keep_best", action="store_true", help="if the threshold is not reached within --max_chunks, "
                   "restore the chunk with the highest running Orig marker (reference marker <= ref_max)")
    p.add_argument("--amp", action="store_true", help="fp16 autocast + loss scaling in proxy training (speed only; output dir "
                   "gets _amp; scoring in fp32)")
    p.add_argument("--eval_bs", type=int, default=128, help="evaluation batch size (eval mode: does not change scores)")
    p.add_argument("--score_only", action="store_true", help="do not train: load the proxies saved in the output "
                   "directory by an earlier run with the same arguments and only score (e.g. after an OOM in scoring)")
    p.add_argument("--extra_null", default="", help="DIAGNOSTIC: comma list of files in verify/refs/<target>/ "
                   "(reference models NOT used here) to score as null<i>")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    names = DOMAINS[cfg.dataset]
    t = names.index(a.target)
    tag = ("prot_" if a.objective == "protect" else "") + f"ref_{a.ref}" + \
        (f"_r{str(a.ref_rounds).replace(',', '-')}" if "fl" in a.ref else "") + (f"_s{str(a.ref_seed).replace(',', '-')}" if str(a.ref_seed) != "7777" else "") + ("_sh" if a.ref_init == "shared" else "") + f"_eps{a.eps}_b{a.beta}" + (f"_auto{a.hold_max:g}" if a.hold_max >= 0 else "") + (f"_mspec{a.spec_beta:g}" if a.spec_beta > 0 else "") \
        + (f"_K{a.n_proxy}" + (a.proxy_refs if a.proxy_refs != "all" else "") if a.n_proxy > 1 else "") \
        + ("_warm" if a.init_proxy else "") + ("_kb" if a.keep_best else "") + ("_amp" if a.amp else "") \
        + (f"_sel{a.select}" if a.select != "top" else "") + (f"_lam{a.lam:g}" if a.lam != 0.5 else "") \
        + (f"_y{a.anchor}" if a.anchor != 9 else "") + (f"_{a.vtag}" if a.vtag else "")
    out = os.path.join(cfg.run_dir(), "verify", tag, names[t])
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    lg = lambda s: log(s, fh)  # noqa: E731
    clip = (-1.0, 1.0)
    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    lg(f"target={names[t]} ref={a.ref} eps={a.eps} beta={a.beta} cfg={cfg}")

    refs, retained_it, groups = [], None, []
    if a.ref in ("local", "fl", "local+fl"):
        for seed in [int(x) for x in str(a.ref_seed).split(",")]:
            g = []
            for kind in a.ref.split("+"):
                g += get_refs(kind, cfg, data, t, a, lg, seed)
            refs += g
            groups.append(g)
    elif a.ref == "domain":
        keep = [c for c in range(data.num_users) if c != t]
        lds = data.train_loaders(seed=a.ref_seed)
        def endless(loaders):  # re-iterates (itertools.cycle would cache every batch in memory)
            while True:
                for batches in itertools.zip_longest(*loaders):
                    for b in batches:
                        if b is not None:
                            yield b
        retained_it = endless([lds[c] for c in keep])
    holds = []
    if a.holdout_seed:
        ha = argparse.Namespace(**{**vars(a), "ref_rounds": str(max(int(r) for r in str(a.ref_rounds).split(",")))})
        for hs in [int(x) for x in a.holdout_seed.split(",")]:
            holds += get_refs("fl", cfg, data, t, ha, lg, hs)
    a.spec_it = None
    if a.spec_beta > 0:
        keep_c = [c for c in range(data.num_users) if c != t]
        slds = {c: data._loader(data.train[c], True, cfg.seed + 7 + c, bs=a.proxy_bs or None) for c in keep_c}

        def endless_spec(loaders):
            while True:
                for batches in itertools.zip_longest(*loaders):
                    for b in batches:
                        if b is not None:
                            yield b
        a.spec_it = endless_spec([slds[c] for c in keep_c])
    set_seed(cfg.seed)

    rec = np.load(os.path.join(cfg.run_dir(), "full", "forget_records.npz"))[f"arr_{t}"]
    rep = select(rec, a.lam, a.select, seed=cfg.seed)
    train_ld = data.subset_loader(t, rep, split="train", shuffle=True, seed=cfg.seed, bs=a.proxy_bs or None)
    rep_ld = data.subset_loader(t, rep, split="train_eval", bs=a.eval_bs)
    test_ld = data.test_loaders(bs=a.eval_bs)[t]

    orig = build_model(cfg.dataset, cfg.device)
    orig.load_state_dict(torch.load(os.path.join(cfg.run_dir(), "full", "weight_global.pt")))
    orig = frozen(orig)
    def fit_one(refs, k=0):
        proxy = new_proxy("unet", cfg.device, cfg.seed + 1000 * k)
        if a.init_proxy:
            ip = a.init_proxy if os.path.isabs(a.init_proxy) else os.path.join(cfg.run_dir(), "verify", a.init_proxy)
            proxy.load_state_dict(torch.load(ip))
            lg(f"warm start from {ip}")
        ld = train_ld if k == 0 else data.subset_loader(t, rep, split="train", shuffle=True, seed=cfg.seed + k,
                                                       bs=a.proxy_bs or None)
        if a.n_proxy > 1:
            lg(f"-- proxy {k + 1}/{a.n_proxy} ({len(refs)} reference model(s))")
        if a.objective == "protect":
            assert refs, "protect needs reference models"
            proxy, hist, ep = train_proxy_protect(orig, refs, proxy, ld, a.eps, a.anchor, a.beta, cfg.device,
                                                  a, clip, lg)
        else:
            proxy, hist, ep = train_proxy_ref(orig, refs, retained_it, proxy, ld, a.eps, a.anchor, a.beta,
                                              cfg.device, a, clip, lg)
        if a.spec_it is not None:
            recalibrate_bn_mixed(proxy, ld, a.spec_it, a.anchor, cfg.device)
        else:
            recalibrate_bn(proxy, ld, cfg.device)
        return proxy, hist, ep

    def fit(refs):
        if a.n_proxy == 1:
            return fit_one(refs)
        ps, hs, eps_ = [], [], []
        for k in range(a.n_proxy):
            if a.proxy_refs == "split" and groups:
                rk = groups[k % len(groups)]
            elif a.proxy_refs == "loo" and len(groups) > 1:
                rk = [r for j, g in enumerate(groups) if j != k % len(groups) for r in g]
            else:
                rk = refs
            pk, hk, ek = fit_one(rk, k)
            ps.append(pk)
            hs.append(hk)
            eps_.append(ek)
        return ps, hs, eps_

    def mscore(net, proxy, loader):
        if a.n_proxy == 1:
            return marker_scores(net, proxy, loader, a.eps, a.anchor, cfg.device, clip)["marker"]
        return consensus(net, proxy, loader, a.eps, a.anchor, cfg.device, clip)["marker"]

    tm = Timer()
    auto = []
    if a.score_only:
        loaded = []
        for k in range(a.n_proxy):
            pr = new_proxy("unet", cfg.device, cfg.seed + 1000 * k)
            pr.load_state_dict(torch.load(os.path.join(out, "proxy.pt" if k == 0 else f"proxy{k}.pt")))
            pr.eval()
            loaded.append(pr)
        proxy, hist, ep = (loaded if a.n_proxy > 1 else loaded[0]), None, None
        lg(f"[score_only] loaded {len(loaded)} saved proxy(ies) from {out}")
    else:
        proxy, hist, ep = fit(refs)
    if a.hold_max >= 0 and holds and not a.score_only:
        # adaptive number of reference seeds (Retrain-free): add one more reference run while the held-out
        # reference model still responds to the marker above hold_max
        pool = [int(x) for x in a.seed_pool.split(",") if x and str(x) not in str(a.ref_seed).split(",")]
        # DIAGNOSTIC ONLY (never used by the stopping rule): Retrain's marker after every step, so that one run
        # with a strict threshold also shows what a looser threshold would have given
        diag_p = os.path.join(cfg.run_dir(), "unlearn", "retrain", names[t], "weight.pt")
        diag = None
        if os.path.exists(diag_p):
            diag = build_model(cfg.dataset, cfg.device)
            diag.load_state_dict(torch.load(diag_p))
        while True:
            hm = max(mscore(h, proxy, rep_ld) for h in holds)
            n_seeds = len(str(a.ref_seed).split(",")) + len(auto)
            dm = mscore(diag, proxy, rep_ld) if diag else None
            lg(f"[auto] {n_seeds} reference seed(s): held-out marker {hm:.2f} (threshold {a.hold_max})"
               + (f"   [diagnostic only] retrain marker {dm:.2f}" if dm is not None else ""))
            auto.append({"n_seeds": n_seeds, "hold_marker": hm, "sec": tm(), "diag_retrain_marker": dm})
            if hm <= a.hold_max or not pool:
                break
            s_new = pool.pop(0)
            g = []
            for kind in a.ref.split("+"):
                g += get_refs(kind, cfg, data, t, a, lg, s_new)
            refs += g
            groups.append(g)
            proxy, hist, ep = fit(refs)
    meta = {"proxy_train_sec": tm(), "proxy_epochs": ep, "proxy_hist": hist, "n_rep": int(len(rep)), "auto": auto,
            "n_proxy": a.n_proxy, "proxy_refs": a.proxy_refs, "score_only": a.score_only}
    proxies = proxy if a.n_proxy > 1 else [proxy]
    for i, pr in enumerate(proxies if not a.score_only else []):
        torch.save({k: v.cpu() for k, v in pr.state_dict().items()},
                   os.path.join(out, "proxy.pt" if i == 0 else f"proxy{i}.pt"))
    proxy = proxies[0]

    models = {"Orig": os.path.join(cfg.run_dir(), "full", "weight_global.pt")}
    for d in sorted(glob.glob(os.path.join(cfg.run_dir(), "unlearn", "*", names[t], "weight.pt"))):
        models[d.split(os.sep)[-3]] = d
    net = build_model(cfg.dataset, cfg.device)
    scores = {}
    # domain-specificity check in EVAL mode (as deployed): the same proxy on <=200 images of every retained domain
    xdom_ld = {}
    for c in [c for c in range(data.num_users) if c != t]:
        idx = np.random.RandomState(cfg.seed + c).permutation(len(data.train_eval[c]))[:200]
        xdom_ld[names[c]] = data.subset_loader(c, idx, split="train_eval", bs=a.eval_bs)
    extra = [(f"ref{i}", r) for i, r in enumerate(refs)] + [(f"hold{i}", h) for i, h in enumerate(holds)]
    for e in [x for x in a.extra_null.split(",") if x]:
        nm = build_model(cfg.dataset, cfg.device)
        sd = torch.load(os.path.join(cfg.run_dir(), "verify", "refs", names[t], e))
        nm.load_state_dict(sd[0] if isinstance(sd, list) else sd)
        extra.append((f"null:{e.replace('.pt', '')}", frozen(nm)))
    for k, path in list(models.items()) + extra:
        m = net if isinstance(path, str) else path
        if not isinstance(path, str):
            path = None
        if path is not None:
            m.load_state_dict(torch.load(path))
        rs = marker_scores(m, proxy, rep_ld, a.eps, a.anchor, cfg.device, clip)
        ts = marker_scores(m, proxy, test_ld, a.eps, a.anchor, cfg.device, clip)
        rs["pert_acc"] = perturbed_acc(m, proxy, rep_ld, a.eps, a.anchor, cfg.device, clip)
        ts["pert_acc"] = perturbed_acc(m, proxy, test_ld, a.eps, a.anchor, cfg.device, clip)
        rs["xdom"] = {dn: marker_scores(m, proxy, ld_, a.eps, a.anchor, cfg.device, clip)["marker"]
                      for dn, ld_ in xdom_ld.items()}
        extra_s = "  other domains " + " ".join(f"{v:5.1f}" for v in rs["xdom"].values())
        if a.n_proxy > 1:
            # "marker" = K-proxy consensus; the single-proxy marker of proxy 0 is kept as "marker_p0"
            rs["marker_p0"], ts["marker_p0"] = rs["marker"], ts["marker"]
            rs.update(consensus(m, proxies, rep_ld, a.eps, a.anchor, cfg.device, clip))
            ts.update(consensus(m, proxies, test_ld, a.eps, a.anchor, cfg.device, clip))
            extra_s += f"  each {' '.join(f'{v:5.1f}' for v in rs['marker_each'])}  majority {rs['marker_major']:5.1f}"
        scores[k] = {"rep": rs, "test": ts}
        lg(f"{k:>16s}: marker(rep) {rs['marker']:6.2f}  acc-under-T(rep) {rs['pert_acc']:6.2f}  "
           f"acc-under-T(test) {ts['pert_acc']:6.2f}  clean-acc(rep) {rs['clean_acc']:6.2f}" + extra_s)
    a.spec_it = None
    save_json({"target": names[t], "args": {k: v for k, v in vars(a).items() if k != "spec_it"}, "cfg": cfg.__dict__, "proxy_meta": meta, "scores": scores},
              os.path.join(out, "scores.json"))


if __name__ == "__main__":
    main()
