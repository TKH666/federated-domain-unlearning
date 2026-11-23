#!/usr/bin/env python
"""Exploratory variant (not used in the paper): held-out-calibrated marker, so that Retrain ~ 0 holds by
construction.

Retrain never saw the forgetting domain D_f, so for Retrain a random member of D_f and a random
same-domain held-out sample (never trained on by any model) are exchangeable. Any statistic of the form
    Delta(U) = s_U(members) - s_U(held-out)
is therefore ~0 for Retrain in expectation, without needing the Retrain model at verification time.

Cross-fitting keeps the proxy's own memorisation out of the score:
  members  : random half A_pool -> proxy training on top-lambda forgetting events of A_pool (Sec. V)
             other half B       -> scoring only
  held-out : test split; mode=plain   -> all of it is H_B (scoring only)
                         mode=contrast-> half H_A used in proxy training (keep true label on the original
                                         model), other half H_B for scoring
Score per model: marker(B) - marker(H_B) and clean-acc(B) - clean-acc(H_B), with binomial standard errors.

Output: <run_dir>/verify/calib_<mode>_eps<eps>[_<vtag>]/<target>/{proxy.pt, scores.json, log.txt}
"""
import argparse
import glob
import itertools
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import Timer, add_cfg_args, cfg_from_args, log, save_json, set_seed  # noqa: E402
from fdu.data import DOMAINS, FedData, _transforms  # noqa: E402
from fdu.forget import select  # noqa: E402
from fdu.models import build_model, logits_of  # noqa: E402
from fdu.ours import clip_image, marker_scores, new_proxy, recalibrate_bn  # noqa: E402


def train_proxy_calib(net, proxy, mem_ld, ho_ld, eps, anchor, device, w_ho, epochs_per_chunk, max_chunks,
                      threshold, clip, log_fn):
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(proxy.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    hist, acc, chunks, epochs = [], 0.0, 0, 0
    while acc <= threshold and chunks < max_chunks:
        for _ in range(epochs_per_chunk):
            proxy.train()
            c_m = n_m = c_h = n_h = 0
            ho_it = itertools.cycle(ho_ld) if ho_ld is not None else None
            for x, _, _ in mem_ld:
                x = x.to(device)
                out = logits_of(net(clip_image(x + proxy(x) * eps, *clip)))
                tgt = torch.full((x.size(0),), anchor, dtype=torch.long, device=device)
                loss = ce(out, tgt)
                c_m += (out.argmax(1) == tgt).sum().item()
                n_m += x.size(0)
                if ho_it is not None:
                    xh, yh, _ = next(ho_it)
                    xh, yh = xh.to(device), yh.to(device)
                    oh = logits_of(net(clip_image(xh + proxy(xh) * eps, *clip)))
                    loss = loss + w_ho * ce(oh, yh)          # held-out: keep the true label
                    keep = yh != anchor
                    c_h += (oh.argmax(1)[keep] == anchor).sum().item()
                    n_h += keep.sum().item()
                opt.zero_grad()
                loss.backward()
                opt.step()
            acc = 100.0 * c_m / n_m
            epochs += 1
            hist.append((acc, 100.0 * c_h / max(n_h, 1)))
        chunks += 1
        log_fn(f"proxy chunk {chunks}: marker(members A, running) {acc:.2f}  marker(held-out H_A) {hist[-1][1]:.2f}")
    return proxy, hist, epochs


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--mode", default="plain", choices=["plain", "contrast"])
    p.add_argument("--w_ho", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--anchor", type=int, default=9)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=95.0)
    p.add_argument("--epochs_per_chunk", type=int, default=20)
    p.add_argument("--max_chunks", type=int, default=10)
    p.add_argument("--vtag", default="")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    names = DOMAINS[cfg.dataset]
    t = names.index(a.target)
    tag = f"calib_{a.mode}_eps{a.eps}" + (f"_w{a.w_ho}" if a.mode == "contrast" and a.w_ho != 1.0 else "") \
        + (f"_{a.vtag}" if a.vtag else "")
    out = os.path.join(cfg.run_dir(), "verify", tag, names[t])
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    lg = lambda s: log(s, fh)  # noqa: E731
    set_seed(cfg.seed)
    clip = (-1.0, 1.0)  # lineage clamp, as verify_ours.py default
    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)

    # ---- splits ----
    n_mem, n_ho = len(data.train[t]), len(data.test[t])
    rs = np.random.RandomState(a.split_seed)
    perm = rs.permutation(n_mem)
    a_pool, b_idx = np.sort(perm[: n_mem // 2]), np.sort(perm[n_mem // 2:])
    rec = np.load(os.path.join(cfg.run_dir(), "full", "forget_records.npz"))[f"arr_{t}"]
    a_idx = a_pool[select(rec[:, a_pool], a.lam, "top")]
    hperm = rs.permutation(n_ho)
    if a.mode == "contrast":
        ha_idx, hb_idx = np.sort(hperm[: n_ho // 2]), np.sort(hperm[n_ho // 2:])
    else:
        ha_idx, hb_idx = None, np.arange(n_ho)
    lg(f"target={names[t]} mode={a.mode} eps={a.eps} w_ho={a.w_ho} |A|={len(a_idx)} |B|={len(b_idx)} "
       f"|H_A|={0 if ha_idx is None else len(ha_idx)} |H_B|={len(hb_idx)} cfg={cfg}")

    mem_ld = data.subset_loader(t, a_idx, split="train", shuffle=True, seed=cfg.seed)  # augmented
    ho_ld = None
    if ha_idx is not None:
        tr_t = _transforms(cfg.dataset, names[t])[0]
        te = data.test[t]
        te_aug = te.with_transform(tr_t) if hasattr(te, "with_transform") else te
        ho_ld = data._loader(Subset(te_aug, list(ha_idx)), True, cfg.seed)
    a_eval = data.subset_loader(t, a_idx, split="train_eval", bs=128)
    b_eval = data.subset_loader(t, b_idx, split="train_eval", bs=128)
    hb_eval = data._loader(Subset(data.test[t], list(hb_idx)), False, bs=128)

    orig = build_model(cfg.dataset, cfg.device)
    orig.load_state_dict(torch.load(os.path.join(cfg.run_dir(), "full", "weight_global.pt")))
    proxy = new_proxy("unet", cfg.device, cfg.seed)
    tm = Timer()
    proxy, hist, ep = train_proxy_calib(orig, proxy, mem_ld, ho_ld, a.eps, a.anchor, cfg.device, a.w_ho,
                                        a.epochs_per_chunk, a.max_chunks, a.threshold, clip, lg)
    recalibrate_bn(proxy, mem_ld, cfg.device)  # same statistics for B and H_B
    meta = {"proxy_train_sec": tm(), "proxy_epochs": ep, "proxy_hist": hist,
            "n_A": int(len(a_idx)), "n_B": int(len(b_idx)), "n_HB": int(len(hb_idx))}
    torch.save({k: v.cpu() for k, v in proxy.state_dict().items()}, os.path.join(out, "proxy.pt"))

    models = {"Orig": os.path.join(cfg.run_dir(), "full", "weight_global.pt")}
    for d in sorted(glob.glob(os.path.join(cfg.run_dir(), "unlearn", "*", names[t], "weight.pt"))):
        models[d.split(os.sep)[-3]] = d
    net = build_model(cfg.dataset, cfg.device)
    scores = {}

    def se(p1, n1, p2, n2):
        p1, p2 = p1 / 100, p2 / 100
        return 100 * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    for k, path in models.items():
        net.load_state_dict(torch.load(path))
        sA = marker_scores(net, proxy, a_eval, a.eps, a.anchor, cfg.device, clip)
        sB = marker_scores(net, proxy, b_eval, a.eps, a.anchor, cfg.device, clip)
        sH = marker_scores(net, proxy, hb_eval, a.eps, a.anchor, cfg.device, clip)
        d_m, d_c = sB["marker"] - sH["marker"], sB["clean_acc"] - sH["clean_acc"]
        e_m = se(sB["marker"], sB["n"], sH["marker"], sH["n"])
        e_c = se(sB["clean_acc"], sB["n"], sH["clean_acc"], sH["n"])
        scores[k] = {"A": sA, "B": sB, "HB": sH, "d_marker": d_m, "se_marker": e_m, "d_clean": d_c, "se_clean": e_c}
        lg(f"{k:>16s}: marker A {sA['marker']:6.2f} B {sB['marker']:6.2f} H {sH['marker']:6.2f} "
           f"| dMarker {d_m:+6.2f}±{e_m:4.2f} | clean B {sB['clean_acc']:6.2f} H {sH['clean_acc']:6.2f} "
           f"dClean {d_c:+6.2f}±{e_c:4.2f}")
    save_json({"target": names[t], "args": vars(a), "cfg": cfg.__dict__, "proxy_meta": meta, "scores": scores},
              os.path.join(out, "scores.json"))


if __name__ == "__main__":
    main()
