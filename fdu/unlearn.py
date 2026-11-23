"""Federated unlearning methods.

Existing baselines are ported from the authors' original code so that the new
experiments use the same implementation as the manuscript:
  retrain   : bl0_retrain.py
  rr        : bl1_rapid_retrain.py   (round 0 FPL/SGD, then AdaHessian local updates)
  fe        : bl2_FedEraser.py       (history-based calibration, all state-dict entries)
  pga       : bl3_increase_loss.py   (projected gradient ascent [13] + 1-epoch recovery rounds)
  cp        : bl4_class_discriminative_pruning.py (TF-IDF channel pruning, R=45, + fine-tuning)
Recent methods:
  fedosd    : FedOSD, AAAI'25 (official zibinpan/FedOSD: full-batch gradients, unlearning CE,
              orthogonal steepest descent; then orthogonalized recovery)
  not       : NoT, CVPR'25 (negate first-layer weights, then fine-tune on retained clients)
  fedunran  : FedUNRAN, IEEE BigData'24 (official alessiomora/FedUNRAN: local training on random
              labels with Adam + early stopping, then weighted FedAvg recovery rounds)
All methods operate on the same original model (<run_dir>/full) and return a state dict.
"""
import copy
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fdu import _ada_hessian
from fdu.common import cpu_sd, log
from fdu.fl import accuracy, fedavg, local_update, proto_aggregation
from fdu.models import build_model, build_pruning_model, logits_of

DEFAULTS = dict(
    unlearn_rounds=50,          # original --unlearn_epoch default (Retrain / RR / FE / PGA / CP)
    rr_chunk=16,                # AdaHessian micro-batch for VGG16 (memory); digits use full batches
    fe_ratio=0.5,               # FedEraser calibration local epochs = ceil(E * 0.5)
    fe_old_gm="start",          # oldGM_t = global at the start of round t (FedEraser paper); 'after' = original code
    fe_bn="avg",                # BN running stats: mean of the calibration clients (default; FedEraser defines no BN rule). "formula": calibrate them like weights (the original code) -> negative running_var, collapse. "recal": after each calibrated step every retained client re-estimates the running stats of the NEW global model on its data (one forward pass, no update) and the server averages them (fixes the weight/buffer mismatch of "avg")
    fe_eval_every=0,            # >0: log test accuracy of the calibrated global model every k rounds (diagnostics)
    pga_local_unlearn_epochs=10, pga_clip=5.0, pga_dist_stop=20.0, pga_recovery_ep=1,
    cp_prune_ratio=45, cp_ft_ep=10,
    fedosd_R=50, fedosd_UR=25, fedosd_lr=0.01, fedosd_r_lr=0.01, fedosd_decay=0.999, fedosd_clip=-1.0,
    not_rounds=10, not_ep=1, not_layers="0",
    unran_lr=1e-4, unran_epochs=10, unran_bs=32, unran_patience=3, unran_rounds=10, unran_ep=1,
)


def _retained(n, target):
    return [c for c in range(n) if c != target]


def fl_rounds(net, loaders, clients, rounds, local_ep, cfg, protos=None, log_fn=None, weights=None):
    """Plain federated rounds of the backbone (FPL or FedAvg) over `clients`."""
    protos = protos if protos is not None else []
    local_protos = {}
    sub = SimpleNamespace(**{**cfg.__dict__, "local_ep": local_ep})
    for r in range(rounds):
        ws = []
        for c in clients:
            m = copy.deepcopy(net)
            _, w, p, _ = local_update(cfg.backbone, m, loaders[c], protos, sub)
            ws.append(cpu_sd(w))
            local_protos[c] = p
            del m
        if cfg.backbone == "fpl":
            protos = proto_aggregation(local_protos)
        net.load_state_dict(weighted_avg(ws, weights) if weights is not None else fedavg(ws))
        if log_fn:
            log_fn(r)
    return net, protos


def weighted_avg(ws, weights):
    tot = float(sum(weights))
    out = {}
    for k in ws[0]:
        if ws[0][k].is_floating_point():
            out[k] = sum(w[k] * (n / tot) for w, n in zip(ws, weights))
        else:
            out[k] = ws[0][k].clone()
    return out


# ---------------------------------------------------------------------------
def retrain(cfg, hp, data, full, target, loaders):
    net = build_model(cfg.dataset, cfg.device)
    net.load_state_dict(full["init"])
    net, _ = fl_rounds(net, loaders, _retained(data.num_users, target), hp["unlearn_rounds"], cfg.local_ep, cfg)
    return net.state_dict()


def rapid_retrain(cfg, hp, data, full, target, loaders):
    """bl1: from init; round 0 = FPL/SGD local update, rounds>=1 = AdaHessian local update
    (DomainClientUpdate_Hesian_record, lr 1e-3); prototypes from round 0 only (as in the original code)."""
    net = build_model(cfg.dataset, cfg.device)
    net.load_state_dict(full["init"])
    keep = _retained(data.num_users, target)
    protos, local_protos = [], {}
    for r in range(hp["unlearn_rounds"]):
        ws = []
        for c in keep:
            m = copy.deepcopy(net)
            if r == 0:
                _, w, p, _ = local_update(cfg.backbone, m, loaders[c], protos, cfg)
                local_protos[c] = p
            else:
                w = _adahessian_update(m, loaders[c], cfg.local_ep, cfg.device,
                                       chunk=hp['rr_chunk'] if cfg.dataset != 'digits' else 0)
            ws.append(cpu_sd(w))
            del m
        if cfg.backbone == "fpl" and local_protos:
            protos = proto_aggregation(local_protos)
        net.load_state_dict(fedavg(ws))
        if r % 5 == 0 or r == hp["unlearn_rounds"] - 1:
            log(f"[RR] round {r + 1}/{hp['unlearn_rounds']} done")
    return net.state_dict()


def _adahessian_update(net, loader, local_ep, device, chunk=0):
    """original DomainClientUpdate_Hesian_record (AdaHessian, lr=1e-3, Hutchinson trace).
    chunk>0 (memory adaptation for VGG16@224 on 24-32GB GPUs): the batch gradient and the
    Hessian-vector product H z are accumulated over micro-batches with the same Rademacher z; both
    are sums over samples, so the result equals the full-batch quantity up to BatchNorm batch stats."""
    net.train()
    opt = _ada_hessian.AdaHessian(net.parameters())
    ce = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    for _ in range(local_ep):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)  # torch>=2 default; grads below are graph tensors
            if not chunk or x.size(0) <= chunk:
                loss = ce(logits_of(net(x)), y)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                for p, g in zip(params, grads):
                    p.grad = g
                opt.step()
                # p.grad holds a create_graph tensor -> reference cycle param<->graph that is never freed
                # (~one model copy + optimizer state per client update). step() has consumed it: release.
                for p in params:
                    p.grad = None
                del grads, loss
                continue
            # --- chunked path: replicate AdaHessian.step with accumulated g and H z ---
            gen = opt.generator
            if gen.device != params[0].device:
                gen = opt.generator = torch.Generator(params[0].device).manual_seed(2147483647)
            for p in params:
                opt.state[p]["hessian step"] += 1
            zs = [torch.randint(0, 2, p.size(), generator=gen, device=p.device) * 2.0 - 1.0 for p in params]
            g_acc = [torch.zeros_like(p) for p in params]
            hz_acc = [torch.zeros_like(p) for p in params]
            n = x.size(0)
            for s0 in range(0, n, chunk):
                xc, yc = x[s0:s0 + chunk], y[s0:s0 + chunk]
                lc = ce(logits_of(net(xc)), yc) * (xc.size(0) / n)
                gc = torch.autograd.grad(lc, params, create_graph=True)
                hz = torch.autograd.grad(gc, params, grad_outputs=zs)
                for i in range(len(params)):
                    g_acc[i] += gc[i].detach()
                    hz_acc[i] += hz[i]
                del lc, gc, hz
            with torch.no_grad():
                for p, g, hzi, z in zip(params, g_acc, hz_acc, zs):
                    p.grad = g
                    p.hess = hzi * z
                _adahessian_apply(opt)
    return net.state_dict()


@torch.no_grad()
def _adahessian_apply(opt):
    """The update part of AdaHessian.step (after the Hessian diagonal estimate is set)."""
    for group in opt.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            p.mul_(1 - group["lr"] * group["weight_decay"])
            state = opt.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)
            exp_avg, exp_h = state["exp_avg"], state["exp_hessian_diag_sq"]
            beta1, beta2 = group["betas"]
            state["step"] += 1
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_h.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            denom = (exp_h / bc2).pow_(group["hessian_power"] / 2).add_(group["eps"])
            p.addcdiv_(exp_avg, denom, value=-group["lr"] / bc1)


def federaser(cfg, hp, data, full, target, loaders):
    """bl2: FedEraser with stored history (client models of round i, global model after round i)."""
    hdir = os.path.join(full["dir"], "history")
    keep = _retained(data.num_users, target)
    h0 = torch.load(os.path.join(hdir, "round_000.pt"))
    net = build_model(cfg.dataset, cfg.device)
    net.load_state_dict(fedavg([h0["locals"][c] for c in keep]))
    cal_ep = int(np.ceil(cfg.local_ep * hp["fe_ratio"]))
    sub = SimpleNamespace(**{**cfg.__dict__, "local_ep": cal_ep})
    protos, local_protos = [], {}
    n_neg = 0
    for i in range(1, min(hp["unlearn_rounds"], cfg.rounds)):
        new_cms = []
        for c in keep:
            m = copy.deepcopy(net)
            _, w, p, _ = local_update(cfg.backbone, m, loaders[c], protos, sub)
            new_cms.append(cpu_sd(w))
            local_protos[c] = p
            del m
        if cfg.backbone == "fpl":
            protos = proto_aggregation(local_protos)
        h = torch.load(os.path.join(hdir, f"round_{i:03d}.pt"))
        old_cms = [h["locals"][c] for c in keep]
        new_gm = cpu_sd(net.state_dict())
        if hp["fe_old_gm"] == "start":
            # FedEraser (Liu et al., IWQoS'21): oldGM_t is the global model the clients STARTED round t
            # from, i.e. the global model after round t-1.
            old_gm = torch.load(os.path.join(hdir, f"round_{i - 1:03d}.pt"))["global_after"]
        else:  # "after": original bl2 indexing (global after round t; yields near-zero step lengths)
            old_gm = h["global_after"]
        sd, neg = _fe_step(old_cms, new_cms, old_gm, new_gm, bn="avg" if hp["fe_bn"] == "recal" else hp["fe_bn"])
        n_neg += neg
        net.load_state_dict(sd)
        if hp["fe_bn"] == "recal":
            _recal_bn_fed(net, [loaders[c] for c in keep], cfg.device)
        if hp["fe_eval_every"] and (i % hp["fe_eval_every"] == 0 or i == 1):
            accs = [round(accuracy(net, ld, cfg.device), 2) for ld in data.test_loaders()]
            log(f"[FE] round {i} test {accs} neg_var {neg}")
    if n_neg:
        log(f"[FE] clamped {n_neg} negative BN running_var entries")
    return net.state_dict()


_BN_BUF = ("running_mean", "running_var", "num_batches_tracked")


@torch.no_grad()
def _recal_bn_fed(net, client_loaders, device):
    """Each client re-estimates BN running stats of the current global model on its own data (cumulative
    average over one pass, train-mode forward, no parameter update); the server averages them uniformly."""
    bns = [k for k, m in net.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    stats = []
    for ld in client_loaders:
        m = copy.deepcopy(net)
        mods = dict(m.named_modules())
        for k in bns:
            mods[k].reset_running_stats()
            mods[k].momentum = None
        m.train()
        for x, _, _ in ld:
            m(x.to(device))
        stats.append({k: (mods[k].running_mean.clone(), mods[k].running_var.clone()) for k in bns})
        del m
    mods = dict(net.named_modules())
    for k in bns:
        mods[k].running_mean.copy_(sum(st[k][0] for st in stats) / len(stats))
        mods[k].running_var.copy_(sum(st[k][1] for st in stats) / len(stats))


def _fe_step(old_cms, new_cms, old_gm, new_gm, bn="formula"):
    """original unlearning_step_once, applied to every state-dict entry (incl. BN buffers if
    bn == 'formula'; bn == 'avg' sets BN buffers to the mean of the calibration clients' buffers)."""
    out, n_neg = {}, 0
    for k in old_gm:
        nc = sum(m[k].float() for m in new_cms) / len(new_cms)
        if bn == "avg" and k.endswith(_BN_BUF):
            out[k] = nc.to(old_gm[k].dtype)
            continue
        oc = sum(m[k].float() for m in old_cms) / len(old_cms)
        ou = oc - old_gm[k].float()
        nu = nc - new_gm[k].float()
        val = new_gm[k].float() + torch.norm(ou) * (nu / torch.norm(nu))
        if not torch.isfinite(val).all():  # ||nu|| == 0 (e.g. num_batches_tracked unchanged)
            val = new_gm[k].float()
        if k.endswith("running_var"):
            neg = (val <= 0).sum().item()
            if neg:
                n_neg += neg
                val = val.clamp_min(1e-5)
        out[k] = val.to(old_gm[k].dtype)
    return out, n_neg


def _sq_dist(m1, m2):
    with torch.no_grad():
        return torch.square(torch.norm(nn.utils.parameters_to_vector(m1.parameters())
                                       - nn.utils.parameters_to_vector(m2.parameters())))


def pga(cfg, hp, data, full, target, loaders):
    """bl3 (PGA [13]): gradient ascent on the target client's data, projected onto an L2 ball around
    the reference model w_ref = N/(N-1) w_glob - 1/(N-1) w_target; then recovery rounds (local_ep=1)."""
    dev = cfg.device
    N = data.num_users
    init_model = build_model(cfg.dataset, dev)
    glob = copy.deepcopy(init_model)
    glob.load_state_dict(full["global"])
    tgt = copy.deepcopy(init_model)
    tgt.load_state_dict(full["locals"][target])
    ref_vec = N / (N - 1) * nn.utils.parameters_to_vector(glob.parameters()) \
        - 1 / (N - 1) * nn.utils.parameters_to_vector(tgt.parameters())
    net_ref = copy.deepcopy(init_model)  # NB: BN buffers stay at their initial values (the original code)
    nn.utils.vector_to_parameters(ref_vec, net_ref.parameters())
    dists = torch.tensor([_sq_dist(net_ref, build_model(cfg.dataset, dev)).item() for _ in range(10)])
    threshold = torch.mean(dists) / 3
    net = copy.deepcopy(net_ref)
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    net.train()
    stop = False
    for _ in range(hp["pga_local_unlearn_epochs"]):
        if stop:
            break
        for x, y, _ in loaders[target]:
            x, y = x.to(dev), y.to(dev)
            net.zero_grad()
            (-ce(logits_of(net(x)), y)).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), hp["pga_clip"])
            opt.step()
            with torch.no_grad():
                if _sq_dist(net, net_ref) > threshold:
                    dv = nn.utils.parameters_to_vector(net.parameters()) - nn.utils.parameters_to_vector(net_ref.parameters())
                    dv = dv / torch.norm(dv) * np.sqrt(threshold.item())
                    nn.utils.vector_to_parameters(nn.utils.parameters_to_vector(net_ref.parameters()) + dv,
                                                  net.parameters())
            if _sq_dist(net, tgt) > hp["pga_dist_stop"]:
                stop = True
                break
    net, _ = fl_rounds(net, loaders, _retained(N, target), hp["unlearn_rounds"], hp["pga_recovery_ep"], cfg)
    return net.state_dict()


def class_pruning(cfg, hp, data, full, target, loaders):
    """bl4: TF-IDF class-discriminative channel pruning (adapted to the target domain), then
    fine-tuning rounds with local_ep=10 on the retained clients."""
    from fdu import _pruning_utils as pu
    args = SimpleNamespace(model="cnn" if cfg.dataset == "digits" else "vgg16", num_users=data.num_users,
                           device=cfg.device, dataset=cfg.dataset)
    pu.vgg_unlrean = lambda **kw: build_pruning_model(cfg.dataset, cfg.device)
    pu.DigitModel_Pruning = lambda: build_pruning_model("digits", cfg.device)
    pu.R_OVERRIDE = hp["cp_prune_ratio"]
    net = build_model(cfg.dataset, cfg.device)
    w_glob = {k: v.to(cfg.device) for k, v in copy.deepcopy(full["global"]).items()}
    w_locals = [{k: v.to(cfg.device) for k, v in w.items()} for w in full["locals"]]
    w_pruned = pu.unlearn_prune(args, [target], w_locals, w_glob, net, loaders)
    net.load_state_dict(w_pruned)
    net, _ = fl_rounds(net, loaders, _retained(data.num_users, target), hp["unlearn_rounds"], hp["cp_ft_ep"], cfg)
    return net.state_dict()


# ---------------------------------------------------------------------------
# FedOSD (official algorithm, torch port)
# ---------------------------------------------------------------------------
def _unlearning_ce(pred, target):
    p = F.softmax(pred, dim=-1)
    return -torch.mean(torch.sum(torch.log(1.0 - p / 2) * F.one_hot(target, pred.shape[1]).float(), dim=1))


def _full_grad(net, loader, loss_fn, device, clip):
    """Client.cal_gradient_loss ('bgd'): gradient over all local batches, weighted by batch size."""
    net.train()
    acc, tot = None, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        loss = loss_fn(logits_of(net(x)), y)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        g = torch.cat([p.grad.detach().flatten() for p in net.parameters()]) * x.size(0)
        acc = g if acc is None else acc + g
        tot += x.size(0)
    return acc / tot


def _pinv(m):
    U, s, V = torch.svd(m)
    idx = torch.where(s >= 1e-6)[0]
    s = s.clone()
    s[idx] = 1 / s[idx]
    return V @ torch.diag(s) @ U.T


def fedosd(cfg, hp, data, full, target, loaders):
    dev = cfg.device
    net = build_model(cfg.dataset, dev)
    net.load_state_dict(full["global"])
    w0 = nn.utils.parameters_to_vector(net.parameters()).detach().clone()
    ce = nn.CrossEntropyLoss()
    lr, r_lr = hp["fedosd_lr"], hp["fedosd_r_lr"]
    keep = _retained(data.num_users, target)
    for r in range(hp["fedosd_R"]):
        recovery = r >= hp["fedosd_UR"]
        clients = keep if recovery else list(range(data.num_users))
        grads = {}
        for c in clients:
            m = copy.deepcopy(net)  # client copy: BN running stats of the global model are not updated
            grads[c] = _full_grad(m, loaders[c], _unlearning_ce if c == target else ce, dev, hp["fedosd_clip"])
            del m
        w = nn.utils.parameters_to_vector(net.parameters()).detach()
        if not recovery:
            gu = grads[target]
            A = torch.stack([grads[c] for c in keep])
            d = gu - (A.T @ (_pinv(A @ A.T) @ (A @ gu.reshape(-1, 1)))).reshape(-1)
            d = d / torch.norm(d) * torch.norm(gu)
            w = w - lr * d
        else:
            ga = w - w0
            gs = []
            for c in keep:
                g = grads[c]
                gn = torch.norm(g)
                g = g - (g @ ga) / (torch.norm(ga) ** 2 + 1e-12) * ga
                gs.append(g / (torch.norm(g) + 1e-12) * gn)
            w = w - r_lr * torch.stack(gs).mean(0)
        nn.utils.vector_to_parameters(w, net.parameters())
        lr *= hp["fedosd_decay"]
        r_lr *= hp["fedosd_decay"]
    return net.state_dict()


# ---------------------------------------------------------------------------
def neg_not(cfg, hp, data, full, target, loaders):
    """NoT: negate the parameters of the selected layer indices (default: index 0 = first conv
    weight), then fine-tune with the retained clients."""
    net = build_model(cfg.dataset, cfg.device)
    net.load_state_dict(full["global"])
    layers = {int(i) for i in str(hp["not_layers"]).split(",")}
    with torch.no_grad():
        for i, p in enumerate(net.parameters()):
            if i in layers:
                p.mul_(-1)
    net, _ = fl_rounds(net, loaders, _retained(data.num_users, target), hp["not_rounds"], hp["not_ep"], cfg)
    return net.state_dict()


def fedunran(cfg, hp, data, full, target, loaders):
    """FedUNRAN: the unlearning client trains on uniformly random labels (KL(onehot(rand) || p) ==
    CE with random labels), Adam, batch 32, early stopping on the loss (patience 3, restore best);
    then sample-weighted FedAvg recovery rounds with the retained clients (local_ep=1)."""
    dev = cfg.device
    net = build_model(cfg.dataset, dev)
    net.load_state_dict(full["global"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["unran_lr"])
    ld = data.subset_loader(target, range(len(data.train[target])), split="train", shuffle=True,
                            seed=cfg.seed, bs=hp["unran_bs"])
    g = torch.Generator().manual_seed(cfg.seed)
    best, best_sd, bad = float("inf"), cpu_sd(net.state_dict()), 0
    for ep in range(hp["unran_epochs"]):
        net.train()
        tot, n = 0.0, 0
        for x, _, _ in ld:
            x = x.to(dev)
            out = logits_of(net(x))
            rl = torch.randint(0, out.shape[1], (x.size(0),), generator=g).to(dev)
            loss = F.cross_entropy(out, rl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        ep_loss = tot / n
        if ep_loss < best:
            best, best_sd, bad = ep_loss, cpu_sd(net.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["unran_patience"]:
                break
    net.load_state_dict(best_sd)
    keep = _retained(data.num_users, target)
    sizes = [len(data.train[c]) for c in keep]
    net, _ = fl_rounds(net, loaders, keep, hp["unran_rounds"], hp["unran_ep"], cfg, weights=sizes)
    return net.state_dict()


METHODS = {"retrain": retrain, "rr": rapid_retrain, "fe": federaser, "pga": pga, "cp": class_pruning,
           "fedosd": fedosd, "not": neg_not, "fedunran": fedunran}
