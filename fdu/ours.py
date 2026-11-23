"""Proposed verification (manuscript Sec. V-A), following the paper; details the paper leaves open
are taken from the authors' original code (Verify.py, models/Update_domain.py::atk_train,
utils/evaluate.py::test):

1. representative samples: top-lambda samples of the forgetting domain by forgetting-event count,
   recorded during regular FL training (fdu.forget.select).
2. proxy validation model M_PV (U-Net): T(z) = clip(z + eps * M_PV(z)) trained with Adam(lr=1e-4)
   so that the FROZEN original global model predicts the anchor class y_T (Eq. 8). Training runs in
   chunks of 20 epochs and stops once the running marker accuracy exceeds 95% (max 10 chunks).
   The proxy is trained on the augmented training transform, as in the original code. Afterwards the
   proxy's BatchNorm statistics are re-estimated on the marker samples (recalibrate_bn).
3. verification (Eq. 9): fraction of T(z) classified as y_T by the evaluated model, over the
   representative samples whose true label != y_T (deterministic transform); also reported on the
   target-domain test set.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fdu.models import build_proxy, logits_of


def clip_image(x, lo=-1.0, hi=1.0):
    return torch.clamp(x, lo, hi)


def train_proxy(net, proxy, loader, eps, anchor, device, epochs_per_chunk=20, max_chunks=10,
                threshold=95.0, lr=1e-4, clip=(-1.0, 1.0), log_fn=None):
    """original atk_train + Verify.py loop. Returns (proxy, history, epochs_used)."""
    net = copy.deepcopy(net).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(proxy.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    hist, acc, chunks, epochs = [], 0.0, 0, 0
    while acc <= threshold and chunks < max_chunks:
        for _ in range(epochs_per_chunk):
            proxy.train()
            correct, n, losses = 0, 0, []
            for x, y, _ in loader:
                x = x.to(device)
                xt = clip_image(x + proxy(x) * eps, *clip)
                tgt = torch.full((x.size(0),), anchor, dtype=torch.long, device=device)
                out = logits_of(net(xt))
                loss = ce(out, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                correct += (out.argmax(1) == tgt).sum().item()
                n += x.size(0)
                losses.append(loss.item())
            acc = 100.0 * correct / n
            epochs += 1
            hist.append((float(np.mean(losses)), acc))
        chunks += 1
        if log_fn:
            log_fn(f"proxy chunk {chunks}: loss {hist[-1][0]:.4f} marker-acc(train,running) {acc:.2f}")
    recalibrate_bn(proxy, loader, device)
    return proxy, hist, epochs


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Re-estimate BatchNorm running statistics of the (frozen) proxy with a cumulative average over
    the marker samples, so that eval-mode perturbations match the trained generator. Without this,
    running statistics lag behind the rapidly changing generator weights and eval-mode markers are
    much weaker than during training (observed: 99% -> 3% marker accuracy on the original model)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
    model.train()
    for x, _, _ in loader:
        model(x.to(device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


@torch.no_grad()
def marker_scores(net, proxy, loader, eps, anchor, device, clip=(-1.0, 1.0), use_proxy=True):
    """Returns dict: marker (fraction predicted y_T on T(z), true label != y_T, %),
    clean_anchor (fraction predicted y_T on z), clean_acc (true-label accuracy on z), n."""
    net.eval()
    if proxy is not None:
        proxy.eval()
    m = c_anchor = c_acc = n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        keep = y != anchor
        if keep.sum() == 0:
            continue
        x, y = x[keep], y[keep]
        p_clean = logits_of(net(x)).argmax(1)
        c_anchor += (p_clean == anchor).sum().item()
        c_acc += (p_clean == y).sum().item()
        if use_proxy and proxy is not None:
            xt = clip_image(x + proxy(x) * eps, *clip)
            m += (logits_of(net(xt)).argmax(1) == anchor).sum().item()
        n += x.size(0)
    n = max(n, 1)
    return {"marker": 100.0 * m / n, "clean_anchor": 100.0 * c_anchor / n, "clean_acc": 100.0 * c_acc / n,
            "n": n}


def new_proxy(kind, device, seed):
    torch.manual_seed(seed)
    return build_proxy(kind, device)
