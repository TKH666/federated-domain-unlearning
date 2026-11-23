#!/usr/bin/env python
"""Representation-level reference for E4b: linear CKA (Kornblith et al. 2019) between the
penultimate-layer features of each model and the ORIGINAL model on the forgetting-domain data.
CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F), X,Y column-centered.
Output: <run_dir>/verify/cka/<target>/cka.json  {model: cka on test, cka on train(eval)}
--layers bn: additionally linear CKA after every BatchNorm2d (+ReLU, adaptive-avg-pooled to 2x2, flattened),
i.e. shallow -> deep conv features (DigitModel: 3 layers, VGG16-BN: 13). Output key "layers".
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import add_cfg_args, cfg_from_args, save_json  # noqa: E402
from fdu.data import DOMAINS, FedData  # noqa: E402
from fdu.models import build_model  # noqa: E402


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


@torch.no_grad()
def feats(net, loader, device, max_n):
    net.eval()
    out, n = [], 0
    for x, _, _ in loader:
        _, f = net(x.to(device), output_feature=True)
        out.append(f.double().cpu().numpy())
        n += len(x)
        if n >= max_n:
            break
    return np.concatenate(out)[:max_n]


@torch.no_grad()
def layer_feats(net, loader, device, max_n):
    """Per-BatchNorm2d features (ReLU + 2x2 adaptive average pooling), shallow -> deep."""
    net.eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    buf = [[] for _ in bns]
    hs = [m.register_forward_hook(lambda mod, i, o, k=k: buf[k].append(
        torch.nn.functional.adaptive_avg_pool2d(torch.relu(o), 2).flatten(1).double().cpu().numpy()))
        for k, m in enumerate(bns)]
    n = 0
    for x, _, _ in loader:
        net(x.to(device))
        n += len(x)
        if n >= max_n:
            break
    for h in hs:
        h.remove()
    return [np.concatenate(b)[:max_n] for b in buf]


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--max_n", type=int, default=2000)
    p.add_argument("--layers", default="", choices=["", "bn"])
    p.add_argument("--eval_bs", type=int, default=128)
    a = p.parse_args()
    cfg = cfg_from_args(a)
    names = DOMAINS[cfg.dataset]
    t = int(a.target) if a.target.isdigit() else names.index(a.target)
    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    loaders = {"test": data.test_loaders(bs=a.eval_bs)[t], "train": data.train_eval_loaders(bs=a.eval_bs)[t]}
    models = {"Orig": os.path.join(cfg.run_dir(), "full", "weight_global.pt")}
    for d in sorted(glob.glob(os.path.join(cfg.run_dir(), "unlearn", "*", names[t], "weight.pt"))):
        models[d.split(os.sep)[-3]] = d
    net = build_model(cfg.dataset, cfg.device)
    ref = {}
    net.load_state_dict(torch.load(models["Orig"]))
    for s, ld in loaders.items():
        ref[s] = feats(net, ld, cfg.device, a.max_n)
    res, lref = {}, {}
    for k, pth in models.items():
        net.load_state_dict(torch.load(pth))
        res[k] = {s: linear_cka(ref[s], feats(net, ld, cfg.device, a.max_n)) for s, ld in loaders.items()}
        msg = ""
        if a.layers == "bn":
            res[k]["layers"] = {}
            for s_, ld in loaders.items():
                if s_ not in lref:
                    net.load_state_dict(torch.load(models["Orig"]))
                    lref[s_] = layer_feats(net, ld, cfg.device, a.max_n)
                    net.load_state_dict(torch.load(pth))
                res[k]["layers"][s_] = [linear_cka(r, f) for r, f in
                                        zip(lref[s_], layer_feats(net, ld, cfg.device, a.max_n))]
            msg = "  layers(train) " + " ".join(f"{v:.3f}" for v in res[k]["layers"]["train"])
        print(f"{k:>16s} CKA(test) {res[k]['test']:.4f}  CKA(train) {res[k]['train']:.4f}" + msg, flush=True)
    save_json({"target": names[t], "cka": res}, os.path.join(cfg.run_dir(), "verify", "cka", names[t], "cka.json"))


if __name__ == "__main__":
    main()
