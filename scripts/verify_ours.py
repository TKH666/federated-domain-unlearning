#!/usr/bin/env python
"""Train the proxy validation model for one target domain on the ORIGINAL model and score every
available model (original + all unlearned models of that target under <run_dir>/unlearn/*/<target>).

Output: <run_dir>/verify/ours[_<vtag>]/<target>/{proxy.pt, scores.json}
Variants (E5): --select top|bottom|random|all, --lam, --eps, --anchor, --proxy unet|auto.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import Timer, add_cfg_args, cfg_from_args, load_json, log, save_json, set_seed  # noqa: E402
from fdu.data import DOMAINS, FedData, input_range  # noqa: E402
from fdu.forget import select  # noqa: E402
from fdu.models import build_model  # noqa: E402
from fdu.ours import marker_scores, new_proxy, train_proxy  # noqa: E402


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--anchor", type=int, default=9)
    p.add_argument("--proxy", default="unet", choices=["unet", "auto"])
    p.add_argument("--select", default="top", choices=["top", "bottom", "random", "all"])
    p.add_argument("--threshold", type=float, default=95.0)
    p.add_argument("--epochs_per_chunk", type=int, default=20)
    p.add_argument("--max_chunks", type=int, default=10)
    p.add_argument("--clip", default="lineage", choices=["lineage", "range"],
                   help="lineage: clamp to [-1,1] for every dataset (the original code); range: dataset input range")
    p.add_argument("--vtag", default="")
    p.add_argument("--retrain_proxy", action="store_true")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    names = DOMAINS[cfg.dataset]
    t = int(a.target) if a.target.isdigit() else names.index(a.target)
    out = os.path.join(cfg.run_dir(), "verify", "ours" + (f"_{a.vtag}" if a.vtag else ""), names[t])
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    set_seed(cfg.seed)
    clip = (-1.0, 1.0) if a.clip == "lineage" else input_range(cfg.dataset)
    log(f"target={names[t]} lam={a.lam} eps={a.eps} anchor={a.anchor} proxy={a.proxy} select={a.select} "
        f"clip={clip} cfg={cfg}", fh)

    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    rec = np.load(os.path.join(cfg.run_dir(), "full", "forget_records.npz"))[f"arr_{t}"]
    rep = select(rec, a.lam, a.select, seed=cfg.seed)
    train_ld = data.subset_loader(t, rep, split="train", shuffle=True, seed=cfg.seed)  # augmented
    rep_ld = data.subset_loader(t, rep, split="train_eval", bs=128)                    # deterministic
    test_ld = data.test_loaders()[t]

    orig = build_model(cfg.dataset, cfg.device)
    orig.load_state_dict(torch.load(os.path.join(cfg.run_dir(), "full", "weight_global.pt")))
    ppath = os.path.join(out, "proxy.pt")
    if os.path.exists(ppath) and not a.retrain_proxy:
        proxy = new_proxy(a.proxy, cfg.device, cfg.seed)
        proxy.load_state_dict(torch.load(ppath))
        meta = load_json(os.path.join(out, "scores.json")).get("proxy_meta", {})
    else:
        proxy = new_proxy(a.proxy, cfg.device, cfg.seed)
        tm = Timer()
        proxy, hist, ep = train_proxy(orig, proxy, train_ld, a.eps, a.anchor, cfg.device, a.epochs_per_chunk,
                                      a.max_chunks, a.threshold, clip=clip, log_fn=lambda s: log(s, fh))
        meta = {"proxy_train_sec": tm(), "proxy_epochs": ep, "proxy_hist": hist, "n_rep": int(len(rep))}
        torch.save({k: v.cpu() for k, v in proxy.state_dict().items()}, ppath)

    models = {"Orig": os.path.join(cfg.run_dir(), "full", "weight_global.pt")}
    for d in sorted(glob.glob(os.path.join(cfg.run_dir(), "unlearn", "*", names[t], "weight.pt"))):
        models[d.split(os.sep)[-3]] = d
    scores = {}
    net = build_model(cfg.dataset, cfg.device)
    for k, path in models.items():
        net.load_state_dict(torch.load(path))
        rs = marker_scores(net, proxy, rep_ld, a.eps, a.anchor, cfg.device, clip)
        ts = marker_scores(net, proxy, test_ld, a.eps, a.anchor, cfg.device, clip)
        scores[k] = {"rep": rs, "test": ts}
        log(f"{k:>12s}: marker(rep) {rs['marker']:6.2f}  marker(test) {ts['marker']:6.2f}  "
            f"clean-acc(rep) {rs['clean_acc']:6.2f}  clean-anchor(rep) {rs['clean_anchor']:5.2f}", fh)
    save_json({"target": names[t], "args": vars(a), "cfg": cfg.__dict__, "proxy_meta": meta, "scores": scores},
              os.path.join(out, "scores.json"))


if __name__ == "__main__":
    main()
