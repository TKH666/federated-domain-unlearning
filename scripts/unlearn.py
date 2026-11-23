#!/usr/bin/env python
"""Run one federated unlearning method on one target domain.

Output: <run_dir>/unlearn/<method>[_<hptag>]/<target>/{weight.pt, metrics.json, log.txt}
metrics.json: per-domain test accuracy, per-domain train accuracy (deterministic transform),
unlearning time, hyperparameters.
"""
import argparse
import ast
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import Timer, add_cfg_args, cfg_from_args, cpu_sd, log, save_json, set_seed  # noqa: E402
from fdu.data import DOMAINS, FedData  # noqa: E402
from fdu.fl import accuracy  # noqa: E402
from fdu.models import build_model  # noqa: E402
from fdu.unlearn import DEFAULTS, METHODS  # noqa: E402


def parse_hp(items):
    hp = dict(DEFAULTS)
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in hp:
            raise KeyError(f"unknown hp {k}")
        try:
            hp[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            hp[k] = v
    return hp


def load_full(run_dir):
    d = os.path.join(run_dir, "full")
    return {"dir": d, "init": torch.load(os.path.join(d, "weight_init.pt")),
            "global": torch.load(os.path.join(d, "weight_global.pt")),
            "locals": torch.load(os.path.join(d, "weight_local.pt"))}


def evaluate_all(net, data, device):
    te = [accuracy(net, ld, device) for ld in data.test_loaders()]
    tr = [accuracy(net, ld, device) for ld in data.train_eval_loaders()]
    return te, tr


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--target", required=True, help="domain name or index")
    p.add_argument("--hp", nargs="*", help="key=value overrides of fdu.unlearn.DEFAULTS")
    p.add_argument("--hptag", default="", help="suffix for the output folder (hp variants)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    hp = parse_hp(a.hp)
    names = DOMAINS[cfg.dataset]
    t_idx = int(a.target) if a.target.isdigit() else names.index(a.target)
    out = os.path.join(cfg.run_dir(), "unlearn", a.method + (f"_{a.hptag}" if a.hptag else ""), names[t_idx])
    if os.path.exists(os.path.join(out, "metrics.json")) and not a.overwrite:
        print(f"exists: {out}")
        return
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    set_seed(cfg.seed)
    log(f"method={a.method} target={names[t_idx]} cfg={cfg} hp={hp}", fh)

    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    loaders = data.train_loaders(seed=cfg.seed + 1000)
    full = load_full(cfg.run_dir())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = Timer()
    sd = METHODS[a.method](cfg, hp, data, full, t_idx, loaders)
    secs = t()
    net = build_model(cfg.dataset, cfg.device)
    net.load_state_dict(sd)
    sd = cpu_sd(net.state_dict())
    finite = all(torch.isfinite(v.float()).all().item() for v in sd.values())
    te, tr = evaluate_all(net, data, cfg.device)
    torch.save(sd, os.path.join(out, "weight.pt"))
    keep = [i for i in range(len(names)) if i != t_idx]
    save_json({"method": a.method, "target": names[t_idx], "target_idx": t_idx, "names": names,
               "test_acc": te, "train_acc": tr, "forget_test": te[t_idx], "forget_train": tr[t_idx],
               "retained_test_avg": float(np.mean([te[i] for i in keep])),
               "unlearn_time_sec": secs, "finite": finite,
               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
               "cfg": cfg.__dict__, "hp": hp}, os.path.join(out, "metrics.json"))
    log(f"done {a.method}/{names[t_idx]} test {np.round(te, 2).tolist()} train {np.round(tr, 2).tolist()} "
        f"time {secs:.0f}s finite={finite}", fh)


if __name__ == "__main__":
    main()
