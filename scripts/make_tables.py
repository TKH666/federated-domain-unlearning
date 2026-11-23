#!/usr/bin/env python
"""E1: LaTeX rows for the new baselines, in the format of the existing accuracy tables.

Digits / Office (Tables II-III): per target, one row per method with mean±std over seeds of the train
accuracy and the test accuracy on every domain (same column order as the manuscript).
DomainNet (Table IV, --delta): train accuracy on the unlearned domain (absolute) and test accuracy on
every domain as a difference to Retrain of the same run (our Retrain, same protocol).

python scripts/make_tables.py --dataset digits --seeds 2024,2025,2026 \
    --methods "FedOSD=fedosd_lr0.3,NoT=not,FedUNRAN=fedunran_lr1e-3"
python scripts/make_tables.py --dataset domainnet --le 5 --seeds 2024 --targets infograph,painting,sketch --delta \
    --methods "FedOSD=fedosd_lr0.3,NoT=not,FedUNRAN=fedunran_lr1e-3"
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.data import DOMAINS  # noqa: E402


def load(runs, ds, le, seed, method, target, backbone="fpl"):
    base = os.path.join(runs, ds, backbone, f"le{le}", f"seed{seed}")
    if method == "full":  # the original model (same for every target)
        f = os.path.join(base, "full", "metrics.json")
        if not os.path.exists(f):
            return None
        m = json.load(open(f))
        return {"train_acc": m["final_train_acc"], "test_acc": m["final_test_acc"]}
    f = os.path.join(base, "unlearn", method, target, "metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None


def cell(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f}±{v.std(ddof=0):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", default="2024,2025,2026")
    ap.add_argument("--le", type=int, default=10)
    ap.add_argument("--targets", default="")
    ap.add_argument("--methods", required=True, help="Label=method_dir,...")
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--backbone", default="fpl", help="fpl (Tables I-III) or fedavg (E2)")
    ap.add_argument("--runs", default=os.environ.get("FDU_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")))
    a = ap.parse_args()
    names = DOMAINS[a.dataset]
    seeds = [int(s) for s in a.seeds.split(",")]
    targets = a.targets.split(",") if a.targets else names
    methods = [m.split("=") for m in a.methods.split(",")]
    for t in targets:
        ti = names.index(t)
        print(f"% ---- target {t}")
        for label, mdir in methods:
            rs = [load(a.runs, a.dataset, a.le, s, mdir, t, a.backbone) for s in seeds]
            miss = [s for s, r in zip(seeds, rs) if r is None]
            rs = [r for r in rs if r is not None]
            if not rs:
                print(f"% {label}: no runs ({mdir})")
                continue
            if miss:
                print(f"% WARNING {label}: missing seeds {miss}")
            tr = np.array([r["train_acc"] for r in rs])
            te = np.array([r["test_acc"] for r in rs])
            if a.delta:
                ret = [load(a.runs, a.dataset, a.le, s, "retrain", t, a.backbone) for s in seeds]
                ret = np.array([r["test_acc"] for r in ret if r is not None])
                if len(ret) != len(te):
                    print(f"% WARNING {label}: retrain missing for some seeds")
                    continue
                d = te - ret
                cells = [f"\\textbf{{{cell(d[:, j])}}}" if j == ti else cell(d[:, j]) for j in range(len(names))]
                print(f"  &  {label:<10s}&\\quad\\quad {cell(tr[:, ti])}    & " + " & ".join(cells) + " \\\\")
            else:
                # same layout as the included Tables (Digits_train_acc / Office10_train_acc): one line per
                # method, train then test accuracy on every domain, unlearned-domain cells in bold
                bold = lambda c, j: f"\\textbf{{{c}}}" if j == ti else c  # noqa: E731
                cells = ([bold(cell(tr[:, j]), j) for j in range(len(names))]
                         + [bold(cell(te[:, j]), j) for j in range(len(names))])
                print(f"    & {label:<10s} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
