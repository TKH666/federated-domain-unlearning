#!/usr/bin/env python
"""Calibration + sensitivity summary of verifier variants for one target.

For each verify/<variant>/<target>/scores.json: Orig, Retrain, reference models, and the rank correlation
(Spearman) between the marker on the representative samples and the ground truth (accuracy of each model
on the forgetting domain's training data) over all unlearned models of that target (incl. hyper-parameter
sweeps such as fedosd_lr*, which give a dose-response curve). Also the no-proxy baseline (clean accuracy).

python scripts/verifier_sens.py --dataset digits --target SVHN [--variants ours_eps0.05,ref_local_eps0.1_b1.0]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.data import DOMAINS  # noqa: E402


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--le", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--variants", default="")
    ap.add_argument("--runs", default=os.environ.get("FDU_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")))
    a = ap.parse_args()
    rd = os.path.join(a.runs, a.dataset, "fpl", f"le{a.le}", f"seed{a.seed}")
    ti = DOMAINS[a.dataset].index(a.target)
    gt = {}
    for f in glob.glob(os.path.join(rd, "unlearn", "*", a.target, "metrics.json")):
        gt[f.split(os.sep)[-3]] = json.load(open(f))["train_acc"][ti]
    full = json.load(open(os.path.join(rd, "full", "metrics.json")))
    gt["Orig"] = full["final_train_acc"][ti]
    vs = a.variants.split(",") if a.variants else sorted(
        os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(rd, "verify", "*", a.target)))
    print(f"### {a.dataset}/{a.target}: GT train acc Orig {gt['Orig']:.1f}, Retrain {gt.get('retrain', float('nan')):.1f}")
    print("| variant | Orig | Retrain | refs | PGA | CP | FedOSD-0.4 | Spearman(marker, GT) over n | no-proxy Spearman |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in vs:
        f = os.path.join(rd, "verify", v, a.target, "scores.json")
        if not os.path.exists(f):
            continue
        sc = json.load(open(f))["scores"]
        if "rep" not in sc.get("Orig", {}):
            continue
        mk = {k: s["rep"]["marker"] for k, s in sc.items()}
        ca = {k: s["rep"]["clean_acc"] for k, s in sc.items()}
        ks = [k for k in mk if k in gt and k != "Orig" and k != "fe_bnformula"]
        rho = spearman([mk[k] for k in ks], [gt[k] for k in ks])
        rho0 = spearman([ca[k] for k in ks], [gt[k] for k in ks])
        refs = [f"{mk[k]:.1f}" for k in mk if k.startswith("ref")]
        g = lambda k: f"{mk[k]:.1f}" if k in mk else "–"  # noqa: E731
        print(f"| {v} | {g('Orig')} | {g('retrain')} | {'/'.join(refs) or '–'} | {g('pga')} | {g('cp')} | "
              f"{g('fedosd_lr0.4')} | {rho:+.2f} (n={len(ks)}) | {rho0:+.2f} |")


if __name__ == "__main__":
    main()
