#!/usr/bin/env python
"""Collect results of one run directory and compare with the manuscript (G1 consistency gate).

python scripts/collect.py --dataset digits [--seed 2024] [--runs DIR] [--backbone fpl]
Prints, per target: forgetting-domain / per-domain test accuracy for each method next to the
manuscript value (Tables I/II/III), and the Ours marker accuracy next to Tables IV-VI.
Writes <run_dir>/summary.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.data import DOMAINS  # noqa: E402

try:
    from paper_tables import ACC as P_ACC, VERIFY as P_VER  # manuscript numbers (scripts/paper_tables.py)
except ImportError:
    P_ACC, P_VER = {}, {}

PAPER_DS = {"digits": "Digits", "office": "Office", "domainnet": "DomainNet"}
PAPER_NAME = {"digits": {}, "office": {"amazon": "Amazon", "caltech": "Caltech", "dslr": "Dslr", "webcam": "Webcam"},
              "domainnet": {"clipart": "C", "infograph": "I", "painting": "P", "quickdraw": "Q", "real": "R",
                            "sketch": "S"}}
M2P = {"retrain": "Retrain", "rr": "RR", "fe": "FE", "pga": "PGA", "cp": "CP"}
ORDER = ["retrain", "rr", "fe", "pga", "cp", "fedosd", "not", "fedunran"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="fpl")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--le", type=int, default=10)
    ap.add_argument("--runs", default=os.environ.get("FDU_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rd = os.path.join(a.runs, a.dataset, a.backbone, f"le{a.le}", f"seed{a.seed}" + (f"_{a.tag}" if a.tag else ""))
    names = DOMAINS[a.dataset]
    pds = PAPER_DS[a.dataset]
    pn = lambda d: PAPER_NAME[a.dataset].get(d, d)  # noqa: E731
    summary = {"run_dir": rd, "full": None, "targets": {}}
    fm = os.path.join(rd, "full", "metrics.json")
    if os.path.exists(fm):
        full = json.load(open(fm))
        summary["full"] = {"test": full["final_test_acc"], "train": full["final_train_acc"]}
        print(f"== {rd}\nFull  test {np.round(full['final_test_acc'], 2).tolist()}")
        if pds in P_ACC:
            ref = [P_ACC[pds][pn(d)]["test"]["Orig"] if pn(d) in P_ACC[pds] else float("nan") for d in names]
            print(f"paper test {np.round(ref, 2).tolist()}  (Full learn row; DomainNet: targets I/P/S only)")
    for t in names:
        rows = {}
        for m in ORDER + sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                                 for p in glob.glob(os.path.join(rd, "unlearn", "*", t, "metrics.json"))} - set(ORDER)):
            f = os.path.join(rd, "unlearn", m, t, "metrics.json")
            if os.path.exists(f):
                rows[m] = json.load(open(f))
        if not rows:
            continue
        print(f"\n-- target {t}")
        print(f"{'method':>10s} | forget-test (paper) | test per domain {names} | time(s)")
        for m, r in rows.items():
            pv = ""
            if pds in P_ACC and pn(t) in P_ACC[pds] and m in M2P:
                pv = f"({P_ACC[pds][pn(t)]['test'][M2P[m]]:6.2f})"
            print(f"{m:>10s} | {r['forget_test']:6.2f} {pv:>8s}    | {np.round(r['test_acc'], 2).tolist()} | "
                  f"{r['unlearn_time_sec']:.0f}")
        summary["targets"][t] = {"unlearn": {m: {k: r[k] for k in ("test_acc", "train_acc", "forget_test",
                                                                    "retained_test_avg", "unlearn_time_sec")}
                                             for m, r in rows.items()}}
        for vdir in sorted(glob.glob(os.path.join(rd, "verify", "ours*", t, "scores.json"))):
            s = json.load(open(vdir))
            vname = vdir.split(os.sep)[-3]
            print(f"   [{vname}] proxy epochs {s['proxy_meta'].get('proxy_epochs')} "
                  f"train {s['proxy_meta'].get('proxy_train_sec', 0):.0f}s")
            for m, sc in s["scores"].items():
                key = "Orig" if m == "Orig" else M2P.get(m)
                pv = ""
                if pds in P_VER and pn(t) in P_VER[pds] and key:
                    pv = f"(paper {P_VER[pds][pn(t)]['Ours'][key]:6.2f})"
                print(f"   {m:>10s} marker(rep) {sc['rep']['marker']:6.2f} {pv:>15s}  marker(test) "
                      f"{sc['test']['marker']:6.2f}  clean-acc(rep) {sc['rep']['clean_acc']:6.2f}")
            summary["targets"][t].setdefault("ours", {})[vname] = {m: sc["rep"]["marker"] for m, sc in s["scores"].items()}
    with open(os.path.join(rd, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
