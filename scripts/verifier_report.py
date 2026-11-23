#!/usr/bin/env python
"""Evidence tables for the verifier discussion: every Ours variant (eps / selection / anchor) per target,
next to the accuracy ground truth of each model on the forgetting domain.

python scripts/verifier_report.py [--runs DIR] > report.md
"""
import argparse
import glob
import json
import os

TARGETS = [("digits", 10, "SVHN"), ("digits", 10, "MNIST-M"),
           ("office", 10, "amazon"), ("office", 10, "caltech"), ("office", 10, "dslr"), ("office", 10, "webcam"),
           ("domainnet", 5, "infograph")]
MODELS = ["Orig", "retrain", "rr", "fe", "pga", "cp", "fedosd_lr0.4", "not", "fedunran_lr1e-3"]
SHORT = {"Orig": "Orig", "retrain": "Retrain", "rr": "RR", "fe": "FE", "pga": "PGA", "cp": "CP",
         "fedosd_lr0.4": "FedOSD", "not": "NoT", "fedunran_lr1e-3": "FedUNRAN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.environ.get("FDU_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")))
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    for ds, le, t in TARGETS:
        rd = os.path.join(a.runs, ds, "fpl", f"le{le}", f"seed{a.seed}")
        gt = {}
        fm = os.path.join(rd, "full", "metrics.json")
        if os.path.exists(fm):
            full = json.load(open(fm))
            from fdu.data import DOMAINS  # noqa: E402
            ti = DOMAINS[ds].index(t)
            gt["Orig"] = (full["final_test_acc"][ti], full["final_train_acc"][ti])
        for m in MODELS[1:]:
            f = os.path.join(rd, "unlearn", m, t, "metrics.json")
            if os.path.exists(f):
                r = json.load(open(f))
                gt[m] = (r["test_acc"][ti], r["train_acc"][ti])
        cols = [m for m in MODELS if m in gt]
        print(f"\n### {ds} / {t}\n")
        print("| variant | proxy ep | " + " | ".join(SHORT[m] for m in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        print("| *test acc (GT)* | | " + " | ".join(f"{gt[m][0]:.1f}" for m in cols) + " |")
        print("| *train acc (GT)* | | " + " | ".join(f"{gt[m][1]:.1f}" for m in cols) + " |")
        first = None
        for f in sorted(glob.glob(os.path.join(rd, "verify", "ours*", t, "scores.json"))):
            s = json.load(open(f))
            sc, meta = s["scores"], s["proxy_meta"]
            if first is None:
                first = sc
                print("| *clean acc on rep (no proxy)* | | " +
                      " | ".join(f"{sc[m]['rep']['clean_acc']:.1f}" if m in sc else "–" for m in cols) + " |")
            v = f.split(os.sep)[-3].replace("ours_", "").replace("ours", "eps0.5 (paper)")
            reached = sc["Orig"]["rep"]["marker"] >= 95.0
            print(f"| {v}{'' if reached else ' (<95%)'} | {meta.get('proxy_epochs')} | " +
                  " | ".join(f"{sc[m]['rep']['marker']:.1f}" if m in sc else "–" for m in cols) + " |")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
