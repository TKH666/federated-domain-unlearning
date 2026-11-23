#!/usr/bin/env python
"""E3: collect forget-domain / retained-domain test accuracy per local-epoch setting into the JSON read by
scripts/plot_e3.py. Entry "<target>|le<E>": {"Full": [forget, retained], "full_time": s, <method dir>: [forget,
retained], ...}; forget = test accuracy on the target domain, retained = mean test accuracy on the other domains.

python scripts/e3_collect.py --runs $FDU_RUNS --spec office:amazon,caltech digits:SVHN,MNIST-M --out e3_acc.json
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.environ.get("FDU_RUNS", "runs"))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--le", default="1,5,10,20")
    p.add_argument("--spec", nargs="+", required=True, help="dataset:target1,target2")
    p.add_argument("--out", required=True)
    p.add_argument("--merge", action="store_true", help="update an existing --out (results spread over machines)")
    a = p.parse_args()
    res = json.load(open(a.out)) if a.merge and os.path.exists(a.out) else {}
    for spec in a.spec:
        ds, targets = spec.split(":")
        for le in a.le.split(","):
            run = os.path.join(a.runs, ds, "fpl", f"le{le}", f"seed{a.seed}")
            fm = os.path.join(run, "full", "metrics.json")
            if not os.path.exists(fm):
                continue
            f = json.load(open(fm))
            names, acc = f["names"], f["final_test_acc"]
            for t in targets.split(","):
                i = names.index(t)
                e = {"Full": [acc[i], sum(x for j, x in enumerate(acc) if j != i) / (len(acc) - 1)],
                     "full_time": f.get("train_time_sec")}
                for mp in sorted(glob.glob(os.path.join(run, "unlearn", "*", t, "metrics.json"))):
                    m = json.load(open(mp))
                    e[mp.split(os.sep)[-3]] = [m["forget_test"], m["retained_test_avg"]]
                res[f"{t}|le{le}"] = e
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{len(res)} entries -> {a.out}")


if __name__ == "__main__":
    main()
