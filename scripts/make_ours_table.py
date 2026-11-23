#!/usr/bin/env python
"""LaTeX table of the verifier (final setting): marker (%) of every model on the representative
samples of the forgotten domain, per target. Missing models print as '--'.

python scripts/make_ours_table.py \
    --spec digits:SVHN:0.05:ref_fl_r20_sh_eps0.05_b20.0_auto5_mspec20_h8888 ... \
    [--runs DIR] > ours_rows.tex
spec = dataset:target:eps:verify_tag   (run = <runs>/<dataset>/fpl/le10/seed2024)
"""
import argparse
import json
import os

# column label -> candidate model keys in scores.json (first existing one wins; FedOSD / FedUNRAN use the
# same learning rates as the E1 table rows)
COLS = [("Orig", ["Orig"]), ("Retrain", ["retrain"]), ("Held-out", ["hold0"]), ("RR", ["rr"]),
        ("FE", ["fe_recal", "fe"]), ("CP", ["cp"]), ("PGA", ["pga"]),
        ("FedOSD", {"digits": ["fedosd_lr0.4"], "office": ["fedosd_lr0.1"], "domainnet": ["fedosd_lr0.1"]}),
        ("NoT", ["not"]),
        ("FedUNRAN", {"digits": ["fedunran_lr1e-3"], "office": ["fedunran_lr3e-5"], "domainnet": ["fedunran"]})]
NAMES = {"SVHN": "SVHN", "MNIST-M": "MNIST-M", "MNIST": "MNIST", "USPS": "USPS", "SynthDigits": "SynthDigits",
         "amazon": "Amazon", "caltech": "Caltech", "dslr": "DSLR", "webcam": "Webcam",
         "infograph": "Infograph"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--runs", default=os.environ.get("FDU_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")))
    ap.add_argument("--le", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    print("% verifier (final setting): marker (%) on the representative samples; "
          "columns: " + " & ".join(c for c, _ in COLS))
    for spec in a.spec:
        ds, t, eps, tag = spec.split(":", 3)
        p = os.path.join(a.runs, ds, "fpl", f"le{a.le}", f"seed{a.seed}", "verify", tag, t, "scores.json")
        sc = json.load(open(p))["scores"] if os.path.exists(p) else {}
        cells = []
        for _, keys in COLS:
            keys = keys[ds] if isinstance(keys, dict) else keys
            k = next((k for k in keys if k in sc), None)
            cells.append(f"{sc[k]['rep']['marker']:.2f}" if k else "--")
        print(f"{NAMES.get(t, t)} ($\\epsilon$={eps}) & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
