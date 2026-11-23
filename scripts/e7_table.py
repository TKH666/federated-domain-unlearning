#!/usr/bin/env python
"""E7: LaTeX rows of the feature-inversion table (mean PSNR dB / SSIM over the inverted images, per VGG16
block), one column group per target, from verify/inversion_<tag>/<target>/results.json written by scripts/invert.py.

python scripts/e7_table.py --results <run>/verify/inversion_n20/infograph/results.json \
    <run>/verify/inversion_n20/clipart/results.json
"""
import argparse
import json

LABEL = [("Orig", "Orig"), ("retrain", "Retrain"), ("rr", "RR"), ("fe", "FE"), ("cp", "CP"), ("pga", "PGA"),
         ("fedosd_lr0.1", "FedOSD"), ("not", "NoT"), ("fedunran", "FedUNRAN")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    a = p.parse_args()
    res = [json.load(open(f)) for f in a.results]
    print("% targets: " + ", ".join(f"{r['target']} ({len(r['images'])} images, {r['iters']} iterations)" for r in res))
    for key, label in LABEL:
        if not any(key in r["results"] for r in res):
            continue
        cells = []
        for r in res:
            m = r["results"].get(key)
            for b in r["blocks"]:
                x = m.get(str(b)) if m else None
                cells.append(f"{x['psnr']:.2f} / {x['ssim']:.3f}" if x else "--")
        print(f"{label} & " + " & ".join(cells) + r" \\" + (r" \hline" if key == "Orig" else ""))


if __name__ == "__main__":
    main()
