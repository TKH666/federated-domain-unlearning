#!/usr/bin/env python
"""Fig. 10: verification accuracy of the verifier for different perturbation scales eps and
selection ratios lambda, on Domain-Digits SVHN and MNIST-M (seed 2024). Each line is one verifier run; the x-axis
lists the evaluated models (Orig, held-out reference, Retrain, RR, FE, PGA, CP).

Input: a directory with one scores.json per run, named <target>__<variant>.json (copied from
<run>/verify/<tag>/<target>/scores.json; the tags are listed in TAGS below).

python scripts/plot_fig10.py --data results/fig10 --out out/fig10_eps_lambda.pdf
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = "ref_fl_r20_sh_eps"
TAGS = {  # (target, panel, value) -> verify tag
    ("SVHN", "eps", 0.05): f"{P}0.05_b20.0_auto5_mspec20_h8888",
    ("SVHN", "eps", 0.1): f"{P}0.1_b20.0_auto5_mspec20_h8888",
    ("SVHN", "eps", 0.3): f"{P}0.3_b20.0_auto5_mspec20_kb_h8888",
    ("SVHN", "lam", 0.2): f"{P}0.05_b20.0_auto5_mspec20_kb_lam0.2_h8888",
    ("SVHN", "lam", 0.5): f"{P}0.05_b20.0_auto5_mspec20_h8888",
    ("SVHN", "lam", 0.8): f"{P}0.05_b20.0_auto5_mspec20_kb_lam0.8_h8888",
    ("MNIST-M", "eps", 0.1): f"{P}0.1_b20.0_auto5_mspec20_kb_h8888",
    ("MNIST-M", "eps", 0.3): f"{P}0.3_b20.0_auto5_mspec20_h8888",
    ("MNIST-M", "eps", 0.5): f"{P}0.5_b20.0_auto5_mspec20_h8888",
    ("MNIST-M", "lam", 0.2): f"{P}0.3_b20.0_auto5_mspec20_kb_lam0.2_h8888",
    ("MNIST-M", "lam", 0.5): f"{P}0.3_b20.0_auto5_mspec20_h8888",
    ("MNIST-M", "lam", 0.8): f"{P}0.3_b20.0_auto5_mspec20_kb_lam0.8_h8888",
}
DEFAULT = {("SVHN", "eps"): 0.05, ("MNIST-M", "eps"): 0.3, ("SVHN", "lam"): 0.5, ("MNIST-M", "lam"): 0.5}
MODELS = [("Orig", "Orig"), ("hold0", "Held-out"), ("retrain", "Retrain"), ("rr", "RR"), ("fe_recal", "FE"),
          ("pga", "PGA"), ("cp", "CP")]
# categorical slots 1-3 of the palette used for Fig. S1 (fixed order); markers + line styles as secondary encoding
STYLE = [("#2a78d6", "o", "-"), ("#eb6834", "s", "--"), ("#1baf7a", "^", ":")]
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3dd"


def fname(target, var):
    return f"{target}__{var}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=95.0)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.3), sharey=True)
    rows = []
    for r, target in enumerate(["SVHN", "MNIST-M"]):
        for c, panel in enumerate(["eps", "lam"]):
            ax = axes[r, c]
            vals = sorted(v for (t, p, v) in TAGS if t == target and p == panel)
            for k, v in enumerate(vals):
                d = json.load(open(os.path.join(a.data, fname(target, f"{panel}{v}"))))
                sc = d["scores"]
                hist = (d.get("proxy_meta") or {}).get("proxy_hist") or []
                reached = bool(hist) and hist[-1][0] > a.threshold
                ys = [sc[m]["rep"]["marker"] for m, _ in MODELS]
                sym = r"\epsilon" if panel == "eps" else r"\lambda"
                lab = f"${sym}={v:g}$" + (" (default)" if DEFAULT[(target, panel)] == v else "") + \
                      ("" if reached else r" $^\dagger$")
                col, mk, ls = STYLE[k]
                ax.plot(range(len(MODELS)), ys, color=col, marker=mk, linestyle=ls, linewidth=1.5, markersize=4.5,
                        label=lab)
                rows.append({"target": target, "panel": panel, "value": v, "reached": reached,
                             **{lbl: round(y, 2) for (_, lbl), y in zip(MODELS, ys)}})
            ax.set_xticks(range(len(MODELS)))
            ax.set_xticklabels([lbl for _, lbl in MODELS])
            ax.set_ylim(-4, 104)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.set_ylabel("Verification acc. (%)")
            ax.set_title(f"({'abcd'[2 * r + c]}) {target}, " + ("$\\epsilon$" if panel == "eps" else "$\\lambda$"),
                         fontsize=8, color=INK)
            # the region above Held-out..FE is empty in every panel (scores <= 20%): place the legend there
            ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, 1.0), handlelength=2.2)
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=1.0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight")
    json.dump(rows, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    for x in rows:
        print(x)


if __name__ == "__main__":
    main()
