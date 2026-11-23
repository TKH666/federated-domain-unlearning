#!/usr/bin/env python
"""E3 local-epoch sensitivity figure: forget-domain and retained-domain test accuracy vs local epochs E
for Full / Retrain / FE / PGA / FedOSD on two Office and two Digits targets (seed 2024).

python scripts/plot_e3.py --data results/e3_local_epochs/accuracy.json --out out/e3_local_epochs.pdf
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-5 of the reference palette, fixed order; markers + line styles as secondary encoding (print)
SERIES = [("Full", "Full", "#2a78d6", "o", "-"),
          ("retrain", "Retrain", "#eb6834", "s", "--"),
          ("fe_recal", "FE", "#1baf7a", "^", "-"),
          ("pga", "PGA", "#eda100", "D", "-"),
          ("fedosd", "FedOSD", "#e87ba4", "v", "-")]
FEDOSD = {"office": "fedosd_lr0.1", "digits": "fedosd_lr0.4"}  # same variants as the E1 table rows
PANELS = [("amazon", "office", "Office: Amazon"), ("caltech", "office", "Office: Caltech"),
          ("SVHN", "digits", "Digits: SVHN"), ("MNIST-M", "digits", "Digits: MNIST-M")]
EPOCHS = [1, 5, 10, 20]
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3dd"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = json.load(open(a.data))
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.2), sharex=True)
    for j, (t, ds, title) in enumerate(PANELS):
        for i, (which, ylab) in enumerate([(0, "Forgotten domain\nacc. (%)"), (1, "Retained domains\nacc. (%)")]):
            ax = axes[i, j]
            for key, label, col, mk, ls in SERIES:
                k = FEDOSD[ds] if key == "fedosd" else key
                xs, ys = [], []
                for e in EPOCHS:
                    r = d.get(f"{t}|le{e}", {})
                    if k in r:
                        xs.append(e)
                        ys.append(r[k][which])
                ax.plot(xs, ys, ls, color=col, marker=mk, ms=3.5, lw=1.4, label=label,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=3 if key != "retrain" else 4)
            ax.set_xscale("log")
            ax.set_xticks(EPOCHS)
            ax.set_xticklabels([str(e) for e in EPOCHS])
            ax.minorticks_off()
            ax.grid(True, axis="y", color=GRID, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axvline(10, color=GRID, lw=3, zorder=0)  # the paper's setting (FPL protocol)
            if i == 0:
                ax.set_title(title, fontsize=8, color=INK)
            else:
                ax.set_xlabel("Local epochs $E$")
            if j == 0:
                ax.set_ylabel(ylab)
    fig.align_ylabels(axes[:, 0])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8, h_pad=0.6)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
