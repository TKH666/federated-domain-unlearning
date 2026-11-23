"""E4a: verifier validity analysis using ONLY numbers already in the manuscript (Tables I-VI).

Reference ("ground-truth residual") for model m on target domain d:
    a_m = accuracy of m on the forgetting domain (Tables I-III);
    residual_m = a_m - a_Retrain   (> 0: retains more than the ideal reference).
For each verifier v with score s_m (Tables IV-VI; higher = more residual):
  * separation   : s_Orig - s_Retrain
  * norm. score  : r_m = (s_m - s_Retrain) / (s_Orig - s_Retrain)
  * Spearman rho : rank agreement between s and a over {Orig, Retrain, RR, FE, PGA, CP}
                   and over the four approximate methods only {RR, FE, PGA, CP}
  * spread       : std of r_m over the four approximate methods (resolution)
  * false-forgotten verdict: residual_m > DELTA (accuracy above Retrain) but r_m < TAU
Run: python3 e4a_verifier_validity.py [test|train]
"""
import sys
import numpy as np
from scipy.stats import spearmanr
from paper_tables import ACC, VERIFY, METHODS, UNLEARN

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
DELTA, TAU = 1.0, 0.10
VERIFIERS = ["MIA", "Backdoor", "Ours"]

rows = []
for ds, targets in ACC.items():
    for tgt, accs in targets.items():
        a = accs[SPLIT]
        for v in VERIFIERS:
            s = VERIFY[ds][tgt][v]
            sep = s["Orig"] - s["Retrain"]
            r = {m: (s[m] - s["Retrain"]) / sep for m in METHODS}
            rho_all = spearmanr([s[m] for m in METHODS], [a[m] for m in METHODS]).correlation
            rho_ul = spearmanr([s[m] for m in UNLEARN], [a[m] for m in UNLEARN]).correlation
            under = [m for m in UNLEARN if a[m] - a["Retrain"] > DELTA]
            ff = [m for m in under if r[m] < TAU]
            rows.append(dict(ds=ds, tgt=tgt, v=v, sep=sep, rho_all=rho_all, rho_ul=rho_ul,
                             spread=float(np.std([r[m] for m in UNLEARN])),
                             n_under=len(under), n_ff=len(ff), ff=ff,
                             r={m: round(r[m], 2) for m in UNLEARN}))

print(f"== ground truth split: {SPLIT}; under-forgetting if acc - acc_Retrain > {DELTA}; "
      f"'forgotten' verdict if normalized score < {TAU}\n")
print(f"{'dataset':10s}{'target':12s}{'verifier':9s}{'sep':>7s}{'rho6':>7s}{'rho4':>7s}{'spread':>8s}  under->FF   normalized scores (RR,FE,PGA,CP)")
for x in rows:
    print(f"{x['ds']:10s}{x['tgt']:12s}{x['v']:9s}{x['sep']:7.1f}{x['rho_all']:7.2f}{x['rho_ul']:7.2f}{x['spread']:8.2f}  "
          f"{x['n_under']}->{x['n_ff']} {str(x['ff']):14s} {x['r']}")

print("\n== Summary per verifier (mean over targets; rho4 NaN-safe) ==")
print(f"{'dataset':10s}{'verifier':9s}{'sep':>7s}{'rho6':>7s}{'rho4':>7s}{'spread':>8s}{'FF/under':>10s}")
for ds in list(ACC) + ["ALL"]:
    for v in VERIFIERS:
        xs = [x for x in rows if x["v"] == v and (ds == "ALL" or x["ds"] == ds)]
        print(f"{ds:10s}{v:9s}{np.mean([x['sep'] for x in xs]):7.1f}{np.nanmean([x['rho_all'] for x in xs]):7.2f}"
              f"{np.nanmean([x['rho_ul'] for x in xs]):7.2f}{np.mean([x['spread'] for x in xs]):8.2f}"
              f"{sum(x['n_ff'] for x in xs):5d}/{sum(x['n_under'] for x in xs):<4d}")
