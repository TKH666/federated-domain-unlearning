#!/usr/bin/env python
"""E4b: Ours (final verifier) vs VeriFi vFM / vEM against ground-truth residual knowledge.

Ground truth per scored model: forget-domain test accuracy relative to Retrain,
    r_acc(m) = (acc_m - acc_Retrain) / (acc_O - acc_Retrain),  O = the model unlearning started from
(the original model for Ours; the MARKED model for VeriFi, whose methods start from it).
Verifier residual in the same direction (high = knowledge still there):
    Ours: marker %;  VeriFi: minus the checking metric (vFM: loss variance, vEM: loss w.r.t. the new label).
Reported per target: Spearman rho(verifier, r_acc) over the common method set, and the decisions
(residual if normalised verifier score >= 0.5, ground truth residual if r_acc >= 0.5) with false-forgotten /
false-residual counts. Optional: rho(Ours, CKA) when verify/cka/<t>/cka.json exists (unmarked models only).

python scripts/e4b_compare.py --run runs/office/fpl/le10/seed2024 \
    --ours amazon=ref_..._h8888 caltech=... [--json out.json]
"""
import argparse
import json
import os

import numpy as np

CORE = ["retrain", "rr", "fe_recal", "cp", "not", "pga", "fedosd", "fedunran"]


def method_key(name):
    """'fedosd_lr0.4' -> 'fedosd'; exact names otherwise (fe vs fe_recal kept apart)."""
    for k in ("fedosd", "fedunran"):
        if name == k or name.startswith(k + "_lr"):
            return k
    return name


def rank(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def forget_acc(path):
    d = json.load(open(path))
    return d["forget_test"]


def collect(run, t, names, hp_pick):
    """Ground-truth accuracies of the unmarked models: {method_key: acc}, plus Orig."""
    ti = names.index(t)
    acc = {"Orig": json.load(open(os.path.join(run, "full", "metrics.json")))["final_test_acc"][ti]}
    udir = os.path.join(run, "unlearn")
    for m in sorted(os.listdir(udir)):
        k = method_key(m)
        p = os.path.join(udir, m, t, "metrics.json")
        if k in CORE and os.path.exists(p) and (k not in hp_pick or m == hp_pick[k]):
            acc[k] = forget_acc(p)
    return acc


def load_cka(run, t, o_key, hp_pick):
    """Penultimate-layer CKA (train split) of every model vs the run's own global model -> {method_key: cka}."""
    cp = os.path.join(run, "verify", "cka", t, "cka.json")
    if not os.path.exists(cp):
        return {}
    out = {}
    for m, v in json.load(open(cp)).get("cka", {}).items():
        if not isinstance(v, dict) or "train" not in v:
            continue
        k = o_key if m == "Orig" else method_key(m)
        if k == o_key or (k in CORE and (k not in hp_pick or m == hp_pick[k])):
            out[k] = v["train"]
    return out


def judge(score, acc, o_key):
    """score, acc: dicts over the same keys (incl. retrain and o_key). Returns per-method decisions."""
    s_r, s_o = score["retrain"], score[o_key]
    a_r, a_o = acc["retrain"], acc[o_key]
    out = {}
    for k in score:
        if k in ("retrain", o_key):
            continue
        rv = (score[k] - s_r) / (s_o - s_r) if s_o != s_r else float("nan")
        ra = (acc[k] - a_r) / (a_o - a_r) if a_o != a_r else float("nan")
        out[k] = {"r_verifier": rv, "r_acc": ra, "v_residual": bool(rv >= 0.5), "gt_residual": bool(ra >= 0.5)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ours", nargs="+", required=True, help="target=verify_tag")
    p.add_argument("--hp", nargs="*", default=[], help="method=dir to pick one hyper-parameter variant, e.g. "
                   "fedosd=fedosd_lr0.4 fedunran=fedunran_lr1e-3 (same as the E1 table rows)")
    p.add_argument("--json", default="")
    a = p.parse_args()
    hp_pick = dict(x.split("=", 1) for x in a.hp)
    names = json.load(open(os.path.join(a.run, "full", "metrics.json")))["names"]
    res = {}
    for spec in a.ours:
        t, tag = spec.split("=", 1)
        ti = names.index(t)
        acc = collect(a.run, t, names, hp_pick)
        row = {"acc": acc}
        # --- Ours (unmarked models)
        sp = os.path.join(a.run, "verify", tag, t, "scores.json")
        if os.path.exists(sp):
            sc = json.load(open(sp))["scores"]
            ours = {}
            for m, v in sc.items():
                k = "Orig" if m == "Orig" else method_key(m)
                if k in acc and (k == "Orig" or k not in hp_pick or m == hp_pick[k]):
                    ours[k] = v["rep"]["marker"]
            keys = [k for k in ours if k in acc]
            row["ours"] = {"score": ours, "rho_acc": spearman([ours[k] for k in keys], [acc[k] for k in keys]),
                           "n": len(keys), "judge": judge({k: ours[k] for k in keys}, acc, "Orig")}
            cka = load_cka(a.run, t, "Orig", hp_pick)
            if cka:
                kk = [k for k in cka if k in ours]
                row["ours"]["rho_cka"] = spearman([ours[k] for k in kk], [cka[k] for k in kk])
                row["ours"]["judge_cka"] = judge({k: ours[k] for k in kk}, {k: cka[k] for k in kk}, "Orig")
                row["cka"] = cka
        # --- VeriFi (methods that start from the global model were re-run from the MARKED model)
        for v in ("vfm", "vem"):
            vd = os.path.join(a.run, "verify", f"verifi_{v}", t)
            if not os.path.exists(os.path.join(vd, "scores.json")):
                continue
            sc = json.load(open(os.path.join(vd, "scores.json")))["scores"]
            mk = json.load(open(os.path.join(vd, "markers.json")))
            mrun = f"{a.run}_mark-{v}-{t}"
            vacc = {"Marked": mk["test_acc_marked"][ti]}
            vs = {}
            for m, val in sc.items():
                if m == "Orig":
                    continue
                k = "Marked" if m == "Marked" else method_key(m)
                if k != "Marked" and (k not in CORE or (k in hp_pick and m != hp_pick[k])):
                    continue
                if k == "Marked":
                    pass
                elif os.path.exists(os.path.join(mrun, "unlearn", m, t, "metrics.json")):
                    vacc[k] = forget_acc(os.path.join(mrun, "unlearn", m, t, "metrics.json"))
                elif k in acc:  # reused from the base run (retrain / rr / fe)
                    vacc[k] = acc[k]
                else:
                    continue
                vs[k] = -val
            keys = [k for k in vs if k in vacc]
            row[v] = {"score": {k: -vs[k] for k in keys}, "acc": vacc, "n": len(keys),
                      "rho_acc": spearman([vs[k] for k in keys], [vacc[k] for k in keys]),
                      "judge": judge({k: vs[k] for k in keys}, vacc, "Marked") if "retrain" in vs else {},
                      "mark_sec": mk.get("marking_round_sec"), "n_markers": len(mk["markers"]),
                      "marked_drop": [round(x - y, 2) for x, y in zip(mk["test_acc_marked"], mk["test_acc_orig"])]}
            cka = load_cka(mrun, t, "Marked", hp_pick)  # CKA of each scored model vs the MARKED model
            if cka:
                kk = [k for k in cka if k in vs]
                row[v]["rho_cka"] = spearman([vs[k] for k in kk], [cka[k] for k in kk])
                row[v]["judge_cka"] = judge({k: vs[k] for k in kk}, {k: cka[k] for k in kk}, "Marked")
                row[v]["cka"] = cka
        res[t] = row
        line = f"{t:12s}"
        for v in ("ours", "vfm", "vem"):
            if v in row:
                j = row[v]["judge"]
                ff = sum(1 for x in j.values() if x["gt_residual"] and not x["v_residual"])
                fr = sum(1 for x in j.values() if not x["gt_residual"] and x["v_residual"])
                line += (f" | {v}: rho_acc {row[v]['rho_acc']:+.2f} (n={row[v]['n']})"
                         + (f" rho_cka {row[v]['rho_cka']:+.2f}" if 'rho_cka' in row[v] else "")
                         + f" agree(acc) {len(j) - ff - fr}/{len(j)} ff {ff} fr {fr}")
                if "judge_cka" in row[v]:
                    jc = row[v]["judge_cka"]
                    ff = sum(1 for x in jc.values() if x["gt_residual"] and not x["v_residual"])
                    fr = sum(1 for x in jc.values() if not x["gt_residual"] and x["v_residual"])
                    line += f" agree(cka) {len(jc) - ff - fr}/{len(jc)} ff {ff} fr {fr}"
        print(line)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
