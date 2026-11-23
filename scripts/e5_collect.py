#!/usr/bin/env python
"""E5: ablations of the verifier, summarised with the E4b metrics.

Per variant and target: marker (%) of Orig / Retrain / held-out reference, Spearman rho of the marker with the
ground truth (forget-domain test accuracy, r_acc) and with the representation-level reference (penultimate CKA vs
Orig), decision agreement (normalised score >= 0.5, see e4b_compare.py), proxy-training time / epochs, and
whether the running Orig marker reached the threshold.
"no-proxy" rows score every model by its CLEAN accuracy on the representative samples (the samples used
directly, without the generated perturbation), read from the scores.json of the base (final) run.

python scripts/e5_collect.py --run runs/digits/fpl/le10/seed2024 --hp fedosd=fedosd_lr0.4 fedunran=fedunran_lr1e-3 \
    --spec "final:SVHN:ref_..._h8888" "random:SVHN:ref_..._selrandom_h8888" ... \
    --noproxy SVHN=ref_..._h8888 [--json out.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4b_compare import collect, judge, load_cka, method_key, spearman  # noqa: E402


def summarise(score, acc, cka):
    keys = [k for k in score if k in acc]
    out = {"rho_acc": spearman([score[k] for k in keys], [acc[k] for k in keys]), "n": len(keys)}
    j = judge({k: score[k] for k in keys}, acc, "Orig")
    out["agree_acc"] = (sum(x["v_residual"] == x["gt_residual"] for x in j.values()), len(j))
    if cka:
        kk = [k for k in cka if k in score]
        out["rho_cka"] = spearman([score[k] for k in kk], [cka[k] for k in kk])
        jc = judge({k: score[k] for k in kk}, {k: cka[k] for k in kk}, "Orig")
        out["agree_cka"] = (sum(x["v_residual"] == x["gt_residual"] for x in jc.values()), len(jc))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--spec", nargs="+", required=True, help="variant:target:verify_tag")
    p.add_argument("--noproxy", nargs="*", default=[], help="target=verify_tag of the base run")
    p.add_argument("--hp", nargs="*", default=[])
    p.add_argument("--threshold", type=float, default=95.0)
    p.add_argument("--json", default="")
    a = p.parse_args()
    hp_pick = dict(x.split("=", 1) for x in a.hp)
    names = json.load(open(os.path.join(a.run, "full", "metrics.json")))["names"]
    rows = []

    def models(sc, field):
        out, extra = {}, {}
        for m, v in sc.items():
            k = "Orig" if m == "Orig" else method_key(m)
            if m.startswith("hold"):
                extra.setdefault("hold", []).append(field(v))
            elif k == "Orig" or k not in hp_pick or m == hp_pick[k]:
                out[k] = field(v)
        return out, extra

    for spec in a.spec:
        var, t, tag = spec.split(":", 2)
        sp = os.path.join(a.run, "verify", tag, t, "scores.json")
        if not os.path.exists(sp):
            rows.append({"variant": var, "target": t, "tag": tag, "missing": True})
            continue
        d = json.load(open(sp))
        acc = collect(a.run, t, names, hp_pick)
        cka = load_cka(a.run, t, "Orig", hp_pick)
        score, extra = models(d["scores"], lambda v: v["rep"]["marker"])
        pm = d.get("proxy_meta", {})
        hist = pm.get("proxy_hist") or []
        last = hist[-1][0] if hist and isinstance(hist[-1], (list, tuple)) else None
        r = {"variant": var, "target": t, "tag": tag, "orig": score.get("Orig"), "retrain": score.get("retrain"),
             "hold": max(extra["hold"]) if extra.get("hold") else None, "sec": pm.get("proxy_train_sec"),
             "epochs": pm.get("proxy_epochs"), "reached": (last is not None and last > a.threshold),
             "anchor": d.get("args", {}).get("anchor", 9)}
        r.update(summarise({k: v for k, v in score.items()}, acc, cka))
        rows.append(r)
    for spec in a.noproxy:
        t, tag = spec.split("=", 1)
        d = json.load(open(os.path.join(a.run, "verify", tag, t, "scores.json")))
        acc = collect(a.run, t, names, hp_pick)
        cka = load_cka(a.run, t, "Orig", hp_pick)
        score, extra = models(d["scores"], lambda v: v["rep"]["clean_acc"])
        r = {"variant": "no-proxy (clean acc.)", "target": t, "tag": tag, "orig": score.get("Orig"),
             "retrain": score.get("retrain"), "hold": max(extra["hold"]) if extra.get("hold") else None,
             "sec": 0.0, "epochs": 0, "reached": None}
        r.update(summarise(score, acc, cka))
        rows.append(r)

    f = lambda x, n=2: "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))  # noqa: E731
    print("| variant | target | Orig | Retrain | Held-out | rho(acc) | rho(CKA) | agree(acc) | agree(CKA) "
          "| proxy s | epochs | reached |")
    print("|" + "---|" * 12)
    for r in rows:
        if r.get("missing"):
            print(f"| {r['variant']} | {r['target']} | missing: {r['tag']} |")
            continue
        ag = lambda k: f"{r[k][0]}/{r[k][1]}" if k in r else "--"  # noqa: E731
        print(f"| {r['variant']} | {r['target']} | {f(r['orig'])} | {f(r['retrain'])} | {f(r['hold'])} "
              f"| {f(r['rho_acc'])} | {f(r.get('rho_cka'))} | {ag('agree_acc')} | {ag('agree_cka')} "
              f"| {f(r['sec'], 0)} | {r['epochs']} | {r['reached']} |")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
