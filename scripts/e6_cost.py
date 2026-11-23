#!/usr/bin/env python
"""E6: GPU time to construct the verification signal (cost table).

Ours = training of the reference models + training of the proxy (all stages of a warm-started chain).
  - reference models: every "trained reference model(s) [fl] in <s>s" line in the logs of the runs that use the same
    reference configuration (verify/ref_fl_r<rounds>_sh_*_<vtag>/<target>/log.txt): the models are trained by the
    first run that needs them and cached in verify/refs/ (seed 7777 snapshots + held-out seed 8888 [+ adaptive seeds]);
  - proxy: proxy_meta.proxy_train_sec of every stage of the chain.
VeriFi = marking round (verify/verifi_<v>/<target>/markers.json).

python scripts/e6_cost.py --runs $FDU_RUNS \
    --spec office:amazon:ref_fl_r10-20_sh_eps0.1_b20.0_auto5_mspec20_h8888+ref_fl_r10-20_sh_eps0.1_b20.0_auto5_mspec20_warm_kb_h8888
"""
import argparse
import glob
import json
import os
import re

LE = {"digits": 10, "office": 10, "domainnet": 5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.environ.get("FDU_RUNS", "runs"))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--spec", nargs="+", required=True, help="dataset:target:tag1[+tag2...] (stages in order)")
    p.add_argument("--json", default="")
    a = p.parse_args()
    rows = []
    for spec in a.spec:
        ds, t, chain = spec.split(":", 2)
        tags = chain.split("+")
        vdir = os.path.join(a.runs, ds, "fpl", f"le{LE[ds]}", f"seed{a.seed}", "verify")
        proxy, stages = 0.0, []
        for tag in tags:
            d = json.load(open(os.path.join(vdir, tag, t, "scores.json")))
            s = d["proxy_meta"]["proxy_train_sec"]
            proxy += s
            stages.append(round(s))
        m = re.match(r"(ref_fl_r[\d-]+_sh)_.*_(h\d+)$", tags[0])
        pat = r"trained reference model\(s\) \[fl\] in (\d+)s"
        ref = 0.0
        for lg in glob.glob(os.path.join(vdir, f"{m.group(1)}_*_{m.group(2)}", t, "log.txt")):
            txt = open(lg).read()
            base, _, after = txt.partition("[auto]")
            # before the first held-out check: seed-7777 snapshots and the held-out seed (shared by all runs)
            ref += sum(float(x) for x in re.findall(pat, base))
            # after it: adaptive extra seeds, counted only for the stages of this chain (other runs' extra seeds
            # are not used by it)
            if os.path.basename(os.path.dirname(os.path.dirname(lg))) in tags:
                ref += sum(float(x) for x in re.findall(pat, after))
        mark = {}
        for v in ("vfm", "vem"):
            mp = os.path.join(vdir, f"verifi_{v}", t, "markers.json")
            mark[v] = round(json.load(open(mp))["marking_round_sec"]) if os.path.exists(mp) else None
        rows.append({"dataset": ds, "target": t, "ref_sec": round(ref), "proxy_sec": round(proxy),
                     "proxy_stages": stages, "total_sec": round(ref + proxy), "verifi_mark_sec": mark})
    print("| target | reference models | proxy (stages) | total | VeriFi vFM / vEM |")
    print("|---|---|---|---|---|")
    for r in rows:
        mk = r["verifi_mark_sec"]
        print(f"| {r['target']} | {r['ref_sec']} | {r['proxy_sec']} ({' + '.join(map(str, r['proxy_stages']))}) "
              f"| {r['total_sec']} | {mk['vfm']} / {mk['vem']} |")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
