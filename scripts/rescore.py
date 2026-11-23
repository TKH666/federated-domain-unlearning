#!/usr/bin/env python
"""Score unlearned models that appeared after a verify_ref.py run, with the SAVED proxy (no re-training).

Reads <run_dir>/verify/<variant>/<target>/{proxy.pt, scores.json}; every unlearn/*/<target>/weight.pt not yet in
scores.json is scored exactly as in verify_ref.py (marker / clean-acc / acc-under-T on the representative samples
and on the test split) and added to scores.json (key "rescored" lists them).

python scripts/rescore.py --dataset office --backbone fpl --seed 2024 --target amazon --variant <verify dir name>
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import add_cfg_args, cfg_from_args, log, save_json  # noqa: E402
from fdu.data import DOMAINS, FedData  # noqa: E402
from fdu.forget import select  # noqa: E402
from fdu.models import build_model  # noqa: E402
from fdu.ours import marker_scores, new_proxy  # noqa: E402

import numpy as np  # noqa: E402

from verify_ref import consensus, perturbed_acc  # noqa: E402


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--force", action="store_true", help="re-score models already in scores.json")
    p.add_argument("--eval_bs", type=int, default=128, help="evaluation batch size (eval mode: does not change scores)")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    names = DOMAINS[cfg.dataset]
    t = names.index(a.target)
    out = os.path.join(cfg.run_dir(), "verify", a.variant, names[t])
    sj = json.load(open(os.path.join(out, "scores.json")))
    va = sj["args"]
    eps, anchor, lam = va["eps"], va["anchor"], va["lam"]
    clip = (-1.0, 1.0)
    fh = open(os.path.join(out, "log.txt"), "a")
    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    rec = np.load(os.path.join(cfg.run_dir(), "full", "forget_records.npz"))[f"arr_{t}"]
    rep = select(rec, lam, "top")
    rep_ld = data.subset_loader(t, rep, split="train_eval", bs=a.eval_bs)
    test_ld = data.test_loaders(bs=a.eval_bs)[t]
    proxy = new_proxy("unet", cfg.device, cfg.seed)
    proxy.load_state_dict(torch.load(os.path.join(out, "proxy.pt")))
    proxy.eval()
    K = int(sj.get("proxy_meta", {}).get("n_proxy", 1))
    proxies = [proxy]
    for i in range(1, K):  # K-proxy consensus variant: proxy{i}.pt
        pr = new_proxy("unet", cfg.device, cfg.seed + 1000 * i)
        pr.load_state_dict(torch.load(os.path.join(out, f"proxy{i}.pt")))
        pr.eval()
        proxies.append(pr)
    net = build_model(cfg.dataset, cfg.device)
    done = []
    for d in sorted(glob.glob(os.path.join(cfg.run_dir(), "unlearn", "*", names[t], "weight.pt"))):
        k = d.split(os.sep)[-3]
        if k in sj["scores"] and not a.force:
            continue
        net.load_state_dict(torch.load(d))
        rs = marker_scores(net, proxy, rep_ld, eps, anchor, cfg.device, clip)
        ts = marker_scores(net, proxy, test_ld, eps, anchor, cfg.device, clip)
        rs["pert_acc"] = perturbed_acc(net, proxy, rep_ld, eps, anchor, cfg.device, clip)
        ts["pert_acc"] = perturbed_acc(net, proxy, test_ld, eps, anchor, cfg.device, clip)
        if K > 1:  # same convention as verify_ref.py: "marker" = consensus, proxy 0 alone kept as "marker_p0"
            rs["marker_p0"], ts["marker_p0"] = rs["marker"], ts["marker"]
            rs.update(consensus(net, proxies, rep_ld, eps, anchor, cfg.device, clip))
            ts.update(consensus(net, proxies, test_ld, eps, anchor, cfg.device, clip))
        sj["scores"][k] = {"rep": rs, "test": ts}
        done.append(k)
        log(f"[rescore] {k:>16s}: marker(rep) {rs['marker']:6.2f}  acc-under-T(rep) {rs['pert_acc']:6.2f}  "
            f"acc-under-T(test) {ts['pert_acc']:6.2f}  clean-acc(rep) {rs['clean_acc']:6.2f}", fh)
    sj["rescored"] = sorted(set(sj.get("rescored", [])) | set(done))
    save_json(sj, os.path.join(out, "scores.json"))


if __name__ == "__main__":
    main()
