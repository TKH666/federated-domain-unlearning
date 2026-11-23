#!/usr/bin/env python
"""VeriFi baseline.

--stage mark : marking round on top of the original model -> marked original model in
               <run_dir>_mark-<variant>-<target>/full (init/history/records are symlinked). Run the
               unlearning methods that start from the global model (pga, cp, fedosd, not, fedunran)
               with --tag mark-<variant>-<target>; Retrain / RR / FE do not use the leaving client's
               updates and are reused from the base run.
--stage score: native checking metric for every model -> <run_dir>/verify/verifi_<variant>/<target>/scores.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import Timer, add_cfg_args, cfg_from_args, log, save_json, set_seed  # noqa: E402
from fdu.data import DOMAINS, FedData  # noqa: E402
from fdu.fl import accuracy  # noqa: E402
from fdu.models import build_model  # noqa: E402
from fdu.verifi import check_metric, marking_round, select_vem, select_vfm  # noqa: E402

REUSE_FROM_BASE = ("retrain", "rr", "fe", "fe_recal")


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--variant", required=True, choices=["vfm", "vem"])
    p.add_argument("--stage", required=True, choices=["mark", "score"])
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--mark_epochs", type=int, default=-1, help="-1: 100 for digits, 50 otherwise (VeriFi Tab. IX)")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    assert not cfg.tag.startswith("mark-"), "use the base run, not a marked run"
    names = DOMAINS[cfg.dataset]
    t = int(a.target) if a.target.isdigit() else names.index(a.target)
    base = cfg.run_dir()
    mtag = f"mark-{a.variant}-{names[t]}"
    mdir = f"{base}_{mtag}"
    out = os.path.join(base, "verify", f"verifi_{a.variant}", names[t])
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    set_seed(cfg.seed)
    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    build = lambda: build_model(cfg.dataset, cfg.device)  # noqa: E731

    if a.stage == "mark":
        orig = build()
        orig.load_state_dict(torch.load(os.path.join(base, "full", "weight_global.pt")))
        if a.variant == "vfm":
            loss_rec = np.load(os.path.join(base, "full", "loss_records.npz"))[f"arr_{t}"]
            markers, new_label = select_vfm(loss_rec, a.ratio)
        else:
            markers, new_label = select_vem(orig, data, t, cfg.device, a.ratio)
        me = a.mark_epochs if a.mark_epochs > 0 else (100 if cfg.dataset == "digits" else 50)
        log(f"[{a.variant}] target {names[t]}: {len(markers)} markers, new_label={new_label}, mark_epochs={me}", fh)
        tm = Timer()
        g, locs = marking_round(cfg, data, orig.state_dict(), t, a.variant, markers, new_label, me, build,
                                data.train_loaders(seed=cfg.seed + 7))
        secs = tm()
        os.makedirs(os.path.join(mdir, "full"), exist_ok=True)
        for f in ["weight_init.pt", "history", "forget_records.npz", "loss_records.npz"]:
            src = os.path.join(base, "full", f)
            dst = os.path.join(mdir, "full", f)
            if os.path.exists(src) and not os.path.lexists(dst):
                os.symlink(src, dst)
        torch.save(g, os.path.join(mdir, "full", "weight_global.pt"))
        torch.save(locs, os.path.join(mdir, "full", "weight_local.pt"))
        marked = build()
        marked.load_state_dict(g)
        te_o = [accuracy(orig, ld, cfg.device) for ld in data.test_loaders()]
        te_m = [accuracy(marked, ld, cfg.device) for ld in data.test_loaders()]
        save_json({"variant": a.variant, "target": names[t], "markers": markers.tolist(), "new_label": new_label,
                   "mark_epochs": me, "marking_round_sec": secs, "test_acc_orig": te_o, "test_acc_marked": te_m},
                  os.path.join(out, "markers.json"))
        log(f"marked: test {np.round(te_m, 2).tolist()} (orig {np.round(te_o, 2).tolist()}) time {secs:.0f}s", fh)
        return

    info = json.load(open(os.path.join(out, "markers.json")))
    markers, new_label = np.array(info["markers"]), info["new_label"]
    models = {"Orig": os.path.join(base, "full", "weight_global.pt"),
              "Marked": os.path.join(mdir, "full", "weight_global.pt")}
    for m in REUSE_FROM_BASE:
        pth = os.path.join(base, "unlearn", m, names[t], "weight.pt")
        if os.path.exists(pth):
            models[m] = pth
    for pth in sorted(glob.glob(os.path.join(mdir, "unlearn", "*", names[t], "weight.pt"))):
        models[pth.split(os.sep)[-3]] = pth
    net = build()
    scores = {}
    for k, pth in models.items():
        net.load_state_dict(torch.load(pth))
        scores[k] = check_metric(net, data, t, a.variant, markers, new_label, cfg.device)
        log(f"{a.variant} {k:>10s}: {scores[k]:.4f}", fh)
    save_json({"variant": a.variant, "target": names[t], "metric": "loss_variance" if a.variant == "vfm" else
               "marker_loss", "scores": scores, "models": models}, os.path.join(out, "scores.json"))


if __name__ == "__main__":
    main()
