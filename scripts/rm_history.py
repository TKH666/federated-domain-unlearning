#!/usr/bin/env python
"""Free disk space: delete <run_dir>/full/history (per-round client models, only used by FedEraser) once every
FedEraser run that needs it has finished. Refuses to touch the main runs (local_ep 10 Digits/Office, 5 DomainNet).

python scripts/rm_history.py --dataset digits --local_ep 5 --targets SVHN,MNIST-M --fe_dir fe_recal
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import add_cfg_args, cfg_from_args  # noqa: E402

MAIN_LE = {"digits": 10, "office": 10, "domainnet": 5}


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--targets", required=True)
    p.add_argument("--fe_dir", default="fe_recal")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    if cfg.local_ep == MAIN_LE.get(cfg.dataset) or cfg.tag:
        sys.exit(f"refusing: {cfg.run_dir()} is a main run")
    run = cfg.run_dir()
    missing = [t for t in a.targets.split(",")
               if not os.path.exists(os.path.join(run, "unlearn", a.fe_dir, t, "metrics.json"))]
    if missing:
        sys.exit(f"keeping history: FedEraser not finished for {missing}")
    h = os.path.join(run, "full", "history")
    if os.path.isdir(h):
        shutil.rmtree(h)
        print(f"deleted {h}")


if __name__ == "__main__":
    main()
