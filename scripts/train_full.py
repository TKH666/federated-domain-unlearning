#!/usr/bin/env python
"""Original federated training over all domains ("Full learn"); original Full_Learn.py (FPL) /
learning.py (FedAvg).

Saves to <run_dir>/full/:
  weight_init.pt, weight_global.pt, weight_local.pt (final-round client states), global_protos.pt,
  forget_records.npz (per-client [rounds, n] correctness of the global model on local train data),
  metrics.json (per-round test accuracy), history/round_XXX.pt (optional; client models trained in
  round t + global model after round t, needed by FedEraser).
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import Timer, add_cfg_args, cfg_from_args, cpu_sd, log, save_json, set_seed  # noqa: E402
from fdu.data import FedData  # noqa: E402
from fdu.fl import ForgetRecorder, accuracy, fedavg, local_update, proto_aggregation  # noqa: E402
from fdu.models import build_model  # noqa: E402


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--history", action="store_true", help="save per-round client/global models (FedEraser)")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    out = os.path.join(cfg.run_dir(), "full")
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    set_seed(cfg.seed)
    log(f"cfg={cfg} history={a.history}", fh)

    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, cfg.num_workers)
    log(f"data {data.summary()}", fh)
    train_loaders = data.train_loaders(seed=cfg.seed)
    test_loaders = data.test_loaders()
    rec = ForgetRecorder([len(d) for d in data.train])

    net = build_model(cfg.dataset, cfg.device)
    w_init = cpu_sd(net.state_dict())
    global_protos, local_protos = [], {}
    hist = {"test_acc": [], "train_acc_aug": [], "round_time": []}
    total = Timer()
    for r in range(cfg.rounds):
        t = Timer()
        w_locals, losses = [], []
        for c in range(data.num_users):
            local = copy.deepcopy(net).to(cfg.device)
            _, w, protos, loss = local_update(cfg.backbone, local, train_loaders[c], global_protos, cfg)
            w_locals.append(cpu_sd(w))
            local_protos[c] = protos
            losses.append(loss)
            del local
        if cfg.backbone == "fpl":
            global_protos = proto_aggregation(local_protos)
        net.load_state_dict(fedavg(w_locals))
        train_time = t()
        tr_acc = rec.record(net, train_loaders, cfg.device)  # forgetting events (original semantics)
        te_acc = [accuracy(net, ld, cfg.device) for ld in test_loaders]
        hist["test_acc"].append(te_acc)
        hist["train_acc_aug"].append(tr_acc)
        hist["round_time"].append(train_time)
        if a.history:
            os.makedirs(os.path.join(out, "history"), exist_ok=True)
            torch.save({"locals": w_locals, "global_after": cpu_sd(net.state_dict())},
                       os.path.join(out, "history", f"round_{r:03d}.pt"))
        log(f"round {r:3d} loss {np.mean(losses):.4f} train(aug) {np.round(tr_acc, 2).tolist()} "
            f"test {np.round(te_acc, 2).tolist()} time {train_time:.1f}s", fh)

    torch.save(w_init, os.path.join(out, "weight_init.pt"))
    torch.save(cpu_sd(net.state_dict()), os.path.join(out, "weight_global.pt"))
    torch.save(w_locals, os.path.join(out, "weight_local.pt"))
    torch.save(global_protos, os.path.join(out, "global_protos.pt"))
    np.savez_compressed(os.path.join(out, "forget_records.npz"), *rec.arrays())
    np.savez_compressed(os.path.join(out, "loss_records.npz"), *rec.loss_arrays())
    tre = [accuracy(net, ld, cfg.device) for ld in data.train_eval_loaders()]
    save_json({"cfg": cfg.__dict__, "names": data.names, "final_test_acc": hist["test_acc"][-1],
               "final_train_acc": tre, "history": hist, "train_time_sec": float(sum(hist["round_time"])),
               "wall_time_sec": total()}, os.path.join(out, "metrics.json"))
    log(f"done. final test {np.round(hist['test_acc'][-1], 2).tolist()} train {np.round(tre, 2).tolist()} "
        f"wall {total():.0f}s", fh)


if __name__ == "__main__":
    main()
