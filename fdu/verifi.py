"""VeriFi (Gao et al., 2024) unique-memory markers, following the paper (Sec. III, Alg. 2,
Table II, Table IX):

vFM (forgettable memory): markers = top-10% samples of the leaving client by the variance of their
  training loss across communication rounds. Marking: the leaving client fine-tunes its local model
  on the markers (SGD lr 0.01, momentum 0.9, wd 2e-4; 100 epochs for 32x32 data, 50 otherwise) to
  reduce their loss variance and uploads it; the server aggregates it in the marking round.
  Checking metric: variance of the per-sample loss on the markers (low = memorized; effective
  unlearning recovers a high variance).
vEM (erroneous memory, Alg. 2): D_l = top-10% highest-loss samples of the leaving client under the
  global model; markers = samples of the majority (true) class in D_l; every marker is relabeled to
  the label most predicted for the markers; the client fine-tunes on (D_a \\ D_m) U D_m(relabeled)
  and uploads. Checking metric: CE loss of the markers w.r.t. their erroneous labels (low = memorized).
Marking round (t_m): the leaving client performs marking while the other clients perform a normal
local update; the server aggregates all updates (one extra FL round after the original training).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from fdu.fl import fedavg, local_update, predict
from fdu.models import logits_of


class Relabeled(Dataset):
    def __init__(self, base, indices, labels):
        self.base, self.indices, self.labels = base, list(indices), dict(zip(indices, labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, _, idx = self.base[self.indices[i]]
        return x, int(self.labels[self.indices[i]]), idx


def select_vfm(loss_records, ratio=0.1):
    var = np.asarray(loss_records, dtype=np.float32).var(axis=0)
    k = max(1, int(round(ratio * len(var))))
    return np.sort(np.argsort(var)[-k:]), None


def select_vem(net, data, client, device, kappa=0.1):
    idx, y, p, loss = predict(net, data.subset_loader(client, range(len(data.train[client])), split="train_eval",
                                                      bs=128), device, with_loss=True)
    k = max(1, int(round(kappa * len(idx))))
    top = np.argsort(loss)[-k:]
    maj = np.bincount(y[top]).argmax()
    sel = top[y[top] == maj]
    markers = idx[sel]
    new_label = int(np.bincount(p[sel]).argmax())  # most predicted label on the markers
    return np.sort(markers), new_label


def mark_finetune(net, loader, epochs, device, lr=0.01, momentum=0.9, wd=2e-4):
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce(logits_of(net(x)), y).backward()
            opt.step()
    return net


def marking_round(cfg, data, global_sd, target, variant, markers, new_label, mark_epochs, build, loaders):
    """Returns (marked global state, list of local states after the marking round)."""
    dev = cfg.device
    w_locals = []
    for c in range(data.num_users):
        net = build()
        net.load_state_dict(global_sd)
        if c == target:
            base = data.train[c]
            if variant == "vfm":
                ds = Subset(base, list(markers))
            else:  # vem: (D_a \ D_m) U D_m relabeled
                rest = [i for i in range(len(base)) if i not in set(markers.tolist())]
                ds = torch.utils.data.ConcatDataset([Subset(base, rest),
                                                     Relabeled(base, markers.tolist(), [new_label] * len(markers))])
            g = torch.Generator().manual_seed(cfg.seed)
            ld = DataLoader(ds, batch_size=cfg.local_bs, shuffle=True, generator=g, num_workers=cfg.num_workers)
            mark_finetune(net, ld, mark_epochs, dev)
            w = net.state_dict()
        else:
            _, w, _, _ = local_update(cfg.backbone, net, loaders[c], [], cfg)
        w_locals.append({k: v.detach().cpu().clone() for k, v in w.items()})
        del net
    return fedavg(w_locals), w_locals


@torch.no_grad()
def check_metric(net, data, target, variant, markers, new_label, device):
    """Native VeriFi checking metric of `net` on the markers."""
    ld = data.subset_loader(target, markers, split="train_eval", bs=128)
    net.eval()
    losses = []
    for x, y, _ in ld:
        out = logits_of(net(x.to(device)))
        yy = y.to(device) if variant == "vfm" else torch.full_like(y, new_label).to(device)
        losses.append(F.cross_entropy(out, yy, reduction="none").cpu())
    l = torch.cat(losses).numpy()
    return float(l.var()) if variant == "vfm" else float(l.mean())
