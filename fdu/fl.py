"""Federated learning core, ported from the authors' original code.

- fpl_local_update   : models/Update_domain.py::DomainClientUpdate.train  (FPL, CVPR'23)
- avg_local_update   : models/Update_domain.py::DomainClientUpdate_avg.train (FedAvg backbone)
- proto_aggregation  : utils/fpl.py::proto_aggregation (FINCH clustering of client prototypes)
- fedavg             : models/Fed.py::FedAvg (uniform average over participating clients)
- evaluate           : utils/evaluate.py::test (accuracy + per-sample correctness record used for
                       forgetting events: the GLOBAL model is evaluated on each client's training
                       data after every round)
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from fdu._fpl import FINCH
from fdu.models import logits_of


# ---------------------------------------------------------------------------
# FPL local update (verbatim semantics)
# ---------------------------------------------------------------------------
def _calculate_infonce(tau, f_now, f_pos, f_neg, device):
    f_proto = torch.cat((f_pos, f_neg), dim=0)
    l = torch.cosine_similarity(f_now, f_proto, dim=1) / tau
    exp_l = torch.exp(l).view(1, -1)
    pos_mask = torch.tensor([1] * f_pos.shape[0] + [0] * f_neg.shape[0], dtype=torch.float32,
                            device=device).view(1, -1)
    return -torch.log((exp_l * pos_mask).sum(1) / exp_l.sum(1))


def _hierarchical_info_loss(tau, f_now, label, all_f, mean_f, keys, device):
    f_pos = all_f[keys == label.item()][0].to(device)
    f_neg = torch.cat(all_f[keys != label.item()].tolist()).to(device)
    xi = _calculate_infonce(tau, f_now, f_pos, f_neg, device)
    mean_f_pos = mean_f[keys == label.item()][0].to(device).view(1, -1)
    cu = nn.MSELoss()(f_now.to(torch.float32), mean_f_pos.to(torch.float32))
    return xi + cu


def _batch_info_loss(tau, feat, labels, P, P_lab, mean_tab, has_key):
    """Vectorized, mathematically identical form of the per-sample hierarchical InfoNCE loop:
    for every sample whose label has global prototypes,
      InfoNCE_i = -log( sum_{p in protos(y_i)} e^{cos(f_i,p)/tau} / sum_{p in all protos} e^{cos(f_i,p)/tau} )
      MSE_i     = mean((f_i - mean_proto(y_i))^2)
    summed and divided by the batch size (as the loop divides by the sample counter)."""
    lab_ok = labels < has_key.numel()
    valid = torch.zeros_like(labels, dtype=torch.bool)
    valid[lab_ok] = has_key[labels[lab_ok]]
    eps = 1e-8  # torch.cosine_similarity clamps the norm product at eps
    num = feat.float() @ P.float().T
    den = torch.clamp(feat.float().norm(dim=1, keepdim=True) * P.float().norm(dim=1)[None, :], min=eps)
    e = torch.exp(num / den / tau)
    pos = (P_lab[None, :] == labels[:, None]).float()
    tot = e.sum(1)
    pos_sum = torch.where(valid, (e * pos).sum(1), tot)  # rows without prototypes -> log(1)=0
    info = -torch.log(pos_sum / tot)
    mse = ((feat.float() - mean_tab[labels.clamp(max=has_key.numel() - 1)]) ** 2).mean(1)
    return ((info + mse) * valid.float()).sum() / labels.numel()


def _agg_func(protos):
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos


def fpl_local_update(net, loader, global_protos, local_ep, lr, momentum, device, tau=0.02, vectorized=True):
    """Returns (net, state_dict, agg_protos, mean_loss)."""
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    ce = nn.CrossEntropyLoss()
    if len(global_protos) != 0:
        keys = np.array(list(global_protos.keys()))
        all_f, mean_f = [], []
        for k in keys:
            t = torch.cat(global_protos[k], dim=0).to(device)
            all_f.append(t.cpu())
            mean_f.append(torch.mean(t, dim=0).cpu())
        all_f_np = np.empty(len(all_f), dtype=object)
        mean_f_np = np.empty(len(mean_f), dtype=object)
        for i in range(len(all_f)):
            all_f_np[i] = all_f[i].detach()
            mean_f_np[i] = mean_f[i].detach()
        all_f, mean_f = all_f_np, mean_f_np
        # vectorized tables: all prototypes (with their label), per-label mean prototype
        P = torch.cat([all_f[i] for i in range(len(keys))], 0).to(device)
        P_lab = torch.cat([torch.full((all_f[i].shape[0],), int(keys[i])) for i in range(len(keys))]).to(device)
        n_cls = int(max(keys)) + 1
        mean_tab = torch.zeros(n_cls, P.shape[1], device=device)
        has_key = torch.zeros(n_cls, dtype=torch.bool, device=device)
        for i, k in enumerate(keys):
            mean_tab[int(k)] = mean_f[i].to(device)
            has_key[int(k)] = True
    epoch_loss = []
    agg_protos_label = {}
    for it in range(local_ep):
        net.train()
        batch_loss = []
        agg_protos_label = {}
        for images, labels, _ in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            net.zero_grad()
            log_probs, feature = net(images, output_feature=True)
            loss_ce = ce(log_probs, labels)
            if len(global_protos) == 0:
                loss_info = 0 * loss_ce
            elif vectorized:
                loss_info = _batch_info_loss(tau, feature, labels, P, P_lab, mean_tab, has_key)
            else:  # original per-sample loop (reference implementation)
                loss_info, n = None, 0
                for i, label in enumerate(labels):
                    if label.item() in global_protos.keys():
                        li = _hierarchical_info_loss(tau, feature[i].unsqueeze(0), label, all_f, mean_f,
                                                     keys, device).to(torch.float32)
                        loss_info = li if loss_info is None else loss_info + li
                    n = i + 1
                loss_info = loss_info / n
            loss = loss_ce + loss_info
            loss.backward()
            batch_loss.append(loss.item())
            opt.step()
            if it == local_ep - 1:  # prototypes from the last local epoch
                for i in range(len(labels)):
                    agg_protos_label.setdefault(labels[i].item(), []).append(feature[i, :].detach())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    agg = _agg_func(agg_protos_label)
    return net, net.state_dict(), agg, sum(epoch_loss) / len(epoch_loss)


def avg_local_update(net, loader, local_ep, lr, momentum, device, weight_decay=1e-4, clip_grad=5.0):
    """FedAvg client update (original DomainClientUpdate_avg): SGD(m=0.9, wd=1e-4) + grad clipping."""
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for _ in range(local_ep):
        ep = 0.0
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = ce(logits_of(net(x)), y)
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            ep += loss.item()
        total += ep / len(loader)
    return net, net.state_dict(), {}, total / local_ep


def local_update(backbone, net, loader, global_protos, cfg):
    if backbone == "fpl":
        return fpl_local_update(net, loader, global_protos, cfg.local_ep, cfg.lr, cfg.momentum, cfg.device,
                                cfg.infonce_t)
    if backbone == "fedavg":
        return avg_local_update(net, loader, cfg.local_ep, cfg.lr, cfg.momentum, cfg.device)
    raise ValueError(backbone)


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------
def proto_aggregation(local_protos, exclude=()):
    """FINCH-clustered prototypes per label over participating clients (original utils/fpl.py)."""
    agg = {}
    for idx, protos in local_protos.items():
        if idx in exclude:
            continue
        for label, p in protos.items():
            agg.setdefault(label, []).append(p)
    for label, plist in agg.items():
        if len(plist) > 1:
            arr = np.array([p.squeeze(0).detach().cpu().numpy().reshape(-1) for p in plist])
            c, _, _ = FINCH(arr, initial_rank=None, req_clust=None, distance="cosine",
                            ensure_early_exit=False, verbose=False)
            cl = c[:, -1]
            agg[label] = [torch.tensor(np.mean(arr[cl == u], axis=0, keepdims=True)) for u in np.unique(cl)]
        else:
            agg[label] = [plist[0].data]
    return agg


def fedavg(w_list):
    """Uniform average (original models/Fed.py::FedAvg). Integer buffers are averaged then cast back."""
    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg.keys():
        if w_avg[k].is_floating_point():
            for i in range(1, len(w_list)):
                w_avg[k] += w_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w_list))
        else:  # num_batches_tracked
            acc = w_avg[k].clone().double()
            for i in range(1, len(w_list)):
                acc += w_list[i][k].double()
            w_avg[k] = torch.div(acc, len(w_list)).to(w_avg[k].dtype)
    return w_avg


# ---------------------------------------------------------------------------
# Evaluation and forgetting-event recording
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(net, loader, device, with_loss=False):
    """Returns (indices, labels, predictions[, per-sample CE loss]) as numpy arrays."""
    net.eval()
    I, Y, P, L = [], [], [], []
    for x, y, idx in loader:
        out = logits_of(net(x.to(device, non_blocking=True)))
        P.append(out.argmax(1).cpu())
        Y.append(y)
        I.append(idx)
        if with_loss:
            L.append(torch.nn.functional.cross_entropy(out, y.to(out.device), reduction="none").cpu())
    res = (torch.cat(I).numpy(), torch.cat(Y).numpy(), torch.cat(P).numpy())
    return res + (torch.cat(L).numpy(),) if with_loss else res


@torch.no_grad()
def accuracy(net, loader, device):
    _, y, p = predict(net, loader, device)
    return 100.0 * float((y == p).mean())


class ForgetRecorder:
    """Per client: correctness (and CE loss) of the global model on every training sample after
    each round.

    original evaluate() records on the (augmented, shuffled) TRAIN loaders; we keep that, keyed by
    the sample index: `arrays()` -> int8 [rounds, n] correctness (forgetting events, ours);
    `loss_arrays()` -> float16 [rounds, n] per-sample loss (VeriFi vFM marker selection).
    """

    def __init__(self, sizes):
        self.rows = [[] for _ in sizes]
        self.loss_rows = [[] for _ in sizes]
        self.sizes = sizes

    def record(self, net, loaders, device):
        accs = []
        for c, ld in enumerate(loaders):
            idx, y, p, l = predict(net, ld, device, with_loss=True)
            row = np.zeros(self.sizes[c], dtype=np.int8)
            row[idx] = (y == p).astype(np.int8)
            lrow = np.zeros(self.sizes[c], dtype=np.float16)
            lrow[idx] = l.astype(np.float16)
            self.rows[c].append(row)
            self.loss_rows[c].append(lrow)
            accs.append(100.0 * float(row.mean()))
        return accs

    def arrays(self):
        return [np.stack(r) if r else np.zeros((0, n), np.int8) for r, n in zip(self.rows, self.sizes)]

    def loss_arrays(self):
        return [np.stack(r) if r else np.zeros((0, n), np.float16) for r, n in zip(self.loss_rows, self.sizes)]
