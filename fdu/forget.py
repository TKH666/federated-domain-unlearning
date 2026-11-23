"""Forgetting events (original utils/forget_event.py, ignore=" " mode used by Verify.py).

records: int8 array [rounds, n_samples]; 1 = sample classified correctly by the global model.
forgetting count R_i = number of 1->0 transitions; never-learned samples get -1 (they sort first,
so they are never selected as representative). Representative samples = top-lambda by R_i.
Ties are broken by sample index (stable sort) for reproducibility.
"""
import numpy as np


def forgetting_counts(records):
    acc = np.asarray(records, dtype=np.int16)
    trans = acc[1:] - acc[:-1]
    counts = (trans == -1).sum(axis=0).astype(np.int64)
    never = acc.max(axis=0) == 0
    counts[never] = -1
    return counts


def rank_ascending(records):
    c = forgetting_counts(records)
    order = np.argsort(c, kind="stable")
    return order, c[order]


def select(records, lam, mode="top", seed=0):
    """Indices of the lambda-fraction of samples used to build markers.
    mode: top (paper), bottom, random, all."""
    order, _ = rank_ascending(records)
    n = len(order)
    k = int(n * lam)
    if mode == "top":
        return np.sort(order[n - k:])
    if mode == "bottom":
        return np.sort(order[:k])
    if mode == "random":
        return np.sort(np.random.RandomState(seed).choice(n, k, replace=False))
    if mode == "all":
        return np.arange(n)
    raise ValueError(mode)
