import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

RUNS = os.environ.get("FDU_RUNS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))


@dataclass
class Cfg:
    dataset: str = "office"          # digits | office | domainnet
    backbone: str = "fpl"            # fpl | fedavg
    seed: int = 2024
    rounds: int = 50
    local_ep: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    local_bs: int = 64
    percent: float = 1.0             # digits: fraction of FedBN partitions
    infonce_t: float = 0.02          # FPL InfoNCE temperature (original default)
    num_workers: int = 4
    gpu: int = 0
    tag: str = ""
    device: str = field(default="cuda", repr=False)

    def run_dir(self):
        d = os.path.join(RUNS, self.dataset, self.backbone, f"le{self.local_ep}", f"seed{self.seed}")
        return d + (f"_{self.tag}" if self.tag else "")


def add_cfg_args(p):
    for k, v in asdict(Cfg()).items():
        if k == "device":
            continue
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p


def cfg_from_args(a):
    c = Cfg(**{k: getattr(a, k) for k in asdict(Cfg()) if k != "device"})
    c.device = f"cuda:{c.gpu}" if torch.cuda.is_available() and c.gpu >= 0 else "cpu"
    return c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def cpu_sd(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def __call__(self):
        return time.perf_counter() - self.t


def log(msg, fh=None):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()
