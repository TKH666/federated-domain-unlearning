"""Datasets and loaders, ported from original utils/init_data_model.py + utils/data_utils.py.

Every dataset returns (image, label, index) so that per-sample forgetting events can be recorded.
One client per domain (original default domain_split_factor = 1).

Faithful details:
- digits    : 32x32, MNIST/USPS grayscale->3ch, Normalize(0.5); `percent` = fraction of the 10
              FedBN partitions (percent=1 -> all 10 partitions); no augmentation.
- office    : Resize 224; train: RandomHorizontalFlip + RandomRotation(+-30); Normalize(0.5).
- domainnet : Resize 224; train: RandomHorizontalFlip + RandomRotation(+-30); NO Normalize
              (inputs in [0,1], as in the original code); train sets truncated to the smallest domain.
Optimization (not semantic): images are decoded+resized once into a shared uint8 array; the random
augmentations are still applied on the fly in the same order (Resize -> flip -> rotate -> tensor).
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

DATA_ROOT = os.environ.get("FDU_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

DOMAINS = {
    "digits": ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"],
    "office": ["amazon", "caltech", "dslr", "webcam"],
    "domainnet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
}
SHORT = {  # abbreviations used in the manuscript tables
    "clipart": "C", "infograph": "I", "painting": "P", "quickdraw": "Q", "real": "R", "sketch": "S",
}
_DIGIT_DIR = {"MNIST": ("MNIST", 1), "SVHN": ("SVHN", 3), "USPS": ("USPS", 1),
              "SynthDigits": ("SynthDigits", 3), "MNIST-M": ("MNIST_M", 3)}
OFFICE_LABELS = {"back_pack": 0, "bike": 1, "calculator": 2, "headphones": 3, "keyboard": 4,
                 "laptop_computer": 5, "monitor": 6, "mouse": 7, "mug": 8, "projector": 9}
DOMAINNET_LABELS = {"bird": 0, "feather": 1, "headphones": 2, "ice_cream": 3, "teapot": 4,
                    "tiger": 5, "whale": 6, "windmill": 7, "wine_glass": 8, "zebra": 9}

NORMALIZED = {"digits": True, "office": True, "domainnet": False}  # input range [-1,1] vs [0,1]


def input_range(dataset):
    return (-1.0, 1.0) if NORMALIZED[dataset] else (0.0, 1.0)


# ---------------------------------------------------------------------------
class DigitsDataset(Dataset):
    def __init__(self, domain, train, percent, transform):
        sub, self.channels = _DIGIT_DIR[domain]
        path = os.path.join(DATA_ROOT, "digitdata", sub)
        if train:
            if percent >= 0.1:
                imgs, labs = [], []
                for part in range(int(percent * 10)):
                    i, l = np.load(os.path.join(path, f"partitions/train_part{part}.pkl"), allow_pickle=True)
                    imgs.append(i)
                    labs.append(l)
                self.images, self.labels = np.concatenate(imgs), np.concatenate(labs)
            else:
                self.images, self.labels = np.load(os.path.join(path, "partitions/train_part0.pkl"),
                                                   allow_pickle=True)
                n = int(self.images.shape[0] * percent * 10)
                self.images, self.labels = self.images[:n], self.labels[:n]
        else:
            self.images, self.labels = np.load(os.path.join(path, "test.pkl"), allow_pickle=True)
        self.labels = self.labels.astype(np.int64).squeeze()
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode="L" if self.channels == 1 else "RGB")
        return self.transform(img), int(self.labels[idx]), idx


class CachedImageDataset(Dataset):
    """Office / DomainNet: image paths in the pkl are relative to DATA_ROOT.

    All images are decoded and resized once (transforms.Resize([s, s]) semantics, bilinear) into a
    shared uint8 array; transforms (flip/rotation/ToTensor/Normalize) are applied per access.
    """

    def __init__(self, pkl, label_map, transform, size=224, array=None):
        paths, text = np.load(pkl, allow_pickle=True)
        self.paths = [os.path.join(DATA_ROOT, p) for p in paths]
        self.labels = np.array([label_map[t] for t in text], dtype=np.int64)
        self.transform = transform
        self.size = size
        self.array = array if array is not None else self._decode_all()

    def _decode_one(self, path):
        img = Image.open(path).convert("RGB")  # == Grayscale(num_output_channels=3) for 1-ch images
        return np.asarray(img.resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)

    def _decode_all(self):
        from concurrent.futures import ThreadPoolExecutor
        arr = np.empty((len(self.paths), self.size, self.size, 3), dtype=np.uint8)
        with ThreadPoolExecutor(16) as ex:
            for i, a in enumerate(ex.map(self._decode_one, self.paths)):
                arr[i] = a
        return arr

    def with_transform(self, transform):
        other = object.__new__(CachedImageDataset)
        other.__dict__.update(self.__dict__)
        other.transform = transform
        return other

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.array[idx])), int(self.labels[idx]), idx


# ---------------------------------------------------------------------------
def _transforms(dataset, domain=None):
    norm = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] if NORMALIZED[dataset] else []
    if dataset == "digits":
        pre = [transforms.Grayscale(num_output_channels=3)] if domain in ("MNIST", "USPS") else []
        t = transforms.Compose(pre + [transforms.Resize([32, 32]), transforms.ToTensor()] + norm)
        return t, t
    train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation((-30, 30)),
                                transforms.ToTensor()] + norm)
    test = transforms.Compose([transforms.ToTensor()] + norm)
    return train, test


class FedData:
    """Holds per-domain train (augmented), train_eval (deterministic) and test datasets."""

    def __init__(self, dataset, percent=1.0, local_bs=64, num_workers=4):
        self.dataset = dataset
        self.names = DOMAINS[dataset]
        self.local_bs = local_bs
        self.num_workers = num_workers
        self.train, self.train_eval, self.test = [], [], []
        for d in self.names:
            tr_t, te_t = _transforms(dataset, d)
            if dataset == "digits":
                tr = DigitsDataset(d, True, percent, tr_t)
                tre = DigitsDataset(d, True, percent, te_t)
                te = DigitsDataset(d, False, percent, te_t)
            else:
                sub, lmap = ("office_caltech_10", OFFICE_LABELS) if dataset == "office" else ("DomainNet", DOMAINNET_LABELS)
                tr = CachedImageDataset(os.path.join(DATA_ROOT, sub, f"{d}_train.pkl"), lmap, tr_t)
                tre = tr.with_transform(te_t)  # same decoded images, deterministic transform
                te = CachedImageDataset(os.path.join(DATA_ROOT, sub, f"{d}_test.pkl"), lmap, te_t)
            self.train.append(tr)
            self.train_eval.append(tre)
            self.test.append(te)
        if dataset == "domainnet":  # original code: truncate every train set to the smallest domain
            m = min(len(t) for t in self.train)
            self.train = [Subset(t, list(range(m))) for t in self.train]
            self.train_eval = [Subset(t, list(range(m))) for t in self.train_eval]
        self.num_users = len(self.names)

    def labels(self, client, split="train"):
        ds = {"train": self.train, "test": self.test}[split][client]
        if isinstance(ds, Subset):
            return ds.dataset.labels[np.asarray(ds.indices)]
        return ds.labels

    # ---- loaders ----
    def _loader(self, ds, shuffle, seed=None, bs=None):
        g = None
        if shuffle and seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
        return DataLoader(ds, batch_size=bs or self.local_bs, shuffle=shuffle, generator=g,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def train_loaders(self, seed=None):
        return [self._loader(d, True, None if seed is None else seed + i) for i, d in enumerate(self.train)]

    def train_eval_loaders(self, bs=128):
        return [self._loader(d, False, bs=bs) for d in self.train_eval]

    def test_loaders(self, bs=128):
        return [self._loader(d, False, bs=bs) for d in self.test]

    def subset_loader(self, client, indices, split="train", shuffle=False, seed=None, bs=None):
        ds = {"train": self.train, "train_eval": self.train_eval, "test": self.test}[split][client]
        return self._loader(Subset(ds, list(indices)), shuffle, seed, bs)

    def summary(self):
        return {n: (len(self.train[i]), len(self.test[i])) for i, n in enumerate(self.names)}
