"""Model definitions (ported verbatim from the authors' original code).

- vgg / vgg_unlrean : models/vggmodule.py (VGG16-BN, 4608-d penultimate input, 10 classes)
- DigitModel / DigitModel_Pruning : models/Nets.py (3 conv + 2 max-pool + 3 FC, FedBN digit model)
- UNet / Autoencoder : models/backdoor.py (proxy validation model / marker generator)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fdu._vggmodule import vgg, vgg_unlrean  # noqa: F401  (verbatim copy)
from fdu._backdoor import UNet, Autoencoder  # noqa: F401  (verbatim copy)


class DigitModel(nn.Module):
    """Model for benchmark experiment on Digits (verbatim from original models/Nets.py)."""

    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x, output_feature=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        feature = x
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.fc3(x)
        if output_feature:
            return x, feature.view(feature.shape[0], -1)
        return x


class DigitModel_Pruning(nn.Module):
    """Feature extractor used by Class-Discriminative Pruning on digits (verbatim)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        conv1_out_non = self.conv1(x)
        x = F.relu(self.bn1(conv1_out_non))
        x = F.max_pool2d(x, 2)
        conv2_out_non = self.conv2(x)
        x = F.relu(self.bn2(conv2_out_non))
        x = F.max_pool2d(x, 2)
        conv3_out_non = self.conv3(x)
        conv1 = F.adaptive_avg_pool2d(F.relu(conv1_out_non), (1, 1))
        conv2 = F.adaptive_avg_pool2d(F.relu(conv2_out_non), (1, 1))
        conv3 = F.adaptive_avg_pool2d(F.relu(conv3_out_non), (1, 1))
        return 1, [conv1, conv2, conv3]


def build_model(dataset, device):
    """digits -> DigitModel ('cnn'); office / domainnet -> VGG16-BN (as in the manuscript)."""
    if dataset == "digits":
        net = DigitModel()
    elif dataset in ("office", "domainnet"):
        net = vgg(dataset="office-caltech10", depth=16, init_weights=True, cfg=None)
    else:
        raise ValueError(dataset)
    return net.to(device)


def build_pruning_model(dataset, device):
    if dataset == "digits":
        return DigitModel_Pruning().to(device)
    return vgg_unlrean(dataset="office-caltech10", depth=16, init_weights=True, cfg=None).to(device)


def build_proxy(kind, device):
    """Proxy validation model M_PV (Sec. V-A). Manuscript: U-Net for all datasets."""
    if kind == "unet":
        return UNet(3).to(device)
    if kind == "auto":
        return Autoencoder().to(device)
    raise ValueError(kind)


def logits_of(output):
    """Some forward() variants return (logits, feature)."""
    return output[0] if isinstance(output, (tuple, list)) else output
