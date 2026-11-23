"""Feature inversion with a Deep Image Prior (Ulyanov et al., CVPR'18 [45]).

Threat model: white-box access to the (unlearned) global model; the adversary observes the
intermediate representation Phi_l(x) of a forgotten-domain sample x at block l and reconstructs x by
optimizing the weights theta of a randomly initialised generator g_theta with a FIXED random input z:
    theta* = argmin_theta ||Phi_l(g_theta(z)) - Phi_l(x)||_2^2 / ||Phi_l(x)||_2^2 ,   x_hat = g_theta*(z).
VGG16-BN blocks (as in Fig. 7): block0..block4 = the five convolutional stages (each ends with the
pooling layer), block5..block7 = FC1..FC3 of the classifier.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def vgg_blocks(net):
    """Split original vgg into callables returning Phi_l(x) for l = 0..7."""
    feats = list(net.feature)
    stages, cur = [], []
    for m in feats:
        cur.append(m)
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            stages.append(nn.Sequential(*cur))
            cur = []
    assert len(stages) == 5, len(stages)
    cls = net.classifier  # Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear

    def phi(x, l):
        h = x
        for s in stages[: min(l, 4) + 1]:
            h = s(h)
        if l <= 4:
            return h
        h = F.avg_pool2d(h, 2, 2).view(h.size(0), -1)  # as in vgg.forward
        h = cls[1](cls[0](h))                            # FC1 (+ReLU)
        if l == 5:
            return h
        h = cls[4](cls[3](h))                            # FC2 (+ReLU), dropout inactive in eval
        if l == 6:
            return h
        return cls[6](h)                                 # FC3 (logits)
    return phi


# ---------------------------------------------------------------------------
class DIPGenerator(nn.Module):
    """Encoder-decoder with skip connections (the 'skip' architecture of the DIP paper, 5 scales,
    128 channels, 4-channel skips, LeakyReLU, bilinear upsampling); input: fixed noise z (32 ch)."""

    def __init__(self, in_ch=32, ch=128, skip=4, scales=5, out_ch=3):
        super().__init__()
        self.scales = scales
        self.down, self.skip, self.up = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = in_ch
        for _ in range(scales):
            self.skip.append(nn.Sequential(nn.Conv2d(c, skip, 1), nn.BatchNorm2d(skip), nn.LeakyReLU(0.2, True)))
            self.down.append(nn.Sequential(
                nn.Conv2d(c, ch, 3, 2, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, True)))
            c = ch
        for i in range(scales):
            self.up.append(nn.Sequential(
                nn.BatchNorm2d(ch + skip),
                nn.Conv2d(ch + skip, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, True)))
        self.out = nn.Conv2d(ch, out_ch, 1)

    def forward(self, z):
        skips, h = [], z
        for i in range(self.scales):
            skips.append(self.skip[i](h))
            h = self.down[i](h)
        for i in reversed(range(self.scales)):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            h = self.up[i](torch.cat([h, skips[i]], 1))
        return torch.sigmoid(self.out(h))


def invert(phi, x, l, normalize, iters=2000, lr=0.01, seed=0, device="cuda", reg_noise=0.03):
    """x: a single image in [0,1], shape [1,3,H,W]. normalize: callable mapping [0,1] -> model input.
    Returns the reconstruction in [0,1] and the final relative feature loss."""
    torch.manual_seed(seed)
    g = DIPGenerator().to(device)
    z0 = torch.rand(1, 32, x.shape[-2], x.shape[-1], device=device) * 0.1
    with torch.no_grad():
        target = phi(normalize(x), l)
        denom = target.pow(2).sum().clamp_min(1e-12)
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    g.train()
    for it in range(iters):
        z = z0 + reg_noise * torch.randn_like(z0)       # DIP input regularisation
        out = g(z)
        loss = (phi(normalize(out), l) - target).pow(2).sum() / denom
        opt.zero_grad()
        loss.backward()
        opt.step()
    g.eval()
    with torch.no_grad():
        rec = g(z0)
        final = ((phi(normalize(rec), l) - target).pow(2).sum() / denom).item()
    return rec.clamp(0, 1), final


# ---------------------------------------------------------------------------
def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(a, b, win=11, sigma=1.5):
    """Mean SSIM over channels, Gaussian window (Wang et al. 2004), inputs in [0,1], [1,3,H,W]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(win, dtype=a.dtype, device=a.device) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(a.shape[1], 1, win, win).contiguous()
    f = lambda t: F.conv2d(t, w, groups=a.shape[1])  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    s_a, s_b, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))
    return m.mean().item()
