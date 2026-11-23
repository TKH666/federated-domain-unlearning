#!/usr/bin/env python
"""E7: DIP feature inversion against the original and unlearned models of one target domain.

Output: <run_dir>/verify/inversion/<target>/{results.json, recon.npz, fig7_grid.png}
results.json: PSNR / SSIM / relative feature loss per model x block x image.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdu.common import add_cfg_args, cfg_from_args, log, save_json, set_seed  # noqa: E402
from fdu.data import DOMAINS, NORMALIZED, FedData  # noqa: E402
from fdu.inversion import invert, psnr, ssim, vgg_blocks  # noqa: E402
from fdu.models import build_model  # noqa: E402

LABEL = {"Orig": "Original", "retrain": "Retrain", "rr": "RR", "fe": "FE", "pga": "PGA", "cp": "CP",
         "fedosd": "FedOSD", "not": "NoT", "fedunran": "FedUNRAN"}


def main():
    p = add_cfg_args(argparse.ArgumentParser())
    p.add_argument("--target", required=True)
    p.add_argument("--blocks", default="0,1,2,3,4,5,6,7")
    p.add_argument("--n_images", type=int, default=5)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--dip_lr", type=float, default=0.01)
    p.add_argument("--models", default="Orig,retrain,rr,fe,pga,cp")
    p.add_argument("--out_tag", default="")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    assert cfg.dataset in ("office", "domainnet"), "inversion is defined for the VGG16 datasets"
    names = DOMAINS[cfg.dataset]
    t = int(a.target) if a.target.isdigit() else names.index(a.target)
    out = os.path.join(cfg.run_dir(), "verify", "inversion" + (f"_{a.out_tag}" if a.out_tag else ""), names[t])
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "log.txt"), "a")
    set_seed(cfg.seed)
    data = FedData(cfg.dataset, cfg.percent, cfg.local_bs, num_workers=0)
    test = data.test[t]
    idx = np.random.RandomState(cfg.seed).choice(len(test), a.n_images, replace=False)
    if NORMALIZED[cfg.dataset]:
        to01, norm = (lambda x: x * 0.5 + 0.5), (lambda x: (x - 0.5) / 0.5)
    else:
        to01, norm = (lambda x: x), (lambda x: x)
    imgs = [to01(test[int(i)][0]).unsqueeze(0).to(cfg.device) for i in idx]

    paths = {"Orig": os.path.join(cfg.run_dir(), "full", "weight_global.pt")}
    for d in glob.glob(os.path.join(cfg.run_dir(), "unlearn", "*", names[t], "weight.pt")):
        paths[d.split(os.sep)[-3]] = d
    models = [m for m in a.models.split(",") if m in paths]
    blocks = [int(b) for b in a.blocks.split(",")]
    log(f"target={names[t]} models={models} blocks={blocks} images={idx.tolist()} iters={a.iters}", fh)

    net = build_model(cfg.dataset, cfg.device)
    res, recon = {}, {}
    for m in models:
        net.load_state_dict(torch.load(paths[m]))
        net.eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        phi = vgg_blocks(net)
        res[m] = {}
        for l in blocks:
            rows = []
            for k, x in enumerate(imgs):
                rec, fl = invert(phi, x, l, norm, iters=a.iters, lr=a.dip_lr, seed=cfg.seed + k, device=cfg.device)
                rows.append({"psnr": psnr(rec, x), "ssim": ssim(rec, x), "feat_loss": fl})
                recon[f"{m}_b{l}_i{k}"] = (rec[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            res[m][l] = {"psnr": float(np.mean([r["psnr"] for r in rows])),
                         "ssim": float(np.mean([r["ssim"] for r in rows])),
                         "feat_loss": float(np.mean([r["feat_loss"] for r in rows])), "per_image": rows}
            log(f"{m:>8s} block{l}: PSNR {res[m][l]['psnr']:.2f}  SSIM {res[m][l]['ssim']:.3f}  "
                f"feat-loss {res[m][l]['feat_loss']:.4f}", fh)
    for k, x in enumerate(imgs):
        recon[f"input_i{k}"] = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    np.savez_compressed(os.path.join(out, "recon.npz"), **recon)
    save_json({"target": names[t], "images": idx.tolist(), "blocks": blocks, "iters": a.iters, "dip_lr": a.dip_lr,
               "results": res}, os.path.join(out, "results.json"))
    try:  # Fig. 7-style grid for image 0
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(models), len(blocks) + 1, figsize=(1.6 * (len(blocks) + 1), 1.6 * len(models)))
        ax = np.atleast_2d(ax)
        for r, m in enumerate(models):
            ax[r, 0].imshow(recon["input_i0"])
            ax[r, 0].set_ylabel(LABEL.get(m, m), fontsize=9)
            for c, l in enumerate(blocks):
                ax[r, c + 1].imshow(recon[f"{m}_b{l}_i0"])
                if r == 0:
                    ax[r, c + 1].set_title(f"Block{l}" + (f" (FC{l - 4})" if l >= 5 else ""), fontsize=8)
            if r == 0:
                ax[r, 0].set_title("Image", fontsize=8)
        for x_ in ax.ravel():
            x_.set_xticks([])
            x_.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out, "fig7_grid.png"), dpi=200)
        plt.savefig(os.path.join(out, "fig7_grid.pdf"))
    except Exception as e:  # noqa: BLE001
        log(f"grid plot failed: {e}", fh)


if __name__ == "__main__":
    main()
