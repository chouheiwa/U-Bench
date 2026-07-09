#!/usr/bin/env python
"""P2 pilot: source-free test-time adaptation via physics self-calibration.

A source-trained calibrated model is adapted on the UNLABELED target val split
(source-free, label-free, transductive) by minimizing an objective, updating only
BatchNorm affine params (Tent-style). Then segmentation IoU is measured on the
same target split. Compares:
  naive     : no adaptation (lower bound = current cross-domain number)
  physcalib : minimize |estimator.last_phys - acoustic_pseudo_gt(feat)| (our P2)
  entropy   : minimize prediction entropy (Tent, generic SFDA reference)

Make-or-break for the P2 headline: does physics self-calibration lift the worst
cross-domain cell (busi->bus, calib naive ~0.729) above the naive baseline, and
does it match/beat generic entropy-min?

Usage: python tools/tta_physics.py --source busi --target bus --seed 42 \
         --mode physcalib --steps 20 --lr 1e-3 --bs 8
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["USEANET_CALIB_PROXY"] = "1"
os.environ.pop("USEANET_CALIB_PHYS", None)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_args(src, tgt, seed, bs):
    return SimpleNamespace(
        model="USEANet", model_id=115, img_size=256,
        base_dir=f"./data/{src}", dataset_name=src,
        batch_size=bs, seed=seed,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path="./pretrained",
        exp_save_dir=f"./output/USEANet/{src}/calib_{src}_s{seed}",
        train_file_dir="train.txt", val_file_dir="val.txt",
        zero_shot_dataset_name=tgt, zero_shot_base_dir=f"./data/{tgt}",
    )


def load_model(args):
    from models import build_model
    model = build_model(args, input_channel=3, num_classes=1,
                        pretrained_model_path=args.pretrained_model_path).to(device)
    ckpt = torch.load(os.path.join(args.exp_save_dir, "checkpoint_best.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model


def last_out(out):
    return out[-1] if isinstance(out, (list, tuple)) else out


def iou_on(model, loader):
    model.eval()
    ious = []
    with torch.no_grad():
        for batch in loader:
            inp = batch["image"].to(device)
            prob = torch.sigmoid(last_out(model(inp))).cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            gt = (batch["label"].numpy() > 0.5).astype(np.uint8)
            for b in range(pred.shape[0]):
                p, g = pred[b, 0], gt[b, 0]
                inter = (p & g).sum(); union = (p | g).sum()
                ious.append(inter / union if union > 0 else 0.0)
    return float(np.mean(ious)), len(ious)


def config_tta(model):
    """Freeze all but BN affine; BN uses batch stats (train mode)."""
    model.eval()
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
            m.requires_grad_(True)
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    for p in model.parameters():
        pass
    # only BN affine trainable
    trainable = set(id(p) for p in params)
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable)
    return params


def physcalib_loss(model):
    from models.Hybrid.USEANet.moe.losses import physics_calib_loss
    total, k = 0.0, 0
    for m in model.modules():
        if m.__class__.__name__ == "PhysicsMoE" and m.estimator is not None \
                and m.estimator.last_phys is not None:
            gt = m.estimator.calib_target(m.last_feat)
            total = total + physics_calib_loss(m.estimator.last_phys, gt)
            k += 1
    return total / max(1, k)


def entropy_loss(out):
    p = torch.sigmoid(last_out(out)).clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def adapt(model, loader, mode, steps, lr):
    params = config_tta(model)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    it = 0
    for _ in range(steps):
        for batch in loader:
            inp = batch["image"].to(device)
            out = model(inp)
            loss = physcalib_loss(model) if mode == "physcalib" else entropy_loss(out)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["naive", "physcalib", "entropy"], default="physcalib")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()

    from dataloader.dataloader import getZeroShotDataloader
    args = build_args(a.source, a.target, a.seed, a.bs)

    model = load_model(args)
    loader = getZeroShotDataloader(args)
    base_iou, n = iou_on(model, loader)
    tag = f"{a.source}->{a.target} s{a.seed}"
    if a.mode == "naive":
        print(f"[tta] {tag} mode=naive IoU={base_iou:.4f} n={n}")
        return

    # reload fresh model so adaptation starts from the trained weights
    model = load_model(args)
    loader = getZeroShotDataloader(args)
    nsteps = adapt(model, loader, a.mode, a.steps, a.lr)
    adapted_iou, _ = iou_on(model, loader)
    print(f"[tta] {tag} mode={a.mode} steps={nsteps} lr={a.lr} bs={a.bs} "
          f"naive={base_iou:.4f} -> adapted={adapted_iou:.4f} "
          f"delta={adapted_iou - base_iou:+.4f} n={n}")


if __name__ == "__main__":
    main()
