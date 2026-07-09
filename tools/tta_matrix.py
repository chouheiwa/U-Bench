#!/usr/bin/env python
"""Full source-free TTA matrix: physics self-calibration vs SFDA baselines.

For every (source, seed) we build the calibrated model ONCE (amortizes the
expensive PVT load), snapshot its weights, then for each target x mode reset to
the snapshot, adapt on the unlabeled target val (BN-affine only, Tent-style),
and record naive->adapted IoU. One row per (source,target,seed,mode) to
result/tta_matrix.csv.

Modes (adaptation objective on the unlabeled target):
  naive     : no adaptation (lower bound)
  bnstats   : AdaBN -- BN in train mode (target batch stats), no param update
  entropy   : Tent -- minimize prediction entropy
  pseudo    : self-training -- BCE against own hard pseudo-labels
  shot      : information maximization -- entropy_min - entropy_of_mean (anti-collapse)
  physcalib : ours -- minimize |estimator - classical pseudo-GT|
  physent   : ours+Tent -- physcalib + entropy (does physics stabilize Tent?)

Usage: python tools/tta_matrix.py --sources busi,bus,BUSBRA --seeds 41,42,43 \
         --steps 3 --lr 1e-3 --gpu 0 [--targets busi,bus,BUSBRA,BrEaST]
"""
import argparse
import copy
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

os.environ["USEANET_CALIB_PROXY"] = "1"
os.environ.pop("USEANET_CALIB_PHYS", None)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV = os.path.join(REPO, "result", "tta_matrix.csv")
MODES = ["naive", "bnstats", "entropy", "pseudo", "shot", "physcalib", "physent"]


def build_args(src, tgt, seed):
    return SimpleNamespace(
        model="USEANet", model_id=115, img_size=256,
        base_dir=f"./data/{src}", dataset_name=src, batch_size=8, seed=seed,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path="./pretrained",
        exp_save_dir=f"./output/USEANet/{src}/calib_{src}_s{seed}",
        train_file_dir="train.txt", val_file_dir="val.txt",
        zero_shot_dataset_name=tgt, zero_shot_base_dir=f"./data/{tgt}")


def last_out(o):
    return o[-1] if isinstance(o, (list, tuple)) else o


def iou_eval(model, loader):
    model.eval()
    ious = []
    with torch.no_grad():
        for b in loader:
            prob = torch.sigmoid(last_out(model(b["image"].to(device)))).cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            gt = (b["label"].numpy() > 0.5).astype(np.uint8)
            for i in range(pred.shape[0]):
                p, g = pred[i, 0], gt[i, 0]
                u = (p | g).sum()
                ious.append((p & g).sum() / u if u > 0 else 0.0)
    return float(np.mean(ious)), len(ious)


def set_bn_affine_trainable(model):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
            for p in (m.weight, m.bias):
                if p is not None:
                    params.append(p)
    tr = set(id(p) for p in params)
    for p in model.parameters():
        p.requires_grad_(id(p) in tr)
    return params


def entropy_of(out):
    p = torch.sigmoid(last_out(out)).clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def shot_loss(out):
    p = torch.sigmoid(last_out(out)).clamp(1e-6, 1 - 1e-6)
    ent = (-(p * p.log() + (1 - p) * (1 - p).log())).mean()
    pm = p.mean().clamp(1e-6, 1 - 1e-6)                 # batch marginal
    div = -(pm * pm.log() + (1 - pm) * (1 - pm).log())  # maximize -> subtract
    return ent - div


def pseudo_loss(out):
    logit = last_out(out)
    p = torch.sigmoid(logit)
    target = (p > 0.5).float().detach()
    return nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), target)


def physcalib_loss(model):
    from models.Hybrid.USEANet.moe.losses import physics_calib_loss
    tot, k = 0.0, 0
    for m in model.modules():
        if m.__class__.__name__ == "PhysicsMoE" and m.estimator is not None \
                and m.estimator.last_phys is not None:
            tot = tot + physics_calib_loss(m.estimator.last_phys,
                                           m.estimator.calib_target(m.last_feat))
            k += 1
    return tot / max(1, k)


def adapt_and_eval(model, init_sd, loader, mode, steps, lr):
    model.load_state_dict(init_sd)          # reset to trained weights
    if mode == "naive":
        model.eval()
        return iou_eval(model, loader)
    if mode == "bnstats":                    # AdaBN: accumulate target running stats
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
        with torch.no_grad():
            for _ in range(steps):
                for b in loader:
                    model(b["image"].to(device))   # momentum-update running_mean/var
        return iou_eval(model, loader)              # eval() now uses target stats
    params = set_bn_affine_trainable(model)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for _ in range(steps):
        for b in loader:
            out = model(b["image"].to(device))
            if mode == "entropy":
                loss = entropy_of(out).mean()
            elif mode == "shot":
                loss = shot_loss(out)
            elif mode == "pseudo":
                loss = pseudo_loss(out)
            elif mode == "physcalib":
                loss = physcalib_loss(model)
            elif mode == "physent":
                loss = physcalib_loss(model) + entropy_of(out).mean()
            else:
                raise ValueError(mode)
            opt.zero_grad(); loss.backward(); opt.step()
    return iou_eval(model, loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="busi,bus,BUSBRA")
    ap.add_argument("--targets", default="busi,bus,BUSBRA,BrEaST")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=CSV)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    out_csv = a.out

    from models import build_model
    from dataloader.dataloader import getZeroShotDataloader
    sources = a.sources.split(","); targets = a.targets.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]; modes = a.modes.split(",")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    new = not os.path.exists(out_csv)
    fcsv = open(out_csv, "a", newline=""); w = csv.writer(fcsv)
    if new:
        w.writerow(["source", "target", "seed", "mode", "naive_iou", "adapted_iou", "delta", "n"])

    for src in sources:
        for seed in seeds:
            args0 = build_args(src, src, seed)
            model = build_model(args0, input_channel=3, num_classes=1,
                                pretrained_model_path=args0.pretrained_model_path).to(device)
            ck = torch.load(os.path.join(args0.exp_save_dir, "checkpoint_best.pth"),
                            map_location=device, weights_only=False)
            init_sd = copy.deepcopy(ck["state_dict"] if "state_dict" in ck else ck)
            model.load_state_dict(init_sd)
            for tgt in targets:
                args = build_args(src, tgt, seed)
                loader = getZeroShotDataloader(args)
                base, n = adapt_and_eval(model, init_sd, loader, "naive", a.steps, a.lr)
                for mode in modes:
                    if mode == "naive":
                        iou = base
                    else:
                        iou, _ = adapt_and_eval(model, init_sd, loader, mode, a.steps, a.lr)
                    w.writerow([src, tgt, seed, mode, f"{base:.4f}", f"{iou:.4f}",
                                f"{iou - base:+.4f}", n]); fcsv.flush()
                    print(f"[m] {src}->{tgt} s{seed} {mode:9s} {base:.4f}->{iou:.4f} ({iou-base:+.4f})")
            del model
            torch.cuda.empty_cache()
    fcsv.close()
    print("WROTE", out_csv)


if __name__ == "__main__":
    main()
