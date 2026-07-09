#!/usr/bin/env python
"""Path 三 extension: label-free gate over P1 ROUTING VARIANTS (per-case).

failure_gate.py dumps per-case label-free signals for a SINGLE routing variant
(calibrated-full). Here we generalize to ALL P1 routing variants so we can ask
the deployable question: given a cross-domain case, can a cheap label-free
signal pick WHICH routing variant to trust -- physics-off (manual) vs which
calibrated physics quantity (full / attenuation / nakagami_m / snr)?

For every (variant, source model, target) we run per-case inference and dump,
alongside the eval-only IoU, the same label-free signals as failure_gate. All
variants share the same target loader (identical case order per (src,tgt,seed)),
so case_idx aligns across variants -> a per-case oracle / gate is well-defined.

Variants (exp prefix, CALIB_PROXY, CALIB_PHYS):
  manual : disc_mult02_ (busi) / c2a_ (bus,BUSBRA), proxy=0, phys=-      (physics off)
  full   : calib_,     proxy=1, phys=""            (all 3 acoustic quantities)
  att    : calibatt_,  proxy=1, phys="attenuation"
  nak    : calibnak_,  proxy=1, phys="nakagami_m"
  snr    : calibsnr_,  proxy=1, phys="snr"

Env (CALIB_PROXY / CALIB_PHYS) is read inside PhysicsMoE/PhysicsEstimator
__init__, so we set it per-variant BEFORE each build_model call in one process.

Usage: python tools/routing_gate.py --sources busi,bus,BUSBRA \
         --variants manual,full,att,nak,snr --seeds 41,42,43 --gpu 0
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
CSV = os.path.join(REPO, "result", "routing_gate_percase.csv")

# variant -> (exp-prefix template, calib_proxy, calib_phys)
# manual prefix is source-dependent (busi has no c2a_busi), resolved in exp_dir().
VARIANTS = {
    "manual": (None, "0", ""),
    "full":   ("calib", "1", ""),
    "att":    ("calibatt", "1", "attenuation"),
    "nak":    ("calibnak", "1", "nakagami_m"),
    "snr":    ("calibsnr", "1", "snr"),
}


def exp_name(variant, src, seed):
    if variant == "manual":
        return f"disc_mult02_s{seed}" if src == "busi" else f"c2a_{src}_s{seed}"
    prefix = VARIANTS[variant][0]
    return f"{prefix}_{src}_s{seed}"


def build_args(src, tgt, seed, edir):
    return SimpleNamespace(
        model="USEANet", model_id=115, img_size=256,
        base_dir=f"./data/{src}", dataset_name=src, batch_size=8, seed=seed,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path="./pretrained",
        exp_save_dir=edir,
        train_file_dir="train.txt", val_file_dir="val.txt",
        zero_shot_dataset_name=tgt, zero_shot_base_dir=f"./data/{tgt}")


def last_out(o):
    return o[-1] if isinstance(o, (list, tuple)) else o


def phys_err(model):
    """Mean |estimator - classical pseudo-GT| and mean physics magnitude.
    Returns (nan, nan) for physics-off (manual) models with no estimator."""
    errs, mags = [], []
    for m in model.modules():
        if m.__class__.__name__ == "PhysicsMoE" and getattr(m, "estimator", None) is not None \
                and m.estimator.last_phys is not None and m.last_feat is not None:
            ph = m.estimator.last_phys
            gt = m.estimator.calib_target(m.last_feat)
            errs.append((ph - gt).abs().mean().item())
            mags.append(ph.mean().item())
    if not errs:
        return float("nan"), float("nan")
    return float(np.mean(errs)), float(np.mean(mags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="busi,bus,BUSBRA")
    ap.add_argument("--targets", default="busi,bus,BUSBRA,BrEaST")
    ap.add_argument("--variants", default="manual,full,att,nak,snr")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=CSV)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from models import build_model
    from dataloader.dataloader import getZeroShotDataloader
    sources = a.sources.split(","); targets = a.targets.split(",")
    variants = a.variants.split(","); seeds = [int(s) for s in a.seeds.split(",")]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    fcsv = open(a.out, "a", newline=""); w = csv.writer(fcsv)
    if new:
        w.writerow(["variant", "source", "target", "seed", "case_idx", "iou",
                    "ent", "margin", "band", "fgfrac", "physerr", "physmag"])

    for variant in variants:
        _, proxy, phys = VARIANTS[variant]
        for src in sources:
            for seed in seeds:
                edir = os.path.join(REPO, "output", "USEANet", src, exp_name(variant, src, seed))
                ckpt = os.path.join(edir, "checkpoint_best.pth")
                if not os.path.exists(ckpt):
                    print(f"!! skip {variant} {src} s{seed}: no {ckpt}")
                    continue
                # set env BEFORE build_model so estimator matches the variant
                os.environ["USEANET_CALIB_PROXY"] = proxy
                if phys:
                    os.environ["USEANET_CALIB_PHYS"] = phys
                else:
                    os.environ.pop("USEANET_CALIB_PHYS", None)

                args0 = build_args(src, src, seed, edir)
                model = build_model(args0, input_channel=3, num_classes=1,
                                    pretrained_model_path=args0.pretrained_model_path).to(device)
                ck = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
                model.eval()
                for tgt in targets:
                    args = build_args(src, tgt, seed, edir)
                    loader = getZeroShotDataloader(args)
                    idx = 0
                    with torch.no_grad():
                        for b in loader:
                            logit = last_out(model(b["image"].to(device)))
                            prob = torch.sigmoid(logit)
                            pe, pm = phys_err(model)
                            p = prob.cpu().numpy()
                            pred = (p > 0.5).astype(np.uint8)
                            gt = (b["label"].numpy() > 0.5).astype(np.uint8)
                            pcl = np.clip(p, 1e-6, 1 - 1e-6)
                            ent = float(np.mean(-(pcl * np.log(pcl) + (1 - pcl) * np.log(1 - pcl))))
                            margin = float(np.mean(np.abs(p - 0.5)))
                            band = float(np.mean((p > 0.3) & (p < 0.7)))
                            fgfrac = float(np.mean(pred))
                            for i in range(pred.shape[0]):
                                pi, gi = pred[i, 0], gt[i, 0]
                                u = (pi | gi).sum()
                                iou = (pi & gi).sum() / u if u > 0 else 1.0
                                w.writerow([variant, src, tgt, seed, idx, f"{iou:.4f}",
                                            f"{ent:.4f}", f"{margin:.4f}", f"{band:.4f}",
                                            f"{fgfrac:.4f}", f"{pe:.4f}", f"{pm:.4f}"])
                                idx += 1
                        fcsv.flush()
                    print(f"[rg] {variant} {src}->{tgt} s{seed}  {idx} cases", flush=True)
                del model
                torch.cuda.empty_cache()
    fcsv.close()
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
