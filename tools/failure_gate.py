#!/usr/bin/env python
"""Path 三: transfer-failure prediction / selective prediction gate.

For every (source model, target) we run per-case inference and dump, alongside
the (eval-only) ground-truth IoU, a set of LABEL-FREE signals available at test
time. The hypothesis: a cheap unsupervised signal predicts per-case failure,
enabling a risk-coverage / selective-prediction gate -- turning the physics
line's source-conditionality into a deployable decision-support feature.

Signals (all label-free, computed from the prediction / physics head only):
  ent      : mean pixelwise prediction entropy (Tent-style uncertainty)
  margin   : mean |p-0.5|  (confidence; higher = more certain)
  band     : fraction of pixels in the ambiguous band 0.3<p<0.7
  fgfrac   : predicted foreground fraction
  physerr  : mean |estimator.last_phys - classical pseudo-GT| over PhysicsMoE
  physmag  : mean estimator.last_phys magnitude

Eval-only label: iou vs GT.

Usage: python tools/failure_gate.py --sources busi,bus,BUSBRA --seeds 41 --gpu 0
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

os.environ["USEANET_CALIB_PROXY"] = "1"
os.environ.pop("USEANET_CALIB_PHYS", None)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV = os.path.join(REPO, "result", "failure_gate_percase.csv")


def build_args(src, tgt, seed):
    return SimpleNamespace(
        model="USEANet", model_id=115, img_size=256,
        base_dir=f"./data/{src}", dataset_name=src, batch_size=8, seed=seed,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path="/home/chouheiwa/experiment/pretrain_model",
        exp_save_dir=f"./output/USEANet/{src}/calib_{src}_s{seed}",
        train_file_dir="train.txt", val_file_dir="val.txt",
        zero_shot_dataset_name=tgt, zero_shot_base_dir=f"./data/{tgt}")


def last_out(o):
    return o[-1] if isinstance(o, (list, tuple)) else o


def phys_err(model):
    """Mean |estimator - classical pseudo-GT| and mean physics magnitude."""
    errs, mags = [], []
    for m in model.modules():
        if m.__class__.__name__ == "PhysicsMoE" and m.estimator is not None \
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
    ap.add_argument("--seeds", default="41")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=CSV)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    from models import build_model
    from dataloader.dataloader import getZeroShotDataloader
    sources = a.sources.split(","); targets = a.targets.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    fcsv = open(a.out, "a", newline=""); w = csv.writer(fcsv)
    if new:
        w.writerow(["source", "target", "seed", "case_idx", "iou",
                    "ent", "margin", "band", "fgfrac", "physerr", "physmag"])

    for src in sources:
        for seed in seeds:
            args0 = build_args(src, src, seed)
            model = build_model(args0, input_channel=3, num_classes=1,
                                pretrained_model_path=args0.pretrained_model_path).to(device)
            ck = torch.load(os.path.join(args0.exp_save_dir, "checkpoint_best.pth"),
                            map_location=device, weights_only=False)
            model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
            model.eval()
            for tgt in targets:
                args = build_args(src, tgt, seed)
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
                            w.writerow([src, tgt, seed, idx, f"{iou:.4f}",
                                        f"{ent:.4f}", f"{margin:.4f}", f"{band:.4f}",
                                        f"{fgfrac:.4f}", f"{pe:.4f}", f"{pm:.4f}"])
                            idx += 1
                    fcsv.flush()
                print(f"[fg] {src}->{tgt} s{seed}  {idx} cases")
            del model
            torch.cuda.empty_cache()
    fcsv.close()
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
