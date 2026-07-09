#!/usr/bin/env python
"""Per-source estimator calibration quality -> does the physics head degrade on
the noisiest source (busi)? Mechanistic check for the routing-failure finding.

For each source S, build its calibrated model (USEANET_CALIB_PROXY=1), load
calib_S/checkpoint_best.pth, run S's own val split, and measure the estimator's
calibration L1 = |estimator.last_phys - acoustic_pseudo_gt(last_feat)| at the
PhysicsMoE blocks (same quantity trained by physics_calib_loss). A high value on
busi = the head cannot fit calibrated physics there (noisy input) -> unreliable
routing, explaining why physics routing hurts most from busi.

Also reports the raw magnitude of the estimated physics maps and the pseudo-GT so
we can tell whether it's the head or the (noisy) target that blows up.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

os.environ["USEANET_CALIB_PROXY"] = "1"   # must be set BEFORE build_model
os.environ.pop("USEANET_CALIB_PHYS", None)  # full 3 quantities

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from models.Hybrid.USEANet.moe.physics_estimator import acoustic_pseudo_gt  # noqa: E402

OUT = os.path.join(REPO, "result", "estimator_reliability.txt")
SRCS = ["busi", "bus", "BUSBRA"]


def build_args(src):
    return SimpleNamespace(
        model="USEANet", model_id=115, img_size=256,
        base_dir=f"./data/{src}", dataset_name=src,
        batch_size=1, seed=42,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path="/home/chouheiwa/experiment/pretrain_model",
        exp_save_dir=f"./output/USEANet/{src}/calib_{src}_s42",
        train_file_dir="train.txt", val_file_dir="val.txt",
        zero_shot_dataset_name=src, zero_shot_base_dir=f"./data/{src}",
    )


def eval_source(src):
    from models import build_model
    from dataloader.dataloader import getZeroShotDataloader
    args = build_args(src)
    model = build_model(args, input_channel=3, num_classes=1,
                        pretrained_model_path=args.pretrained_model_path).to(device)
    ckpt = torch.load(os.path.join(args.exp_save_dir, "checkpoint_best.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()

    moe = [m for m in model.modules()
           if m.__class__.__name__ == "PhysicsMoE" and m.estimator is not None]
    if not moe:
        return None

    loader = getZeroShotDataloader(args)
    calib_l1, pred_mag, gt_mag, n = [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["image"].to(device)
            _ = model(inp)
            for blk in moe:
                phys = blk.estimator.last_phys                     # [B,n,H,W]
                gt = blk.estimator.calib_target(blk.last_feat)     # [B,n,H,W] detached
                calib_l1.append(float((phys - gt).abs().mean()))
                pred_mag.append(float(phys.mean()))
                gt_mag.append(float(gt.mean()))
            n += 1
    return (float(np.mean(calib_l1)), float(np.std(calib_l1)),
            float(np.mean(pred_mag)), float(np.mean(gt_mag)), n, len(moe))


def main():
    L = ["=== per-source estimator calibration quality (in-domain val) ===",
         "calib_L1 = |estimator - classical pseudo-GT| (lower = head fits physics better)",
         "%-8s %5s %6s %11s %9s %9s %9s" %
         ("source", "n", "nMoE", "calib_L1", "L1_std", "pred_mag", "gt_mag")]
    res = {}
    for s in SRCS:
        r = eval_source(s)
        if r is None:
            L.append("%-8s  (no estimator)" % s); continue
        l1, sd, pm, gm, n, nmoe = r
        res[s] = l1
        L.append("%-8s %5d %6d %11.5f %9.5f %9.4f %9.4f" % (s, n, nmoe, l1, sd, pm, gm))
    L.append("")
    if res:
        order = sorted(res, key=res.get, reverse=True)
        L.append("calib_L1 排序 (高->低, 高=估计器越难拟合): " +
                 " > ".join("%s(%.4f)" % (s, res[s]) for s in order))
        L.append("对齐 att_gain: busi=-0.0008 bus=+0.0121 BUSBRA=+0.0069 "
                 "(SNR: busi=1.66最噪 bus=1.97 BUSBRA=2.15最净)")
        L.append("预期机制成立 <=> busi 的 calib_L1 最高 (最噪源估计器最不可靠)")
    txt = "\n".join(L) + "\n"
    with open(OUT, "w") as f:
        f.write(txt)
    print(txt)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
