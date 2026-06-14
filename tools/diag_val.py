"""Offline diagnostic: load a USEANet+MoE checkpoint and inspect WHY val IoU is ~0.

Loads the real BUSI val set via the same dataloader main.py uses, runs the model,
and prints per-head sigmoid stats + label stats + IoU. No training.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch.nn.functional as F
from types import SimpleNamespace
from models import build_model
from dataloader.dataloader import getDataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args_run = ap.parse_args()

    cfg = SimpleNamespace(
        model="USEANet", base_dir="hf_data/data/busi", dataset_name="busi",
        train_file_dir="train.txt", val_file_dir="val.txt", batch_size=8,
        img_size=256, num_classes=1, input_channel=3,
        pretrained_model_path="/home/chouheiwa/experiment/pretrain_model",
        model_id=115, do_deeps=True, seed=41,
    )
    model = build_model(config=cfg, input_channel=3, num_classes=1,
                        pretrained_model_path=cfg.pretrained_model_path).to(device)
    ck = torch.load(args_run.ckpt, map_location=device, weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args_run.ckpt} (epoch={ck.get('epoch','?')})")

    _, valloader = getDataloader(cfg)

    head_inter = [0]*4; head_union = [0]*4
    n_imgs = 0; lbl_pos = 0; lbl_tot = 0
    head_pos_frac = [0.0]*4
    sig_min = [9.0]*4; sig_max = [-9.0]*4; sig_mean = [0.0]*4
    with torch.no_grad():
        for sb in valloader:
            x = sb["image"].to(device); y = sb["label"].to(device)
            n_imgs += x.size(0)
            lbl_pos += float((y > 0.5).sum()); lbl_tot += y.numel()
            outs = model(x)
            outs = outs if isinstance(outs, (list, tuple)) else [outs]
            for i, o in enumerate(outs):
                if o.shape[-2:] != y.shape[-2:]:
                    o = F.interpolate(o, size=y.shape[-2:], mode="bilinear", align_corners=True)
                s = torch.sigmoid(o)
                p = (s > 0.5)
                t = (y > 0.5)
                head_inter[i] += float((p & t).sum())
                head_union[i] += float((p | t).sum())
                head_pos_frac[i] += float(p.float().mean()) * x.size(0)
                sig_min[i] = min(sig_min[i], float(s.min()))
                sig_max[i] = max(sig_max[i], float(s.max()))
                sig_mean[i] += float(s.mean()) * x.size(0)

    print(f"val imgs={n_imgs}  label pos frac={lbl_pos/lbl_tot:.4f}")
    for i in range(4):
        iou = head_inter[i]/head_union[i] if head_union[i] > 0 else 0.0
        print(f"head[{i}]: IoU={iou:.4f}  pred_pos_frac={head_pos_frac[i]/n_imgs:.4f}  "
              f"sigmoid[min={sig_min[i]:.3f} mean={sig_mean[i]/n_imgs:.3f} max={sig_max[i]:.3f}]")
    i = 3
    print(f"\noutputs[-1] (head[3]) is what main.py scores -> "
          f"IoU={head_inter[i]/head_union[i] if head_union[i]>0 else 0:.4f}")


if __name__ == "__main__":
    main()
