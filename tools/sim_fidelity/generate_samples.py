"""Generate synthetic-degraded ultrasound samples for fidelity inspection.

CLI dumps side-by-side PNGs (real | synthetic) and a .npz holding the real and
synthetic batches consumed by ``gate.py``. Standalone — imports only the
simulator package.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from models.Hybrid.USEANet.simulator import UltrasoundDegradationSimulator


def synthesize_batch(real, intensity=1.0, seed=0):
    """real: [N,1,H,W] float32 in [0,1] -> (synth [N,H,W], maps {name:[N,H,W]})."""
    sim = UltrasoundDegradationSimulator(seed=seed)
    out = sim(torch.from_numpy(real).float(), intensity=intensity)
    synth = out.image.squeeze(1).detach().cpu().numpy()
    maps = {k: v.squeeze(1).detach().cpu().numpy() for k, v in out.degradation_maps.items()}
    return synth, maps


def _load_images(images_dir, limit, size):
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))[:limit]
    imgs = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (size, size)).astype("float32") / 255.0
        imgs.append(g[None])  # [1,H,W]
    return np.stack(imgs, axis=0)  # [N,1,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="dir of real US PNGs (e.g. BUSI/images)")
    ap.add_argument("--out_dir", default="output/sim_fidelity")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    real = _load_images(args.images_dir, args.limit, args.size)
    synth, _ = synthesize_batch(real, intensity=args.intensity, seed=args.seed)

    for i in range(min(16, real.shape[0])):
        pair = np.concatenate([real[i, 0], synth[i]], axis=1)
        cv2.imwrite(os.path.join(args.out_dir, f"pair_{i:03d}.png"), (pair * 255).astype("uint8"))

    np.savez(os.path.join(args.out_dir, "batches.npz"),
             real=real[:, 0], synth=synth)
    print(f"wrote {min(16, real.shape[0])} pairs + batches.npz to {args.out_dir}")


if __name__ == "__main__":
    main()
