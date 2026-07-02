"""Dump per-image validation IoU from a trained best-checkpoint (no retraining).

Reproduces exactly the metric that main.py reports:
  - val DataLoader is batch_size=1, shuffle=False  -> get_metrics() is already per-image
  - reported val_iou = simple mean over these per-image IoUs
so writing each image's IoU here is byte-consistent with the aggregate best_iou.

Usage:
  python tools/dump_percase.py --model USEANet --dataset_name bus --exp_name c2a_bus_s41 --gpu 0

Locates output/<model>/<dataset>/<exp>/config.json + checkpoint_best.pth, runs one
inference pass over the val split, and writes result/percase_<model>_<dataset>_<exp>.csv
with columns: case,iou. For USEANet `nomoe_*` runs it sets USEANET_NO_MOE=1 so the
MoE-off architecture is reconstructed to match the checkpoint.
"""
import os
import sys
import argparse
import json
import csv

# allow `import models` / `import dataloader` when run as tools/dump_percase.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- resolve GPU + env BEFORE importing torch / models ---
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu', type=str, default='0')
_pre.add_argument('--model', type=str, required=True)
_pre.add_argument('--dataset_name', type=str, required=True)
_pre.add_argument('--exp_name', type=str, required=True)
_pre.add_argument('--out_dir', type=str, default='./result')
_args, _ = _pre.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
# thread caps mirror main.py for determinism parity
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

EXP_DIR = f'./output/{_args.model}/{_args.dataset_name}/{_args.exp_name}'
CFG_PATH = os.path.join(EXP_DIR, 'config.json')
if not os.path.isfile(CFG_PATH):
    raise SystemExit(f'config.json not found: {CFG_PATH}')
cfg = json.load(open(CFG_PATH))

# MoE-off runs need the env flag set so usea_core builds MultiBranchFeatureProcessor.
if _args.model == 'USEANet' and _args.exp_name.startswith('nomoe'):
    os.environ['USEANET_NO_MOE'] = '1'
    print('[dump] USEANET_NO_MOE=1 (MoE-off architecture)')

import random
import numpy as np
import torch

torch.set_num_threads(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from models import build_model
from utils.metrics_medpy import get_metrics
from dataloader.dataloader import getDataloader


class Args:
    """Lightweight namespace reconstructed from the run's saved config.json."""
    pass


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)
    # fields the dataloader/model builder read; keep them exactly as trained
    seed_torch(int(getattr(args, 'seed', 41)))

    _pk = {'pretrained_model_path': args.pretrained_model_path} \
        if getattr(args, 'pretrained_model_path', None) else {}
    # build_model mutates args.do_deeps via load_model_id -> same value main.py used
    model = build_model(args, input_channel=args.input_channel,
                        num_classes=args.num_classes, **_pk).to(device)

    ckpt_path = os.path.join(EXP_DIR, 'checkpoint_best.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _, valloader = getDataloader(args)

    rows = []
    ious = []
    with torch.no_grad():
        for sampled in valloader:
            inp, target = sampled['image'].to(device), sampled['label'].to(device)
            case = sampled['case']
            case = case[0] if isinstance(case, (list, tuple)) else str(case)
            out = model(inp)
            out = out[-1] if args.do_deeps else out
            iou, _, _, _, _, _, _ = get_metrics(out, target)
            iou = float(iou)
            rows.append((case, iou))
            ious.append(iou)

    os.makedirs(_args.out_dir, exist_ok=True)
    out_csv = os.path.join(
        _args.out_dir,
        f'percase_{_args.model}_{_args.dataset_name}_{_args.exp_name}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'iou'])
        w.writerows(rows)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f'[dump] {out_csv}  n={len(ious)}  mean_iou={mean_iou:.4f}')


if __name__ == '__main__':
    main()
