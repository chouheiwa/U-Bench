"""Render the routing-interpretability figure for the paper.

Runs the trained PUMA-Net over the BUSI val split, picks the cases with the
strongest shadow-proxy activation, and renders, for the higher-resolution MoE
block, a gate-vs-proxy heatmap grid over four specialized experts
(despeckle / shadow / posterior / contrast). Saves paper/puma-net/figs/routing_maps.png.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from models import build_model
from dataloader.dataloader import getDataloader
from models.Hybrid.USEANet.moe.physics_moe import PhysicsMoE
from models.Hybrid.USEANet.moe import EXPERT_NAMES

DS = 'busi'
EXP = 'disc_mult02_busi'
EXP_DIR = f'./output/USEANet/{DS}/{EXP}'
EXPERTS_SHOWN = ['despeckle', 'shadow', 'posterior', 'contrast']
N_CASES = 2
BLOCK = 0  # higher-resolution (x3) MoE block
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Args:
    pass


def main():
    cfg = json.load(open(os.path.join(EXP_DIR, 'config.json')))
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)
    torch.manual_seed(int(getattr(args, 'seed', 41)))

    _pk = {'pretrained_model_path': args.pretrained_model_path} \
        if getattr(args, 'pretrained_model_path', None) else {}
    model = build_model(args, input_channel=args.input_channel,
                        num_classes=args.num_classes, **_pk).to(device)
    ck = torch.load(os.path.join(EXP_DIR, 'checkpoint_best.pth'),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    model.eval()

    e_idx = {n: i for i, n in enumerate(EXPERT_NAMES[:6])}
    shadow_i = e_idx['shadow']

    _, valloader = getDataloader(args)
    # online top-N by shadow-proxy peak
    kept = []  # (peak, case, img, gate[E,h,w], proxy[E,h,w])
    with torch.no_grad():
        for sampled in valloader:
            inp = sampled['image'].to(device)
            case = sampled['case']
            case = case[0] if isinstance(case, (list, tuple)) else str(case)
            _ = model(inp)
            moes = [m for m in model.modules() if isinstance(m, PhysicsMoE)]
            g = moes[BLOCK].last_gate[0].float().cpu().numpy()
            p = moes[BLOCK].last_proxy[0].float().cpu().numpy()
            peak = float(p[shadow_i].max())
            img = inp[0].mean(0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            kept.append((peak, case, img, g, p))
            kept.sort(key=lambda t: -t[0])
            kept = kept[:N_CASES]

    H = W = args.img_size
    ncol = 1 + len(EXPERTS_SHOWN)
    nrow = 2 * N_CASES  # gate row + proxy row per case
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow))

    def up(m):
        return zoom(m, (H / m.shape[0], W / m.shape[1]), order=1)

    for ci, (peak, case, img, g, p) in enumerate(kept):
        for which, mat in ((0, g), (1, p)):  # 0=gate,1=proxy
            r = 2 * ci + which
            ax0 = axes[r][0]
            ax0.imshow(img, cmap='gray')
            ax0.set_ylabel('gate' if which == 0 else 'proxy', fontsize=11)
            if which == 0:
                ax0.set_title(f'US image\n({case})', fontsize=8)
            ax0.set_xticks([]); ax0.set_yticks([])
            for k, ename in enumerate(EXPERTS_SHOWN):
                ax = axes[r][k + 1]
                ax.imshow(img, cmap='gray')
                heat = up(mat[e_idx[ename]])
                heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                ax.imshow(heat, cmap='jet', alpha=0.5)
                if r == 0:
                    ax.set_title(ename, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    os.makedirs('paper/puma-net/figs', exist_ok=True)
    out = 'paper/puma-net/figs/routing_maps.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}  cases={[k[1] for k in kept]}')


if __name__ == '__main__':
    main()
