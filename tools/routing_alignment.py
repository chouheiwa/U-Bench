"""Quantitative routing-interpretability metrics for PUMA-Net's PhysicsMoE.

Runs one inference pass per seed over a val split and, from each PhysicsMoE block's
stashed last_gate / last_proxy ([B,E,H,W]), aggregates over ALL images x positions:
  - per-expert Pearson r(gate_e, proxy_e)        -> does the gate track its physics cue
  - per-expert specialization ratio               -> gate weight where cue e dominates
                                                      vs elsewhere (argmax(proxy)==e)
  - argmax agreement fraction                      -> gate top-expert == proxy top-expert
  - mean effective #experts per position           -> routing sparsity (vs uniform 6)
Reported as mean+/-std over the seeds passed in --exp_names.

Honesty note: the gate is KL-supervised toward the proxy, so *some* alignment is by
construction. The point of these numbers is to show the supervision takes, routing
is sparse + differentiated (not collapsed/uniform), and yet the gate is not a copy
of the proxy (only moderate correlation) -- it uses physics as a prior then refines.

Usage:
  python tools/routing_alignment.py --dataset_name busi \
      --exp_names disc_mult02_busi,disc_mult02_s42,disc_mult02_s43 --gpu 0
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu', type=str, default='0')
_pre.add_argument('--model', type=str, default='USEANet')
_pre.add_argument('--dataset_name', type=str, required=True)
_pre.add_argument('--exp_names', type=str, required=True,
                  help='comma-separated exp dirs (one per seed)')
_pre.add_argument('--out_dir', type=str, default='./result')
_args, _ = _pre.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import random
import numpy as np
import torch

torch.set_num_threads(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from models import build_model
from dataloader.dataloader import getDataloader
from models.Hybrid.USEANet.moe.physics_moe import PhysicsMoE
from models.Hybrid.USEANet.moe import EXPERT_NAMES


class Args:
    pass


def compute_for_exp(exp_name):
    exp_dir = f'./output/{_args.model}/{_args.dataset_name}/{exp_name}'
    cfg = json.load(open(os.path.join(exp_dir, 'config.json')))
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)
    seed = int(getattr(args, 'seed', 41))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    _pk = {'pretrained_model_path': args.pretrained_model_path} \
        if getattr(args, 'pretrained_model_path', None) else {}
    model = build_model(args, input_channel=args.input_channel,
                        num_classes=args.num_classes, **_pk).to(device)
    ckpt = torch.load(os.path.join(exp_dir, 'checkpoint_best.pth'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.eval()

    moes = [m for m in model.modules() if isinstance(m, PhysicsMoE)]
    E = moes[0].num_experts

    sx = np.zeros(E); sy = np.zeros(E); sxy = np.zeros(E)
    sx2 = np.zeros(E); sy2 = np.zeros(E); cnt = np.zeros(E)
    in_sum = np.zeros(E); in_cnt = np.zeros(E)
    out_sum = np.zeros(E); out_cnt = np.zeros(E)
    agree = 0.0; total_pos = 0.0
    eff_sum = 0.0; eff_n = 0

    _, valloader = getDataloader(args)
    with torch.no_grad():
        for sampled in valloader:
            _ = model(sampled['image'].to(device))
            for moe in moes:
                g = moe.last_gate[0].float().cpu().numpy()
                p = moe.last_proxy[0].float().cpu().numpy()
                gf = g.reshape(E, -1); pf = p.reshape(E, -1)
                for e in range(E):
                    a = gf[e]; b = pf[e]
                    sx[e] += a.sum(); sy[e] += b.sum(); sxy[e] += (a * b).sum()
                    sx2[e] += (a * a).sum(); sy2[e] += (b * b).sum(); cnt[e] += a.size
                pa = pf.argmax(axis=0); ga = gf.argmax(axis=0)
                agree += (pa == ga).sum(); total_pos += pa.size
                for e in range(E):
                    mask = (pa == e)
                    in_sum[e] += gf[e][mask].sum(); in_cnt[e] += mask.sum()
                    out_sum[e] += gf[e][~mask].sum(); out_cnt[e] += (~mask).sum()
                w = gf / (gf.sum(axis=0, keepdims=True) + 1e-8)
                eff = 1.0 / (np.square(w).sum(axis=0) + 1e-8)
                eff_sum += eff.mean(); eff_n += 1

    r = np.zeros(E); inm = np.zeros(E); outm = np.zeros(E); ratio = np.zeros(E)
    for e in range(E):
        n = cnt[e]
        num = sxy[e] - sx[e] * sy[e] / n
        den = np.sqrt((sx2[e] - sx[e]**2 / n) * (sy2[e] - sy[e]**2 / n))
        r[e] = num / den if den > 0 else np.nan
        inm[e] = in_sum[e] / max(in_cnt[e], 1)
        outm[e] = out_sum[e] / max(out_cnt[e], 1)
        ratio[e] = inm[e] / outm[e] if outm[e] > 0 else np.nan
    return dict(E=E, nb=len(moes), r=r, inm=inm, outm=outm, ratio=ratio,
                agree=agree / total_pos, eff=eff_sum / eff_n)


def main():
    exps = [e.strip() for e in _args.exp_names.split(',') if e.strip()]
    res = [compute_for_exp(e) for e in exps]
    E = res[0]['E']; nb = res[0]['nb']
    names = EXPERT_NAMES[:E]

    def ms(key, e=None):
        vals = np.array([x[key][e] if e is not None else x[key] for x in res])
        return float(vals.mean()), float(vals.std())

    print(f'\n[align] {nb} blocks, {E} experts, {len(exps)} seeds: {exps}')
    print(f'{"expert":12s} {"Pearson":>14s} {"gate|cue":>12s} {"ratio":>12s}')
    rows = []
    for e in range(E):
        rm, rs = ms('r', e); im, isd = ms('inm', e); ram, rasd = ms('ratio', e)
        rows.append((names[e], rm, rs, im, isd, ram, rasd))
        print(f'{names[e]:12s} {rm:6.3f}+/-{rs:4.3f} {im:6.3f}+/-{isd:4.3f} {ram:6.2f}+/-{rasd:4.2f}')
    agm, ags = ms('agree'); efm, efs = ms('eff')
    print(f'\nargmax agreement: {agm:.3f}+/-{ags:.3f}  (chance={1.0/E:.3f})')
    print(f'effective #experts/pos: {efm:.2f}+/-{efs:.2f}  (uniform={E})')

    os.makedirs(_args.out_dir, exist_ok=True)
    out_md = os.path.join(_args.out_dir, f'routing_alignment_{_args.dataset_name}.md')
    with open(out_md, 'w') as f:
        f.write(f'# Routing interpretability metrics ({_args.dataset_name})\n\n')
        f.write(f'{nb} PhysicsMoE blocks, {E} experts, mean+/-std over '
                f'{len(exps)} seeds ({", ".join(exps)}), pooled over all val '
                f'images x positions.\n\n')
        f.write('| Expert | Pearson r(gate,proxy) | mean gate where cue dominates | '
                'specialization ratio |\n|---|--:|--:|--:|\n')
        for name, rm, rs, im, isd, ram, rasd in rows:
            f.write(f'| {name} | {rm:.3f}$\\pm${rs:.3f} | {im:.3f}$\\pm${isd:.3f} '
                    f'| {ram:.2f}$\\pm${rasd:.2f} |\n')
        f.write(f'\n- **argmax agreement** (gate top == proxy top): '
                f'**{agm:.3f}$\\pm${ags:.3f}** (chance = {1.0/E:.3f})\n')
        f.write(f'- **mean effective #experts / position**: **{efm:.2f}$\\pm${efs:.2f}** '
                f'(uniform = {E}; top-2 ideal ~2)\n')
    print(f'\nwrote {out_md}')


if __name__ == '__main__':
    main()
