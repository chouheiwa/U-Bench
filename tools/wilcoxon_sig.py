"""Paired significance test: PUMA-Net vs baselines, per dataset (pure CPU).

Reads result/percase_<model>_<ds>_<exp>.csv (written by tools/dump_percase.py),
averages per-image IoU across the 3 seeds for each method, pairs by case-id, and
runs Wilcoxon signed-rank (primary) + paired t-test (cross-check) for
  PUMA-Net vs H2Former   (4 datasets)
  PUMA-Net vs MoE-off    (3 datasets; tuscui has no nomoe run)
Bonferroni-corrects across the whole family of comparisons.

Narrative note (efficiency paper): a NON-significant PUMA-vs-H2Former gap is the
desired result — statistical parity at ~1/40 the FLOPs. A significant PUMA>MoE-off
gap on BUSI supports the physics-MoE contribution.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from scipy import stats

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')

# method -> {dataset -> [exp glob patterns for the 3 seeds]}
SPEC = {
    'PUMA-Net': {
        'busi':   ['USEANet_busi_disc_mult02_busi', 'USEANet_busi_disc_mult02_s42', 'USEANet_busi_disc_mult02_s43'],
        'bus':    ['USEANet_bus_c2a_bus_s41', 'USEANet_bus_c2a_bus_s42', 'USEANet_bus_c2a_bus_s43'],
        'BUSBRA': ['USEANet_BUSBRA_c2a_BUSBRA_s41', 'USEANet_BUSBRA_c2a_BUSBRA_s42', 'USEANet_BUSBRA_c2a_BUSBRA_s43'],
        'tuscui': ['USEANet_tuscui_c2a_tuscui_s41', 'USEANet_tuscui_c2a_tuscui_s42', 'USEANet_tuscui_c2a_tuscui_s43'],
    },
    'H2Former': {
        'busi':   ['H2Former_busi_baseline_s41', 'H2Former_busi_mtx_s42', 'H2Former_busi_mtx_s43'],
        'bus':    ['H2Former_bus_mtx_s41', 'H2Former_bus_mtx_s42', 'H2Former_bus_mtx_s43'],
        'BUSBRA': ['H2Former_BUSBRA_mtx_s41', 'H2Former_BUSBRA_mtx_s42', 'H2Former_BUSBRA_mtx_s43'],
        'tuscui': ['H2Former_tuscui_mtx_s41', 'H2Former_tuscui_mtx_s42', 'H2Former_tuscui_mtx_s43'],
    },
    'MoE-off': {
        'busi':   ['USEANet_busi_nomoe_mult02_s41', 'USEANet_busi_nomoe_mult02_s42', 'USEANet_busi_nomoe_mult02_s43'],
        'bus':    ['USEANet_bus_nomoe_bus_s41', 'USEANet_bus_nomoe_bus_s42', 'USEANet_bus_nomoe_bus_s43'],
        'BUSBRA': ['USEANet_BUSBRA_nomoe_BUSBRA_s41', 'USEANet_BUSBRA_nomoe_BUSBRA_s42', 'USEANet_BUSBRA_nomoe_BUSBRA_s43'],
    },
}

# comparisons to run: (baseline, dataset)
COMPARISONS = [
    ('H2Former', 'busi'), ('H2Former', 'bus'), ('H2Former', 'BUSBRA'), ('H2Former', 'tuscui'),
    ('MoE-off', 'busi'), ('MoE-off', 'bus'), ('MoE-off', 'BUSBRA'),
]
DS_LABEL = {'busi': 'BUSI', 'bus': 'BUS', 'BUSBRA': 'BUS-BRA', 'tuscui': 'TUSCUI'}


def load_percase(stem):
    path = os.path.join(RESULT_DIR, f'percase_{stem}.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['case']] = float(r['iou'])
    return out


def seed_mean(method, ds):
    """Per-case IoU averaged over the 3 seed dumps for (method, ds)."""
    stems = SPEC[method][ds]
    dicts = [load_percase(s) for s in stems]
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def main():
    m = len(COMPARISONS)  # Bonferroni family size
    rows = []
    print(f'Bonferroni family size m = {m}\n')
    for baseline, ds in COMPARISONS:
        puma = seed_mean('PUMA-Net', ds)
        base = seed_mean(baseline, ds)
        cases = sorted(set(puma) & set(base))
        a = np.array([puma[c] for c in cases])
        b = np.array([base[c] for c in cases])
        diff = a - b
        n = len(cases)
        mean_d = float(diff.mean())
        median_d = float(np.median(diff))
        # Wilcoxon signed-rank (two-sided). zero-diffs dropped (default 'wilcox').
        try:
            w_stat, p_w = stats.wilcoxon(a, b, alternative='two-sided')
        except ValueError:
            w_stat, p_w = float('nan'), 1.0
        t_stat, p_t = stats.ttest_rel(a, b)
        p_w_adj = min(p_w * m, 1.0)
        sig = 'significant' if p_w_adj < 0.05 else 'n.s.'
        rows.append(dict(cmp=f'PUMA vs {baseline}', ds=DS_LABEL[ds], n=n,
                         puma=float(a.mean()), base=float(b.mean()),
                         mean_d=mean_d, median_d=median_d,
                         p_w=p_w, p_w_adj=p_w_adj, p_t=float(p_t), sig=sig))
        print(f'PUMA vs {baseline:8s} | {DS_LABEL[ds]:8s} n={n:4d} '
              f'| PUMA={a.mean():.4f} base={b.mean():.4f} Δ={mean_d:+.4f} '
              f'| W p={p_w:.3g} (adj {p_w_adj:.3g}) t p={p_t:.3g} -> {sig}')

    # markdown table
    out_md = os.path.join(RESULT_DIR, 'result_significance.md')
    with open(out_md, 'w') as f:
        f.write('# Paired significance: PUMA-Net vs baselines\n\n')
        f.write('Per-image IoU averaged over 3 seeds (41/42/43), paired by case-id. '
                'Wilcoxon signed-rank two-sided (primary), paired t-test cross-check. '
                f'Bonferroni over m={m} comparisons.\n\n')
        f.write('| Comparison | Dataset | n | PUMA IoU | Base IoU | ΔIoU (mean) | Δ (median) | p (Wilcoxon) | p (Bonf.) | p (paired t) | Verdict |\n')
        f.write('|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|\n')
        for r in rows:
            f.write(f"| {r['cmp']} | {r['ds']} | {r['n']} | {r['puma']:.4f} | {r['base']:.4f} "
                    f"| {r['mean_d']:+.4f} | {r['median_d']:+.4f} | {r['p_w']:.3g} | {r['p_w_adj']:.3g} "
                    f"| {r['p_t']:.3g} | {r['sig']} |\n")
    print(f'\nwrote {out_md}')


if __name__ == '__main__':
    main()
