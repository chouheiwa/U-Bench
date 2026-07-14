#!/usr/bin/env python
"""Single canonical statistics generator for the reliability-gate paper.

Every number the reviewers asked to harden is produced HERE, from the already
dumped per-case CSVs (no new inference, no fabricated numbers), with a single
fixed RNG so the manuscript never disagrees with itself again. Answers fix.md
items #2, #4, #8, #9, #10, #11, #12.

Inputs
  result/failure_gate_percase.csv   selective gate (USEANet full calib model)
  result/routing_gate_percase.csv   K routing variants, per case
  result/percase_cross_dataset.csv  main matrix -> Dice / HD95 for calib_ model

Output
  result/paper_stats.txt            canonical numbers, section-tagged to fix.md
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 0
RNG = np.random.default_rng(SEED)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = ["full", "att", "nak", "snr"]          # 4 calibrated routing variants
FAIL = 0.5                                      # main failure threshold IoU<0.5
OUT = []


def w(s=""):
    OUT.append(s)
    print(s)


# ----------------------------------------------------------------- helpers
def cv_auroc(X, y, seed=SEED):
    """5-fold out-of-fold AUROC of a logistic combiner (fixed seed = canonical)."""
    X = np.asarray(X, float); y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    risk = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        risk[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return risk, float(roc_auc_score(y, risk))


def auroc(y_fail, score):
    y = np.asarray(y_fail, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, score))


def aurc(risk, conf):
    """Area under risk-coverage curve (order by confidence desc) + E-AURC.
    risk_i in [0,1] (=1-IoU), conf_i higher=more confident."""
    order = np.argsort(-conf)                       # most confident first
    r = np.asarray(risk)[order]
    cum = np.cumsum(r) / np.arange(1, len(r) + 1)   # selective risk at each coverage
    a = float(cum.mean())
    ro = np.sort(risk)                              # oracle: lowest risk first
    cum_o = np.cumsum(ro) / np.arange(1, len(ro) + 1)
    return a, a - float(cum_o.mean())               # AURC, E-AURC


def risk_at(risk, conf, cov):
    order = np.argsort(-conf)
    k = max(1, int(round(cov * len(risk))))
    return float(np.asarray(risk)[order][:k].mean())


def ece(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, int); e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, int)) ** 2))


def nll(p, y):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7); y = np.asarray(y, int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ================================================================= load
fg = pd.read_csv(os.path.join(REPO, "result/failure_gate_percase.csv"))
fgx = fg[fg.source != fg.target].reset_index(drop=True)          # 6702 cross cases
rg = pd.read_csv(os.path.join(REPO, "result/routing_gate_percase.csv"))
cd = pd.read_csv(os.path.join(REPO, "result/percase_cross_dataset.csv"))

w("=" * 78)
w("CANONICAL PAPER STATISTICS  (single RNG seed=%d; regenerate, never hand-copy)" % SEED)
w("=" * 78)
w(f"selective-gate cross-domain cases: {len(fgx)}  seeds {sorted(fgx.seed.unique())}")

# cell id = (source,target,seed); the resampling unit for cluster bootstrap
fgx["cell"] = list(zip(fgx.source, fgx.target, fgx.seed))
cells = sorted(fgx.cell.unique())
w(f"cross-domain cells (bootstrap resampling unit): {len(cells)}")
w("")

iou = fgx.iou.values
ent = fgx.ent.values
risk = 1.0 - iou
conf = -ent                                     # higher conf = lower entropy

# ================================================= #8 selective-prediction metrics
w("### [#8] Selective prediction — proper metrics (fail=IoU<%.1f, cross-domain)" % FAIL)
y50 = (iou < 0.5).astype(int)
A, EA = aurc(risk, conf)
Arand = float(risk.mean())                       # random-order AURC = base risk
_, Aopt_ea = aurc(risk, -risk)                   # sanity (E-AURC vs self ~0)
w(f"  base risk (1-IoU)        : {risk.mean():.4f}   [no-gate mean risk]")
w(f"  AURC (entropy order)     : {A:.4f}")
w(f"  AURC (random order)      : {Arand:.4f}")
w(f"  E-AURC (excess over oracle): {EA:.4f}")
w("  risk@coverage (risk=1-IoU, lower=better):")
for c in [1.00, 0.90, 0.80, 0.70]:
    w(f"    cov {c:.2f}: entropy-gate {risk_at(risk, conf, c):.4f}  |  random {risk.mean():.4f}"
      f"  |  oracle {risk_at(risk, -risk, c):.4f}")
w("  failure-detector AUROC by threshold & baseline signal:")
w("    thr | random |  ent  | margin |  band | fgfrac | combo4(OOF)")
for thr in [0.3, 0.5, 0.7]:
    y = (iou < thr).astype(int)
    Xc = fgx[["ent", "margin", "band", "fgfrac"]].values
    _, cmb = cv_auroc(Xc, y)
    w(f"    {thr:.1f} | 0.500  | {auroc(y, ent):.3f} | {auroc(y, -fgx.margin.values):.3f}"
      f"  | {auroc(y, fgx.band.values):.3f} | {auroc(y, fgx.fgfrac.values):.3f} | {cmb:.3f}")
# calibration of the OFFLINE combiner (clearly labelled: supervised upper bound)
Xc = fgx[["ent", "margin", "band", "fgfrac"]].values
oof, cmb50 = cv_auroc(Xc, y50)
w(f"  offline 4-signal combiner (OOF, fail<0.5): AUROC={cmb50:.3f}  "
  f"Brier={brier(oof, y50):.4f}  NLL={nll(oof, y50):.4f}  ECE={ece(oof, y50):.4f}")
w("")

# ================================================= #11 canonical combo AUROC
w("### [#11] Canonical AUROC values (seed=%d, used everywhere in the paper)" % SEED)
_, a_ent = cv_auroc(fgx[["ent"]].values, y50)
_, a4 = cv_auroc(fgx[["ent", "margin", "band", "fgfrac"]].values, y50)
d6 = fgx.dropna(subset=["physerr", "physmag"])
y6 = (d6.iou.values < 0.5).astype(int)
_, a6 = cv_auroc(d6[["ent", "margin", "band", "fgfrac", "physerr", "physmag"]].values, y6)
w(f"  entropy-only (label-free deployment)   AUROC = {a_ent:.3f}")
w(f"  4-signal confidence combo (offline UB) AUROC = {a4:.3f}")
w(f"  6-signal (+physics) combo (offline UB) AUROC = {a6:.3f}")
w(f"  -> physics adds {a6 - a4:+.3f} over confidence-only (redundant)")
w("")

# ================================================= #9 per-cell / macro / LODO AUROC
w("### [#9] Failure-detection AUROC is NOT just domain mixture (entropy vs fail<0.5)")
w(f"  pooled AUROC                : {auroc(y50, ent):.3f}")
percell = []
for (s, t), g in fgx.groupby(["source", "target"]):
    a = auroc((g.iou.values < 0.5).astype(int), g.ent.values)
    if not np.isnan(a):
        percell.append(((s, t), a, len(g)))
w("  per (source->target) cell AUROC:")
for (s, t), a, n in percell:
    w(f"    {s:6s} -> {t:7s}  n={n:4d}  AUROC={a:.3f}")
w(f"  macro-average over {len(percell)} cells : {np.mean([a for _, a, _ in percell]):.3f}")
lodo = []
for t, g in fgx.groupby("target"):
    a = auroc((g.iou.values < 0.5).astype(int), g.ent.values)
    lodo.append((t, a, len(g)))
    w(f"  within target-domain {t:7s}: AUROC={a:.3f} (n={len(g)})")
w(f"  macro over held-out target domains    : {np.nanmean([a for _, a, _ in lodo]):.3f}")
w("")

# ================================================= #4 cluster bootstrap CIs
w("### [#4] Cluster bootstrap 95%% CI (resample %d cells, B=2000, seed=%d)" % (len(cells), SEED))
cell_idx = {c: fgx.index[fgx.cell == c].values for c in cells}


def boot_ci(fn, B=2000):
    vals = []
    for _ in range(B):
        pick = RNG.choice(len(cells), len(cells), replace=True)
        rows = np.concatenate([cell_idx[cells[i]] for i in pick])
        vals.append(fn(rows))
    vals = np.array(vals)
    return float(np.nanmean(vals)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def sel_gain(rows):                              # defer worst 20% by entropy
    io = iou[rows]; en = ent[rows]
    k = max(1, int(round(0.8 * len(io))))
    keep = np.argsort(en)[:k]
    return io[keep].mean() - io.mean()


def ent_auroc(rows):
    return auroc((iou[rows] < 0.5).astype(int), ent[rows])


m, lo, hi = boot_ci(sel_gain)
w(f"  selective defer-20%% IoU gain : {m:+.4f}  95%% CI [{lo:+.4f}, {hi:+.4f}]")
m, lo, hi = boot_ci(ent_auroc)
w(f"  entropy failure AUROC        : {m:.4f}  95%% CI [{lo:.4f}, {hi:.4f}]")
w("")

# ================================================= routing gate (pivot once)
rgx = rg[rg.source != rg.target].copy()
key = ["source", "target", "seed", "case_idx"]
piv_iou = rgx.pivot_table(index=key, columns="variant", values="iou")
piv_ent = rgx.pivot_table(index=key, columns="variant", values="ent")
piv_mar = rgx.pivot_table(index=key, columns="variant", values="margin")
both = (piv_iou.dropna(subset=CALIB)
        .join(piv_ent[CALIB], rsuffix="_e")
        .join(piv_mar[CALIB], rsuffix="_m").reset_index())
iouV = both[CALIB].values
entV = both[[v + "_e" for v in CALIB]].values
marV = both[[v + "_m" for v in CALIB]].values
both["cell"] = list(zip(both.source, both.target, both.seed))
rcells = sorted(both.cell.unique())
rcidx = {c: both.index[both.cell == c].values for c in rcells}
w("### [#2] Routing-variant selection — deployable baselines (cross-domain, n=%d)" % len(both))
fixmean = {v: both[v].mean() for v in CALIB}
w("  per-case IoU by fixed variant: " + ", ".join(f"{v}={fixmean[v]:.4f}" for v in CALIB))
best_fixed_global = max(fixmean, key=fixmean.get)
w(f"  (A) best-fixed by GLOBAL cross-domain acc [uses target labels]: {best_fixed_global} "
  f"= {fixmean[best_fixed_global]:.4f}")
# source-only: rank variants by source-domain (src==tgt) IoU, apply per source
rin = rg[(rg.source == rg.target)]
src_rank = {}
for s, g in rin.groupby("source"):
    mv = {v: g[g.variant == v].iou.mean() for v in CALIB}
    src_rank[s] = max(mv, key=mv.get)
w(f"  (B) source-validation pick (label-free on target): {src_rank}")
so = both.apply(lambda r: r[src_rank[r.source]], axis=1).values
w(f"      source-only selection mean IoU: {so.mean():.4f}")
rand_v = np.mean([fixmean[v] for v in CALIB])
w(f"  (C) random-variant selection (expected)          : {rand_v:.4f}")
pick = entV.argmin(axis=1)
gate = iouV[np.arange(len(iouV)), pick]
w(f"  (D) label-free min-entropy gate (per case)       : {gate.mean():.4f}  "
  f"(+{gate.mean()-fixmean[best_fixed_global]:+.4f} vs best-fixed, +{gate.mean()-so.mean():+.4f} vs source-only)")
# within-variant standardized entropy (remove per-model scale bias) -- #10
entZ = (entV - entV.mean(0)) / (entV.std(0) + 1e-9)
gateZ = iouV[np.arange(len(iouV)), entZ.argmin(axis=1)]
w(f"  (E) min standardized-entropy gate                : {gateZ.mean():.4f}")
w(f"  (F) oracle per-case max                          : {iouV.max(axis=1).mean():.4f}")
w("")


def routing_gain(rows):                          # gate vs best-fixed(global att)
    g = iouV[rows][np.arange(len(rows)), entV[rows].argmin(axis=1)].mean()
    return g - iouV[rows][:, CALIB.index(best_fixed_global)].mean()


vals = []
for _ in range(2000):
    pk = RNG.choice(len(rcells), len(rcells), replace=True)
    rows = np.concatenate([rcidx[rcells[i]] for i in pk])
    vals.append(routing_gain(rows))
vals = np.array(vals)
w("### [#4/#10] Routing gate vs best-fixed — paired cluster bootstrap 95%% CI")
w(f"  gate - best-fixed IoU gain : {vals.mean():+.4f}  95%% CI [{np.percentile(vals,2.5):+.4f}, "
  f"{np.percentile(vals,97.5):+.4f}]  (cells={len(rcells)})")
w("")

# ================================================= #10 per-variant calibration
w("### [#10] Per-variant confidence calibration (does min-entropy favour an")
w("          over-confident variant? case-level, conf=2|margin|, success=IoU>=0.5)")
w("    variant | meanIoU | AUROC(ent->fail) | Brier | ECE")
for j, v in enumerate(CALIB):
    io_v = iouV[:, j]
    yv = (io_v < 0.5).astype(int)
    cv = np.clip(2 * marV[:, j], 0, 1)           # case-level confidence in [0,1]
    succ = (io_v >= 0.5).astype(int)
    w(f"    {v:5s}   | {io_v.mean():.4f} | {auroc(yv, entV[:, j]):.3f}            "
      f"| {brier(cv, succ):.4f} | {ece(cv, succ):.4f}")
w("")

# ================================================= #12 Dice / HD95 for the gate
w("### [#12] Segmentation metrics for the gate model (calib_, cross-domain, dedup)")
cc = cd[cd.exp_name.str.startswith("calib_") & ~cd.exp_name.str.startswith("calibatt")
        & ~cd.exp_name.str.startswith("calibnak") & ~cd.exp_name.str.startswith("calibsnr")].copy()
cc["seed"] = cc.exp_name.str.extract(r"s(\d+)$").astype(int)
cc = cc[cc.source != cc.target].drop_duplicates(subset=key)
j = fgx.merge(cc[["source", "target", "seed", "case_idx", "dice", "hd95"]], on=key, how="inner")
w(f"  merged gate cases with Dice/HD95: {len(j)}")
w(f"  no-gate (all)   : IoU {j.iou.mean():.4f}  Dice {j.dice.mean():.4f}  HD95 {j.hd95.mean():.3f}")
k = max(1, int(round(0.8 * len(j))))
keep = j.iloc[np.argsort(j.ent.values)[:k]]
w(f"  entropy defer-20%%: IoU {keep.iou.mean():.4f}  Dice {keep.dice.mean():.4f}  "
  f"HD95 {keep.hd95.mean():.3f}")
w("")

with open(os.path.join(REPO, "result/paper_stats.txt"), "w") as f:
    f.write("\n".join(OUT) + "\n")
print("\nWROTE result/paper_stats.txt")
