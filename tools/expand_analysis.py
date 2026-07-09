"""Extra REAL analyses for the reliability-gate paper, all computed from the
already-dumped per-case CSVs (no new inference, no fabricated numbers):

  1. Selective prediction: fine-grained risk-coverage sweep + per-seed lift.
  2. Confidence-signal ablation: 5-fold CV out-of-fold AUROC for signal subsets
     (shows physics signals are redundant with ordinary confidence).
  3. Per-case routing gate: K-variant sweep (K=2..5) headroom / gate recovery.
  4. Full 9-cell cross-domain matrix: best-fixed vs gate vs oracle per cell.
  5. Per-seed routing-gate gain.

Inputs:  result/failure_gate_percase.csv, result/routing_gate_percase.csv
Output:  result/expand_analysis.txt
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = []


def w(s=""):
    OUT.append(s)
    print(s)


def cv_auroc(X, y, seed=0):
    risk = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        risk[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, risk)


# ============================================================ load
fg = pd.read_csv(os.path.join(REPO, "result/failure_gate_percase.csv"))
rg = pd.read_csv(os.path.join(REPO, "result/routing_gate_percase.csv"))
fg_x = fg[fg.source != fg.target].copy()          # cross-domain only
w(f"failure_gate cross-domain cases: {len(fg_x)} (seeds {sorted(fg_x.seed.unique())})")
w(f"routing_gate rows: {len(rg)}  variants {sorted(rg.variant.unique())}")
w("")

# ============================================================ 1. risk-coverage
w("### 1. Selective prediction: risk-coverage sweep (cross-domain, pooled 3 seeds)")
w("coverage | entropy-gate IoU | oracle IoU | no-gate IoU")
ent = fg_x.ent.values
iou = fg_x.iou.values
order_ent = np.argsort(ent)                 # ascending entropy = most confident first
order_iou = np.argsort(-iou)                # oracle: best iou first
base = iou.mean()
n = len(iou)
for c in [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    k = max(1, int(round(c * n)))
    g = iou[order_ent[:k]].mean()
    o = iou[order_iou[:k]].mean()
    w(f"  {c:.2f}   |     {g:.4f}     |   {o:.4f} |   {base:.4f}")
w("")
w("### 1b. Per-seed defer-20% lift (coverage=0.80)")
for s in sorted(fg_x.seed.unique()):
    d = fg_x[fg_x.seed == s]
    io = d.iou.values
    oe = np.argsort(d.ent.values)
    k = max(1, int(round(0.8 * len(io))))
    w(f"  seed {s}: base {io.mean():.4f} -> kept {io[oe[:k]].mean():.4f}  (+{io[oe[:k]].mean()-io.mean():.4f})")
w("")

# ============================================================ 2. signal ablation
w("### 2. Confidence-signal ablation: 5-fold CV out-of-fold AUROC (failure=IoU<0.5)")
d = fg_x.dropna(subset=["ent", "margin", "band", "fgfrac", "physerr", "physmag"]).copy()
y = (d.iou.values < 0.5).astype(int)
w(f"n={len(d)}, failures={int(y.sum())} ({100*y.mean():.1f}%)")
subsets = {
    "entropy": ["ent"],
    "margin": ["margin"],
    "band": ["band"],
    "fg-fraction": ["fgfrac"],
    "physics-calib err": ["physerr"],
    "physics magnitude": ["physmag"],
    "entropy+margin": ["ent", "margin"],
    "entropy+margin+band": ["ent", "margin", "band"],
    "confidence-only (4)": ["ent", "margin", "band", "fgfrac"],
    "all 6 (+physics)": ["ent", "margin", "band", "fgfrac", "physerr", "physmag"],
}
for name, cols in subsets.items():
    a = cv_auroc(d[cols].values, y)
    w(f"  {name:22s} AUROC={a:.3f}")
w("")

# ============================================================ 3. K-variant sweep
w("### 3. Per-case routing gate: K-variant sweep (cross-domain, calibrated variants)")
CALIB = ["full", "att", "nak", "snr"]
rgx = rg[rg.source != rg.target].copy()
# pivot to one row per (source,target,seed,case) x variant
key = ["source", "target", "seed", "case_idx"]
piv_iou = rgx.pivot_table(index=key, columns="variant", values="iou")
piv_ent = rgx.pivot_table(index=key, columns="variant", values="ent")
both = piv_iou.dropna(subset=CALIB).join(piv_ent[CALIB], rsuffix="_ent")
w(f"cases with all 4 calibrated variants: {len(both)}")
# fixed-variant means
fixmean = {v: both[v].mean() for v in CALIB}
order = sorted(CALIB, key=lambda v: -fixmean[v])       # best fixed first
w("fixed variant means: " + ", ".join(f"{v}={fixmean[v]:.4f}" for v in order))
w("K | variants                    | best-fixed | gate(min-ent) | oracle | gate-gain | oracle-rec")
for K in range(2, 5):
    vs = order[:K]
    ent_cols = [v + "_ent" for v in vs]
    bestfix = max(both[v].mean() for v in vs)
    # gate: pick variant with min entropy per case
    pick = both[ent_cols].values.argmin(axis=1)
    gate = both[vs].values[np.arange(len(both)), pick].mean()
    orac = both[vs].values.max(axis=1).mean()
    rec = 100 * (gate - bestfix) / (orac - bestfix) if orac > bestfix else 0
    w(f"{K} | {','.join(vs):28s} | {bestfix:.4f}   |   {gate:.4f}    | {orac:.4f} | +{gate-bestfix:.4f} | {rec:.1f}%")
# K=5 with manual
ALL5 = CALIB + ["manual"]
both5 = piv_iou.dropna(subset=ALL5).join(piv_ent[ALL5], rsuffix="_ent")
if len(both5):
    ent5 = [v + "_ent" for v in ALL5]
    bestfix5 = max(both5[v].mean() for v in ALL5)
    pick5 = both5[ent5].values.argmin(axis=1)
    gate5 = both5[ALL5].values[np.arange(len(both5)), pick5].mean()
    orac5 = both5[ALL5].values.max(axis=1).mean()
    rec5 = 100 * (gate5 - bestfix5) / (orac5 - bestfix5)
    w(f"5 | {'+manual':28s} | {bestfix5:.4f}   |   {gate5:.4f}    | {orac5:.4f} | +{gate5-bestfix5:.4f} | {rec5:.1f}%  (n={len(both5)})")
w("")

# ============================================================ 4. 9-cell matrix
w("### 4. Full cross-domain matrix: best-fixed / gate / oracle IoU per (source->target)")
w("source -> target |   n  | best-fixed | gate | oracle | gate-gain")
bi = both.reset_index()
for (src, tgt), g in bi.groupby(["source", "target"]):
    ent_cols = [v + "_ent" for v in CALIB]
    bestfix = g["att"].mean()   # global best-fixed variant (att), matches main table
    pick = g[ent_cols].values.argmin(axis=1)
    gate = g[CALIB].values[np.arange(len(g)), pick].mean()
    orac = g[CALIB].values.max(axis=1).mean()
    w(f"  {src:6s} -> {tgt:7s} | {len(g):4d} |  {bestfix:.4f}   | {gate:.4f} | {orac:.4f} | +{gate-bestfix:.4f}")
w("")

# ============================================================ 5. per-seed gate
w("### 5. Per-seed routing-gate gain (min-entropy, 4 calibrated variants)")
for s in sorted(bi.seed.unique()):
    g = bi[bi.seed == s]
    ent_cols = [v + "_ent" for v in CALIB]
    bestfix = max(g[v].mean() for v in CALIB)
    pick = g[ent_cols].values.argmin(axis=1)
    gate = g[CALIB].values[np.arange(len(g)), pick].mean()
    orac = g[CALIB].values.max(axis=1).mean()
    w(f"  seed {s}: best-fixed {bestfix:.4f}  gate {gate:.4f}  (+{gate-bestfix:.4f})  oracle {orac:.4f}  n={len(g)}")

with open(os.path.join(REPO, "result/expand_analysis.txt"), "w") as f:
    f.write("\n".join(OUT) + "\n")
print("\nwrote result/expand_analysis.txt")
