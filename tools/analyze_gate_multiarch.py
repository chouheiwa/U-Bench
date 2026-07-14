#!/usr/bin/env python
"""Cross-architecture selective-prediction analysis (paper rebuttal: does the
label-free reliability gate transfer to backbones OTHER than the physics-anchored
engine?).

Reads result/percase_gate_multiarch.csv (per-case IoU + prediction-side signals
for several standard backbones, produced by tools/run_gate_multiarch.sh) and, for
each backbone, on the CROSS-DOMAIN cells only (source != target), reports:

  * IoU(all)      : mean IoU with no gate (the baseline a deployer would ship)
  * IoU(keep 80%) : mean IoU after deferring the 20% highest-entropy cases
  * dIoU          : the selective-prediction gain from that 20% deferral
  * IoU(defer 20%): mean IoU of the deferred set (should be far lower — the gate
                    is dumping the bad cases, not random ones)
  * AUROC         : entropy as a detector of failure (IoU < 0.5), pooled

An AUROC > 0.5 and dIoU > 0 on a backbone means the gate's core mechanism —
ranking cases by prediction entropy and deferring the least confident — is
model-agnostic, exactly what the reviewer question demands.

Signals are the SAME formulas as the USEANet gate (tools/failure_gate.py), so the
per-architecture numbers are directly comparable to the paper's USEANet results.
"""
import csv
import os
import sys

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def cv_auroc(X, y, seed=0):
    """5-fold out-of-fold AUROC of a logistic combiner over multiple signals.
    Identical recipe to tools/expand_analysis.py:31-37 so the multi-architecture
    combo numbers are comparable to the paper's USEANet combo AUROC."""
    if not HAVE_SK or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    risk = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        risk[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, risk))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "result", "percase_gate_multiarch.csv")
FAIL_TAU = 0.5          # a case "fails" if cross-domain IoU < 0.5
DEFER_FRAC = 0.20       # defer the least-confident 20%


def auroc(y_fail, score):
    """AUROC of `score` (higher = more likely to fail) vs binary y_fail. Falls
    back to a rank-based (Mann-Whitney) estimate if sklearn is unavailable."""
    y = np.asarray(y_fail).astype(int)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")            # undefined without both classes
    if HAVE_SK:
        return float(roc_auc_score(y, s))
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load(rows_path):
    rows = []
    with open(rows_path) as f:
        for r in csv.DictReader(f):
            if r["source"] == r["target"]:
                continue                # cross-domain only
            rows.append({
                "model": r["modelname"], "seed": r["seed"],
                "iou": float(r["iou"]), "ent": float(r["ent"]),
                "margin": float(r["margin"]), "band": float(r["band"]),
                "fgfrac": float(r["fgfrac"]),
            })
    return rows


def gate_stats(sub):
    iou = np.array([x["iou"] for x in sub])
    ent = np.array([x["ent"] for x in sub])
    n = len(iou)
    order = np.argsort(ent)             # ascending entropy = descending confidence
    k_keep = int(round(n * (1 - DEFER_FRAC)))
    keep = order[:k_keep]; defer = order[k_keep:]
    fail = (iou < FAIL_TAU).astype(int)
    # multi-signal combo: logistic combiner over the four label-free signals
    X = np.array([[x["ent"], x["margin"], x["band"], x["fgfrac"]] for x in sub])
    return {
        "n": n, "fail_rate": float(fail.mean()),
        "iou_all": float(iou.mean()),
        "iou_keep": float(iou[keep].mean()),
        "iou_defer": float(iou[defer].mean()) if len(defer) else float("nan"),
        "d_iou": float(iou[keep].mean() - iou.mean()),
        "auroc": auroc(fail, ent),
        "auroc_combo": cv_auroc(X, fail),
    }


def main():
    if not os.path.exists(CSV):
        sys.exit(f"missing {CSV} — run tools/run_gate_multiarch.sh first")
    rows = load(CSV)
    models = sorted({x["model"] for x in rows})
    lines = []
    lines.append(f"Cross-architecture selective-prediction gate (cross-domain cells, "
                 f"fail=IoU<{FAIL_TAU}, defer {int(DEFER_FRAC*100)}%)")
    lines.append("=" * 92)
    hdr = (f"{'Backbone':<14}{'n':>6}{'fail%':>8}{'IoU(all)':>10}"
           f"{'IoU(keep)':>11}{'dIoU':>9}{'IoU(defer)':>12}{'AUROC.ent':>11}{'AUROC.4sig':>12}")
    lines.append(hdr)
    lines.append("-" * 92)
    for m in models:
        s = gate_stats([x for x in rows if x["model"] == m])
        lines.append(f"{m:<14}{s['n']:>6}{100*s['fail_rate']:>8.1f}"
                     f"{s['iou_all']:>10.4f}{s['iou_keep']:>11.4f}"
                     f"{s['d_iou']:>+9.4f}{s['iou_defer']:>12.4f}"
                     f"{s['auroc']:>11.3f}{s['auroc_combo']:>12.3f}")
    lines.append("-" * 92)
    # per-seed AUROC range, to show the effect is not a single-seed fluke
    lines.append("")
    lines.append("Per-seed AUROC (entropy vs failure), cross-domain pooled:")
    for m in models:
        per = []
        for sd in sorted({x["seed"] for x in rows if x["model"] == m}):
            sub = [x for x in rows if x["model"] == m and x["seed"] == sd]
            iou = np.array([x["iou"] for x in sub]); ent = np.array([x["ent"] for x in sub])
            per.append(f"s{sd}={auroc((iou < FAIL_TAU).astype(int), ent):.3f}")
        lines.append(f"  {m:<12} " + "  ".join(per))
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(REPO, "result", "gate_multiarch_analysis.txt"), "w") as f:
        f.write(out + "\n")
    print("\nWROTE result/gate_multiarch_analysis.txt")


if __name__ == "__main__":
    main()
