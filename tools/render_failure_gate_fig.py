#!/usr/bin/env python
"""Render Fig. (failure gate) — selective-prediction risk-coverage curve +
label-free failure-detector AUROC bars, computed from the CANONICAL per-case CSV
(result/failure_gate_percase.csv). Regenerating guarantees it matches the tables.

Panel (a): mean IoU on the retained set vs. coverage, for the entropy gate,
           the per-case oracle, and no gate (constant). Annotates the defer-20%
           operating point.
Panel (b): failure-detection AUROC (target IoU<0.5) for each label-free signal
           and the 5-fold out-of-fold multi-confidence combo, vs. chance.

Output: result/fig_failure_gate.png (+ .pdf)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "result", "failure_gate_percase.csv")
SEED = 0


def auroc_dir(y, s):
    a = roc_auc_score(y, s)
    return a if a >= 0.5 else 1 - a


def cv_auroc(X, y):
    risk = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        risk[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, risk)


def main():
    fg = pd.read_csv(CSV)
    fg = fg[fg.source != fg.target].reset_index(drop=True)
    iou = fg.iou.values
    ent = fg.ent.values
    n = len(iou)
    base = iou.mean()

    # ---- panel (a): risk-coverage (retained mean IoU) ----
    covs = np.linspace(0.30, 1.00, 36)
    order_ent = np.argsort(ent)             # ascending entropy = most confident first
    order_or = np.argsort(-iou)             # oracle: best IoU first
    gate_c, oracle_c = [], []
    for c in covs:
        k = max(1, int(round(c * n)))
        gate_c.append(iou[order_ent[:k]].mean())
        oracle_c.append(iou[order_or[:k]].mean())

    # defer-20% operating point
    k80 = max(1, int(round(0.80 * n)))
    gate80 = iou[order_ent[:k80]].mean()

    # ---- panel (b): failure-detector AUROC ----
    y = (iou < 0.5).astype(int)
    a_ent = auroc_dir(y, ent)
    a_mar = auroc_dir(y, fg.margin.values)
    a_band = auroc_dir(y, fg.band.values)
    a_phys = auroc_dir(y, fg.physerr.values)
    X = fg[["ent", "margin", "band", "fgfrac"]].values
    a_combo = cv_auroc(X, y)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # (a)
    ax0.plot(covs, oracle_c, "--", color="#888888", lw=2.4, label="Oracle (true IoU)")
    ax0.plot(covs, gate_c, "-", color="#c0392b", lw=2.6, label="Entropy gate (label-free)")
    ax0.axhline(base, ls=":", color="#2c3e50", lw=2.0, label="No gate (random defer)")
    ax0.axvline(0.80, ls=":", color="#999999", lw=1.2)
    ax0.annotate(f"defer 20%\n+{gate80 - base:.3f} IoU",
                 xy=(0.80, gate80), xytext=(0.60, gate80 + 0.055),
                 fontsize=11, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
    ax0.set_xlabel("Coverage (fraction auto-segmented)", fontsize=12)
    ax0.set_ylabel("Mean IoU on retained cases", fontsize=12)
    ax0.set_title("(a) Selective prediction: risk--coverage", fontsize=13)
    ax0.set_xlim(0.30, 1.00)
    ax0.legend(fontsize=10.5, loc="lower left")
    ax0.grid(ls=":", alpha=0.45)

    # (b)
    names = ["entropy", "margin", "band", "phys-calib\nerr", "multi-conf\ncombo"]
    vals = [a_ent, a_mar, a_band, a_phys, a_combo]
    colors = ["#c0392b", "#e08e0b", "#e08e0b", "#7f8c8d", "#27ae60"]
    bars = ax1.bar(range(len(vals)), vals, color=colors, edgecolor="#555", lw=0.8)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax1.axhline(0.5, ls="--", color="#888", lw=1.4, label="chance")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=10.5)
    ax1.set_ylabel("Failure-detection AUROC (IoU<0.5)", fontsize=12)
    ax1.set_title("(b) Label-free failure detectors", fontsize=13)
    ax1.set_ylim(0.45, 0.92)
    ax1.legend(fontsize=10.5, loc="upper left")
    ax1.grid(axis="y", ls=":", alpha=0.45)

    fig.tight_layout()
    png = os.path.join(REPO, "result", "fig_failure_gate.png")
    pdf = os.path.join(REPO, "result", "fig_failure_gate.pdf")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"base={base:.4f} gate@80={gate80:.4f} (+{gate80-base:.4f})")
    print(f"AUROC ent={a_ent:.3f} margin={a_mar:.3f} band={a_band:.3f} "
          f"phys={a_phys:.3f} combo={a_combo:.3f}")
    print("wrote", png)


if __name__ == "__main__":
    main()
