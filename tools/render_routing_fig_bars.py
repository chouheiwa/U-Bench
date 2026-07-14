#!/usr/bin/env python
"""Render Fig. (routing gate) — two-panel bar chart of per-case routing-variant
selection, from the CANONICAL high-precision routing analysis.

All numbers are read from the same canonical source as the paper tables:
result/routing_gate_percase.csv (8-decimal entropy, tie-free). Regenerating this
figure guarantees it never drifts from the tables again.

Panel (a): fixed variants vs. label-free min-entropy gate vs. per-case oracle.
Panel (b): per-source best-fixed / gate / oracle, with the gate's gain annotated.

Output: result/fig_routing_gate.png (+ .pdf)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "result", "routing_gate_percase.csv")
CALIB = ["att", "nak", "snr", "full"]        # display order (a): best-fixed first
C_FIX = "#c3d3e2"; C_ATT = "#8ba6bf"; C_GATE = "#2f74b5"; C_ORACLE = "#b8b8b8"
C_ANN = "#b02020"


def load():
    rg = pd.read_csv(CSV); rgx = rg[rg.source != rg.target]
    key = ["source", "target", "seed", "case_idx"]
    pi = rgx.pivot_table(index=key, columns="variant", values="iou")
    pe = rgx.pivot_table(index=key, columns="variant", values="ent")
    b = pi.dropna(subset=CALIB).join(pe[CALIB], rsuffix="_e").reset_index()
    return b


def gate_of(b):
    iv = b[CALIB].values
    ev = b[[v + "_e" for v in CALIB]].values
    return iv[np.arange(len(iv)), ev.argmin(1)]


def main():
    b = load()
    fixmean = {v: b[v].mean() for v in CALIB}
    gate = gate_of(b).mean()
    oracle = b[CALIB].values.max(1).mean()
    bestf = max(fixmean.values())
    gain = gate - bestf
    rec = 100 * gain / (oracle - bestf)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # ----- panel (a) -----
    labels = [f"route:{v}" for v in CALIB] + ["gate\n(min-entropy)", "oracle\n(per-case)"]
    vals = [fixmean[v] for v in CALIB] + [gate, oracle]
    colors = [C_ATT] + [C_FIX] * 3 + [C_GATE, C_ORACLE]
    bars = ax0.bar(range(len(vals)), vals, color=colors, edgecolor="#555", linewidth=0.8)
    bars[-1].set_hatch("//")
    for i, v in enumerate(vals):
        ax0.text(i, v + 0.0012, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax0.annotate("", xy=(4, gate), xytext=(4, bestf),
                 arrowprops=dict(arrowstyle="<->", color=C_ANN, lw=1.8))
    # place the label to the LEFT of the arrow, in the empty space above the
    # snr/full bars, so it never runs into the oracle bar on the right
    ax0.text(3.35, (gate + bestf) / 2, f"+{gain:.4f}\n({rec:.0f}% of gap)",
             color=C_ANN, fontsize=10.5, ha="right", va="center", fontweight="bold")
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, fontsize=9.5)
    ax0.set_ylabel("cross-domain mean IoU", fontsize=11)
    ax0.set_ylim(0.60, 0.715)
    ax0.set_title("(a) Per-case routing-variant selection\nbeats the best fixed variant",
                  fontsize=12)
    ax0.grid(axis="y", ls=":", alpha=0.5)

    # ----- panel (b): per source -----
    srcs = ["busi", "bus", "BUSBRA"]; disp = {"busi": "BUSI", "bus": "BUS", "BUSBRA": "BUS-BRA"}
    bf, gt, orc = [], [], []
    for s in srcs:
        sb = b[b.source == s]
        iv = sb[CALIB].values; ev = sb[[v + "_e" for v in CALIB]].values
        bf.append(max(sb[v].mean() for v in CALIB))
        gt.append(iv[np.arange(len(iv)), ev.argmin(1)].mean())
        orc.append(iv.max(1).mean())
    x = np.arange(len(srcs)); w = 0.26
    ax1.bar(x - w, bf, w, label="best fixed variant", color=C_FIX, edgecolor="#555", lw=0.8)
    ax1.bar(x, gt, w, label="label-free gate", color=C_GATE, edgecolor="#555", lw=0.8)
    ax1.bar(x + w, orc, w, label="oracle", color=C_ORACLE, edgecolor="#555", lw=0.8, hatch="//")
    # place each gate-gain label above the blue (gate) bar, shifted left so it
    # sits over the gate bar / best-fixed gap and never runs into the oracle bar
    for i in range(len(srcs)):
        ax1.text(x[i] - 0.11, gt[i] + 0.006, f"+{gt[i]-bf[i]:.4f}", ha="center",
                 va="bottom", color=C_ANN, fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([disp[s] for s in srcs], fontsize=11)
    ax1.set_ylabel("cross-domain mean IoU", fontsize=11)
    ax1.set_ylim(0.55, 0.77)
    ax1.set_title("(b) Net-positive from every source;\nlargest gain on BUSI (the harmful source)",
                  fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    png = os.path.join(REPO, "result", "fig_routing_gate.png")
    pdf = os.path.join(REPO, "result", "fig_routing_gate.pdf")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"gate={gate:.4f} gain=+{gain:.4f} rec={rec:.1f}%")
    for s, a, g, o in zip(srcs, bf, gt, orc):
        print(f"  {s}: best={a:.4f} gate={g:.4f} (+{g-a:.4f}) oracle={o:.4f}")
    print("wrote", png)


if __name__ == "__main__":
    main()
