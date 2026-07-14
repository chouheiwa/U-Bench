#!/usr/bin/env python3
"""Render the system-overview schematic (Fig. 1) for the reliability-gate paper.

Redesigned per figure1_redesign.md:
  - Clean left-to-right main pipeline (unseen image -> reused engine ->
    prediction-side confidence -> reliability gate), then three EQUAL-SIZED
    decision cards (select / adapt / defer) fed by the gate, then three outputs.
  - No vertical bracket, no rotated long text.
  - Three-colour scheme: deep blue = model / confidence / gate; green = trusted
    output; orange = defer / human review / fallback.
  - All numbers are the paper's CANONICAL results (+0.0195, +0.056, AUROC 0.79)
    and adaptation is stated as a TARGET-DOMAIN-LEVEL decision.
  - Vector output (PDF + PNG); the engine is a REUSED black box (its internals
    belong to a separate, anonymized submission; double-blind safe).

Outputs: result/fig_gate_system.png and result/fig_gate_system.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "result")

# palette (three families) --------------------------------------------------
BLUE_F = "#e7eef6"; BLUE_E = "#3f6ea5"          # model / signals (light blue)
GATE_F = "#2f5c8f"; GATE_E = "#21456e"          # the gate (deep blue, focal)
CARD_F = "#eaf1f8"; CARD_E = "#3f6ea5"          # decision cards
BADGE = "#2f5c8f"
GREEN_F = "#dcece0"; GREEN_E = "#3f8a5a"        # trusted output
ORANGE_F = "#f6e7d3"; ORANGE_E = "#c8802a"      # defer / human review
GREY = "#8a94a0"; TXT = "#1d232b"; MUTED = "#5a6673"


def box(ax, x, y, w, h, fc, ec, lw=1.6, rounding=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def arrow(ax, p0, p1, color=GREY, lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17,
        linewidth=lw, color=color, zorder=3,
        connectionstyle=f"arc3,rad={rad}"))


def badge(ax, xc, yc, text, fs=11.5):
    w = 0.92 * len(text) + 3.4                        # data-unit width enclosing text
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - 2.6), w, 5.2,
        boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=0, facecolor=BADGE, zorder=4))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs,
            color="white", weight="bold", zorder=5)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.8, 6.9))
    ax.set_xlim(0, 118); ax.set_ylim(0, 69); ax.axis("off")

    # ============ main pipeline (top, left -> right) ============
    py, ph = 50, 13

    box(ax, 2, py, 21, ph, BLUE_F, BLUE_E)
    ax.text(12.5, py + ph - 3.8, "Unseen target\nimage", ha="center", va="center",
            fontsize=14.5, color=TXT, weight="bold")
    ax.text(12.5, py + 2.8, "from an unseen\ntarget site", ha="center", va="center",
            fontsize=11.5, color=MUTED, style="italic")
    arrow(ax, (23.3, py + ph / 2), (27.7, py + ph / 2))

    box(ax, 28, py - 1, 23, ph + 2, "#d9d4cc", "#7d766a", lw=1.9)
    ax.text(39.5, py + ph - 2.8, "Reused MoE\nengine", ha="center", va="center",
            fontsize=14.5, color=TXT, weight="bold")
    ax.text(39.5, py + 3.0, "black box\n3.66 M params", ha="center",
            va="center", fontsize=11, color=MUTED, style="italic")
    ax.text(55, py + ph + 2.4, "K routing variants", ha="center",
            va="center", fontsize=11, color=MUTED, style="italic")
    arrow(ax, (51.3, py + ph / 2), (58.7, py + ph / 2))

    box(ax, 59, py, 23, ph, BLUE_F, BLUE_E)
    ax.text(70.5, py + ph - 2.8, "Prediction-side\nconfidence", ha="center",
            va="center", fontsize=14.5, color=TXT, weight="bold")
    ax.text(70.5, py + 3.6, "entropy $\\cdot$ margin\n$\\cdot$ ambiguous-band",
            ha="center", va="center", fontsize=11, color=MUTED)
    arrow(ax, (82.3, py + ph / 2), (87.7, py + ph / 2))

    # 4. reliability gate (focal, deep blue)
    box(ax, 88, py - 2, 28, ph + 4, GATE_F, GATE_E, lw=2.2)
    ax.text(102, py + ph - 1.6, "Reliability gate", ha="center", va="center",
            fontsize=17, color="white", weight="bold")
    ax.text(102, py + 3.2, "reads confidence only\nno target labels",
            ha="center", va="center", fontsize=11.5, color="#dce7f2", style="italic")

    # ============ distributor: gate feeds a bus that fans to three cards ============
    gx = 102; bus_y = 44; card_top = 40
    cxs = [19, 59, 99]
    # plain elbow from the gate down into the horizontal distribution bus
    # (NO arrowhead here: this is the wire, not a decision); only the three
    # branches to the cards carry arrowheads.
    ax.plot([gx, gx], [py - 2, bus_y], color=GATE_E, lw=2.0, zorder=1)
    ax.plot([cxs[0], gx], [bus_y, bus_y], color=GATE_E, lw=2.0, zorder=1)
    for cx in cxs:
        arrow(ax, (cx, bus_y), (cx, card_top + 0.4), color=GATE_E, lw=2.0)
    ax.text(59, bus_y + 1.8, "three independent, label-free decisions",
            ha="center", va="bottom", fontsize=11.5, color=MUTED, style="italic")

    # ============ three equal decision cards ============
    cw, cch, cy = 34, 20, 19
    cards = [
        ("Select the variant", "keep the most confident\nprediction",
         "+0.0195 IoU", "per-case $\\cdot$ label-free"),
        ("Adapt the target domain", "adapt only when domain-level\nuncertainty is high",
         "helps 5/6 SFDA", "target-domain decision"),
        ("Defer uncertain cases", "defer the least-confident 20%",
         "+0.056 retained IoU", "AUROC 0.79"),
    ]
    for cx, (title, body, bdg, note) in zip(cxs, cards):
        x0 = cx - cw / 2
        box(ax, x0, cy, cw, cch, CARD_F, CARD_E, lw=1.7)
        ax.text(cx, cy + cch - 3.5, title, ha="center", va="center",
                fontsize=14.5, color=TXT, weight="bold")
        ax.text(cx, cy + cch - 9.0, body, ha="center", va="center",
                fontsize=11.5, color=MUTED)
        badge(ax, cx, cy + 6.0, bdg, fs=11.5)
        if note:
            ax.text(cx, cy + 1.8, note, ha="center", va="center",
                    fontsize=10, color=MUTED, style="italic")

    # ============ outputs ============
    oy, oh = 4, 12
    outs = [
        (19, "Trusted mask", GREEN_F, GREEN_E),
        (59, "Target-domain\nupdate", BLUE_F, BLUE_E),
        (99, "Human review\n/ fallback", ORANGE_F, ORANGE_E),
    ]
    for cx, label, fc, ec in outs:
        arrow(ax, (cx, cy - 0.3), (cx, oy + oh + 0.4), color=ec, lw=2.0)
        box(ax, cx - 15, oy, 30, oh, fc, ec, lw=1.7)
        ax.text(cx, oy + oh / 2, label, ha="center", va="center",
                fontsize=14, color=TXT, weight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    png = os.path.join(OUT, "fig_gate_system.png")
    pdf = os.path.join(OUT, "fig_gate_system.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png); print("wrote", pdf)


if __name__ == "__main__":
    main()
