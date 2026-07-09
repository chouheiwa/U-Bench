"""Render a clean IoU-vs-FLOPs Pareto scatter for the paper (zero-GPU; data
hard-coded from the final efficiency table). Leader lines keep labels off the
markers; y-axis focuses on the competitive band (Swin-Unet at 0.534 is far
below and noted in the caption instead of stretching the axis)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (name, FLOPs G, BUSI IoU, label_x, label_y, halign)  -- label pos in data coords
PTS = [
    ("PUMA-Net (ours)", 1.6, 0.709, 2.7, 0.7025, "left"),
    ("MSLAU-Net", 12.5, 0.709, 6.5, 0.7135, "center"),
    ("H2Former", 64.5, 0.709, 52, 0.7150, "center"),
    ("TransUNet", 64.5, 0.708, 88, 0.7010, "left"),
    ("CMU-Net", 182.5, 0.705, 235, 0.7050, "left"),
    ("CMUNeXt", 14.8, 0.698, 8.5, 0.6905, "center"),
    ("UNet++", 75.3, 0.686, 90, 0.6825, "left"),
    ("VM-UNet", 15.1, 0.673, 9.0, 0.6690, "center"),
    ("Att U-Net", 133.3, 0.669, 116, 0.6745, "right"),
    ("U-Net", 131.0, 0.668, 92, 0.6640, "left"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.6))

for name, f, iou, lx, ly, ha in PTS:
    ours = name.startswith("PUMA")
    ax.scatter(f, iou, s=190 if ours else 42, zorder=4 if ours else 3,
               marker="*" if ours else "o",
               color="#c0392b" if ours else "#5b8fb9",
               edgecolors="black", linewidths=0.7 if ours else 0.5)
    ax.annotate(name, xy=(f, iou), xytext=(lx, ly), ha=ha, va="center",
                fontsize=7, fontweight="bold" if ours else "normal",
                color="#c0392b" if ours else "black",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55",
                                shrinkA=0, shrinkB=2))

# shade PUMA's dominance corner (lower FLOPs at the top IoU tier)
ax.axhspan(0.7075, 0.716, xmin=0, xmax=0.30, color="#c0392b", alpha=0.07, zorder=0)

ax.set_xscale("log")
ax.set_xlim(1.2, 430)
ax.set_ylim(0.660, 0.717)
ax.set_xlabel("FLOPs (G, log scale) — lower is better", fontsize=8)
ax.set_ylabel("BUSI IoU — higher is better", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, which="both", ls=":", alpha=0.35)
plt.tight_layout()
out = "paper/puma-net/figs/pareto.pdf"
plt.savefig(out, bbox_inches="tight")
print("wrote", out)
