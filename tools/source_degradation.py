#!/usr/bin/env python
"""Per-source acoustic degradation metrics -> correlate with attenuation cross-gain.

For each source's val split, load grayscale images (256px) and compute classical
acoustic-moment degradation proxies (same family as acoustic_pseudo_gt):
  - nakagami_m : m = E[X]^2/Var(X), X=intensity^2. m~1 fully-developed speckle
                 (noisy/degraded), m>1 structured/specular (cleaner boundaries).
  - snr        : global mean/std. Higher = cleaner.
  - attenuation: normalized top-vs-bottom depth brightness deficit. Higher = more
                 cumulative acoustic attenuation along depth.
  - contrast   : std/mean (speckle contrast). Higher = more speckle/noise.

Then align each source's degradation with its measured attenuation-routing cross
gain (att - manual, from wilcoxon_cross Part B) and rank-correlate. Pure CPU.
"""
import csv
import glob
import os

import numpy as np
import cv2

REPO = "/home/chouheiwa/python/U-Bench"
OUT = os.path.join(REPO, "result", "source_degradation.txt")

# path convention per source: (dir, whether val.txt entries already carry .png)
SRC_CFG = {
    "busi":   ("images", False),
    "bus":    ("images", False),
    "BUSBRA": ("Images", True),
    "BrEaST": ("images", False),
}

# measured cross-domain gains (meanΔ, per-case pooled) from wilcoxon_cross.py Part B
ATT_GAIN  = {"busi": -0.0008, "bus": +0.0121, "BUSBRA": +0.0069}   # att - manual
FULL_GAIN = {"busi": -0.0354, "bus": +0.0171, "BUSBRA": -0.0026}   # calib(full) - manual


def img_path(src, name):
    d, has_ext = SRC_CFG[src]
    fn = name if has_ext else name + ".png"
    return os.path.join(REPO, "data", src, d, fn)


def metrics_for_image(g):
    """g: float grayscale in [0,1], shape [H,W]. Returns dict of degradation metrics."""
    eps = 1e-6
    x = g * g                                  # intensity (envelope^2)
    m_nak = float(np.mean(x) ** 2 / (np.var(x) + eps))
    snr = float(np.mean(g) / (np.std(g) + eps))
    h = g.shape[0]
    band = max(1, h // 8)
    top = float(np.mean(g[:band]))
    bot = float(np.mean(g[-band:]))
    atten = float((top - bot) / (np.mean(g) + eps))
    contrast = float(np.std(g) / (np.mean(g) + eps))
    return dict(nakagami_m=m_nak, snr=snr, attenuation=atten, contrast=contrast)


def source_metrics(src):
    vals = [l.strip() for l in open(os.path.join(REPO, "data", src, "val.txt")) if l.strip()]
    acc, n, miss = {}, 0, 0
    for name in vals:
        p = img_path(src, name)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            miss += 1
            continue
        im = cv2.resize(im, (256, 256)).astype(np.float32) / 255.0
        for k, v in metrics_for_image(im).items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in acc.items()}, n, miss


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    L = []
    srcs = ["busi", "bus", "BUSBRA"]
    rows = {}
    L.append("=== per-source acoustic degradation (val split, 256px grayscale) ===")
    L.append("%-8s %5s %11s %8s %12s %9s" % ("source", "n", "nakagami_m", "snr", "attenuation", "contrast"))
    for s in srcs:
        m, n, miss = source_metrics(s)
        rows[s] = m
        L.append("%-8s %5d %11.4f %8.4f %12.4f %9.4f%s" %
                 (s, n, m["nakagami_m"], m["snr"], m["attenuation"], m["contrast"],
                  ("  (miss=%d)" % miss) if miss else ""))

    L.append("")
    L.append("=== align with cross-domain routing gains ===")
    L.append("%-8s %11s %11s | %10s %10s" %
             ("source", "att_gain", "full_gain", "1/snr", "contrast"))
    for s in srcs:
        L.append("%-8s %+11.4f %+11.4f | %10.4f %10.4f" %
                 (s, ATT_GAIN[s], FULL_GAIN[s], 1.0 / rows[s]["snr"], rows[s]["contrast"]))

    # Spearman rank corr of each degradation metric vs att_gain (3 points -> illustrative)
    L.append("")
    L.append("=== Spearman rank corr (degradation metric vs att_gain), 3 sources ===")
    att = np.array([ATT_GAIN[s] for s in srcs])
    for metric in ["nakagami_m", "snr", "attenuation", "contrast"]:
        vals = np.array([rows[s][metric] for s in srcs])
        L.append("  att_gain vs %-12s rho=%+.3f   (values: %s)" %
                 (metric, spearman(vals, att),
                  ", ".join("%s=%.3f" % (s, rows[s][metric]) for s in srcs)))
    # also vs 1/snr (noise proxy) and contrast, expected POSITIVE if degradation->gain
    for metric, name in [("snr", "1/snr"), ("contrast", "contrast")]:
        vals = np.array([(1.0 / rows[s]["snr"]) if name == "1/snr" else rows[s]["contrast"] for s in srcs])
        L.append("  att_gain vs %-12s rho=%+.3f" % (name, spearman(vals, att)))

    txt = "\n".join(L) + "\n"
    with open(OUT, "w") as f:
        f.write(txt)
    print(txt)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
