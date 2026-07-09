#!/usr/bin/env python
"""Cross-domain paired significance for the new-paper P1 deepening (pure CPU).

Consumes result/percase_cross_dataset.csv (written by tools/cross_dataset_eval.py
--dump_cases). Columns: exp_name,source,target,seed,case_idx,iou,dice,recall,hd95.

The target val split is fixed (val.txt, no shuffle) and independent of the seed,
so case_idx i is the SAME physical image across every seed and every method -> we
average per-case IoU over the 3 seeds per method, then pair by case_idx and run
Wilcoxon signed-rank (paired t-test as cross-check).

Two analyses:
  A. 12/12 calibrated-vs-manual routing: per (source,target) cell + a pooled
     cross-domain test (the headline "calib significantly beats hand-crafted
     proxy off-domain"). Bonferroni across the 12 cells.
  B. Per-quantity attribution (bus source): each single quantity (nak/att/snr)
     vs manual, and attenuation vs full-3, on the 3 cross-domain targets pooled.

Method mapping from exp_name:
  calibnak_bus* -> nak   calibatt_bus* -> att   calibsnr_bus* -> snr
  calib_*       -> calib (== 'full' on bus source; same checkpoint)
  c2a_* / disc_mult02* -> manual
"""
import csv
import os
import statistics
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "result", "percase_cross_dataset.csv")
OUT = os.path.join(REPO, "result", "wilcoxon_cross_report.txt")

SRCS = ["busi", "bus", "BUSBRA"]
TGTS = ["busi", "bus", "BUSBRA", "BrEaST"]


def method(exp):
    # Variant-only prefixes (source-independent). NOTE: must not hardcode the
    # source token -- "calibatt_bus" would match bus/busi but MISS BUSBRA (case).
    if exp.startswith("calibnak_"):
        return "nak"
    if exp.startswith("calibatt_"):
        return "att"
    if exp.startswith("calibsnr_"):
        return "snr"
    if exp.startswith("calib_"):
        return "calib"
    if exp.startswith("c2a_") or exp.startswith("disc_mult02"):
        return "manual"
    return None


def load():
    # raw[(method, src, tgt, case_idx)] = [iou per seed]
    raw = defaultdict(list)
    if not os.path.exists(CSV):
        return raw
    with open(CSV) as f:
        for r in csv.DictReader(f):
            m = method(r["exp_name"])
            if m is None:
                continue
            raw[(m, r["source"], r["target"], int(r["case_idx"]))].append(float(r["iou"]))
    return raw


def seed_avg(raw):
    # avg[(method, src, tgt)] = {case_idx: mean_iou_over_seeds}
    avg = defaultdict(dict)
    for (m, s, t, ci), xs in raw.items():
        avg[(m, s, t)][ci] = sum(xs) / len(xs)
    return avg


def paired(avg, mA, mB, src, tgt):
    a = avg.get((mA, src, tgt), {})
    b = avg.get((mB, src, tgt), {})
    common = sorted(set(a) & set(b))
    xa = np.array([a[i] for i in common])
    xb = np.array([b[i] for i in common])
    return xa, xb


def wilcox(xa, xb):
    """(n, median_diff, mean_diff, p_wilcoxon, p_ttest). p=1.0 when all diffs 0."""
    if len(xa) == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    d = xa - xb
    med = float(np.median(d))
    mean = float(np.mean(d))
    if np.allclose(d, 0):
        return len(xa), med, mean, 1.0, 1.0
    try:
        _, pw = stats.wilcoxon(xa, xb, zero_method="wilcox")
    except ValueError:
        pw = 1.0
    try:
        _, pt = stats.ttest_rel(xa, xb)
    except ValueError:
        pt = float("nan")
    return len(xa), med, mean, float(pw), float(pt)


def main():
    raw = load()
    avg = seed_avg(raw)
    L = []
    have = sorted({(m, s, t) for (m, s, t) in avg.keys()})
    L.append("data cells present (method,src,tgt): %d" % len(have))
    methods_present = sorted({m for (m, _, _) in have})
    L.append("methods present: %s" % ", ".join(methods_present))
    L.append("")

    # ---------- A. 12/12 calib vs manual ----------
    L.append("=" * 72)
    L.append("A. Calibrated vs hand-crafted routing  (calib - manual, per case)")
    L.append("=" * 72)
    L.append("%-8s %-8s %5s %9s %9s %11s %10s %4s" %
             ("source", "target", "n", "medΔ", "meanΔ", "p_wilcoxon", "p_ttest", "sig"))
    cell_ps = []
    cross_a, cross_b = [], []
    diag_a, diag_b = [], []
    n_win = n_cells = 0
    for s in SRCS:
        for t in TGTS:
            xa, xb = paired(avg, "calib", "manual", s, t)
            if len(xa) == 0:
                L.append("%-8s %-8s %5s  (no data)" % (s, t, "-"))
                continue
            n, med, mean, pw, pt = wilcox(xa, xb)
            cell_ps.append(pw)
            n_cells += 1
            if mean > 0:
                n_win += 1
            sig = "*" if pw < 0.05 else ""
            L.append("%-8s %-8s %5d %+9.4f %+9.4f %11.2e %10.2e %4s" %
                     (s, t, n, med, mean, pw, pt, sig))
            if s == t:
                diag_a.append(xa); diag_b.append(xb)
            else:
                cross_a.append(xa); cross_b.append(xb)
    if n_cells:
        m = min(1.0, min(cell_ps) * n_cells)
        L.append("")
        L.append("cells with mean Δ>0 (calib better): %d/%d" % (n_win, n_cells))
        L.append("Bonferroni (x%d) smallest corrected p: %.2e" % (n_cells, m))

    # pooled cross-domain & in-domain
    def pool(alist, blist, name):
        if not alist:
            return
        xa = np.concatenate(alist); xb = np.concatenate(blist)
        n, med, mean, pw, pt = wilcox(xa, xb)
        L.append("%-26s n=%-5d medΔ=%+.4f meanΔ=%+.4f  p_wilcoxon=%.2e  p_ttest=%.2e %s" %
                 (name, n, med, mean, pw, pt, "*" if pw < 0.05 else ""))
    L.append("")
    L.append("--- pooled (all cases across cells) ---")
    pool(cross_a, cross_b, "CROSS-domain (off-diag)")
    pool(diag_a, diag_b, "IN-domain (diagonal)")

    # ---------- B. per-quantity attribution, PER SOURCE (degradation scaling) ----------
    L.append("")
    L.append("=" * 72)
    L.append("B. Per-quantity attribution PER SOURCE (cross targets pooled)")
    L.append("   -> tests 'source degradation -> attenuation drives cross-domain gain'")
    L.append("=" * 72)

    def pooled_cmp(src, mA, mB, label):
        cross = [t for t in TGTS if t != src]  # off-diagonal targets for this source
        A, B = [], []
        for t in cross:
            xa, xb = paired(avg, mA, mB, src, t)
            if len(xa):
                A.append(xa); B.append(xb)
        if not A:
            L.append("  %-16s (no data)" % label); return
        xa = np.concatenate(A); xb = np.concatenate(B)
        n, med, mean, pw, pt = wilcox(xa, xb)
        L.append("  %-16s n=%-5d medΔ=%+.4f meanΔ=%+.4f  p=%.2e %s" %
                 (label, n, med, mean, pw, "*" if pw < 0.05 else ""))

    L.append("(Δ = first - second; * = p<0.05; each block = one training source)")
    for src in SRCS:
        L.append("")
        L.append("--- source = %s (cross targets: %s) ---" %
                 (src, ", ".join(t for t in TGTS if t != src)))
        pooled_cmp(src, "calib", "manual", "full - manual")
        pooled_cmp(src, "att", "manual", "att  - manual")
        pooled_cmp(src, "nak", "manual", "nak  - manual")
        pooled_cmp(src, "snr", "manual", "snr  - manual")
        pooled_cmp(src, "att", "nak", "att  - nak")
        pooled_cmp(src, "att", "snr", "att  - snr")

    txt = "\n".join(L) + "\n"
    with open(OUT, "w") as f:
        f.write(txt)
    print(txt)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
