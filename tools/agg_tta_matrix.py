#!/usr/bin/env python
"""Aggregate the TTA matrix CSVs into multi-baseline + robustness tables."""
import csv, collections, os, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "result", "tta_matrix_summary.txt")
L = []


def load(files):
    rows = []
    for f in files:
        p = os.path.join(REPO, f)
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    return rows


def agg(subset, label):
    a = collections.defaultdict(list)
    for r in subset:
        a[r["mode"]].append(float(r["delta"]))
    ncell = len(subset) // max(1, len({(r["source"], r["target"], r["seed"]) for r in subset})) if subset else 0
    L.append(f"\n=== {label}  (cells x seeds = {len({(r['source'],r['target'],r['seed']) for r in subset})}) ===")
    L.append("%-10s | %8s | %7s | %8s" % ("mode", "meanD", "pos/n", "worst"))
    for m in ["naive", "bnstats", "entropy", "pseudo", "shot", "physcalib", "physent"]:
        d = a.get(m, [])
        if d:
            L.append("%-10s | %+.4f | %2d/%2d  | %+.4f" %
                     (m, sum(d) / len(d), sum(1 for x in d if x > 0), len(d), min(d)))


def main():
    s3 = load(["result/tta_matrix_g0.csv", "result/tta_matrix_g1.csv"])
    cross = [r for r in s3 if r["source"] != r["target"]]
    ind = [r for r in s3 if r["source"] == r["target"]]
    L.append("############ STEPS=3 (fair few-step comparison) ############")
    agg(cross, "CROSS-domain")
    agg(ind, "IN-domain")

    # per-cell physcalib vs best baseline
    L.append("\n=== per cross-cell: physcalib vs best non-physics baseline (3-seed mean) ===")
    byc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in cross:
        byc[(r["source"], r["target"])][r["mode"]].append(float(r["delta"]))
    for (s, t), md in sorted(byc.items()):
        pc = sum(md["physcalib"]) / len(md["physcalib"])
        base = {m: sum(v) / len(v) for m, v in md.items() if m not in ("physcalib", "naive", "physent")}
        bb = max(base, key=base.get)
        L.append("%6s->%-7s physcalib=%+.4f  best_base=%s(%+.4f)  %s" %
                 (s, t, pc, bb, base[bb], "phys wins" if pc >= base[bb] else ""))

    # steps=20 robustness (does physent stabilize Tent?)
    s20 = load(["result/tta_s20_a.csv", "result/tta_s20_b.csv"])
    if s20:
        c20 = [r for r in s20 if r["source"] != r["target"]]
        L.append("\n############ STEPS=20 (over-adaptation / robustness) ############")
        agg(c20, "CROSS-domain steps=20")
        L.append("\n=== per cross-cell steps=20: entropy vs physcalib vs physent (3-seed mean) ===")
        b20 = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in c20:
            b20[(r["source"], r["target"])][r["mode"]].append(float(r["delta"]))
        L.append("%-16s %9s %9s %9s" % ("cell", "entropy", "physcalib", "physent"))
        for (s, t), md in sorted(b20.items()):
            def mv(m):
                return ("%+.4f" % (sum(md[m]) / len(md[m]))) if m in md else "  --  "
            L.append("%6s->%-9s %9s %9s %9s" % (s, t, mv("entropy"), mv("physcalib"), mv("physent")))

    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
