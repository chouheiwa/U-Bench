#!/usr/bin/env python
"""Path 三 bridge: gated test-time adaptation.

Ungated SFDA is source-conditional -- it helps some (source,target) cells and
hurts others, netting ~0 or negative (see tta_matrix_summary). Here we ask
whether a LABEL-FREE cell-level signal (mean naive prediction entropy on the
target) predicts which cells benefit, so a gate "adapt only high-risk cells,
keep naive elsewhere" recovers a net-positive adaptation.

Merges:
  result/failure_gate_percase.csv  -> per (src,tgt,seed) mean naive entropy
  result/tta_matrix_g0/g1.csv      -> per (src,tgt,seed,mode) naive/adapted/delta

For each adaptation mode we (1) Spearman-correlate cell mean-entropy vs cell
delta, and (2) simulate a gate: pick the entropy threshold that maximizes net
mean delta on a train split, apply to a test split (leave-one-source-out), and
compare gated vs always-adapt vs oracle (adapt iff true delta>0).
"""
import csv
import collections
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FG = os.path.join(REPO, "result", "failure_gate_percase.csv")
OUT = os.path.join(REPO, "result", "gated_adapt.txt")
L = []


def rankdata(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b); n = len(a)
    ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main():
    # cell mean naive entropy (cross-domain only)
    fg = list(csv.DictReader(open(FG)))
    ent = collections.defaultdict(list)
    for r in fg:
        if r["source"] != r["target"]:
            ent[(r["source"], r["target"], r["seed"])].append(float(r["ent"]))
    cell_ent = {k: sum(v) / len(v) for k, v in ent.items()}

    tta = []
    for f in ["result/tta_matrix_g0.csv", "result/tta_matrix_g1.csv"]:
        p = os.path.join(REPO, f)
        if os.path.exists(p):
            tta += list(csv.DictReader(open(p)))
    tta = [r for r in tta if r["source"] != r["target"]]
    modes = ["bnstats", "entropy", "pseudo", "physcalib", "physent"]

    L.append("### gated adaptation: cell mean-entropy vs cell TTA delta (cross-domain)")
    L.append("cells with both signals: %d" % len(cell_ent))
    L.append("%-10s | %8s | %10s | %10s | %10s | %10s" %
             ("mode", "rho(ent,d)", "always", "gated-LOSO", "oracle", "gate>=always"))
    for mode in modes:
        pts = []
        for r in tta:
            if r["mode"] != mode:
                continue
            k = (r["source"], r["target"], r["seed"])
            if k in cell_ent:
                pts.append((cell_ent[k], float(r["delta"]), r["source"]))
        if len(pts) < 4:
            continue
        e = [p[0] for p in pts]; d = [p[1] for p in pts]
        rho = spearman(e, d)
        always = sum(d) / len(d)
        oracle = sum(max(0.0, x) for x in d) / len(d)   # adapt iff true delta>0
        # leave-one-source-out gate: threshold from train sources maximizing net delta
        srcs = sorted(set(p[2] for p in pts))
        gated_deltas = []
        for hold in srcs:
            tr = [p for p in pts if p[2] != hold]
            te = [p for p in pts if p[2] == hold]
            cands = sorted(set(p[0] for p in tr))
            best_t, best_net = None, -1e9
            for t in cands:
                net = sum((p[1] if p[0] >= t else 0.0) for p in tr) / len(tr)
                if net > best_net:
                    best_net, best_t = net, t
            for p in te:
                gated_deltas.append(p[1] if p[0] >= best_t else 0.0)
        gated = sum(gated_deltas) / len(gated_deltas)
        L.append("%-10s | %+8.3f | %+10.4f | %+10.4f | %+10.4f | %10s" %
                 (mode, rho, always, gated, oracle, "Y" if gated >= always else "n"))

    # ---- steps=20 over-adaptation: does the gate prevent catastrophic collapse? ----
    s20 = []
    for f in ["result/tta_s20_a.csv", "result/tta_s20_b.csv",
              "result/tta_s20_c.csv", "result/tta_s20_d.csv"]:
        p = os.path.join(REPO, f)
        if os.path.exists(p):
            s20 += list(csv.DictReader(open(p)))
    s20 = [r for r in s20 if r["source"] != r["target"]]
    if s20:
        L.append("\n### steps=20 collapse prevention (gate = keep naive on high-entropy cells)")
        L.append("%-10s | %10s | %10s | %10s | %10s | %10s" %
                 ("mode", "always", "worst", "gated-LOSO", "gWorst", "oracle"))
        for mode in ["entropy", "physcalib", "physent"]:
            pts = []
            for r in s20:
                if r["mode"] != mode:
                    continue
                k = (r["source"], r["target"], r["seed"])
                if k in cell_ent:
                    pts.append((cell_ent[k], float(r["delta"]), r["source"]))
            if len(pts) < 4:
                continue
            d = [p[1] for p in pts]
            always = sum(d) / len(d); worst = min(d)
            oracle = sum(max(0.0, x) for x in d) / len(d)
            # gate learned on steps=3 relationship is unavailable here; use LOSO on s20
            srcs = sorted(set(p[2] for p in pts))
            gd = []
            for hold in srcs:
                tr = [p for p in pts if p[2] != hold]
                te = [p for p in pts if p[2] == hold]
                cands = sorted(set(p[0] for p in tr))
                bt, bn = None, -1e9
                for t in cands:
                    net = sum((p[1] if p[0] >= t else 0.0) for p in tr) / len(tr)
                    if net > bn:
                        bn, bt = net, t
                for p in te:
                    gd.append(p[1] if p[0] >= bt else 0.0)
            gated = sum(gd) / len(gd); gworst = min(gd)
            L.append("%-10s | %+10.4f | %+10.4f | %+10.4f | %+10.4f | %+10.4f" %
                     (mode, always, worst, gated, gworst, oracle))

    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
