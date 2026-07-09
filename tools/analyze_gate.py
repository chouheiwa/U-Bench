#!/usr/bin/env python
"""Path 三 analysis: does a label-free signal predict per-case transfer failure?

Reads result/failure_gate_percase.csv. For the cross-domain cases we test each
label-free signal as (a) a rank predictor of IoU (Spearman), (b) a failure
detector (AUROC for iou < tau), and (c) a selective-prediction gate
(risk-coverage: reject worst-scoring cases, measure mean IoU on the retained
set vs the oracle that sorts by true IoU and vs random rejection).

Positive result criterion: some signal gives AUROC clearly > 0.5 and a
risk-coverage curve close to the oracle -> a deployable failure gate.
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "result", "failure_gate_percase.csv")
OUT = os.path.join(REPO, "result", "gate_analysis.txt")

# signal -> whether HIGHER means WORSE (more risk). ent/band/physerr: high=bad.
# margin/fgfrac ambiguous; we test both directions and report the better.
SIGNALS = ["ent", "margin", "band", "fgfrac", "physerr", "physmag"]
L = []


def rankdata(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def auroc(scores, labels):
    """AUROC of `scores` predicting positive label (label=1). Higher score -> pos."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    # Mann-Whitney U via rank of pos among all
    alls = scores
    r = rankdata(alls)
    rpos = sum(r[i] for i, y in enumerate(labels) if y == 1)
    npos, nneg = len(pos), len(neg)
    u = rpos - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def risk_coverage(iou, score_risk):
    """Sort by ascending risk score, keep the LEAST risky coverage-fraction,
    report mean IoU on retained. Returns list of (coverage, mean_iou)."""
    order = sorted(range(len(iou)), key=lambda i: score_risk[i])  # low risk first
    out = []
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(round(cov * len(iou))))
        keep = order[:k]
        out.append((cov, sum(iou[i] for i in keep) / k))
    return out


def logreg_cv_auroc(X, y, folds=5, iters=400, lr=0.3, seed=0):
    """Standardize features, 5-fold CV logistic regression, return mean AUROC.
    Pure-python; deterministic fold split (index modulo). Predicts P(fail)."""
    n, d = len(X), len(X[0])
    # standardize per feature
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    sds = [(sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) ** 0.5 or 1.0 for j in range(d)]
    Z = [[(X[i][j] - means[j]) / sds[j] for j in range(d)] for i in range(n)]
    aurocs = []
    for f in range(folds):
        tr = [i for i in range(n) if i % folds != f]
        te = [i for i in range(n) if i % folds == f]
        w = [0.0] * d; b = 0.0
        for _ in range(iters):
            gw = [0.0] * d; gb = 0.0
            for i in tr:
                z = b + sum(w[j] * Z[i][j] for j in range(d))
                p = 1.0 / (1.0 + pow(2.718281828, -z))
                e = p - y[i]
                for j in range(d):
                    gw[j] += e * Z[i][j]
                gb += e
            m = len(tr)
            for j in range(d):
                w[j] -= lr * gw[j] / m
            b -= lr * gb / m
        sc = [b + sum(w[j] * Z[i][j] for j in range(d)) for i in te]
        yl = [y[i] for i in te]
        a = auroc(sc, yl)
        if a == a:
            aurocs.append(a)
    return sum(aurocs) / len(aurocs) if aurocs else float("nan")


def main():
    rows = list(csv.DictReader(open(CSV)))
    cross = [r for r in rows if r["source"] != r["target"]]
    L.append(f"failure_gate: {len(rows)} rows, {len(cross)} cross-domain cases")
    seeds = sorted(set(r["seed"] for r in cross))
    L.append(f"seeds present: {seeds}")
    iou = [float(r["iou"]) for r in cross]
    n = len(iou)
    L.append(f"cross-domain mean IoU = {sum(iou)/n:.4f}")
    for tau in (0.5, 0.3):
        fail = [1 if v < tau else 0 for v in iou]
        L.append(f"\n### failure = IoU < {tau}  ({sum(fail)}/{n} = {sum(fail)/n:.1%} fail)")
        L.append("%-8s | %7s | %7s(dir) | %7s" % ("signal", "spearman", "AUROC", "best"))
        for s in SIGNALS:
            vals = [float(r[s]) for r in cross]
            if len(set(vals)) <= 1:
                continue
            sp = spearman(vals, iou)
            a_hi = auroc(vals, fail)         # high signal -> failure
            a_lo = auroc([-v for v in vals], fail)
            best = max(a_hi, a_lo)
            d = "hi=bad" if a_hi >= a_lo else "lo=bad"
            L.append("%-8s | %+7.3f | %7.3f | %7.3f %s" % (s, sp, best, best, d))

    # risk-coverage with the single best signal by |spearman|
    best_sig, best_sp = None, 0.0
    for s in SIGNALS:
        vals = [float(r[s]) for r in cross]
        if len(set(vals)) <= 1:
            continue
        sp = spearman(vals, iou)
        if abs(sp) > abs(best_sp):
            best_sp, best_sig = sp, s
    vals = [float(r[s]) for r in cross for s in [best_sig]]
    # risk high when signal predicts low iou: if spearman<0 higher val => lower iou => higher risk
    risk = vals if best_sp < 0 else [-v for v in vals]
    L.append(f"\n### risk-coverage gate (best signal = {best_sig}, spearman={best_sp:+.3f})")
    L.append("%-8s | %8s | %8s | %8s" % ("coverage", "gate", "oracle", "random"))
    rc_gate = risk_coverage(iou, risk)
    rc_oracle = risk_coverage(iou, [-v for v in iou])   # sort by true iou desc = keep best
    for (cov, g), (_, o) in zip(rc_gate, rc_oracle):
        rnd = sum(iou) / n   # random rejection keeps mean unchanged in expectation
        L.append("%-8.2f | %8.4f | %8.4f | %8.4f" % (cov, g, o, rnd))

    # ---- multi-signal logistic regression: does physics add to entropy? ----
    fail05 = [1 if v < 0.5 else 0 for v in iou]
    L.append("\n### multi-signal failure detector (5-fold CV AUROC, fail=IoU<0.5)")
    combos = [
        ("ent only", ["ent"]),
        ("margin+band", ["margin", "band"]),
        ("ent+physerr", ["ent", "physerr"]),
        ("ent+physerr+physmag", ["ent", "physerr", "physmag"]),
        ("all-6", SIGNALS),
        ("all-conf(no phys)", ["ent", "margin", "band", "fgfrac"]),
    ]
    for name, feats in combos:
        X = [[float(r[s]) for s in feats] for r in cross]
        a = logreg_cv_auroc(X, fail05)
        L.append("%-22s | CV-AUROC = %.3f" % (name, a))

    # ---- per-seed robustness of the entropy gate ----
    L.append("\n### per-seed robustness (entropy gate, reject worst 20%)")
    L.append("%-6s | %8s | %8s | %8s" % ("seed", "full", "cov0.8", "lift"))
    for sd in seeds:
        sub = [r for r in cross if r["seed"] == sd]
        io = [float(r["iou"]) for r in sub]
        en = [float(r["ent"]) for r in sub]
        rc = risk_coverage(io, en)   # high ent = high risk
        full = rc[0][1]; c8 = [v for c, v in rc if abs(c - 0.8) < 1e-6][0]
        L.append("%-6s | %8.4f | %8.4f | %+8.4f" % (sd, full, c8, c8 - full))

    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
