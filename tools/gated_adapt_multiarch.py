#!/usr/bin/env python
"""Cross-architecture gated adaptation (mechanism 3 generality check).

Same leave-one-source-out entropy-gate logic as tools/gated_adapt.py, but for a
STANDARD backbone instead of the physics-anchored engine. Answers the reviewer's
"does gated adaptation also transfer off your engine?" for backbones that admit
BN-based source-free adaptation (Tent / AdaBN / pseudo-label).

Per cross-domain cell (source, target, seed) we take:
  * mean prediction entropy  -> from result/percase_gate_multiarch.csv (this model)
  * naive->adapted IoU delta -> from the TTA matrix CSV (tools/tta_matrix.py --model ...)
For each adaptation mode we compare:
  * always : mean delta if we adapt on every cell
  * gated  : leave-one-source-out entropy threshold; adapt only where mean
             entropy >= tau (learned on the other sources), else keep naive
  * oracle : adapt iff the true delta > 0 (upper bound)
gate>=always ("Y") means the label-free entropy gate is at least as good as
blindly always-adapting — i.e. it avoids adapting where it would hurt.

Honest scope: BN-based SFDA requires BatchNorm; pure-Transformer backbones
(Swin-UNet) have none, so this mechanism is only meaningful for BN-bearing
backbones (U-Net, and the CNN part of TransUNet). Mechanism 1 (selective
prediction, forward-only) is the truly architecture-agnostic result.

Usage: python tools/gated_adapt_multiarch.py --model U_Net --tta result/tta_multiarch_unet.csv
"""
import argparse
import collections
import csv
import os

from scipy.stats import rankdata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCASE = os.path.join(REPO, "result", "percase_gate_multiarch.csv")


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="U_Net")
    ap.add_argument("--tta", default=os.path.join(REPO, "result", "tta_multiarch_unet.csv"))
    ap.add_argument("--modes", default="bnstats,entropy,pseudo,shot")
    a = ap.parse_args()
    modes = a.modes.split(",")

    # cell mean entropy for this backbone, cross-domain only
    ent = collections.defaultdict(list)
    for r in csv.DictReader(open(PERCASE)):
        if r["modelname"] == a.model and r["source"] != r["target"]:
            ent[(r["source"], r["target"], r["seed"])].append(float(r["ent"]))
    cell_ent = {k: sum(v) / len(v) for k, v in ent.items()}

    tta = [r for r in csv.DictReader(open(a.tta)) if r["source"] != r["target"]]

    L = [f"### gated adaptation cross-architecture: {a.model} "
         f"(cell mean-entropy vs TTA delta, cross-domain)",
         f"cells with entropy signal: {len(cell_ent)}",
         "%-10s | %10s | %10s | %10s | %10s | %11s" %
         ("mode", "rho(ent,d)", "always", "gated-LOSO", "oracle", "gate>=always")]
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
        oracle = sum(max(0.0, x) for x in d) / len(d)
        srcs = sorted(set(p[2] for p in pts))
        gated_deltas = []
        for hold in srcs:                       # leave-one-source-out
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
        L.append("%-10s | %+10.3f | %+10.4f | %+10.4f | %+10.4f | %11s" %
                 (mode, rho, always, gated, oracle, "Y" if gated >= always else "n"))
    out = "\n".join(L) + "\n"
    print(out)
    dst = os.path.join(REPO, "result", f"gated_adapt_multiarch_{a.model}.txt")
    open(dst, "w").write(out)
    print("WROTE", dst)


if __name__ == "__main__":
    main()
