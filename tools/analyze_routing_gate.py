#!/usr/bin/env python
"""Path 三 extension analysis: label-free per-case ROUTING-VARIANT selection.

Reads result/routing_gate_percase.csv (per-case IoU + label-free signals for
each P1 routing variant). Cross-domain only. Asks: can a label-free signal
pick, per case, WHICH routing variant to trust, beating any fixed variant and
recovering a share of the per-case oracle?

Variant sets:
  CALIB = full/att/nak/snr        (which acoustic quantity to route on)
  ALL   = manual/full/att/nak/snr (physics off vs on + which quantity)

Selection rules (all label-free, parameter-free):
  ent   : pick variant with LOWEST mean prediction entropy (most confident)
  margin: pick variant with HIGHEST confidence margin
  band  : pick variant with SMALLEST ambiguous-band fraction
Baselines: each fixed variant's mean IoU; oracle = per-case max IoU.
"""
import csv
import collections
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "result", "routing_gate_percase.csv")
OUT = os.path.join(REPO, "result", "routing_gate_analysis.txt")
CALIB = ["full", "att", "nak", "snr"]
ALL5 = ["manual", "full", "att", "nak", "snr"]
L = []


def load():
    rows = list(csv.DictReader(open(IN)))
    cross = [r for r in rows if r["source"] != r["target"]]
    G = collections.defaultdict(dict)
    for r in cross:
        k = (r["source"], r["target"], r["seed"], r["case_idx"])
        G[k][r["variant"]] = {
            "iou": float(r["iou"]), "ent": float(r["ent"]),
            "margin": float(r["margin"]), "band": float(r["band"]),
            "physerr": float(r["physerr"]), "physmag": float(r["physmag"]),
            "src": r["source"]}
    return G


def cases_with(G, S):
    return [(k, vd) for k, vd in G.items() if all(v in vd for v in S)]


def fixed_mean(cases, v):
    return sum(vd[v]["iou"] for _, vd in cases) / len(cases)


def oracle_mean(cases, S):
    return sum(max(vd[v]["iou"] for v in S) for _, vd in cases) / len(cases)


def gate_select(cases, S, sig, mode):
    """Return (mean IoU, pick-frequency dict) selecting per case by label-free sig."""
    tot = 0.0
    freq = collections.Counter()
    for _, vd in cases:
        bv, bval = None, None
        for v in S:
            val = vd[v][sig]
            if bval is None or (mode == "min" and val < bval) or (mode == "max" and val > bval):
                bval, bv = val, v
        tot += vd[bv]["iou"]
        freq[bv] += 1
    return tot / len(cases), freq


def report_set(G, S, name):
    cases = cases_with(G, S)
    L.append("\n" + "=" * 70)
    L.append(f"### variant set {name} = {S}   (cases with all present: {len(cases)})")
    if not cases:
        L.append("  (no cases)")
        return
    fixed = {v: fixed_mean(cases, v) for v in S}
    best_v = max(fixed, key=fixed.get)
    best_fixed = fixed[best_v]
    oracle = oracle_mean(cases, S)
    L.append("  fixed-variant mean IoU:")
    for v in sorted(S, key=lambda x: -fixed[x]):
        star = "  <- best fixed" if v == best_v else ""
        L.append(f"     {v:8s} {fixed[v]:.4f}{star}")
    L.append(f"  oracle (per-case max)         {oracle:.4f}   headroom vs best-fixed +{oracle - best_fixed:.4f}")
    L.append("  label-free per-case gate:")
    L.append("     %-8s | %-4s | %7s | %+8s | %6s | pick-freq" %
             ("signal", "dir", "IoU", "vs-best", "rec%"))
    for sig, mode in [("ent", "min"), ("margin", "max"), ("band", "min")]:
        g, freq = gate_select(cases, S, sig, mode)
        vs = g - best_fixed
        rec = 100.0 * vs / (oracle - best_fixed) if oracle > best_fixed else float("nan")
        fq = " ".join(f"{v}:{100*freq[v]/len(cases):.0f}%" for v in S if freq[v])
        L.append("     %-8s | %-4s | %.4f | %+8.4f | %5.1f | %s" %
                 (sig, mode, g, vs, rec, fq))
    return cases, fixed, best_v, best_fixed, oracle


def per_source(G, S, sig, mode):
    """Robustness: gate vs best-fixed within each source (LOSO-free, parameter-free)."""
    L.append(f"\n  per-source (gate={sig}/{mode}, set={S}):")
    by_src = collections.defaultdict(list)
    for k, vd in cases_with(G, S):
        by_src[vd[S[0]]["src"]].append((k, vd))
    for src in sorted(by_src):
        cs = by_src[src]
        fixed = {v: fixed_mean(cs, v) for v in S}
        bf = max(fixed.values())
        g, _ = gate_select(cs, S, sig, mode)
        orc = oracle_mean(cs, S)
        L.append(f"     {src:8s} n={len(cs):4d}  best-fixed {bf:.4f}  gate {g:.4f} ({g-bf:+.4f})  oracle {orc:.4f}")


def main():
    G = load()
    L.append("# Label-free per-case routing-variant selection (cross-domain)")
    L.append(f"total cross-domain case-groups: {len(G)}")
    variants_seen = collections.Counter()
    for vd in G.values():
        for v in vd:
            variants_seen[v] += 1
    L.append("variant presence (case-groups): " +
             ", ".join(f"{v}:{variants_seen[v]}" for v in ALL5 if variants_seen[v]))

    report_set(G, CALIB, "CALIB")
    per_source(G, CALIB, "ent", "min")
    per_source(G, CALIB, "margin", "max")
    report_set(G, ALL5, "ALL5")
    per_source(G, ALL5, "margin", "max")

    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
