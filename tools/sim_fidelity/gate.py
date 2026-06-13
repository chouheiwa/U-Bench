"""Hard-gate decision for the ultrasound degradation simulator (design §4).

Fidelity is PASS iff intensity Wasserstein, speckle-SNR delta, and radial
spectrum distance are all within thresholds. The simulator is PROMOTEd to
Pillar 2 only if fidelity passes AND it is non-harmful downstream
(downstream_dice_gain >= min). Otherwise DEMOTE_OR_DROP (use as plain aug or
drop); Pillar 1 carries the paper regardless.
"""
import argparse
import json

import numpy as np

from tools.sim_fidelity.metrics import (
    intensity_wasserstein, speckle_snr, radial_spectrum_distance,
)

# Pinned thresholds (design "仍待细化" item resolved here).
DEFAULT_THRESHOLDS = {
    "intensity_w1": 0.10,            # <= : intensity histograms close
    "speckle_snr_delta": 0.30,       # <= : |synth SNR - real SNR| small
    "radial_spectrum_distance": 0.30,  # <= : frequency content close
    "min_downstream_dice_gain": 0.0,   # >= : sim aug must not hurt (target +0.5)
}


def evaluate_gate(metrics, thresholds=None):
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    fidelity_pass = (
        metrics["intensity_w1"] <= t["intensity_w1"]
        and metrics["speckle_snr_delta"] <= t["speckle_snr_delta"]
        and metrics["radial_spectrum_distance"] <= t["radial_spectrum_distance"]
    )
    downstream_ok = metrics.get("downstream_dice_gain", 0.0) >= t["min_downstream_dice_gain"]
    decision = "PROMOTE" if (fidelity_pass and downstream_ok) else "DEMOTE_OR_DROP"
    return {
        "metrics": metrics,
        "thresholds": t,
        "fidelity_pass": bool(fidelity_pass),
        "downstream_ok": bool(downstream_ok),
        "decision": decision,
    }


def compute_fidelity_metrics(npz_path):
    data = np.load(npz_path)
    real, synth = data["real"], data["synth"]
    return {
        "intensity_w1": intensity_wasserstein(real, synth),
        "speckle_snr_delta": abs(speckle_snr(synth) - speckle_snr(real)),
        "radial_spectrum_distance": radial_spectrum_distance(real, synth),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="batches.npz from generate_samples.py")
    ap.add_argument("--downstream_dice_gain", type=float, default=0.0,
                    help="absolute Dice gain (sim-aug minus baseline) from the training experiment")
    ap.add_argument("--out", default="output/sim_fidelity/gate_report.json")
    args = ap.parse_args()

    metrics = compute_fidelity_metrics(args.npz)
    metrics["downstream_dice_gain"] = args.downstream_dice_gain
    report = evaluate_gate(metrics)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nGATE DECISION: {report['decision']}")


if __name__ == "__main__":
    main()
