from tools.sim_fidelity.gate import evaluate_gate, DEFAULT_THRESHOLDS


def test_pass_when_all_within_thresholds():
    metrics = {"intensity_w1": 0.05, "speckle_snr_delta": 0.1,
               "radial_spectrum_distance": 0.1, "downstream_dice_gain": 0.6}
    report = evaluate_gate(metrics)
    assert report["fidelity_pass"] is True
    assert report["decision"] == "PROMOTE"


def test_fidelity_fail_demotes():
    metrics = {"intensity_w1": 0.5, "speckle_snr_delta": 1.2,
               "radial_spectrum_distance": 0.9, "downstream_dice_gain": 0.6}
    report = evaluate_gate(metrics)
    assert report["fidelity_pass"] is False
    assert report["decision"] == "DEMOTE_OR_DROP"


def test_fidelity_ok_but_harmful_downstream_demotes():
    metrics = {"intensity_w1": 0.05, "speckle_snr_delta": 0.1,
               "radial_spectrum_distance": 0.1, "downstream_dice_gain": -0.3}
    report = evaluate_gate(metrics)
    assert report["fidelity_pass"] is True
    assert report["decision"] == "DEMOTE_OR_DROP"


def test_defaults_present():
    for k in ("intensity_w1", "speckle_snr_delta", "radial_spectrum_distance",
              "min_downstream_dice_gain"):
        assert k in DEFAULT_THRESHOLDS
