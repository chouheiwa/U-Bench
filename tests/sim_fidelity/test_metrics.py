import numpy as np
from tools.sim_fidelity.metrics import (
    intensity_wasserstein, speckle_snr, radial_spectrum_distance,
)


def test_intensity_wasserstein_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.random((4, 64, 64)).astype("float32")
    assert intensity_wasserstein(a, a) < 1e-6


def test_intensity_wasserstein_positive_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.random((4, 64, 64)).astype("float32") * 0.3
    b = a + 0.4
    assert intensity_wasserstein(a, b) > 0.2


def test_speckle_snr_higher_for_smoother_image():
    flat = np.full((2, 64, 64), 0.5, dtype="float32")
    rng = np.random.default_rng(1)
    noisy = np.clip(flat + rng.normal(0, 0.2, flat.shape), 0, 1).astype("float32")
    assert speckle_snr(flat) > speckle_snr(noisy)


def test_radial_spectrum_distance_zero_for_identical():
    rng = np.random.default_rng(2)
    a = rng.random((3, 64, 64)).astype("float32")
    assert radial_spectrum_distance(a, a) < 1e-6
