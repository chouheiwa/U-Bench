import numpy as np
from tools.sim_fidelity.generate_samples import synthesize_batch


def test_synthesize_batch_shapes_and_range():
    rng = np.random.default_rng(0)
    real = rng.random((5, 1, 64, 64)).astype("float32")
    synth, maps = synthesize_batch(real, intensity=1.0, seed=0)
    assert synth.shape == (5, 64, 64)
    assert synth.min() >= 0.0 and synth.max() <= 1.0
    assert "shadow" in maps and maps["shadow"].shape == (5, 64, 64)
