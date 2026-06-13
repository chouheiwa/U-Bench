import torch
from models.Hybrid.USEANet.simulator import UltrasoundDegradationSimulator, SimOutput

EXPECTED_MAPS = {"speckle", "attenuation", "shadow", "posterior"}


def _sim():
    return UltrasoundDegradationSimulator(seed=0)


def test_forward_returns_simoutput_with_maps():
    x = torch.rand(2, 1, 64, 64)
    out = _sim()(x, intensity=1.0)
    assert isinstance(out, SimOutput)
    assert out.image.shape == x.shape
    assert out.image.min() >= 0.0 and out.image.max() <= 1.0
    assert EXPECTED_MAPS.issubset(set(out.degradation_maps.keys()))
    for m in out.degradation_maps.values():
        assert m.shape == x.shape


def test_intensity_zero_is_near_identity():
    x = torch.rand(1, 1, 64, 64)
    out = _sim()(x, intensity=0.0)
    assert torch.allclose(out.degradation_maps["shadow"], torch.zeros_like(x), atol=1e-6)
    assert torch.allclose(out.degradation_maps["posterior"], torch.zeros_like(x), atol=1e-6)


def test_higher_intensity_increases_total_degradation():
    x = torch.rand(1, 1, 64, 64)
    lo = _sim()(x, intensity=0.2)
    hi = _sim()(x, intensity=1.0)
    lo_mag = sum(m.mean() for m in lo.degradation_maps.values())
    hi_mag = sum(m.mean() for m in hi.degradation_maps.values())
    assert hi_mag > lo_mag


def test_simulator_is_differentiable():
    x = torch.rand(1, 1, 32, 32, requires_grad=True)
    _sim()(x, intensity=1.0).image.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
