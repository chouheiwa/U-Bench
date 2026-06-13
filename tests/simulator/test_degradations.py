import torch
from models.Hybrid.USEANet.simulator.degradations import (
    log_compression, add_speckle,
)


def _img():
    torch.manual_seed(0)
    return torch.rand(2, 1, 64, 64)


def test_log_compression_keeps_shape_and_range():
    x = _img()
    out = log_compression(x, dynamic_range_db=50.0)
    assert out.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_log_compression_is_differentiable():
    x = _img().requires_grad_(True)
    log_compression(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_add_speckle_returns_image_and_map():
    x = _img()
    out, dmap = add_speckle(x, sigma=0.3, generator=torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    assert dmap.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_add_speckle_increases_variance_in_flat_region():
    flat = torch.full((1, 1, 64, 64), 0.5)
    out, _ = add_speckle(flat, sigma=0.4, generator=torch.Generator().manual_seed(2))
    assert out.var() > flat.var() + 1e-4


def test_add_speckle_is_differentiable_wrt_input():
    x = _img().requires_grad_(True)
    out, _ = add_speckle(x, sigma=0.3, generator=torch.Generator().manual_seed(3))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


from models.Hybrid.USEANet.simulator.degradations import (
    depth_attenuation, acoustic_shadow, posterior_enhancement,
)


def test_depth_attenuation_darkens_with_depth():
    x = torch.full((1, 1, 64, 64), 0.8)
    out, amap = depth_attenuation(x, coeff=1.0)
    assert out.shape == x.shape and amap.shape == x.shape
    assert out[..., -1, :].mean() < out[..., 0, :].mean()
    assert amap[..., -1, :].mean() > amap[..., 0, :].mean()


def test_acoustic_shadow_darkens_below_column_band():
    x = torch.full((1, 1, 64, 64), 0.7)
    out, smap = acoustic_shadow(x, col_start=20, col_end=30, row_start=30, strength=0.8)
    shadowed = out[..., 40:, 20:30].mean()
    clear = out[..., 40:, 40:50].mean()
    assert shadowed < clear
    assert smap[..., 40:, 20:30].mean() > smap[..., 40:, 40:50].mean()


def test_posterior_enhancement_brightens_below_band():
    x = torch.full((1, 1, 64, 64), 0.4)
    out, emap = posterior_enhancement(x, col_start=20, col_end=30, row_start=30, strength=0.6)
    enhanced = out[..., 40:, 20:30].mean()
    clear = out[..., 40:, 40:50].mean()
    assert enhanced > clear
    assert emap[..., 40:, 20:30].mean() > emap[..., 40:, 40:50].mean()


def test_new_ops_are_differentiable():
    for op in (depth_attenuation, acoustic_shadow, posterior_enhancement):
        x = torch.rand(1, 1, 32, 32, requires_grad=True)
        out, _ = op(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
