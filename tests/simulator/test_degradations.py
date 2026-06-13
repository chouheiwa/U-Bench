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
