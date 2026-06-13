import torch
from models.Hybrid.USEANet.moe import EXPERT_NAMES
from models.Hybrid.USEANet.moe.proxy import degradation_proxies


def test_proxy_shape_matches_experts():
    torch.manual_seed(0)
    feat = torch.rand(2, 16, 16, 16)
    proxy = degradation_proxies(feat)
    assert proxy.shape == (2, len(EXPERT_NAMES), 16, 16)


def test_proxy_is_nonnegative_and_normalised_per_position():
    torch.manual_seed(0)
    feat = torch.rand(2, 16, 16, 16)
    proxy = degradation_proxies(feat)
    assert (proxy >= 0).all()
    s = proxy.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_edge_proxy_high_on_vertical_edge():
    feat = torch.zeros(1, 4, 16, 16)
    feat[..., :, 8:] = 1.0
    proxy = degradation_proxies(feat)
    edge_idx = EXPERT_NAMES.index("edge")
    at_edge = proxy[0, :, 8, 7]
    assert at_edge.argmax().item() == edge_idx


def test_proxy_is_differentiable():
    feat = torch.rand(1, 8, 16, 16, requires_grad=True)
    degradation_proxies(feat).sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
