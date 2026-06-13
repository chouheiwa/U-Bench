import torch
from models.Hybrid.USEANet.moe import EXPERT_NAMES
from models.Hybrid.USEANet.moe.router import DegradationAwareRouter


def _router():
    torch.manual_seed(0)
    return DegradationAwareRouter(in_channel=16, num_experts=len(EXPERT_NAMES), k=2)


def test_gate_shape():
    feat = torch.rand(2, 16, 16, 16)
    proxy = torch.rand(2, 6, 16, 16).softmax(dim=1)
    gate = _router()(feat, proxy)
    assert gate.shape == (2, 6, 16, 16)


def test_top2_exactly_two_active_per_position():
    feat = torch.rand(2, 16, 16, 16)
    proxy = torch.rand(2, 6, 16, 16).softmax(dim=1)
    gate = _router()(feat, proxy)
    active = (gate > 0).sum(dim=1)        # [B,H,W]
    assert (active == 2).all()


def test_gate_normalised_over_active_experts():
    feat = torch.rand(2, 16, 16, 16)
    proxy = torch.rand(2, 6, 16, 16).softmax(dim=1)
    gate = _router()(feat, proxy)
    s = gate.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_router_differentiable():
    feat = torch.rand(1, 16, 16, 16, requires_grad=True)
    proxy = torch.rand(1, 6, 16, 16).softmax(dim=1)
    _router()(feat, proxy).sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
