import torch
from models.Hybrid.USEANet.moe import EXPERT_NAMES
from models.Hybrid.USEANet.moe.experts import build_experts


def test_build_experts_count_and_names():
    experts = build_experts(in_channel=16, out_channel=32)
    assert len(experts) == len(EXPERT_NAMES)


def test_each_expert_output_shape():
    torch.manual_seed(0)
    feat = torch.rand(2, 16, 16, 16)
    experts = build_experts(in_channel=16, out_channel=32)
    for e in experts:
        out = e(feat)
        assert out.shape == (2, 32, 16, 16)


def test_experts_are_lightweight():
    experts = build_experts(in_channel=16, out_channel=32)
    for e in experts:
        n = sum(p.numel() for p in e.parameters() if p.requires_grad)
        assert n < 30000, f"expert too heavy: {n}"


def test_experts_differentiable():
    feat = torch.rand(1, 16, 16, 16, requires_grad=True)
    out = build_experts(16, 32)[0](feat)
    out.sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
