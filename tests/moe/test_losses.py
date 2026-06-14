import torch
from models.Hybrid.USEANet.moe.losses import (
    router_supervision_loss, load_balance_loss, effective_experts,
)


def test_router_supervision_zero_when_gate_matches_proxy():
    torch.manual_seed(0)
    proxy = torch.rand(2, 6, 8, 8).softmax(dim=1)
    loss = router_supervision_loss(proxy, proxy)
    assert loss.item() < 1e-4


def test_router_supervision_positive_on_mismatch():
    proxy = torch.zeros(1, 6, 4, 4); proxy[:, 0] = 1.0
    gate = torch.zeros(1, 6, 4, 4); gate[:, 1] = 1.0
    assert router_supervision_loss(gate, proxy).item() > 0.1


def test_load_balance_minimised_when_uniform():
    uniform = torch.full((2, 6, 8, 8), 1.0 / 6)
    skewed = torch.zeros(2, 6, 8, 8); skewed[:, 0] = 1.0
    assert load_balance_loss(uniform) < load_balance_loss(skewed)


def test_effective_experts_range():
    uniform = torch.full((2, 6, 8, 8), 1.0 / 6)
    single = torch.zeros(2, 6, 8, 8); single[:, 0] = 1.0
    assert abs(effective_experts(uniform) - 6.0) < 0.1
    assert abs(effective_experts(single) - 1.0) < 0.1


def test_losses_differentiable():
    gate_raw = torch.rand(1, 6, 4, 4, requires_grad=True)
    gate = gate_raw.softmax(dim=1)
    proxy = torch.rand(1, 6, 4, 4).softmax(dim=1)
    (router_supervision_loss(gate, proxy) + load_balance_loss(gate)).backward()
    assert gate_raw.grad is not None
