import torch
from models.Hybrid.USEANet.moe import PhysicsMoE


def _moe(in_c=160, out_c=32):
    torch.manual_seed(0)
    return PhysicsMoE(in_c, out_c)


def test_output_shape_matches_multibranch_contract():
    feat = torch.rand(2, 160, 16, 16)
    out = _moe()(feat)
    assert out.shape == (2, 32, 16, 16)


def test_exposes_last_gate_and_proxy_after_forward():
    feat = torch.rand(2, 160, 16, 16)
    m = _moe()
    m(feat)
    assert m.last_gate.shape == (2, 6, 16, 16)
    assert m.last_proxy.shape == (2, 6, 16, 16)


def test_aux_loss_is_scalar_and_finite():
    feat = torch.rand(2, 160, 16, 16)
    m = _moe()
    m(feat)
    aux = m.aux_loss(route_weight=0.5, lb_weight=0.01)
    assert aux.dim() == 0 and torch.isfinite(aux)


def test_forward_backward():
    feat = torch.rand(1, 160, 16, 16, requires_grad=True)
    m = _moe()
    out = m(feat)
    (out.sum() + m.aux_loss(0.5, 0.01)).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
