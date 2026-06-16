import os
import torch
import pytest
from models.Hybrid.USEANet.moe import EXPERT_NAMES


def test_learnable_prefilter_init_and_shape():
    from models.Hybrid.USEANet.moe.experts import _LearnablePrefilter
    k = torch.tensor([[1.0, 2.0, 1.0], [0, 0, 0], [-1.0, -2.0, -1.0]])
    pf = _LearnablePrefilter(k)
    # 权重以物理核初始化(单核,broadcast over channels)
    assert torch.allclose(pf.weight.detach().view(3, 3), k)
    assert pf.weight.requires_grad
    feat = torch.rand(2, 16, 16, 16)
    out = pf(feat)
    assert out.shape == (2, 16, 16, 16)  # depthwise 保持通道与尺寸


def test_learnable_prefilter_anisotropic_and_dilated():
    from models.Hybrid.USEANet.moe.experts import _LearnablePrefilter
    vk = torch.tensor([[1.0], [1.0], [1.0], [0.0], [-1.0], [-1.0], [-1.0]])  # 7x1
    pf = _LearnablePrefilter(vk)
    out = pf(torch.rand(1, 8, 12, 12))
    assert out.shape == (1, 8, 12, 12)
    dk = torch.ones(3, 3) / 9.0
    pfd = _LearnablePrefilter(dk, dilation=2)
    outd = pfd(torch.rand(1, 8, 12, 12))
    assert outd.shape == (1, 8, 12, 12)


def test_hetero_expert_shapes_x3_x4():
    from models.Hybrid.USEANet.moe.experts import _build_hetero_experts
    for in_ch in (160, 256):                       # x3, x4
        experts = _build_hetero_experts(in_ch, 32, 32)
        assert len(experts) == len(EXPERT_NAMES)
        feat = torch.rand(2, in_ch, 8, 8)
        for e in experts:
            assert e(feat).shape == (2, 32, 8, 8)


def test_hetero_kernels_cover_all_experts():
    from models.Hybrid.USEANet.moe.experts import _hetero_kernels
    ks = _hetero_kernels()
    for name in EXPERT_NAMES:
        assert name in ks, f"missing hetero kernel spec for {name}"


def test_build_experts_dispatch(monkeypatch):
    from models.Hybrid.USEANet.moe.experts import (
        build_experts, _AnchoredExpert, _HeteroExpert,
    )
    monkeypatch.delenv("USEANET_HETERO_EXPERTS", raising=False)
    homo = build_experts(160, 32)
    assert all(isinstance(e, _AnchoredExpert) for e in homo)
    assert len(homo) == len(EXPERT_NAMES)

    monkeypatch.setenv("USEANET_HETERO_EXPERTS", "1")
    hetero = build_experts(160, 32)
    assert all(isinstance(e, _HeteroExpert) for e in hetero)
    assert len(hetero) == len(EXPERT_NAMES)


def test_hetero_param_budget():
    from models.Hybrid.USEANet.moe.experts import _build_hetero_experts, build_experts
    import os as _os
    # 异构每专家 < 60k(增容但不爆);整组 < 同构整组的 3x
    hetero = _build_hetero_experts(256, 32, 32)
    for e in hetero:
        n = sum(p.numel() for p in e.parameters() if p.requires_grad)
        assert n < 60000, f"hetero expert too heavy: {n}"
    homo_total = sum(p.numel() for e in build_experts(256, 32) for p in e.parameters())
    hetero_total = sum(p.numel() for e in hetero for p in e.parameters())
    assert hetero_total < 3 * homo_total, f"hetero {hetero_total} vs homo {homo_total}"


def test_hetero_prefilter_gets_gradient():
    from models.Hybrid.USEANet.moe.experts import _build_hetero_experts
    e = _build_hetero_experts(16, 32, 32)[2]   # shadow, 7x1 learnable kernel
    out = e(torch.rand(1, 16, 12, 12))
    out.sum().backward()
    g = e.prefilters[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_hetero_structure_is_real():
    from models.Hybrid.USEANet.moe.experts import _build_hetero_experts
    experts = _build_hetero_experts(160, 32, 32)
    idx = {name: i for i, name in enumerate(EXPERT_NAMES)}
    # SE only on contrast
    assert hasattr(experts[idx["contrast"]], "se")
    assert not hasattr(experts[idx["edge"]], "se")
    # anisotropic shadow prefilter (7x1) differs in shape from edge (3x3)
    shadow_k = experts[idx["shadow"]].prefilters[0].weight.shape[-2:]
    edge_k = experts[idx["edge"]].prefilters[0].weight.shape[-2:]
    assert shadow_k == (7, 1)
    assert edge_k == (3, 3)
    assert shadow_k != edge_k


def test_physics_moe_end_to_end_hetero(monkeypatch):
    monkeypatch.setenv("USEANET_HETERO_EXPERTS", "1")
    from models.Hybrid.USEANet.moe.physics_moe import PhysicsMoE
    from models.Hybrid.USEANet.moe.experts import _HeteroExpert
    moe = PhysicsMoE(in_channel=160, out_channel=32)
    assert all(isinstance(e, _HeteroExpert) for e in moe.experts)
    x = torch.rand(2, 160, 16, 16)
    out = moe(x)
    assert out.shape == (2, 32, 16, 16)
    loss = moe.aux_loss(route_weight=0.1, lb_weight=0.1)
    assert torch.isfinite(loss)
    eff = moe.eff_experts()
    assert 1.0 <= eff <= len(EXPERT_NAMES) + 1e-4


def test_hetero_env_knobs(monkeypatch):
    """Overfit-reduction env knobs: channel override, no-SE, freeze, dropout."""
    from models.Hybrid.USEANet.moe.experts import _build_hetero_experts
    idx = {name: i for i, name in enumerate(EXPERT_NAMES)}
    monkeypatch.setenv("USEANET_HETERO_CHANNEL", "32")
    monkeypatch.setenv("USEANET_HETERO_NO_SE", "1")
    monkeypatch.setenv("USEANET_HETERO_FREEZE", "1")
    monkeypatch.setenv("USEANET_HETERO_DROPOUT", "0.3")
    experts = _build_hetero_experts(256, 32, 32)
    # channel override -> contrast head first conv out = 32 (was 48)
    assert experts[idx["contrast"]].head[0].out_channels == 32
    # no-SE -> contrast loses its se branch
    assert not hasattr(experts[idx["contrast"]], "se")
    # freeze -> prefilter kernels not trainable
    assert experts[idx["shadow"]].prefilters[0].weight.requires_grad is False
    # dropout override -> Dropout2d p = 0.3
    drops = [m for m in experts[idx["edge"]].head if isinstance(m, __import__("torch").nn.Dropout2d)]
    assert drops and abs(drops[0].p - 0.3) < 1e-9
    # forward still contract-correct
    assert experts[idx["contrast"]](__import__("torch").rand(2, 256, 8, 8)).shape == (2, 32, 8, 8)
