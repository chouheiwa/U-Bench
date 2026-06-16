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
