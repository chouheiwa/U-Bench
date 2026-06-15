import math
import os

import pytest
import torch
import torch.nn.functional as F

from models.Hybrid.USEANet import usea_loss as L


def _mk(seed=0, n=2, h=16, w=16):
    """小型二分类(num_classes=1)样本:pred/bg 为 logits,mask 为 {0,1} float。"""
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(n, 1, h, w, generator=g, requires_grad=True)
    pred_bg = torch.randn(n, 1, h, w, generator=g)
    mask_fg = torch.zeros(n, 1, h, w)
    mask_fg[:, :, 4:12, 4:12] = 1.0  # 中央前景块
    mask_bg = 1.0 - mask_fg
    return pred, pred_bg, mask_fg, mask_bg


def _old_structure_loss(pred, pred_bg, mask_fg, mask_bg):
    """重构前的原始公式,作为字节级参照(num_classes=1,形状已对齐)。"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_fg, 31, 1, 15) - mask_fg)
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    p = torch.sigmoid(pred)
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    union = ((p + mask_fg) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou + 0.8 * wbce2).mean()


def test_default_is_byte_identical_to_old(monkeypatch):
    monkeypatch.delenv("USEANET_LOSS_REGION", raising=False)
    pred, pred_bg, mask_fg, mask_bg = _mk()
    got = L.structure_loss(pred, pred_bg, mask_fg, mask_bg, num_classes=1)
    ref = _old_structure_loss(pred, pred_bg, mask_fg, mask_bg)
    assert torch.allclose(got, ref, rtol=0, atol=0)


def test_iou_explicit_equals_default(monkeypatch):
    pred, pred_bg, mask_fg, mask_bg = _mk()
    monkeypatch.delenv("USEANET_LOSS_REGION", raising=False)
    d = L.structure_loss(pred, pred_bg, mask_fg, mask_bg, 1)
    monkeypatch.setenv("USEANET_LOSS_REGION", "iou")
    i = L.structure_loss(pred, pred_bg, mask_fg, mask_bg, 1)
    assert torch.allclose(d, i, rtol=0, atol=0)


def test_region_dice_perfect_pred_near_zero():
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1.0
    weit = torch.ones_like(mask)
    p = mask.clone()  # 完美预测
    r = L._region_dice(p, mask, weit)
    assert torch.all(r < 0.05)


def test_dice_path_finite_and_backprops(monkeypatch):
    monkeypatch.setenv("USEANET_LOSS_REGION", "dice")
    pred, pred_bg, mask_fg, mask_bg = _mk()
    loss = L.structure_loss(pred, pred_bg, mask_fg, mask_bg, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.all(torch.isfinite(pred.grad))


def test_focal_tversky_finite_and_backprops(monkeypatch):
    monkeypatch.setenv("USEANET_LOSS_REGION", "focal_tversky")
    pred, pred_bg, mask_fg, mask_bg = _mk()
    loss = L.structure_loss(pred, pred_bg, mask_fg, mask_bg, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.all(torch.isfinite(pred.grad))


def test_focal_tversky_beta_penalises_false_negatives():
    # 构造一个"漏检(FN 高、FP 低)"样本:有前景但预测几乎全背景。
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1.0
    weit = torch.ones_like(mask)
    p = torch.full_like(mask, 0.1)  # 欠预测 -> FN 大、FP 小
    gamma = 4.0 / 3.0
    high_fn_penalty = L._region_focal_tversky(p, mask, weit, alpha=0.3, beta=0.7, gamma=gamma)
    low_fn_penalty = L._region_focal_tversky(p, mask, weit, alpha=0.7, beta=0.3, gamma=gamma)
    # β>α(更狠惩罚漏检)应给出更大的损失
    assert (high_fn_penalty > low_fn_penalty).all()


def test_ft_params_read_from_env(monkeypatch):
    monkeypatch.setenv("USEANET_FT_ALPHA", "0.25")
    monkeypatch.setenv("USEANET_FT_BETA", "0.75")
    monkeypatch.setenv("USEANET_FT_GAMMA", "2.0")
    assert L._ft_params() == (0.25, 0.75, 2.0)


def test_ft_params_defaults(monkeypatch):
    for k in ("USEANET_FT_ALPHA", "USEANET_FT_BETA", "USEANET_FT_GAMMA"):
        monkeypatch.delenv(k, raising=False)
    a, b, g = L._ft_params()
    assert (a, b) == (0.3, 0.7)
    assert g == pytest.approx(4.0 / 3.0)


def test_invalid_region_falls_back_to_iou(monkeypatch, recwarn):
    pred, pred_bg, mask_fg, mask_bg = _mk()
    monkeypatch.delenv("USEANET_LOSS_REGION", raising=False)
    ref = L.structure_loss(pred.detach().requires_grad_(True), pred_bg, mask_fg, mask_bg, 1)
    monkeypatch.setenv("USEANET_LOSS_REGION", "bogus")
    got = L.structure_loss(pred.detach().requires_grad_(True), pred_bg, mask_fg, mask_bg, 1)
    assert torch.allclose(got, ref, rtol=0, atol=0)
    assert any("falling back to 'iou'" in str(w.message) for w in recwarn.list)


@pytest.mark.parametrize("region", ["iou", "dice", "focal_tversky"])
def test_end_to_end_via_deep_supervision(monkeypatch, region):
    monkeypatch.setenv("USEANET_LOSS_REGION", region)
    from models.Hybrid.USEANet import USEANet
    torch.manual_seed(0)
    model = USEANet(input_channel=3, num_classes=1)
    x = torch.randn(2, 3, 64, 64)
    label = torch.zeros(2, 1, 64, 64)
    label[:, :, 16:48, 16:48] = 1.0
    outputs = model(x)
    loss = model.deep_supervision_loss(outputs, label)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()  # 与 MoE 辅助 + 多尺度 + 背景分支组合后仍可反传


def test_focal_tversky_no_nan_grad_at_saturation(monkeypatch):
    # Perfect, saturated prediction makes TI hit exactly 1 -> (1-TI)=0.
    # The clamp must keep the gradient finite (no NaN through ^(1/gamma)).
    monkeypatch.setenv("USEANET_LOSS_REGION", "focal_tversky")
    mask_fg = torch.zeros(1, 1, 16, 16)
    mask_fg[:, :, 4:12, 4:12] = 1.0
    mask_bg = 1.0 - mask_fg
    # logits that saturate sigmoid: huge positive on fg, huge negative on bg
    pred = (mask_fg * 60.0 - 30.0).clone().requires_grad_(True)
    pred_bg = (mask_bg * 60.0 - 30.0)
    loss = L.structure_loss(pred, pred_bg, mask_fg, mask_bg, 1)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.all(torch.isfinite(pred.grad))
