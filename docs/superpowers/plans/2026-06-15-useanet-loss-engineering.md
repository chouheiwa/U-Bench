# USEANet 损失工程(B 路线)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `usea_loss.py` 内引入可切换区域项(`iou` 默认 / `dice` / `focal_tversky`),保留 USEANet 边界权重 + 背景分支(B1 形态),默认路径字节级等价现状。

**Architecture:** 把区域项抽成三个纯函数(接已 sigmoid 的 `p`、`mask_fg`、`weit`),`structure_loss` 内按 `USEANET_LOSS_REGION` 解析一次并 dispatch。签名不变,`__init__.py`/`main.py` 零改动。focal-Tversky 超参经 `USEANET_FT_*` env 读取,带文献默认 α=0.3/β=0.7/γ=4/3。

**Tech Stack:** PyTorch 2.7、pytest、conda env `ubench1`。分支 `feat/useanet-loss`(已创建)。

**参考 spec:** `docs/superpowers/specs/2026-06-15-useanet-loss-engineering-design.md`

---

## File Structure

- **Modify:** `models/Hybrid/USEANet/usea_loss.py` — 新增三个区域项函数 + env 解析 helper,重构 `structure_loss` 走 dispatch。唯一改动的生产文件。
- **Create:** `tests/test_usea_loss.py` — 区域项行为、零影响不变量、env 解析、端到端测试。
- **不动:** `models/Hybrid/USEANet/__init__.py`(`deep_supervision_loss` 适配器)、`main.py`、其他模型。

测试约定参考现有 `tests/test_training_recipe.py`(顶层 `tests/`、`import torch`、`pytest.approx`)。

---

### Task 1: 区域项 dispatch + `dice` + 零影响不变量

**Files:**
- Modify: `models/Hybrid/USEANet/usea_loss.py`
- Test: `tests/test_usea_loss.py` (create)

- [ ] **Step 1: 写失败测试**

新建 `tests/test_usea_loss.py`:

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_usea_loss.py -q`
Expected: FAIL —`AttributeError: module ... has no attribute '_region_dice'`。

- [ ] **Step 3: 重构 `usea_loss.py`**

把 `structure_loss` 改成 dispatch,并新增区域项函数 + env helper。文件改为:

```python
"""USEANet's weighted dual-branch (foreground/background) structure loss.

(原 docstring 保留;区域项现可经 USEANET_LOSS_REGION 在 iou/dice/focal_tversky 间切换。)
"""

import os
import warnings

import torch
import torch.nn.functional as F


def expand_as_one_hot(input_tensor, num_classes):
    """(N, H, W) or (N, 1, H, W) label -> (N, num_classes, H, W) one-hot float."""
    if input_tensor.dim() == 4 and input_tensor.size(1) == 1:
        input_tensor = input_tensor.squeeze(1)
    input_tensor = input_tensor.long()
    one_hot = F.one_hot(input_tensor, num_classes=num_classes)
    return one_hot.permute(0, 3, 1, 2).float()


def _region_iou(p, mask_fg, weit):
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    union = ((p + mask_fg) * weit).sum(dim=(2, 3))
    return 1 - (inter + 1) / (union - inter + 1)


def _region_dice(p, mask_fg, weit):
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    psum = (p * weit).sum(dim=(2, 3))
    msum = (mask_fg * weit).sum(dim=(2, 3))
    return 1 - (2 * inter + 1) / (psum + msum + 1)


def _region_focal_tversky(p, mask_fg, weit, alpha, beta, gamma):
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    fp = ((p * (1 - mask_fg)) * weit).sum(dim=(2, 3))
    fn = (((1 - p) * mask_fg) * weit).sum(dim=(2, 3))
    ti = (inter + 1) / (inter + alpha * fp + beta * fn + 1)
    return (1 - ti) ** (1.0 / gamma)


def _resolve_region():
    mode = os.environ.get("USEANET_LOSS_REGION", "iou").lower()
    if mode not in ("iou", "dice", "focal_tversky"):
        warnings.warn(
            f"USEANET_LOSS_REGION={mode!r} invalid; falling back to 'iou'"
        )
        mode = "iou"
    return mode


def _ft_params():
    alpha = float(os.environ.get("USEANET_FT_ALPHA", "0.3"))
    beta = float(os.environ.get("USEANET_FT_BETA", "0.7"))
    gamma = float(os.environ.get("USEANET_FT_GAMMA", str(4.0 / 3.0)))
    return alpha, beta, gamma


def structure_loss(pred, pred_bg, mask_fg, mask_bg, num_classes):
    if pred.shape != mask_fg.shape:
        mask_fg = expand_as_one_hot(mask_fg.long(), num_classes)
        mask_bg = expand_as_one_hot(mask_bg.long(), num_classes)

    # Boundary-emphasising weight: amplify the penalty near object edges.
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_fg, kernel_size=31, stride=1, padding=15) - mask_fg)

    # Weighted BCE (foreground) — classification anchor, kept across all region modes.
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted BCE (background)
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)

    # Switchable region term (B1): iou (default) / dice / focal_tversky.
    mode = _resolve_region()
    if mode == "dice":
        region = _region_dice(pred, mask_fg, weit)
    elif mode == "focal_tversky":
        alpha, beta, gamma = _ft_params()
        region = _region_focal_tversky(pred, mask_fg, weit, alpha, beta, gamma)
    else:
        region = _region_iou(pred, mask_fg, weit)

    return (wbce + region + 0.8 * wbce2).mean()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_usea_loss.py -q`
Expected: PASS(5 个测试)。

- [ ] **Step 5: 提交**

```bash
git add models/Hybrid/USEANet/usea_loss.py tests/test_usea_loss.py
git commit -m "feat(USEANet): switchable region term (iou default/dice), zero-impact invariant"
```

---

### Task 2: `focal_tversky` 区域项 + FN 方向性 + env 超参

**Files:**
- Modify: `models/Hybrid/USEANet/usea_loss.py`(`_region_focal_tversky`/`_ft_params` 已在 Task 1 落地)
- Test: `tests/test_usea_loss.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_usea_loss.py`:

```python
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
```

- [ ] **Step 2: 跑测试**

Run: `conda run -n ubench1 python -m pytest tests/test_usea_loss.py -q`
Expected: PASS(`_region_focal_tversky`/`_ft_params` 已在 Task 1 实现,这些测试应直接绿)。
若任一失败,修 `usea_loss.py` 对应函数直到通过。

- [ ] **Step 3: 提交**

```bash
git add tests/test_usea_loss.py
git commit -m "test(USEANet): focal-Tversky FN-direction + env hyperparam coverage"
```

---

### Task 3: 非法 `USEANET_LOSS_REGION` fallback 到 `iou`

**Files:**
- Test: `tests/test_usea_loss.py`

- [ ] **Step 1: 写测试**

追加:

```python
def test_invalid_region_falls_back_to_iou(monkeypatch, recwarn):
    pred, pred_bg, mask_fg, mask_bg = _mk()
    monkeypatch.delenv("USEANET_LOSS_REGION", raising=False)
    ref = L.structure_loss(pred.detach().requires_grad_(True), pred_bg, mask_fg, mask_bg, 1)
    monkeypatch.setenv("USEANET_LOSS_REGION", "bogus")
    got = L.structure_loss(pred.detach().requires_grad_(True), pred_bg, mask_fg, mask_bg, 1)
    assert torch.allclose(got, ref, rtol=0, atol=0)
    assert any("falling back to 'iou'" in str(w.message) for w in recwarn.list)
```

- [ ] **Step 2: 跑测试**

Run: `conda run -n ubench1 python -m pytest tests/test_usea_loss.py -q`
Expected: PASS(`_resolve_region` 已含 fallback + warning)。

- [ ] **Step 3: 提交**

```bash
git add tests/test_usea_loss.py
git commit -m "test(USEANet): invalid USEANET_LOSS_REGION falls back to iou with warning"
```

---

### Task 4: 端到端经 `deep_supervision_loss` 跑三种区域项

**Files:**
- Test: `tests/test_usea_loss.py`

- [ ] **Step 1: 写测试**

追加(构造真实 USEANet,forward 后调适配器;CPU、小输入):

```python
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
```

> 注:若 `USEANet(...)` 构造或 forward 需要额外参数,参照 `tests/moe/test_adapter_integration.py:test_deep_supervision_loss_includes_moe_aux` 的构造方式对齐(同仓已有可用范例)。

- [ ] **Step 2: 跑测试**

Run: `conda run -n ubench1 python -m pytest tests/test_usea_loss.py -q`
Expected: PASS(3 个参数化)。

- [ ] **Step 3: 全量回归**

Run: `conda run -n ubench1 python -m pytest tests/ -q`
Expected: PASS,数量 = 原 74 + 本计划新增(无回归)。

- [ ] **Step 4: 提交**

```bash
git add tests/test_usea_loss.py
git commit -m "test(USEANet): end-to-end region-term switch through deep_supervision_loss"
```

---

## 验收(实现完成后,非本计划自动步骤)

按 spec 第 7 节,以 DISC_LR+strong-aug+250ep 为基线,seed41 跑 `dice`/`focal_tversky` 单点,超 0.7085+~0.01 才晋级 3-seed,结论写回 `docs/superpowers/NEXT-STEPS-useanet-moe.md`。GPU 训练不在本计划内。

## 自检结果

- **spec 覆盖**:§3 区域项公式→Task1/2;§4 开关+默认+fallback→Task1/3,FT 超参→Task2;§4 硬不变量→Task1 `test_default_is_byte_identical_to_old`;§6 测试 1-5→Task1-4 全覆盖;§5 落点→Task1 重构;§7 验收→「验收」节。
- **占位符**:无 TBD/TODO,所有 step 含完整代码或确切命令。
- **类型一致**:`_region_iou/_region_dice/_region_focal_tversky`、`_resolve_region`、`_ft_params`、`structure_loss` 跨 Task 命名/签名一致。
