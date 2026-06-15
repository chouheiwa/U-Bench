# USEANet 配方阶梯 · 优化器组 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把判别式 LR / warmup / 权重 EMA / AdamW 做成一个 env 门控、默认全关的隔离模块,并接入 `main.py`,env 全关时训练行为与改动前完全一致。

**Architecture:** 新增纯逻辑模块 `models/Hybrid/USEANet/training_recipe.py`(可单测,不依赖真模型/GPU);`main.py` 的 `train()` 把优化器构造、LR 调度、验证/best 存盘三处用 **`args.model == 'USEANet'`** 门控——其余模型走与原文件逐字节一致的 `optim.SGD(...)` 分支,且根本不 import 该模块。

**Tech Stack:** PyTorch (`torch`, `torch.optim`)、pytest、Python 标准库 `os`/`warnings`。

**Spec:** `docs/superpowers/specs/2026-06-15-useanet-recipe-optimizer-design.md`

**关键事实(实查得到):**
- 适配器 `models/Hybrid/USEANet/__init__.py` 的 `USEANet` 持有 `self.net = _USEANetCore(...)`,core 持有 `self.backbone = pvt_v2_b0()`。故 backbone 参数前缀为 **`net.backbone.`** → `BACKBONE_PARAM_PREFIXES = ("net.backbone.",)`。
- `main.py:269` 原优化器:`optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)`。
- `main.py:345-347` 原 LR 调度:`lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9` 后写入所有 `param_groups`。
- `main.py:343` 是 `optimizer.step()`;`:353` 是验证块的 `model.eval()`;`:413-417` best 存盘(`model.state_dict()`);`:428-436` `checkpoint_final`(存 `state_dict + optimizer`)。
- 已有 env 开关命名约定 `USEANET_*`(见 `USEANET_NO_MOE` / `USEANET_WEIGHTED_DS`)。

**铁律:** env 全关 ⇒ 训练行为与改动前一致;其余模型零影响(连 import 都不触发)。

**测试命令:** `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -q`

---

## Task 1: `lr_at` —— 调度函数(warmup + poly,默认退化为原公式)

**Files:**
- Create: `models/Hybrid/USEANet/training_recipe.py`
- Test: `tests/test_training_recipe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training_recipe.py
import math
import pytest
from models.Hybrid.USEANet import training_recipe as tr


def _orig_poly(base, i, n):
    return base * (1.0 - i / n) ** 0.9


def test_lr_at_no_warmup_matches_original_poly():
    base, n = 0.01, 1000
    for i in [0, 1, 250, 500, 999]:
        assert tr.lr_at(base, i, n, 0) == pytest.approx(_orig_poly(base, i, n))


def test_lr_at_warmup_is_linear_then_poly():
    base, n, w = 0.01, 1000, 100
    # linear ramp inside warmup: (i+1)/w * base
    assert tr.lr_at(base, 0, n, w) == pytest.approx(base * 1 / w)
    assert tr.lr_at(base, w - 1, n, w) == pytest.approx(base * w / w)  # peak == base
    # just after warmup, poly over the remaining (n-w) iters, starts at base
    assert tr.lr_at(base, w, n, w) == pytest.approx(base * (1.0 - 0 / (n - w)) ** 0.9)
    # monotonic non-increasing after the peak
    prev = base
    for i in range(w, n):
        cur = tr.lr_at(base, i, n, w)
        assert cur <= prev + 1e-12
        prev = cur
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -q`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'lr_at'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/training_recipe.py
"""USEANet-only training recipe levers (discriminative LR / warmup / weight EMA /
AdamW). Every lever is gated by a ``USEANET_*`` environment variable and is
default-off; with all switches off the helpers reproduce ``main.py``'s original
SGD + poly schedule exactly. This module imports only torch + stdlib so it never
pulls in the heavy USEANet backbone."""

import os
import warnings

import torch
from torch import optim

# Adapter (models/Hybrid/USEANet/__init__.py) holds self.net -> core.self.backbone,
# so the pretrained PVT-B0 params are prefixed "net.backbone." in the adapter's
# named_parameters().
BACKBONE_PARAM_PREFIXES = ("net.backbone.",)


def lr_at(group_base_lr, iter_num, max_iterations, warmup_iters):
    """Per-group LR. warmup_iters==0 reduces to main.py's original poly schedule:
    group_base_lr * (1 - iter_num/max_iterations) ** 0.9."""
    if warmup_iters > 0 and iter_num < warmup_iters:
        return group_base_lr * (iter_num + 1) / warmup_iters
    denom = max_iterations - warmup_iters
    return group_base_lr * (1.0 - (iter_num - warmup_iters) / denom) ** 0.9
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/training_recipe.py tests/test_training_recipe.py
git commit -m "feat(USEANet): add lr_at warmup+poly schedule helper"
```

---

## Task 2: `warmup_iters` —— 读 env 算 warmup 迭代数

**Files:**
- Modify: `models/Hybrid/USEANet/training_recipe.py`
- Test: `tests/test_training_recipe.py`

- [ ] **Step 1: Write the failing test**

```python
def test_warmup_iters_default_zero(monkeypatch):
    monkeypatch.delenv("USEANET_WARMUP_EPOCHS", raising=False)
    assert tr.warmup_iters(125) == 0


def test_warmup_iters_reads_env(monkeypatch):
    monkeypatch.setenv("USEANET_WARMUP_EPOCHS", "5")
    assert tr.warmup_iters(125) == 5 * 125
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -k warmup_iters -q`
Expected: FAIL — `AttributeError: ... 'warmup_iters'`.

- [ ] **Step 3: Write minimal implementation**

Append to `training_recipe.py`:

```python
def warmup_iters(iters_per_epoch):
    """USEANET_WARMUP_EPOCHS (default 0) * iters_per_epoch."""
    epochs = int(os.environ.get("USEANET_WARMUP_EPOCHS", "0"))
    return epochs * iters_per_epoch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -k warmup_iters -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/training_recipe.py tests/test_training_recipe.py
git commit -m "feat(USEANet): add warmup_iters env helper"
```

---

## Task 3: `build_optimizer` —— 默认 passthrough(与原 SGD 逐字节等价)

**Files:**
- Modify: `models/Hybrid/USEANet/training_recipe.py`
- Test: `tests/test_training_recipe.py`

- [ ] **Step 1: Write the failing test**

```python
import torch.nn as nn
from torch import optim


class _TinyNet(nn.Module):
    """Mimics the adapter's name layout: a 'net.backbone.*' subtree + a head."""
    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.backbone = nn.Linear(4, 4)   # -> params named net.backbone.*
        self.head = nn.Linear(4, 1)           # -> params named head.*


def test_build_optimizer_default_is_plain_sgd(monkeypatch):
    for k in ("USEANET_DISC_LR", "USEANET_ADAMW", "USEANET_BACKBONE_LR_MULT"):
        monkeypatch.delenv(k, raising=False)
    model = _TinyNet()
    opt, group_base_lrs = tr.build_optimizer(model, 0.01)
    assert isinstance(opt, optim.SGD)
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["lr"] == pytest.approx(0.01)
    assert g["momentum"] == pytest.approx(0.9)
    assert g["weight_decay"] == pytest.approx(0.0001)
    assert group_base_lrs == [0.01]
    # all trainable params present, none dropped
    n_model = sum(1 for p in model.parameters() if p.requires_grad)
    n_opt = sum(len(grp["params"]) for grp in opt.param_groups)
    assert n_opt == n_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -k build_optimizer_default -q`
Expected: FAIL — `AttributeError: ... 'build_optimizer'`.

- [ ] **Step 3: Write minimal implementation**

Append to `training_recipe.py`:

```python
def _split_params(model):
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pref) for pref in BACKBONE_PARAM_PREFIXES):
            backbone.append(p)
        else:
            rest.append(p)
    return backbone, rest


def build_optimizer(model, base_lr):
    """Return (optimizer, group_base_lrs). With USEANET_DISC_LR and USEANET_ADAMW
    both off this is byte-identical to main.py's original:
    optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)."""
    disc = os.environ.get("USEANET_DISC_LR") == "1"
    adamw = os.environ.get("USEANET_ADAMW") == "1"

    if not disc and not adamw:
        opt = optim.SGD(model.parameters(), lr=base_lr,
                        momentum=0.9, weight_decay=0.0001)
        return opt, [base_lr]

    if disc:
        mult = float(os.environ.get("USEANET_BACKBONE_LR_MULT", "0.1"))
        backbone, rest = _split_params(model)
        if not backbone:
            warnings.warn(
                "USEANET_DISC_LR: no params matched %s; falling back to a single "
                "group" % (BACKBONE_PARAM_PREFIXES,))
            groups = [{"params": rest, "lr": base_lr}]
            group_base_lrs = [base_lr]
        else:
            backbone_lr = base_lr * mult
            groups = [{"params": backbone, "lr": backbone_lr},
                      {"params": rest, "lr": base_lr}]
            group_base_lrs = [backbone_lr, base_lr]
    else:  # adamw only, single group
        groups = [{"params": list(model.parameters()), "lr": base_lr}]
        group_base_lrs = [base_lr]

    if adamw:
        opt = optim.AdamW(groups, lr=base_lr, weight_decay=0.0001)
    else:
        opt = optim.SGD(groups, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    return opt, group_base_lrs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -k build_optimizer_default -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/training_recipe.py tests/test_training_recipe.py
git commit -m "feat(USEANet): build_optimizer default passthrough to plain SGD"
```

---

## Task 4: `build_optimizer` —— 判别式 LR 分组 + AdamW

**Files:**
- Modify: `tests/test_training_recipe.py`(无源码改动;逻辑已在 Task 3 写好,本任务是分组语义验证 gate)

- [ ] **Step 1: Write the failing test**

```python
def test_build_optimizer_disc_lr_two_groups(monkeypatch):
    monkeypatch.setenv("USEANET_DISC_LR", "1")
    monkeypatch.setenv("USEANET_BACKBONE_LR_MULT", "0.1")
    monkeypatch.delenv("USEANET_ADAMW", raising=False)
    model = _TinyNet()
    opt, group_base_lrs = tr.build_optimizer(model, 0.01)
    assert isinstance(opt, optim.SGD)
    assert len(opt.param_groups) == 2
    assert group_base_lrs == [pytest.approx(0.001), pytest.approx(0.01)]
    assert opt.param_groups[0]["lr"] == pytest.approx(0.001)  # backbone group
    assert opt.param_groups[1]["lr"] == pytest.approx(0.01)   # rest group
    # union == all trainable params, no overlap, no drop
    n_model = sum(1 for p in model.parameters() if p.requires_grad)
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    assert n_opt == n_model
    bb = {id(p) for p in opt.param_groups[0]["params"]}
    rest = {id(p) for p in opt.param_groups[1]["params"]}
    assert bb.isdisjoint(rest)


def test_build_optimizer_adamw(monkeypatch):
    monkeypatch.setenv("USEANET_ADAMW", "1")
    monkeypatch.delenv("USEANET_DISC_LR", raising=False)
    model = _TinyNet()
    opt, group_base_lrs = tr.build_optimizer(model, 0.01)
    assert isinstance(opt, optim.AdamW)
    assert group_base_lrs == [0.01]


def test_build_optimizer_disc_no_backbone_falls_back(monkeypatch):
    monkeypatch.setenv("USEANET_DISC_LR", "1")
    monkeypatch.delenv("USEANET_ADAMW", raising=False)
    model = nn.Linear(4, 1)  # no 'net.backbone.*' params
    with pytest.warns(UserWarning):
        opt, group_base_lrs = tr.build_optimizer(model, 0.01)
    assert len(opt.param_groups) == 1
    assert group_base_lrs == [0.01]
```

- [ ] **Step 2: Run tests**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -k "disc_lr or adamw or backbone" -q`
Expected: PASS immediately (logic already in Task 3). If any FAIL, fix `build_optimizer` until green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_training_recipe.py
git commit -m "test(USEANet): cover discriminative-LR grouping and AdamW"
```

---

## Task 5: `ModelEMA` —— 影子权重

**Files:**
- Modify: `models/Hybrid/USEANet/training_recipe.py`
- Test: `tests/test_training_recipe.py`

- [ ] **Step 1: Write the failing test**

```python
import torch


def test_model_ema_update_formula():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = tr.ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    # shadow = 0.9*1.0 + 0.1*2.0 = 1.1
    assert ema.shadow["weight"].item() == pytest.approx(1.1)


def test_model_ema_store_copy_restore_roundtrip():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema = tr.ModelEMA(model, decay=0.9)  # shadow == 2.0
    with torch.no_grad():
        model.weight.fill_(5.0)          # "real" training weights
    ema.store(model)                      # back up 5.0
    ema.copy_to(model)                    # load shadow 2.0 for eval
    assert model.weight.item() == pytest.approx(2.0)
    ema.restore(model)                    # restore 5.0
    assert model.weight.item() == pytest.approx(5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -k model_ema -q`
Expected: FAIL — `AttributeError: ... 'ModelEMA'`.

- [ ] **Step 3: Write minimal implementation**

Append to `training_recipe.py`:

```python
class ModelEMA:
    """Exponential moving average of model weights. best ckpt is saved from the
    EMA shadow; checkpoint_final keeps the real training weights (resume-correct)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self._backup = None

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                s = self.shadow[k]
                if torch.is_floating_point(v):
                    s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    s.copy_(v)  # buffers (e.g. counters) track the latest value

    def store(self, model):
        self._backup = {k: v.detach().clone()
                        for k, v in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

    def state_dict(self):
        return self.shadow
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/test_training_recipe.py -q`
Expected: PASS (all module tests green).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/training_recipe.py tests/test_training_recipe.py
git commit -m "feat(USEANet): add ModelEMA shadow-weight helper"
```

---

## Task 6: 接入 `main.py`(USEANet-only 门控,默认 passthrough)

**Files:**
- Modify: `main.py:269`(optimizer 构造)、`main.py:343`(EMA update)、`main.py:345-347`(LR 调度)、`main.py:353`(eval 前 copy EMA)、`main.py:428`(checkpoint_final 前 restore)

> 没有自动化测试覆盖 `main.py`(需 GPU + 数据)。正确性靠两点保证:(1) 非 USEANet 分支与原文件**逐字符一致**;(2) USEANet 全关时 `build_optimizer`→原 SGD、`warmup_iters`→0、`_ema=None`,EMA 三处是 no-op。验证见 Step 6。

- [ ] **Step 1: 替换 `main.py:269` 的优化器构造**

Old (`main.py:269`):
```python
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
```
New:
```python
    # USEANet-only optional training recipe (discriminative LR / warmup / EMA /
    # AdamW), all env-gated and default-off. Other models keep the exact SGD in
    # the else branch and never import the USEANet recipe module.
    if args.model == 'USEANet':
        from models.Hybrid.USEANet import training_recipe as _recipe
        optimizer, _group_base_lrs = _recipe.build_optimizer(model, base_lr)
        _warmup_iters = _recipe.warmup_iters(len(trainloader))
        _ema = (_recipe.ModelEMA(model, float(os.environ.get("USEANET_EMA_DECAY", "0.999")))
                if os.environ.get("USEANET_EMA") == "1" else None)
    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        _group_base_lrs = [base_lr]
        _warmup_iters = 0
        _ema = None
```

- [ ] **Step 2: EMA update —— 在 `optimizer.step()`(`main.py:343`)后加**

Old:
```python
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```
New:
```python
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if _ema is not None:
                _ema.update(model)
```

- [ ] **Step 3: 替换 LR 调度(`main.py:345-347`)**

Old:
```python
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
```
New:
```python
            if args.model == 'USEANet':
                for param_group, _gbase in zip(optimizer.param_groups, _group_base_lrs):
                    param_group['lr'] = _recipe.lr_at(_gbase, iter_num, max_iterations, _warmup_iters)
            else:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
```

- [ ] **Step 4: eval 前换上 EMA 权重(`main.py:353` 的 `model.eval()` 前)**

Old:
```python
        model.eval()
        with torch.no_grad():
```
New:
```python
        if _ema is not None:
            _ema.store(model)
            _ema.copy_to(model)
        model.eval()
        with torch.no_grad():
```

- [ ] **Step 5: best 存盘后、`checkpoint_final`(`main.py:428`)前还原真实权重**

Old(`main.py:426-428`):
```python
            train_metric_dict["last_ACC"] = avg_meters['ACC'].avg

        checkpoint_path = os.path.join(exp_save_dir, f'checkpoint_final.pth')
```
New:
```python
            train_metric_dict["last_ACC"] = avg_meters['ACC'].avg

        if _ema is not None:
            _ema.restore(model)

        checkpoint_path = os.path.join(exp_save_dir, f'checkpoint_final.pth')
```

> 顺序保证:eval/val 与 best 存盘(`:413` 的 `model.state_dict()`)在 EMA 权重下进行 → best ckpt 存 EMA;`last_*` 块只读 `avg_meters`,不碰 model;restore 后 `checkpoint_final` 存真实权重 + optimizer(resume 正确);下一轮 `model.train()` 从真实权重继续。

- [ ] **Step 6: 验证默认行为不变 + 语法**

Run:
```bash
# 语法/import 不报错
conda run -n ubench1 python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"
# 模块测试仍全绿
conda run -n ubench1 python -m pytest tests/test_training_recipe.py -q
# 回归:确认非 USEANet 分支与原始 SGD 行完全一致(人工 diff 审阅本次改动)
git diff main.py
```
Expected: parse OK;module tests 全绿;`git diff` 中非 USEANet 路径是原始 `optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)` 与原始 poly 公式,逐字符一致。

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat(USEANet): wire optional recipe (disc-LR/warmup/EMA/AdamW) into train(), gated USEANet-only"
```

---

## 收尾

- [ ] 全量测试:`conda run -n ubench1 python -m pytest tests/ -q`(原 49 + 本次新增,应全绿)。
- [ ] 派 final code reviewer 审整支;然后用 superpowers:finishing-a-development-branch 合并。
- [ ] 实验阶段(非本计划编码范围):按 spec §8 用 `USEANET_DISC_LR/WARMUP/EMA` 在 BUSI 强增强 250ep 上跑 3 seed(41/42/43),对照 0.682 基线做 keep/drop。
