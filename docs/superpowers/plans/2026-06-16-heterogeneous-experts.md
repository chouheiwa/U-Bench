# 异构物理专家(Heterogeneous Physics Experts)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 USEANet 的 6 个同构 MoE 专家改造为物理驱动的异构专家(可学物理初始核 + 异构结构 + 增容 + dropout),env 门控默认关,救回 C1a 弱增益。

**Architecture:** 全部改动局部化在 `models/Hybrid/USEANet/moe/experts.py`,新增「可学预滤波 + 异构专家」类与 `build_experts` 的 env 派发;`USEANET_HETERO_EXPERTS=1` 时返回异构专家,默认返回现同构 `_AnchoredExpert`。router/proxy/losses/physics_moe 的调用契约与通道(`[B,in,H,W]→[B,out=32,H,W]`、`EXPERT_NAMES` 顺序)完全不变。

**Tech Stack:** PyTorch 2.7,conda env `ubench1`,pytest。

参考 spec:`docs/superpowers/specs/2026-06-16-heterogeneous-experts-design.md`

---

## File Structure

- **Modify** `models/Hybrid/USEANet/moe/experts.py`:在文件内追加
  - `_LearnablePrefilter`(可学深度卷积,物理核初始化,支持非对称/膨胀核)
  - `_hetero_kernels()`(返回 6 类异构初始化核张量)
  - `_HeteroExpert`(统一异构专家:1~2 预滤波支 + 可选 SE + 带 Dropout 的头)
  - `_build_hetero_experts(in_channel, out_channel, channel)`
  - 改 `build_experts` 顶部加 env 派发
- **Create** `tests/moe/test_hetero_experts.py`(异构专家专属测试,显式设/清 env)
- 现有 `tests/moe/test_experts.py` 不改(env 默认关,继续测同构路径)。

测试运行环境:`conda run -n ubench1 python -m pytest tests/moe/ -q`(repo 根有 `conftest.py`)。

---

### Task 1: 可学物理初始预滤波 `_LearnablePrefilter`

**Files:**
- Modify: `models/Hybrid/USEANet/moe/experts.py`(在 `import` 段下方、`_AnchoredExpert` 之后追加)
- Test: `tests/moe/test_hetero_experts.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/moe/test_hetero_experts.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: FAIL — `ImportError: cannot import name '_LearnablePrefilter'`

- [ ] **Step 3: Write minimal implementation**

在 `experts.py` 顶部已有 `import os`?没有则在 `import torch` 上方加 `import os`。
在 `_AnchoredExpert` 类定义之后追加:

```python
class _LearnablePrefilter(nn.Module):
    """Depthwise prefilter shared across channels, physics-initialised but learnable.

    A single [kh,kw] kernel is broadcast over all input channels (groups=C) so the
    physical prior is anchored at init yet can drift during training. Supports
    anisotropic (e.g. 7x1) and dilated kernels for direction/scale-specialised experts.
    """

    def __init__(self, kernel, dilation=1):
        super().__init__()
        kh, kw = kernel.shape
        self.kh, self.kw, self.d = kh, kw, dilation
        self.weight = nn.Parameter(kernel.view(1, 1, kh, kw).clone().float())

    def forward(self, x):
        c = x.shape[1]
        k = self.weight.expand(c, 1, self.kh, self.kw)
        pad = (self.d * (self.kh // 2), self.d * (self.kw // 2))
        return F.conv2d(x, k, padding=pad, dilation=self.d, groups=c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/experts.py tests/moe/test_hetero_experts.py
git commit -m "feat(USEANet): learnable physics-initialised prefilter for hetero experts"
```

---

### Task 2: 异构专家类 `_HeteroExpert` + 6 类初始化核

**Files:**
- Modify: `models/Hybrid/USEANet/moe/experts.py`(追加 `_hetero_kernels`、`_HeteroExpert`)
- Test: `tests/moe/test_hetero_experts.py`

- [ ] **Step 1: Write the failing test**

在 `tests/moe/test_hetero_experts.py` 追加:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: FAIL — `ImportError: cannot import name '_build_hetero_experts'`

- [ ] **Step 3: Write minimal implementation**

在 `experts.py` 的 `_LearnablePrefilter` 之后、`_kernels()` 之后追加。先加异构核规格,
每个专家给一个 `(kernel_tensor, dilation)` 的预滤波支列表 + 头宽 `channel` + 是否带 SE:

```python
def _hetero_kernels():
    """Per-expert init kernels + structural knobs (design table). Reuses _kernels() bases."""
    base = _kernels()
    box5 = torch.ones(5, 5) / 25.0                                   # large despeckle
    vgrad7 = torch.tensor([[1.0], [1.0], [1.0], [0.0],
                           [-1.0], [-1.0], [-1.0]])                  # 7x1 shadow (vertical atten.)
    post5 = torch.tensor([[0.0], [0.0], [1.0], [2.0], [1.0]]) / 4.0  # 5x1 below-structure posterior
    return {
        # name: dict(prefilters=[(kernel, dilation), ...], channel=int, use_se=bool)
        "despeckle": dict(prefilters=[(box5, 1), (base["despeckle"], 2)], channel=48, use_se=False),
        "edge":      dict(prefilters=[(base["edge"], 1)],                 channel=48, use_se=False),
        "shadow":    dict(prefilters=[(vgrad7, 1)],                       channel=32, use_se=False),
        "posterior": dict(prefilters=[(post5, 1)],                       channel=32, use_se=False),
        "contrast":  dict(prefilters=[(base["contrast"], 1)],            channel=48, use_se=True),
        "hf":        dict(prefilters=[(base["hf"], 1), (base["hf"], 2)], channel=32, use_se=False),
    }


class _HeteroExpert(nn.Module):
    """Physics-anchored heterogeneous expert: 1-2 learnable prefilter branches (summed),
    optional channel-SE gate, then a Dropout-regularised lightweight head."""

    def __init__(self, in_channel, out_channel, prefilters, channel=32, use_se=False):
        super().__init__()
        self.prefilters = nn.ModuleList(
            _LearnablePrefilter(k, d) for (k, d) in prefilters
        )
        self.use_se = use_se
        if use_se:
            hidden = max(8, in_channel // 16)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, hidden, 1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, in_channel, 1), nn.Sigmoid(),
            )
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(channel, out_channel, 1, bias=False),
        )

    def forward(self, x):
        y = self.prefilters[0](x)
        for pf in self.prefilters[1:]:
            y = y + pf(x)
        if self.use_se:
            y = y * self.se(x)
        return self.head(y)


def _build_hetero_experts(in_channel, out_channel, channel=32):
    specs = _hetero_kernels()
    return nn.ModuleList(
        _HeteroExpert(in_channel, out_channel,
                      prefilters=specs[name]["prefilters"],
                      channel=specs[name]["channel"],
                      use_se=specs[name]["use_se"])
        for name in EXPERT_NAMES
    )
```

注:`_HeteroExpert` 的头宽用各专家自己的 `channel`,忽略 `_build_hetero_experts` 的 `channel` 形参
(形参仅为与 `build_experts` 签名兼容)。

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/experts.py tests/moe/test_hetero_experts.py
git commit -m "feat(USEANet): 6 heterogeneous physics experts (anisotropic/dilated/SE)"
```

---

### Task 3: `build_experts` 的 env 派发(默认关)

**Files:**
- Modify: `models/Hybrid/USEANet/moe/experts.py:51-56`(现有 `build_experts`)
- Test: `tests/moe/test_hetero_experts.py`

- [ ] **Step 1: Write the failing test**

在 `tests/moe/test_hetero_experts.py` 追加(用 monkeypatch 安全设/清 env):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py::test_build_experts_dispatch -q`
Expected: FAIL — hetero 分支返回的仍是 `_AnchoredExpert`(assert 失败)

- [ ] **Step 3: Write minimal implementation**

把现有 `build_experts` 改为:

```python
def build_experts(in_channel, out_channel, channel=32):
    if os.environ.get("USEANET_HETERO_EXPERTS") == "1":
        return _build_hetero_experts(in_channel, out_channel, channel)
    ks = _kernels()
    return nn.ModuleList(
        _AnchoredExpert(in_channel, out_channel, ks[name], channel)
        for name in EXPERT_NAMES
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: PASS(5 passed)

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/experts.py tests/moe/test_hetero_experts.py
git commit -m "feat(USEANet): env-gate hetero experts via USEANET_HETERO_EXPERTS (default off)"
```

---

### Task 4: 参数预算 + 梯度流回归

**Files:**
- Test: `tests/moe/test_hetero_experts.py`

- [ ] **Step 1: Write the failing test**

追加:

```python
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
```

- [ ] **Step 2: Run test to verify it fails (then likely pass)**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: 这两测无新代码,直接验证已实现行为;若 `test_hetero_param_budget` FAIL
(某专家超 60k),收 `contrast` 的 SE hidden 或把 `channel=48` 调回 32,再跑通。

- [ ] **Step 3: (仅当 Step 2 失败时)收容量**

若超预算,把 `_hetero_kernels()` 中 `contrast` 的 `channel` 由 48 改 32;仍超则 SE hidden
`max(8, in_channel // 16)` 改 `max(8, in_channel // 32)`。重跑直到 PASS。

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py -q`
Expected: PASS(7 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/moe/test_hetero_experts.py models/Hybrid/USEANet/moe/experts.py
git commit -m "test(USEANet): hetero expert param-budget + prefilter gradient regression"
```

---

### Task 5: 端到端 — `PhysicsMoE` 在 hetero env 下前向 + aux_loss

**Files:**
- Test: `tests/moe/test_hetero_experts.py`

- [ ] **Step 1: Write the failing test**

追加:

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_hetero_experts.py::test_physics_moe_end_to_end_hetero -q`
Expected: PASS(`PhysicsMoE` 经 `build_experts` 自动拿到异构专家,契约不变)

- [ ] **Step 3: 全套回归(确认未破坏同构路径)**

Run: `conda run -n ubench1 python -m pytest tests/moe/ -q`
Expected: 全部 PASS(原 `test_experts.py` 等同构测试不受影响,env 默认关)

- [ ] **Step 4: Commit**

```bash
git add tests/moe/test_hetero_experts.py
git commit -m "test(USEANet): end-to-end PhysicsMoE forward+aux_loss under hetero experts"
```

---

### Task 6: 冒烟验证整模型可构建可前向(GPU 非必需)

**Files:** 无(纯验证)

- [ ] **Step 1: env 开启下整模型前向冒烟**

Run:
```bash
cd /home/chouheiwa/python/U-Bench
USEANET_HETERO_EXPERTS=1 conda run -n ubench1 python -c "
import torch
from types import SimpleNamespace
from models import build_model
args=SimpleNamespace(model='USEANet',model_id=115,img_size=256,num_classes=1,input_channel=3)
m=build_model(args,input_channel=3,num_classes=1).eval()
y=m(torch.randn(2,3,256,256))
print('OK out=', (y[-1].shape if isinstance(y,(list,tuple)) else y.shape),
      'params=%.3fM'%(sum(p.numel() for p in m.parameters())/1e6))
"
```
Expected: `OK out= (2, 1, 256, 256) params≈3.79M`(同构基线 ~3.66M,Δ≈+0.13M,符合 spec 预算)

- [ ] **Step 2: 确认 env 关时参数回到同构基线**

Run:
```bash
conda run -n ubench1 python -c "
import torch
from types import SimpleNamespace
from models import build_model
args=SimpleNamespace(model='USEANet',model_id=115,img_size=256,num_classes=1,input_channel=3)
m=build_model(args,input_channel=3,num_classes=1)
print('homo params=%.3fM'%(sum(p.numel() for p in m.parameters())/1e6))
"
```
Expected: `homo params≈3.66M`(与现状一致 → 默认路径零回归)

实现完成后,异构专家可用以下配方上 seed41 单点验证(见 spec「验证配方」),
对照组 = A 路线同构 `disc_mult02_s41`:
```bash
USEANET_HETERO_EXPERTS=1 USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 \
USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  conda run -n ubench1 python main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name hetero_s41
```

---

## Self-Review

**Spec coverage:** 6 异构专家结构(Task 2 `_hetero_kernels`/`_HeteroExpert`)✓;可学物理初始核(Task 1)✓;
增容 channel=48 + SE + Dropout2d(0.1)(Task 2)✓;env 门控默认关(Task 3)✓;I/O 契约不变(Task 2/5 shape 测)✓;
参数预算 Δ≈+130k(Task 4 预算测 + Task 6 整模型测)✓;测试 6 项(spec §测试:形状/初始化/派发等价/端到端/梯度流/预算)
全覆盖 ✓;不碰 router/proxy/losses/优化器 ✓。

**Placeholder scan:** 无 TBD/TODO;每个代码步给出完整代码;Task 4 Step 3 的「收容量」是明确条件分支非占位。

**Type consistency:** `_LearnablePrefilter(kernel, dilation)`、`_HeteroExpert(in_channel, out_channel, prefilters, channel, use_se)`、
`_build_hetero_experts(in_channel, out_channel, channel)`、`build_experts(in_channel, out_channel, channel=32)` 全程签名一致;
`EXPERT_NAMES` 顺序在 `_build_hetero_experts` 与 `_hetero_kernels` 保持一致(router gate 索引依赖)。
