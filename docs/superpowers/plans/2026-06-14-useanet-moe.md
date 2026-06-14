# USEANet-MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade USEANet's weak dense soft-MoE into a sparse *ultrasound-physics degradation MoE* (Pillar 1), and build a differentiable ultrasound degradation simulator whose standalone fidelity is hard-gated before it is allowed to feed the model (Pillar 2).

**Architecture:** Two **decoupled** tracks. **Track A (Pillar 2)** builds a standalone differentiable simulator + a fidelity validation harness that emits a PASS/FAIL gate report; nothing in the model imports the simulator until the gate passes. **Track B (Pillar 1)** replaces `MultiBranchFeatureProcessor` at the x3/x4 layers with a `PhysicsMoE` (6 anchored physical experts + degradation-aware top-2 spatial router) whose router supervision uses **label-free proxy maps** by default — so Track B does **not** depend on the simulator passing. Track C is a thin, gate-conditional integration that upgrades router supervision from proxy maps to the simulator's physical ground truth, only after Track A's gate is green.

**Tech Stack:** Python 3.x, PyTorch 2.7 (env `ubench1`), numpy, scipy, PyWavelets (`pywt`), OpenCV (`cv2`), pytest. Existing repo: PVT-B0 backbone, `main.py` training loop with a `deep_supervision_loss` hook, BUSI-style `MedicalDataSets` dataloader.

---

## Conventions (read once before starting)

- **Python interpreter / test runner:** all commands use the `ubench1` conda env. Run tests as:
  `conda run -n ubench1 python -m pytest <path> -v`
- **First task only:** install pytest into the env (Task 0).
- **Fixed expert order (used everywhere — never reorder):**
  `EXPERT_NAMES = ["despeckle", "edge", "shadow", "posterior", "contrast", "hf"]` (index 0..5).
- **Tensor conventions:** images/feature maps are `[B, C, H, W]`, float32. Grayscale ultrasound for the simulator is `[B, 1, H, W]` in `[0, 1]`.
- **Determinism in tests:** call `torch.manual_seed(0)` at the top of every test that constructs modules or random tensors.
- **Commit discipline:** one commit per task (the final step of each task). Conventional Commits, no attribution footer (repo convention).

## File Structure

**Track A — Pillar 2 (simulator + hard gate), no model imports:**
- Create `models/Hybrid/USEANet/simulator/__init__.py` — exports `UltrasoundDegradationSimulator`, `SimOutput`.
- Create `models/Hybrid/USEANet/simulator/degradations.py` — differentiable ops, one per degradation type, each returns `(degraded_image, degradation_map)`.
- Create `models/Hybrid/USEANet/simulator/simulator.py` — `UltrasoundDegradationSimulator` composing the ops; `SimOutput` dataclass.
- Create `tools/sim_fidelity/__init__.py` — empty package marker.
- Create `tools/sim_fidelity/metrics.py` — intensity-histogram Wasserstein, speckle-SNR statistic, radial power-spectrum distance.
- Create `tools/sim_fidelity/generate_samples.py` — CLI: load real US images, synthesize degradations, dump side-by-side PNGs + a `.npz` of metric inputs.
- Create `tools/sim_fidelity/gate.py` — CLI: aggregate metrics vs thresholds → `gate_report.json` with `PASS`/`FAIL`.
- Tests: `tests/simulator/test_degradations.py`, `tests/simulator/test_simulator.py`, `tests/sim_fidelity/test_metrics.py`, `tests/sim_fidelity/test_gate.py`.

**Track B — Pillar 1 (physics MoE), decoupled from simulator:**
- Create `models/Hybrid/USEANet/moe/__init__.py` — exports `PhysicsMoE`, `EXPERT_NAMES`, loss/metric fns.
- Create `models/Hybrid/USEANet/moe/proxy.py` — `degradation_proxies(feat) -> [B,6,H,W]` label-free cues.
- Create `models/Hybrid/USEANet/moe/experts.py` — 6 anchored experts + `build_experts`.
- Create `models/Hybrid/USEANet/moe/router.py` — `DegradationAwareRouter` (per-position top-2).
- Create `models/Hybrid/USEANet/moe/physics_moe.py` — `PhysicsMoE` drop-in for `MultiBranchFeatureProcessor`.
- Create `models/Hybrid/USEANet/moe/losses.py` — `router_supervision_loss`, `load_balance_loss`, `effective_experts`.
- Modify `models/Hybrid/USEANet/usea_core.py` — x3/x4 use `PhysicsMoE`; x2 keeps `MultiBranchFeatureProcessor`; expose MoE submodules.
- Modify `models/Hybrid/USEANet/__init__.py` — aggregate MoE aux losses into `deep_supervision_loss`; add `set_training_progress`.
- Modify `main.py:321-329` — call `set_training_progress` when present (one guarded line).
- Tests: `tests/moe/test_proxy.py`, `tests/moe/test_experts.py`, `tests/moe/test_router.py`, `tests/moe/test_physics_moe.py`, `tests/moe/test_losses.py`, `tests/moe/test_core_wiring.py`, `tests/moe/test_adapter_integration.py`.

**Track C — gate-conditional simulator→router supervision (only after Track A PASS):**
- Modify `models/Hybrid/USEANet/moe/physics_moe.py` + adapter to accept an optional physical-GT degradation map for router supervision.

---

# Track A — Pillar 2: Differentiable Simulator + Hard Gate

> Build and validate the simulator **in isolation**. Success criterion is the gate report. Track B may proceed in parallel; it never imports anything from `simulator/`.

### Task 0: Test tooling

**Files:**
- None (environment only).

- [ ] **Step 1: Install pytest into the training env**

Run: `conda run -n ubench1 python -m pip install pytest`
Expected: `Successfully installed pytest-...` (or "already satisfied").

- [ ] **Step 2: Create the tests package roots**

```bash
mkdir -p tests/simulator tests/sim_fidelity tests/moe
touch tests/__init__.py tests/simulator/__init__.py tests/sim_fidelity/__init__.py tests/moe/__init__.py
```

- [ ] **Step 3: Verify pytest runs (collects zero tests cleanly)**

Run: `conda run -n ubench1 python -m pytest tests/ -q`
Expected: `no tests ran` (exit code 5 is fine) — confirms collection works.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/simulator/__init__.py tests/sim_fidelity/__init__.py tests/moe/__init__.py
git commit -m "chore: add pytest test scaffolding"
```

---

### Task A1: Differentiable degradation ops — speckle & log compression

**Files:**
- Create: `models/Hybrid/USEANet/simulator/__init__.py`
- Create: `models/Hybrid/USEANet/simulator/degradations.py`
- Test: `tests/simulator/test_degradations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulator/test_degradations.py
import torch
from models.Hybrid.USEANet.simulator.degradations import (
    log_compression, add_speckle,
)


def _img():
    torch.manual_seed(0)
    return torch.rand(2, 1, 64, 64)


def test_log_compression_keeps_shape_and_range():
    x = _img()
    out = log_compression(x, dynamic_range_db=50.0)
    assert out.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_log_compression_is_differentiable():
    x = _img().requires_grad_(True)
    log_compression(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_add_speckle_returns_image_and_map():
    x = _img()
    out, dmap = add_speckle(x, sigma=0.3, generator=torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    assert dmap.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_add_speckle_increases_variance_in_flat_region():
    # Multiplicative speckle on a flat patch must raise its local variance.
    flat = torch.full((1, 1, 64, 64), 0.5)
    out, _ = add_speckle(flat, sigma=0.4, generator=torch.Generator().manual_seed(2))
    assert out.var() > flat.var() + 1e-4


def test_add_speckle_is_differentiable_wrt_input():
    x = _img().requires_grad_(True)
    out, _ = add_speckle(x, sigma=0.3, generator=torch.Generator().manual_seed(3))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/simulator/test_degradations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.Hybrid.USEANet.simulator'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/simulator/__init__.py
from .simulator import UltrasoundDegradationSimulator, SimOutput

__all__ = ["UltrasoundDegradationSimulator", "SimOutput"]
```

```python
# models/Hybrid/USEANet/simulator/degradations.py
"""Differentiable ultrasound image-formation degradation ops.

Each op takes a grayscale image ``[B, 1, H, W]`` in ``[0, 1]`` and returns
``(degraded_image, degradation_map)`` where ``degradation_map`` (also
``[B, 1, H, W]``, in ``[0, 1]``) marks *where/how strongly* that degradation
was applied. The maps double as physical ground truth for router supervision
(Track C). All ops are autograd-differentiable w.r.t. the input image.
"""
import torch
import torch.nn.functional as F


def log_compression(img, dynamic_range_db=50.0):
    """Mimic the log compression of the ultrasound scan-conversion pipeline."""
    eps = 1e-6
    floor = 10.0 ** (-dynamic_range_db / 20.0)
    x = torch.clamp(img, floor, 1.0)
    out = (20.0 * torch.log10(x + eps) + dynamic_range_db) / dynamic_range_db
    return torch.clamp(out, 0.0, 1.0)


def add_speckle(img, sigma=0.3, distribution="rayleigh", generator=None):
    """Multiplicative speckle (Rayleigh-like fully-developed speckle).

    Returns ``(degraded, speckle_map)`` where ``speckle_map`` is the normalised
    magnitude of the multiplicative perturbation (high where speckle is strong).
    """
    if distribution == "rayleigh":
        # Rayleigh magnitude = sqrt(n1^2 + n2^2) of two N(0, sigma) draws.
        n1 = torch.empty_like(img).normal_(0.0, sigma, generator=generator)
        n2 = torch.empty_like(img).normal_(0.0, sigma, generator=generator)
        mult = torch.sqrt(n1 * n1 + n2 * n2)
        mult = mult / (sigma * (3.14159265 / 2.0) ** 0.5 + 1e-6)  # mean ~= 1
    else:
        raise ValueError(f"unknown speckle distribution: {distribution}")
    out = torch.clamp(img * mult, 0.0, 1.0)
    speckle_map = torch.clamp((mult - 1.0).abs(), 0.0, 1.0)
    return out, speckle_map
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/simulator/test_degradations.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/simulator/__init__.py models/Hybrid/USEANet/simulator/degradations.py tests/simulator/test_degradations.py
git commit -m "feat(sim): differentiable log-compression and speckle ops"
```

---

### Task A2: Differentiable degradation ops — depth attenuation, acoustic shadow, posterior enhancement

**Files:**
- Modify: `models/Hybrid/USEANet/simulator/degradations.py`
- Test: `tests/simulator/test_degradations.py` (append)

- [ ] **Step 1: Write the failing test (append to file)**

```python
# tests/simulator/test_degradations.py  (append)
from models.Hybrid.USEANet.simulator.degradations import (
    depth_attenuation, acoustic_shadow, posterior_enhancement,
)


def test_depth_attenuation_darkens_with_depth():
    x = torch.full((1, 1, 64, 64), 0.8)
    out, amap = depth_attenuation(x, coeff=1.0)
    assert out.shape == x.shape and amap.shape == x.shape
    # Deeper rows (larger H index) must be darker than shallow rows.
    assert out[..., -1, :].mean() < out[..., 0, :].mean()
    assert amap[..., -1, :].mean() > amap[..., 0, :].mean()


def test_acoustic_shadow_darkens_below_column_band():
    x = torch.full((1, 1, 64, 64), 0.7)
    # Shadow under columns 20..30 starting at row 30.
    out, smap = acoustic_shadow(x, col_start=20, col_end=30, row_start=30, strength=0.8)
    shadowed = out[..., 40:, 20:30].mean()
    clear = out[..., 40:, 40:50].mean()
    assert shadowed < clear
    assert smap[..., 40:, 20:30].mean() > smap[..., 40:, 40:50].mean()


def test_posterior_enhancement_brightens_below_band():
    x = torch.full((1, 1, 64, 64), 0.4)
    out, emap = posterior_enhancement(x, col_start=20, col_end=30, row_start=30, strength=0.6)
    enhanced = out[..., 40:, 20:30].mean()
    clear = out[..., 40:, 40:50].mean()
    assert enhanced > clear
    assert emap[..., 40:, 20:30].mean() > emap[..., 40:, 40:50].mean()


def test_new_ops_are_differentiable():
    for op in (depth_attenuation, acoustic_shadow, posterior_enhancement):
        x = torch.rand(1, 1, 32, 32, requires_grad=True)
        out, _ = op(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/simulator/test_degradations.py -v -k "depth or shadow or posterior or new_ops"`
Expected: FAIL — `ImportError: cannot import name 'depth_attenuation'`.

- [ ] **Step 3: Write minimal implementation (append to `degradations.py`)**

```python
# models/Hybrid/USEANet/simulator/degradations.py  (append)

def _depth_ramp(img):
    """Row index normalised to [0, 1] along the depth (H) axis, shaped [1,1,H,1]."""
    h = img.shape[-2]
    ramp = torch.linspace(0.0, 1.0, h, device=img.device, dtype=img.dtype)
    return ramp.view(1, 1, h, 1)


def depth_attenuation(img, coeff=0.5):
    """Exponential brightness falloff with depth (deeper rows are dimmer)."""
    ramp = _depth_ramp(img)
    atten = torch.exp(-coeff * ramp)            # [1,1,H,1], 1 at top -> small at bottom
    out = torch.clamp(img * atten, 0.0, 1.0)
    atten_map = (1.0 - atten).expand_as(img)     # strong (->1) where most attenuated
    return out, atten_map


def _vertical_band(img, col_start, col_end, row_start):
    """Soft mask: 1 inside columns [col_start,col_end) for rows >= row_start."""
    b, _, h, w = img.shape
    cols = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, 1, w)
    rows = torch.arange(h, device=img.device, dtype=img.dtype).view(1, 1, h, 1)
    col_mask = ((cols >= col_start) & (cols < col_end)).to(img.dtype)
    row_mask = (rows >= row_start).to(img.dtype)
    return (col_mask * row_mask).expand(b, 1, h, w)


def acoustic_shadow(img, col_start=None, col_end=None, row_start=None, strength=0.6):
    """Darken a vertical column band below a (highly attenuating) structure."""
    _, _, h, w = img.shape
    col_start = w // 3 if col_start is None else col_start
    col_end = 2 * w // 3 if col_end is None else col_end
    row_start = h // 2 if row_start is None else row_start
    band = _vertical_band(img, col_start, col_end, row_start)
    out = torch.clamp(img * (1.0 - strength * band), 0.0, 1.0)
    return out, band * strength


def posterior_enhancement(img, col_start=None, col_end=None, row_start=None, strength=0.4):
    """Brighten a vertical column band below an (anechoic) structure."""
    _, _, h, w = img.shape
    col_start = w // 3 if col_start is None else col_start
    col_end = 2 * w // 3 if col_end is None else col_end
    row_start = h // 2 if row_start is None else row_start
    band = _vertical_band(img, col_start, col_end, row_start)
    out = torch.clamp(img + strength * band * (1.0 - img), 0.0, 1.0)
    return out, band * strength
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/simulator/test_degradations.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/simulator/degradations.py tests/simulator/test_degradations.py
git commit -m "feat(sim): depth attenuation, acoustic shadow, posterior enhancement ops"
```

---

### Task A3: Compose ops into `UltrasoundDegradationSimulator`

**Files:**
- Create: `models/Hybrid/USEANet/simulator/simulator.py`
- Test: `tests/simulator/test_simulator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulator/test_simulator.py
import torch
from models.Hybrid.USEANet.simulator import UltrasoundDegradationSimulator, SimOutput

EXPECTED_MAPS = {"speckle", "attenuation", "shadow", "posterior"}


def _sim():
    return UltrasoundDegradationSimulator(seed=0)


def test_forward_returns_simoutput_with_maps():
    x = torch.rand(2, 1, 64, 64)
    out = _sim()(x, intensity=1.0)
    assert isinstance(out, SimOutput)
    assert out.image.shape == x.shape
    assert out.image.min() >= 0.0 and out.image.max() <= 1.0
    assert EXPECTED_MAPS.issubset(set(out.degradation_maps.keys()))
    for m in out.degradation_maps.values():
        assert m.shape == x.shape


def test_intensity_zero_is_near_identity():
    x = torch.rand(1, 1, 64, 64)
    out = _sim()(x, intensity=0.0)
    # No degradation strength -> output close to (log-compressed) input, maps ~0.
    assert torch.allclose(out.degradation_maps["shadow"], torch.zeros_like(x), atol=1e-6)
    assert torch.allclose(out.degradation_maps["posterior"], torch.zeros_like(x), atol=1e-6)


def test_higher_intensity_increases_total_degradation():
    x = torch.rand(1, 1, 64, 64)
    lo = _sim()(x, intensity=0.2)
    hi = _sim()(x, intensity=1.0)
    lo_mag = sum(m.mean() for m in lo.degradation_maps.values())
    hi_mag = sum(m.mean() for m in hi.degradation_maps.values())
    assert hi_mag > lo_mag


def test_simulator_is_differentiable():
    x = torch.rand(1, 1, 32, 32, requires_grad=True)
    _sim()(x, intensity=1.0).image.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/simulator/test_simulator.py -v`
Expected: FAIL — `ImportError: cannot import name 'SimOutput'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/simulator/simulator.py
"""Composable differentiable ultrasound degradation simulator.

Used three ways (see design doc §4): online augmentation, physical GT for
router supervision (Track C), and controllable degradation experiments. This
module has NO dependency on the segmentation model and must never import it.
"""
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from .degradations import (
    log_compression, add_speckle, depth_attenuation,
    acoustic_shadow, posterior_enhancement,
)


@dataclass
class SimOutput:
    image: torch.Tensor
    degradation_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    applied: Dict[str, float] = field(default_factory=dict)


class UltrasoundDegradationSimulator(nn.Module):
    """Apply log-compression + speckle + depth attenuation + (random) shadow /
    posterior-enhancement bands, scaled by ``intensity`` in ``[0, 1]``.
    """

    def __init__(self, seed=None, speckle_sigma=0.3, atten_coeff=0.6,
                 shadow_strength=0.7, posterior_strength=0.5, dynamic_range_db=50.0):
        super().__init__()
        self.speckle_sigma = speckle_sigma
        self.atten_coeff = atten_coeff
        self.shadow_strength = shadow_strength
        self.posterior_strength = posterior_strength
        self.dynamic_range_db = dynamic_range_db
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def _rand_band(self, w):
        # Random column band ~ one third of the width.
        width = max(1, w // 3)
        start = int(torch.randint(0, max(1, w - width), (1,), generator=self._gen).item())
        return start, start + width

    def forward(self, img, intensity=1.0):
        if img.shape[1] != 1:
            img = img.mean(dim=1, keepdim=True)
        h, w = img.shape[-2:]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=self._gen).item()))

        x = log_compression(img, self.dynamic_range_db)

        x, speckle_map = add_speckle(
            x, sigma=self.speckle_sigma * intensity, generator=gen)

        x, atten_map = depth_attenuation(x, coeff=self.atten_coeff * intensity)

        cs, ce = self._rand_band(w)
        x, shadow_map = acoustic_shadow(
            x, col_start=cs, col_end=ce, row_start=h // 2,
            strength=self.shadow_strength * intensity)

        cs2, ce2 = self._rand_band(w)
        x, posterior_map = posterior_enhancement(
            x, col_start=cs2, col_end=ce2, row_start=h // 2,
            strength=self.posterior_strength * intensity)

        return SimOutput(
            image=x,
            degradation_maps={
                "speckle": speckle_map,
                "attenuation": atten_map,
                "shadow": shadow_map,
                "posterior": posterior_map,
            },
            applied={
                "speckle": self.speckle_sigma * intensity,
                "attenuation": self.atten_coeff * intensity,
                "shadow": self.shadow_strength * intensity,
                "posterior": self.posterior_strength * intensity,
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/simulator/test_simulator.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/simulator/simulator.py tests/simulator/test_simulator.py
git commit -m "feat(sim): compose UltrasoundDegradationSimulator with SimOutput"
```

---

### Task A4: Fidelity metrics (intensity Wasserstein, speckle SNR, radial spectrum)

**Files:**
- Create: `tools/sim_fidelity/__init__.py`
- Create: `tools/sim_fidelity/metrics.py`
- Test: `tests/sim_fidelity/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim_fidelity/test_metrics.py
import numpy as np
from tools.sim_fidelity.metrics import (
    intensity_wasserstein, speckle_snr, radial_spectrum_distance,
)


def test_intensity_wasserstein_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.random((4, 64, 64)).astype("float32")
    assert intensity_wasserstein(a, a) < 1e-6


def test_intensity_wasserstein_positive_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.random((4, 64, 64)).astype("float32") * 0.3
    b = a + 0.4
    assert intensity_wasserstein(a, b) > 0.2


def test_speckle_snr_higher_for_smoother_image():
    flat = np.full((2, 64, 64), 0.5, dtype="float32")
    rng = np.random.default_rng(1)
    noisy = np.clip(flat + rng.normal(0, 0.2, flat.shape), 0, 1).astype("float32")
    assert speckle_snr(flat) > speckle_snr(noisy)


def test_radial_spectrum_distance_zero_for_identical():
    rng = np.random.default_rng(2)
    a = rng.random((3, 64, 64)).astype("float32")
    assert radial_spectrum_distance(a, a) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/sim_fidelity/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.sim_fidelity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/sim_fidelity/__init__.py
```

```python
# tools/sim_fidelity/metrics.py
"""Fidelity metrics comparing synthetic vs real ultrasound batches.

All inputs are float32 numpy arrays ``[N, H, W]`` in ``[0, 1]``.
"""
import numpy as np
from scipy.stats import wasserstein_distance


def intensity_wasserstein(real, synth, bins=256):
    """Wasserstein-1 distance between flattened intensity distributions."""
    r = np.clip(real.reshape(-1), 0.0, 1.0)
    s = np.clip(synth.reshape(-1), 0.0, 1.0)
    return float(wasserstein_distance(r, s))


def speckle_snr(images, patch=16):
    """Median local SNR (mean/std) over non-overlapping patches.

    Fully-developed Rayleigh speckle has a characteristic SNR ~ 1.91; this
    statistic lets the gate compare synthetic speckle texture against real.
    """
    snrs = []
    for img in images:
        h, w = img.shape
        for i in range(0, h - patch + 1, patch):
            for j in range(0, w - patch + 1, patch):
                p = img[i:i + patch, j:j + patch]
                mu, sd = float(p.mean()), float(p.std())
                if sd > 1e-6:
                    snrs.append(mu / sd)
    return float(np.median(snrs)) if snrs else 0.0


def _radial_profile(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def radial_spectrum_distance(real, synth):
    """L1 distance between mean (log) radially-averaged power spectra."""
    def mean_profile(batch):
        profs = [np.log1p(_radial_profile(im)) for im in batch]
        n = min(len(p) for p in profs)
        return np.mean([p[:n] for p in profs], axis=0)
    pr = mean_profile(real)
    ps = mean_profile(synth)
    n = min(len(pr), len(ps))
    denom = np.abs(pr[:n]).mean() + 1e-6
    return float(np.abs(pr[:n] - ps[:n]).mean() / denom)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/sim_fidelity/test_metrics.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/sim_fidelity/__init__.py tools/sim_fidelity/metrics.py tests/sim_fidelity/test_metrics.py
git commit -m "feat(sim-fidelity): intensity Wasserstein, speckle SNR, radial spectrum metrics"
```

---

### Task A5: Sample generator CLI

**Files:**
- Create: `tools/sim_fidelity/generate_samples.py`
- Test: `tests/sim_fidelity/test_generate_samples.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim_fidelity/test_generate_samples.py
import numpy as np
from tools.sim_fidelity.generate_samples import synthesize_batch


def test_synthesize_batch_shapes_and_range():
    rng = np.random.default_rng(0)
    real = rng.random((5, 1, 64, 64)).astype("float32")
    synth, maps = synthesize_batch(real, intensity=1.0, seed=0)
    assert synth.shape == (5, 64, 64)
    assert synth.min() >= 0.0 and synth.max() <= 1.0
    assert "shadow" in maps and maps["shadow"].shape == (5, 64, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/sim_fidelity/test_generate_samples.py -v`
Expected: FAIL — `ImportError: cannot import name 'synthesize_batch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/sim_fidelity/generate_samples.py
"""Generate synthetic-degraded ultrasound samples for fidelity inspection.

CLI dumps side-by-side PNGs (real | synthetic) and a .npz holding the real and
synthetic batches consumed by ``gate.py``. Standalone — imports only the
simulator package.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from models.Hybrid.USEANet.simulator import UltrasoundDegradationSimulator


def synthesize_batch(real, intensity=1.0, seed=0):
    """real: [N,1,H,W] float32 in [0,1] -> (synth [N,H,W], maps {name:[N,H,W]})."""
    sim = UltrasoundDegradationSimulator(seed=seed)
    out = sim(torch.from_numpy(real).float(), intensity=intensity)
    synth = out.image.squeeze(1).detach().cpu().numpy()
    maps = {k: v.squeeze(1).detach().cpu().numpy() for k, v in out.degradation_maps.items()}
    return synth, maps


def _load_images(images_dir, limit, size):
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))[:limit]
    imgs = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (size, size)).astype("float32") / 255.0
        imgs.append(g[None])  # [1,H,W]
    return np.stack(imgs, axis=0)  # [N,1,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="dir of real US PNGs (e.g. BUSI/images)")
    ap.add_argument("--out_dir", default="output/sim_fidelity")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    real = _load_images(args.images_dir, args.limit, args.size)
    synth, _ = synthesize_batch(real, intensity=args.intensity, seed=args.seed)

    for i in range(min(16, real.shape[0])):
        pair = np.concatenate([real[i, 0], synth[i]], axis=1)
        cv2.imwrite(os.path.join(args.out_dir, f"pair_{i:03d}.png"), (pair * 255).astype("uint8"))

    np.savez(os.path.join(args.out_dir, "batches.npz"),
             real=real[:, 0], synth=synth)
    print(f"wrote {min(16, real.shape[0])} pairs + batches.npz to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/sim_fidelity/test_generate_samples.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/sim_fidelity/generate_samples.py tests/sim_fidelity/test_generate_samples.py
git commit -m "feat(sim-fidelity): sample generator CLI"
```

---

### Task A6: Hard-gate report (fidelity thresholds + decision)

> Encodes the §4 hard gate. The downstream-Dice-gain criterion is run separately as a normal training experiment (see Step 6 notes); this task implements the **fidelity** half of the gate and the decision aggregation so the report is reproducible.

**Files:**
- Create: `tools/sim_fidelity/gate.py`
- Test: `tests/sim_fidelity/test_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim_fidelity/test_gate.py
from tools.sim_fidelity.gate import evaluate_gate, DEFAULT_THRESHOLDS


def test_pass_when_all_within_thresholds():
    metrics = {"intensity_w1": 0.05, "speckle_snr_delta": 0.1,
               "radial_spectrum_distance": 0.1, "downstream_dice_gain": 0.6}
    report = evaluate_gate(metrics)
    assert report["fidelity_pass"] is True
    assert report["decision"] == "PROMOTE"


def test_fidelity_fail_demotes():
    metrics = {"intensity_w1": 0.5, "speckle_snr_delta": 1.2,
               "radial_spectrum_distance": 0.9, "downstream_dice_gain": 0.6}
    report = evaluate_gate(metrics)
    assert report["fidelity_pass"] is False
    assert report["decision"] == "DEMOTE_OR_DROP"


def test_fidelity_ok_but_harmful_downstream_demotes():
    metrics = {"intensity_w1": 0.05, "speckle_snr_delta": 0.1,
               "radial_spectrum_distance": 0.1, "downstream_dice_gain": -0.3}
    report = evaluate_gate(metrics)
    assert report["fidelity_pass"] is True
    assert report["decision"] == "DEMOTE_OR_DROP"


def test_defaults_present():
    for k in ("intensity_w1", "speckle_snr_delta", "radial_spectrum_distance",
              "min_downstream_dice_gain"):
        assert k in DEFAULT_THRESHOLDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/sim_fidelity/test_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_gate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/sim_fidelity/gate.py
"""Hard-gate decision for the ultrasound degradation simulator (design §4).

Fidelity is PASS iff intensity Wasserstein, speckle-SNR delta, and radial
spectrum distance are all within thresholds. The simulator is PROMOTEd to
Pillar 2 only if fidelity passes AND it is non-harmful downstream
(downstream_dice_gain >= min). Otherwise DEMOTE_OR_DROP (use as plain aug or
drop); Pillar 1 carries the paper regardless.
"""
import argparse
import json

import numpy as np

from tools.sim_fidelity.metrics import (
    intensity_wasserstein, speckle_snr, radial_spectrum_distance,
)

# Pinned thresholds (design "仍待细化" item resolved here).
DEFAULT_THRESHOLDS = {
    "intensity_w1": 0.10,            # <= : intensity histograms close
    "speckle_snr_delta": 0.30,       # <= : |synth SNR - real SNR| small
    "radial_spectrum_distance": 0.30,  # <= : frequency content close
    "min_downstream_dice_gain": 0.0,   # >= : sim aug must not hurt (target +0.5)
}


def evaluate_gate(metrics, thresholds=None):
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    fidelity_pass = (
        metrics["intensity_w1"] <= t["intensity_w1"]
        and metrics["speckle_snr_delta"] <= t["speckle_snr_delta"]
        and metrics["radial_spectrum_distance"] <= t["radial_spectrum_distance"]
    )
    downstream_ok = metrics.get("downstream_dice_gain", 0.0) >= t["min_downstream_dice_gain"]
    decision = "PROMOTE" if (fidelity_pass and downstream_ok) else "DEMOTE_OR_DROP"
    return {
        "metrics": metrics,
        "thresholds": t,
        "fidelity_pass": bool(fidelity_pass),
        "downstream_ok": bool(downstream_ok),
        "decision": decision,
    }


def compute_fidelity_metrics(npz_path):
    data = np.load(npz_path)
    real, synth = data["real"], data["synth"]
    return {
        "intensity_w1": intensity_wasserstein(real, synth),
        "speckle_snr_delta": abs(speckle_snr(synth) - speckle_snr(real)),
        "radial_spectrum_distance": radial_spectrum_distance(real, synth),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="batches.npz from generate_samples.py")
    ap.add_argument("--downstream_dice_gain", type=float, default=0.0,
                    help="absolute Dice gain (sim-aug minus baseline) from the training experiment")
    ap.add_argument("--out", default="output/sim_fidelity/gate_report.json")
    args = ap.parse_args()

    metrics = compute_fidelity_metrics(args.npz)
    metrics["downstream_dice_gain"] = args.downstream_dice_gain
    report = evaluate_gate(metrics)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nGATE DECISION: {report['decision']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/sim_fidelity/test_gate.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/sim_fidelity/gate.py tests/sim_fidelity/test_gate.py
git commit -m "feat(sim-fidelity): hard-gate decision report"
```

- [ ] **Step 6: Run the gate end-to-end (manual, produces the decision artifact)**

Fidelity half (fast):
```bash
conda run -n ubench1 python -m tools.sim_fidelity.generate_samples \
  --images_dir <path-to-busi>/images --out_dir output/sim_fidelity --limit 64 --size 256
conda run -n ubench1 python -m tools.sim_fidelity.gate \
  --npz output/sim_fidelity/batches.npz --downstream_dice_gain 0.0
```
Downstream half (the decision experiment): train USEANet (id 115) on BUSI **with** vs **without** simulator online augmentation; record `val Dice` of each; pass the difference (sim − baseline) via `--downstream_dice_gain`. Re-run the gate to refresh `gate_report.json`.

**HARD GATE:** If `decision == "PROMOTE"`, Track C is unlocked. If `DEMOTE_OR_DROP`, **do not** start Track C — the simulator stays as optional plain augmentation; Pillar 1 (Track B) is the paper's core and is unaffected.

---

# Track B — Pillar 1: Ultrasound-Physics MoE (decoupled; parallelizable with Track A)

> None of these tasks import `simulator/`. Router supervision uses label-free proxy maps. Default expert count = 6 (design §3.1, §10).

### Task B1: Label-free degradation proxy maps

**Files:**
- Create: `models/Hybrid/USEANet/moe/__init__.py`
- Create: `models/Hybrid/USEANet/moe/proxy.py`
- Test: `tests/moe/test_proxy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_proxy.py
import torch
from models.Hybrid.USEANet.moe import EXPERT_NAMES
from models.Hybrid.USEANet.moe.proxy import degradation_proxies


def test_proxy_shape_matches_experts():
    torch.manual_seed(0)
    feat = torch.rand(2, 16, 16, 16)
    proxy = degradation_proxies(feat)
    assert proxy.shape == (2, len(EXPERT_NAMES), 16, 16)


def test_proxy_is_nonnegative_and_normalised_per_position():
    torch.manual_seed(0)
    feat = torch.rand(2, 16, 16, 16)
    proxy = degradation_proxies(feat)
    assert (proxy >= 0).all()
    s = proxy.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_edge_proxy_high_on_vertical_edge():
    # Build a feature with a sharp vertical edge -> edge proxy (idx 1) should be
    # the dominant channel at the edge column.
    feat = torch.zeros(1, 4, 16, 16)
    feat[..., :, 8:] = 1.0
    proxy = degradation_proxies(feat)
    edge_idx = EXPERT_NAMES.index("edge")
    at_edge = proxy[0, :, 8, 7]
    assert at_edge.argmax().item() == edge_idx


def test_proxy_is_differentiable():
    feat = torch.rand(1, 8, 16, 16, requires_grad=True)
    degradation_proxies(feat).sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_proxy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.Hybrid.USEANet.moe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/moe/__init__.py
"""Ultrasound-physics MoE for USEANet (design Pillar 1). No simulator imports."""

EXPERT_NAMES = ["despeckle", "edge", "shadow", "posterior", "contrast", "hf"]

# physics_moe / losses are added in Tasks B2-B5; keep the package importable
# while only proxy.py exists (proxy.py does `from . import EXPERT_NAMES`).
try:
    from .physics_moe import PhysicsMoE
    from .losses import router_supervision_loss, load_balance_loss, effective_experts
except ImportError:
    pass

__all__ = [
    "EXPERT_NAMES", "PhysicsMoE",
    "router_supervision_loss", "load_balance_loss", "effective_experts",
]
```

```python
# models/Hybrid/USEANet/moe/proxy.py
"""Label-free degradation proxy maps, one channel per expert (design §3.1).

Computed from the channel-mean of a feature map (or grayscale image). Each
channel is a cheap cue for *where* that expert's degradation is present; the
stack is softmax-normalised across experts per position so it can directly
supervise the router gate.
"""
import torch
import torch.nn.functional as F

from . import EXPERT_NAMES


def _local_variance(g, k=5):
    pad = k // 2
    mean = F.avg_pool2d(g, k, 1, pad)
    mean_sq = F.avg_pool2d(g * g, k, 1, pad)
    return torch.clamp(mean_sq - mean * mean, min=0.0)


def _gradient_magnitude(g):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _vertical_attenuation(g):
    # Brightness deficit relative to the shallow (top) rows of each column.
    top = g[..., :max(1, g.shape[-2] // 8), :].mean(dim=-2, keepdim=True)
    return torch.clamp(top - g, min=0.0)


def _posterior_brightness(g):
    # Local excess brightness vs a large neighbourhood mean (posterior enhance).
    bg = F.avg_pool2d(g, 9, 1, 4)
    return torch.clamp(g - bg, min=0.0)


def _low_freq_energy(g):
    return F.avg_pool2d(g, 9, 1, 4)


def _high_freq_energy(g):
    return torch.clamp(g - F.avg_pool2d(g, 5, 1, 2), min=0.0).abs()


def degradation_proxies(feat, eps=1e-6):
    """feat: [B,C,H,W] -> proxy [B, len(EXPERT_NAMES), H, W], softmax over experts."""
    g = feat.mean(dim=1, keepdim=True)  # [B,1,H,W]
    cues = {
        "despeckle": _local_variance(g),
        "edge": _gradient_magnitude(g),
        "shadow": _vertical_attenuation(g),
        "posterior": _posterior_brightness(g),
        "contrast": _low_freq_energy(g),
        "hf": _high_freq_energy(g),
    }
    stack = torch.cat([cues[name] for name in EXPERT_NAMES], dim=1)  # [B,6,H,W]
    # Per-channel min-max to comparable scale, then softmax across experts.
    flat = stack.flatten(2)
    mn = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    mx = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    norm = (stack - mn) / (mx - mn + eps)
    return F.softmax(norm, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_proxy.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/__init__.py models/Hybrid/USEANet/moe/proxy.py tests/moe/test_proxy.py
git commit -m "feat(moe): label-free degradation proxy maps"
```

---

### Task B2: Anchored physical experts

**Files:**
- Create: `models/Hybrid/USEANet/moe/experts.py`
- Test: `tests/moe/test_experts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_experts.py
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
    # Anchored small experts: each well under 30k params (fixed kernels + few ch).
    experts = build_experts(in_channel=16, out_channel=32)
    for e in experts:
        n = sum(p.numel() for p in e.parameters() if p.requires_grad)
        assert n < 30000, f"expert too heavy: {n}"


def test_experts_differentiable():
    feat = torch.rand(1, 16, 16, 16, requires_grad=True)
    out = build_experts(16, 32)[0](feat)
    out.sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_experts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.Hybrid.USEANet.moe.experts'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/moe/experts.py
"""Anchored ultrasound-physics experts (design §3.1).

Each expert = a fixed (non-learnable) physical pre-filter + a small learnable
1x1->depthwise->1x1 head (channel anchored, few params). Fixed kernels keep
experts physically specialised and resist overfitting on small data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EXPERT_NAMES


class _AnchoredExpert(nn.Module):
    """fixed depthwise prefilter (broadcast over channels) + small learnable head."""

    def __init__(self, in_channel, out_channel, kernel, channel=32):
        super().__init__()
        self.register_buffer("kernel", kernel.view(1, 1, *kernel.shape))
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, out_channel, 1, bias=False),
        )

    def _prefilter(self, x):
        c = x.shape[1]
        k = self.kernel.expand(c, 1, -1, -1)
        return F.conv2d(x, k, padding=self.kernel.shape[-1] // 2, groups=c)

    def forward(self, x):
        return self.head(self._prefilter(x))


# Fixed physical kernels (3x3).
def _kernels():
    box = torch.ones(3, 3) / 9.0                                   # despeckle (low-pass)
    lap = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)  # edge
    vgrad = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)     # shadow (vertical atten.)
    vsum = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 2, 1]], dtype=torch.float32) / 4.0   # posterior (below-structure)
    gauss = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0  # contrast (low-freq)
    hf = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)        # hf texture (sharpen)
    return {
        "despeckle": box, "edge": lap, "shadow": vgrad,
        "posterior": vsum, "contrast": gauss, "hf": hf,
    }


def build_experts(in_channel, out_channel, channel=32):
    ks = _kernels()
    return nn.ModuleList(
        _AnchoredExpert(in_channel, out_channel, ks[name], channel)
        for name in EXPERT_NAMES
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_experts.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/experts.py tests/moe/test_experts.py
git commit -m "feat(moe): anchored ultrasound-physics experts"
```

---

### Task B3: Degradation-aware top-2 spatial router

**Files:**
- Create: `models/Hybrid/USEANet/moe/router.py`
- Test: `tests/moe/test_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_router.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.Hybrid.USEANet.moe.router'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/moe/router.py
"""Degradation-aware per-position top-2 router (design §3.2).

Eats the feature map AND the label-free degradation proxy stack, emits a sparse
gate: at every spatial position exactly k=2 experts are active, gate weights
renormalised over the active two. Implemented as dense logits + a straight
top-k mask (the maps are tiny, so true sparse dispatch is not worth it).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationAwareRouter(nn.Module):
    def __init__(self, in_channel, num_experts=6, k=2, hidden=32):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        # Input = feature map + proxy stack (num_experts channels).
        self.net = nn.Sequential(
            nn.Conv2d(in_channel + num_experts, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )

    def forward(self, feat, proxy):
        logits = self.net(torch.cat([feat, proxy], dim=1))   # [B,E,H,W]
        self.last_logits = logits
        # Top-k over the expert dim at each position.
        topv, topi = logits.topk(self.k, dim=1)              # [B,k,H,W]
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        # Softmax over active experts only.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
        gate = F.softmax(masked_logits, dim=1) * mask
        return gate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_router.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/router.py tests/moe/test_router.py
git commit -m "feat(moe): degradation-aware top-2 spatial router"
```

---

### Task B4: MoE losses & utilization metric

**Files:**
- Create: `models/Hybrid/USEANet/moe/losses.py`
- Test: `tests/moe/test_losses.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_losses.py
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
    gate = torch.rand(1, 6, 4, 4, requires_grad=True).softmax(dim=1)
    proxy = torch.rand(1, 6, 4, 4).softmax(dim=1)
    (router_supervision_loss(gate, proxy) + load_balance_loss(gate)).backward()
    assert gate.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.Hybrid.USEANet.moe.losses'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/moe/losses.py
"""MoE auxiliary losses and utilization metric (design §3.4, §6).

router_supervision_loss: KL(proxy || gate) anchoring the gate to the label-free
  physical proxy (or physical GT in Track C).
load_balance_loss: switch-transformer style importance/load balancing.
effective_experts: exp(entropy(mean gate)) in [1, E]; a monitoring metric.
"""
import torch


def router_supervision_loss(gate, proxy, eps=1e-8):
    """KL(proxy || gate) averaged over positions. Both [B,E,H,W], sum-1 over E."""
    g = gate.clamp_min(eps)
    p = proxy.clamp_min(eps)
    kl = (p * (p.log() - g.log())).sum(dim=1)   # [B,H,W]
    return kl.mean()


def load_balance_loss(gate):
    """Switch loss: E * sum_e (f_e * P_e), minimised at uniform usage."""
    e = gate.shape[1]
    importance = gate.mean(dim=(0, 2, 3))                 # P_e
    load = (gate > 0).float().mean(dim=(0, 2, 3))         # f_e
    return e * torch.sum(importance * load)


def effective_experts(gate, eps=1e-8):
    """exp(entropy(mean gate over all positions)); float in [1, E]."""
    mean_gate = gate.mean(dim=(0, 2, 3))                  # [E]
    mean_gate = mean_gate / mean_gate.sum().clamp_min(eps)
    entropy = -(mean_gate * (mean_gate + eps).log()).sum()
    return float(torch.exp(entropy).item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_losses.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/moe/losses.py tests/moe/test_losses.py
git commit -m "feat(moe): router supervision, load-balance losses and eff-experts metric"
```

---

### Task B5: `PhysicsMoE` drop-in module

**Files:**
- Create: `models/Hybrid/USEANet/moe/physics_moe.py`
- Modify: `models/Hybrid/USEANet/moe/__init__.py` (remove the B1 import guard; restore plain imports)
- Test: `tests/moe/test_physics_moe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_physics_moe.py
import torch
from models.Hybrid.USEANet.moe import PhysicsMoE


def _moe(in_c=160, out_c=32):
    torch.manual_seed(0)
    return PhysicsMoE(in_c, out_c)


def test_output_shape_matches_multibranch_contract():
    # Drop-in for MultiBranchFeatureProcessor: [B,in,H,W] -> [B,out,H,W].
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_physics_moe.py -v`
Expected: FAIL — `ImportError: cannot import name 'PhysicsMoE'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/Hybrid/USEANet/moe/physics_moe.py
"""PhysicsMoE: sparse ultrasound-physics MoE block (design §3).

Drop-in replacement for MultiBranchFeatureProcessor at the x3/x4 layers:
``[B, in, H, W] -> [B, out, H, W]`` with a residual. Computes label-free proxy
maps, routes per-position top-2 over 6 anchored experts, combines densely with
the gate mask. Stashes ``last_gate`` / ``last_proxy`` and exposes ``aux_loss``
for the adapter to aggregate (router supervision + load balance).
"""
import torch
import torch.nn as nn

from . import EXPERT_NAMES
from .experts import build_experts
from .proxy import degradation_proxies
from .router import DegradationAwareRouter
from .losses import router_supervision_loss, load_balance_loss, effective_experts


class PhysicsMoE(nn.Module):
    def __init__(self, in_channel, out_channel, num_experts=None, k=2, channel=32):
        super().__init__()
        num_experts = num_experts or len(EXPERT_NAMES)
        self.experts = build_experts(in_channel, out_channel, channel)
        self.router = DegradationAwareRouter(in_channel, num_experts, k)
        self.res = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # Optional physical-GT supervision target (Track C); None -> use proxy.
        self.supervision_target = None
        self.last_gate = None
        self.last_proxy = None

    def forward(self, x):
        proxy = degradation_proxies(x)                       # [B,E,H,W]
        gate = self.router(x, proxy)                          # [B,E,H,W]
        # Dense expert compute + per-position top-2 mask combine.
        out = 0.0
        for e_idx, expert in enumerate(self.experts):
            out = out + expert(x) * gate[:, e_idx:e_idx + 1]
        self.last_gate = gate
        self.last_proxy = proxy
        return self.relu(out + self.res(x))

    def aux_loss(self, route_weight, lb_weight):
        if self.last_gate is None:
            raise RuntimeError("aux_loss called before forward()")
        target = self.supervision_target if self.supervision_target is not None else self.last_proxy
        route = router_supervision_loss(self.last_gate, target)
        lb = load_balance_loss(self.last_gate)
        return route_weight * route + lb_weight * lb

    def eff_experts(self):
        return effective_experts(self.last_gate)
```

- [ ] **Step 4: Restore plain imports in `__init__.py`**

Replace the `try/except ImportError` guard added in Task B1 so `models/Hybrid/USEANet/moe/__init__.py` reads:
```python
"""Ultrasound-physics MoE for USEANet (design Pillar 1). No simulator imports."""

EXPERT_NAMES = ["despeckle", "edge", "shadow", "posterior", "contrast", "hf"]

from .physics_moe import PhysicsMoE
from .losses import router_supervision_loss, load_balance_loss, effective_experts

__all__ = [
    "EXPERT_NAMES", "PhysicsMoE",
    "router_supervision_loss", "load_balance_loss", "effective_experts",
]
```

- [ ] **Step 5: Run all MoE unit tests to verify they pass**

Run: `conda run -n ubench1 python -m pytest tests/moe/ -v`
Expected: PASS (all proxy/experts/router/losses/physics_moe tests green).

- [ ] **Step 6: Commit**

```bash
git add models/Hybrid/USEANet/moe/physics_moe.py models/Hybrid/USEANet/moe/__init__.py tests/moe/test_physics_moe.py
git commit -m "feat(moe): PhysicsMoE drop-in block with aux loss aggregation"
```

---

### Task B6: Wire `PhysicsMoE` into `usea_core.py` (x3/x4 only)

**Files:**
- Modify: `models/Hybrid/USEANet/usea_core.py:380-382` (feature processors) and import near `:5`
- Test: `tests/moe/test_core_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_core_wiring.py
import torch
from models.Hybrid.USEANet.usea_core import USEANet, MultiBranchFeatureProcessor
from models.Hybrid.USEANet.moe import PhysicsMoE


def _net():
    torch.manual_seed(0)
    return USEANet(channel=32, num_classes=1, sem_downsample=1, use_softmax=False)


def test_x3_x4_are_physics_moe_x2_is_multibranch():
    net = _net()
    assert isinstance(net.feature_processor_3, PhysicsMoE)
    assert isinstance(net.feature_processor_4, PhysicsMoE)
    assert isinstance(net.feature_processor_2, MultiBranchFeatureProcessor)


def test_forward_still_returns_eight_maps():
    net = _net()
    x = torch.rand(1, 3, 256, 256)
    out = net(x)
    assert len(out) == 8
    for m in out:
        assert m.shape[0] == 1


def test_moe_modules_collectible():
    net = _net()
    moes = [m for m in net.modules() if isinstance(m, PhysicsMoE)]
    assert len(moes) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_core_wiring.py -v`
Expected: FAIL — `feature_processor_3` is `MultiBranchFeatureProcessor`, not `PhysicsMoE`.

- [ ] **Step 3: Apply the implementation change**

In `models/Hybrid/USEANet/usea_core.py`, add the import after the `pvtv2` import on line 5:
```python
from .moe import PhysicsMoE
```

Replace lines 380-382:
```python
        self.feature_processor_2 = MultiBranchFeatureProcessor(64, channel)
        self.feature_processor_3 = MultiBranchFeatureProcessor(160, channel)
        self.feature_processor_4 = MultiBranchFeatureProcessor(256, channel)
```
with:
```python
        # x2 keeps the plain fused branch; x3/x4 use the ultrasound-physics MoE
        # (design §3.3: avoid multi-layer CNN MoE variance, MoE only on semantic layers).
        self.feature_processor_2 = MultiBranchFeatureProcessor(64, channel)
        self.feature_processor_3 = PhysicsMoE(160, channel)
        self.feature_processor_4 = PhysicsMoE(256, channel)
```
The `forward` calls at lines 424-426 are unchanged (`PhysicsMoE.forward(x)` has the same `[B,in,H,W] -> [B,out,H,W]` contract).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_core_wiring.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/usea_core.py tests/moe/test_core_wiring.py
git commit -m "feat(moe): place PhysicsMoE at x3/x4 in USEANet core"
```

---

### Task B7: Aggregate MoE aux loss in the adapter + annealing

**Files:**
- Modify: `models/Hybrid/USEANet/__init__.py`
- Test: `tests/moe/test_adapter_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_adapter_integration.py
import torch
from models.Hybrid.USEANet import USEANet


def _model():
    torch.manual_seed(0)
    return USEANet(input_channel=3, num_classes=1, channel=32)


def test_set_training_progress_controls_route_weight():
    m = _model()
    m.set_training_progress(0.0)
    assert abs(m._route_weight() - 0.5) < 1e-6          # ROUTE_WEIGHT_MAX at start
    m.set_training_progress(1.0)
    assert abs(m._route_weight() - 0.05) < 1e-6         # ROUTE_WEIGHT_MIN at end


def test_deep_supervision_loss_includes_moe_aux():
    m = _model()
    m.set_training_progress(0.0)
    x = torch.rand(1, 1, 256, 256)
    label = (torch.rand(1, 1, 256, 256) > 0.5).float()
    outputs = m(x)
    loss = m.deep_supervision_loss(outputs, label)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_eff_experts_reported():
    m = _model()
    x = torch.rand(1, 1, 256, 256)
    m(x)
    stats = m.moe_stats()
    assert "eff_experts" in stats and 1.0 <= stats["eff_experts"] <= 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_adapter_integration.py -v`
Expected: FAIL — `AttributeError: 'USEANet' object has no attribute 'set_training_progress'`.

- [ ] **Step 3: Apply the implementation change**

In `models/Hybrid/USEANet/__init__.py`, add imports and constants after the existing imports (after line 27, `from .usea_loss import structure_loss`):
```python
from .moe import PhysicsMoE

# Router-supervision annealing (design §6: small weight + linear anneal -> anchor
# early, free the router later). Anneal from MAX to MIN over the first
# ANNEAL_FRACTION of training; load-balance weight is constant.
ROUTE_WEIGHT_MAX = 0.5
ROUTE_WEIGHT_MIN = 0.05
ANNEAL_FRACTION = 0.5
LB_WEIGHT = 0.01
```

In `USEANet.__init__` (after `self._bg_maps = None` on line 36) add:
```python
        self._progress = 0.0  # training fraction in [0,1], set by main.py loop
```

Add these methods to the `USEANet` adapter class (after `forward`, before `deep_supervision_loss`):
```python
    def set_training_progress(self, frac):
        self._progress = float(max(0.0, min(1.0, frac)))

    def _route_weight(self):
        t = min(self._progress / ANNEAL_FRACTION, 1.0)
        return ROUTE_WEIGHT_MAX + (ROUTE_WEIGHT_MIN - ROUTE_WEIGHT_MAX) * t

    def _moe_modules(self):
        return [m for m in self.modules() if isinstance(m, PhysicsMoE)]

    def moe_stats(self):
        moes = self._moe_modules()
        if not moes or moes[0].last_gate is None:
            return {}
        eff = sum(m.eff_experts() for m in moes) / len(moes)
        return {"eff_experts": eff}
```

In `deep_supervision_loss`, change the final loop + return so it reads:
```python
        for fg, bg in zip(fg_outputs, bg_outputs):
            total = total + structure_loss(fg, bg, label_batch, bg_mask, nc)
        # Add MoE auxiliary losses (router supervision + load balance), annealed.
        rw = self._route_weight()
        for moe in self._moe_modules():
            total = total + moe.aux_loss(route_weight=rw, lb_weight=LB_WEIGHT)
        return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_adapter_integration.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add models/Hybrid/USEANet/__init__.py tests/moe/test_adapter_integration.py
git commit -m "feat(moe): aggregate annealed MoE aux loss in USEANet adapter"
```

---

### Task B8: Pass training progress from `main.py`

**Files:**
- Modify: `main.py:321-329`
- Test: manual smoke (no unit test — touches the global training loop)

- [ ] **Step 1: Verify `max_iterations` is defined before the training loop**

Run: `grep -n "max_iterations" main.py | head`
Expected: a definition (e.g. `max_iterations = max_epoch * len(trainloader)`) at a line number lower than 317 (the `for i_batch ...` loop). If it is defined *after*, relocate the new hook or compute progress from `iter_num` against a precomputed total; confirm before editing.

- [ ] **Step 2: Apply the implementation change**

In `main.py`, inside the `if args.do_deeps:` block, immediately before `outputs = model(volume_batch)` (currently line 322), add:
```python
                if hasattr(model, 'set_training_progress'):
                    model.set_training_progress(iter_num / max(1, max_iterations))
```
So the block becomes:
```python
            if args.do_deeps:
                if hasattr(model, 'set_training_progress'):
                    model.set_training_progress(iter_num / max(1, max_iterations))
                outputs = model(volume_batch)
                # Models may supply their own deep-supervision loss (e.g. USEANet's
                # weighted fg/bg structure loss); otherwise use the shared criterion.
                if hasattr(model, 'deep_supervision_loss'):
                    loss = model.deep_supervision_loss(outputs, label_batch)
                else:
                    loss = deep_supervision_loss(outputs=outputs,label_batch=label_batch,loss_metric=criterion)
                outputs=outputs[-1]
```

- [ ] **Step 3: Smoke-test a few training iterations on BUSI**

Run (adjust dataset flags to your environment; keep it to 1 epoch / small subset):
```bash
conda run -n ubench1 python main.py --model USEANet --dataset_name busi \
  --base_dir <path-to-busi> --do_deeps 1 --max_epochs 1 --batch_size 4 2>&1 | tail -30
```
Expected: training runs without exceptions; loss is finite.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(moe): feed training progress to USEANet for router-supervision annealing"
```

---

### Task B9: Full Track-B regression + eff_experts sanity run

**Files:**
- None (verification only).

- [ ] **Step 1: Run the entire test suite**

Run: `conda run -n ubench1 python -m pytest tests/ -v`
Expected: PASS — all simulator, sim_fidelity, and moe tests green.

- [ ] **Step 2: Short training run, watch eff_experts for route collapse**

Read `model.moe_stats()` once per epoch in a scratch run. Train ~5 epochs on BUSI:
```bash
conda run -n ubench1 python main.py --model USEANet --dataset_name busi \
  --base_dir <path-to-busi> --do_deeps 1 --max_epochs 5 --batch_size 8 2>&1 | tail -40
```
Health check (design §3.4): `eff_experts` should settle in **2–4**. If it trends to **1.0**, the router is collapsing — raise `LB_WEIGHT` (0.01 → 0.05) or slow the anneal (`ANNEAL_FRACTION` 0.5 → 0.7) in `models/Hybrid/USEANet/__init__.py` and re-run. If a single expert's load `f_e → 0`, an expert is dead — same remedy.

- [ ] **Step 3: Commit any tuning**

```bash
git add models/Hybrid/USEANet/__init__.py
git commit -m "chore(moe): tune load-balance/anneal for healthy expert utilization"
```

---

# Track C — Gate-conditional: simulator → router physical supervision

> **DO NOT START** unless Track A's `gate_report.json` says `decision == "PROMOTE"`. If demoted, the simulator stays optional plain augmentation and Pillar 1 stands alone.

### Task C1: Feed simulator physical-GT degradation maps as router supervision

**Files:**
- Modify: `models/Hybrid/USEANet/__init__.py` (add `set_physical_supervision` / `clear_physical_supervision`)
- Modify: `main.py` (opt-in `--use_simulator` augmentation in the training step)
- Test: `tests/moe/test_physical_supervision.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/moe/test_physical_supervision.py
import torch
from models.Hybrid.USEANet.moe import PhysicsMoE
from models.Hybrid.USEANet import USEANet


def test_supervision_target_overrides_proxy_in_aux_loss():
    torch.manual_seed(0)
    m = PhysicsMoE(160, 32)
    feat = torch.rand(1, 160, 16, 16)
    m(feat)
    target = torch.rand(1, 6, 16, 16).softmax(dim=1)
    m.supervision_target = target
    aux_phys = m.aux_loss(0.5, 0.01)
    m.supervision_target = None
    aux_proxy = m.aux_loss(0.5, 0.01)
    assert torch.isfinite(aux_phys) and (aux_phys.item() != aux_proxy.item())


def test_set_physical_supervision_assigns_targets_per_layer():
    torch.manual_seed(0)
    m = USEANet(input_channel=3, num_classes=1, channel=32)
    x = torch.rand(1, 1, 256, 256)
    m(x)
    maps = {k: torch.rand(1, 1, 256, 256) for k in ("speckle", "shadow", "posterior")}
    m.set_physical_supervision(maps)
    for moe in m._moe_modules():
        assert moe.supervision_target is not None
        s = moe.supervision_target.sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    m.clear_physical_supervision()
    assert all(moe.supervision_target is None for moe in m._moe_modules())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_physical_supervision.py -v`
Expected: `test_supervision_target_overrides_proxy_in_aux_loss` PASSES (slot exists from B5); `test_set_physical_supervision_assigns_targets_per_layer` FAILS — `AttributeError: ... 'set_physical_supervision'`.

- [ ] **Step 3: Implement the target-projection helpers in the adapter**

In `models/Hybrid/USEANet/__init__.py`, add these methods to the `USEANet` adapter class (after `moe_stats`):
```python
    def set_physical_supervision(self, degradation_maps):
        """degradation_maps: dict[str -> [B,1,H,W]] from the simulator.

        Maps simulator types onto the 6 expert channels (missing experts -> 0),
        resizes per MoE layer, softmax-normalises, and stashes for aux_loss.
        Call AFTER forward() (needs each MoE's last_gate shape).
        """
        import torch.nn.functional as F
        from .moe import EXPERT_NAMES
        type_to_expert = {"speckle": "despeckle", "attenuation": "shadow",
                          "shadow": "shadow", "posterior": "posterior"}
        for moe in self._moe_modules():
            if moe.last_gate is None:
                continue
            b, e, h, w = moe.last_gate.shape
            stack = moe.last_gate.new_zeros(b, e, h, w)
            for sim_type, dm in degradation_maps.items():
                name = type_to_expert.get(sim_type)
                if name is None:
                    continue
                idx = EXPERT_NAMES.index(name)
                resized = F.interpolate(dm, size=(h, w), mode="bilinear", align_corners=False)
                stack[:, idx] = stack[:, idx] + resized[:, 0]
            moe.supervision_target = F.softmax(stack, dim=1)

    def clear_physical_supervision(self):
        for moe in self._moe_modules():
            moe.supervision_target = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n ubench1 python -m pytest tests/moe/test_physical_supervision.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit the adapter change**

```bash
git add models/Hybrid/USEANet/__init__.py tests/moe/test_physical_supervision.py
git commit -m "feat(moe): physical-GT router supervision from simulator maps"
```

- [ ] **Step 6: Wire simulator augmentation into the training step (opt-in)**

In `main.py`, add an `--use_simulator` arg (default off, mirroring the `--do_deeps` pattern). When set: construct one `UltrasoundDegradationSimulator` before the epoch loop; inside the `if args.do_deeps:` branch, with probability `p=0.5` per batch, run `sim_out = simulator(volume_batch)`, set `volume_batch = sim_out.image.repeat(1,3,1,1) if needed`, call `model(volume_batch)`, then `model.set_physical_supervision(sim_out.degradation_maps)` before the loss; on the no-sim branch call `model.clear_physical_supervision()`. Keep all of this behind the flag so non-simulator runs are byte-for-byte unchanged.

- [ ] **Step 7: Smoke-test the simulator-augmented path**

Run:
```bash
conda run -n ubench1 python main.py --model USEANet --dataset_name busi \
  --base_dir <path-to-busi> --do_deeps 1 --use_simulator 1 --max_epochs 1 --batch_size 4 2>&1 | tail -30
```
Expected: runs without exceptions; loss finite.

- [ ] **Step 8: Commit**

```bash
git add main.py
git commit -m "feat(moe): opt-in simulator augmentation with physical router supervision"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §3.1 6 experts → B2. §3.2 top-2 spatial degradation-aware router → B3. §3.3 x3/x4 placement, x2 plain → B6. §3.4 eff_experts/load monitoring → B4, B7, B9. §4 simulator + three uses + **hard gate** → A1–A6 (gate), C1 (router-supervision use), B8/C6 (aug use). §6 loss (native structure + router sup + load balance, anneal) → B7. §10 default 6 experts, decoupling, gate-first → track split + Track-C gate guard.
- **Left to experiments (not implementation):** ablations (§7) and cross-modality runs (§2) are training configs; the eff_experts/load hooks they need exist (B4/B7). No code task required.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step has complete code. Two manual steps (A6 Step 6 downstream experiment, C1 Step 6 main.py aug plumbing) are explicitly experiments / opt-in wiring with concrete instructions.

**Type consistency:** `EXPERT_NAMES` fixed order identical across proxy/experts/router/losses/physics_moe/adapter. Gate API `[B,E,H,W]` sum-1 over E consistent router→losses→supervision. `aux_loss(route_weight, lb_weight)`, `set_training_progress`, `_route_weight`, `_moe_modules`, `moe_stats`, `eff_experts`, `supervision_target` names consistent B5/B7/C1. `SimOutput.image` / `.degradation_maps` keys (`speckle`/`attenuation`/`shadow`/`posterior`) consistent A3→C1. B1 ships an import-guarded `__init__.py` so the package imports with only `proxy.py` present; B5 Step 4 removes the guard once `physics_moe`/`losses` exist.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-14-useanet-moe.md`.**
