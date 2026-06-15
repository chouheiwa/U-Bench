# tests/test_training_recipe.py
import math
import pytest
import torch
import torch.nn as nn
from torch import optim
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


def test_warmup_iters_default_zero(monkeypatch):
    monkeypatch.delenv("USEANET_WARMUP_EPOCHS", raising=False)
    assert tr.warmup_iters(125) == 0


def test_warmup_iters_reads_env(monkeypatch):
    monkeypatch.setenv("USEANET_WARMUP_EPOCHS", "5")
    assert tr.warmup_iters(125) == 5 * 125


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
