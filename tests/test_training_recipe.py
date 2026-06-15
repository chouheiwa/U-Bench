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


def test_build_optimizer_disc_lr_default_mult_is_0p2(monkeypatch):
    # A 路线定论:默认 backbone_lr_mult=0.2(均值持平 0.1、方差减半)。
    monkeypatch.setenv("USEANET_DISC_LR", "1")
    monkeypatch.delenv("USEANET_BACKBONE_LR_MULT", raising=False)
    monkeypatch.delenv("USEANET_ADAMW", raising=False)
    model = _TinyNet()
    opt, group_base_lrs = tr.build_optimizer(model, 0.01)
    assert group_base_lrs == [pytest.approx(0.002), pytest.approx(0.01)]
    assert opt.param_groups[0]["lr"] == pytest.approx(0.002)  # backbone = base*0.2


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


def test_model_ema_update_formula():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = tr.ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    # shadow = 0.9*1.0 + 0.1*2.0 = 1.1
    assert ema.shadow["weight"].item() == pytest.approx(1.1)


def test_model_ema_store_copy_restore_roundtrip():
    model = nn.Linear(1, 1, bias=False)
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
