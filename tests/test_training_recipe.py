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
