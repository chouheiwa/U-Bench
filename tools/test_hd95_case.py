#!/usr/bin/env python
"""Unit test for offline_hd95.hd95_case empty-pred penalty + medpy parity.

Run: conda run -n ubench1 python tools/test_hd95_case.py
"""
import math
import os
import sys

import numpy as np
from medpy.metric.binary import hd95 as medpy_hd95

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_hd95 as oh  # noqa: E402


def main():
    # both empty -> 0
    z = np.zeros((256, 256), np.uint8)
    assert oh.hd95_case(z, z) == 0.0, "both-empty should be 0"

    # pred empty, GT present -> diagonal penalty at GT resolution
    gt = z.copy(); gt[10:20, 10:20] = 1
    diag256 = math.sqrt(256**2 + 256**2)
    assert abs(oh.hd95_case(z, gt) - diag256) < 1e-9, "empty-pred 256 penalty"
    gt224 = np.zeros((224, 224), np.uint8); gt224[5:15, 5:15] = 1
    diag224 = math.sqrt(224**2 + 224**2)
    assert abs(oh.hd95_case(np.zeros((224, 224), np.uint8), gt224) - diag224) < 1e-9, \
        "empty-pred 224 penalty"

    # perfect overlap -> 0
    assert oh.hd95_case(gt, gt) == 0.0, "perfect overlap should be 0"

    # non-trivial geometry -> must equal raw medpy hd95
    rng = np.random.RandomState(0)
    for _ in range(20):
        a = np.zeros((64, 64), np.uint8)
        b = np.zeros((64, 64), np.uint8)
        a[rng.randint(0, 30):rng.randint(31, 60), rng.randint(0, 30):rng.randint(31, 60)] = 1
        b[rng.randint(0, 30):rng.randint(31, 60), rng.randint(0, 30):rng.randint(31, 60)] = 1
        if a.sum() == 0 or b.sum() == 0:
            continue
        ref = float(medpy_hd95(a, b))
        got = oh.hd95_case(a, b)
        assert abs(ref - got) < 1e-9, f"parity {got} != {ref}"

    print("ALL HD95_CASE TESTS PASSED")


if __name__ == "__main__":
    main()
