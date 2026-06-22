#!/usr/bin/env python
"""IoU parity test: eval_nnunet.iou_score must equal U-Bench's metrics_medpy IoU.

Run: conda run -n ubench1 python tools/nnunet/test_iou_parity.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import eval_nnunet  # noqa: E402


def ubench_iou(output, target):
    """Reference: utils/metrics_medpy.py:19-21 semantics on binary arrays."""
    output = np.asarray(output)
    target = np.asarray(target)
    intersection = np.sum(output * target)
    union = np.sum(output) + np.sum(target) - intersection
    return intersection / union if union > 0 else 0.0


def main():
    rng = np.random.RandomState(0)
    cases = [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),  # partial
        (np.zeros((4, 4), int), np.zeros((4, 4), int)),            # union==0 -> 0
        (np.ones((4, 4), int), np.ones((4, 4), int)),              # perfect -> 1
        (np.ones((4, 4), int), np.zeros((4, 4), int)),             # disjoint -> 0
    ]
    for _ in range(50):
        cases.append((rng.randint(0, 2, (16, 16)), rng.randint(0, 2, (16, 16))))

    for i, (o, t) in enumerate(cases):
        ref = float(ubench_iou(o, t))
        got = eval_nnunet.iou_score(o, t)
        assert abs(ref - got) < 1e-12, f"case {i}: parity {got} != {ref}"
    print(f"ALL IOU PARITY TESTS PASSED ({len(cases)} cases)")


if __name__ == "__main__":
    main()
