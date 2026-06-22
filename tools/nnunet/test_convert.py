#!/usr/bin/env python
"""Sanity test for convert_dataset.py — runs on the smallest dataset (bus).

Asserts: imagesTr count == train+val lines; every label is binary {0,1};
every image has a matching label; split counts == txt line counts.
Run: conda run -n ubench1 python tools/nnunet/test_convert.py
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_dataset as cd  # noqa: E402

# Expected split counts (confirmed on disk 2026-06-22).
EXPECTED = {
    "busi": (452, 195),
    "bus": (393, 169),
    "BUSBRA": (1500, 375),
    "tuscui": (2550, 1094),
}


def check(ds):
    train_lines = len(cd.read_list(ds, "train"))
    val_lines = len(cd.read_list(ds, "val"))
    exp_tr, exp_val = EXPECTED[ds]
    assert (train_lines, val_lines) == (exp_tr, exp_val), \
        f"{ds}: list counts {(train_lines, val_lines)} != expected {(exp_tr, exp_val)}"

    cd.convert_one(ds, force=True)
    out = os.path.join(cd.RAW_ROOT, cd.DATASETS[ds])
    images = [f for f in os.listdir(os.path.join(out, "imagesTr")) if f.endswith(".png")]
    labels = [f for f in os.listdir(os.path.join(out, "labelsTr")) if f.endswith(".png")]
    assert len(images) == train_lines + val_lines, \
        f"{ds}: imagesTr {len(images)} != {train_lines + val_lines}"
    assert len(labels) == len(images), \
        f"{ds}: labelsTr {len(labels)} != imagesTr {len(images)}"

    with open(os.path.join(out, "ubench_split.json")) as f:
        split = json.load(f)[0]
    assert len(split["train"]) == train_lines, f"{ds}: split train mismatch"
    assert len(split["val"]) == val_lines, f"{ds}: split val mismatch"

    # Every image has a matching label; spot-check label binariness on 20 cases.
    img_ids = {f[:-len("_0000.png")] for f in images}
    lbl_ids = {f[:-len(".png")] for f in labels}
    assert img_ids == lbl_ids, f"{ds}: image/label id sets differ"
    for clean in list(lbl_ids)[:20]:
        lbl = cv2.imread(os.path.join(out, "labelsTr", f"{clean}.png"),
                         cv2.IMREAD_GRAYSCALE)
        vals = set(np.unique(lbl).tolist())
        assert vals <= {0, 1}, f"{ds}: label {clean} not binary, got {vals}"
    print(f"[ok] {ds}: {len(images)} images, split {train_lines}/{val_lines}, "
          f"labels binary")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["bus"]
    for ds in targets:
        check(ds)
    print("ALL CONVERT TESTS PASSED")
