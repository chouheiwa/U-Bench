#!/usr/bin/env python
"""Convert U-Bench ultrasound datasets into nnU-Net v2 raw datasets.

Holds U-Bench's exact train/val split (via a sidecar splits file the driver
injects as splits_final.json) and U-Bench's binary mask definition
(label[label>0]=1, see dataloader/dataset.py:60-63). Everything else
(resampling/normalization/architecture) stays native nnU-Net.

Per-dataset source layout (confirmed on disk 2026-06-22):
  busi/bus/tuscui : images/<name>.png   + masks/0/<name>.png
  BUSBRA          : Images/<name>.png   + Masks/<name with bus_->mask_>.png
Case-name sanitization is mandatory: nnU-Net case IDs must be clean
identifiers (no spaces/parens/dots), so we mint <ds>_<idx5> ids and persist
case_map.json {clean_id: {orig_name, val}} for eval to map predictions back
to the original val GT.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

# Repo root = two levels up from this file (tools/nnunet/).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO, "hf_data", "data")
RAW_ROOT = os.environ.get("nnUNet_raw", os.path.join(REPO, "nnunet", "raw"))

# U-Bench dataset -> nnU-Net DatasetID name.
DATASETS = {
    "busi": "Dataset501_busi",
    "bus": "Dataset502_bus",
    "BUSBRA": "Dataset503_BUSBRA",
    "tuscui": "Dataset504_tuscui",
}


def src_paths(ds, entry):
    """Return (image_path, mask_path) for one split-list entry of dataset ds."""
    root = os.path.join(DATA_ROOT, ds)
    if ds == "BUSBRA":
        # List entries already carry the .png suffix; mask = bus_ -> mask_.
        name = entry if entry.endswith(".png") else entry + ".png"
        img = os.path.join(root, "Images", name)
        mask = os.path.join(root, "Masks", name.replace("bus_", "mask_", 1))
        return img, mask
    # busi / bus / tuscui
    name = entry if entry.endswith(".png") else entry + ".png"
    img = os.path.join(root, "images", name)
    mask = os.path.join(root, "masks", "0", name)
    return img, mask


def read_list(ds, which):
    path = os.path.join(DATA_ROOT, ds, f"{which}.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def convert_one(ds, force=False):
    name = DATASETS[ds]
    out = os.path.join(RAW_ROOT, name)
    imagesTr = os.path.join(out, "imagesTr")
    labelsTr = os.path.join(out, "labelsTr")
    imagesVal = os.path.join(out, "imagesVal")  # val-only input folder for predict
    done_flag = os.path.join(out, ".convert_done")
    if os.path.exists(done_flag) and not force:
        print(f"[skip] {name} already converted")
        return
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesVal, exist_ok=True)

    train = read_list(ds, "train")
    val = read_list(ds, "val")
    case_map = {}
    train_ids, val_ids = [], []
    idx = 0
    for entries, is_val, bucket in ((train, False, train_ids), (val, True, val_ids)):
        for entry in entries:
            clean = f"{ds}_{idx:05d}"
            idx += 1
            img_p, mask_p = src_paths(ds, entry)
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                sys.exit(f"[err] {ds}: cannot read image {img_p} (entry {entry!r})")
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                sys.exit(f"[err] {ds}: cannot read mask {mask_p} (entry {entry!r})")
            lbl = np.zeros_like(mask, dtype=np.uint8)
            lbl[mask > 0] = 1  # binarize, mirrors dataset.py:60-63
            img_name = f"{clean}_0000.png"
            cv2.imwrite(os.path.join(imagesTr, img_name), img)
            cv2.imwrite(os.path.join(labelsTr, f"{clean}.png"), lbl)
            if is_val:  # symlink val image into the predict input folder
                link = os.path.join(imagesVal, img_name)
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(os.path.join("..", "imagesTr", img_name), link)
            case_map[clean] = {"orig_name": entry, "val": is_val}
            bucket.append(clean)

    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump({
            "channel_names": {"0": "ultrasound"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": len(train_ids) + len(val_ids),
            "file_ending": ".png",
        }, f, indent=2)

    # Sidecar in splits_final.json shape; driver copies into nnUNet_preprocessed.
    with open(os.path.join(out, "ubench_split.json"), "w") as f:
        json.dump([{"train": train_ids, "val": val_ids}], f, indent=2)
    with open(os.path.join(out, "case_map.json"), "w") as f:
        json.dump(case_map, f, indent=2)

    open(done_flag, "w").close()
    print(f"[done] {name}: train={len(train_ids)} val={len(val_ids)} "
          f"total={len(case_map)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all", *DATASETS.keys()])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        convert_one(ds, force=args.force)


if __name__ == "__main__":
    main()
