# nnU-Net Baseline Integration — Design Spec

**Date:** 2026-06-22
**Goal:** Add **nnU-Net v2** as a comparison baseline to the U-Bench matrix study, run in its **native self-configuring pipeline** but on **U-Bench's exact train/val split**, across **4 ultrasound datasets × 3 seeds**, producing IoU numbers **comparable** to the existing 10-method × 4-dataset × 3-seed matrix and to PUMA-Net.

**Decided (user, 2026-06-22):**
- nnU-Net = **native pipeline + same split** (NOT forced into U-Bench's 256px/250ep loop).
- MedSAM = **dropped** (GT-box prompt breaks fairness vs fully-automatic methods).
- Scheduling = **queue after the 120-cell matrix finishes** (no GPU contention).

---

## The comparability problem

nnU-Net's value *is* its self-configuring preprocessing + training. We must NOT cripple that, but we MUST hold three things identical to U-Bench so the IoU is a fair head-to-head:

| Held identical to U-Bench | How |
|---|---|
| **Which images are train vs val** | nnU-Net custom `splits_final.json`: fold 0 train = `train.txt` basenames, val = `val.txt` basenames (per dataset). nnU-Net's internal best-checkpoint selection then happens on **our** val set. |
| **The val set IoU is measured on** | Run inference on exactly the `val.txt` images; compute IoU on full-resolution GT masks. |
| **The IoU definition** | Reuse U-Bench's formula `utils/metrics_medpy.py:19-21`: binary masks, `inter/union`, per-image then mean over val. nnU-Net outputs argmax labels (0/1) → no sigmoid/threshold needed, same set algebra. |
| **The 3 seeds** | 41 / 42 / 43, matching `tools/run_matrix.sh`. |

What is intentionally **different** (this is nnU-Net's contribution, keeping it is the honest comparison):
- nnU-Net's own resampling/normalization/patch-based 2d preprocessing (not 256px /255).
- nnU-Net's own augmentation (not U-Bench's `USEANET_STRONG_AUG`).
- nnU-Net's own architecture/optimizer/LR schedule/deep-supervision.

### The one deviation from nnU-Net default — flagged

nnU-Net default = **1000 epochs**. At ~real wall-clock that is **>1 day per (dataset,seed)**; 4×3 = 12 runs would take ~2 weeks on 2 GPUs. We cap to **250 epochs** via the `nnUNetTrainer_250epochs` preset, which **also matches U-Bench's `--max_epochs 250`** → kills two birds (feasibility + epoch-budget parity). This is the **only** departure from nnU-Net defaults and must be stated in the paper's methods ("nnU-Net trained for 250 epochs to match the common training budget"). Everything else stays native.

---

## Architecture / components

All new files; **nothing in the existing training code or `run_matrix.sh` is touched**.

### 1. Isolated environment
- New conda env **`nnunet`** (separate from `ubench1`): `pip install nnunetv2`. nnU-Net v2 pins its own torch; must not perturb `ubench1` (the running matrix).
- nnU-Net's required env vars: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` → point under `./nnunet/{raw,preprocessed,results}` in the repo (gitignored).

### 2. Dataset conversion — `tools/nnunet/convert_dataset.py`
For each U-Bench dataset, build an nnU-Net raw dataset:
- One `DatasetID` per U-Bench dataset: `Dataset501_busi`, `Dataset502_bus`, `Dataset503_BUSBRA`, `Dataset504_tuscui`.
- Copy **both** train.txt and val.txt images into `imagesTr/` (nnU-Net needs all referenced images in `imagesTr`; the split decides train vs val). Channel suffix `_0000`: `<case>_0000.png`. Labels → `labelsTr/<case>.png`, binarized `label[label>0]=1` (matches `dataset.py:60-63`).
- Resolve per-dataset path quirks (from Explore findings): busi/bus/tuscui images `images/`, masks `masks/0/`; **BUSBRA** images `Images/` (cap I), masks `Masks/mask_<id>.png` (filename remap). Source roots under `hf_data/data/<ds>/`.
- Write `dataset.json`: 1 input channel (grayscale; nnU-Net handles 1-ch), labels `{background:0, lesion:1}`, file ending `.png`, `numTraining` = train+val count.
- Write `splits_final.json` into `nnUNet_preprocessed/Dataset50X_*/`: a single fold whose `train`=train.txt basenames, `val`=val.txt basenames. (Generated **after** `plan_and_preprocess` so the folder exists, or pre-created then preserved.)

### 3. Seed×budget trainer — `tools/nnunet/nnUNetTrainer_seeds.py`
nnU-Net v2 `nnUNetv2_train` has no `--seed` flag. Provide 3 trainer subclasses registered into nnunetv2's trainer namespace:
- `nnUNetTrainer_s41_250e`, `_s42_250e`, `_s43_250e` — each subclasses `nnUNetTrainer`, sets `self.num_epochs = 250`, and seeds `torch.manual_seed / cuda / np.random / random` to {41,42,43} in `__init__` (mirrors `main.py:58-68`). Installed via a path nnU-Net discovers (its `nnunetv2/training/nnUNetTrainer/variants/` or a documented plugin import).
- (Confirm the exact registration hook against the installed nnunetv2 version during implementation — fall back to copying the file into the package's variants dir if dynamic registration is unavailable.)

### 4. Driver — `tools/run_nnunet.sh <GPU>`
Mirrors `run_matrix.sh` conventions (idempotent, restartable, work-stealing, GPU-free guard, FAILED markers, lock dirs under `locks/nnunet/`):
- Task list = 4 datasets × 3 seeds = 12 cells.
- **GPU-free guard**: waits until the target GPU is `<2500MiB` used → only starts after the matrix vacates the card (satisfies "queue after matrix" automatically, no manual coordination).
- Per cell: ensure preprocess done once per dataset (`nnUNetv2_plan_and_preprocess -d 50X`), inject splits, then `CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train 50X 2d 0 -tr nnUNetTrainer_s<seed>_250e` (fold 0 = our split).
- `is_done()` check: a results CSV row for (nnUNet, ds, seed) exists with IoU>0, OR a FAILED marker.

### 5. Inference + IoU — `tools/nnunet/eval_nnunet.py`
- `nnUNetv2_predict` on the `val.txt` images (from `imagesTs/` copy, or re-use imagesTr subset) using the best checkpoint of the trained fold.
- For each val case: load prediction (0/1) and GT mask (binarized), compute IoU with the **exact** U-Bench formula (`intersection/union`, union==0→0). Mean over val = the cell's IoU.
- Append a row to **`result/result_<ds>_train.csv`** with `modelname=nnUNet`, `seed`, `exp_name=nnunet_s<seed>`, and the IoU written into the **same `best_iou` column position** the matrix uses — BUT note the CSV is 34-col schema (`best_iou` is the field right after the `./output/...` path). Simplest robust approach: write a **separate** `result/result_nnunet.csv` with `modelname,dataset,seed,best_iou`, and merge into the final table at report time (avoids fragile column alignment in the shared CSV). The matrix's `is_done` only scans for matrix methods, so nnUNet rows in the shared CSV would be ignored anyway → separate file is cleaner.

### 6. Final-table merge
At the end, the "method × dataset" table (3-seed mean ± std) gains an **nnU-Net** row alongside the 10 matrix methods + PUMA-Net. Update memory `useanet-c3a-baseline-and-hetero.md`.

---

## Data flow

```
hf_data/data/<ds>/{images,masks}  ──convert_dataset.py──▶  nnUNet_raw/Dataset50X_<ds>/{imagesTr,labelsTr,dataset.json}
                                                                      │
                                              nnUNetv2_plan_and_preprocess -d 50X
                                                                      │  + inject splits_final.json (= U-Bench train/val)
                                                                      ▼
                                   nnUNetv2_train 50X 2d 0 -tr nnUNetTrainer_s<seed>_250e   (×3 seeds ×4 ds, GPU-guarded queue)
                                                                      │  checkpoint_best (selected on our val set)
                                                                      ▼
                                   nnUNetv2_predict on val.txt images  ──eval_nnunet.py (U-Bench IoU formula)──▶ result/result_nnunet.csv
                                                                      │
                                                                      ▼
                                                        final method×dataset table (+ nnU-Net row)
```

## Error handling
- Each cell isolated; failure → FAILED marker (no retry loop), logged, other cells proceed (same pattern as matrix).
- Conversion is idempotent (skip if Dataset50X already built).
- Preprocess once per dataset, guarded by a sentinel file.

## Testing / verification
- **Conversion sanity**: per dataset, assert `len(imagesTr)==train+val count`, every label is binary {0,1}, every image has a matching label, splits_final train/val counts == train.txt/val.txt line counts (busi 452/195, bus 393/169, BUSBRA 1500/375, tuscui 2550/1094).
- **IoU formula parity**: unit-test `eval_nnunet`'s IoU against `utils/metrics_medpy.py` on a synthetic pair → identical value.
- **Smoke**: 1 dataset (bus, smallest non-trivial) × seed41 × a few epochs → confirms the full train→predict→IoU→CSV chain end-to-end before launching all 12.
- **End-to-end**: the 12-cell driver run, IoU values land in `result/result_nnunet.csv`, plausible range (ultrasound lesion IoU ~0.6–0.85).

## Out of scope
- MedSAM (dropped).
- 3D nnU-Net configs / ensembling / cascade (these are 2D images → `2d` only).
- 5-fold CV (we use the single U-Bench split as one fold).

## Open item for user confirmation
- **250-epoch cap** on nnU-Net (vs native 1000) — needed for feasibility + epoch parity; only deviation from defaults. Confirm acceptable.
