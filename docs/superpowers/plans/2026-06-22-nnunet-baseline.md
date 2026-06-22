# nnU-Net Baseline — Implementation Plan

> Executes spec `docs/superpowers/specs/2026-06-22-nnunet-baseline-design.md`. Inline execution (cost-aware). GPU-bound steps run later via the GPU-guarded driver after the matrix frees the cards.

**Goal:** nnU-Net v2 baseline, native pipeline, U-Bench split, 4 datasets × 3 seeds, U-Bench-comparable IoU into `result/result_nnunet.csv`.

**Tech:** isolated conda env `nnunet` (`nnunetv2`), Python conversion/eval scripts, bash work-stealing driver.

---

## Status (2026-06-22)
- **Phase A: DONE** (commit `31d08e9`). A1 env `nnunet`+nnunetv2 2.8.0; trainers resolve by name. A2 seed trainers symlinked into nnunetv2 `variants/`. A3 convert (+imagesVal symlinks, case_map/ubench_split sidecars). A4 convert sanity (bus 393/169) green. A5 eval + IoU parity (54 cases) green. A6 driver `tools/run_nnunet.sh` (`bash -n` clean, GPU-guarded).
- Discovery: official `nnUNetTrainer_250epochs`/`_5epochs` presets exist → seed trainers just subclass them (no manual epoch wiring). cudnn left native (not forced deterministic).
- **Phase B: BLOCKED on matrix** (68/120 as of 10:30). Do NOT launch `run_nnunet.sh` until matrix 120/120 — its GPU guard would grab brief inter-cell windows and contend. Launch smoke (`_s41_5e` on bus) first, then full 12-cell.

## Phase A — buildable now (no GPU)

### Task A1: Isolated env + registration smoke
- Create env: `conda create -y -n nnunet python=3.11`; `conda run -n nnunet pip install nnunetv2`.
- Verify import + locate trainer base + variants dir:
  `conda run -n nnunet python -c "import nnunetv2, nnunetv2.training.nnUNetTrainer.nnUNetTrainer as t; print(nnunetv2.__path__)"`.
- Decide registration hook for custom trainers (dynamic vs copy-into-variants). Record the working path.

### Task A2: Custom trainers `tools/nnunet/nnUNetTrainer_seeds.py`
- 3 subclasses `nnUNetTrainer_s41_250e/_s42_250e/_s43_250e`: set `self.num_epochs=250`; seed torch/cuda/np/random in `__init__` (mirror `main.py:58-68`).
- Install into the discovered location (A1).
- Verify nnU-Net can resolve them: `nnUNetv2_train -h` then a dry `-tr nnUNetTrainer_s41_250e` resolution check (no data needed — just class lookup, may need a stub; otherwise defer resolution check to smoke).

### Task A3: Conversion `tools/nnunet/convert_dataset.py`
- Map U-Bench ds → Dataset501_busi/502_bus/503_BUSBRA/504_tuscui.
- Read `train.txt`+`val.txt`; copy images → `imagesTr/<case>_0000.png`, masks → `labelsTr/<case>.png` binarized.
- Per-ds path quirks: busi/bus/tuscui `images/`+`masks/0/`; BUSBRA `Images/`+`Masks/mask_<id>.png` remap. Roots `hf_data/data/<ds>/`.
- Emit `dataset.json` (1 channel, labels bg0/lesion1, `.png`).
- Emit a sidecar `splits.json` (train/val basename lists) for later injection.
- Idempotent (skip built datasets).

### Task A4: Conversion unit/sanity test `tools/nnunet/test_convert.py`
- Run convert on each ds (or smallest `bus`), assert: imagesTr count == train+val lines; every label ∈{0,1}; every image has matching label; split counts == txt line counts (busi 452/195, bus 393/169, BUSBRA 1500/375, tuscui 2550/1094).

### Task A5: Eval + IoU `tools/nnunet/eval_nnunet.py`
- IoU fn = exact copy of `utils/metrics_medpy.py:19-21` semantics (`inter/union`, union0→0).
- Given a pred dir + the ds val list: load pred (0/1) & GT (binarized), per-case IoU, mean → cell IoU.
- Append `modelname=nnUNet,dataset,seed,exp_name=nnunet_s<seed>,best_iou` to `result/result_nnunet.csv`.
- **IoU parity unit test** `test_iou_parity.py`: synthetic pred/GT pairs → assert eval IoU == `utils/metrics_medpy` IoU bit-for-bit.

### Task A6: Driver `tools/run_nnunet.sh <GPU>`
- Mirror `run_matrix.sh`: tasks = 4 ds × {41,42,43}; locks `locks/nnunet/`; FAILED markers; `is_done` = row in `result/result_nnunet.csv` for (ds,seed) with iou>0 OR FAILED.
- **GPU-free guard** (`<2500MiB` wait-loop) → auto-queues behind the matrix.
- Per cell: ensure `convert` done; ensure `nnUNetv2_plan_and_preprocess -d 50X` done once/ds (sentinel); inject `splits_final.json` into preprocessed dir; `CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train 50X 2d 0 -tr nnUNetTrainer_s<seed>_250e`; then `nnUNetv2_predict` val → `eval_nnunet.py`.
- `.gitignore` the `nnunet/{raw,preprocessed,results}` work dirs.

## Phase B — GPU-bound (auto, after matrix)

### Task B1: Smoke
- `bus` × seed41, **5-epoch** trainer variant (or `--val_disable`/short), run full convert→preprocess→train→predict→eval→CSV. Confirm chain + plausible IoU. Fix any registration/path breakage.

### Task B2: Full 12-cell run
- Launch `tools/run_nnunet.sh` on both GPUs (after matrix done). 12 cells, ~250ep each. Monitor under the same 2h cadence.

### Task B3: Merge + report
- Add nnU-Net row (3-seed mean±std per ds) to the final method×dataset table.
- Update memory `useanet-c3a-baseline-and-hetero.md`.

---

## Verification
- A: unit tests green (convert sanity + IoU parity); trainers resolve; driver `bash -n` clean; dry `is_done`/guard logic.
- B: smoke produces a CSV row; full run → `result/result_nnunet.csv` has 12 rows, IoU ∈ ~0.6–0.85.
- Final: nnU-Net vs PUMA-Net vs 10 methods table, 3-seed CIs.

## Data layout gotchas (confirmed on disk 2026-06-22)
Split files are `hf_data/data/<ds>/{train,val}.txt`. Counts verified: busi 452/195, bus 393/169, BUSBRA 1500/375, tuscui 2550/1094.
**Case-name sanitization is REQUIRED** (nnU-Net case IDs must be clean identifiers — no spaces/parens/dots):
- **busi**: list entries like `malignant (84)` → file `images/malignant (84).png`, mask `masks/0/malignant (84).png`. Sanitize → e.g. `busi_<idx>` with a kept `{clean→original}` map for val GT lookup.
- **bus**: `case0493` (clean) → `images/case0493.png`, `masks/0/case0493.png`.
- **BUSBRA**: list entries include `.png` (`bus_0523-l.png`); image `Images/bus_0523-l.png`, mask `Masks/mask_0523-l.png` (strip `bus_`→`mask`, drop extension when sanitizing case id). Verify mask naming on a sample during impl.
- **tuscui**: pure numeric `1698` → `images/1698.png`, `masks/0/1698.png`. Prefix to `tuscui_1698` for safety.
Convert must persist a per-dataset `case_map.json {clean_id: {orig_name, val:bool}}` so `eval_nnunet.py` can match predictions back to the original val GT masks.

## Notes
- Only deviation from nnU-Net default: **250 epochs** (parity + feasibility) — confirmed by user.
- Do not touch `ubench1`, `run_matrix.sh`, or the running matrix.
