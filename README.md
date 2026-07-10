# A Label-Free Reliability Gate for Cross-Domain Ultrasound Lesion Segmentation

> **Anonymous release for double-blind review.** Author and affiliation
> information is intentionally omitted.

This repository accompanies a paper on a **label-free reliability gate** for
cross-domain breast-ultrasound (BUS) lesion segmentation. Deep segmenters degrade
sharply under domain shift across scanners and sites; a deployable decision-support
system must know *when to trust a prediction* and *when to adapt*, using only
information available at test time (no target labels).

The system pairs a lightweight, physics-anchored mixture-of-experts **engine**
(a reused backbone, ~3.66M params / ~1.59 GFLOPs) with a single-pass, label-free
gate that provides three deployable capabilities:

- **Selective prediction** — rank cases by prediction confidence and defer the
  least-confident fraction to a human (entropy detects per-case transfer failure
  at AUROC ~0.79; deferring 20% lifts retained IoU by +0.056).
- **Per-case routing-variant selection** — run *K* lightweight routing variants
  and keep the most confident prediction; parameter-free and label-free, this
  beats the best fixed variant by +0.0196 IoU and is net-positive from every
  source domain.
- **Gated adaptation** — adapt only where a label-free signal predicts adaptation
  helps, beating always-on adaptation for every source-free method tested.

We report the negatives with equal weight: physics-calibrated routing is
*source-conditional* (it helps from some source domains and hurts from others),
and physics-head signals are redundant with ordinary confidence for the gate.

## Relationship to the code

- **Engine (reused backbone).** The lightweight physics-anchored MoE lives in
  [`models/Hybrid/USEANet/moe/`](models/Hybrid/USEANet/moe/): a calibrated
  acoustic-physics estimator (`physics_estimator.py`), a degradation-aware top-2
  router (`physics_moe.py`), and its losses (`losses.py`). It is treated as a
  **fixed engine**, not a contribution of this paper; efficiency is a property of
  the backbone, not a claim we make.
- **Gate and analysis tools** are in [`tools/`](tools/) (see table below).
- The benchmark harness, datasets, and baseline models come from the **U-Bench**
  segmentation benchmark, on which this repository is built. Please cite U-Bench
  if you use the harness or baselines.

## Setup

```bash
conda create -n gate python=3.10 -y && conda activate gate
pip install -r requirements.txt
# ImageNet-pretrained PVT-B0 backbone weights (pvt_v2_b0.pth):
export PRETRAINED_MODEL_PATH=/path/to/pretrained   # dir holding pvt_v2_b0.pth
```

Datasets (BUSI, BUS, BUS-BRA, BrEaST) follow the U-Bench splits; place them under
`hf_data/data/<dataset>/` (or adjust `--base_dir`). Our regime is zero-shot
cross-domain: a model trained on one source is evaluated, without any target
labels, on a different target.

## Reproduce

**Train an engine routing variant** (single dataset / seed). The calibrated
physics estimator is enabled by `USEANET_CALIB_PROXY=1`; `USEANET_CALIB_PHYS`
selects the acoustic quantity subset (empty = all three):

```bash
USEANET_CALIB_PROXY=1 USEANET_CALIB_PHYS=attenuation \
python main.py --model USEANet --model_id 115 --do_deeps 1 \
  --base_dir hf_data/data/busi --dataset_name busi \
  --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name calib_att_busi_s41
```

**Gate and analysis tools** (reproduce every table and figure; all label-free at
test time):

| Script | Produces |
|---|---|
| `tools/routing_gate.py` | per-case dump across the five routing variants (IoU + confidence signals) |
| `tools/analyze_routing_gate.py` | per-case routing-variant selection analysis |
| `tools/failure_gate.py` | label-free failure-detection signals (entropy / margin / band / fg-fraction) |
| `tools/gated_adapt.py` | entropy-gated source-free adaptation vs. always-on |
| `tools/expand_analysis.py` | risk–coverage sweep, signal-ablation AUROC, K-variant sweep, full cross-domain matrix |
| `tools/render_gate_qualitative.py` | selective-prediction qualitative figure (gate-keep vs. gate-defer) |

Before publishing this repository anywhere, run `bash tools/prepare_submission.sh`
to verify no identity-revealing paths or files are included.

## License

See `LICENSE`. Third-party components (U-Bench harness, backbone and baseline
implementations) retain their original licenses and attributions.
