# PUMA-Net: A Lightweight Physics-Anchored Ultrasound Mixture-of-Experts

> **Anonymous release for double-blind review.**
> This repository accompanies the paper *"PUMA-Net: A Lightweight
> Physics-Anchored Ultrasound Mixture-of-Experts for Interpretable Lesion
> Segmentation."* All author/affiliation information is intentionally omitted.

PUMA-Net is a compute-efficient ultrasound lesion segmentation network. It
anchors a sparse Mixture-of-Experts in **ultrasound image-formation physics**:
six experts defined from degradation cues (despeckle / edge / shadow /
posterior / contrast / high-frequency), a degradation-aware top-2 router
supervised by **label-free** proxy maps, and heterogeneous expert kernels. It
reaches accuracy competitive with much larger hybrid/Transformer baselines at
**~1.59 GFLOPs** and **~3.7M parameters**.

## Relationship to the code

- The paper's **PUMA-Net** contribution is the **PhysicsMoE** block in
  [`models/Hybrid/USEANet/moe/`](models/Hybrid/USEANet/moe/) — `proxy.py`
  (label-free degradation proxies), `router.py` (degradation-aware gate),
  `experts.py` (heterogeneous physics experts), `physics_moe.py`, `losses.py`.
- The surrounding encoder–decoder (PVT-B0 backbone + reverse-attention decoder)
  is the base architecture the MoE is dropped into; in the code it is named
  `USEANet`. Read `USEANet` + `PhysicsMoE` as **PUMA-Net**.
- The benchmark harness, datasets, and baseline models come from the **U-Bench**
  100-variant segmentation benchmark, on which this repository is built. Please
  cite U-Bench if you use the harness or baselines.

## Setup

```bash
conda create -n puma python=3.10 -y && conda activate puma
pip install -r requirements.txt
# ImageNet-pretrained PVT-B0 backbone weights (pvt_v2_b0.pth):
export PRETRAINED_MODEL_PATH=/path/to/pretrained   # dir holding pvt_v2_b0.pth
```

Datasets (BUSI, BUS, BUS-BRA, TUSCUI) follow the U-Bench splits; place them
under `hf_data/data/<dataset>/` (or adjust `--base_dir`).

## Reproduce

**Train PUMA-Net** (single dataset, one seed):

```bash
USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 USEANET_STRONG_AUG=1 \
USEANET_LOSS_REGION=iou \
python main.py --model USEANet --model_id 115 --do_deeps 1 \
  --base_dir hf_data/data/busi --dataset_name busi \
  --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name puma_busi_s41
```

Ablation switches (default-off → the full 6-expert top-2 model): set
`USEANET_NO_MOE=1` (MoE off), `USEANET_NUM_EXPERTS=N`, `USEANET_TOPK=k`.

**Analysis tools** (from the paper):

| Script | Produces |
|---|---|
| `tools/dump_percase.py` | per-image validation IoU from a best checkpoint |
| `tools/wilcoxon_sig.py` | paired Wilcoxon significance vs. baselines |
| `tools/routing_alignment.py` | expert↔proxy specialization metrics |
| `tools/render_routing_fig.py` | routing interpretability figure |
| `tools/measure_efficiency.py` | Params / FLOPs |

## License

See `LICENSE`. Third-party components (U-Bench harness, backbone/baseline
implementations) retain their original licenses and attributions.
