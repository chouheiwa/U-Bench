# USEANet-MoE — 下一步待办(新会话接续用)

> 更新:2026-06-14 · 已合并到 main。分支 `feat/useanet-moe`。
> 环境:conda `ubench1`(torch 2.7,2× RTX 2080 Ti,用 `--gpu 0`)。测试:`conda run -n ubench1 python -m pytest tests/ -q`(当前 49 passed)。

## 现状(本轮成果)
- 修复两处致命 bug:输出头顺序(`a3903cf`)、预训练 backbone 输入尺度(`a3903cf`)。
- BUSI 进展:`0.000 → 0.13 → 0.623 → 0.667(强增强) → 0.682(+TTA)`,**最佳 IoU 0.682 / Dice 0.773**。
- 结论:TTA 真增益(采纳);最大连通域净负(丢弃);加权深监督噪声内无明确收益(默认关)。
- 三个 env 开关(默认全关,对其他模型零影响):
  - `USEANET_STRONG_AUG=1` — 强增强(**需配长训练 ~250ep 才兑现**)
  - `USEANET_WEIGHTED_DS=1` — 分辨率加权深监督(消融旋钮)
  - `USEANET_NO_MOE=1` — 关掉 MoE(消融基线)
- 工具:`tools/diag_val.py`(逐头诊断)、`tools/eval_tta.py`(TTA/后处理评估)。

## 复现当前最佳(BUSI)
```bash
USEANET_STRONG_AUG=1 conda run -n ubench1 python main.py --gpu 0 \
  --model USEANet --model_id 115 --base_dir hf_data/data/busi --dataset_name busi \
  --do_deeps 1 --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name moe_busi_strongaug_e250
# 再叠加 TTA:
conda run -n ubench1 python tools/eval_tta.py \
  --ckpt output/USEANet/busi/moe_busi_strongaug_e250/checkpoint_best.pth
```

## 待办(按论文价值排序)

### 1. MoE 消融(novelty 核心证据,最高优先)
有/无 MoE 在**完全相同条件**(强增强 250ep)下对比:
```bash
USEANET_STRONG_AUG=1 USEANET_NO_MOE=1 conda run -n ubench1 python main.py --gpu 0 \
  --model USEANet --model_id 115 --base_dir hf_data/data/busi --dataset_name busi \
  --do_deeps 1 --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name nomoe_busi_strongaug_e250
```
对照 `moe_busi_strongaug_e250`(0.667)。**若 MoE 无明显增益,论文叙事需调整**(转向鲁棒性/可解释性,而非纯精度)。
- 还需:专家数消融、硬/软门控消融、各物理专家贡献、`eff_experts` 监控。

### 2. 跨数据集(泛化 > BUSI 峰值)
对 bus / BUSBRA / tuscui 各跑强增强 250ep(注意各自 `--base_dir` / `--dataset_name` / dataset 类的目录结构;BUSBRA 用 `BUSBRADatasets`)。数据在 `hf_data/data.zip`(按需解压)。
- 另找一个**外部**超声集(BUSIS / UDIAT)做 zero-shot 跨数据集泛化(`--zero_shot_*` 参数)。

### 3. 多 seed 出显著性
当前都是单 seed(41)。论文需 ≥3 seed 报均值±std(加权深监督等"噪声内"结论也需多 seed 才能定论)。

### 4. 必备 baseline(在同一 split 重跑)
- **nnU-Net**(独立框架跑,别塞进 U-Bench)、一个 Mamba(VM-UNet/U-Mamba)、一个 SAM 类(MedSAM)。
- 详见 `docs/superpowers/research/2026-06-14-segmentation-sota-and-nnunet.md`。

### 5. 可选精度补强
- 加权深监督多 seed 复核;更长训练(300ep)。
- TTA 已证明有效,可考虑写进 `main.py` 的 `validate()`(目前仅在 `eval_tta.py` 里离线叠加)。

## 重要陷阱(别再踩)
- `do_deeps` 是 `type=bool`,传 `--do_deeps 0` 也会变 True;USEANet 必须深监督,保持 `--do_deeps 1`。
- 数据管线有"双重归一化"(Normalize 后又 /255),对预训练 PVT 致命——已由适配器 `x*255` 补偿。**别再"修"这个 /255**,否则要同步改 `x*255`。
- `checkpoint_best` 仅在 val_iou 提升时写;早期若 val 恒 0 则不会写(本轮 bug 的连带现象,已随头顺序修复解决)。
- 大数据/输出(`hf_data/ output/ result/`)已 gitignore,别提交。
