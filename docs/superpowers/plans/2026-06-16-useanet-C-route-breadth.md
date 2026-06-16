# C 路线 — 转广度(论文骨架)执行清单

> 2026-06-16 立。前置:B(损失=iou)、A(`backbone_lr_mult=0.2`)已收尾,BUSI 精度稳在
> **IoU ~0.709 / Dice ~0.79**,结构旋钮用尽。C 不再追单点精度,而是攒一篇可投稿论文需要的
> 三类证据:**① 消融证卖点 · ② 跨数据集证泛化 · ③ baseline 证竞争力**。
> 当前最优配方(所有 C 训练的统一底座):
> ```bash
> USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
> conda run -n ubench1 python main.py --gpu 0 --model USEANet --model_id 115 \
>   --base_dir <DATA> --dataset_name <NAME> --do_deeps 1 \
>   --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
>   --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed <S> --exp_name <EXP>
> ```
> 多卡并行钉物理卡:前缀 `CUDA_VISIBLE_DEVICES=N`、参数仍用 `--gpu 0`(`--gpu N>0` 有 bug 卡死)。

---

## 优先级总览

| 块 | 内容 | 直接能跑? | 论文权重 | 建议顺序 |
|---|---|---|---|---|
| **C1a** | 有/无 MoE 对比(novelty 命门) | ✅ `USEANET_NO_MOE` | ★★★ 最高 | **1** |
| **C2a** | 跨超声集复现(bus/BUSBRA/tuscui) | ✅(需确认 dataset 类/目录) | ★★★ | **2** |
| **C3a** | Mamba baseline(VMUNet 等,同 split) | ✅ U-Bench 自带 | ★★ | **3** |
| C1b | 专家数 / top-k / 硬软门控消融 | ⚠️ 需先写 env 开关 | ★★ | 4(代码任务) |
| C2b | 外部集 zero-shot 泛化 | ⚠️ 需确认 `--zero_shot_*` | ★★ | 5 |
| C3b | nnU-Net baseline | ⚠️ 独立框架,工程量大 | ★★ | 6 |
| C3c | SAM 类(MedSAM)baseline | ⚠️ repo 没有,需引入 | ★ | 7(可选) |

**最小可投稿集 = C1a + C2a + C3a**(都能直接跑);C1b/C2b/C3b 加分;C3c 可选。

---

## C1 — MoE 消融(novelty 核心证据)

### C1a. 有/无 MoE(✅ 直接跑,最高优先)
**目的:** 证明 Physics MoE 真带来增益——否则论文卖点不成立(若无增益,叙事转鲁棒性/可解释性)。
**做法:** 完全相同配方下 3-seed(41/42/43)对比 `USEANET_NO_MOE=1` vs 默认(有 MoE)。
有 MoE 端 = A 路线已有的 `disc_mult02_s{41,42,43}`(IoU 0.7086±0.0043)可直接复用,**只需补跑无 MoE 三 seed**:
```bash
# 无 MoE,seed 41/42/43,其余同最优配方,exp_name nomoe_mult02_s4x
USEANET_NO_MOE=1 USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  ... --seed 41 --exp_name nomoe_mult02_s41
```
**判定:** 有 MoE 均值 − 无 MoE 均值,超 seed 噪声(~0.01)才算 MoE 有效。

### C1b. 专家数 / top-k / 门控消融(⚠️ 需先写 env 开关)
现状:`moe/router.py` `num_experts=6,k=2`、`moe/physics_moe.py` 均为**构造参数,未接 env**。
要做需先加开关(小代码任务,仿 `training_recipe.py` 风格):
- `USEANET_MOE_EXPERTS`(扫 4/6/8)、`USEANET_MOE_TOPK`(扫 1/2/3)、`USEANET_MOE_HARD`(硬 vs 软门控)。
然后各 3-seed 扫。**另:`eff_experts` 监控已有**(`__init__.py:94` 暴露 metric),可直接记录利用率;
各物理专家贡献分析需另写小工具(关单个专家看掉点)。

---

## C2 — 跨数据集(泛化 > BUSI 峰值)

### C2a. 同框架跨超声集复现(✅ 直接跑)
对 `bus` / `BUSBRA` / `tuscui`(均超声乳腺)各跑最优配方。**⚠️ 跑前确认每集的
`--base_dir` / `--dataset_name` / dataset 类**(BUSBRA 用 `BUSBRADatasets`,目录结构与 busi 不同;
`tuscui` 在 `hf_data/data.zip` 内,需先解压)。建议先各跑 **seed41 单点**确认管线通,再补 3-seed。
```bash
# 例:bus(确认 dataset 类后)
... --base_dir hf_data/data/bus --dataset_name bus --seed 41 --exp_name cross_bus_s41
```
**产出:** 一张「方法 × 数据集」的 IoU/Dice 表,证明配方不是只在 BUSI 过拟合。

### C2b. 外部集 zero-shot(⚠️ 需确认参数)
训练集训、完全未见的外部超声集(如 BUSIS/UDIAT)测,走 `--zero_shot_*` 参数(先 grep main.py 确认用法)。

---

## C3 — Baseline 对标(同 split 重跑)

### C3a. Mamba 类(✅ U-Bench 自带,最省事)
`models/Mamba/` 已有 VMUNet / VMUNetV2 / Swin_umamba / MambaUnet 等。同 `--base_dir busi` 同 split
跑(`--model VMUNet` 等,`model_id` 查 `models/model_id.json`)。是性价比最高的 baseline。

### C3b. nnU-Net(⚠️ 独立框架)
按惯例**别塞进 U-Bench**,独立装 nnU-Net 框架、转 BUSI 为其数据格式跑。工程量最大但审稿常要。
参考 `docs/superpowers/research/2026-06-14-segmentation-sota-and-nnunet.md`。

### C3c. SAM 类 MedSAM(⚠️ repo 无,可选)
需引入 MedSAM 权重与推理代码。优先级最低。

---

## 执行节奏建议
1. **先 C1a + C2a(seed41 单点) + C3a 起一批** —— 一两轮多卡并行就能拿到论文骨架的核心三表。
2. 看 C1a 结果定叙事:MoE 有增益 → 主打精度+物理可解释;无增益 → 转鲁棒/泛化。
3. 再决定要不要投入 C1b(写开关)/ C3b(nnU-Net)这些重活。
4. 体量大、跨多数据集×多 seed,建议**每块单独会话**执行,避免单会话上下文/成本爆。
