# USEANet-MoE — 下一步待办(新会话接续用)

> 更新:2026-06-14 · 已合并到 main。分支 `feat/useanet-moe`。
> 环境:conda `ubench1`(torch 2.7,2× RTX 2080 Ti,用 `--gpu 0`)。测试:`conda run -n ubench1 python -m pytest tests/ -q`(当前 74 passed)。

---

## 2026-06-15 更新:优化器配方扫描结论 + 下一步路线(B→A→C)

**已合并 main**(merge `48d0b48`):新增 `models/Hybrid/USEANet/training_recipe.py`,4 个 env 开关(默认全关,其他模型零影响,详见 memory [[useanet-recipe-optim]]):`USEANET_DISC_LR`(+`USEANET_BACKBONE_LR_MULT`,默认 0.1)、`USEANET_WARMUP_EPOCHS`、`USEANET_EMA`(+`USEANET_EMA_DECAY`)、`USEANET_ADAMW`。

**3-seed 扫描结果(41/42/43,250ep,strong-aug,无 TTA,BUSI):**

| 配置 | IoU mean±std | Dice | 判定 |
|---|---|---|---|
| 对照(仅 strong-aug) | 0.6481 ± 0.017 | — | 基线 |
| **+DISC_LR** | **0.7085 ± 0.009** | **0.7923** | **KEEP**(gain +0.060 ≫ std;最佳 seed42 0.720/0.805) |
| 三件套(+warmup5+EMA) | 0.7009 ± 0.007 | — | DROP(均值反低于 DISC_LR 单项) |

**取舍已定:只留 `USEANET_DISC_LR=1`;EMA、warmup、TTA 全 DROP**(EMA 单独 −0.003;TTA 在强化后的模型上 plain ≥ tta,不再有用)。新最佳 **IoU 0.7085 / Dice 0.7923(均值),最佳 seed Dice 0.805**,踏入文献 SOTA 带(~0.80–0.83)下沿(旧最佳 0.682/0.773)。扫描脚本+日志在 gitignored `output/_sweep/`。

**正式复现(就这一个开关):**
```bash
USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.1 USEANET_STRONG_AUG=1 \
conda run -n ubench1 python main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name disc_busi
```

**下一步路线(顺序有依赖,别颠倒):**

- **B 先 — 损失工程(唯一没碰的结构杠杆)。✅ 代码已实现并合并 main**(merge `e9e7fe5`,2026-06-15;spec/plan 在 `docs/superpowers/{specs,plans}/2026-06-15-useanet-loss-engineering*`,memory `[[useanet-loss-engineering]]`)。B1 形态:`usea_loss.py` 的 `structure_loss` 保留边界权重+背景分支+加权 wBCE 分类锚,**只切区域项**:`USEANET_LOSS_REGION=iou`(默认=字节级等价旧实现)/`dice`/`focal_tversky`(α/β/γ 经 `USEANET_FT_*`,默认 0.3/0.7/(4/3),已 clamp 防 NaN)。`tests/test_usea_loss.py` 13 测试,全套 87 passed,默认全关对其他模型零影响。
  - **剩 GPU 验收(下一步):** 以 **DISC_LR strong-aug 250ep(IoU 0.7085/Dice 0.7923)为基线**,seed41 单点跑 `dice`/`focal_tversky`,超基线 ~0.01 才晋级 3-seed(41/42/43);均值±std 赢才 KEEP,结论写回本文件。命令在复现块基础上加 `USEANET_LOSS_REGION=dice`(或 `focal_tversky`)即可。
- **A 后 — 调 `backbone_lr_mult`。** 用 B 胜出的损失,再扫 0.05/0.1/0.2 定操作点。**为什么在 B 之后:** lr_mult 是操作点微调、依赖损失函数;先调 A 再换 B 会让 A 作废。
- **最后 C — 转广度(论文骨架,价值 > BUSI 峰值)。** 跨数据集(bus/BUSBRA/tuscui)复现 DISC_LR + 最佳损失;MoE 消融(专家数/硬软门控/各物理专家贡献/`eff_experts`);nnU-Net/Mamba(VM-UNet)/MedSAM baseline 同 split 重跑。详见下方原始待办 + `docs/superpowers/research/2026-06-14-segmentation-sota-and-nnunet.md`。

---

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
