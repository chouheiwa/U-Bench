# USEANet 损失工程(B 路线)设计 spec

> 日期:2026-06-15 · 路线 B(B→A→C 的第一步)· 分支建议:`feat/useanet-loss`
> 环境:conda `ubench1`(torch 2.7,2× RTX 2080 Ti,用 `--gpu 0`)
> 测试:`conda run -n ubench1 python -m pytest tests/ -q`

## 1. 背景与目标

USEANet 的损失是模型隔离的,定义在 `models/Hybrid/USEANet/usea_loss.py`:
`structure_loss` = **边界加权 wBCE(fg) + wIoU(fg) + 0.8·wBCE(bg)**,由 `__init__.py`
的 `deep_supervision_loss` 套在 4 个深监督尺度上,再叠加 MoE 辅助损失。`main.py`
只调用 `model.deep_supervision_loss(...)`,不感知内部结构。

B 路线是路线图里唯一没碰过的结构杠杆——损失函数。目标:在**不碰共享 `main.py`、
不碰其他模型**的前提下,引入可切换的区域项,检验 nnU-Net 同款 Dice、以及
focal-Tversky(治小病灶 FN 漏检)能否超过现有 wIoU。

**新基线**(已合并 main,见 memory `[[useanet-recipe-optim]]`):
DISC_LR + strong-aug + 250ep → BUSI **IoU 0.7085 / Dice 0.7923**(3-seed 均值)。

## 2. 设计决策(已敲定)

- **B1 形态**:保留 USEANet 脚手架(边界权重 `weit` + 背景分支 `+0.8·wBCE(bg)`),
  前景项统一写成 `加权wBCE + 区域项`,**只把区域项**在 `wIoU / wDice / focal-Tversky`
  间切换。分类锚(加权 wBCE)不变,消融变量单一。
  - 排除 A(直接替换、绕开 weit/bg):放弃保留 USEANet 既有归纳偏置。
  - 排除 B2(整块替换前景项):同时动「去 wBCE」和「换区域项」两个变量,难归因。
- **focal-Tversky 超参**:文献默认 **α=0.3, β=0.7, γ=4/3**,env 可调,B 阶段只筛单点
  (不扫网格)。β>α 以更狠惩罚假阴(小病灶漏检)。
- **改动范围**:全部收在 `usea_loss.py`,`structure_loss` 签名不变,`__init__.py` /
  `main.py` 零改动。

## 3. 损失结构(B1)

前景项统一形态:
```
fg_loss = 加权wBCE(fg) + 区域项(fg)        # 区域项三选一
total   = fg_loss + 0.8 · 加权wBCE(bg)      # 背景分支不动
```
`weit`(边缘 ±5× 惩罚)、背景分支、`.mean()` 规约——全保留不变。
区域项三选一,**全部用 `weit` 加权、`+1` 平滑**,与现有 wIoU 同款数值约定
(`p = sigmoid(pred)`;`inter / FP / FN` 均按 `weit` 加权后在 `dim=(2,3)` 求和):

| 区域项 | 公式 |
|---|---|
| `iou`(默认 = 现状) | `1 − (inter+1)/(union−inter+1)`,`union = Σ((p+m)·weit)` |
| `dice` | `1 − (2·inter+1)/(Σ(p·weit) + Σ(m·weit) + 1)` |
| `focal_tversky` | `(1 − (inter+s)/(inter + α·FP + β·FN + s))^(1/γ)`,`s=1` |

其中 `inter = Σ(p·m·weit)`,`FP = Σ(p·(1−m)·weit)`,`FN = Σ((1−p)·m·weit)`。

## 4. 开关面(沿用 `USEANET_*` 风格,默认全关 = 字节级等价现状)

- `USEANET_LOSS_REGION` ∈ `{iou(默认), dice, focal_tversky}`
- `USEANET_FT_ALPHA=0.3` · `USEANET_FT_BETA=0.7` · `USEANET_FT_GAMMA=1.333`
  (仅 `focal_tversky` 读取;带默认值)
- 非法 `USEANET_LOSS_REGION` 值:fallback 到 `iou` + 打一次 warning(与现有 env
  开关容错一致)。

**硬不变量**:`USEANET_LOSS_REGION` 未设或 `=iou` 时,`structure_loss` 输出与当前
实现**逐元素相同**——保证默认路径 / 其他模型零影响。

## 5. 实现落点

`usea_loss.py` 内新增三个区域项小函数(纯函数,接 `p, mask_fg, weit` 等已算好的张量):
`_region_iou`、`_region_dice`、`_region_focal_tversky`。`structure_loss` 内部按 env
解析一次区域项 + FT 超参,dispatch 调用;wBCE(fg)、wBCE(bg)、weit、规约逻辑保持原样。
保持 `structure_loss` 自包含,`__init__.py` 的适配器无需改签名。

## 6. 测试计划(TDD,纯 CPU 小张量,不需 GPU/BUSI)

1. **零影响不变量**:`USEANET_LOSS_REGION` 未设 与 `=iou` 时,`structure_loss` 输出与
   「写死旧实现」参照 `torch.allclose` 逐元素相等。
2. **dice 路径**:完美预测→区域项≈0;全错→接近上界;有限、`loss.backward()` 不 NaN。
3. **focal_tversky 路径**:同上的数值健全性;**方向性**——构造一个「漏检(FN)」样本,
   确认 β=0.7 的 loss > α=0.7 的 loss,证明小病灶 FN 惩罚生效;γ 读取生效。
4. **env 解析**:非法值 fallback 到 `iou`(+ 一次 warning)。
5. **端到端**:经 `model.deep_supervision_loss(...)` 跑三种区域项,确认与 MoE 辅助、
   多尺度、背景分支组合后仍为可反传 scalar。

## 7. 实验协议(seed41 先筛)

固定 DISC_LR + strong-aug + 250ep,只切 `USEANET_LOSS_REGION`:
```bash
USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.1 USEANET_STRONG_AUG=1 \
USEANET_LOSS_REGION=dice \   # 或 focal_tversky
conda run -n ubench1 python main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name disc_dice_busi
```
**判定**:seed41 上 dice / focal_tversky 任一明显超过 0.7085(超出 seed 噪声 ~0.01)
→ 晋级 3-seed(41/42/43)复核;均值±std 赢基线才 KEEP,写回 `NEXT-STEPS-useanet-moe.md`。
没赢则 DROP、保留 iou,直接进 A 路线(调 `backbone_lr_mult`)。

## 8. 不做(YAGNI)

- 不在 B 阶段扫 α/β/γ 网格(赢家阶段或 A 路线再调)。
- 不碰 `backbone_lr_mult`(那是 A 路线)。
- 不动 TTA / EMA / warmup(已 DROP)。
- 不改 `main.py` / 共享损失 / 其他模型路径。
