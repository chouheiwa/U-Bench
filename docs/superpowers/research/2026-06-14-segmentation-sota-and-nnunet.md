# 分割 SOTA 调研 & nnU-Net 能否集成到 USEANet-MoE

> 日期:2026-06-14 · 分支 `feat/useanet-moe`
> 背景:USEANet-MoE(PVT-B0 + 超声物理 MoE + 深监督),BUSI 当前 **val_IoU 0.623 / Dice 0.727**(100ep,两处 bug 修复后)。
>
> ⚠️ **数字可靠性**:下表 BUSI 的 Dice/IoU 多为"典型区间综合",**非逐篇核对的表格原值**。BUSI 的 split 协议混乱(normal 类是否剔除、划分比例、近重复泄漏、尺寸/TTA),跨论文不可直接比较。引用前必须逐篇复核。

---

## 1. 当前 SOTA 架构格局(2025–2026)

五大家族,彼此重叠:

| 家族 | 代表 | 2026 现状 |
|---|---|---|
| **混合 CNN-Transformer** | TransUNet、Swin-UNet、**CASCADE / MERIT / G-CASCADE**(MaxViT) | 小数据医学图最实用 SOTA;纯 ViT 增益见顶。本仓库已含 MERIT、G_CASCADE |
| **Mamba / 状态空间 SSM** | U-Mamba、VM-UNet(V2)、SegMamba、Swin-UMamba、LightM-UNet | 2024-25 最活跃;**到 2026 已是必备 baseline,本身不再是 novelty** |
| **SAM / 基础模型** | SAM2、MedSAM、MedSAM-2、SAM-Med2D/3D | 趋势主线,走 adapter/LoRA/prompt 微调。超声零样本弱(SAM Dice<0.65),需域适配 |
| **MoE 分割** | 多器官/多模态路由(零散) | **医学分割里仍稀疏 → USEANet-MoE 的 novelty 空间**。风险:路由要"有原理(物理/退化驱动)",不能是普通门控集成 |
| **nnU-Net / 扩散** | nnU-Net、MedSegDiff | nnU-Net 仍是"**必须打过**"的沉默强 baseline(详见 §3) |

### BUSI 参考区间(均需复核)

| 模型 | BUSI Dice | BUSI IoU | 年 | 来源(arXiv,待核) |
|---|---|---|---|---|
| U-Net | ~0.70–0.76 | ~0.58–0.63 | 2015 | 1505.04597 |
| Attention U-Net | ~0.74–0.78 | ~0.61–0.65 | 2018 | 1804.03999 |
| TransUNet | ~0.76–0.81 | ~0.64–0.69 | 2021 | 2102.04306 |
| Swin-UNet | ~0.77–0.81 | ~0.65–0.69 | 2022 | 2105.05537 |
| CASCADE / MERIT | ~0.80–0.83 | ~0.68–0.72 | 2023-24 | 2303.16892 |
| VM-UNet (Mamba) | ~0.78–0.82 | ~0.66–0.70 | 2024 | 2402.02491 |
| U-Mamba | ~0.78–0.82 | ~0.66–0.70 | 2024 | 2401.04722 |
| MedSAM(微调) | ~0.78–0.83 | ~0.66–0.71 | 2024 | 2304.12306 |
| SAM(零样本) | ~0.55–0.65 | ~0.40–0.50 | 2023 | 2304.02643 |
| **USEANet-MoE(本工作)** | **0.727** | **0.623** | 2026 | 本仓库 |

**定位**:比经典 U-Net 持平略高(体面),比 2025 混合/Mamba SOTA(~0.80–0.83 Dice)**低约 7–10 个 Dice 点**。强依赖 split——若为严格无泄漏的硬 split,差距部分是 split 假象。**结论:0.727 Dice 单靠精度撑不起 2026 论文,必须靠机制 + 鲁棒性 + 消融。**

### 2026 审稿人硬要求
- **必备 baseline(且在你自己的 split 上重跑)**:nnU-Net + 一个混合(TransUNet/MERIT)+ 一个 Mamba(VM-UNet/U-Mamba)+ 一个 SAM 类(MedSAM)。
- **跨数据集泛化**:再加一个超声集(BUSIS / UDIAT / 甲状腺等)。泛化赢 > BUSI 峰值。
- **消融必做**:专家数、硬/软门控、退化模拟器开关、各物理专家贡献。
- **诚实 split + 多 seed/CV + 标准差/显著性**。
- **效率(MoE 必问)**:激活参数 vs 总参数、FLOPs、延迟。

---

## 2. nnU-Net 到底是什么(关键澄清)

**nnU-Net 不是一种新架构,而是一套"自配置框架/训练配方"。** 它的网络就是一个朴素 U-Net(InstanceNorm + LeakyReLU + 深监督);其威力 100% 来自**自动化的数据指纹 → 预处理/拓扑/训练/后处理配方 + 重训练**:

1. **数据指纹 → 自配置**:由数据集的 spacing/强度分布/尺寸自动决定 patch size、归一化方案、网络层数。
2. **预处理**:重采样到目标 spacing(3D/CT 相关,2D 超声 PNG **不适用**);强度归一化——**非 CT 模态用逐图 z-score**(超声属此类)。
3. **架构**:按显存自动扩缩的朴素 U-Net(**非 transformer**)。
4. **训练**:1000 epoch、SGD momentum **0.99**、poly LR(0.01 起)、**Dice + CE** 损失、**重数据增强**(旋转/缩放/高斯噪声/高斯模糊/亮度/对比度/gamma/镜像)、**分辨率加权的深监督**、**5 折 CV**。
5. **推理**:高斯加权滑窗 + **TTA(镜像)** + **5 折集成** + 跨配置集成 + **自动后处理(最大连通域)**。

> 一句话:nnU-Net 强在"配方 + 重训 + 集成",不在"网络结构"。所以"集成 nnU-Net" ≠ 加一个模块,而是**借它的配方**。

---

## 3. 能否集成到 USEANet-MoE?——逐组件分析

USEANet-MoE 是 U-Bench 框架内的**固定架构**(PVT-B0 + MoE),而 nnU-Net 是独立框架。因此:

> **整体结论:不要把 nnU-Net 代码库塞进 U-Bench(高成本、低收益、且其自配置假设的是从零训练的朴素 U-Net,与"预训练 PVT + MoE"冲突)。但 nnU-Net 的训练/推理配方里有几样是高价值、可直接借的。**

| nnU-Net 组件 | 能否用到 USEANet-MoE | 价值 | 说明 |
|---|---|---|---|
| 自配置 / 数据指纹 | ❌ 不适用 | — | 假设从零训练朴素 U-Net,自动定层数/patch。我们用预训练固定 backbone,无从配 |
| 朴素 U-Net 架构 | ❌ 不用 | — | 我们的卖点正是 PVT+物理 MoE,不能换掉 |
| 逐图 z-score 归一化 | ⚠️ 谨慎 | 中 | 对从零模型好;**但预训练 PVT 期望 ImageNet 均值/方差**,二者冲突。**结论:保留 ImageNet 归一化**(我们刚修的 `x*255` 即为此),不要换 z-score |
| **重数据增强**(缩放/高斯噪声/模糊/亮度/对比度/gamma/弹性) | ✅ 易加 | **高** | U-Bench 现有 aug 极弱(仅 RandomRotate90+Flip+Resize)。这是**最高价值的可借项**,对 BUSI 通常 +几个 Dice。需 USEANet 专属 transform 或加开关,避免动全仓库 |
| **分辨率加权深监督** | ✅ 易加 | 中-高 | 我们 4 个 fg/bg 尺度目前**等权**;nnU-Net 按分辨率递减加权(粗尺度权重减半)。在 `deep_supervision_loss` 里给 fg5/fg4/fg3/fg2 配 `[1/8,1/4,1/2,1]` 之类权重,廉价且原理正 |
| Dice + CE 损失 | ➖ 已等价 | 低 | 我们的 `structure_loss`(wBCE+wIoU+bg)已是 PraNet 风格的同类物。可作为消融对照,不必替换 |
| poly LR / SGD momentum | ➖ 已有 | 低 | main.py 已 poly LR + SGD 0.9。可试 momentum 0.99 + 更长训练 |
| **TTA(推理镜像)** | ✅ 易加 | 中 | 在 `validate()` 里对输入做水平/垂直翻转、平均 logits,通常 +1–2 Dice。廉价 |
| **后处理(最大连通域)** | ✅ 易加 | 中 | BUSI 多为单病灶,取最大连通域去除碎块,常 +1 Dice。廉价 |
| **5 折 CV** | ✅ 协议层 | 高(论文必需) | 不改模型,改评估协议。审稿人要的"诚实 split + 多折"正是这个 |
| 5 折 / 多配置集成 | ⚠️ 可选 | 中 | 涨点但增推理成本,且会"稀释" MoE 的机制叙事。主表可不做,附录可加 |
| nnU-Net 作为 **baseline** | ✅ 独立跑 | 高(论文必需) | **独立**用官方 nnU-Net 在你的同一 split 上训练评估,作对照。**不要**塞进 U-Bench |

---

## 4. 建议优先级(围绕论文 novelty)

精度差距要靠"配方"补一点,但论文骨架仍是 **MoE 机制 + 鲁棒性 + 消融**。建议顺序:

**A. 论文骨架(必做,优先)**
1. `USEANET_NO_MOE` 有/无 MoE 公平对比(开关已就绪,尺度修在共享 forward,baseline 也享有)。
2. 第二个超声数据集(BUS / BUSBRA / tuscui 已在仓库;另找 BUSIS/UDIAT 做跨数据集泛化)。
3. MoE 消融:专家数、硬/软门控、各物理专家贡献。
4. **nnU-Net 独立 baseline** + 一个 Mamba(VM-UNet/U-Mamba)+ 一个 SAM 类(MedSAM),全部在你的 split 上重跑。
5. 5 折 CV + 多 seed + 标准差。

**B. 借 nnU-Net 配方补精度(性价比从高到低)**
1. **重数据增强**(USEANet 专属 transform / 开关)—— 预期增益最大。
2. **分辨率加权深监督**(改 `deep_supervision_loss`)。
3. **推理 TTA(镜像平均)** + **后处理(最大连通域)**。
4. 更长训练(200–300ep)+ momentum 0.99 试探。

**C. 不要做**
- 不把 nnU-Net 框架/数据格式塞进 U-Bench。
- 不用逐图 z-score 替换 ImageNet 归一化(与预训练 PVT 冲突)。
- 不靠堆集成去刷 BUSI 峰值(稀释机制叙事,且 BUSI 已饱和)。

---

## 5. 待验证 / 风险
- 所有 BUSI 数字需逐篇核对(split 不一致)。
- "重数据增强一定涨点"需在 BUSI 上实测(超声斑点噪声下,某些 aug 如过强高斯噪声可能伤害)。
- z-score vs ImageNet 归一化:若改用非预训练 backbone,结论反转——届时 z-score 可能更优。
- nnU-Net baseline 的 split 必须与 USEANet-MoE 完全一致才公平。

---

## 附:来源(arXiv ID,引用前复核)
U-Net 1505.04597 · Attention U-Net 1804.03999 · TransUNet 2102.04306 · Swin-UNet 2105.05537 · CASCADE 2303.16892 · VM-UNet 2402.02491 · U-Mamba 2401.04722 · MedSAM 2304.12306 · SAM 2304.02643 · BUSI 数据集:Al-Dhabyani et al., *Data in Brief* 2020。nnU-Net:Isensee et al., *Nature Methods* 2021(arXiv 1809.10486)。
