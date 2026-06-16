# 异构物理专家(Heterogeneous Physics Experts)设计

> 2026-06-16 立。前置:C1a 消融显示 Physics MoE 峰值精度增益弱不显著
> (IoU Δ+0.0059,< seed 噪声阈值 0.01)。根因诊断:现 6 个专家**架构同构**——
> 除输入端固定物理预滤波核不同外,后端 learnable head 结构与容量**完全一样**
> (`1×1→BN→ReLU→3×3 depthwise→1×1`, channel=32),特化不充分。
> 本设计把专家**异构化**:让每个专家的结构/容量匹配它建模的超声物理,给予真正
> 不同的归纳偏置,目标是把 MoE 从"弱增益"救成可立的精度卖点。

## 目标与非目标

**目标:**
- 把同构专家头改为**物理驱动的异构专家**(理念B:可学物理初始核 + 异构结构 + 适度增容 + dropout 正则)。
- 改动**完全局部化在 `models/Hybrid/USEANet/moe/experts.py`**(+ `build_experts` 派发)。
  router / proxy / losses / physics_moe 的调用契约不变。
- 提供干净的"同构 vs 异构"消融能力(env 门控,默认关)。

**非目标:**
- 不改 router / proxy / aux-loss / 通道数契约(专家 I/O 仍 `[B,in,H,W]→[B,out=32,H,W]`)。
- 不改优化器 / `training_recipe`(per-group weight-decay 太侵入;正则只用 `Dropout2d`)。
- 不动专家数(仍 6)、top-k(仍 2)、MoE 层位(仍 x3/x4)。

## 约束(继承 MoE 论文 memory)

- BUSI 仅 ~452 train img → **过拟合是头号风险**。异构必须保持轻量 + 物理锚定。
- 现 MoE 实参:x3 = `PhysicsMoE(in=160, out=32)`、x4 = `PhysicsMoE(in=256, out=32)`;
  特征图 16×16 / 8×8(256 输入)→ **FLOPs 在任何方案下都可忽略,约束轴是参数量**。
- 现同构专家参数 ≈ 6.5k(x3)/9.6k(x4)/专家,6×2 层 ≈ 96k。

## 架构:6 个异构专家

每个专家 = **可学深度卷积(用对应物理核初始化)→ 异构结构 → 小投影头**,
保持 I/O 契约 `[B,in,H,W]→[B,out=32,H,W]` 与残差由 `PhysicsMoE.res` 外部提供(不变)。
可学预滤波核**用现有固定物理核初始化**(起点即物理锚定,训练中可微调 = 理念B 的"增容"核心)。

| 专家 | 物理 | 可学预滤波(物理初始化) | 头 channel | 额外结构 |
|---|---|---|---|---|
| **despeckle** | 斑点去噪 | 5×5 dw(init=box low-pass)+ 3×3 dw dilation2(init=box)双支相加 | **48** | `Dropout2d(0.1)` |
| **edge** | 边界 | 3×3 dw(init=Laplacian) | **48** | `Dropout2d(0.1)` |
| **shadow** | 声影竖向衰减 | **7×1 竖长 dw**(init=竖向梯度 vgrad) | 32 | 各向异性;`Dropout2d(0.1)` |
| **posterior** | 后方增强 | **5×1 竖向偏下非对称 dw**(init=below-sum vsum) | 32 | `Dropout2d(0.1)` |
| **contrast** | 低频全局对比 | 3×3 dw(init=gauss)+ **全局池化 SE 支** | **48** | `Dropout2d(0.1)` |
| **hf** | 高频纹理 | 3×3 dw(init=sharpen)+ 3×3 dw dilation2 双尺度 | 32 | `Dropout2d(0.1)` |

**设计要点:**
- **despeckle / contrast**:要大感受野 / 全局上下文 → 大核 / 膨胀 / 全局池化 SE。
- **edge / hf**:高频 → 小核;edge 升 channel 容纳方向多样性,hf 用双尺度膨胀核当微型小波。
- **shadow / posterior**:超声里这两类退化是**竖向各向异性**(沿声束方向)→ 竖长 / 竖向偏下非对称核,
  这是同构 3×3 给不了的归纳偏置,也是异构最强的物理动机点。
- 头结构统一为 `learnable_prefilter → 1×1(in→ch) → BN → ReLU → 3×3 dw(ch) → Dropout2d(0.1) → 1×1(ch→out)`,
  各专家差异在 **prefilter 形状/初始化 + ch + 是否带 SE/双支**。

**参数预算:** Δ ≈ +130k(总模型 ~3.66M → ~3.79M,+3.5%);FLOPs <0.5%。

## env 门控(与项目惯例一致,默认关)

- `USEANET_HETERO_EXPERTS=1` → 启用异构专家;**默认(未设)走现同构 `_AnchoredExpert`**。
- 保住现 0.709 配方复现性;天然给出"同构 vs 异构"干净消融(同 `USEANET_NO_MOE`/`STRONG_AUG` 风格)。
- 门控读取点:`build_experts`(按 env 派发到同构或异构专家工厂),`PhysicsMoE.__init__` 调用不变。

## 接口契约(不变)

- `build_experts(in_channel, out_channel, channel=32) -> nn.ModuleList`,长度 = `len(EXPERT_NAMES)`,
  顺序与 `EXPERT_NAMES` 一致(router gate 索引依赖此顺序)。
- 每个专家 `forward(x: [B,in,H,W]) -> [B,out,H,W]`。
- `PhysicsMoE.forward` 的 `out = Σ_e expert(x) * gate[:,e]` 组合逻辑不动。

## 测试

仿现有 `tests/` MoE 测试风格(pytest,合成输入):
1. **形状契约**:每个异构专家 `forward` 输出 `[B,32,H,W]`,x3(in=160)/x4(in=256)两组尺寸。
2. **可学核物理初始化**:构造后专家 prefilter 权重 == 对应物理核(init 正确性)。
3. **门控等价性**:`build_experts` 在 env 关时返回同构专家、开时返回异构专家,数量/顺序均 = `EXPERT_NAMES`。
4. **端到端**:`USEANET_HETERO_EXPERTS=1` 下 `PhysicsMoE` 前向 + `aux_loss` 不报错,`eff_experts` 在 [1,6]。
5. **梯度流**:异构专家可学核 `requires_grad` 且反传后有非零梯度。
6. **参数预算回归**:开启后总专家参数 < 同构的 3×(防止误增容爆参数)。

## 监控护栏(训练期)

- 盯 `eff_experts`(健康带 2–4)+ val 曲线;若过拟合(train-val gap 扩大 / 某专家长期休眠)
  → 回退理念A(prefilter 改回固定、撤 channel 增容,只保异构形状)。
- seed41 单点先验:异构 vs 同构在统一最优配方下对比 IoU,过 seed 噪声(~0.01)才算异构有效。

## 验证配方(与 C 路线统一底座一致)

```bash
USEANET_HETERO_EXPERTS=1 USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 \
USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  conda run -n ubench1 python main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name hetero_s41
```

对照组 = A 路线已有同构 `disc_mult02_s41`(IoU 0.7086±0.0043 的 seed41 单点)。
