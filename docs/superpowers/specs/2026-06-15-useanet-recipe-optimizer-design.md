# USEANet 配方阶梯 · 优化器组(判别式LR / warmup / EMA / AdamW) — 设计 Spec

> 日期:2026-06-15 · 仓库 U-Bench · 分支 `feat/useanet-recipe-optim`
> 上游:配方阶梯方案 A(保持 PVT-B0、锁 256、单模型、多 seed gate)。本 spec 落地阶1+阶2+阶4(损失阶3、多尺度 TTA 阶5 另开)。

## 1. 目标与范围

把"优化器配方"三根杠杆做成一个隔离模块,全部 **env 门控、默认关**,用于在 BUSI 强增强 250ep 上把单模型 val_IoU 从 0.682 往 0.79+ 推:
- **阶1a 判别式 LR**:预训练 backbone 用更低 LR(默认 0.1×),新初始化的 MoE/decoder 用全 LR。
- **阶1b warmup**:前 N 个 epoch 线性 warmup,再接原 poly 衰减。
- **阶2 权重 EMA**:验证与 best 存盘用 EMA 影子权重。
- **阶4 AdamW**:可选用 AdamW 替 SGD(沿用判别式 LR 分组)。

**铁律(最高优先):env 全关时,`main.py` 行为与改动前逐字节等价**(其余模型零影响)。

**不做(YAGNI):** 不改损失、不改分辨率、不动数据管线、不做集成、不改其他模型的默认训练。

## 2. Env 开关(默认全关)

| 变量 | 默认 | 作用 |
|---|---|---|
| `USEANET_DISC_LR` | off | 开判别式 LR(backbone/rest 分组) |
| `USEANET_BACKBONE_LR_MULT` | `0.1` | backbone 组 LR = `base_lr × 此值`(仅 DISC_LR 开时生效) |
| `USEANET_WARMUP_EPOCHS` | `0` | 线性 warmup 的 epoch 数(0=无 warmup) |
| `USEANET_EMA` | off | 开权重 EMA |
| `USEANET_EMA_DECAY` | `0.999` | EMA 衰减 |
| `USEANET_ADAMW` | off | 用 AdamW 替 SGD |

> 命名沿用既有 `USEANET_*` 约定。这些是 USEANet 实验旋钮;虽然下文 helper 由共享 `main.py` 调用,但全关时是 passthrough,故对其他模型无副作用。

## 3. 模块:`models/Hybrid/USEANet/training_recipe.py`(纯逻辑,可测)

### 3.1 `build_optimizer(model, base_lr) -> (optimizer, group_base_lrs)`
- **全关(默认)**:返回 `optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)` 与 `group_base_lrs=[base_lr]` —— 与现 `main.py:269` **逐字节一致**。
- `USEANET_DISC_LR` 开:把参数按名字前缀分成 **backbone 组**(名字含 backbone 前缀,见 3.4)与 **rest 组**;两组 base_lr 分别为 `base_lr×mult`、`base_lr`。`group_base_lrs` 与 `optimizer.param_groups` 顺序对齐(返回 `[backbone_lr, base_lr]`)。
- `USEANET_ADAMW` 开:用 `optim.AdamW(groups, lr=base_lr, weight_decay=0.0001)`,分组同上(未开 DISC_LR 时为单组 `model.parameters()`)。
- backbone 组为空(非预训练/找不到前缀)时回退为单组 + 警告,不崩。

### 3.2 `lr_at(group_base_lr, iter_num, max_iterations, warmup_iters) -> float`
- `warmup_iters>0` 且 `iter_num<warmup_iters`:线性 `group_base_lr × (iter_num+1)/warmup_iters`。
- 否则:poly `group_base_lr × (1 - (iter_num-warmup_iters)/(max_iterations-warmup_iters))**0.9`。
- **`warmup_iters=0` 时退化为 `group_base_lr×(1-iter_num/max_iterations)**0.9`**,与现 `main.py:345` 逐字节一致。

### 3.3 `warmup_iters(iters_per_epoch) -> int`
- 读 `USEANET_WARMUP_EPOCHS`(默认 0)× `iters_per_epoch`。

### 3.4 backbone 前缀识别
- USEANet 适配器里 PVT-B0 backbone 子模块的参数名前缀(实现时从 `model.named_parameters()` 实查确定,例如 `backbone.` / 适配器实际属性名)。spec 不臆测,plan 任务第一步即打印 named_parameters 定位前缀并写进常量 `BACKBONE_PARAM_PREFIXES`。

### 3.5 `ModelEMA`
- `__init__(model, decay)`:深拷贝一份 `state_dict` 影子(detach、跟随 model 设备)。
- `update(model)`:`shadow = decay*shadow + (1-decay)*param`(对 float 参数;buffer 直接 copy 最新)。
- `store(model)` / `copy_to(model)` / `restore(model)`:存当前真实权重 → 把影子灌入 model → 训练继续前还原。
- `state_dict()`:返回影子,用于 best 存盘。

## 4. `main.py` 接入(全部 env 门控,默认 passthrough)

在 `train()` 内**局部 import** 该模块(避免共享文件无条件耦合 USEANet 路径):
1. `:269` → `optimizer, group_base_lrs = build_optimizer(model, base_lr)`。
2. 取 `warmup = warmup_iters(len(trainloader))`;`ema = ModelEMA(model, decay) if USEANET_EMA else None`。
3. `optimizer.step()` 后:`if ema: ema.update(model)`。
4. LR 调度 `:345-347` → `for pg, gbase in zip(optimizer.param_groups, group_base_lrs): pg['lr'] = lr_at(gbase, iter_num, max_iterations, warmup)`。
5. 验证块 `:353` 前:`if ema: ema.store(model); ema.copy_to(model)`。验证 + best 存盘(`:400-417`)用 EMA 权重;best 之后、`checkpoint_final`(`:430`,存真实训练权重+optimizer 供 resume)之前:`if ema: ema.restore(model)`。

> 全关时:`build_optimizer` 返回原 SGD 单组、`warmup=0`、`ema=None`,步骤 3/5 是 no-op,步骤 4 退化为原公式 → **行为不变**。

## 5. 健壮性 / 正确性
- best ckpt 在 EMA 开时保存 **EMA 权重**(评测/TTA 用它);`checkpoint_final` 始终保存 **真实训练权重 + optimizer**(resume 正确)。
- 分组覆盖完整:backbone∪rest = 全部 `requires_grad` 参数,无遗漏无重复(测试断言)。
- AdamW + 判别式 LR 同开时分组一致。

## 6. 测试(pytest)
- `lr_at`:`warmup_iters=0` 时多个 `iter_num` 上等于原 poly 公式(逐点);`warmup_iters>0` 时 0→peak 线性、之后 poly 单调降、端点连续。
- `build_optimizer`:全关 → 单组、SGD、lr/momentum/wd 与原值一致、`group_base_lrs==[base_lr]`;DISC_LR 开 → 两组且并集=全参数(数量与去重断言)、backbone 组 lr=base×mult;ADAMW 开 → optimizer 类型为 AdamW。用一个小 `nn.Module`(含命名子模块模拟 backbone 前缀)即可,无需真模型/GPU。
- `ModelEMA`:`update` 后影子 = `decay*old+(1-decay)*new`(已知数值);`copy_to`/`restore` 往返后 model 权重复原。
- **回归**:构造一个无任何 env 的最小场景,断言 `build_optimizer` 产出的 optimizer 的 param_groups 结构/超参与手写 `optim.SGD(...)` 等价,`lr_at(base,i,N,0)==base*(1-i/N)**0.9`。

## 7. 足迹
- 新增:`models/Hybrid/USEANet/training_recipe.py`、`tests/test_training_recipe.py`。
- 改动:`main.py` `train()`(优化器构造、LR 调度、EMA 接入;均门控)。
- 依赖:无新增。
- 复现命令(阶1+阶2 示例):
```bash
USEANET_DISC_LR=1 USEANET_WARMUP_EPOCHS=5 USEANET_EMA=1 USEANET_STRONG_AUG=1 \
conda run -n ubench1 python main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name recipe_busi_s41
```

## 8. 测量 gate(实验阶段,非本次编码范围)
- 每根杠杆 3 seed(41/42/43)对照 0.682 基线,均值增益 > seed 间 std 才留。本 spec 只交付**代码 + 开关**;keep/drop 在 GPU 跑后定。
