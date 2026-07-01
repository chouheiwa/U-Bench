#!/usr/bin/env python3
"""汇总 method×dataset 主表: 10 baseline + PUMA(USEANet) + nnU-Net。

IoU 3-seed mean±std(全 12 方法可得);HD95 仅 10 baseline 可得
(PUMA/nnU-Net 的 HD95 尚未离线补算 -> 标 N/A)。

数据源:
  result/result_<ds>_train.csv  (best_iou, 经 ./output 路径定位以容忍列错位)
  result/result_nnunet.csv      (nnUNet best_iou)
  result/result_hd95.csv        (baseline HD95)
"""
import csv
import statistics as st

DATASETS = ["busi", "bus", "BUSBRA", "tuscui"]
SEEDS = ["41", "42", "43"]
BASELINES = ["U_Net", "AttU_Net", "ResNet34UnetPlus", "SwinUnet", "VMUNet",
             "H2Former", "CMU_Net", "CMUNeXt", "MSLAU_Net", "TransUnet"]

# PUMA(USEANet) 每数据集的 canonical exp 候选(按优先级)
PUMA_EXP = {
    "busi":   lambda s: [f"disc_mult02_s{s}", "disc_mult02_busi"],
    "bus":    lambda s: [f"c2a_bus_s{s}"],
    "BUSBRA": lambda s: [f"c2a_BUSBRA_s{s}"],
    "tuscui": lambda s: [f"c2a_tuscui_s{s}"],
}
# baseline 每 seed 的 exp 候选
BASE_EXP = lambda s: [f"mtx_s{s}", f"baseline_s{s}"]


def parse_train_csv(path):
    """返回 {(model, exp, seed): best_iou}。用 ./output 路径定位以容忍列数不一致。"""
    out = {}
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        for fields in rd:
            idx = next((i for i, x in enumerate(fields) if x.startswith("./output/")), None)
            if idx is None or idx + 1 >= len(fields):
                continue
            model = fields[0]
            seed = fields[9]
            exp = fields[idx].rstrip("/").split("/")[-1]
            try:
                iou = float(fields[idx + 1])
            except ValueError:
                continue
            # 同 (model,exp,seed) 保留最后一条(CSV 追加写,后者较新)
            out[(model, exp, seed)] = iou
    return out


def pick(table, model, candidates, seed):
    for exp in candidates:
        if (model, exp, seed) in table:
            return table[(model, exp, seed)]
    return None


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return "N/A", None, None
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}", m, s


# --- IoU ---
iou_tables = {ds: parse_train_csv(f"result/result_{ds}_train.csv") for ds in DATASETS}

# nnUNet IoU
nn_iou = {}
with open("result/result_nnunet.csv") as f:
    for r in csv.DictReader(f):
        nn_iou.setdefault((r["dataset"], r["seed"]), float(r["best_iou"]))

# HD95 baseline
hd95 = {}
with open("result/result_hd95.csv") as f:
    for r in csv.DictReader(f):
        hd95.setdefault((r["modelname"], r["dataset"], r["seed"]), float(r["hd95"]))


def hd95_pick(model, ds, seed):
    return hd95.get((model, ds, seed))


missing = []
rows = []  # (method, {ds: (iou_str, hd95_str)})

methods = BASELINES + ["USEANet", "nnUNet"]
for method in methods:
    cell = {}
    for ds in DATASETS:
        # IoU
        if method == "nnUNet":
            ivals = [nn_iou.get((ds, s)) for s in SEEDS]
        elif method == "USEANet":
            ivals = [pick(iou_tables[ds], "USEANet", PUMA_EXP[ds](s), s) for s in SEEDS]
        else:
            ivals = [pick(iou_tables[ds], method, BASE_EXP(s), s) for s in SEEDS]
        iou_str, _, _ = fmt(ivals)
        if any(v is None for v in ivals):
            missing.append(f"IoU {method}/{ds}: {ivals}")
        # HD95: 全 12 方法可得(baseline/PUMA/nnUNet 均在 result_hd95.csv,
        # modelname 与 method 同名; nnUNet 已 256-resize 口径)。
        hvals = [hd95_pick(method, ds, s) for s in SEEDS]
        hd_str, _, _ = fmt(hvals)
        if any(v is None for v in hvals):
            missing.append(f"HD95 {method}/{ds}: {hvals}")
        cell[ds] = (iou_str, hd_str)
    rows.append((method, cell))

# 友好显示名
DISP = {"USEANet": "PUMA-Net (ours)", "nnUNet": "nnU-Net"}

# --- 写 CSV ---
with open("result/result_main_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    header = ["method"]
    for ds in DATASETS:
        header += [f"{ds}_IoU", f"{ds}_HD95"]
    w.writerow(header)
    for method, cell in rows:
        line = [DISP.get(method, method)]
        for ds in DATASETS:
            line += [cell[ds][0], cell[ds][1]]
        w.writerow(line)

# --- 写 Markdown ---
with open("result/result_main_table.md", "w") as f:
    f.write("# 主结果表 — IoU (3-seed mean±std)\n\n")
    f.write("| Method | " + " | ".join(DATASETS) + " |\n")
    f.write("|" + "---|" * (len(DATASETS) + 1) + "\n")
    for method, cell in rows:
        f.write(f"| {DISP.get(method, method)} | "
                + " | ".join(cell[ds][0] for ds in DATASETS) + " |\n")
    f.write("\n# HD95 (3-seed mean±std, 256-px 空间, 空预测罚对角线~362)\n\n")
    f.write("> 全 12 方法齐全。baseline/PUMA 在 val_transform 的 256 空间原生推理; "
            "nnU-Net 原生输出为原图分辨率，pred+GT 均最近邻 resize 到 256 再算(口径可比)。\n\n")
    f.write("| Method | " + " | ".join(DATASETS) + " |\n")
    f.write("|" + "---|" * (len(DATASETS) + 1) + "\n")
    for method, cell in rows:
        f.write(f"| {DISP.get(method, method)} | "
                + " | ".join(cell[ds][1] for ds in DATASETS) + " |\n")

print("=== IoU main table ===")
print(f"{'Method':18s} " + " ".join(f"{d:>16s}" for d in DATASETS))
for method, cell in rows:
    print(f"{DISP.get(method, method):18s} " + " ".join(f"{cell[d][0]:>16s}" for d in DATASETS))

print("\n=== missing cells ===")
print("\n".join(missing) if missing else "(none — 全部 12 方法 4 数据集 IoU 齐全)")
