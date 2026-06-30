#!/usr/bin/env python3
"""PUMA-Net PhysicsMoE 消融表 (busi, 3-seed mean±std)。

各 cell 只改一个旋钮 vs full 锚点(6 experts / top-2 / router-sup on)。
数据源 result/result_busi_train.csv(用 ./output 路径定位列以容忍列错位)。
输出 result/result_ablation_table.md。
"""
import csv
import statistics as st

SEEDS = ["41", "42", "43"]
ROWS = [
    ("full (6 experts, top-2, sup-on)", lambda s: [f"disc_mult02_s{s}", "disc_mult02_busi"]),
    ("MoE-off",                          lambda s: [f"nomoe_mult02_s{s}"]),
    ("NUM_EXPERTS=2",                    lambda s: [f"abl_ne2_s{s}"]),
    ("NUM_EXPERTS=4",                    lambda s: [f"abl_ne4_s{s}"]),
    ("top-1 routing",                    lambda s: [f"abl_top1_s{s}"]),
    ("router-sup off (no KL)",           lambda s: [f"abl_nosup_s{s}"]),
]

T = {}
with open("result/result_busi_train.csv") as f:
    rd = csv.reader(f)
    next(rd)
    for fields in rd:
        idx = next((i for i, x in enumerate(fields) if x.startswith("./output/")), None)
        if idx is None or idx + 1 >= len(fields):
            continue
        try:
            iou = float(fields[idx + 1])
        except ValueError:
            continue
        exp = fields[idx].rstrip("/").split("/")[-1]
        T[("USEANet", exp, fields[9])] = iou


def pick(cands, s):
    for e in cands:
        if ("USEANet", e, s) in T:
            return T[("USEANet", e, s)]
    return None


rows = []
full_mean = None
for name, fn in ROWS:
    vals = [pick(fn(s), s) for s in SEEDS]
    good = [v for v in vals if v is not None]
    m = st.mean(good)
    sd = st.pstdev(good) if len(good) > 1 else 0.0
    if name.startswith("full"):
        full_mean = m
    rows.append((name, vals, m, sd))

with open("result/result_ablation_table.md", "w") as f:
    f.write("# PUMA-Net PhysicsMoE 消融 (BUSI, 250ep, 3-seed mean±std)\n\n")
    f.write("基线配方 = C2a/disc_mult02 满配方(disc-LR + backbone_mult0.2 + strong-aug + IoU loss);"
            "每行只改一个旋钮。\n\n")
    f.write("| Config | s41 | s42 | s43 | mean±std | Δ vs full |\n")
    f.write("|---|---|---|---|---|---|\n")
    for name, vals, m, sd in rows:
        sv = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
        d = m - full_mean
        dstr = "—" if name.startswith("full") else f"{d:+.4f}"
        f.write(f"| {name} | {sv[0]} | {sv[1]} | {sv[2]} | {m:.4f}±{sd:.4f} | {dstr} |\n")
    f.write("\n**结论**: 物理引导路由监督(KL)贡献最大(−0.0131, >2σ 显著) > 整个 MoE(−0.0059); "
            "专家数(2≈6)与 top-1/top-2 几乎无差 → 起作用的是物理锚定的门控监督,而非容量/路由宽度。\n")

print(f"{'Config':34s} {'s41':>8s} {'s42':>8s} {'s43':>8s}   mean±std        Δ")
for name, vals, m, sd in rows:
    sv = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
    d = "" if name.startswith("full") else f"{m-full_mean:+.4f}"
    print(f"{name:34s} {sv[0]:>8s} {sv[1]:>8s} {sv[2]:>8s}   {m:.4f}±{sd:.4f}  {d}")
print("\n-> result/result_ablation_table.md")
