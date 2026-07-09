#!/usr/bin/env python
"""P1.4 消融汇总: 标定物理路由 vs 手工代理,跨域迁移矩阵对比。

从 result/result_cross_dataset.csv 读取,按 exp_name 前缀分组:
  - calib_*              -> 标定组(本次新训 9 seed)
  - disc_mult02_*/c2a_*  -> 手工基线(PUMA 现配方)
对每 (source,target) 取 3-seed IoU 均值。源限 busi/bus/BUSBRA,
目标 busi/bus/BUSBRA/BrEaST。输出标定/手工矩阵 + Δ,并分 in-domain(对角)
与 cross-domain(非对角) 汇总平均 Δ —— 核心假设: 跨域 Δ > in-domain Δ。
"""
import csv
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "result", "result_cross_dataset.csv")
SRCS = ["busi", "bus", "BUSBRA"]
TGTS = ["busi", "bus", "BUSBRA", "BrEaST"]


def group_of(exp):
    if exp.startswith("calib_"):
        return "calib"
    if exp.startswith("disc_mult02") or exp.startswith("c2a_"):
        return "manual"
    return None


def load():
    # (group, source, target) -> {seed: iou}  (dedup: last row per seed wins)
    acc = defaultdict(dict)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            g = group_of(r["exp_name"])
            if g is None:
                continue
            s, t = r["source"], r["target"]
            if s not in SRCS or t not in TGTS:
                continue
            acc[(g, s, t)][r["seed"]] = float(r["iou"])
    # mean over seeds
    mean = {}
    for k, seedmap in acc.items():
        mean[k] = (sum(seedmap.values()) / len(seedmap), len(seedmap))
    return mean


def fmt_matrix(mean, group, title):
    print(f"\n=== {title} (3-seed IoU 均值) ===")
    print("源\\目标      " + "".join(f"{t:>10}" for t in TGTS))
    for s in SRCS:
        row = f"{s:<10}"
        for t in TGTS:
            v = mean.get((group, s, t))
            cell = f"{v[0]:.4f}" if v else "  --  "
            mark = "*" if s == t else " "
            row += f"{cell+mark:>10}"
        print(row)


def main():
    mean = load()
    fmt_matrix(mean, "manual", "手工代理基线 (PUMA)")
    fmt_matrix(mean, "calib", "标定物理路由 (本文)")

    print("\n=== Δ 矩阵 (标定 - 手工, 正=标定更好) ===")
    print("源\\目标      " + "".join(f"{t:>10}" for t in TGTS))
    indom_deltas, cross_deltas = [], []
    for s in SRCS:
        row = f"{s:<10}"
        for t in TGTS:
            c = mean.get(("calib", s, t))
            m = mean.get(("manual", s, t))
            if c and m:
                d = c[0] - m[0]
                row += f"{d:>+10.4f}"
                (indom_deltas if s == t else cross_deltas).append(d)
            else:
                row += f"{'--':>10}"
        print(row)

    def summ(name, ds):
        if not ds:
            print(f"  {name}: 无数据")
            return
        avg = sum(ds) / len(ds)
        wins = sum(1 for d in ds if d > 0)
        print(f"  {name}: 平均 Δ={avg:+.4f}  ({wins}/{len(ds)} 单元标定更好)")

    print("\n=== 核心假设检验 (标定物理量域不变 -> 跨域收益 > 同域) ===")
    summ("in-domain (对角)", indom_deltas)
    summ("cross-domain (非对角)", cross_deltas)
    if indom_deltas and cross_deltas:
        ai = sum(indom_deltas) / len(indom_deltas)
        ac = sum(cross_deltas) / len(cross_deltas)
        verdict = "支持" if ac > ai else "不支持"
        print(f"\n  跨域Δ({ac:+.4f}) {'>' if ac > ai else '<='} 同域Δ({ai:+.4f}) "
              f"-> 核心假设 [{verdict}]")


if __name__ == "__main__":
    main()
