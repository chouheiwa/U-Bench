#!/usr/bin/env python
"""统一测量模型效率指标: Params / FLOPs / FPS。

与数据集解耦——只喂 torch.randn 假输入,形状对齐训练口径 (1,C,H,W) 即可,
因为这三个指标只取决于张量形状,与像素数值无关。

实例化复用 models.build_model,确保和 main.py 训练时的构造口径完全一致。

用法:
  # 默认对比集 (baseline + USEANet + H2Former)
  python tools/measure_efficiency.py

  # 指定模型
  python tools/measure_efficiency.py --models U_Net H2Former USEANet

  # 全部 model_id.json 里的模型 (慢,部分可能 OOM/报错,会跳过并记录)
  python tools/measure_efficiency.py --all

  # USEANet 需要 PVT backbone 权重时
  python tools/measure_efficiency.py --models USEANet --pretrained /path/to/pvt_v2_b0.pth
"""
import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

# 让 "from models import build_model" 可用 (脚本在 tools/ 下运行)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model  # noqa: E402

# 默认对比集: 已有 baseline_s41 结果的古典模型 + 自研 USEANet + 最强对手 H2Former
DEFAULT_MODELS = [
    "U_Net", "AttU_Net", "ResNet34UnetPlus", "SwinUnet", "VMUNet",
    "H2Former", "USEANet",
]

# 部分模型的输入尺寸在仓库里被写死,与全局 --img-size 不同。
# 必须按其原生尺寸喂输入,否则会触发 patch_embed 的 size 断言。
# (这些模型的 FLOPs 因此不在同一分辨率,报表时需注明。)
PER_MODEL_INPUT_SIZE = {
    "SwinUnet": 224,  # swinunet() 硬编码 img_size=224
}


def make_config(model_name, num_classes, input_channel, img_size, pretrained):
    """构造 build_model 需要的最小 config (对齐 main.py 的 argparse Namespace)。"""
    cfg = SimpleNamespace(
        model=model_name,
        num_classes=num_classes,
        input_channel=input_channel,
        img_size=img_size,
        model_id=0,
        do_deeps=0,
        pretrained_model_path=pretrained,
    )
    return cfg


def build(model_name, num_classes, input_channel, img_size, pretrained, device):
    cfg = make_config(model_name, num_classes, input_channel, img_size, pretrained)
    # 镜像 main.py: 只在提供了权重路径时才传 pretrained_model_path
    pre = {"pretrained_model_path": pretrained} if pretrained else {}
    model = build_model(
        cfg, input_channel=input_channel, num_classes=num_classes, **pre
    ).to(device)
    model.eval()
    return model


def count_params(model):
    """直接统计参数量 (比 thop 返回的更可靠)。返回 (total_M, trainable_M)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def measure_flops(model, x):
    """返回 (GMac, GFLOPs)。优先 thop;失败回退 fvcore。FLOPs ≈ 2 * MACs。"""
    try:
        from thop import profile
        import copy
        # thop 会给模块挂 buffer,用 deepcopy 避免污染后续 FPS 测量
        macs, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
        gmac = macs / 1e9
        return gmac, gmac * 2, "thop"
    except Exception as e_thop:
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(model, x).unsupported_ops_warnings(False)
            # fvcore 的 .total() 计的是 MAC (它命名为 flops 但语义是乘加)
            gmac = flops.total() / 1e9
            return gmac, gmac * 2, "fvcore"
        except Exception as e_fv:
            return float("nan"), float("nan"), f"failed({type(e_thop).__name__}/{type(e_fv).__name__})"


@torch.no_grad()
def measure_fps(model, x, warmup=20, iters=100, repeats=3):
    """规范化 FPS 测量: warmup + cuda.synchronize + 多轮取最优。

    返回 (fps, latency_ms_mean)。batch=1 时 fps = 1000/latency_ms。
    """
    cuda = x.is_cuda
    # warmup: 触发 cuDNN autotune / 首次 kernel 编译
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()

    best_total = float("inf")
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        best_total = min(best_total, time.perf_counter() - t0)

    latency_ms = best_total / iters * 1000.0
    bs = x.shape[0]
    fps = bs * iters / best_total
    return fps, latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="模型名列表;默认对比集")
    ap.add_argument("--all", action="store_true", help="测 model_id.json 里全部模型")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--input-channel", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1, help="FPS 测量 batch;论文常用 1")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pretrained", type=str, default=None, help="USEANet 的 PVT backbone 权重路径")
    ap.add_argument("--out", type=str, default="result/efficiency.csv")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.all:
        with open("models/model_id.json") as f:
            models = [m["modelname"] for m in json.load(f)]
    else:
        models = args.models or DEFAULT_MODELS

    print(f"Device: {device} | img={args.img_size} C={args.input_channel} "
          f"bs(fps)={args.batch_size} | {len(models)} models\n")

    rows = []
    for name in models:
        print(f"[{name}] building ...", flush=True)
        try:
            model = build(name, args.num_classes, args.input_channel,
                          args.img_size, args.pretrained, device)
        except Exception as e:
            print(f"  ✗ build failed: {type(e).__name__}: {e}\n")
            rows.append({"model": name, "input_size": "", "params_M": "", "trainable_M": "",
                         "gmac": "", "gflops": "", "fps": "", "latency_ms": "",
                         "flops_tool": "", "status": f"build_failed:{type(e).__name__}"})
            continue

        size = PER_MODEL_INPUT_SIZE.get(name, args.img_size)
        if size != args.img_size:
            print(f"  (note: {name} 使用原生输入尺寸 {size}×{size},FLOPs 不与 {args.img_size} 同分辨率)")
        x_flops = torch.randn(1, args.input_channel, size, size, device=device)
        x_fps = torch.randn(args.batch_size, args.input_channel, size, size, device=device)

        total_m, train_m = count_params(model)
        try:
            gmac, gflops, tool = measure_flops(model, x_flops)
        except Exception as e:
            gmac, gflops, tool = float("nan"), float("nan"), f"err:{type(e).__name__}"

        try:
            fps, lat = measure_fps(model, x_fps, args.warmup, args.iters, args.repeats)
            status = "ok"
        except Exception as e:
            fps, lat, status = float("nan"), float("nan"), f"fps_failed:{type(e).__name__}"

        print(f"  Params={total_m:.2f}M  FLOPs={gflops:.2f}G ({gmac:.2f}GMac,{tool})  "
              f"FPS={fps:.1f}  Latency={lat:.2f}ms\n", flush=True)
        rows.append({
            "model": name,
            "input_size": size,
            "params_M": f"{total_m:.3f}",
            "trainable_M": f"{train_m:.3f}",
            "gmac": f"{gmac:.3f}",
            "gflops": f"{gflops:.3f}",
            "fps": f"{fps:.2f}",
            "latency_ms": f"{lat:.3f}",
            "flops_tool": tool,
            "status": status,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 终端 markdown 表
    print("\n| Model | Input | Params(M) | FLOPs(G) | GMac | FPS | Latency(ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r.get('input_size','')} | {r['params_M']} | {r['gflops']} | {r['gmac']} | {r['fps']} | {r['latency_ms']} |")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
