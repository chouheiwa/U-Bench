"""只读训练进程看板的采集 + 解析逻辑(无 HTTP,可单测)。"""
from __future__ import annotations

import re

_EPOCH_RE = re.compile(r"epoch \[(\d+)/(\d+)\]")
_VAL_IOU_RE = re.compile(r"val_iou ([\d.]+)")
_BEST_RE = re.compile(r"Best IoU: ([\d.]+), Best Epoch: (\d+)")


def parse_training_log(text: str) -> dict:
    """从 training.log 文本(通常是末尾片段)解析进度。缺失字段返回 None。"""
    epoch = total = val_iou = best_iou = best_epoch = None
    # 取最后一条 epoch 行
    last_epoch_line = None
    for line in text.splitlines():
        if _EPOCH_RE.search(line):
            last_epoch_line = line
    if last_epoch_line is not None:
        m = _EPOCH_RE.search(last_epoch_line)
        epoch, total = int(m.group(1)), int(m.group(2))
        vm = _VAL_IOU_RE.search(last_epoch_line)
        if vm:
            val_iou = float(vm.group(1))
    # 全文最后一次 Best IoU
    best_matches = _BEST_RE.findall(text)
    if best_matches:
        best_iou = float(best_matches[-1][0])
        best_epoch = int(best_matches[-1][1])
    return {
        "epoch": epoch,
        "total_epochs": total,
        "val_iou": val_iou,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "completed": "Training completed" in text,
    }


def _flag(cmdline: str, name: str):
    m = re.search(rf"--{name}\s+(\S+)", cmdline)
    return m.group(1) if m else None


def parse_cmdline_args(cmdline: str) -> dict:
    """从进程命令行解析关心的训练参数;缺失为 None。"""
    def as_int(v):
        return int(v) if v is not None else None

    gpu_raw = _flag(cmdline, "gpu")
    gpu = None
    if gpu_raw is not None:
        try:
            gpu = int(gpu_raw.split(",")[0])
        except ValueError:
            gpu = None
    lr_raw = _flag(cmdline, "base_lr")
    return {
        "model": _flag(cmdline, "model"),
        "dataset_name": _flag(cmdline, "dataset_name"),
        "exp_name": _flag(cmdline, "exp_name"),
        "gpu": gpu,
        "max_epochs": as_int(_flag(cmdline, "max_epochs")),
        "seed": as_int(_flag(cmdline, "seed")),
        "base_lr": float(lr_raw) if lr_raw is not None else None,
    }


def parse_processes(ps_output: str) -> list:
    """解析 `ps -eo pid=,etimes=,pcpu=,pmem=,args=` 输出,仅保留训练进程(main.py)。"""
    procs = []
    for line in ps_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid_s, etimes_s, cpu_s, mem_s, cmdline = parts
        if "main.py" not in cmdline:
            continue
        # 排除 grep / 监控自身等非训练进程
        if "tools.monitor" in cmdline or cmdline.startswith("grep"):
            continue
        try:
            proc = {
                "pid": int(pid_s),
                "uptime_sec": int(etimes_s),
                "cpu": float(cpu_s),
                "mem": float(mem_s),
                "cmdline": cmdline,
                "args": parse_cmdline_args(cmdline),
            }
        except ValueError:
            continue
        procs.append(proc)
    return procs


def list_jobs(output_root: str = "./output", runner=None, now=None) -> dict:
    """Placeholder — full implementation in Task 6."""
    raise NotImplementedError("list_jobs not yet implemented")
