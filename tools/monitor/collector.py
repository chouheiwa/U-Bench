"""只读训练进程看板的采集 + 解析逻辑(无 HTTP,可单测)。"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime

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


def parse_gpus(nvidia_smi_csv: str) -> list:
    """解析 nvidia-smi CSV(noheader,nounits)为每卡状态;解析不了的行跳过。"""
    gpus = []
    for line in nvidia_smi_csv.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 6:
            continue
        try:
            gpus.append({
                "index": int(cells[0]),
                "name": cells[1],
                "util": int(cells[2]),
                "mem_used": int(cells[3]),
                "mem_total": int(cells[4]),
                "temp": int(cells[5]),
            })
        except ValueError:
            continue
    return gpus


DEFAULT_STALE_SEC = 180
TAIL_BYTES = 64 * 1024


def _tail(path: str, nbytes: int = TAIL_BYTES) -> str:
    with open(path, "rb") as f:
        try:
            f.seek(-nbytes, os.SEEK_END)
        except OSError:
            f.seek(0)
        return f.read().decode("utf-8", errors="replace")


def read_run_dir(output_root: str, model, dataset, exp_name) -> dict:
    """定位 output_root/<model>/<dataset>/<exp_name>/,读 config.json + tail training.log。"""
    info = {"config": {}, "has_log": False, "log_mtime": None,
            "epoch": None, "total_epochs": None, "val_iou": None,
            "best_iou": None, "best_epoch": None, "completed": False}
    if not (model and dataset and exp_name):
        return info
    d = os.path.join(output_root, str(model), str(dataset), str(exp_name))
    cfg_path = os.path.join(d, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                info["config"] = json.load(f)
        except (OSError, ValueError):
            info["config"] = {}
    log_path = os.path.join(d, "training.log")
    if os.path.isfile(log_path):
        info["has_log"] = True
        try:
            info["log_mtime"] = os.path.getmtime(log_path)
            info.update(parse_training_log(_tail(log_path)))
        except OSError:
            pass
    return info


def classify_status(progress: dict, log_mtime, now: float,
                    has_log: bool = True, stale_sec: int = DEFAULT_STALE_SEC) -> str:
    if progress.get("completed"):
        return "finished"
    if not has_log:
        return "no_log"
    if progress.get("epoch") is None:
        return "starting"
    if log_mtime is not None and (now - log_mtime) > stale_sec:
        return "stale"
    return "running"


_PS_ARGS = ["ps", "-eo", "pid=,etimes=,pcpu=,pmem=,args="]
_NVSMI_ARGS = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def run_cmd(args: list, timeout: float = 5.0) -> str:
    """运行命令,返回 stdout;任何失败(找不到命令/超时/非零)返回空串。"""
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return res.stdout if res.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def list_jobs(output_root: str = "./output", runner=run_cmd, now=None) -> dict:
    """组装看板数据:gpus + 当前训练 jobs。runner/now 可注入以便测试。"""
    if now is None:
        now = time.time()
    gpus = parse_gpus(runner(_NVSMI_ARGS))
    procs = parse_processes(runner(_PS_ARGS))

    jobs = []
    for p in procs:
        a = p["args"]
        info = read_run_dir(output_root, a["model"], a["dataset_name"], a["exp_name"])
        cfg = info["config"]
        status = classify_status(info, info["log_mtime"], now, has_log=info["has_log"])
        jobs.append({
            "pid": p["pid"],
            "gpu": a["gpu"],
            "exp_name": a["exp_name"],
            "model": a["model"],
            "dataset": a["dataset_name"],
            "uptime_sec": p["uptime_sec"],
            "cpu": p["cpu"],
            "mem": p["mem"],
            "epoch": info["epoch"],
            "total_epochs": info["total_epochs"] or a["max_epochs"],
            "val_iou": info["val_iou"],
            "best_iou": info["best_iou"],
            "best_epoch": info["best_epoch"],
            "status": status,
            "seed": a["seed"] if a["seed"] is not None else cfg.get("seed"),
            "base_lr": a["base_lr"] if a["base_lr"] is not None else cfg.get("base_lr"),
            "log_mtime": (datetime.fromtimestamp(info["log_mtime"]).isoformat()
                          if info["log_mtime"] else None),
        })
    return {
        "generated_at": datetime.fromtimestamp(now).isoformat(),
        "gpus": gpus,
        "jobs": jobs,
    }
