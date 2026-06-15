# 训练进程 Web 看板(只读) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 U-Bench 加一个零依赖、只读的 Web 看板,实时显示正在跑的训练 job(所在 GPU、epoch 进度、当前/最佳 val_IoU)与每张卡的 util/显存/温度。

**Architecture:** 纯逻辑(`collector.py`,采集 + 解析 `ps`/`nvidia-smi`/`config.json`/`training.log`,通过注入的 `runner` 间接调命令,可单测)与 web 层(`server.py` 用 stdlib `ThreadingHTTPServer` 提供 `GET /` 和 `GET /api/jobs`)分离;前端为单页原生 JS 每 4s 轮询。零改动训练代码、零新依赖。

**Tech Stack:** Python 标准库(`http.server`、`subprocess`、`json`、`re`、`argparse`)、原生 JS/HTML、pytest。

**测试命令(全程统一):** `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`

**Spec:** `docs/superpowers/specs/2026-06-15-training-monitor-web-viewer-design.md`

---

## 文件结构

- Create: `tools/monitor/__init__.py` — 包标记,re-export `list_jobs`。
- Create: `tools/monitor/collector.py` — 采集 + 解析全部纯逻辑。
- Create: `tools/monitor/server.py` — HTTP 服务(两个路由 + argparse CLI)。
- Create: `tools/monitor/__main__.py` — `python -m tools.monitor` 入口。
- Create: `tools/monitor/index.html` — 单页看板 UI。
- Create: `tests/test_monitor_collector.py` — collector 单测。

> 现有 `conftest.py` 在仓库根,`tests/` 下测试以仓库根为导入根,故 `from tools.monitor.collector import ...` 可直接解析。

---

## Task 1: 包脚手架 + `parse_training_log`

**Files:**
- Create: `tools/monitor/__init__.py`
- Create: `tools/monitor/collector.py`
- Test: `tests/test_monitor_collector.py`

- [ ] **Step 1: 写失败测试**

`tests/test_monitor_collector.py`:
```python
from tools.monitor.collector import parse_training_log

REAL_LOG_TAIL = """2026-06-14 12:12:10,622 - INFO - epoch [98/100]  train_loss: 3.3426, train_iou: 0.7445 - val_loss 0.3795 - val_iou 0.6195 - val_SE 0.7876 - val_PC 0.7288 - val_F1 0.7207 - val_ACC 0.9507
2026-06-14 12:12:27,677 - INFO - epoch [99/100]  train_loss: 3.3208, train_iou: 0.7489 - val_loss 0.3798 - val_iou 0.6194 - val_SE 0.7893 - val_PC 0.7279 - val_F1 0.7214 - val_ACC 0.9504
2026-06-14 12:12:28,213 - INFO - Training completed. Best IoU: 0.6228384228172058, Best Epoch: 83, Best SE: 0.78, Best PC: 0.73, Best F1: 0.72, Best ACC: 0.95
"""

RUNNING_LOG_TAIL = """2026-06-14 12:11:01,713 - INFO - epoch [94/100]  train_loss: 3.37, train_iou: 0.73 - val_loss 0.38 - val_iou 0.6163 - val_SE 0.78 - val_PC 0.72 - val_F1 0.71 - val_ACC 0.95
"""


def test_parse_completed_log():
    out = parse_training_log(REAL_LOG_TAIL)
    assert out["epoch"] == 99
    assert out["total_epochs"] == 100
    assert abs(out["val_iou"] - 0.6194) < 1e-4
    assert abs(out["best_iou"] - 0.6228384228172058) < 1e-9
    assert out["best_epoch"] == 83
    assert out["completed"] is True


def test_parse_running_log():
    out = parse_training_log(RUNNING_LOG_TAIL)
    assert out["epoch"] == 94
    assert out["total_epochs"] == 100
    assert abs(out["val_iou"] - 0.6163) < 1e-4
    assert out["best_iou"] is None
    assert out["best_epoch"] is None
    assert out["completed"] is False


def test_parse_empty_log():
    out = parse_training_log("")
    assert out["epoch"] is None
    assert out["total_epochs"] is None
    assert out["val_iou"] is None
    assert out["completed"] is False
```

- [ ] **Step 2: 运行,确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: FAIL,`ModuleNotFoundError: No module named 'tools.monitor'`(或 import 错误)。

- [ ] **Step 3: 写最小实现**

`tools/monitor/__init__.py`:
```python
from .collector import list_jobs

__all__ = ["list_jobs"]
```

`tools/monitor/collector.py`:
```python
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
```

- [ ] **Step 4: 运行,确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: PASS(3 passed)。

- [ ] **Step 5: 提交**

```bash
git add tools/monitor/__init__.py tools/monitor/collector.py tests/test_monitor_collector.py
git commit -m "feat(monitor): training.log progress parser"
```

---

## Task 2: `parse_cmdline_args`

**Files:**
- Modify: `tools/monitor/collector.py`
- Test: `tests/test_monitor_collector.py`

- [ ] **Step 1: 写失败测试**(追加到测试文件)

```python
from tools.monitor.collector import parse_cmdline_args


def test_parse_cmdline_args_full():
    cmd = ("python main.py --gpu 1 --model USEANet --model_id 115 "
           "--base_dir hf_data/data/busi --dataset_name busi --exp_name moe_busi_e250 "
           "--max_epochs 250 --seed 42 --base_lr 0.01")
    out = parse_cmdline_args(cmd)
    assert out["model"] == "USEANet"
    assert out["dataset_name"] == "busi"
    assert out["exp_name"] == "moe_busi_e250"
    assert out["gpu"] == 1
    assert out["max_epochs"] == 250
    assert out["seed"] == 42
    assert abs(out["base_lr"] - 0.01) < 1e-9


def test_parse_cmdline_args_gpu_list_takes_first():
    out = parse_cmdline_args("python main.py --gpu 0,1 --model USEANet")
    assert out["gpu"] == 0


def test_parse_cmdline_args_missing_are_none():
    out = parse_cmdline_args("python main.py")
    assert out["model"] is None
    assert out["gpu"] is None
    assert out["exp_name"] is None
```

- [ ] **Step 2: 运行,确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: FAIL,`ImportError: cannot import name 'parse_cmdline_args'`。

- [ ] **Step 3: 写最小实现**(追加到 `collector.py`)

```python
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
```

> 注意:`_flag(cmdline, "model")` 不会误匹配 `--model_id`,因为正则要求 `--model` 后跟空白。

- [ ] **Step 4: 运行,确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: PASS(6 passed)。

- [ ] **Step 5: 提交**

```bash
git add tools/monitor/collector.py tests/test_monitor_collector.py
git commit -m "feat(monitor): parse training args from process cmdline"
```

---

## Task 3: `parse_processes`(解析 `ps` 输出,过滤 main.py)

**Files:**
- Modify: `tools/monitor/collector.py`
- Test: `tests/test_monitor_collector.py`

`ps` 调用约定:`ps -eo pid=,etimes=,pcpu=,pmem=,args=`(无表头;`etimes` 为已运行秒数)。每行示例:
`  31542   3600  98.5  2.1  python main.py --model USEANet --gpu 1 --exp_name X ...`

- [ ] **Step 1: 写失败测试**(追加)

```python
from tools.monitor.collector import parse_processes

PS_OUTPUT = """  31542   3600  98.5  2.1  python main.py --model USEANet --gpu 1 --dataset_name busi --exp_name run_a --max_epochs 250
  31999    120  50.0  1.0  python main.py --model U_Net --gpu 0 --dataset_name busi --exp_name run_b
   4242     10   0.0  0.1  python -m tools.monitor --port 8800
   5151      5   0.0  0.1  grep main.py
"""


def test_parse_processes_filters_to_main_py():
    procs = parse_processes(PS_OUTPUT)
    assert len(procs) == 2  # monitor 与 grep 行被排除
    pids = {p["pid"] for p in procs}
    assert pids == {31542, 31999}


def test_parse_processes_fields():
    procs = parse_processes(PS_OUTPUT)
    a = next(p for p in procs if p["pid"] == 31542)
    assert a["uptime_sec"] == 3600
    assert abs(a["cpu"] - 98.5) < 1e-6
    assert abs(a["mem"] - 2.1) < 1e-6
    assert "main.py" in a["cmdline"]
    assert a["args"]["model"] == "USEANet"  # 已内联 parse_cmdline_args
    assert a["args"]["gpu"] == 1
```

- [ ] **Step 2: 运行,确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: FAIL,`ImportError: cannot import name 'parse_processes'`。

- [ ] **Step 3: 写最小实现**(追加)

```python
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
```

- [ ] **Step 4: 运行,确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: PASS(8 passed)。

- [ ] **Step 5: 提交**

```bash
git add tools/monitor/collector.py tests/test_monitor_collector.py
git commit -m "feat(monitor): parse and filter training processes from ps"
```

---

## Task 4: `parse_gpus`(解析 nvidia-smi CSV)

**Files:**
- Modify: `tools/monitor/collector.py`
- Test: `tests/test_monitor_collector.py`

nvidia-smi 调用约定:`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits`。

- [ ] **Step 1: 写失败测试**(追加)

```python
from tools.monitor.collector import parse_gpus

NVIDIA_CSV = """0, NVIDIA GeForce RTX 2080 Ti, 95, 8000, 11264, 72
1, NVIDIA GeForce RTX 2080 Ti, 0, 12, 11264, 41
"""


def test_parse_gpus():
    gpus = parse_gpus(NVIDIA_CSV)
    assert len(gpus) == 2
    g0 = gpus[0]
    assert g0["index"] == 0
    assert g0["name"] == "NVIDIA GeForce RTX 2080 Ti"
    assert g0["util"] == 95
    assert g0["mem_used"] == 8000
    assert g0["mem_total"] == 11264
    assert g0["temp"] == 72


def test_parse_gpus_empty_returns_empty_list():
    assert parse_gpus("") == []
```

- [ ] **Step 2: 运行,确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: FAIL,`ImportError: cannot import name 'parse_gpus'`。

- [ ] **Step 3: 写最小实现**(追加)

```python
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
```

- [ ] **Step 4: 运行,确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: PASS(10 passed)。

- [ ] **Step 5: 提交**

```bash
git add tools/monitor/collector.py tests/test_monitor_collector.py
git commit -m "feat(monitor): parse nvidia-smi gpu stats"
```

---

## Task 5: `read_run_dir` + `classify_status`

**Files:**
- Modify: `tools/monitor/collector.py`
- Test: `tests/test_monitor_collector.py`

- [ ] **Step 1: 写失败测试**(追加)

```python
import json

from tools.monitor.collector import read_run_dir, classify_status


def _make_run(tmp_path, model, dataset, exp_name, log_text, config=None):
    d = tmp_path / model / dataset / exp_name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or {"seed": 41, "base_lr": 0.01}))
    (d / "training.log").write_text(log_text)
    return str(tmp_path)


def test_read_run_dir_joins_config_and_log(tmp_path):
    root = _make_run(tmp_path, "USEANet", "busi", "run_a", RUNNING_LOG_TAIL)
    info = read_run_dir(root, "USEANet", "busi", "run_a")
    assert info["epoch"] == 94
    assert info["config"]["seed"] == 41
    assert info["log_mtime"] is not None
    assert info["has_log"] is True


def test_read_run_dir_missing_dir(tmp_path):
    info = read_run_dir(str(tmp_path), "USEANet", "busi", "nope")
    assert info["has_log"] is False
    assert info["epoch"] is None
    assert info["config"] == {}


def test_classify_status():
    now = 1_000_000.0
    # completed
    assert classify_status({"completed": True, "epoch": 99}, now - 10, now) == "finished"
    # no log
    assert classify_status({"completed": False, "epoch": None}, None, now,
                           has_log=False) == "no_log"
    # log exists but no epoch yet
    assert classify_status({"completed": False, "epoch": None}, now - 1, now,
                           has_log=True) == "starting"
    # stale (mtime old)
    assert classify_status({"completed": False, "epoch": 5}, now - 999, now,
                           has_log=True, stale_sec=180) == "stale"
    # running
    assert classify_status({"completed": False, "epoch": 5}, now - 10, now,
                           has_log=True) == "running"
```

- [ ] **Step 2: 运行,确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: FAIL,`ImportError: cannot import name 'read_run_dir'`。

- [ ] **Step 3: 写最小实现**(追加;在 `collector.py` 顶部 import 区补 `import json`、`import os`)

```python
import json
import os

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
```

- [ ] **Step 4: 运行,确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: PASS(15 passed)。

- [ ] **Step 5: 提交**

```bash
git add tools/monitor/collector.py tests/test_monitor_collector.py
git commit -m "feat(monitor): read run dir (config + log) and classify status"
```

---

## Task 6: `run_cmd` + `list_jobs`(总装,注入 runner)

**Files:**
- Modify: `tools/monitor/collector.py`
- Test: `tests/test_monitor_collector.py`

- [ ] **Step 1: 写失败测试**(追加)

```python
import time
from datetime import datetime

from tools.monitor.collector import list_jobs


def test_list_jobs_assembles_gpus_and_jobs(tmp_path):
    root = _make_run(tmp_path, "USEANet", "busi", "run_a", RUNNING_LOG_TAIL,
                     config={"seed": 41, "base_lr": 0.01})

    def fake_runner(args):
        if args[0] == "ps":
            return ("  31542 3600 98.5 2.1 python main.py --model USEANet "
                    "--gpu 1 --dataset_name busi --exp_name run_a --max_epochs 100\n")
        if args[0] == "nvidia-smi":
            return NVIDIA_CSV
        return ""

    out = list_jobs(output_root=root, runner=fake_runner, now=time.time())
    assert len(out["gpus"]) == 2
    assert len(out["jobs"]) == 1
    job = out["jobs"][0]
    assert job["pid"] == 31542
    assert job["gpu"] == 1
    assert job["exp_name"] == "run_a"
    assert job["model"] == "USEANet"
    assert job["dataset"] == "busi"
    assert job["epoch"] == 94
    assert job["total_epochs"] == 100
    assert abs(job["val_iou"] - 0.6163) < 1e-4
    assert job["seed"] == 41
    assert job["status"] == "running"
    # generated_at 是可解析的 ISO 时间串
    datetime.fromisoformat(out["generated_at"])


def test_list_jobs_nvidia_smi_failure_degrades(tmp_path):
    root = _make_run(tmp_path, "USEANet", "busi", "run_a", RUNNING_LOG_TAIL)

    def fake_runner(args):
        if args[0] == "ps":
            return ("  31542 3600 98.5 2.1 python main.py --model USEANet "
                    "--gpu 0 --dataset_name busi --exp_name run_a\n")
        return ""  # nvidia-smi 失败 → 空

    out = list_jobs(output_root=root, runner=fake_runner, now=time.time())
    assert out["gpus"] == []
    assert len(out["jobs"]) == 1  # 进程仍正常列出
```

- [ ] **Step 2: 运行,确认失败**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: FAIL,`ImportError: cannot import name 'list_jobs'`。

- [ ] **Step 3: 写最小实现**(追加;在 `collector.py` 顶部 import 区补 `import subprocess`、`import time`、`from datetime import datetime`)

```python
import subprocess
import time
from datetime import datetime

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
```

- [ ] **Step 4: 运行,确认通过**

Run: `conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
Expected: PASS(17 passed)。

- [ ] **Step 5: 提交**

```bash
git add tools/monitor/collector.py tests/test_monitor_collector.py
git commit -m "feat(monitor): list_jobs assembles gpus + jobs with injectable runner"
```

---

## Task 7: HTTP 服务 + 入口

**Files:**
- Create: `tools/monitor/server.py`
- Create: `tools/monitor/__main__.py`

> 此层薄、依赖真实环境(端口/进程),不做单测;以手动 smoke 验证。

- [ ] **Step 1: 写 `tools/monitor/server.py`**

```python
"""只读训练看板 HTTP 服务(stdlib)。"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collector import list_jobs

_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")


def make_handler(output_root: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index.html"):
                try:
                    with open(_HTML_PATH, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    self._send(500, b"index.html missing", "text/plain")
            elif self.path.startswith("/api/jobs"):
                try:
                    payload = json.dumps(list_jobs(output_root)).encode("utf-8")
                except Exception as e:  # 端点永不 500
                    payload = json.dumps(
                        {"generated_at": None, "gpus": [], "jobs": [],
                         "error": str(e)}).encode("utf-8")
                self._send(200, payload, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # 静音默认访问日志
            pass

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="U-Bench 只读训练看板")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--output-root", default="./output")
    args = parser.parse_args(argv)

    handler = make_handler(args.output_root)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"训练看板: http://{args.host}:{args.port}  (output-root={args.output_root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        httpd.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 `tools/monitor/__main__.py`**

```python
from .server import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: smoke 验证(确认服务起得来 + API 不崩)**

Run:
```bash
conda run -n ubench1 python -m tools.monitor --port 8899 &
sleep 1
curl -s http://127.0.0.1:8899/api/jobs | head -c 400
echo
kill %1
```
Expected: 打印形如 `{"generated_at": "...", "gpus": [...], "jobs": [...]}` 的 JSON(无 traceback)。

- [ ] **Step 4: 提交**

```bash
git add tools/monitor/server.py tools/monitor/__main__.py
git commit -m "feat(monitor): read-only http server + python -m entrypoint"
```

---

## Task 8: 前端单页看板

**Files:**
- Create: `tools/monitor/index.html`

- [ ] **Step 1: 写 `tools/monitor/index.html`**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>U-Bench 训练看板</title>
<style>
  body { font-family: -apple-system, Menlo, monospace; background:#0f1115; color:#e6e6e6; margin:0; padding:16px; }
  h1 { font-size:16px; font-weight:600; margin:0 0 12px; }
  .muted { color:#8a8f98; font-size:12px; }
  .bar { background:#262b33; border-radius:4px; height:8px; overflow:hidden; }
  .bar > span { display:block; height:100%; background:#3b82f6; }
  .grid { display:grid; gap:10px; }
  .gpus { grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); margin-bottom:18px; }
  .card { background:#171a21; border:1px solid #262b33; border-radius:8px; padding:10px 12px; }
  .job { display:grid; grid-template-columns:1.6fr 1fr 1.4fr 1fr; gap:10px; align-items:center; }
  .pill { font-size:11px; padding:2px 8px; border-radius:10px; }
  .running { background:#14532d; color:#7ee2a8; }
  .starting { background:#5a4a14; color:#f5d779; }
  .finished { background:#2a2f37; color:#a8b0bd; }
  .stale, .no_log { background:#5a1d1d; color:#f3a2a2; }
  .banner { background:#5a1d1d; color:#f3a2a2; padding:6px 10px; border-radius:6px; margin-bottom:10px; display:none; }
  .big { font-size:15px; font-weight:600; }
</style>
</head>
<body>
  <h1>U-Bench 训练看板 <span class="muted" id="ts"></span></h1>
  <div class="banner" id="banner">连接丢失,显示的是上次数据…</div>
  <div class="grid gpus" id="gpus"></div>
  <div class="grid" id="jobs"></div>

<script>
function pct(used, total){ return total ? Math.round(100*used/total) : 0; }
function fmtDur(s){ if(s==null) return '-'; const h=Math.floor(s/3600), m=Math.floor((s%3600)/60); return h>0?`${h}h${m}m`:`${m}m`; }
function num(v, d){ return (v==null)?'-':Number(v).toFixed(d); }

function renderGpus(gpus){
  document.getElementById('gpus').innerHTML = gpus.map(g => `
    <div class="card">
      <div><b>GPU ${g.index}</b> <span class="muted">${g.name}</span></div>
      <div class="muted">util ${g.util}% · ${g.temp}°C</div>
      <div class="bar"><span style="width:${g.util}%"></span></div>
      <div class="muted">显存 ${g.mem_used} / ${g.mem_total} MB (${pct(g.mem_used,g.mem_total)}%)</div>
      <div class="bar"><span style="width:${pct(g.mem_used,g.mem_total)}%"></span></div>
    </div>`).join('') || '<div class="muted">无 GPU 数据(nvidia-smi 不可用)</div>';
}

function renderJobs(jobs){
  if(!jobs.length){ document.getElementById('jobs').innerHTML = '<div class="muted">当前没有正在跑的训练进程。</div>'; return; }
  document.getElementById('jobs').innerHTML = jobs.map(j => {
    const ep = (j.epoch!=null && j.total_epochs) ? `${j.epoch}/${j.total_epochs}` : '-';
    const p = (j.epoch!=null && j.total_epochs) ? Math.round(100*j.epoch/j.total_epochs) : 0;
    return `<div class="card job">
      <div>
        <div class="big">${j.exp_name ?? '(no exp)'}</div>
        <div class="muted">${j.model ?? '?'} · ${j.dataset ?? '?'} · GPU ${j.gpu ?? '?'} · pid ${j.pid} · ${fmtDur(j.uptime_sec)}</div>
      </div>
      <div><span class="pill ${j.status}">${j.status}</span></div>
      <div>
        <div class="muted">epoch ${ep}</div>
        <div class="bar"><span style="width:${p}%"></span></div>
      </div>
      <div class="muted">
        val_IoU <b>${num(j.val_iou,4)}</b><br>
        best <b>${num(j.best_iou,4)}</b>${j.best_epoch!=null?` @${j.best_epoch}`:''}<br>
        seed ${j.seed ?? '-'} · lr ${j.base_lr ?? '-'}
      </div>
    </div>`;
  }).join('');
}

async function tick(){
  try {
    const r = await fetch('/api/jobs');
    const d = await r.json();
    document.getElementById('banner').style.display = 'none';
    document.getElementById('ts').textContent = d.generated_at ? '· ' + d.generated_at.replace('T',' ').slice(0,19) : '';
    renderGpus(d.gpus || []);
    renderJobs(d.jobs || []);
  } catch(e){
    document.getElementById('banner').style.display = 'block';
  }
}
tick();
setInterval(tick, 4000);
</script>
</body>
</html>
```

- [ ] **Step 2: 手动验证(浏览器)**

Run:
```bash
conda run -n ubench1 python -m tools.monitor --port 8899 &
```
浏览器打开 `http://127.0.0.1:8899`。Expected: 看到 GPU 卡片 + job 列表(无在跑训练时显示"当前没有正在跑的训练进程");停掉服务时出现"连接丢失"横幅。验证后 `kill %1`。

- [ ] **Step 3: 提交**

```bash
git add tools/monitor/index.html
git commit -m "feat(monitor): single-page dashboard UI"
```

---

## Task 9: 端到端联调 + 文档

**Files:**
- Modify: `docs/superpowers/specs/2026-06-15-training-monitor-web-viewer-design.md`(底部追加"用法"小节)

- [ ] **Step 1: 真实联调(若此刻有训练在跑则最佳)**

Run:
```bash
conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q
conda run -n ubench1 python -m tools.monitor --port 8899 &
sleep 1
curl -s http://127.0.0.1:8899/api/jobs | python -c "import sys,json; d=json.load(sys.stdin); print('gpus',len(d['gpus']),'jobs',len(d['jobs']))"
kill %1
```
Expected: 全部测试 PASS;curl 打印 `gpus N jobs M`,无 traceback。

- [ ] **Step 2: 在 spec 文档底部追加用法说明**

在文件末尾追加:
```markdown

---

## 用法(实现完成后)

启动:`conda run -n ubench1 python -m tools.monitor --port 8800`,浏览器打开 `http://127.0.0.1:8800`(每 4s 自动刷新)。
单测:`conda run -n ubench1 python -m pytest tests/test_monitor_collector.py -q`
```

- [ ] **Step 3: 提交**

```bash
git add docs/superpowers/specs/2026-06-15-training-monitor-web-viewer-design.md
git commit -m "docs(monitor): usage notes"
```

---

## 完成判据

- [ ] `tests/test_monitor_collector.py` 全绿(17+ 用例)。
- [ ] `python -m tools.monitor` 起得来,`/api/jobs` 返回合法 JSON,nvidia-smi 缺失时降级不崩。
- [ ] 浏览器看板能显示 GPU 状态 + 在跑 job 的 epoch 进度/IoU。
- [ ] 训练代码(`main.py` 等)零改动;无新增第三方依赖。
```
