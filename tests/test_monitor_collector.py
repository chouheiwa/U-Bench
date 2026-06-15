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
