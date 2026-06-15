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


def list_jobs(output_root: str = "./output", runner=None, now=None) -> dict:
    """Placeholder — full implementation in Task 6."""
    raise NotImplementedError("list_jobs not yet implemented")
