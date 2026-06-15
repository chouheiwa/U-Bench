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
