import torch
from models.Hybrid.USEANet import USEANet


def _model():
    torch.manual_seed(0)
    return USEANet(input_channel=3, num_classes=1, channel=32)


def test_set_training_progress_controls_route_weight():
    m = _model()
    m.set_training_progress(0.0)
    assert abs(m._route_weight() - 0.5) < 1e-6          # ROUTE_WEIGHT_MAX at start
    m.set_training_progress(1.0)
    assert abs(m._route_weight() - 0.05) < 1e-6         # ROUTE_WEIGHT_MIN at end


def test_deep_supervision_loss_includes_moe_aux():
    m = _model()
    m.set_training_progress(0.0)
    x = torch.rand(1, 1, 256, 256)
    label = (torch.rand(1, 1, 256, 256) > 0.5).float()
    outputs = m(x)
    loss = m.deep_supervision_loss(outputs, label)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_eff_experts_reported():
    m = _model()
    x = torch.rand(1, 1, 256, 256)
    m(x)
    stats = m.moe_stats()
    assert "eff_experts" in stats and 1.0 <= stats["eff_experts"] <= 6.0
