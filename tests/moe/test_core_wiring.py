import torch
from models.Hybrid.USEANet.usea_core import USEANet, MultiBranchFeatureProcessor
from models.Hybrid.USEANet.moe import PhysicsMoE


def _net():
    torch.manual_seed(0)
    return USEANet(channel=32, num_classes=1, sem_downsample=1, use_softmax=False)


def test_x3_x4_are_physics_moe_x2_is_multibranch():
    net = _net()
    assert isinstance(net.feature_processor_3, PhysicsMoE)
    assert isinstance(net.feature_processor_4, PhysicsMoE)
    assert isinstance(net.feature_processor_2, MultiBranchFeatureProcessor)


def test_forward_still_returns_eight_maps():
    net = _net()
    x = torch.rand(1, 3, 256, 256)
    out = net(x)
    assert len(out) == 8
    for m in out:
        assert m.shape[0] == 1


def test_moe_modules_collectible():
    net = _net()
    moes = [m for m in net.modules() if isinstance(m, PhysicsMoE)]
    assert len(moes) == 2
