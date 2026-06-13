"""Ultrasound-physics MoE for USEANet (design Pillar 1). No simulator imports."""

EXPERT_NAMES = ["despeckle", "edge", "shadow", "posterior", "contrast", "hf"]

# physics_moe / losses are added in Tasks B2-B5; keep the package importable
# while only proxy.py exists (proxy.py does `from . import EXPERT_NAMES`).
try:
    from .physics_moe import PhysicsMoE
    from .losses import router_supervision_loss, load_balance_loss, effective_experts
except ImportError:
    pass

__all__ = [
    "EXPERT_NAMES", "PhysicsMoE",
    "router_supervision_loss", "load_balance_loss", "effective_experts",
]
