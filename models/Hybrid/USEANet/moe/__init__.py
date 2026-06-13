"""Ultrasound-physics MoE for USEANet (design Pillar 1). No simulator imports."""

EXPERT_NAMES = ["despeckle", "edge", "shadow", "posterior", "contrast", "hf"]

from .physics_moe import PhysicsMoE
from .losses import router_supervision_loss, load_balance_loss, effective_experts

__all__ = [
    "EXPERT_NAMES", "PhysicsMoE",
    "router_supervision_loss", "load_balance_loss", "effective_experts",
]
