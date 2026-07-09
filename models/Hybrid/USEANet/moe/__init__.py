"""Ultrasound-physics MoE for USEANet (design Pillar 1). No simulator imports."""

EXPERT_NAMES = ["despeckle", "edge", "shadow", "posterior", "contrast", "hf"]

from .physics_moe import PhysicsMoE
from .physics_estimator import PhysicsEstimator, acoustic_pseudo_gt, PHYS_NAMES
from .losses import (router_supervision_loss, load_balance_loss,
                     physics_calib_loss, effective_experts)

__all__ = [
    "EXPERT_NAMES", "PhysicsMoE",
    "PhysicsEstimator", "acoustic_pseudo_gt", "PHYS_NAMES",
    "router_supervision_loss", "load_balance_loss", "physics_calib_loss",
    "effective_experts",
]
