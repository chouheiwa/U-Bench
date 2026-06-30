"""PhysicsMoE: sparse ultrasound-physics MoE block (design §3).

Drop-in replacement for MultiBranchFeatureProcessor at the x3/x4 layers:
``[B, in, H, W] -> [B, out, H, W]`` with a residual. Computes label-free proxy
maps, routes per-position top-2 over 6 anchored experts, combines densely with
the gate mask. Stashes ``last_gate`` / ``last_proxy`` and exposes ``aux_loss``
for the adapter to aggregate (router supervision + load balance).
"""
import os

import torch
import torch.nn as nn

from . import EXPERT_NAMES
from .experts import build_experts
from .proxy import degradation_proxies
from .router import DegradationAwareRouter
from .losses import router_supervision_loss, load_balance_loss, effective_experts


class PhysicsMoE(nn.Module):
    def __init__(self, in_channel, out_channel, num_experts=None, k=2, channel=32):
        super().__init__()
        # Ablation env switches (default-off -> identical 6-expert top-2 model):
        #   USEANET_NUM_EXPERTS=N -> use the first N physics cues/experts (default 6)
        #   USEANET_TOPK=k        -> per-position active experts (default 2; 1 = top-1)
        if num_experts is None:
            num_experts = int(os.environ.get("USEANET_NUM_EXPERTS", len(EXPERT_NAMES)))
        num_experts = max(1, min(num_experts, len(EXPERT_NAMES)))
        k = int(os.environ.get("USEANET_TOPK", k))
        k = max(1, min(k, num_experts))
        self.num_experts = num_experts
        self.k = k
        self.experts = build_experts(in_channel, out_channel, channel, num_experts)
        self.router = DegradationAwareRouter(in_channel, num_experts, k)
        self.res = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # Optional physical-GT supervision target (Track C); None -> use proxy.
        self.supervision_target = None
        self.last_gate = None
        self.last_proxy = None

    def forward(self, x):
        proxy = degradation_proxies(x, num_experts=self.num_experts)  # [B,E,H,W]
        gate = self.router(x, proxy)                          # [B,E,H,W]
        # Dense expert compute + per-position top-2 mask combine.
        out = 0.0
        for e_idx, expert in enumerate(self.experts):
            out = out + expert(x) * gate[:, e_idx:e_idx + 1]
        self.last_gate = gate
        self.last_proxy = proxy
        return self.relu(out + self.res(x))

    def aux_loss(self, route_weight, lb_weight):
        if self.last_gate is None:
            raise RuntimeError("aux_loss called before forward()")
        # The router-supervision target (proxy or physical GT) is a pseudo-label:
        # detach it so KL pulls the gate toward the target, not the target (and
        # the backbone features behind the proxy) toward the gate.
        target = (self.supervision_target if self.supervision_target is not None
                  else self.last_proxy).detach()
        route = router_supervision_loss(self.last_gate, target)
        lb = load_balance_loss(self.last_gate)
        return route_weight * route + lb_weight * lb

    def eff_experts(self):
        return effective_experts(self.last_gate)
