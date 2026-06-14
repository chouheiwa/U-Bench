"""Degradation-aware per-position top-2 router (design §3.2).

Eats the feature map AND the label-free degradation proxy stack, emits a sparse
gate: at every spatial position exactly k=2 experts are active, gate weights
renormalised over the active two. Implemented as dense logits + a straight
top-k mask (the maps are tiny, so true sparse dispatch is not worth it).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationAwareRouter(nn.Module):
    def __init__(self, in_channel, num_experts=6, k=2, hidden=32):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        # Input = feature map + proxy stack (num_experts channels).
        self.net = nn.Sequential(
            nn.Conv2d(in_channel + num_experts, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )

    def forward(self, feat, proxy):
        logits = self.net(torch.cat([feat, proxy], dim=1))   # [B,E,H,W]
        self.last_logits = logits
        # Top-k over the expert dim at each position.
        topv, topi = logits.topk(self.k, dim=1)              # [B,k,H,W]
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        # Softmax over active experts only.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
        gate = F.softmax(masked_logits, dim=1) * mask
        return gate
