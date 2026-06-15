"""USEANet-only training recipe levers (discriminative LR / warmup / weight EMA /
AdamW). Every lever is gated by a ``USEANET_*`` environment variable and is
default-off; with all switches off the helpers reproduce ``main.py``'s original
SGD + poly schedule exactly. This module imports only torch + stdlib so it never
pulls in the heavy USEANet backbone."""

import os
import warnings

import torch
from torch import optim

# Adapter (models/Hybrid/USEANet/__init__.py) holds self.net -> core.self.backbone,
# so the pretrained PVT-B0 params are prefixed "net.backbone." in the adapter's
# named_parameters().
BACKBONE_PARAM_PREFIXES = ("net.backbone.",)


def lr_at(group_base_lr, iter_num, max_iterations, warmup_iters):
    """Per-group LR. warmup_iters==0 reduces to main.py's original poly schedule:
    group_base_lr * (1 - iter_num/max_iterations) ** 0.9."""
    if warmup_iters > 0 and iter_num < warmup_iters:
        return group_base_lr * (iter_num + 1) / warmup_iters
    denom = max_iterations - warmup_iters
    return group_base_lr * (1.0 - (iter_num - warmup_iters) / denom) ** 0.9
