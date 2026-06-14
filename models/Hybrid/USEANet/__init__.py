"""U-Bench adapter for USEANet (Ultrasound Segmentation Enhancement Attention Network).

The original USEANet returns 8 maps (4 foreground + 4 background) and is trained
with a custom dual fg/bg ``structure_loss`` (boundary-weighted wBCE + wIoU).
This adapter keeps that weighted loss instead of U-Bench's shared ``BCEDiceLoss``:

* runs the core net with ``num_classes`` channels per head (default 1 = binary),
* repeats a 1-channel input to 3 channels for the PVT backbone,
* ``forward`` returns the 4 foreground maps as a deep-supervision tuple, ordered
  coarse->fine so the primary refined prediction ``lateral_map_2_fg`` is last
  (U-Bench uses ``outputs[-1]`` for metrics and validation; ``lateral_map_5_fg``
  is only the coarse global aggregate), and caches the paired background maps on
  ``self._bg_maps`` in the same order (the bg heads are an intrinsic forward
  product, so no extra compute),
* ``deep_supervision_loss`` applies the original weighted ``structure_loss`` over
  all 4 fg/bg scales. ``main.py`` calls this method when present, so the weighted
  loss is used for training while every other model keeps ``BCEDiceLoss``.

NOTE: caching the bg maps assumes the single-GPU, forward-then-loss order that
``main.py`` uses; it is not safe for ``nn.DataParallel`` replicas.

Register with ``deeps_supervision: 1`` in ``models/model_id.json``.
"""

import torch.nn as nn

from .usea_core import USEANet as _USEANetCore
from .usea_loss import structure_loss
from .moe import PhysicsMoE

# Router-supervision annealing (design §6: small weight + linear anneal -> anchor
# early, free the router later). Anneal from MAX to MIN over the first
# ANNEAL_FRACTION of training; load-balance weight is constant.
ROUTE_WEIGHT_MAX = 0.5
ROUTE_WEIGHT_MIN = 0.05
ANNEAL_FRACTION = 0.5
LB_WEIGHT = 0.01


class USEANet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, channel=32,
                 pretrained_model_path=None):
        super().__init__()
        self.input_channel = input_channel
        self.num_classes = num_classes
        self._bg_maps = None  # paired background maps from the last forward()
        self._progress = 0.0  # training fraction in [0,1], set by main.py loop
        self.net = _USEANetCore(
            channel=channel,
            num_classes=num_classes,
            sem_downsample=1,
            # softmax over the channel dim is degenerate for single-channel
            # (binary) heads; keep the plain fg-vs-bg residual instead.
            use_softmax=num_classes > 1,
            pretrained_model_path=pretrained_model_path,
        )

    def forward(self, x):
        # U-Bench's pipeline applies ImageNet Normalize() then divides by 255
        # again in dataset.py, so inputs arrive at ~+-0.01. Undo that erroneous
        # /255 to restore the ImageNet-normalized scale the pretrained PVT-B0
        # backbone expects (without it the pretrained features are degenerate and
        # IoU caps near 0.13). From-scratch models are unaffected and keep the
        # repo-wide pipeline untouched.
        x = x * 255.0
        # PVT-B0 backbone expects 3-channel input.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        (lateral_map_2_fg, lateral_map_3_fg, lateral_map_4_fg, lateral_map_5_fg,
         bg2, bg3, bg4, bg5) = self.net(x)
        # lateral_map_2 is the final refined full-res prediction (PraNet/USEANet
        # convention); lateral_map_5 is the coarse global aggregate. U-Bench scores
        # and validates on outputs[-1], so order coarse->fine to put the primary
        # head (fg2) last. Cache bg maps in the SAME order for the paired loss.
        self._bg_maps = (bg5, bg4, bg3, bg2)
        return (lateral_map_5_fg, lateral_map_4_fg, lateral_map_3_fg, lateral_map_2_fg)

    def set_training_progress(self, frac):
        self._progress = float(max(0.0, min(1.0, frac)))

    def _route_weight(self):
        t = min(self._progress / ANNEAL_FRACTION, 1.0)
        return ROUTE_WEIGHT_MAX + (ROUTE_WEIGHT_MIN - ROUTE_WEIGHT_MAX) * t

    def _moe_modules(self):
        return [m for m in self.modules() if isinstance(m, PhysicsMoE)]

    def moe_stats(self):
        moes = self._moe_modules()
        if not moes or moes[0].last_gate is None:
            return {}
        eff = sum(m.eff_experts() for m in moes) / len(moes)
        return {"eff_experts": eff}

    def deep_supervision_loss(self, fg_outputs, label_batch):
        """USEANet's weighted multi-scale fg/bg structure loss.

        ``fg_outputs`` is the tuple returned by ``forward`` (fg2, fg3, fg4, fg5);
        the paired bg maps are read from ``self._bg_maps`` set during that call.
        """
        if self._bg_maps is None:
            raise RuntimeError("deep_supervision_loss called before forward()")
        bg_outputs = self._bg_maps
        bg_mask = 1.0 - label_batch
        nc = self.num_classes
        total = 0.0
        for fg, bg in zip(fg_outputs, bg_outputs):
            total = total + structure_loss(fg, bg, label_batch, bg_mask, nc)
        # Add MoE auxiliary losses (router supervision + load balance), annealed.
        rw = self._route_weight()
        for moe in self._moe_modules():
            total = total + moe.aux_loss(route_weight=rw, lb_weight=LB_WEIGHT)
        return total


def useanet(input_channel=3, num_classes=1):
    return USEANet(input_channel=input_channel, num_classes=num_classes)


__all__ = ['USEANet', 'useanet']
