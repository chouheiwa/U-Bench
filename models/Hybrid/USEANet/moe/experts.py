"""Anchored ultrasound-physics experts (design §3.1).

Each expert = a fixed (non-learnable) physical pre-filter + a small learnable
1x1->depthwise->1x1 head (channel anchored, few params). Fixed kernels keep
experts physically specialised and resist overfitting on small data.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EXPERT_NAMES


class _AnchoredExpert(nn.Module):
    """fixed depthwise prefilter (broadcast over channels) + small learnable head."""

    def __init__(self, in_channel, out_channel, kernel, channel=32):
        super().__init__()
        self.register_buffer("kernel", kernel.view(1, 1, *kernel.shape))
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, out_channel, 1, bias=False),
        )

    def _prefilter(self, x):
        c = x.shape[1]
        k = self.kernel.expand(c, 1, -1, -1)
        return F.conv2d(x, k, padding=self.kernel.shape[-1] // 2, groups=c)

    def forward(self, x):
        return self.head(self._prefilter(x))


class _LearnablePrefilter(nn.Module):
    """Depthwise prefilter shared across channels, physics-initialised but learnable.

    A single [kh,kw] kernel is broadcast over all input channels (groups=C) so the
    physical prior is anchored at init yet can drift during training. Supports
    anisotropic (e.g. 7x1) and dilated kernels for direction/scale-specialised experts.
    """

    def __init__(self, kernel, dilation=1):
        super().__init__()
        kh, kw = kernel.shape
        self.kh, self.kw, self.d = kh, kw, dilation
        self.weight = nn.Parameter(kernel.view(1, 1, kh, kw).clone().float())

    def forward(self, x):
        c = x.shape[1]
        k = self.weight.expand(c, 1, self.kh, self.kw)
        pad = (self.d * (self.kh // 2), self.d * (self.kw // 2))
        return F.conv2d(x, k, padding=pad, dilation=self.d, groups=c)


# Fixed physical kernels (3x3).
def _kernels():
    box = torch.ones(3, 3) / 9.0                                   # despeckle (low-pass)
    lap = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)  # edge
    vgrad = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)     # shadow (vertical atten.)
    vsum = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 2, 1]], dtype=torch.float32) / 4.0   # posterior (below-structure)
    gauss = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0  # contrast (low-freq)
    hf = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)        # hf texture (sharpen)
    return {
        "despeckle": box, "edge": lap, "shadow": vgrad,
        "posterior": vsum, "contrast": gauss, "hf": hf,
    }


def _hetero_kernels():
    """Per-expert init kernels + structural knobs (design table). Reuses _kernels() bases."""
    base = _kernels()
    box5 = torch.ones(5, 5) / 25.0                                   # large despeckle
    vgrad7 = torch.tensor([[1.0], [1.0], [1.0], [0.0],
                           [-1.0], [-1.0], [-1.0]])                  # 7x1 shadow (vertical atten.)
    post5 = torch.tensor([[0.0], [0.0], [1.0], [2.0], [1.0]]) / 4.0  # 5x1 below-structure posterior
    return {
        # name: dict(prefilters=[(kernel, dilation), ...], channel=int, use_se=bool)
        "despeckle": dict(prefilters=[(box5, 1), (base["despeckle"], 2)], channel=48, use_se=False),
        "edge":      dict(prefilters=[(base["edge"], 1)],                 channel=48, use_se=False),
        "shadow":    dict(prefilters=[(vgrad7, 1)],                       channel=32, use_se=False),
        "posterior": dict(prefilters=[(post5, 1)],                       channel=32, use_se=False),
        "contrast":  dict(prefilters=[(base["contrast"], 1)],            channel=48, use_se=True),
        "hf":        dict(prefilters=[(base["hf"], 1), (base["hf"], 2)], channel=32, use_se=False),
    }


class _HeteroExpert(nn.Module):
    """Physics-anchored heterogeneous expert: 1-2 learnable prefilter branches (summed),
    optional channel-SE gate, then a Dropout-regularised lightweight head."""

    def __init__(self, in_channel, out_channel, prefilters, channel=32, use_se=False):
        super().__init__()
        self.prefilters = nn.ModuleList(
            _LearnablePrefilter(k, d) for (k, d) in prefilters
        )
        self.use_se = use_se
        if use_se:
            hidden = max(8, in_channel // 16)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, hidden, 1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, in_channel, 1), nn.Sigmoid(),
            )
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(channel, out_channel, 1, bias=False),
        )

    def forward(self, x):
        y = self.prefilters[0](x)
        for pf in self.prefilters[1:]:
            y = y + pf(x)
        if self.use_se:
            y = y * self.se(x)
        return self.head(y)


def _build_hetero_experts(in_channel, out_channel, channel=32):
    specs = _hetero_kernels()
    return nn.ModuleList(
        _HeteroExpert(in_channel, out_channel,
                      prefilters=specs[name]["prefilters"],
                      channel=specs[name]["channel"],
                      use_se=specs[name]["use_se"])
        for name in EXPERT_NAMES
    )


def build_experts(in_channel, out_channel, channel=32):
    ks = _kernels()
    return nn.ModuleList(
        _AnchoredExpert(in_channel, out_channel, ks[name], channel)
        for name in EXPERT_NAMES
    )
