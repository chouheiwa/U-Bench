"""Anchored ultrasound-physics experts (design §3.1).

Each expert = a fixed (non-learnable) physical pre-filter + a small learnable
1x1->depthwise->1x1 head (channel anchored, few params). Fixed kernels keep
experts physically specialised and resist overfitting on small data.
"""
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


def build_experts(in_channel, out_channel, channel=32):
    ks = _kernels()
    return nn.ModuleList(
        _AnchoredExpert(in_channel, out_channel, ks[name], channel)
        for name in EXPERT_NAMES
    )
