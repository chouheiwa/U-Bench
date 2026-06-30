"""Label-free degradation proxy maps, one channel per expert (design §3.1).

Computed from the channel-mean of a feature map (or grayscale image). Each
channel is a cheap cue for *where* that expert's degradation is present; the
stack is softmax-normalised across experts per position so it can directly
supervise the router gate.
"""
import torch
import torch.nn.functional as F

from . import EXPERT_NAMES


def _local_variance(g, k=3):
    pad = k // 2
    mean = F.avg_pool2d(g, k, 1, pad)
    mean_sq = F.avg_pool2d(g * g, k, 1, pad)
    return torch.clamp(mean_sq - mean * mean, min=0.0)


def _gradient_magnitude(g):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=g.dtype, device=g.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _vertical_attenuation(g):
    # Brightness deficit relative to the shallow (top) rows of each column.
    top = g[..., :max(1, g.shape[-2] // 8), :].mean(dim=-2, keepdim=True)
    return torch.clamp(top - g, min=0.0)


def _posterior_brightness(g):
    # Local excess brightness vs a large neighbourhood mean (posterior enhance).
    bg = F.avg_pool2d(g, 9, 1, 4)
    return torch.clamp(g - bg, min=0.0)


def _low_freq_energy(g):
    return F.avg_pool2d(g, 9, 1, 4)


def _high_freq_energy(g):
    return torch.clamp(g - F.avg_pool2d(g, 5, 1, 2), min=0.0).abs()


def degradation_proxies(feat, eps=1e-6, num_experts=None):
    """feat: [B,C,H,W] -> proxy [B, N, H, W], softmax over experts.

    ``num_experts`` (ablation switch) selects the first N physical cues from
    ``EXPERT_NAMES``; default None -> all 6 (byte-identical to the original).
    """
    names = EXPERT_NAMES if num_experts is None else EXPERT_NAMES[:num_experts]
    g = feat.mean(dim=1, keepdim=True)  # [B,1,H,W]
    cues = {
        "despeckle": _local_variance(g),
        "edge": _gradient_magnitude(g),
        "shadow": _vertical_attenuation(g),
        "posterior": _posterior_brightness(g),
        "contrast": _low_freq_energy(g),
        "hf": _high_freq_energy(g),
    }
    stack = torch.cat([cues[name] for name in names], dim=1)  # [B,N,H,W]
    # Divide each channel by its spatial maximum so all channels live in [0,1]
    # while preserving the per-position ratio between channels (unlike min-max
    # which can over-equalise cues that happen to peak at the same location).
    flat = stack.flatten(2)
    mx = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)  # [B,6,1,1]
    norm = stack / (mx + eps)
    return F.softmax(norm, dim=1)
