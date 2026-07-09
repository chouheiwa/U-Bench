"""Calibrated acoustic-physics estimator head (new-paper Pillar 1).

Upgrades the hand-crafted ``degradation_proxies`` (fixed image filters) into a
*learnable* head that estimates **calibrated acoustic physics maps** — the
speckle Nakagami-m shape, the depth-wise attenuation slope, and the local SNR —
and derives the router proxy from them.

Why calibrated physics rather than generic image cues: Nakagami-m and the
attenuation slope are *device/protocol-invariant acoustic properties* of the
tissue-transducer interaction, so a router keyed on them is expected to transfer
across datasets far better than one keyed on raw variance/gradient statistics
(the cross-domain claim, contributions 2/3).

The estimator is weakly supervised toward classical moment estimates of these
quantities (``acoustic_pseudo_gt``), computed online from the same feature map —
no offline precompute, no simulator, no manual labels.

Drop-in: ``forward(feat) -> proxy [B, E, H, W]`` (softmax over experts), the same
contract as ``degradation_proxies``, so ``DegradationAwareRouter`` is untouched.
Gated behind ``USEANET_CALIB_PROXY=1`` (default off -> PUMA reproduces exactly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Physical quantities the head estimates, in channel order. Kept as a module
# constant so per-quantity ablations (e.g. Nakagami-only) can slice it.
PHYS_NAMES = ["nakagami_m", "attenuation", "snr"]


def _perchannel_unit_norm(x, eps=1e-6):
    """Scale each [B,C,H,W] channel by its per-sample spatial max -> [0,1].

    Matches ``degradation_proxies``' normalisation so the estimator output and
    its pseudo-GT live on the same scale for the L1 calibration loss.
    """
    flat = x.flatten(2)
    mx = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)  # [B,C,1,1]
    return x / (mx + eps)


def acoustic_pseudo_gt(feat, k=5, eps=1e-6):
    """Classical moment estimates of the acoustic maps -> [B, 3, H, W] in [0,1].

    Weak-supervision target for the estimator; detach before use as a label.

    - nakagami_m: local Nakagami shape m = (E[R^2])^2 / Var(R^2), R the B-mode
      envelope (grayscale proxy). m ~ 1 = fully developed speckle, m > 1 =
      structured/specular, m < 1 = pre-Rayleigh. Windowed via avg_pool.
    - attenuation: depth-wise brightness deficit vs the shallow rows, a proxy for
      cumulative acoustic attenuation / shadowing along each A-line (column).
    - snr: local mean / std (Rayleigh SNR ~ 1.91 for pure speckle); high where
      signal is coherent, low in noisy/anechoic regions.
    """
    # B-mode envelope is a non-negative amplitude; take |mean| so the acoustic
    # moments stay physically valid (and unit-norm well-defined) even when the
    # feature map carries signed activations.
    g = feat.mean(dim=1, keepdim=True).abs()               # [B,1,H,W] envelope proxy
    pad = k // 2
    r2 = g * g
    m2 = F.avg_pool2d(r2, k, 1, pad)                        # E[R^2]
    m4 = F.avg_pool2d(r2 * r2, k, 1, pad)                   # E[R^4]
    var_r2 = torch.clamp(m4 - m2 * m2, min=eps)
    nakagami = (m2 * m2) / var_r2                           # m, in (0, inf)
    nakagami = torch.clamp(nakagami, max=10.0)              # tame the long tail

    top = g[..., :max(1, g.shape[-2] // 8), :].mean(dim=-2, keepdim=True)
    attenuation = torch.clamp(top - g, min=0.0)            # brightness deficit

    mean = F.avg_pool2d(g, k, 1, pad)
    meansq = F.avg_pool2d(g * g, k, 1, pad)
    std = torch.sqrt(torch.clamp(meansq - mean * mean, min=eps))
    snr = mean / (std + eps)

    stack = torch.cat([nakagami, attenuation, snr], dim=1)  # [B,3,H,W]
    return _perchannel_unit_norm(stack, eps)


class PhysicsEstimator(nn.Module):
    """Lightweight head: feature map -> calibrated physics maps -> router proxy.

    Params are tiny (two 3x3 convs + a 1x1 projection) to preserve USEANet's
    lightweight budget. ``last_phys`` is stashed for the calibration loss and for
    interpretability (the physics maps are directly visualisable in [0,1]).
    """

    def __init__(self, in_channel, num_experts, n_phys=3, hidden=32):
        super().__init__()
        # Per-quantity ablation (USEANET_CALIB_PHYS): comma-separated subset of
        # PHYS_NAMES, e.g. "nakagami_m" or "attenuation,snr"; empty = all 3.
        # Isolates which calibrated acoustic quantity drives the routing gain.
        import os
        sel = os.environ.get("USEANET_CALIB_PHYS", "").strip()
        if sel:
            self.phys_idx = [PHYS_NAMES.index(s.strip()) for s in sel.split(",") if s.strip()]
        else:
            self.phys_idx = list(range(len(PHYS_NAMES)))
        n_phys = len(self.phys_idx)
        self.n_phys = n_phys
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channel, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_phys, 3, padding=1),
        )
        # Project calibrated physics maps to per-expert routing logits.
        self.to_proxy = nn.Conv2d(n_phys, num_experts, 1)
        self.last_phys = None

    def forward(self, feat):
        phys = torch.sigmoid(self.trunk(feat))              # [B,n_phys,H,W] in [0,1]
        self.last_phys = phys
        logits = self.to_proxy(phys)                        # [B,E,H,W]
        return F.softmax(logits, dim=1)

    def calib_target(self, feat):
        """Detached classical pseudo-GT for ``last_phys`` (selected quantities)."""
        gt = acoustic_pseudo_gt(feat)                       # [B,3,H,W]
        return gt[:, self.phys_idx].detach()
