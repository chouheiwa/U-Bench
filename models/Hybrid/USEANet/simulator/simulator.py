"""Composable differentiable ultrasound degradation simulator.

Used three ways (see design doc §4): online augmentation, physical GT for
router supervision (Track C), and controllable degradation experiments. This
module has NO dependency on the segmentation model and must never import it.
"""
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from .degradations import (
    log_compression, add_speckle, depth_attenuation,
    acoustic_shadow, posterior_enhancement,
)


@dataclass
class SimOutput:
    image: torch.Tensor
    degradation_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    applied: Dict[str, float] = field(default_factory=dict)


class UltrasoundDegradationSimulator(nn.Module):
    """Apply log-compression + speckle + depth attenuation + (random) shadow /
    posterior-enhancement bands, scaled by ``intensity`` in ``[0, 1]``.
    """

    def __init__(self, seed=None, speckle_sigma=0.3, atten_coeff=0.6,
                 shadow_strength=0.7, posterior_strength=0.5, dynamic_range_db=50.0):
        super().__init__()
        self.speckle_sigma = speckle_sigma
        self.atten_coeff = atten_coeff
        self.shadow_strength = shadow_strength
        self.posterior_strength = posterior_strength
        self.dynamic_range_db = dynamic_range_db
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def _rand_band(self, w):
        # Random column band ~ one third of the width.
        width = max(1, w // 3)
        start = int(torch.randint(0, max(1, w - width), (1,), generator=self._gen).item())
        return start, start + width

    def forward(self, img, intensity=1.0):
        if img.shape[1] != 1:
            img = img.mean(dim=1, keepdim=True)
        h, w = img.shape[-2:]
        gen = torch.Generator(device=img.device)
        gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=self._gen).item()))

        x = log_compression(img, self.dynamic_range_db)

        x, speckle_map = add_speckle(
            x, sigma=self.speckle_sigma * intensity, generator=gen)

        x, atten_map = depth_attenuation(x, coeff=self.atten_coeff * intensity)

        cs, ce = self._rand_band(w)
        x, shadow_map = acoustic_shadow(
            x, col_start=cs, col_end=ce, row_start=h // 2,
            strength=self.shadow_strength * intensity)

        cs2, ce2 = self._rand_band(w)
        x, posterior_map = posterior_enhancement(
            x, col_start=cs2, col_end=ce2, row_start=h // 2,
            strength=self.posterior_strength * intensity)

        return SimOutput(
            image=x,
            degradation_maps={
                "speckle": speckle_map,
                "attenuation": atten_map,
                "shadow": shadow_map,
                "posterior": posterior_map,
            },
            applied={
                "speckle": self.speckle_sigma * intensity,
                "attenuation": self.atten_coeff * intensity,
                "shadow": self.shadow_strength * intensity,
                "posterior": self.posterior_strength * intensity,
            },
        )
