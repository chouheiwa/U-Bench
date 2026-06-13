"""Differentiable ultrasound image-formation degradation ops.

``log_compression`` is a global tonal transform returning a bare image tensor.
The localized degradation ops (speckle, attenuation, shadow, posterior) take a
grayscale image ``[B, 1, H, W]`` in ``[0, 1]`` and return
``(degraded_image, degradation_map)`` where ``degradation_map`` (also
``[B, 1, H, W]``, in ``[0, 1]``) marks *where/how strongly* that degradation
was applied. The maps double as physical ground truth for router supervision
(Track C). All ops are autograd-differentiable w.r.t. the input image.
"""
import torch


def log_compression(img, dynamic_range_db=50.0):
    """Mimic the log compression of the ultrasound scan-conversion pipeline."""
    floor = 10.0 ** (-dynamic_range_db / 20.0)
    x = torch.clamp(img, floor, 1.0)
    out = (20.0 * torch.log10(x) + dynamic_range_db) / dynamic_range_db
    return torch.clamp(out, 0.0, 1.0)


def add_speckle(img, sigma=0.3, distribution="rayleigh", generator=None):
    """Multiplicative speckle (Rayleigh-like fully-developed speckle).

    Returns ``(degraded, speckle_map)`` where ``speckle_map`` is the normalised
    magnitude of the multiplicative perturbation (high where speckle is strong).
    """
    if distribution == "rayleigh":
        # Rayleigh magnitude = sqrt(n1^2 + n2^2) of two N(0, sigma) draws.
        n1 = torch.empty_like(img).normal_(0.0, sigma, generator=generator)
        n2 = torch.empty_like(img).normal_(0.0, sigma, generator=generator)
        mult = torch.sqrt(n1 * n1 + n2 * n2)
        mult = mult / (sigma * (3.14159265 / 2.0) ** 0.5 + 1e-6)  # mean ~= 1
    else:
        raise ValueError(f"unknown speckle distribution: {distribution}")
    out = torch.clamp(img * mult, 0.0, 1.0)
    speckle_map = torch.clamp((mult - 1.0).abs(), 0.0, 1.0)
    return out, speckle_map
