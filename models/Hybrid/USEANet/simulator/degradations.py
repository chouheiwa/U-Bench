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


def _depth_ramp(img):
    """Row index normalised to [0, 1] along the depth (H) axis, shaped [1,1,H,1]."""
    h = img.shape[-2]
    ramp = torch.linspace(0.0, 1.0, h, device=img.device, dtype=img.dtype)
    return ramp.view(1, 1, h, 1)


def depth_attenuation(img, coeff=0.5):
    """Exponential brightness falloff with depth (deeper rows are dimmer)."""
    ramp = _depth_ramp(img)
    atten = torch.exp(-coeff * ramp)            # [1,1,H,1], 1 at top -> small at bottom
    out = torch.clamp(img * atten, 0.0, 1.0)
    atten_map = (1.0 - atten).expand_as(img)     # strong (->1) where most attenuated
    return out, atten_map


def _vertical_band(img, col_start, col_end, row_start):
    """Soft mask: 1 inside columns [col_start,col_end) for rows >= row_start."""
    b, _, h, w = img.shape
    cols = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, 1, w)
    rows = torch.arange(h, device=img.device, dtype=img.dtype).view(1, 1, h, 1)
    col_mask = ((cols >= col_start) & (cols < col_end)).to(img.dtype)
    row_mask = (rows >= row_start).to(img.dtype)
    return (col_mask * row_mask).expand(b, 1, h, w)


def acoustic_shadow(img, col_start=None, col_end=None, row_start=None, strength=0.6):
    """Darken a vertical column band below a (highly attenuating) structure."""
    _, _, h, w = img.shape
    col_start = w // 3 if col_start is None else col_start
    col_end = 2 * w // 3 if col_end is None else col_end
    row_start = h // 2 if row_start is None else row_start
    band = _vertical_band(img, col_start, col_end, row_start)
    out = torch.clamp(img * (1.0 - strength * band), 0.0, 1.0)
    return out, band * strength


def posterior_enhancement(img, col_start=None, col_end=None, row_start=None, strength=0.4):
    """Brighten a vertical column band below an (anechoic) structure."""
    _, _, h, w = img.shape
    col_start = w // 3 if col_start is None else col_start
    col_end = 2 * w // 3 if col_end is None else col_end
    row_start = h // 2 if row_start is None else row_start
    band = _vertical_band(img, col_start, col_end, row_start)
    out = torch.clamp(img + strength * band * (1.0 - img), 0.0, 1.0)
    return out, band * strength
