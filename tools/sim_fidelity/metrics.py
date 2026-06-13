"""Fidelity metrics comparing synthetic vs real ultrasound batches.

All inputs are float32 numpy arrays ``[N, H, W]`` in ``[0, 1]``.
"""
import numpy as np
from scipy.stats import wasserstein_distance


def intensity_wasserstein(real, synth, bins=256):
    """Wasserstein-1 distance between flattened intensity distributions."""
    r = np.clip(real.reshape(-1), 0.0, 1.0)
    s = np.clip(synth.reshape(-1), 0.0, 1.0)
    return float(wasserstein_distance(r, s))


def speckle_snr(images, patch=16):
    """Median local SNR (mean/std) over non-overlapping patches.

    Fully-developed Rayleigh speckle has a characteristic SNR ~ 1.91; this
    statistic lets the gate compare synthetic speckle texture against real.
    """
    snrs = []
    for img in images:
        h, w = img.shape
        for i in range(0, h - patch + 1, patch):
            for j in range(0, w - patch + 1, patch):
                p = img[i:i + patch, j:j + patch]
                mu, sd = float(p.mean()), float(p.std())
                if sd > 1e-6:
                    snrs.append(mu / sd)
                else:
                    # Zero-variance patch: perfectly flat, SNR is unbounded.
                    snrs.append(np.inf)
    return float(np.median(snrs)) if snrs else 0.0


def _radial_profile(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def radial_spectrum_distance(real, synth):
    """L1 distance between mean (log) radially-averaged power spectra."""
    def mean_profile(batch):
        profs = [np.log1p(_radial_profile(im)) for im in batch]
        n = min(len(p) for p in profs)
        return np.mean([p[:n] for p in profs], axis=0)
    pr = mean_profile(real)
    ps = mean_profile(synth)
    n = min(len(pr), len(ps))
    denom = np.abs(pr[:n]).mean() + 1e-6
    return float(np.abs(pr[:n] - ps[:n]).mean() / denom)
