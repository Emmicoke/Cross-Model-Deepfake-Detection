"""Frequency-domain and baseline feature extractors."""
from __future__ import annotations
import numpy as np


def log_spectrum(X: np.ndarray, window: bool = True) -> np.ndarray:
    """X [N,H,W] -> centred log-magnitude spectrum M = log(1+|F|), shape [N,H,W].
    A Hann window suppresses edge leakage; the per-image mean is removed first (kills the DC spike)."""
    X = X - X.mean(axis=(1, 2), keepdims=True)
    if window:
        h, w = X.shape[1:]
        X = X * (np.hanning(h)[:, None] * np.hanning(w)[None, :]).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(X, axes=(1, 2)), axes=(1, 2))
    return np.log1p(np.abs(F)).astype(np.float32)


def azimuthal_average(S: np.ndarray) -> np.ndarray:
    """S [N,H,W] centred spectrum -> radial profile [N, H//2] (mean over rings of equal radius)."""
    n, h, w = S.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    nb = min(h, w) // 2
    mask = r < nb
    counts = np.bincount(r[mask], minlength=nb)
    flat = S.reshape(n, -1)[:, mask.ravel()]
    out = np.zeros((n, nb), np.float32)
    rr = r[mask]
    for k in range(nb):
        out[:, k] = flat[:, rr == k].mean(axis=1)
    return out


def standardize_map(S: np.ndarray, mu=None, sd=None):
    """Per-frequency z-score using training statistics (spectrum normalisation)."""
    if mu is None:
        mu, sd = S.mean(axis=0, keepdims=True), S.std(axis=0, keepdims=True) + 1e-6
    return (S - mu) / sd, mu, sd


def spatial_stats(X: np.ndarray) -> np.ndarray:
    """Naive baseline features: mean and variance of pixel values."""
    return np.stack([X.mean(axis=(1, 2)), X.var(axis=(1, 2))], axis=1)


def lbp_hist(X: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """Uniform LBP histogram per image (P+2 bins)."""
    from skimage.feature import local_binary_pattern
    out = np.zeros((len(X), P + 2), np.float32)
    for i, img in enumerate(X):
        lbp = local_binary_pattern((img * 255).astype(np.uint8), P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        out[i] = h / max(h.sum(), 1)
    return out
