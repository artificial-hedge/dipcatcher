"""Heisenberg uncertainty: Delta_t * Delta_f >= 1/(4 pi) with equality for Gaussians (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def time_spread(f: np.ndarray, t: np.ndarray) -> float:
    p = np.abs(f) ** 2
    p = p / p.sum()
    mu = float(np.sum(t * p))
    return float(np.sqrt(np.sum((t - mu) ** 2 * p)))


def freq_spread(f: np.ndarray, t: np.ndarray) -> float:
    fh = np.fft.fft(f)
    p = np.abs(fh) ** 2
    p = p / p.sum()
    n = len(f)
    dt = t[1] - t[0]
    w = np.fft.fftfreq(n, dt) * 2 * np.pi
    mu = float(np.sum(w * p))
    return float(np.sqrt(np.sum((w - mu) ** 2 * p)))


def _bench_uncertainty(seed: int = 0) -> float:
    checks = []
    t = np.linspace(-10, 10, 8192)
    # Gaussian achieves equality: sigma_t * sigma_w = 1/2 in angular freq (i.e. >= 1/2)
    for s in (0.5, 1.0, 2.0):
        g = np.exp(-(t**2) / (2 * s**2))
        prod = time_spread(g, t) * freq_spread(g, t)
        checks.append(abs(prod - 0.5) < 0.02)
    # a top-hat has a larger product (not minimal uncertainty)
    h = (np.abs(t) < 1.0).astype(float)
    checks.append(time_spread(h, t) * freq_spread(h, t) > 0.5)
    # narrower in time -> wider in frequency
    g1 = np.exp(-(t**2) / (2 * 0.5**2))
    g2 = np.exp(-(t**2) / (2 * 2.0**2))
    checks.append(time_spread(g1, t) < time_spread(g2, t))
    checks.append(freq_spread(g1, t) > freq_spread(g2, t))
    return float(min(1.0, sum(checks) / len(checks)))


def bench_uncertainty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uncertainty": _bench_uncertainty(seed)}
