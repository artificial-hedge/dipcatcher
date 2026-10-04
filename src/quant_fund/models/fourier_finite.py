"""DFT on Z_n: unitarity, Parseval, convolution theorem (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    k = np.arange(n)
    m = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return np.asarray(m @ x)


def idft(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    k = np.arange(n)
    m = np.exp(2j * np.pi * np.outer(k, k) / n)
    return np.asarray(m @ y / n)


def circular_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n)
    for m_ in range(n):
        out[m_] = sum(x[j] * h[(m_ - j) % n] for j in range(n))
    return out


def _bench_fourier_finite(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(8)
    xh = dft(x)
    # inversion
    checks.append(np.allclose(idft(xh), x))
    # Parseval: ||x||^2 = ||X||^2 / n
    checks.append(np.isclose(np.sum(x**2), np.sum(np.abs(xh) ** 2) / 8))
    # convolution theorem
    h = rng.standard_normal(8)
    checks.append(np.allclose(dft(circular_conv(x, h)), xh * dft(h)))
    # shift theorem: DFT of x shifted by 1 = X_k * exp(-2pi i k/n)
    xs = np.roll(x, 1)
    k = np.arange(8)
    checks.append(np.allclose(dft(xs), xh * np.exp(-2j * np.pi * k / 8)))
    # DC component = sum
    checks.append(np.isclose(xh[0].real, np.sum(x)))
    return float(sum(checks) / len(checks))


def bench_fourier_finite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_finite": _bench_fourier_finite(seed)}
