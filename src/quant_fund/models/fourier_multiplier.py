"""Fourier multipliers: m(D)u = F^{-1}(m(k) F u) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def apply_multiplier(u: np.ndarray, m: np.ndarray) -> np.ndarray:
    uh = np.fft.fft(u)
    return np.asarray(np.real(np.fft.ifft(uh * m)))


def wave_numbers(n: int, period: float = 1.0) -> np.ndarray:
    return np.asarray(np.fft.fftfreq(n, period / n) * 2 * np.pi)


def _bench_fourier_multiplier(seed: int = 0) -> float:
    checks = []
    n = 512
    x = np.arange(n) / n
    u = np.sin(2 * np.pi * x) + 0.5 * np.cos(6 * np.pi * x)
    k = wave_numbers(n)
    # derivative multiplier ik: du/dx
    du = apply_multiplier(u, 1j * k)
    exact_du = 2 * np.pi * np.cos(2 * np.pi * x) - 3 * np.pi * np.sin(6 * np.pi * x)
    checks.append(float(np.max(np.abs(du - exact_du))) < 1e-8)
    # Laplacian multiplier -k^2
    d2u = apply_multiplier(u, -(k**2))
    exact_d2 = -4 * np.pi**2 * np.sin(2 * np.pi * x) - 18 * np.pi**2 * np.cos(6 * np.pi * x)
    checks.append(float(np.max(np.abs(d2u - exact_d2))) < 1e-7)
    # low-pass multiplier keeps low modes only
    lp = apply_multiplier(u, (np.abs(k) <= 4 * np.pi).astype(float))
    checks.append(np.allclose(lp, np.sin(2 * np.pi * x), atol=1e-10))
    # shift multiplier exp(i k x0) translates periodic function
    sh = apply_multiplier(u, np.exp(-1j * k * 0.25))
    checks.append(np.allclose(sh, np.roll(u, n // 4), atol=1e-9))
    return float(sum(checks) / len(checks))


def bench_fourier_multiplier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_multiplier": _bench_fourier_multiplier(seed)}
