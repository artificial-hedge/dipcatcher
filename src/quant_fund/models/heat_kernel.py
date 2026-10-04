"""Heat kernel: mass conservation + convolution solves u_t = u_xx (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def heat_kernel(x: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(np.exp(-(x**2) / (4 * t)) / np.sqrt(4 * np.pi * t))


def convolve_periodic(k: np.ndarray, f: np.ndarray, dx: float) -> np.ndarray:
    n = len(f)
    out = np.zeros(n)
    for i in range(n):
        out[i] = float(np.sum(k[(i - np.arange(n)) % n] * f)) * dx
    return out


def _bench_heat_kernel(seed: int = 0) -> float:
    checks = []
    # kernel integrates to 1
    x = np.linspace(-20, 20, 40001)
    for t in (0.5, 1.0, 2.0):
        checks.append(abs(float(np.trapezoid(heat_kernel(x, t), x)) - 1.0) < 1e-8)
    # semigroup: K_t * K_s = K_{t+s} on a wide periodic grid
    n = 2048
    xs = np.linspace(-16, 16, n, endpoint=False)
    dx = xs[1] - xs[0]
    k_c = np.fft.ifftshift(heat_kernel(np.linspace(-16, 16, n, endpoint=False), 1.0))
    conv = convolve_periodic(
        k_c, np.fft.ifftshift(heat_kernel(np.linspace(-16, 16, n, endpoint=False), 0.5)), dx
    )
    k15 = np.fft.ifftshift(heat_kernel(np.linspace(-16, 16, n, endpoint=False), 1.5))
    checks.append(float(np.max(np.abs(conv - k15))) < 1e-3)
    # delta initial data -> kernel itself
    checks.append(abs(float(np.sum(k_c) * dx) - 1.0) < 1e-6)
    return float(sum(checks) / len(checks))


def bench_heat_kernel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heat_kernel": _bench_heat_kernel(seed)}
