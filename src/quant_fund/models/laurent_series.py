"""Laurent coefficients of f on a circle via Fourier extraction (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def laurent_coeffs(f, radius: float, center: complex = 0j, n: int = 512) -> dict[int, complex]:
    """c_k = (1/2pi) int f(c + r e^{it}) (r e^{it})^{-k} dt for k in range."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = center + radius * np.exp(1j * t)
    vals = f(z)
    coeffs: dict[int, complex] = {}
    half = n // 4
    for k in range(-half, half):
        coeffs[k] = complex(np.mean(vals * np.exp(-1j * k * t)) * radius ** (-k))
    return coeffs


def _bench_laurent_series(seed: int = 0) -> float:
    checks = []
    # f(z) = 1/(1 - z) on |z| = 0.5: geometric series, all c_k = 1 for k >= 0
    cs = laurent_coeffs(lambda z: 1.0 / (1.0 - z), radius=0.5)
    checks.append(all(abs(cs[k] - 1.0) < 1e-6 for k in range(0, 8)))
    # negative coefficients vanish (f analytic inside |z|=0.5)
    checks.append(all(abs(cs[k]) < 1e-6 for k in range(-8, 0)))
    # f(z) = e^{1/z}: Laurent coeffs c_k = 1/(-k)! for k <= 0; c_0 = 1, c_{-1} = 1, c_{-2} = 1/2
    ce = laurent_coeffs(lambda z: np.exp(1.0 / z), radius=1.0)
    checks.append(abs(ce[0] - 1.0) < 1e-6)
    checks.append(abs(ce[-1] - 1.0) < 1e-6)
    checks.append(abs(ce[-2] - 0.5) < 1e-6)
    # polynomial f = z^2 + 3z: c_2 = 1, c_1 = 3, others 0
    cp = laurent_coeffs(lambda z: z**2 + 3 * z, radius=1.0)
    checks.append(abs(cp[2] - 1.0) < 1e-9 and abs(cp[1] - 3.0) < 1e-9 and abs(cp[0]) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_laurent_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_laurent_series": _bench_laurent_series(seed)}
