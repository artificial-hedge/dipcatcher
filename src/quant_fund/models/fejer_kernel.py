"""Fejer vs Dirichlet kernels: Cesaro means kill the Gibbs overshoot (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def dirichlet_partial(f_vals: np.ndarray, n_modes: int) -> np.ndarray:
    """Partial Fourier sum of a sampled periodic function."""
    n = len(f_vals)
    fhat = np.fft.fft(f_vals) / n
    modes = np.arange(-n_modes, n_modes + 1)
    x = np.arange(n) / n
    out = np.zeros(n, dtype=complex)
    for k in modes:
        out += fhat[k % n] * np.exp(2j * np.pi * k * x)
    return np.asarray(np.real(out))


def fejer_mean(f_vals: np.ndarray, n_modes: int) -> np.ndarray:
    """Cesaro mean of partial sums."""
    acc = np.zeros(len(f_vals))
    for m in range(1, n_modes + 1):
        acc += dirichlet_partial(f_vals, m)
    return np.asarray(acc / n_modes)


def _bench_fejer_kernel(seed: int = 0) -> float:
    checks = []
    n = 512
    x = np.arange(n) / n
    # square wave: Gibbs overshoot ~9% of jump near discontinuity
    sq = np.where(x < 0.5, 1.0, -1.0)
    d = dirichlet_partial(sq, 40)
    f = fejer_mean(sq, 40)
    checks.append(float(np.max(np.abs(d))) > 1.05)  # Gibbs overshoot present
    checks.append(float(np.max(np.abs(f))) < 1.02)  # Fejer stays inside range
    # Fejer kernel is nonnegative -> positivity preserving: the Cesaro
    # mean of a nonnegative signal stays ~>= 0 (up to discretization),
    # while the Dirichlet partial sum undershoots below the signal floor
    box = np.where((x > 0.3) & (x < 0.7), 1.0, 0.0)
    fb = fejer_mean(box, 40)
    db = dirichlet_partial(box, 40)
    checks.append(float(fb.min()) > -0.02)
    checks.append(float(db.min()) < -0.05)
    # away from the jump Fejer still converges
    mid = n // 4
    checks.append(abs(f[mid] - 1.0) < 0.05)
    # smooth function: both converge well
    sm = np.sin(2 * np.pi * x)
    checks.append(float(np.max(np.abs(dirichlet_partial(sm, 10) - sm))) < 1e-9)
    checks.append(float(np.max(np.abs(fejer_mean(sm, 10) - sm))) < 0.02)
    return float(sum(checks) / len(checks))


def bench_fejer_kernel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fejer_kernel": _bench_fejer_kernel(seed)}
