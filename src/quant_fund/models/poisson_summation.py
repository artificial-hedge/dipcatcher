"""Poisson summation: sum f(n) = sum f_hat(k) for periodization (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def periodize_gauss(sigma: float, n_grid: int = 4000) -> np.ndarray:
    """Period-1 sum of a narrow Gaussian: theta function F(x) = sum_n exp(-(x-n)^2/(2s^2))."""
    x = np.linspace(0, 1, n_grid, endpoint=False)
    out = np.zeros(n_grid)
    for n in range(-10, 11):
        out += np.exp(-((x - n) ** 2) / (2 * sigma**2))
    return out


def poisson_rhs(sigma: float, k_max: int = 50) -> np.ndarray:
    """sum_k f_hat(k) e^{2pi i k x} with f_hat(k) = s*sqrt(2pi) exp(-2 pi^2 s^2 k^2)."""
    x = np.linspace(0, 1, 4000, endpoint=False)
    out = np.zeros(len(x))
    for k in range(-k_max, k_max + 1):
        c = sigma * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi**2 * sigma**2 * k * k)
        out += c * np.cos(2 * np.pi * k * x)
    return np.asarray(out)


def _bench_poisson_summation(seed: int = 0) -> float:
    checks = []
    sigma = 0.05
    lhs = periodize_gauss(sigma)
    rhs = poisson_rhs(sigma)
    checks.append(float(np.max(np.abs(lhs - rhs))) < 1e-8)
    # at sigma -> small, F(0) -> 1 (only n=0 contributes at x=0)
    checks.append(abs(periodize_gauss(0.01)[0] - 1.0) < 1e-6)
    # mean of F over period = integral of f = s*sqrt(2pi)
    checks.append(abs(float(np.mean(lhs)) - sigma * np.sqrt(2 * np.pi)) < 1e-6)
    # wide sigma -> F nearly constant equal to s*sqrt(2pi)
    wide = periodize_gauss(1.0)
    checks.append(float(np.ptp(wide)) < 1e-6)
    checks.append(abs(float(np.mean(wide)) - np.sqrt(2 * np.pi)) < 1e-6)
    return float(sum(checks) / len(checks))


def bench_poisson_summation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poisson_summation": _bench_poisson_summation(seed)}
