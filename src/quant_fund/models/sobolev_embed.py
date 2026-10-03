"""Sobolev embedding H^1 -> C^0 in one dimension (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def h1_norm(u: np.ndarray, du: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.trapezoid(u**2 + du**2, dx=dx)))


def sup_norm(u: np.ndarray) -> float:
    return float(np.max(np.abs(u)))


def _bench_sobolev_embed(seed: int = 0) -> float:
    checks = []
    x = np.linspace(0, 1, 4001)
    # u with a narrow spike: sup norm large vs L2 small; H1 controls sup
    u = np.exp(-((x - 0.5) ** 2) / (2 * 0.01**2))
    du = np.gradient(u, x)
    checks.append(sup_norm(u) <= 2.0 * h1_norm(u, du, x[1] - x[0]))
    # mean-zero functions on the circle: ||u||_inf <= C ||u'||_2 (Poincare-Wirtinger)
    w = np.sin(2 * np.pi * x)
    dw = np.gradient(w, x)
    checks.append(sup_norm(w) <= np.sqrt(np.trapezoid(dw**2, dx=x[1] - x[0])) * 0.5 + 1e-9)
    # constant function: sup = L2 (embedding constant >= 1)
    c = np.ones_like(x)
    checks.append(abs(sup_norm(c) - h1_norm(c, np.zeros_like(x), x[1] - x[0])) < 1e-9)
    # scaling: bump of width eps has H1 ~ eps^{-1/2} * sup
    for eps in (0.05, 0.01):
        b = np.exp(-((x - 0.5) ** 2) / (2 * eps**2))
        db = np.gradient(b, x)
        ratio = h1_norm(b, db, x[1] - x[0]) / sup_norm(b)
        checks.append(ratio > 1.0)
    return float(min(1.0, sum(checks) / len(checks)))


def bench_sobolev_embed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sobolev_embed": _bench_sobolev_embed(seed)}
