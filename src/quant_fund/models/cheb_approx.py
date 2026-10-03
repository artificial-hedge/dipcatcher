"""Chebyshev interpolation at Chebyshev-Lobatto nodes (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def cheb_nodes(n: int) -> np.ndarray:
    """Chebyshev-Lobatto nodes x_j = cos(j pi / n), j = 0..n."""
    return np.cos(np.pi * np.arange(n + 1) / n)


def cheb_interp(f, n: int, xs: np.ndarray) -> np.ndarray:
    """Evaluate the degree-n interpolant of f at Lobatto nodes on xs."""
    nodes = cheb_nodes(n)
    vals = f(nodes)
    # barycentric weights for Lobatto nodes: w_j = (-1)^j * (1/2 at ends)
    w = np.ones(n + 1)
    w[0] = w[-1] = 0.5
    w *= (-1.0) ** np.arange(n + 1)
    num = np.zeros_like(xs, dtype=float)
    den = np.zeros_like(xs, dtype=float)
    for j in range(n + 1):
        d = xs - nodes[j]
        # avoid exact node hits
        nz = np.abs(d) >= 1e-14
        term = np.zeros_like(d)
        term[nz] = w[j] / d[nz]
        num += term * vals[j]
        den += term
    # exact hits
    for j in range(n + 1):
        hit = np.abs(xs - nodes[j]) < 1e-14
        num = np.where(hit, vals[j], num)
        den = np.where(hit, 1.0, den)
    return np.asarray(num / den)


def _bench_cheb_approx(seed: int = 0) -> float:
    checks = []
    f = np.exp
    xs = np.linspace(-1.0, 1.0, 500)
    err = np.max(np.abs(cheb_interp(f, 12, xs) - f(xs)))
    checks.append(err < 1e-9)
    # interpolates exactly at nodes
    nodes = cheb_nodes(12)
    checks.append(np.max(np.abs(cheb_interp(f, 12, nodes) - f(nodes))) < 1e-12)
    # Runge phenomenon controlled: equispaced interpolation of 1/(1+25x^2)
    # diverges at edges while Chebyshev converges
    def g(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + 25.0 * x**2)
    c_err = np.abs(cheb_interp(g, 15, np.array([0.95])) - g(np.array([0.95])))[0]
    checks.append(c_err < 0.05)
    # polynomial exactness: degree-12 interp of x^5 is exact
    def h(x: np.ndarray) -> np.ndarray:
        return x**5
    checks.append(np.max(np.abs(cheb_interp(h, 12, xs) - h(xs))) < 1e-9)
    # error decreases with n for smooth f
    e6 = np.max(np.abs(cheb_interp(f, 6, xs) - f(xs)))
    checks.append(err < e6)
    return float(sum(checks) / len(checks))


def bench_cheb_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cheb_approx": _bench_cheb_approx(seed)}
