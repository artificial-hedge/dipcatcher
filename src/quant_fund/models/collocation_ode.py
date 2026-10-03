"""Spectral collocation for ODE boundary value problems (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _cheb_diff_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Trefethen cheb differentiation matrix on Lobatto nodes."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(n + 1)
    xg = np.tile(x[:, None], (1, n + 1))
    dx = xg - xg.T
    d = (np.outer(c, 1.0 / c)) / (dx + np.eye(n + 1))
    d -= np.diag(np.sum(d, axis=1))
    return d, x


def _bench_collocation_ode(seed: int = 0) -> float:
    checks = []
    # y' = -y, y(0) = 1 on [0,1] mapped from [-1,1]: y = exp(-(x+1)/2)
    n = 16
    d, x = _cheb_diff_matrix(n)
    d2 = 2.0 * d  # dx/dt = 1/2 -> d/dt = 2 d/dx on mapped domain
    # solve D y = -y with boundary row replaced
    a = d2 + np.eye(n + 1)
    a[-1, :] = 0.0
    a[-1, -1] = 1.0  # y at x=-1 (t=0) = 1
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    y = np.linalg.solve(a, rhs)
    exact = np.exp(-(x + 1.0) / 2.0)
    checks.append(np.max(np.abs(y - exact)) < 1e-10)
    # y(1) ~ e^-1 at x=1
    checks.append(abs(y[0] - np.exp(-1.0)) < 1e-9)
    # BVP: y'' = -pi^2 sin(pi t)/... check y'' = -y, y(0)=0,y(pi/2->mapped)=1
    # on t in [0, pi/2]: y = sin(t). D2 = (2/(pi/2) D)^2
    dmat, xn = _cheb_diff_matrix(n)
    scale = 4.0 / np.pi  # dt = (pi/2)(x+1)/2 -> d/dt = 4/pi d/dx
    dmat *= scale
    dd = dmat @ dmat
    a2 = dd + np.eye(n + 1)
    a2[-1, :] = 0.0
    a2[-1, -1] = 1.0
    a2[0, :] = 0.0
    a2[0, 0] = 1.0
    b = np.zeros(n + 1)
    b[-1] = 0.0  # y(0)=0 at x=-1
    b[0] = 1.0  # y(pi/2)=1 at x=+1
    y2 = np.linalg.solve(a2, b)
    t_nodes = (xn + 1.0) * np.pi / 4.0
    checks.append(np.max(np.abs(y2 - np.sin(t_nodes))) < 1e-8)
    # residual small
    res = (dd + np.eye(n + 1)) @ y2
    checks.append(np.max(np.abs(res[1:-1])) < 1e-6)
    return float(sum(checks) / len(checks))


def bench_collocation_ode(seed: int = 0) -> dict[str, float]:
    return {"synthetic_collocation_ode": _bench_collocation_ode(seed)}
