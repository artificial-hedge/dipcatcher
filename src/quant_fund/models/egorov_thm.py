"""Egorov: a.e. convergence -> uniform convergence off a small set (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def uniform_tail_error(fs: list[np.ndarray], f: np.ndarray, n_tail: int) -> float:
    """sup over grid of max_{k >= n_tail} |f_k - f|."""
    tail = np.stack(fs[n_tail:])
    return float(np.max(np.abs(tail - f)))


def _bench_egorov_thm(seed: int = 0) -> float:
    checks = []
    # f_n(x) = x^n -> 0 a.e. on [0,1]; uniform off [0, 1-eps]
    x = np.linspace(0, 1, 4001)
    eps = 0.01
    xs = x[x <= 1 - eps]
    fs = [xs**k for k in range(1, 101)]
    f = np.zeros_like(xs)
    # tail of sup error shrinks to ~eps^N
    err = uniform_tail_error(fs, f, 80)
    checks.append(err < (1 - eps) ** 80 + 1e-9)
    # on whole [0,1] sup error stays 1 (convergence is not uniform)
    xw = np.linspace(0, 1, 4001)
    fsw = [xw**k for k in range(80, 101)]
    checks.append(uniform_tail_error(fsw, np.zeros_like(xw), 0) > 0.99)
    # removed set has measure eps <= chosen delta
    checks.append(eps <= 0.05)
    # monotone sequence f_n = x/n -> 0 uniformly already (small delta ok)
    fs2 = [x / k for k in range(1, 60)]
    checks.append(uniform_tail_error(fs2, np.zeros_like(x), 30) < 0.04)
    return float(sum(checks) / len(checks))


def bench_egorov_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_egorov_thm": _bench_egorov_thm(seed)}
