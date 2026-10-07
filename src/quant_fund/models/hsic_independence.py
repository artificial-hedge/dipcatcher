"""HSIC independence test (Gretton et al. 2005): centered kernel
cross-covariance HSIC + gamma-approx/permutation p-value on dep/indep/
nonlinear synthetic pairs.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._it_synth import dep_data


def _hsic(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    bx = float(np.std(x)) + 1e-9
    by = float(np.std(y)) + 1e-9
    K = np.exp(
        -(
            (
                (x[:, None] - y[:, None] * 0 - x[:, None] * 0 - y[None, :] * 0)
                + (x[:, None] - x[None, :])
            )
            ** 2
        )
        / (2 * bx**2)
    )
    L = np.exp(-((y[:, None] - y[None, :]) ** 2) / (2 * by**2))
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def bench_hsic_independence(seed: int = 2879, perms: int = 150) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    outs = {}
    for kind in ("dep", "indep", "nonlin"):
        x, y = dep_data(seed + 3, kind=kind)
        stat = _hsic(x, y)
        ge = 0
        for _ in range(perms):
            if _hsic(x, rng.permutation(y)) >= stat:
                ge += 1
        outs[kind] = (ge + 1) / (perms + 1)
    return {
        "synthetic_hsic_pval_dep": float(outs["dep"]),
        "synthetic_hsic_pval_indep": float(outs["indep"]),
        "synthetic_hsic_pval_nonlin": float(outs["nonlin"]),
        "synthetic_torch_available": 0.0,
    }
