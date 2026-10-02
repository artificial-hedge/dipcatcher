"""LSD-lite (least-squares dependence, Yamada & Sugiyama 2013 family):
pointwise L2 between the kernel joint density and the product of
kernel marginals, permutation p-value. Torch-free.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._it_synth import dep_data


def _lsd(x: np.ndarray, y: np.ndarray) -> float:
    sx = np.std(x) + 1e-9
    sy = np.std(y) + 1e-9
    Kx = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sx**2))
    Ky = np.exp(-((y[:, None] - y[None, :]) ** 2) / (2 * sy**2))
    dxy = (Kx * Ky).mean(1)
    dpx = Kx.mean(1)
    dpy = Ky.mean(1)
    return float(((dxy - dpx * dpy) ** 2).mean())


def bench_lsd_deptest(seed: int = 2903, perms: int = 120) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    outs = {}
    for kind in ("dep", "indep", "nonlin"):
        x, y = dep_data(seed + 5, kind=kind)
        stat = _lsd(x, y)
        ge = 0
        for _ in range(perms):
            if _lsd(x, rng.permutation(y)) >= stat:
                ge += 1
        outs[kind] = (ge + 1) / (perms + 1)
    return {
        "synthetic_lsd_pval_dep": float(outs["dep"]),
        "synthetic_lsd_pval_indep": float(outs["indep"]),
        "synthetic_lsd_pval_nonlin": float(outs["nonlin"]),
        "torch_available": 0.0,
    }
