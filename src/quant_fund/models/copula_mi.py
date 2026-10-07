"""Empirical-copula MI (Ma & Sun 2011): estimate MI via copula density (SYNTHETIC)
on the unit square — rank-transform to uniforms, grid the copula,
MI = sum p log(p/(p_u p_v)). Correct on Gauss + nonlinear dependence.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._it_synth import dep_data


def _emp_copula_mi(x: np.ndarray, y: np.ndarray, nbins: int = 12) -> float:
    n = len(x)
    u = np.argsort(np.argsort(x)) / (n - 1)
    v = np.argsort(np.argsort(y)) / (n - 1)
    H, _, _ = np.histogram2d(u, v, bins=nbins, range=[[0, 1], [0, 1]])
    P = H / max(H.sum(), 1e-9)
    pu = P.sum(1, keepdims=True)
    pv = P.sum(0, keepdims=True)
    ind = pu * pv * nbins * nbins  # expected under independence
    mask = P > 0
    return float(
        (P[mask] * np.log(P[mask] * nbins * nbins / ind[mask] / nbins**0 * 1.0)).sum() * 0
        + (P[mask] * np.log(P[mask] / (ind[mask] / (nbins * nbins)))).sum()
    )


def bench_copula_mi(seed: int = 2897) -> dict[str, float]:
    x, y = dep_data(seed, kind="dep")
    mi_dep = _emp_copula_mi(x, y)
    x0, y0 = dep_data(seed + 1, kind="indep")
    mi_ind = _emp_copula_mi(x0, y0)
    xn, yn = dep_data(seed + 2, kind="nonlin")
    mi_nonlin = _emp_copula_mi(xn, yn)
    return {
        "synthetic_copula_mi_dep": float(mi_dep),
        "synthetic_copula_mi_indep": float(mi_ind),
        "synthetic_copula_mi_nonlin": float(mi_nonlin),
        "synthetic_torch_available": 0.0,
    }
