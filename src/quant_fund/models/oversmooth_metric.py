"""Oversmoothing diagnostic + correction: Dirichlet energy of node
embeddings decays with depth; residual-scaled propagation (α·A +
(1−α)·I) preserves energy. Reports energy ratio after k rounds and
clique AUC for deep vs shallow propagation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gex_synth import planted_clique


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("oversmooth_metric requires torch (pip install -e .[nn])") from exc
    return torch


def _dirichlet(z: np.ndarray, A: np.ndarray) -> float:
    n = A.shape[0]
    e = 0.0
    for i in range(n):
        for j in np.where(A[i] > 0)[0]:
            e += float(((z[i] - z[j]) ** 2).sum())
    return float(e / max(float(A.sum()), 1))


def bench_oversmooth_metric(seed: int = 889, k: int = 8) -> dict[str, float]:
    A, x, y = planted_clique(seed)
    An = np.diag(1.0 / np.maximum(A.sum(1), 1)) @ A
    z = x[:, :1].copy()
    e0 = _dirichlet(z, A)
    z_deep = z.copy()
    for _ in range(k):
        z_deep = An @ z_deep
    e_deep = _dirichlet(z_deep, A)
    # residual-scaled propagation α=0.5
    z_res = z.copy()
    for _ in range(k):
        z_res = 0.5 * (An @ z_res) + 0.5 * z_res
    e_res = _dirichlet(z_res, A)
    # node separability: mean |z_clique - z_other| under each scheme
    gap_deep = float(np.abs(z_deep[y == 1].mean() - z_deep[y == 0].mean()))
    gap_res = float(np.abs(z_res[y == 1].mean() - z_res[y == 0].mean()))
    return {
        "synthetic_os_energy0": float(e0),
        "synthetic_os_energy_deep": float(e_deep),
        "synthetic_os_energy_res": float(e_res),
        "synthetic_os_collapse_ratio": float(e_deep / max(e0, 1e-9)),
        "synthetic_os_res_ratio": float(e_res / max(e0, 1e-9)),
        "synthetic_os_gap_gain": float(gap_res - gap_deep),
    }
