"""ICP canon: iterative closest point — nearest-neighbor
correspondences + Kabsch rigid step, iterated to a fixed-point
on the registration residual. Bench: converges from a perturbed
initial pose on a synthetic shape pair, final RMS below the
perturbation scale. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from quant_fund.models.kabsch import apply_transform, kabsch

FloatArray = NDArray[np.float64]


def icp(
    P: FloatArray,
    Q: FloatArray,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> tuple[FloatArray, FloatArray, float, int]:
    """Register P onto Q (point-to-point ICP).

    Returns (R, t, rms, iters).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    tree = cKDTree(Q)
    R_tot = np.eye(P.shape[1])
    t_tot = np.zeros(P.shape[1])
    P_cur = P.copy()
    prev = np.inf
    for it in range(max_iter):
        idx = tree.query(P_cur)[1]
        Qc = Q[idx]
        R, t, s, rms = kabsch(P_cur, Qc)
        P_cur = apply_transform(P_cur, R, t)
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t
        if abs(prev - rms) < tol:
            prev = rms
            return R_tot, t_tot, rms, it + 1
        prev = rms
    return (
        np.asarray(R_tot, dtype=np.float64),
        np.asarray(t_tot, dtype=np.float64),
        float(prev),
        max_iter,
    )


def bench_icp(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    # synthetic shape: a "L" + blob — deterministic structure
    th = rng.uniform(0, 2 * np.pi, 200)
    shape = np.stack([np.cos(th), np.sin(th), 0.3 * np.sin(3 * th)], axis=1)
    P = shape + rng.normal(0, 0.01, shape.shape)
    ang = 0.25
    ca, sa = np.cos(ang), np.sin(ang)
    R_true = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    t_true = np.array([0.4, -0.2, 0.1])
    Q = shape @ R_true + t_true + rng.normal(0, 0.01, shape.shape)
    R, t, rms, iters = icp(P, Q, max_iter=80)
    out["synthetic_icp_rms"] = rms
    out["synthetic_icp_iters"] = float(iters)
    out["synthetic_icp_converged"] = float(iters < 80)
    # apply recovered transform: mean residual distance
    aligned = apply_transform(P, R, t)
    tree = cKDTree(Q)
    md = float(np.mean(tree.query(aligned)[0]))
    out["synthetic_icp_nn_dist"] = md
    # harder: bigger rotation (still inside convergence basin)
    ang2 = 0.7
    ca2, sa2 = np.cos(ang2), np.sin(ang2)
    R2 = np.array([[ca2, -sa2, 0], [sa2, ca2, 0], [0, 0, 1.0]])
    Q2 = shape @ R2 + np.array([0.3, 0.1, -0.2])
    _, _, rms2, iters2 = icp(P, Q2, max_iter=120)
    out["synthetic_icp_hard_rms"] = rms2
    out["synthetic_icp_hard_iters"] = float(iters2)
    return out
