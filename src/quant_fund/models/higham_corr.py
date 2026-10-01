"""Nearest correlation matrix repair — Higham (2002) / Qi & Sun (2006).

Given a symmetric unit-diagonal matrix that is *not* positive
semidefinite (mixed-frequency estimates, asynchronously sampled
correlations, stress perturbed surfaces), recover the nearest
correlation matrix in the Frobenius norm.

Two algorithms are implemented:

- ``higham_nearcorr`` — Higham's (2002) Dykstra alternating-correction
  algorithm between the convex sets U (symmetric, unit diagonal) and
  S (symmetric, PSD). Robust, linear convergence, O(n^3) per step.
- ``newton_nearcorr`` — Qi & Sun (2006) semismooth Newton method on the
  dual problem. Quadratic convergence; uses the generalized Jacobian
  W_ij = max(lambda_i,0)/( |lambda_i| + |lambda_j| ) with the eigen-
  decomposition of A + Diag(y). Falls back to Dykstra if the Newton
  solve stalls.

References
----------
- Higham (2002) IMA J. Numer. Anal. 22, "Computing the nearest
  correlation matrix — a problem from finance".
- Qi & Sun (2006) SIAM J. Matrix Anal. 28, "A quadratically convergent
  Newton method for computing the nearest correlation matrix".

Honesty
-------
Deterministic linear algebra — no randomness, no tuning. The bench
checks: output is a valid correlation matrix (PSD + unit diag), is
nearer than the naive eigen-clip heuristic, and that an already-valid
input is returned unchanged. All numbers SYNTHETIC.

Composition
-----------
Called by ``quant_fund.research.benches_w63.bench_higham_corr``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_TOL = 1e-10


def _unit_diag(x: FloatArray) -> FloatArray:
    y = x.copy()
    np.fill_diagonal(y, 1.0)
    return y


def _proj_psd(x: FloatArray) -> FloatArray:
    w, v = np.linalg.eigh(x)
    return (v * np.maximum(w, 0.0)) @ v.T


def higham_nearcorr(a: FloatArray, tol: float = 1e-8, max_iter: int = 200) -> FloatArray:
    """Dykstra alternating projections onto {unit diag} and {PSD}.

    Fail-closed on non-square, non-symmetric, or non-finite input.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be square")
    if not np.all(np.isfinite(a)):
        raise ValueError("a must be finite")
    if not np.allclose(a, a.T, atol=1e-8):
        raise ValueError("a must be symmetric")
    n = a.shape[0]
    ds = np.zeros((n, n))
    y = a.copy()
    for _ in range(max_iter):
        r = y - ds
        x = _proj_psd(r)
        ds = x - r
        y_new = _unit_diag(x)
        if np.max(np.abs(y_new - y)) < tol:
            return (y_new + y_new.T) / 2.0
        y = y_new
    return _unit_diag((y + y.T) / 2.0)


def newton_nearcorr(a: FloatArray, tol: float = 1e-8, max_iter: int = 50) -> FloatArray:
    """Qi-Sun semismooth Newton for the nearest correlation matrix.

    Minimizes over y the dual objective; at the optimum y*, the primal
    solution is (A + Diag(y*))_+. On any numerical stall, falls back to
    :func:`higham_nearcorr`.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be square")
    if not np.all(np.isfinite(a)):
        raise ValueError("a must be finite")
    if not np.allclose(a, a.T, atol=1e-8):
        raise ValueError("a must be symmetric")
    n = a.shape[0]
    y = np.zeros(n)
    try:
        for _ in range(max_iter):
            w, p = np.linalg.eigh(a + np.diag(y))
            wp = np.maximum(w, 0.0)
            x_plus = (p * wp) @ p.T
            g = np.diag(x_plus) - 1.0
            if np.max(np.abs(g)) < tol:
                x = _unit_diag(x_plus)
                return (x + x.T) / 2.0
            dif = w[:, None] - w[None, :]
            w_gen = np.where(
                np.abs(dif) > _TOL,
                (wp[:, None] - wp[None, :]) / np.where(np.abs(dif) > _TOL, dif, 1.0),
                (w[:, None] > 0.0).astype(float),
            )
            j_mat = np.einsum("ik,il,jk,jl,kl->ij", p, p, p, p, w_gen)
            dy = np.linalg.solve(j_mat + 1e-12 * np.eye(n), g)
            if not np.all(np.isfinite(dy)):
                return higham_nearcorr(a, tol=tol)
            step = 1.0
            for _ls in range(10):
                y_new = y - step * dy
                w_new = np.linalg.eigvalsh(a + np.diag(y_new))
                if np.all(np.isfinite(w_new)):
                    break
                step /= 2.0
            y = y - step * dy
        w, p = np.linalg.eigh(a + np.diag(y))
        x = _unit_diag((p * np.maximum(w, 0.0)) @ p.T)
        return (x + x.T) / 2.0
    except np.linalg.LinAlgError:
        return higham_nearcorr(a, tol=tol)


def _is_corr(x: FloatArray, tol: float = 1e-7) -> bool:
    return (
        np.allclose(np.diag(x), 1.0, atol=1e-8)
        and np.allclose(x, x.T, atol=1e-10)
        and float(np.linalg.eigvalsh(x).min()) > -tol
    )


def _broken_corr(rng: np.random.Generator, n: int = 6) -> FloatArray:
    """Symmetric unit-diagonal matrix that is NOT PSD."""
    q = rng.standard_normal((n, n))
    c = np.asarray(np.corrcoef(q), dtype=np.float64)
    e = np.eye(n)
    # Blend toward -I off-diagonal enough to break PSD deterministically.
    m = 0.5 * c + 0.5 * (-e + rng.uniform(-0.3, 0.3, (n, n)))
    m = (m + m.T) / 2.0
    m = _unit_diag(m)
    if np.linalg.eigvalsh(m).min() >= 0.0:
        # Force a negative eigenvalue: shrink an off-diagonal block.
        m = _unit_diag(c)
        m[: n // 2, n // 2 :] *= 3.0
        m[n // 2 :, : n // 2] *= 3.0
        m = np.asarray(np.clip(m, -1.5, 1.5), dtype=np.float64)
        m = _unit_diag((m + m.T) / 2.0)
    return m


def bench_higham(seed: int = 20261231 + 366) -> dict[str, float]:
    """SYNTHETIC check — nearest-correlation repair properties."""
    rng = np.random.default_rng(seed)
    a = _broken_corr(rng)
    eig_min_in = float(np.linalg.eigvalsh(a).min())
    h = higham_nearcorr(a)
    q = newton_nearcorr(a)
    if not (_is_corr(h) and _is_corr(q)):
        raise ValueError("repair output not a valid correlation matrix")
    # Feasible baseline: plain alternating projection (clip -> unit diag
    # repeated). Converges to a valid correlation matrix but not the
    # nearest one, so the true projections must be no farther away.
    naive = a.copy()
    for _ in range(80):
        w, v = np.linalg.eigh(naive)
        naive = _unit_diag((v * np.maximum(w, 0.0)) @ v.T)
    d_h = float(np.linalg.norm(h - a))
    d_q = float(np.linalg.norm(q - a))
    d_n = float(np.linalg.norm(naive - a))
    # Identity input must pass through.
    eye = np.eye(4)
    d_id = float(np.linalg.norm(higham_nearcorr(eye) - eye))
    if eig_min_in >= -1e-6:
        raise ValueError("synthetic input unexpectedly already PSD")
    if d_h > d_n + 1e-9 or d_q > d_n + 1e-9:
        raise ValueError("projection farther than naive baseline")
    if d_id > 1e-8:
        raise ValueError("identity not preserved")
    if abs(d_h - d_q) > 0.02 * (1.0 + d_h):
        raise ValueError("Dykstra and Newton disagree on the optimum")
    return {
        "synthetic_nc_eigmin_in": eig_min_in,
        "synthetic_nc_dist_higham": d_h,
        "synthetic_nc_dist_newton": d_q,
        "synthetic_nc_dist_altproj": d_n,
        "synthetic_nc_id_err": d_id,
        "score": 1.0,
    }
