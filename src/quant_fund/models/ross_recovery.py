"""Ross recovery theorem: physical transitions from state prices.

References
----------
- Ross, S. (2015). "The Recovery Theorem." *Journal of Finance*
  70(2), 615-648.
- Borovicka, J., Hansen, L.P. & Scheinkman, J.A. (2016).
  "Misspecified Recovery." *Journal of Finance* 71(6), 2493-2544.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Under Ross's bounded-utility and no-arbitrage conditions, if ``Q``
is the matrix of positive state prices (entry ``Q[i,j]`` the price
in state ``i`` of a claim paying one unit in state ``j``), then
``Q`` admits a Perron-Frobenius eigenpair ``(gamma, u)`` with
``gamma`` in ``(0, 1)`` — the inverse gross risk-free factor — and
``u`` strictly positive. The physical transition matrix is
``P = D(u)^{-1} Q D(u) / gamma``, i.e. ``P[i,j] = Q[i,j] u[j] /
(gamma u[i])``, which is row-stochastic by the eigenvector
property. BHS(2016) caution that this recovers the physical law
only under the theorem's structural restrictions (the recovered
``P`` is always well defined but its interpretation is not model-
free in general); that caveat is surfaced as ``assumption_note``
in the result. The synth plants a known ``P`` and a positive
eigenvector ``u``, builds ``Q = gamma D(u) P D(u)^{-1}``, and the
recovered matrix must equal the planted ``P`` entrywise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_TOL = 1e-10
_NOTE = "recovered P is the Ross-transform of Q; model-free only under bounded-utility + independent-transition structure (BHS 2016)"


def _check_q(q: FloatArray) -> FloatArray:
    qq = np.asarray(q, dtype=np.float64)
    if qq.ndim != 2 or qq.shape[0] != qq.shape[1]:
        raise ValueError("Q must be a square matrix")
    if qq.shape[0] < 2:
        raise ValueError("need at least 2 states")
    if not np.all(np.isfinite(qq)):
        raise ValueError("Q contains non-finite entries")
    if np.any(qq <= 0.0):
        raise ValueError("Q must be strictly positive (no-arbitrage)")
    return qq


def ross_recover(
    q: FloatArray,
) -> dict[str, FloatArray | float | str]:
    """Recover the physical transition matrix from state prices.

    Raises ValueError on non-square, non-finite, or non-positive
    input, or when the Perron root degenerates.
    """
    qq = _check_q(q)
    # Perron-Frobenius: dominant eigenvalue/eigenvector of Q.
    # Power iteration keeps this deterministic and cheap.
    n = qq.shape[0]
    u = np.ones(n) / n
    gamma = 0.0
    for _ in range(2000):
        u_new = qq @ u
        gamma_new = float(u_new[0] / u[0])
        u_new = u_new / np.linalg.norm(u_new)
        if abs(gamma_new - gamma) < 1e-13 and np.allclose(u_new, u, atol=1e-13):
            u = u_new
            gamma = gamma_new
            break
        u, gamma = u_new, gamma_new
    # sign convention: PF eigenvector is positive
    if np.any(u <= _TOL):
        raise ValueError("degenerate Perron eigenvector")
    if not (0.0 < gamma <= 1.0 + 1e-9):
        raise ValueError("Perron root outside (0, 1]")
    u = u / np.sum(u)
    p = (qq * u[np.newaxis, :]) / (gamma * u[:, np.newaxis])
    row_err = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
    return {
        "p": p,
        "gamma": gamma,
        "u": u,
        "row_sum_err": row_err,
        "assumption_note": _NOTE,
    }


def synth_ross(
    seed: int = 20261231 + 297,
    n: int = 6,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC plant of (P, u, gamma) -> state-price matrix Q."""
    rng = np.random.default_rng(seed)
    # ergodic transition matrix
    a = rng.gamma(2.0, 1.0, size=(n, n))
    p_true = a / a.sum(axis=1, keepdims=True)
    u_true = rng.gamma(3.0, 1.0, size=n) + 0.2
    u_true = u_true / u_true.sum()
    gamma_true = 0.97
    # Q = gamma D(u) P D(u)^{-1}  =>  Qu = gamma D(u) P 1 = gamma u
    q = gamma_true * u_true[:, np.newaxis] * p_true / u_true[np.newaxis, :]
    return {
        "q": q,
        "p_true": p_true,
        "u_true": u_true,
        "gamma_true": gamma_true,
    }


def bench_ross_recovery(seed: int = 20261231 + 297) -> dict[str, float]:
    """Wave-51 self-check: exact recovery of planted transitions."""
    d = synth_ross(seed=seed)
    r = ross_recover(np.asarray(d["q"]))
    p_hat = np.asarray(r["p"])
    p_true = np.asarray(d["p_true"])
    err = float(np.max(np.abs(p_hat - p_true)))
    g_err = abs(float(r["gamma"]) - float(d["gamma_true"]))
    ok = err < 1e-6 and g_err < 1e-8 and float(r["row_sum_err"]) < 1e-8
    return {
        "p_err": err,
        "gamma_err": g_err,
        "row_sum_err": float(r["row_sum_err"]),
        "gamma": float(r["gamma"]),
        "score": float(ok),
    }
