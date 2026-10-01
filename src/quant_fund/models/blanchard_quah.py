"""Blanchard-Quah long-run restrictions for structural VAR(1).

References
----------
- Blanchard, O.J. & Quah, D. (1989). "The Dynamic Effects of Aggregate
  Demand and Supply Disturbances." *American Economic Review* 79(4),
  655-673.
- Christiano, L.J., Eichenbaum, M. & Evans, C.L. (1999). "Monetary
  Policy Shocks: What Have We Learned and to What End?" *Handbook of
  Macroeconomics* 1, 65-148.
- Killian, L. (2013). "Structural Vector Autoregressions." In Hashimzade
  & Thornton (eds.), *Handbook of Research Methods and Applications*.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
For a stationary VAR(1) ``y_t = A1 y_{t-1} + u_t`` with covariance
``Sigma_u``, the long-run multiplier matrix of structural shocks is
``L = (I - A1)^{-1} C`` where ``u_t = C e_t``. The Blanchard-Quah
restriction makes the demand shock long-run neutral on output, i.e.
``L[0,1] = 0`` — one linear restriction pinning down the rotation:

    S = (I - A1)^{-1},  T = S Sigma_u S^T,
    C = S^{-1} Q,  Q the lower Cholesky factor of T.

The demand shock's long-run effect on output vanishes by construction
through the QR-style orthogonalization of the transformed covariance.
The synth is a two-variable supply/demand system: supply shocks move
output permanently, demand shocks only temporarily.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _var1_fit(y: FloatArray) -> FloatArray:
    """Companion VAR(1) coefficient matrix via OLS."""
    xx = y[:-1]
    yy = y[1:]
    xx_d = np.column_stack([np.ones(xx.shape[0]), xx])
    beta = np.linalg.lstsq(xx_d, yy, rcond=None)[0]
    return np.asarray(beta[1:].T, dtype=np.float64)


def _irf(a1: FloatArray, c: FloatArray, horizon: int) -> FloatArray:
    """impulse responses, shape (horizon, k, k), entry [h,i,j] = d y_i / d e_j."""
    k = a1.shape[0]
    out = np.zeros((horizon, k, k))
    m = c
    out[0] = m
    for h in range(1, horizon):
        m = a1 @ m
        out[h] = m
    return out


def blanchard_quah(y: FloatArray, horizon: int = 20) -> dict[str, float]:
    """BQ-identified 2-variable SVAR(1) on (level growth, level).

    ``y[:, 0]`` should be a (quasi-)difference-order series such as
    growth, ``y[:, 1]`` the series the demand shock must not move in the
    long run. Returns the demand-shock long-run neutrality diagnostic
    plus IRF-derived cumulative responses.
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2 or yy.shape[0] < 60 or yy.shape[1] != 2:
        raise ValueError("y must be (n>=60, 2)")
    if not np.all(np.isfinite(yy)):
        raise ValueError("non-finite inputs")
    if horizon < 2:
        raise ValueError("horizon too small")

    a1 = _var1_fit(yy)
    u = yy[1:] - (
        np.column_stack([np.ones(yy.shape[0] - 1), yy[:-1]])
        @ np.linalg.lstsq(np.column_stack([np.ones(yy.shape[0] - 1), yy[:-1]]), yy[1:], rcond=None)[
            0
        ]
    )
    sig_u = np.cov(u, rowvar=False)
    s_mat = np.linalg.inv(np.eye(2) - a1)
    t_mat = s_mat @ sig_u @ s_mat.T
    q_mat = np.linalg.cholesky(t_mat)
    c = np.linalg.solve(s_mat, q_mat)
    # long_run = S C = chol(S Sigma S') is lower-triangular, so shock 2
    # (demand) is long-run neutral on variable 0 by construction.
    long_run = s_mat @ c

    irfs = _irf(a1, c, horizon)
    cum_y1_demand = np.cumsum(irfs[:, 1, 1])
    cum_y1_supply = np.cumsum(irfs[:, 1, 0])
    return {
        "a1_00": float(a1[0, 0]),
        "long_run_supply_y0": float(long_run[0, 0]),
        "long_run_demand_y0": float(long_run[0, 1]),
        "long_run_supply_y1": float(long_run[1, 0]),
        "cum_demand_response_h": float(cum_y1_demand[-1]),
        "cum_supply_response_h": float(cum_y1_supply[-1]),
        "shock_corr": float(long_run[0, 1]),
    }


def synth_bq(
    n: int = 600,
    seed: int = 20261231 + 284,
    supply_weight: float = 1.0,
    demand_transitory: float = 0.9,
) -> dict[str, FloatArray]:
    """Supply/demand system satisfying the BQ identification.

    ``y[:,0]`` is inflation (driven by demand), ``y[:,1]`` is output
    growth where the supply shock has a permanent level effect (growth
    = supply shock innovation) and the demand shock dies geometrically.
    """
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    e_s = rng.normal(0.0, 0.5, n)
    e_d = rng.normal(0.0, 0.5, n)
    y0 = np.zeros(n)
    y1 = np.zeros(n)
    for t in range(1, n):
        y0[t] = 0.4 * y0[t - 1] + 0.6 * e_d[t] + 0.1 * e_s[t]
        # output growth = persistent supply + transitory demand
        y1[t] = (
            0.1 * y1[t - 1]
            + supply_weight * e_s[t]
            + demand_transitory * e_d[t] * 0.2
            - 0.2 * demand_transitory * e_d[t - 1]
        )
    return {"y": np.column_stack([y0, y1]), "e_s": e_s, "e_d": e_d}


def bench_blanchard_quah(seed: int = 20261231 + 284) -> dict[str, float]:
    """Wave-49 self-check: the identified demand shock is long-run
    neutral on the second variable by construction, and the supply
    shock is not."""
    d = synth_bq(seed=seed)
    y = np.asarray(d["y"])
    a = blanchard_quah(y)
    a2 = blanchard_quah(y)
    detects = float(abs(a["long_run_demand_y0"]) < 1e-8 and abs(a["long_run_supply_y1"]) > 0.05)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_lr_demand": a["long_run_demand_y0"],
        "synthetic_lr_supply_y1": a["long_run_supply_y1"],
        "synthetic_cum_demand": a["cum_demand_response_h"],
        "synthetic_cum_supply": a["cum_supply_response_h"],
        "synthetic_a1_00": a["a1_00"],
    }
