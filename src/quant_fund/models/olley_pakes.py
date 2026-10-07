"""Olley-Pakes (1996) production function estimation (SYNTHETIC).

Simultaneity: the firm sees productivity ω before choosing
variable inputs, so OLS on lnk is biased. OP uses investment
as a monotone proxy for ω: stage 1 inverts ω from a
polynomial in (i, k) to get β_l unbundled; stage 2 runs NLS
on y − β_l·l = β_k·k + g(ω_{t-1}(β_k)) + η with g a
polynomial in lagged ω.

Honesty: synthetic panels generate output with known β_k,
β_l and an AR(1) productivity process; the bench checks the
recovered coefficients against OLS — proper diagnostics,
never market evidence.

References:
- Olley, G. S., Pakes, A. (1996). The dynamics of
  productivity in the telecommunications equipment industry.
  *Econometrica* 64 — the proxy-inversion estimator.
- Levinsohn, J., Petrin, A. (2003). Estimating production
  functions using inputs to control for unobservables.
  *Review of Economic Studies* 70 — the intermediate-input
  variant the first stage nests.
- Ackerberg, D. A., Caves, K., Frazer, G. (2015).
  Identification properties of recent production function
  estimators. *Econometrica* 83 — functional-form cautions.
- Wooldridge, J. M. (2009). On estimating firm-level
  production functions using proxy variables to control for
  unobservables. *Economics Letters* 104 — one-step GMM
  alternative.

Composition: numpy + scipy.optimize only — two-stage
polynomial inversion + scalar NLS; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]


def _poly3(x: FloatArray, coef: FloatArray) -> FloatArray:
    return np.asarray(
        coef[0] + coef[1] * x + coef[2] * x**2 + coef[3] * x**3,
        dtype=np.float64,
    )


def _fit_poly3(x: FloatArray, y: FloatArray) -> FloatArray:
    return np.asarray(np.polyfit(x, y, 3)[::-1], dtype=np.float64)  # ascending powers


def op_estimate(
    y: FloatArray,
    lnk: FloatArray,
    lnl: FloatArray,
    inv: FloatArray,
    firm: FloatArray,
    period: FloatArray,
) -> dict[str, float]:
    """Two-stage OP estimation. Panel required for the lagged-
    productivity stage; all arrays (N,) matching."""
    yy = np.asarray(y, dtype=np.float64)
    k = np.asarray(lnk, dtype=np.float64)
    lab = np.asarray(lnl, dtype=np.float64)
    i = np.asarray(inv, dtype=np.float64)
    f = np.asarray(firm, dtype=np.float64)
    t = np.asarray(period, dtype=np.float64)
    n = yy.shape[0]
    if n < 100 or not all(a.shape == (n,) for a in (k, lab, i, f, t)):
        raise ValueError("matched (N,) arrays, N>=100 required")
    if not all(np.all(np.isfinite(a)) for a in (yy, k, lab, i)):
        raise ValueError("finite inputs required")
    # stage 1: y = β_l·l + φ(i,k) + e — partial out l via
    # semiparametric OLS with poly basis in (i,k)
    basis = np.column_stack([np.ones(n), i, k, i * k, i**2, k**2, i**3, i * k**2, i**2 * k, k**3])
    xb = np.column_stack([lab, basis])
    coef = np.linalg.lstsq(xb, yy, rcond=None)[0]
    beta_l = float(coef[0])
    phi = basis @ coef[1:]
    # stage 2: ω_t = φ_t − β_k·k_t; E[ω_t|ω_{t-1}] = poly3(ω_{t-1})
    order = np.lexsort((t, f))
    fs, ts = f[order], t[order]
    k_o, lab_o, phi_o, y_o = k[order], lab[order], phi[order], yy[order]
    lag_ok = (fs[1:] == fs[:-1]) & (ts[1:] > ts[:-1])

    def obj(bk: float) -> float:
        omega = phi_o - bk * k_o
        om1, om0 = omega[1:], omega[:-1]
        g = _fit_poly3(om0[lag_ok], om1[lag_ok])
        eta = (y_o[1:] - beta_l * lab_o[1:] - bk * k_o[1:])[lag_ok] - _poly3(om0[lag_ok], g)
        # OP moments: η_t ⊥ {k_t, l_{t-1}, k_{t-1}} — inputs
        # committed before the innovation; residual-SSE
        # profiling alone degenerates to a corner when k
        # co-moves with ω
        return float(
            np.sum((eta * k_o[1:][lag_ok]) ** 2)
            + np.sum((eta * lab_o[:-1][lag_ok]) ** 2)
            + np.sum((eta * k_o[:-1][lag_ok]) ** 2)
        )

    res = minimize_scalar(obj, bounds=(-1.0, 2.0), method="bounded")
    beta_k = float(res.x)
    return {
        "beta_k": beta_k,
        "beta_l": beta_l,
        "n_lagged": float(np.sum(lag_ok)),
        "obj": float(res.fun),
    }


def synth_op_panel(
    n_firms: int = 200,
    t: int = 10,
    beta_k: float = 0.6,
    beta_l: float = 0.4,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Panel: ln y = βk ln k + βl ln l + ω + η, ω AR(1),
    investment i = h(ω) + shock — the OP endogeneity."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, float, float, float]] = []
    for fi in range(n_firms):
        omega = float(rng.normal(0, 0.8))
        k_it = float(rng.uniform(1.0, 3.0))
        for tp in range(t):
            if tp:
                omega = 0.7 * omega + float(rng.normal(0, 0.4))
            l_it = 0.3 * omega + rng.normal(0, 0.4)  # labor responds to ω → OLS bias
            i_it = 2.0 * omega + rng.normal(0, 0.05)  # steep, tight proxy for ω
            y_it = beta_k * k_it + beta_l * l_it + omega + rng.normal(0, 0.15)
            rows.append((y_it, k_it, l_it, i_it, float(fi), float(tp)))
            # capital accumulates: k_t+1 = (1−δ)·k + i, δ=.1 —
            # enough within-firm variation to separate k from ω
            k_it = 0.9 * k_it + i_it
    arr = np.array(rows)
    return {
        "y": arr[:, 0],
        "lnk": arr[:, 1],
        "lnl": arr[:, 2],
        "inv": arr[:, 3],
        "firm": arr[:, 4],
        "period": arr[:, 5],
    }


def bench_olley_pakes(seed: int = 20261231 + 264) -> dict[str, float]:
    """OP self-check: recovers β_k≈0.6 where OLS (l correlated
    with ω) is upward-biased. All ``synthetic_*``."""
    d = synth_op_panel(seed=seed)
    out = op_estimate(
        np.asarray(d["y"]),
        np.asarray(d["lnk"]),
        np.asarray(d["lnl"]),
        np.asarray(d["inv"]),
        np.asarray(d["firm"]),
        np.asarray(d["period"]),
    )
    x = np.column_stack([np.ones(d["y"].size), np.asarray(d["lnk"]), np.asarray(d["lnl"])])
    b = np.linalg.lstsq(x, np.asarray(d["y"]), rcond=None)[0]
    ols_l = float(b[2])
    out2 = op_estimate(
        np.asarray(d["y"]),
        np.asarray(d["lnk"]),
        np.asarray(d["lnl"]),
        np.asarray(d["inv"]),
        np.asarray(d["firm"]),
        np.asarray(d["period"]),
    )
    return {
        "synthetic_beta_k": out["beta_k"],
        "synthetic_beta_l": out["beta_l"],
        "synthetic_ols_beta_l": ols_l,
        "synthetic_true_beta_k": 0.6,
        "synthetic_detects": float(
            abs(out["beta_k"] - 0.6) < 0.2
            and abs(out["beta_l"] - 0.4) < 0.15
            and abs(ols_l - 0.4) > abs(out["beta_l"] - 0.4)
        ),
        "synthetic_determinism": float(out2["beta_k"] == out["beta_k"]),
    }
