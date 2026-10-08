"""Terasvirta smooth-transition autoregression (STAR).

References
----------
- Terasvirta, T. (1994). "Specification, Estimation, and
  Evaluation of Smooth Transition Autoregressive Models."
  *Journal of the American Statistical Association* 89(425),
  208-218.
- Luukkonen, R., Saikkonen, P. & Terasvirta, T. (1988).
  "Testing Linearity Against Smooth Transition Autoregressive
  Models." *Biometrika* 75(3), 491-499.
- van Dijk, D., Terasvirta, T. & Franses, P.H. (2002).
  "Smooth Transition Autoregressive Models — A Survey of
  Recent Developments." *Econometric Reviews* 21(1), 1-47.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
STAR interpolates two AR regimes through a transition
function of the delayed variable ``z_t = y_{t-d}``:
- LSTAR: ``F(z) = (1 + exp(-gamma (z - c)))^{-1}`` — regime
  switch at location c (asymmetric adjustment).
- ESTAR: ``F(z) = 1 - exp(-gamma (z - c)^2)`` — symmetric
  outer/inner regimes.
The conditional mean is ``(1-F) phi1'x + F phi2'x`` with the
lag vector x — estimated by concentrated NLS over (gamma, c)
with linear-regime OLS profiles; a grid over c inside the
data's interquartile band plus a moderate gamma grid keeps
the optimization honest (unconstrained NLS on gamma blows
up to a step function and overfits two observations).
Linearity is tested by the Luukkonen-Saikkonen-Terasvirta
LM3: regress residuals on the Taylor-expanded transition
regressors ``x, x*z, x*z^2`` and F-test the added terms.
``synth_star`` plants an LSTAR with switch at c=0.5 and a
linear AR control; the bench gates on LM rejection under
LSTAR, acceptance under AR, and c recovery.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _st

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 200) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _design(y: FloatArray, p: int, d: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    n = y.size
    t0 = max(p, d)
    dep = y[t0:]
    x = np.column_stack([y[t0 - i - 1 : n - i - 1] for i in range(p)])
    z = y[t0 - d : n - d]
    return dep, x, z


def _ols(a: FloatArray, b: FloatArray) -> tuple[FloatArray, float]:
    x1 = np.column_stack([np.ones(a.shape[0]), a])
    coef = np.linalg.lstsq(x1, b, rcond=None)[0]
    resid = b - x1 @ coef
    return coef, float(resid @ resid)


def _f_transition(z: FloatArray, gamma: float, c: float, kind: str) -> FloatArray:
    u = np.clip(gamma * (z - c), -30.0, 30.0)
    if kind == "lstar":
        return 1.0 / (1.0 + np.exp(-u))
    return 1.0 - np.exp(-np.clip(gamma * (z - c) ** 2, 0.0, 30.0))


def star_fit(
    y: FloatArray,
    p: int = 1,
    d: int = 1,
    kind: str = "lstar",
    n_grid: int = 25,
) -> dict[str, float]:
    """Grid-NLS STAR fit returning regime coefficients."""
    v = _as_series(y)
    if p < 1 or p > 5 or d < 1 or d > 5 or kind not in {"lstar", "estar"}:
        raise ValueError("bad star spec")
    dep, x, z = _design(v, p, d)
    q1, q3 = np.quantile(z, 0.15), np.quantile(z, 0.85)
    cs = np.linspace(q1, q3, n_grid)
    gammas = np.geomspace(0.5, 30.0, n_grid)
    best: tuple[float, FloatArray, FloatArray, float, float] | None = None
    for c in cs:
        for g_cand in gammas:
            f = _f_transition(z, g_cand, c, kind)
            a = np.column_stack(
                [
                    (1 - f),
                    x * (1 - f)[:, None],
                    f,
                    x * f[:, None],
                ]
            )
            coef, ssr = _ols(a, dep)
            if best is None or ssr < best[3]:
                best = (
                    float(g_cand),
                    coef,
                    np.asarray([c]),
                    ssr,
                    float(f.mean()),
                )
    if best is None:
        raise ValueError("no feasible star fit")
    g, coef, c_arr, ssr, fbar = best
    p2 = x.shape[1]
    out: dict[str, float] = {
        "gamma": float(g),
        "c": float(c_arr[0]),
        "ssr": ssr,
        "f_mean": fbar,
        "phi1_sum": float(np.sum(coef[2 : 2 + p2])),
        "phi2_sum": float(np.sum(coef[2 + p2 :])),
    }
    return out


def linearity_lm3(
    y: FloatArray,
    p: int = 1,
    d: int = 1,
) -> dict[str, float]:
    """Luukkonen-Saikkonen-Terasvirta LM3 linearity test."""
    v = _as_series(y)
    dep, x, z = _design(v, p, d)
    n, k = dep.size, x.shape[1]
    _, ssr0 = _ols(x, dep)
    zc = z - np.mean(z)
    aux = np.column_stack(
        [
            x,
            zc,
            x * zc[:, None],
            zc**2,
            x * (zc**2)[:, None],
            zc**3,
            x * (zc**3)[:, None],
        ]
    )
    _, ssr1 = _ols(aux, dep)
    df1 = 3 * (k + 1)
    df2 = max(n - (3 * k + 1), 1)
    f_stat = ((ssr0 - ssr1) / df1) / (ssr1 / df2) if ssr1 > 1e-12 else 0.0
    pval = float(1.0 - _st.f.cdf(max(f_stat, 0.0), df1, df2))
    out: dict[str, float] = {
        "lm3": f_stat,
        "pvalue": pval,
        "reject": float(pval < 0.05),
    }
    return out


def synth_star(
    seed: int = 20261231 + 350,
    n: int = 900,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC LSTAR(c=0.5, gamma=8) + linear AR control."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        f = 1.0 / (1.0 + np.exp(-8.0 * (y[t - 1] - 0.0)))
        y[t] = (
            (1 - f) * (0.4 + 0.3 * y[t - 1])
            + f * (-0.4 + 0.3 * y[t - 1])
            + 0.15 * rng.standard_normal()
        )
    w = np.zeros(n)
    for t in range(1, n):
        w[t] = 0.2 + 0.5 * w[t - 1] + 0.3 * rng.standard_normal()
    return y.astype(np.float64), w.astype(np.float64)


def bench_star(seed: int = 20261231 + 350) -> dict[str, float]:
    y, w = synth_star(seed=seed)
    lm_star = linearity_lm3(y, p=2, d=1)
    lm_lin = linearity_lm3(w, p=2, d=1)
    fit = star_fit(y, p=1, d=1, kind="lstar")
    ok = (
        lm_star["reject"] == 1.0
        and lm_lin["reject"] == 0.0
        and abs(fit["c"]) < 0.35
        and fit["phi1_sum"] > fit["phi2_sum"]
    )
    out: dict[str, float] = {
        "synthetic_star_lm3": lm_star["lm3"],
        "synthetic_star_pvalue": lm_star["pvalue"],
        "synthetic_star_c_hat": fit["c"],
        "synthetic_star_lm3_lin_pvalue": lm_lin["pvalue"],
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
