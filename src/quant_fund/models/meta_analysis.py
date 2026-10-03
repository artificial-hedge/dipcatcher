"""Random-effects meta-analysis of effect estimates across studies.

References
----------
- DerSimonian, R. & Laird, N. (1986). "Meta-analysis in Clinical
  Trials." *Controlled Clinical Trials* 7(3), 177-188.
- Higgins, J.P.T. & Thompson, S.G. (2002). "Quantifying Heterogeneity
  in a Meta-analysis." *Statistics in Medicine* 21(11), 1539-1558.
- Egger, M., Davey Smith, G., Schneider, M. & Minder, C. (1997). "Bias
  in Meta-analysis Detected by a Simple, Graphical Test." *BMJ* 315,
  629-634.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
DerSimonian-Laird heterogeneity: ``Q = sum w_i (t_i - t_fe)^2`` with
``w_i = 1/se_i^2``, ``tau2 = max(0, (Q - df) / C)``; then random-effects
pooling ``theta = sum w_i^* t_i / sum w_i^*`` with
``w_i^* = 1/(se_i^2 + tau2)``; ``I^2 = max(0, (Q-df)/Q)``; Egger's test
regresses the standardized effect on precision (intercept test). The
synth generates k studies estimating a common true effect with study-
specific bias for the funnel-asymmetry diagnostic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ols_slope_se(y: FloatArray, x: FloatArray) -> tuple[float, float, float, float]:
    """(intercept, slope, se_slope, se_intercept) of OLS y ~ 1 + x."""
    xx = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.linalg.lstsq(xx, y, rcond=None)[0]
    res = y - xx @ beta
    dof = max(y.shape[0] - 2, 1)
    s2 = float(res @ res) / dof
    xtx_inv = np.linalg.inv(xx.T @ xx)
    return (
        float(beta[0]),
        float(beta[1]),
        float(np.sqrt(s2 * xtx_inv[1, 1])),
        float(np.sqrt(s2 * xtx_inv[0, 0])),
    )


def meta_analysis(
    effects: FloatArray,
    se: FloatArray,
) -> dict[str, float]:
    """DerSimonian-Laird random-effects pooling + Egger asymmetry.

    Returns the pooled RE estimate, its SE, heterogeneity stats
    (Q, tau^2, I^2), and Egger regression diagnostics.
    """
    t = np.asarray(effects, dtype=np.float64)
    s = np.asarray(se, dtype=np.float64)
    if t.ndim != 1 or s.ndim != 1 or t.shape != s.shape:
        raise ValueError("effects and se must be same-length vectors")
    k = t.shape[0]
    if k < 4:
        raise ValueError("need k >= 4 studies")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(s)):
        raise ValueError("non-finite inputs")
    if np.any(s <= 0):
        raise ValueError("se must be positive")

    w = 1.0 / s**2
    t_fe = float(np.sum(w * t) / np.sum(w))
    q_stat = float(np.sum(w * (t - t_fe) ** 2))
    df = k - 1
    c_denom = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q_stat - df) / c_denom) if c_denom > 0 else 0.0
    ws = 1.0 / (s**2 + tau2)
    t_re = float(np.sum(ws * t) / np.sum(ws))
    se_re = float(np.sqrt(1.0 / np.sum(ws)))
    i2 = max(0.0, (q_stat - df) / q_stat) if q_stat > 0 else 0.0

    # Egger: regress standardized effect on precision
    snd = t / s
    prec = 1.0 / s
    eg_int, eg_slope, _se_s, se_int = _ols_slope_se(snd, prec)
    egger_t = eg_int / se_int if se_int > 0 else 0.0

    return {
        "theta_fe": t_fe,
        "theta_re": t_re,
        "se_re": se_re,
        "q_stat": q_stat,
        "tau2": float(tau2),
        "i2": float(i2),
        "egger_intercept": float(eg_int),
        "egger_t": float(egger_t),
        "egger_slope": float(eg_slope),
        "k": float(k),
    }


def synth_meta(
    k: int = 15,
    seed: int = 20261231 + 286,
    theta: float = 0.4,
    hetero_sd: float = 0.10,
    bias_small: float = 0.0,
) -> dict[str, FloatArray]:
    """k studies around a common effect; ``bias_small`` adds small-study
    publication bias proportional to se."""
    rng = np.random.default_rng(seed)
    if k < 4:
        raise ValueError("k too small")
    se = rng.uniform(0.08, 0.35, k)
    t = theta + rng.normal(0.0, hetero_sd, k) + rng.normal(0.0, se) + bias_small * se
    return {"effects": t, "se": se}


def bench_meta_analysis(seed: int = 20261231 + 286) -> dict[str, float]:
    """Wave-49 self-check: RE pooling recovers the common effect; the
    biased small-study synth trips Egger's test."""
    d = synth_meta(seed=seed)
    a = meta_analysis(np.asarray(d["effects"]), np.asarray(d["se"]))
    d2 = synth_meta(seed=seed, bias_small=3.0)
    b = meta_analysis(np.asarray(d2["effects"]), np.asarray(d2["se"]))
    a2 = meta_analysis(np.asarray(d["effects"]), np.asarray(d["se"]))
    detects = float(
        abs(a["theta_re"] - 0.4) < 0.12
        and a["i2"] > 0.0
        and abs(b["egger_t"]) > abs(a["egger_t"]) + 0.5
    )
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_theta_re": a["theta_re"],
        "synthetic_tau2": a["tau2"],
        "synthetic_i2": a["i2"],
        "synthetic_egger_t_biased": b["egger_t"],
        "synthetic_egger_t_null": a["egger_t"],
        "synthetic_q_stat": a["q_stat"],
    }
