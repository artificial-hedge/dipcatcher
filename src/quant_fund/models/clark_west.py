"""Clark-West MSPE-adjusted test for nested forecasts.

When model 2 nests model 1, Diebold-Mariano is degenerate
under the null (the models coincide, so the loss difference
has no well-defined limit distribution). Clark-West's
adjustment f_t = e1_t² − (e2_t² − (ŷ1_t − ŷ2_t)²) restores a
standard test: under H0 its expectation is zero, and a
positive mean rejects in favor of the larger model.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure rejection behavior on
generated forecast panels — never market evidence.

References:
- Clark, T. E., West, K. D. (2007). Approximately normal tests
  for equal predictive accuracy in nested models. *Journal of
  Econometrics* 138, 291-311 — the MSPE-adjusted statistic and
  its standard normal limit.
- Clark, T. E., McCracken, M. W. (2001). Tests of equal
  forecast accuracy and encompassing for nested models.
  *J. Econometrics* 105 — why DM fails under nesting.
- West, K. D. (1996). Asymptotic inference about predictive
  ability. *Econometrica* 64.
- Giacomini, R., White, H. (2006). Tests of conditional
  predictive ability. *Econometrica* 74.

Composition: pure numpy + scipy — loss-difference means with
Newey-West variance; deterministic ``np.random.default_rng``;
no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _nw_se(x: FloatArray, lag: int = 4) -> float:
    t = x.size
    xc = x - x.mean()
    s0 = float(np.sum(xc**2)) / t
    s = s0
    for lag_i in range(1, lag + 1):
        g = float(np.sum(xc[lag_i:] * xc[:-lag_i])) / t
        s += 2 * (1 - lag_i / (lag + 1)) * g
    return float(np.sqrt(max(s / t, 1e-300)))


def clark_west(
    y: FloatArray,
    pred1: FloatArray,
    pred2: FloatArray,
    nw_lag: int = 4,
) -> dict[str, float]:
    """Test whether the larger model (pred2) beats the nested
    benchmark (pred1). f_t = e1² − (e2² − (pred1−pred2)²)."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    p1 = np.asarray(pred1, dtype=np.float64).ravel()
    p2 = np.asarray(pred2, dtype=np.float64).ravel()
    if not (yy.size == p1.size == p2.size):
        raise ValueError("aligned y/pred1/pred2 required")
    if yy.size < 50:
        raise ValueError("T>=50")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        raise ValueError("finite inputs required")
    if np.max(np.abs(p1 - p2)) < 1e-12:
        raise ValueError("pred1 and pred2 coincide")

    e1 = yy - p1
    e2 = yy - p2
    f = e1**2 - (e2**2 - (p1 - p2) ** 2)
    mf = float(np.mean(f))
    se = _nw_se(f, lag=nw_lag)
    t_cw = mf / se
    mspe1 = float(np.mean(e1**2))
    mspe2 = float(np.mean(e2**2))

    return {
        "t": float(yy.size),
        "mspe1": mspe1,
        "mspe2": mspe2,
        "mspe_adj": mf,
        "cw_t": float(t_cw),
        "p_one_sided": float(1 - stats.norm.cdf(t_cw)),
        "mspe_ratio": mspe2 / mspe1,
    }


def synth_forecasts(
    t: int = 400,
    effect: float = 0.4,
    noise_x: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y_t = effect·x_{t-1} + ε_t; pred1 = AR(1)-free benchmark
    (rolling mean), pred2 = OLS on x. noise_x inflates the
    estimation noise inside the larger model."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, t)
    y = effect * x + rng.normal(0.0, 1.0, t)
    # expanding-window benchmark: mean of past y
    cs = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(t)
    pred1 = cs[idx] / np.maximum(idx, 1)
    # larger model: y ~ x fitted on expanding window via corr
    beta_hat = np.cumsum(y * x) / np.maximum(np.cumsum(x * x), 1e-12)
    pred2 = np.clip(beta_hat, -2, 2) * x * noise_x + 0.0 * pred1
    return {"y": y, "pred1": pred1, "pred2": pred2, "x": x}


def bench_clark_west(seed: int = 20261231 + 241) -> dict[str, float]:
    """Clark-West self-check: real predictability rejects in
    favor of the larger model (t positive, p small); a pure-noise
    predictor leaves f≈0 and does not reject. All ``synthetic_*``."""
    d = synth_forecasts(effect=0.5, seed=seed)
    out = clark_west(d["y"], d["pred1"], d["pred2"])
    dn = synth_forecasts(effect=0.0, seed=seed + 1)
    outn = clark_west(dn["y"], dn["pred1"], dn["pred2"])
    out_b = clark_west(d["y"], d["pred1"], d["pred2"])

    tstat = float(out["cw_t"])
    return {
        "synthetic_cw_t": tstat,
        "synthetic_p": float(out["p_one_sided"]),
        "synthetic_mspe_ratio": float(out["mspe_ratio"]),
        "synthetic_adj": float(out["mspe_adj"]),
        "synthetic_cw_t_null": float(outn["cw_t"]),
        "synthetic_p_null": float(outn["p_one_sided"]),
        "synthetic_detects": float(
            tstat > 1.28 and float(out["p_one_sided"]) < 0.1 and abs(float(outn["cw_t"])) < 2.5
        ),
        "synthetic_determinism": float(tstat == float(out_b["cw_t"])),
    }
