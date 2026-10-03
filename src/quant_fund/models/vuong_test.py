"""Vuong likelihood-ratio test for nonnested models.

Two models that share no nesting still compete on fit: the
per-observation log-likelihood ratio ω_t = ℓ1_t − ℓ2_t has
mean zero under equivalence (in Kullback-Leibler distance),
so a t-test on ω̄ picks the closer model — with a variance
test first to check whether the models are even distinguishable.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure model-selection behavior
on generated DGPs — never market evidence.

References:
- Vuong, Q. H. (1989). Likelihood ratio tests for model
  selection and non-nested hypotheses. *Econometrica* 57,
  307-333 — the ω² test of distinguishability and the LR
  model-selection statistic.
- Shi, X. (2015). A nondegenerate Vuong test. *Quantitative
  Economics* 6 — size corrections when models are close.
- Rivers, D., Vuong, Q. (2002). Model selection tests for
  nonlinear dynamic models. *Econometrics Journal* 5 —
  time-series version with HAC variance.
- Clarke, K. A. (2007). A simple distribution-free test for
  nonnested model selection. *Political Analysis* 15 —
  sign-test alternative reported alongside.

Composition: pure numpy + scipy — Gaussian log-likelihood
per obs, HAC variance on ω; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _gauss_ll(y: FloatArray, mu: FloatArray, s2: float) -> FloatArray:
    return np.asarray(
        -0.5 * (np.log(2 * np.pi * s2) + (y - mu) ** 2 / s2),
        dtype=np.float64,
    )


def _fit_mean_var(y: FloatArray, x: FloatArray) -> tuple[FloatArray, float]:
    xd = np.column_stack([np.ones(x.shape[0]), x])
    b, *_ = np.linalg.lstsq(xd, y, rcond=None)
    mu = xd @ b
    s2 = float(np.mean((y - mu) ** 2))
    return mu, max(s2, 1e-12)


def _hac_var(w: FloatArray, lag: int = 4) -> float:
    t = w.size
    wc = w - w.mean()
    s = float(np.sum(wc**2)) / t
    for lag_i in range(1, lag + 1):
        g = float(np.sum(wc[lag_i:] * wc[:-lag_i])) / t
        s += 2 * (1 - lag_i / (lag + 1)) * g
    return max(s, 1e-300)


def vuong_test(
    y: FloatArray,
    x1: FloatArray,
    x2: FloatArray,
    nw_lag: int = 4,
) -> dict[str, float]:
    """Compare two regressions y ~ x1 vs y ~ x2 (nonnested).

    Returns the ω² distinguishability stat, the LR selection
    statistic (positive → model 1 preferred), and the Clarke
    sign-test complement."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    if a1.ndim == 1:
        a1 = a1[:, None]
    if a2.ndim == 1:
        a2 = a2[:, None]
    t = yy.size
    if a1.shape[0] != t or a2.shape[0] != t:
        raise ValueError("aligned y/x1/x2 required")
    if t < 80:
        raise ValueError("T>=80")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(a1)) and np.all(np.isfinite(a2))):
        raise ValueError("finite inputs required")

    mu1, s21 = _fit_mean_var(yy, a1)
    mu2, s22 = _fit_mean_var(yy, a2)
    w = _gauss_ll(yy, mu1, s21) - _gauss_ll(yy, mu2, s22)

    # distinguishability: omega^2 test
    w2 = float(np.mean(w**2))
    v_w2 = _hac_var(w**2, lag=nw_lag)
    z_w2 = w2 / np.sqrt(v_w2 / t)
    p_dist = float(2 * (1 - stats.norm.cdf(abs(z_w2))))

    # model selection: mean(w) / hac_se
    mw = float(np.mean(w))
    v_w = _hac_var(w, lag=nw_lag)
    z_v = mw / np.sqrt(v_w / t)
    p_sel = float(2 * (1 - stats.norm.cdf(abs(z_v))))

    # Clarke sign test (distribution-free)
    n_pos = int(np.sum(w > 0))
    p_sign = float(2 * stats.binomtest(min(n_pos, t - n_pos), t, 0.5).pvalue / 2)

    return {
        "t": float(t),
        "omega2": w2,
        "z_distinguish": float(z_w2),
        "p_distinguish": p_dist,
        "lr_stat": mw,
        "z_vuong": float(z_v),
        "p_vuong": p_sel,
        "n_model1": float(n_pos),
        "p_sign": p_sign,
        "preferred": float(1.0 if mw > 0 else -1.0),
    }


def synth_vuong(
    t: int = 400,
    true_model: int = 1,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y driven by x1 (model 1) or x2 (model 2); the other
    regressor is a correlated distractor."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, t)
    x2 = 0.4 * x1 + np.sqrt(1 - 0.4**2) * rng.normal(0, 1, t)
    if true_model == 1:
        y = 0.8 * x1 + rng.normal(0, 1, t)
    else:
        y = 0.8 * x2 + rng.normal(0, 1, t)
    return {"y": y, "x1": x1, "x2": x2}


def bench_vuong_test(seed: int = 20261231 + 248) -> dict[str, float]:
    """Vuong self-check: the test prefers the correct regressor
    on both orientations and the ω² test marks the models as
    distinguishable. All ``synthetic_*``."""
    d1 = synth_vuong(true_model=1, seed=seed)
    o1 = vuong_test(d1["y"], d1["x1"], d1["x2"])
    d2 = synth_vuong(true_model=2, seed=seed + 1)
    o2 = vuong_test(d2["y"], d2["x1"], d2["x2"])
    o1b = vuong_test(d1["y"], d1["x1"], d1["x2"])

    z1 = float(o1["z_vuong"])
    return {
        "synthetic_z_model1": z1,
        "synthetic_p_model1": float(o1["p_vuong"]),
        "synthetic_z_model2": float(o2["z_vuong"]),
        "synthetic_p_model2": float(o2["p_vuong"]),
        "synthetic_p_distinguish": float(o1["p_distinguish"]),
        "synthetic_detects": float(
            z1 > 1.96 and float(o2["z_vuong"]) < -1.96 and float(o1["p_distinguish"]) < 0.05
        ),
        "synthetic_determinism": float(z1 == float(o1b["z_vuong"])),
    }
