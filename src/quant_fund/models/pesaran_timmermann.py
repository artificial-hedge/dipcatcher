"""Pesaran-Timmermann (1992) market-timing direction test.

References
----------
- Pesaran, M.H. & Timmermann, A. (1992). "A Simple
  Nonparametric Test of Predictive Performance." *JBES* 10(4),
  461-465.
- Pesaran, M.H. & Timmermann, A. (1994). "A Generalization of
  the Nonparametric Henriksson-Merton Test of Market Timing."
  *Economics Letters* 44(1-2), 1-7.
- Henriksson, R.D. & Merton, R.C. (1981). "On Market Timing and
  Investment Performance. II. Statistical Procedures for
  Evaluating Forecasting Skills." *Journal of Business* 54(4),
  513-533.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Given a sign prediction ``f_t in {-1, +1}`` for the sign of the
realization ``y_t``, the PT statistic compares the empirical
correct-direction frequency ``p_hat`` against the frequency
expected under independence, ``p_* = f_up * y_up + (1-f_up) *
(1-y_up)`` where ``f_up``, ``y_up`` are the empirical up-shares.
``S = (p_hat - p_*) / se(p_hat - p_*)`` with the PT variance
formula ``Var = p_*(1-p_*) * (1/(n*y_up*(1-y_up)) +
1/(n*f_up*(1-f_up)))`` — valid asymptotically N(0,1) under the
null that forecasts and realizations are independent. The
Henriksson-Merton hypergeometric variant reports the exact
small-sample version. The synth plants forecasts with planted
directional skill rho in [-1, 1]: at rho=0 the test must accept
(p>0.05) and at rho=0.35 it must reject (p<0.01).
``pt_test(y, f)`` returns ``s_stat``, ``p_value``, ``p_hat``,
``p_star``; ``hm_test`` returns the hypergeometric tail.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def pt_test(y: FloatArray, f: FloatArray) -> dict[str, float]:
    """Pesaran-Timmermann sign-accuracy test, asymptotic p."""
    yy = np.asarray(y, dtype=np.float64)
    ff = np.asarray(f, dtype=np.float64)
    if (
        yy.ndim != 1
        or ff.ndim != 1
        or yy.size != ff.size
        or yy.size < 30
        or not np.all(np.isfinite(yy))
        or not np.all(np.isfinite(ff))
    ):
        raise ValueError("bad inputs")
    n = yy.size
    y_up = float(np.mean(yy > 0))
    f_up = float(np.mean(ff > 0))
    if min(y_up, 1 - y_up, f_up, 1 - f_up) < 1e-6:
        raise ValueError("degenerate sign split")
    p_hat = float(np.mean((yy > 0) == (ff > 0)))
    p_star = y_up * f_up + (1 - y_up) * (1 - f_up)
    var = p_star * (1 - p_star) * (1.0 / (n * y_up * (1 - y_up)) + 1.0 / (n * f_up * (1 - f_up)))
    s = (p_hat - p_star) / np.sqrt(var)
    p = float(1.0 - _stats.norm.cdf(s))
    return {
        "s_stat": float(s),
        "p_value": p,
        "p_hat": p_hat,
        "p_star": float(p_star),
        "y_up": y_up,
        "f_up": f_up,
    }


def hm_test(y: FloatArray, f: FloatArray) -> dict[str, float]:
    """Henriksson-Merton hypergeometric exact tail.

    Under the null, the number of correctly predicted ups is
    hypergeometric: draw ``n1 = #ups in y`` from n with
    ``m1 = #ups predicted``; tail = P(correct >= observed).
    """
    yy = np.asarray(y, dtype=np.float64)
    ff = np.asarray(f, dtype=np.float64)
    if (
        yy.ndim != 1
        or ff.ndim != 1
        or yy.size != ff.size
        or yy.size < 30
        or not np.all(np.isfinite(yy))
        or not np.all(np.isfinite(ff))
    ):
        raise ValueError("bad inputs")
    n = yy.size
    n1 = int(np.sum(yy > 0))
    m1 = int(np.sum(ff > 0))
    x = int(np.sum((yy > 0) & (ff > 0)))
    p = float(_stats.hypergeom.sf(x - 1, n, n1, m1))
    return {
        "hm_p": p,
        "correct_ups": float(x),
        "n_up": float(n1),
        "m_up": float(m1),
    }


def synth_pt(
    seed: int = 20261231 + 311,
    t: int = 800,
    rho: float = 0.35,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC signal with directional skill ``rho``.

    y_t = z_t; f_t = sign(rho*z_{t-1-ish} + noise) built so
    corr(sign(f_t), sign(y_t)) rises with rho.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(t)
    y = np.roll(z, 1)  # make forecast see z_{t-1} → predictable share
    y[0] = rng.standard_normal()
    noise = rng.standard_normal(t)
    signal = rho * np.roll(z, 1) + np.sqrt(1 - rho * rho) * noise
    f = np.sign(signal)
    return y, f


def bench_pesaran_timmermann(
    seed: int = 20261231 + 311,
) -> dict[str, float]:
    """Wave-54 self-check: accepts rho=0, rejects rho=0.45."""
    y0, f0 = synth_pt(seed=seed, rho=0.0)
    r_null = pt_test(y0, f0)
    y1, f1 = synth_pt(seed=seed + 7, rho=0.45)
    r_alt = pt_test(y1, f1)
    r_hm = hm_test(y1, f1)
    ok = r_null["p_value"] > 0.05 and r_alt["p_value"] < 0.01 and r_hm["hm_p"] < 0.01
    return {
        "p_null": r_null["p_value"],
        "p_alt": r_alt["p_value"],
        "hm_p": r_hm["hm_p"],
        "p_hat_alt": r_alt["p_hat"],
        "score": float(ok),
    }
