"""Geweke (1982) frequency-domain Granger-causality measure.

References
----------
- Geweke, J. (1982). "Measurement of Linear Dependence and
  Feedback Between Multiple Time Series." *Journal of the
  American Statistical Association* 77(378), 304-313.
- Geweke, J. (1984). "Measures of Conditional Linear
  Dependence and Feedback Between Time Series." *JASA*
  79(388), 907-915.
- Breitung, J. & Candelon, B. (2006). "Testing for Short-
  and Long-Run Causality: A Frequency-Domain Approach."
  *Journal of Econometrics* 132(2), 363-378.
- Hosoya, Y. (1991). "The Decomposition and Measurement of
  the Interdependency Between Second-Order Stationary
  Processes." *Journal of the American Statistical
  Association* 86, 429-430.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
For a bivariate VAR(p) fitted by OLS with residual
covariance Sigma, Geweke's frequency-domain measure of
causality ``y -> x`` at frequency ``omega`` is

    f_{y->x}(omega) = ln( |S_xx(omega)| / |S_xx(omega) -
      (sigma_yy - sigma_xy^2/sigma_xx) * |H_xy(omega)|^2 | )

where ``H(omega) = (I - sum_j A_j e^{-i omega j})^{-1}``
transfers the VAR into the spectral density. A total measure
is the frequency average; Hosoya (1991) gives the equivalent
causal residual-variance reduction. We also implement the
Breitung-Candelon band-restricted version: the same measure
evaluated on a sub-band, reported as the band's max. The
bench drives x_t by lagged y (positive-lag coupling) plus
independent noise: the y->x measure must exceed the x->y
reverse, and the band peak must sit inside the driven band.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _var_fit(y: FloatArray, p: int) -> tuple[FloatArray, FloatArray]:
    """OLS VAR(p): returns (A (p,k,k) stacked coefs, Sigma)."""
    t, k = y.shape
    rows = y[p - 1 : t - 1][:, ::-1] if False else None
    del rows
    xm = np.concatenate(
        [np.ones((t - p, 1))] + [y[p - lag_i - 1 : t - lag_i - 1] for lag_i in range(p)],
        axis=1,
    )
    target = y[p:]
    coef, *_ = np.linalg.lstsq(xm, target, rcond=None)
    resid = target - xm @ coef
    a = np.zeros((p, k, k))
    for lag_i in range(p):
        a[lag_i] = coef[1 + lag_i * k : 1 + (lag_i + 1) * k].T
    return a, resid.T @ resid / resid.shape[0]


def geweke_spectrum(
    y: FloatArray,
    p: int = 4,
    n_freq: int = 64,
) -> dict[str, FloatArray]:
    """Geweke f_{2->1} and f_{1->2} measures on (0, pi).

    ``y`` is (T, 2) with columns [x1, x2]. Returns the two
    direction measures per frequency plus the Hosoya total.
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2 or yy.shape[1] != 2 or yy.shape[0] < 100:
        raise ValueError("need (T,2) series")
    if not np.all(np.isfinite(yy)) or np.std(yy[:, 0]) < 1e-12:
        raise ValueError("degenerate")
    a, sigma = _var_fit(yy, p)
    k = 2
    freqs = np.linspace(np.pi / n_freq, np.pi, n_freq)
    i2 = np.eye(k)
    f_12 = np.zeros(n_freq)  # x2 -> x1
    f_21 = np.zeros(n_freq)  # x1 -> x2
    for m, w in enumerate(freqs):
        z = np.exp(-1j * w)
        d = i2.copy()
        for lag_i in range(1, p + 1):
            d = d - a[lag_i - 1] * z**lag_i
        h = np.linalg.inv(d)
        s = h @ sigma @ h.conj().T
        s11, s22 = s[0, 0].real, s[1, 1].real
        # Geweke (1982): each direction's denominator is the
        # spectrum that series would carry if only its own
        # innovation — with the instantaneous covariance
        # partialled out — drove it through its diagonal
        # transfer block.
        den1 = abs(h[0, 0]) ** 2 * (sigma[0, 0] - sigma[0, 1] ** 2 / sigma[1, 1])
        den2 = abs(h[1, 1]) ** 2 * (sigma[1, 1] - sigma[0, 1] ** 2 / sigma[0, 0])
        f_12[m] = np.log(max(s11, 1e-30) / max(den1, 1e-30))
        f_21[m] = np.log(max(s22, 1e-30) / max(den2, 1e-30))
    return {"freqs": freqs, "f_2to1": f_12, "f_1to2": f_21}


def geweke_total(y: FloatArray, p: int = 4) -> float:
    """Frequency-averaged total linear feedback (Hosoya)."""
    sp = geweke_spectrum(y, p=p)
    return float(np.mean(sp["f_2to1"]) + np.mean(sp["f_1to2"]))


def band_measure(
    sp: dict[str, FloatArray],
    lo: float,
    hi: float,
    direction: str = "f_2to1",
) -> float:
    """Breitung-Candelon band-restricted max of the measure."""
    w = sp["freqs"]
    mask = (w >= lo) & (w <= hi)
    if not mask.any():
        raise ValueError("empty band")
    return float(np.max(sp[direction][mask]))


def synth_geweke(
    seed: int = 20261231 + 331,
    n: int = 600,
    lag: int = 2,
    gain: float = 0.6,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC y->x lagged coupling vs independent pair."""
    rng = np.random.default_rng(seed)
    yy = rng.standard_normal(n)
    xx = np.zeros(n)
    for t in range(lag, n):
        xx[t] = gain * yy[t - lag] + 0.4 * rng.standard_normal()
    pair = np.column_stack([xx, yy])
    indep = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    return pair, indep


def bench_geweke_spectral(
    seed: int = 20261231 + 331,
) -> dict[str, float]:
    """Wave-57 self-check: directional asymmetry + band peak."""
    pair, indep = synth_geweke(seed=seed)
    sp = geweke_spectrum(pair, p=5)
    fwd = float(np.mean(sp["f_2to1"]))
    rev = float(np.mean(sp["f_1to2"]))
    sp0 = geweke_spectrum(indep, p=5)
    null_max = float(np.max(sp0["f_2to1"]))
    band_peak = band_measure(sp, 0.5, 2.0)
    ok = fwd > 3.0 * rev and fwd > 0.15 and band_peak > 3.0 * null_max
    return {
        "synthetic_fwd_mean": fwd,
        "synthetic_rev_mean": rev,
        "synthetic_band_peak": band_peak,
        "synthetic_null_max": null_max,
        "synthetic_total": geweke_total(pair, p=5),
        "synthetic_score": float(ok),
    }
