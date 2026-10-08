"""Early-warning signals of critical transitions — Scheffer et al.

Before a fold-type tipping point, a system loses resilience and its
fluctuations recover more slowly after perturbations ("critical
slowing down"). Observable consequences in a rolling window:

- lag-1 autocorrelation rises toward 1,
- variance (and sometimes skewness) rises.

The significance of the upward trend is assessed with Kendall's tau
on the indicator series against time, and calibrated against
phase-randomized surrogates (IAAFT-lite: FFT magnitude preserved,
phases randomized) which destroy the temporal structure while
keeping the marginal distribution — the fraction of surrogates
whose Kendall tau exceeds the observed value is the p-value.

References
----------
- Scheffer, M., Bascompte, J., Brock, W.A., et al. (2009).
  "Early-warning signals for critical transitions." *Nature* 461.
- Dakos, V., Scheffer, M., van Nes, E.H., et al. (2008). "Slowing
  down as an early warning signal for abrupt climate change."
  *PNAS* 105(38).
- Dakos, V., Carpenter, S.R., Brock, W.A., et al. (2012). "Methods
  for detecting early warnings of critical transitions in time
  series." *PLoS ONE* 7(7) — rolling-window protocol + surrogates.

Honesty
-------
SYNTHETIC AR(1) drift toward collapse only; bench verifies the
indicator recovers the imposed destabilization — not a real
tipping-point claim.

Composition
-----------
Called by ``quant_fund.research.benches_w66.bench_ews_signals``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau

FloatArray = NDArray[np.float64]


def rolling_indicator(x: FloatArray, window: int, kind: str = "ac1") -> FloatArray:
    """Rolling EWS indicator over windows of length ``window``.

    kind: 'ac1' lag-1 autocorrelation, 'var' variance, 'skew'
    skewness. Returns one value per complete window.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 * window or window < 10:
        raise ValueError("series too short for window")
    out = np.empty(x.size - window + 1)
    for i in range(out.size):
        w = x[i : i + window]
        if kind == "ac1":
            w0, w1 = w[:-1], w[1:]
            v0, v1 = w0 - w0.mean(), w1 - w1.mean()
            d = np.sqrt(np.sum(v0 * v0) * np.sum(v1 * v1))
            out[i] = float(np.sum(v0 * v1) / d) if d > 0 else 0.0
        elif kind == "var":
            out[i] = float(np.var(w, ddof=1))
        elif kind == "skew":
            m3 = np.mean((w - w.mean()) ** 3)
            s3 = np.std(w) ** 3
            out[i] = float(m3 / s3) if s3 > 0 else 0.0
        else:
            raise ValueError(f"unknown indicator {kind}")
    return out


def indicator_trend(ind: FloatArray) -> float:
    """Kendall tau of the indicator series against its index."""
    ind = np.asarray(ind, dtype=float)
    if ind.ndim != 1 or ind.size < 5:
        raise ValueError("indicator too short")
    tau, _ = kendalltau(np.arange(ind.size), ind)
    return float(tau) if np.isfinite(tau) else 0.0


def surrogate_pvalue(x: FloatArray, window: int, kind: str, n_surr: int, seed: int) -> float:
    """Fraction of phase-randomized surrogates with Kendall tau at
    least as large as the observed indicator trend."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    tau_obs = indicator_trend(rolling_indicator(x, window, kind))
    count = 0
    n = x.size
    amp = np.abs(np.fft.rfft(x))
    for _ in range(n_surr):
        phases = rng.uniform(0, 2 * np.pi, amp.size)
        spec = amp * np.exp(1j * phases)
        spec[0] = amp[0]  # keep DC real
        if n % 2 == 0:
            spec[-1] = amp[-1]
        xs = np.fft.irfft(spec, n)
        tau_s = indicator_trend(rolling_indicator(xs, window, kind))
        if tau_s >= tau_obs - 1e-12:
            count += 1
    return (count + 1) / (n_surr + 1)


def bench_ews_signals(seed: int = 20261231 + 387) -> dict[str, float]:
    """SYNTHETIC check — AC1/variance trends flag imposed CSD."""
    rng = np.random.default_rng(seed)
    n = 2000
    # AR(1) with persistence rising linearly phi: 0.2 -> 0.99 and a
    # terminal collapse — classic CSD test-bed.
    phi = np.linspace(0.2, 0.99, n)
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi[i] * x[i - 1] + eps[i]
    x[:200] = 0.0  # burn-in
    window = 250
    tau_ac = indicator_trend(rolling_indicator(x, window, "ac1"))
    tau_var = indicator_trend(rolling_indicator(x, window, "var"))
    if tau_ac < 0.3 or tau_var < 0.3:
        raise ValueError("CSD trend not detected")
    p = surrogate_pvalue(x, window, "ac1", n_surr=60, seed=seed + 1)
    if p > 0.15:
        raise ValueError("surrogate p-value not significant")
    # Stationary control: AR(1) phi=0.5 must not flag a strong trend.
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + np.sqrt(0.75) * eps[i]
    tau_ctl = indicator_trend(rolling_indicator(y, window, "ac1"))
    if tau_ctl > 0.3:
        raise ValueError("false positive on stationary control")
    return {
        "synthetic_ews_tau_ac1": tau_ac,
        "synthetic_ews_tau_var": tau_var,
        "synthetic_ews_surrogate_p": p,
        "synthetic_ews_tau_control": tau_ctl,
        "synthetic_score": 1.0,
    }
