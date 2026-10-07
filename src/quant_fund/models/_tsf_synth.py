"""Shared multi-frequency forecasting fixture for the TS-foundation (SYNTHETIC)
canon: sinusoid + trend + AR noise at random frequencies; score = pinball
(mean over quantiles) vs seasonal-naive baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ts_series(seed: int = 7, n: int = 128, period: int = 24) -> NDArray[np.float64]:
    if n < 2 or period < 1:
        raise ValueError(f"need n>=2 and period>=1, got {n},{period}")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    per = period * rng.uniform(0.8, 1.25)
    s = np.sin(2 * np.pi * t / per + rng.uniform(0, 6)) + 0.002 * t
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.standard_normal() * 0.15
    out: NDArray[np.float64] = (s + e).astype(np.float64)
    return out


def pinball(y: NDArray[np.float64], qs: NDArray[np.float64], taus: NDArray[np.float64]) -> float:
    """Mean pinball loss: qs (n_preds, n_taus), y (n_preds,)."""
    y = np.asarray(y, dtype=np.float64)
    qs = np.asarray(qs, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if qs.ndim != 2 or qs.shape != (len(y), len(taus)) or len(y) < 1:
        raise ValueError(f"qs must be (n_preds={len(y)}, n_taus={len(taus)}), got {qs.shape}")
    if ((taus <= 0) | (taus >= 1)).any():
        raise ValueError("taus must lie in (0,1)")
    if not np.isfinite(y).all() or not np.isfinite(qs).all():
        raise ValueError("non-finite input")
    diff = y[:, None] - qs
    loss = np.maximum(taus[None, :] * diff, (taus[None, :] - 1) * diff)
    return float(loss.mean())


def naive_quantiles(
    hist: NDArray[np.float64], taus: NDArray[np.float64], period: int = 24
) -> NDArray[np.float64]:
    """Seasonal-naive quantiles: repeat last-season value + empirical noise."""
    hist = np.asarray(hist, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if period < 2 or len(hist) < period:
        raise ValueError(f"need period>=2 and len(hist)>=period, got {len(hist)},{period}")
    if ((taus <= 0) | (taus >= 1)).any():
        raise ValueError("taus must lie in (0,1)")
    pred = hist[-period:]
    resid = np.diff(hist[-period:])
    qs = []
    for h in range(1, len(pred) + 1):
        qs.append(pred[h - 1] + np.quantile(resid, taus))
    return np.asarray(qs)
