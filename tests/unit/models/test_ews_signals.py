"""Unit tests for quant_fund.models.ews_signals."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ews_signals import (
    bench_ews_signals,
    indicator_trend,
    rolling_indicator,
    surrogate_pvalue,
)


def _ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi * phi) * e[i]
    return x


def test_rolling_indicator_shape_and_range() -> None:
    x = _ar1(800, 0.5)
    ind = rolling_indicator(x, 200, "ac1")
    assert ind.shape == (800 - 200 + 1,)
    assert np.all((ind >= -1.0) & (ind <= 1.0))


def test_ac1_tracks_persistence() -> None:
    x_low = _ar1(800, 0.2, 1)
    x_high = _ar1(800, 0.9, 1)
    a = rolling_indicator(x_low, 300, "ac1").mean()
    b = rolling_indicator(x_high, 300, "ac1").mean()
    assert b > a


def test_indicator_kinds_and_bad_kind() -> None:
    x = _ar1(600, 0.4)
    for kind in ("ac1", "var", "skew"):
        ind = rolling_indicator(x, 200, kind)
        assert ind.size == 401 and np.all(np.isfinite(ind))
    with pytest.raises(ValueError):
        rolling_indicator(x, 200, "bogus")
    with pytest.raises(ValueError):
        rolling_indicator(x[:50], 200, "ac1")


def test_indicator_trend_sign() -> None:
    rising = np.linspace(0.1, 0.9, 200)
    flat = np.full(200, 0.5)
    assert indicator_trend(rising) > 0.9
    assert abs(indicator_trend(flat)) < 0.05
    with pytest.raises(ValueError):
        indicator_trend(np.array([0.1, 0.2]))


def test_surrogate_pvalue_in_unit_interval() -> None:
    x = _ar1(900, 0.85, 2)
    p = surrogate_pvalue(x, 250, "ac1", n_surr=20, seed=3)
    assert 0.0 < p <= 1.0


def test_bench_ews_signals_score() -> None:
    out = bench_ews_signals()
    assert out["synthetic_score"] == pytest.approx(1.0)
    assert out["synthetic_ews_tau_ac1"] > 0.3
    assert out["synthetic_ews_surrogate_p"] <= 0.15
