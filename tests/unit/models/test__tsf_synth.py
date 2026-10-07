"""Adversarial probes for _tsf_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series


def test_ts_series_deterministic():
    np.testing.assert_array_equal(ts_series(1), ts_series(1))
    s = ts_series(1)
    assert len(s) == 128
    assert np.isfinite(s).all()


@pytest.mark.parametrize("n,period", [(1, 24), (0, 24), (64, 0)])
def test_ts_series_hostile(n, period):
    with pytest.raises(ValueError):
        ts_series(n=n, period=period)


def test_pinball_oracle_zero():
    y = np.arange(10.0)
    taus = np.array([0.1, 0.5, 0.9])
    qs = np.repeat(y[:, None], 3, axis=1)
    assert pinball(y, qs, taus) == pytest.approx(0.0)


def test_pinball_median_is_mae_half():
    y = np.linspace(0, 1, 20)
    taus = np.array([0.5])
    qs = np.zeros((20, 1))
    assert pinball(y, qs, taus) == pytest.approx(0.5 * np.abs(y).mean())


def test_pinball_hostile():
    y = np.ones(5)
    taus = np.array([0.5])
    qs = np.ones((5, 1))
    with pytest.raises(ValueError):
        pinball(y[:0], qs[:0], taus)  # empty
    with pytest.raises(ValueError):
        pinball(y, np.ones((4, 1)), taus)  # row mismatch
    with pytest.raises(ValueError):
        pinball(y, np.ones((5, 2)), taus)  # tau-count mismatch
    with pytest.raises(ValueError):
        pinball(y, qs, np.array([0.0]))  # tau boundary
    with pytest.raises(ValueError):
        pinball(y, qs, np.array([1.0]))
    with pytest.raises(ValueError):
        pinball(np.array([1.0, np.nan, 0.0]), np.ones((3, 1)), taus)


def test_naive_quantiles_shape_and_monotone():
    hist = ts_series(2)
    taus = np.array([0.1, 0.5, 0.9])
    qs = naive_quantiles(hist, taus, period=24)
    assert qs.shape == (24, 3)
    assert (np.diff(qs, axis=1) >= -1e-12).all()  # quantiles non-decreasing


def test_naive_quantiles_hostile():
    hist = np.arange(30.0)
    taus = np.array([0.5])
    with pytest.raises(ValueError):
        naive_quantiles(hist[:10], taus, period=24)  # too-short history
    with pytest.raises(ValueError):
        naive_quantiles(hist, taus, period=1)  # degenerate resid
    with pytest.raises(ValueError):
        naive_quantiles(hist, np.array([0.0]), period=4)
