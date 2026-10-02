import numpy as np
import pytest

from quant_fund.models.innovations_ets import (
    bench_innovations_ets,
    croston_sba,
    ets_aan,
    ets_forecast,
    theta_method,
)


def _trend_series(seed=0, n=120):
    rng = np.random.default_rng(seed)
    lv, bs = 0.0, 0.3
    y = np.empty(n)
    for t in range(n):
        lv += 0.9 * bs + 0.05 * rng.standard_normal()
        bs = 0.9 * bs + 0.02 * rng.standard_normal()
        y[t] = lv + 0.2 * rng.standard_normal()
    return y


def test_ets_aan_shapes():
    y = _trend_series()
    out = ets_aan(y)
    assert np.asarray(out["levels"]).shape == (120,)
    assert 0 < float(out["alpha"]) < 1
    assert float(out["sse"]) > 0


def test_ets_forecast_horizon():
    y = _trend_series(1)
    fit = ets_aan(y)
    fc = np.asarray(ets_forecast(fit, 8))
    assert fc.shape == (8,)
    assert np.isfinite(fc).all()


def test_ets_beats_naive():
    y = _trend_series(2, 160)
    fit = ets_aan(y[:120])
    fc = np.asarray(ets_forecast(fit, 40))
    err = np.abs(fc - y[120:]).mean()
    naive = np.abs(y[119] - y[120:]).mean()
    assert err < naive


def test_theta_method_shape():
    y = _trend_series(3)
    fc = np.asarray(theta_method(y, 6))
    assert fc.shape == (6,)
    assert np.isfinite(fc).all()


def test_croston_sba_shrinks():
    rng = np.random.default_rng(4)
    z = rng.binomial(1, 0.3, 60) * (1 + rng.poisson(3.0, 60))
    out = croston_sba(z, alpha=0.2)
    assert out["sba_rate"] < out["croston_rate"]
    assert out["interval_forecast"] > 0
    assert out["size_forecast"] > 0


def test_ets_input_validation():
    with pytest.raises(ValueError):
        ets_aan(np.arange(5.0))
    with pytest.raises(ValueError):
        croston_sba(np.zeros(30))


def test_bench_innovations_ets():
    out = bench_innovations_ets()
    assert out["score"] == 1.0
    assert out["synthetic_ets_mase_ratio"] < 0.85
