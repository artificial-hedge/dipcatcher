"""Tests for stl_loess (wave-58)."""

import numpy as np
import pytest

from quant_fund.models.stl_loess import bench_stl, stl_decompose, stl_strength, synth_stl


def test_recovers_trend() -> None:
    y, trend, _ = synth_stl(seed=1)
    d = stl_decompose(y, period=24)
    assert np.corrcoef(d["trend"], trend)[0, 1] > 0.97


def test_resid_variance_collapses() -> None:
    y, _, _ = synth_stl(seed=2)
    d = stl_decompose(y, period=24)
    assert np.var(d["resid"]) < 0.3 * np.var(y)


def test_seasonal_strength_high() -> None:
    y, _, _ = synth_stl(seed=3)
    assert stl_strength(y, period=24)["seasonal_strength"] > 0.8


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        stl_decompose(np.ones(20), period=4)
    with pytest.raises(ValueError):
        stl_decompose(np.full(200, np.nan), period=12)
    with pytest.raises(ValueError):
        stl_decompose(np.ones(200), period=1)


def test_bench_schema_and_score() -> None:
    r = bench_stl()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
