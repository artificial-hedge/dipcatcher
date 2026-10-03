"""Tests for christoffersen_pelletier — duration-based VaR backtest."""

import numpy as np
import pytest

from quant_fund.models.christoffersen_pelletier import (
    bench_christoffersen_pelletier,
    cp_backtest,
    synth_cp,
    violation_durations,
)


def test_iid_hits_pass() -> None:
    hits = synth_cp(seed=1, clustered=False)
    r = cp_backtest(hits)
    assert r["p_value"] > 0.01
    assert 0.6 < r["b_hat"] < 1.6


def test_clustered_hits_rejected() -> None:
    hits = synth_cp(seed=2, clustered=True)
    r = cp_backtest(hits)
    assert r["p_value"] < 0.05


def test_duration_structure() -> None:
    hits = np.zeros(100)
    hits[[10, 30, 60, 70, 80, 90]] = 1.0
    d = violation_durations(hits)
    # leading: idx[0]+1 = 11; gaps: 20,30,10,10,10; trailing: 100-90-1 = 9
    np.testing.assert_allclose(d, [11.0, 20.0, 30.0, 10.0, 10.0, 10.0, 9.0])


def test_exponential_ll_boundary() -> None:
    hits = synth_cp(seed=4)
    r = cp_backtest(hits)
    assert r["lr"] >= 0.0
    assert 0.0 <= r["p_value"] <= 1.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        cp_backtest(np.zeros(200))  # no violations
    with pytest.raises(ValueError):
        cp_backtest(np.ones(30))
    with pytest.raises(ValueError):
        cp_backtest(np.array([np.nan] * 100))


def test_determinism() -> None:
    hits = synth_cp(seed=6)
    a = cp_backtest(hits)
    b = cp_backtest(hits)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_christoffersen_pelletier()
    for k in ("b_iid", "p_iid", "b_clustered", "p_clustered", "lr_gap", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
