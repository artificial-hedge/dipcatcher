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
    for k in (
        "synthetic_b_iid",
        "synthetic_p_iid",
        "synthetic_b_clustered",
        "synthetic_p_clustered",
        "synthetic_lr_gap",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0


def test_terminal_violation_does_not_crash():
    """A hit on the final index yields a zero right-censored duration;
    it contributes log S(0)=0, not a hard error (old code raised)."""
    hits = np.zeros(200)
    hits[[10, 40, 90, 140, 180, 199]] = 1.0
    d = violation_durations(hits)
    assert d[-1] == 0.0
    r = cp_backtest(hits)
    assert np.isfinite(r["b_hat"])


def test_clustered_synth_both_regimes_persist():
    """The two-state chain must persist in BOTH states (p_stay each);
    the old asymmetric version made state-0 transient (2% persistence),
    collapsing the stream to ~25% iid hits."""
    hits = synth_cp(seed=7, clustered=True, alpha=0.05)
    rate = float(hits.mean())
    # symmetric persistence gives roughly half low-rate (1.5%) and
    # half high-rate (25%) -> ~13%; the broken chain sat near 25%.
    assert 0.05 < rate < 0.20
