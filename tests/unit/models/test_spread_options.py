"""Unit tests for quant_fund.models.spread_options."""

from __future__ import annotations

import pytest

from quant_fund.models.spread_options import (
    bench_spread_options,
    kirk_spread,
    margrabe,
    quanto_call,
)


def test_margrabe_itm() -> None:
    # S2 >> S1 -> price ~ S2 - S1 (forward exchange)
    v = margrabe(50.0, 150.0, 0.2, 0.2, 0.0, 0.5)
    assert v > 90.0


def test_margrabe_zero_vol() -> None:
    v = margrabe(100.0, 100.0, 0.2, 0.2, 1.0, 1.0)
    # identical assets -> zero spread option value
    assert v == pytest.approx(0.0, abs=1e-9)


def test_kirk_at_zero_matches_margrabe() -> None:
    # K=0 -> Kirk reduces to Margrabe exactly? (Kirk is for K>0;
    # at K->0 the adjusted asset -> S1) -- just check proximity.
    v_m = margrabe(80.0, 100.0, 0.25, 0.3, 0.5, 1.0)
    v_k = kirk_spread(80.0, 100.0, 1e-4, 0.25, 0.3, 0.5, 1.0)
    assert v_k == pytest.approx(v_m, rel=0.02)


def test_quanto_drift_sign() -> None:
    q_neg = quanto_call(100.0, 100.0, 0.03, 0.02, 0.2, 0.1, -0.5, 1.0)
    q_pos = quanto_call(100.0, 100.0, 0.03, 0.02, 0.2, 0.1, 0.5, 1.0)
    assert q_neg > q_pos


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        margrabe(-1.0, 100.0, 0.2, 0.2, 0.5, 1.0)
    with pytest.raises(ValueError):
        margrabe(80.0, 100.0, 0.2, 0.2, 1.5, 1.0)
    with pytest.raises(ValueError):
        kirk_spread(80.0, 100.0, -1.0, 0.25, 0.3, 0.5, 1.0)


def test_bench_score() -> None:
    out = bench_spread_options()
    assert out["score"] == 1.0
    assert out["synthetic_spread_kirk_err"] < 0.05
