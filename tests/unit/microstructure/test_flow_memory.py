"""Tests for microstructure/flow_memory.py — order-flow Hurst."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.flow_memory import (
    FLOW_MEMORY_SCHEMA,
    flow_memory_bench,
    hurst_var_ratio,
    lag1_autocorr,
    sign_sequence,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def test_hurst_iid_near_half() -> None:
    rng = np.random.default_rng(0)
    h = hurst_var_ratio(rng.choice([-1.0, 1.0], 8192))
    assert h == pytest.approx(0.5, abs=0.08)


def test_hurst_persistent_above_half() -> None:
    # planted persistent sequence: blocks of +1 / -1 of length ~50
    rng = np.random.default_rng(1)
    block = 50
    signs = np.concatenate([np.full(block, s) for s in rng.choice([-1.0, 1.0], 160)])
    h = hurst_var_ratio(signs)
    assert h > 0.7


def test_hurst_validation() -> None:
    with pytest.raises(ValueError, match="need >="):
        hurst_var_ratio(np.ones(100))


def test_lag1_autocorr() -> None:
    # Σ a_t a_{t+1} / n = -(n-1)/n for a perfectly alternating sequence
    assert lag1_autocorr(np.tile([1.0, -1.0], 500)) == pytest.approx(-0.999)
    assert lag1_autocorr(np.ones(500)) == pytest.approx(0.998)
    assert np.isnan(lag1_autocorr(np.ones(3)))


def test_sign_sequence_deterministic() -> None:
    cfg = ZILobConfig(seed=3, init_depth=8, band=8)
    a = sign_sequence(config=cfg, horizon=400.0)
    b = sign_sequence(config=cfg, horizon=400.0)
    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) <= {-1.0, 1.0}


def test_bench_smoke_and_schema() -> None:
    out = flow_memory_bench(n_seeds=2, horizon=3000.0)
    assert out["schema"] == FLOW_MEMORY_SCHEMA
    for arm in ("calm", "regime"):
        assert "hurst" in out["arms"][arm]
        assert "rho1" in out["arms"][arm]
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64
