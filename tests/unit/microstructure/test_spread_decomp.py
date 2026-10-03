"""Tests for microstructure/spread_decomp.py — Huang–Stoll λ estimator."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.spread_decomp import (
    SPREAD_DECOMP_SCHEMA,
    decompose,
    lambda_session,
    spread_decomp_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def _planted_lambda_stream(
    lam: float, n: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Trades where each signed trade moves the mid permanently by lam ticks."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n)
    mid = np.empty(n)
    m = 100.0
    for i, s in enumerate(signs):
        m += lam * 0.01 * s + rng.normal(0.0, 0.002)
        mid[i] = m
    return signs, mid


def test_decompose_recovers_planted_lambda() -> None:
    lam = 1.5  # ticks of permanent revision per signed trade
    signs, mids = _planted_lambda_stream(lam, n=2000)
    fit = decompose(signs, mids, horizon_trades=1)
    assert fit.lam == pytest.approx(lam * 0.01, abs=0.005)


def test_decompose_needs_enough_trades() -> None:
    signs = np.array([1.0, -1.0])
    mids = np.array([100.0, 100.01])
    with pytest.raises(ValueError, match="trades"):
        decompose(signs, mids)


def test_lambda_session_deterministic() -> None:
    a = lambda_session(config=ZILobConfig(seed=42, init_depth=8, band=8), horizon=400.0)
    b = lambda_session(config=ZILobConfig(seed=42, init_depth=8, band=8), horizon=400.0)
    assert a.lam == b.lam and a.n_trades == b.n_trades and a.se == b.se


def test_lambda_session_returns_fit() -> None:
    fit = lambda_session(config=ZILobConfig(seed=7, init_depth=8, band=8), horizon=400.0)
    assert fit.n_trades > 0
    assert np.isfinite(fit.lam)


def test_bench_smoke_and_schema() -> None:
    out = spread_decomp_bench(n_seeds=2, horizon=120.0)
    assert out["schema"] == SPREAD_DECOMP_SCHEMA
    assert set(out["arms"]) == {"calm", "trend"}
    assert out["data_label"] == "SYNTHETIC"
    assert isinstance(out["lambda_trend_gt_calm"], bool)
    assert len(out["payload_sha256"]) == 64
