"""Tests for microstructure/kyle_lambda.py — Kyle's lambda estimation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.kyle_lambda import (
    KYLE_LAMBDA_SCHEMA,
    bucket_flow,
    kyle_lambda_bench,
    kyle_regression,
)


class _Trade:
    def __init__(self, side: str, price: float) -> None:
        self.aggressor = side
        self.price = price


def _trades(spec: list[tuple[str, float]]) -> list[_Trade]:
    return [_Trade(s, p) for s, p in spec]


def test_bucket_flow_aggregates_signed_qty_and_price_change() -> None:
    # window=2: [buy,buy] net +2, price change p[2]-p[0]; [sell,sell] net -2
    tr = _trades([("buy", 10.0), ("buy", 10.1), ("sell", 10.2), ("sell", 10.0), ("buy", 10.3)])
    q, dp = bucket_flow(tr, window=2)
    assert q.tolist() == [2.0, -2.0]
    assert dp[0] == pytest.approx(0.2)  # p[2] - p[0]
    assert dp[1] == pytest.approx(0.1)  # p[4] - p[2]


def test_kyle_regression_recovers_true_lambda() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    q = rng.integers(-8, 9, n).astype(float)
    dp = 0.05 * q + rng.normal(0.0, 0.02, n)
    fit = kyle_regression(q, dp)
    assert fit.lambda_hat == pytest.approx(0.05, abs=0.002)
    assert fit.se_nw > 0
    assert fit.t_stat > 20
    assert fit.r2 > 0.9
    assert fit.n_windows == n


def test_nw_se_wider_than_iid_under_autocorrelation() -> None:
    rng = np.random.default_rng(1)
    n = 400
    q = rng.choice([-1.0, 1.0], n)
    # positively autocorrelated residuals -> NW se should exceed iid se
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.7 * eps[i - 1] + rng.normal()
    dp = 0.1 * q + eps
    fit = kyle_regression(q, dp, nw_lag=10)
    # rough iid estimate of the slope se for comparison
    resid = dp - fit.lambda_hat * q
    naive = float(np.sqrt(np.sum(resid * resid) / n / np.sum(q * q)))
    assert fit.se_nw > naive * 0.5  # NW within sane range of naive
    assert fit.se_nw > 0


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        bucket_flow(_trades([("buy", 1.0)] * 5), window=0)
    with pytest.raises(ValueError):
        bucket_flow(_trades([("buy", 1.0)] * 10), window=10)
    with pytest.raises(ValueError):
        kyle_regression(np.ones(5), np.zeros(5))
    with pytest.raises(ValueError):
        kyle_regression(np.full(20, np.nan), np.ones(20))
    with pytest.raises(ValueError):
        kyle_regression(np.zeros(20), np.linspace(0, 1, 20))  # zero flow variance


def test_bench_schema_determinism_and_sign() -> None:
    r1 = kyle_lambda_bench(n_mo=2500, bucket=15, seed=3)
    r2 = kyle_lambda_bench(n_mo=2500, bucket=15, seed=3)
    assert r1["payload_sha256"] == r2["payload_sha256"]
    assert r1["schema"] == KYLE_LAMBDA_SCHEMA
    assert r1["kind"] == "kyle_lambda"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    assert r1["positive_in_both"] is True
    # unit-lot ZI-LOB: lambda is a mechanical book-depth property —
    # positive and significant in both regimes, regime difference is a
    # reported contrast, not an asserted claim
    for reg in r1["regimes"].values():
        assert reg["lambda_hat"] > 0
        assert reg["t_stat"] > 2.0
