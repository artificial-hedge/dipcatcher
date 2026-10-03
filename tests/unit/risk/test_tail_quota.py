"""Tests for quant_fund.risk.tail_quota."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.risk.tail_quota import tail_quota_bench, tail_quota_eprocess

ALPHA = 0.05
C = -0.02


def _honest(n: int, seed: int) -> np.ndarray:
    sd = abs(C) / abs(norm.ppf(ALPHA))
    return np.random.default_rng(seed).normal(0.0, sd, n)


def test_honest_stream_stays_calm():
    res = tail_quota_eprocess(_honest(600, 0), c=C, alpha=ALPHA)
    assert not res["alarmed"]
    assert res["e_max"] < 100.0


def test_fat_tail_alarms():
    rng = np.random.default_rng(1)
    fat = rng.standard_t(3.0, 600) * 0.015  # breach at -0.02 ≈ 0.12 >> alpha
    res = tail_quota_eprocess(fat, c=C, alpha=ALPHA)
    assert res["alarmed"]
    assert res["breach_rate"] > ALPHA


def test_e_process_supermartingale_null():
    # under H0, e-path should not systematically exceed 1/alpha
    alarms = 0
    for s in range(30):
        res = tail_quota_eprocess(_honest(400, 1000 + s), c=C, alpha=ALPHA)
        alarms += int(res["e_max"] > 20.0)  # Ville at 5%: P(e_max>20)<=.05
    assert alarms <= 30 * 0.1  # generous slack over the 5% bound


def test_validation_edges():
    with pytest.raises(ValueError):
        tail_quota_eprocess(np.array([0.01]), c=C, alpha=ALPHA)
    with pytest.raises(ValueError):
        tail_quota_eprocess(np.array([0.0, np.inf]), c=C, alpha=ALPHA)
    with pytest.raises(ValueError):
        tail_quota_eprocess(np.zeros(10), c=C, alpha=1.0)


def test_bench_sealed():
    r = tail_quota_bench(seed=0)
    assert r["schema"] == "tail_quota.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r == tail_quota_bench(seed=0)
