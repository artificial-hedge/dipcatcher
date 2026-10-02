"""Streaming-sketch tests (t-digest, HLL, CMS, GK)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.sketches import (
    CountMinSketch,
    GKSketch,
    HyperLogLog,
    TDigest,
    bench_sketches,
)


def test_tdigest_tail_accuracy():
    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 1, 15000)
    td = TDigest(delta=80)
    for v in x:
        td.add(float(v))
    for q in (0.5, 0.9, 0.99):
        est = td.quantile(q)
        true = float(np.quantile(x, q))
        assert abs(est - true) / true < 0.05


def test_hll_cardinality():
    rng = np.random.default_rng(1)
    items = rng.integers(0, 50000, 30000).astype(np.float64)
    h = HyperLogLog(b=10)
    for v in items:
        h.add(float(v))
    true = float(np.unique(items).size)
    assert abs(h.estimate() - true) / true < 0.08


def test_cms_no_underestimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 300, 20000).astype(np.float64)
    cms = CountMinSketch(512, 5, seed=7)
    for v in y:
        cms.add(float(v))
    counts = np.bincount(y.astype(np.int64), minlength=300)
    for i in range(0, 300, 25):
        assert cms.estimate(float(i)) >= counts[i] - 1e-9


def test_gk_quantile_rank_bound():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(12000)
    gk = GKSketch(eps=0.01)
    for v in x:
        gk.add(float(v))
    est = gk.query(0.9)
    rank = float((x <= est).mean())
    assert abs(rank - 0.9) < 0.03


def test_fail_closed():
    td = TDigest(80)
    with pytest.raises(ValueError):
        td.add(float("nan"))
    with pytest.raises(ValueError):
        TDigest(5)
    with pytest.raises(ValueError):
        HyperLogLog(b=2)
    with pytest.raises(ValueError):
        GKSketch(eps=0.5)
    gk = GKSketch()
    with pytest.raises(ValueError):
        gk.query(0.5)


def test_bench_passes():
    out = bench_sketches()
    assert out["synthetic_tdigest_qrelerr"] < 0.05
    assert out["synthetic_hll_card_relerr"] < 0.08
