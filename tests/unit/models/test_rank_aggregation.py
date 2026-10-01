"""Tests for rank_aggregation — PL/Borda/Condorcet/MC3."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.rank_aggregation import (
    bench_rank_aggregation,
    borda_count,
    condorcet_copeland,
    mc3_ranking,
    plackett_luce,
)


def _rankings(seed: int = 0, n_r: int = 300):
    rng = np.random.default_rng(seed)
    w_true = np.array([0.4, 0.3, 0.2, 0.1])
    rs = []
    for _ in range(n_r):
        pool = list(range(4))
        wr = w_true.copy()
        order = []
        for _j in range(4):
            pr = wr / wr.sum()
            pick = rng.choice(len(pool), p=pr)
            order.append(pool.pop(pick))
            wr = np.delete(wr, pick)
        rs.append(order)
    return np.array(rs)


def test_pl_top_item():
    r = _rankings()
    out = plackett_luce(r)
    order = np.asarray(out["item_order"], dtype=np.int64)
    assert order[0] == 0
    w = np.asarray(out["w"])
    assert np.argmax(w) == 0


def test_borda_order():
    r = _rankings()
    out = borda_count(r)
    order = np.asarray(out["item_order"], dtype=np.int64)
    assert order[0] == 0


def test_copeland_transitive():
    r = _rankings()
    out = condorcet_copeland(r)
    order = np.asarray(out["item_order"], dtype=np.int64)
    assert order[0] == 0
    win = np.asarray(out["win_matrix"])
    assert win.shape == (4, 4)


def test_mc3_stationary():
    r = _rankings()
    out = mc3_ranking(r)
    pi = np.asarray(out["stationary"])
    assert pi.sum() == pytest.approx(1.0)
    order = np.asarray(out["item_order"], dtype=np.int64)
    assert order[0] == 0


def test_fail_closed_inconsistent():
    bad = np.array([[0, 1, 2], [0, 1, 3]])  # different item sets
    with pytest.raises(ValueError):
        plackett_luce(bad)


def test_fail_closed_too_few():
    with pytest.raises(ValueError):
        borda_count(np.array([[0, 1, 2]]))


def test_bench():
    out = bench_rank_aggregation()
    assert out["score"] == 1.0
