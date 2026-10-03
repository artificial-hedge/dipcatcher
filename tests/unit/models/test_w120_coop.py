"""Tests for wave-120 cooperative-game + mechanism-design canon:
nucleolus, banzhaf, owen, myerson_auction, groves, envy_free."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.banzhaf import banzhaf_index, bench_banzhaf
from quant_fund.models.envy_free import (
    bench_envy_free,
    ef1_violations,
    envy_value,
    round_robin,
)
from quant_fund.models.groves import bench_groves, groves_mechanism
from quant_fund.models.myerson_auction import (
    bench_myerson,
    myerson_alloc_pay,
    virtual_value_uniform,
)
from quant_fund.models.nucleolus import (
    CoopGame,
    bench_nucleolus,
    glove_game,
    nucleolus,
    voting_game,
)
from quant_fund.models.owen import bench_owen, owen_value


def test_glove_game_values():
    g = glove_game(2, 1)
    assert g.v(7) == 1.0 and g.v(3) == 0.0


def test_nucleolus_pivot_share():
    g = glove_game(2, 1)
    x = nucleolus(g)
    assert x[2] == pytest.approx(1.0, abs=1e-4)


def test_nucleolus_efficiency():
    g = voting_game(np.ones(3), 2.0)
    x = nucleolus(g)
    assert x.sum() == pytest.approx(1.0)


def test_banzhaf_normalization():
    g = voting_game(np.array([4.0, 3, 2, 1]), 6.0)
    idx = banzhaf_index(g)
    assert idx.sum() == pytest.approx(1.0)
    assert idx[0] > idx[3]


def test_owen_efficient():
    n = 4
    vals = np.zeros(2**n)
    for s in range(2**n):
        vals[s] = bin(s).count("1") * 0.5
    g = CoopGame(n, vals)
    phi = owen_value(g, [[0, 1], [2, 3]])
    assert phi.sum() == pytest.approx(g.v(15))


def test_myerson_reserve():
    assert virtual_value_uniform(np.array([0.5]))[0] == pytest.approx(0.0)


def test_myerson_no_sale():
    w, _ = myerson_alloc_pay(np.array([0.3, 0.4]), [virtual_value_uniform] * 2)
    assert w == -1


def test_myerson_threshold_payment():
    w, p = myerson_alloc_pay(np.array([0.8, 0.6]), [virtual_value_uniform] * 2)
    assert w == 0 and p == pytest.approx(0.6, abs=0.02)


def test_groves_welfare_max():
    v = [np.array([5.0, 1.0]), np.array([2.0, 4.0])]
    x, _ = groves_mechanism(v)
    assert x == 0


def test_groves_pivot_payment():
    v = [np.array([5.0, 1.0]), np.array([2.0, 4.0])]
    _, pay = groves_mechanism(v)
    # agent 0's pivot payment: others' max without 0 (4) − others' welfare at x* (2)
    assert pay[0] == pytest.approx(4.0 - 2.0)


def test_round_robin_partition():
    rng = np.random.default_rng(0)
    u = rng.uniform(1, 10, (3, 9))
    b = round_robin(u)
    allocs = sorted(it for bl in b for it in bl)
    assert allocs == list(range(9))


def test_round_robin_ef1():
    rng = np.random.default_rng(1)
    u = rng.uniform(1, 10, (4, 12))
    b = round_robin(u)
    assert ef1_violations(u, b) == 0
    assert envy_value(u, b) >= 0.0


def test_benches():
    for fn in (
        bench_nucleolus,
        bench_banzhaf,
        bench_owen,
        bench_myerson,
        bench_groves,
        bench_envy_free,
    ):
        out = fn()
        assert out and all(k.startswith("synthetic_") for k in out)
