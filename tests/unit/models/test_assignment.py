import numpy as np

from quant_fund.models.assignment import (
    bench_assignment,
    brute_assignment,
    hopcroft_karp,
    hungarian,
)


def test_hungarian_small():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    match, val = hungarian(cost)
    assert abs(val - brute_assignment(cost)) < 1e-9
    assert sorted(match.tolist()) == [0, 1, 2]


def test_hungarian_random():
    rng = np.random.default_rng(1)
    c = rng.random((6, 6)) * 8
    _, val = hungarian(c)
    assert abs(val - brute_assignment(c)) < 1e-9


def test_hopcroft_karp_perfect():
    edges = [(i, i) for i in range(10)] + [(i, (i + 1) % 10) for i in range(10)]
    sz, ml = hopcroft_karp(10, 10, edges)
    assert sz == 10


def test_hk_maximal_not_perfect():
    edges = [(0, 0), (0, 1), (1, 0)]
    sz, _ = hopcroft_karp(3, 3, edges)
    assert sz == 2


def test_bench_keys():
    out = bench_assignment()
    assert out["synthetic_hungarian_err"] < 1e-9
    assert out["synthetic_hk_valid"] == 1.0
