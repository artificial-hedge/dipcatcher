import numpy as np

from quant_fund.models.puct import (
    bench_puct,
    heuristic_prior,
    puct_search,
)


def test_prior_favors_optimal():
    p = heuristic_prior(31)
    assert p[2] > 0.5  # take-3 child (index 2) is optimal


def test_search_visits():
    rng = np.random.default_rng(0)
    visits, wins = puct_search(11, 50, rng)
    assert sum(visits[c] for c in (10, 9, 8)) > 0


def test_bench():
    out = bench_puct(seed=3)
    assert out["synthetic_prior_correct"] == 1.0
    assert out["synthetic_prior_opt_visits"] > 0
