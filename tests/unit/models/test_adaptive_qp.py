"""Tests for adaptive query execution (models/adaptive_qp.py)."""

import numpy as np

from quant_fund.models.adaptive_qp import bench_adaptive_qp


def test_adaptive_choice_matches_oracle_at_full_probe():
    from quant_fund.models.adaptive_qp import _adaptive_choice, _run_order

    rng = np.random.RandomState(0)
    for _ in range(25):
        a = np.arange(rng.randint(10, 500))
        b = np.arange(rng.randint(10, 500))
        c = np.arange(rng.randint(10, 500))
        chosen = _adaptive_choice(a, b, c, rng, frac=1.0)
        oracle = min(("ab_c", "bc_a"), key=lambda o: _run_order(a, b, c, o))
        assert _run_order(a, b, c, chosen) == _run_order(a, b, c, oracle)


def test_bench_reports_oracle_metrics():
    out = bench_adaptive_qp(seed=3)
    assert 0.0 <= out["synthetic_adaptive_oracle_match"] <= 1.0
    assert 0.0 <= out["synthetic_adaptive_beats_worst"] <= 1.0
    assert out["synthetic_adaptive_regret"] >= 0.0
    assert out["synthetic_probe_frac"] > 0.0


def test_bench_deterministic():
    assert bench_adaptive_qp(seed=9) == bench_adaptive_qp(seed=9)
