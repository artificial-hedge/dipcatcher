"""Tests for quant_fund.hmm.duration_check."""

from __future__ import annotations

import numpy as np

from quant_fund.hmm.discrete import DiscreteHMM
from quant_fund.hmm.duration_check import duration_bench, dwell_discrepancy


def _hmm() -> DiscreteHMM:
    return DiscreteHMM(
        A=np.array([[0.95, 0.05], [0.05, 0.95]]),
        B=np.array([[0.8, 0.2], [0.15, 0.85]]),
        pi=np.array([0.5, 0.5]),
    )


def test_geometric_dwell_small_gap():
    rng = np.random.default_rng(0)
    states = np.concatenate([[i % 2] * int(d) for i, d in enumerate(rng.geometric(0.05, 300))])[
        :2000
    ]
    rep = dwell_discrepancy(_hmm(), np.asarray(states, dtype=np.int64))
    assert all(v["max_ecdf_gap"] < 0.2 for v in rep.values() if v.get("status") == "ok")


def test_fixed_dwell_flagged():
    states = np.tile(np.repeat([0, 1], 20), 50)
    rep = dwell_discrepancy(_hmm(), states.astype(np.int64))
    assert rep["state_0"]["max_ecdf_gap"] > 0.4


def test_bench_sealed():
    r = duration_bench(seed=0)
    assert r["schema"] == "duration_check.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r == duration_bench(seed=0)
