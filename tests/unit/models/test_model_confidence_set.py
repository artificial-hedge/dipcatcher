"""Tests for model_confidence_set — Hansen-Lunde-Nason MCS."""

import numpy as np
import pytest

from quant_fund.models.model_confidence_set import (
    bench_model_confidence_set,
    mcs_test,
    synth_mcs,
)


def test_best_model_survives() -> None:
    losses = synth_mcs(seed=1)
    r = mcs_test(losses, alpha=0.10, n_boot=150, seed=1)
    assert 0 in r["survivors"]


def test_worst_eliminated() -> None:
    losses = synth_mcs(seed=2)
    r = mcs_test(losses, alpha=0.10, n_boot=150, seed=2)
    assert 2 not in r["survivors"]


def test_pvalues_ordered() -> None:
    losses = synth_mcs(seed=3)
    r = mcs_test(losses, alpha=0.10, n_boot=150, seed=3)
    p = r["p_values"]
    assert p[0] >= p[2]


def test_identical_models_all_survive() -> None:
    rng = np.random.default_rng(4)
    base = rng.standard_normal(500) ** 2
    losses = np.column_stack([base + 1e-9 * rng.standard_normal(500) for _ in range(3)])
    r = mcs_test(losses, alpha=0.10, n_boot=100, seed=4)
    assert len(list(r["survivors"])) == 3


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        mcs_test(np.ones((30, 2)))
    with pytest.raises(ValueError):
        mcs_test(np.ones((200, 1)))
    with pytest.raises(ValueError):
        mcs_test(np.full((200, 2), np.nan))


def test_determinism() -> None:
    losses = synth_mcs(seed=6)
    a = mcs_test(losses, alpha=0.10, n_boot=100, seed=6)
    b = mcs_test(losses, alpha=0.10, n_boot=100, seed=6)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_model_confidence_set()
    for k in ("n_survivors", "best_survives", "worst_dropped", "p_best", "p_worst", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
