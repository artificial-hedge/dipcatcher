"""Unit tests for quant_fund.models.entropy_pooling."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.entropy_pooling import (
    bench_entropy_pool,
    entropy_pool,
    synth_entropy_pool,
)


def test_view_satisfied() -> None:
    x, a, b = synth_entropy_pool(seed=1)
    out = entropy_pool(x, a, b)
    assert float(out["view_viol"]) < 1e-4


def test_posterior_normalized_and_positive() -> None:
    x, a, b = synth_entropy_pool(seed=2)
    p = np.asarray(entropy_pool(x, a, b)["p"])
    assert abs(float(np.sum(p)) - 1.0) < 1e-9
    assert np.all(p > 0)


def test_effective_n_shrinks_but_not_collapsed() -> None:
    x, a, b = synth_entropy_pool(seed=3)
    out = entropy_pool(x, a, b)
    assert x.size / 4.0 < float(out["effective_n"]) < x.size


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        entropy_pool(np.ones(10), np.ones((1, 10)), np.array([1.0]))
    x, a, b = synth_entropy_pool(seed=4)
    with pytest.raises(ValueError):
        entropy_pool(x, a[:, :50], b)


def test_bench_contract() -> None:
    out = bench_entropy_pool()
    assert out["score"] == 1.0
    assert out["synthetic_ep_view_viol"] < 1e-4
