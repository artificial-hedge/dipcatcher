"""Unit tests for quant_fund.models.moment_inequalities."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.moment_inequalities import (
    bench_moment_inequalities,
    moment_inequality_test,
)


def test_null_not_rejected() -> None:
    rng = np.random.default_rng(0)
    n, p = 300, 3
    means = np.array([0.0, 0.4, 0.2])
    m = means + rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    out = moment_inequality_test(m, alpha=0.05, n_boot=300, seed=0)
    assert out["reject"] == 0.0
    assert out["p_value"] > 0.05


def test_violated_null_rejected() -> None:
    rng = np.random.default_rng(1)
    n, p = 400, 3
    means = np.array([-0.5, 0.3, 0.2])
    m = means + rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    out = moment_inequality_test(m, alpha=0.05, n_boot=300, seed=0)
    assert out["reject"] == 1.0
    assert out["stat"] > out["crit"]


def test_output_keys() -> None:
    rng = np.random.default_rng(2)
    m = rng.standard_normal((100, 2)) + 0.3
    out = moment_inequality_test(m, alpha=0.1, n_boot=100, seed=0)
    for k in ("stat", "crit", "p_value", "reject"):
        assert k in out
        assert np.isfinite(out[k])


def test_deterministic() -> None:
    rng = np.random.default_rng(3)
    m = rng.standard_normal((150, 3))
    a = moment_inequality_test(m, alpha=0.05, n_boot=150, seed=9)
    b = moment_inequality_test(m, alpha=0.05, n_boot=150, seed=9)
    assert a["stat"] == b["stat"]
    assert a["p_value"] == b["p_value"]


def test_rejects_degenerate() -> None:
    m = np.ones((200, 2))
    with pytest.raises(ValueError):
        moment_inequality_test(m, seed=0)


def test_rejects_too_few_obs() -> None:
    m = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError):
        moment_inequality_test(m, seed=0)


def test_bench_moment_inequalities_score() -> None:
    out = bench_moment_inequalities()
    assert out["synthetic_score"] == pytest.approx(1.0)
