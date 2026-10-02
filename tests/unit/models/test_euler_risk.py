"""Unit tests for quant_fund.models.euler_risk."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.euler_risk import (
    bench_euler_risk,
    euler_es_contributions,
    euler_var_contributions,
    expected_shortfall,
)


def test_es_exceeds_var() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 3))
    w = np.array([0.5, 0.3, 0.2])
    loss = x @ w
    var_ = np.quantile(loss, 0.95)
    es = expected_shortfall(x, w, alpha=0.95)
    assert es > var_


def test_es_contributions_sum_to_es() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4000, 4))
    w = np.array([0.4, 0.2, 0.25, 0.15])
    es = expected_shortfall(x, w, alpha=0.975)
    c = euler_es_contributions(x, w, alpha=0.975)
    assert c.sum() == pytest.approx(es, rel=1e-10)


def test_es_concentration() -> None:
    rng = np.random.default_rng(2)
    base = rng.standard_normal(3000)
    x = np.column_stack([base, 0.01 * rng.standard_normal(3000)])
    w = np.array([0.5, 0.5])
    c = euler_es_contributions(x, w, alpha=0.95)
    assert c[0] > 10 * c[1]


def test_var_contributions_finite() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2000, 3))
    w = np.array([0.6, 0.3, 0.1])
    c = euler_var_contributions(x, w, alpha=0.95)
    assert np.all(np.isfinite(c))
    assert c.shape == (3,)


def test_rejects_mismatched_weights() -> None:
    x = np.random.default_rng(0).standard_normal((100, 3))
    w = np.array([1.0, 0.5])
    with pytest.raises(ValueError):
        expected_shortfall(x, w)
    with pytest.raises(ValueError):
        euler_es_contributions(x, w)


def test_rejects_bad_alpha() -> None:
    x = np.random.default_rng(0).standard_normal((100, 2))
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        expected_shortfall(x, w, alpha=0.3)


def test_bench_euler_risk_score() -> None:
    out = bench_euler_risk()
    assert out["score"] == pytest.approx(1.0)
