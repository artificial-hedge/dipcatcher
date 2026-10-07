"""Unit tests for quant_fund.models.mutual_info."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mutual_info import (
    bench_mutual_info,
    cond_ksg_mi,
    ksg_mi,
)


def test_independent_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1500)
    y = rng.standard_normal(1500)
    assert abs(ksg_mi(x, y, k=4)) < 0.1


def test_gaussian_recovers_closed_form() -> None:
    rng = np.random.default_rng(1)
    rho = 0.7
    x = rng.standard_normal(2000)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(2000)
    mi = ksg_mi(x, y, k=4)
    true_mi = -0.5 * np.log(1 - rho**2)
    assert mi == pytest.approx(true_mi, abs=0.08)


def test_conditional_zero_when_mediation() -> None:
    # x -> z -> y chain: I(x;y|z) should be much smaller than I(x;y).
    rng = np.random.default_rng(2)
    x = rng.standard_normal(800)
    z = x + 0.1 * rng.standard_normal(800)
    y = z + 0.1 * rng.standard_normal(800)
    cmi = cond_ksg_mi(x, y, z, k=4)
    mi = ksg_mi(x, y, k=4)
    assert cmi < mi
    assert cmi < 0.2


def test_input_validation() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(100)
    with pytest.raises(ValueError):
        ksg_mi(x, rng.standard_normal(50))
    with pytest.raises(ValueError):
        ksg_mi(np.zeros(100), x)
    with pytest.raises(ValueError):
        cond_ksg_mi(x, rng.standard_normal(100), np.zeros(100))


def test_bench_score() -> None:
    out = bench_mutual_info()
    assert out["synthetic_score"] == 1.0
