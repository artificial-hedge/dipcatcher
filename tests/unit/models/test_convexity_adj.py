"""Unit tests for quant_fund.models.convexity_adj."""

from __future__ import annotations

import pytest

from quant_fund.models.convexity_adj import (
    _int_variance,
    bench_convexity,
    convexity_adj_hull,
    convexity_adj_vasicek,
)


def test_positive_and_increasing() -> None:
    c1 = convexity_adj_vasicek(0.1, 0.01, 2.0, 2.25)
    c2 = convexity_adj_vasicek(0.1, 0.01, 2.0, 2.75)
    assert c1 > 0
    assert c2 > c1


def test_hull_approx_positive() -> None:
    assert convexity_adj_hull(0.01, 2.0, 2.25) == pytest.approx(0.5 * 1e-4 * 4.5)


def test_holee_limit() -> None:
    v = _int_variance(1e-12, 0.01, 2.0, 2.25)
    v_ref = 0.01**2 * (0.25**3 / 3.0 + 0.25**2 * 2.0)
    assert v == pytest.approx(v_ref, rel=1e-4)


def test_flat_rate_scales_output() -> None:
    c0 = convexity_adj_vasicek(0.1, 0.01, 2.0, 2.25, flat_rate=0.0)
    c5 = convexity_adj_vasicek(0.1, 0.01, 2.0, 2.25, flat_rate=0.05)
    assert c5 > c0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        convexity_adj_vasicek(-0.1, 0.01, 2.0, 2.25)
    with pytest.raises(ValueError):
        convexity_adj_vasicek(0.1, 0.01, 2.25, 2.0)
    with pytest.raises(ValueError):
        convexity_adj_hull(0.0, 2.0, 2.25)


def test_bench_contract() -> None:
    out = bench_convexity()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_cx_mc_err_bp"] < 1.0
