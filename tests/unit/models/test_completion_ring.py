"""Tests for completion_ring — m-adic filtration and Krull intersection."""

from __future__ import annotations

from quant_fund.models.completion_ring import (
    bench_completion_ring,
    filtr_mult,
    gr_piece,
)


def test_m_pow_membership():
    from quant_fund.models.completion_ring import _in_m_pow

    assert _in_m_pow(3, 3)
    assert _in_m_pow(5, 3)
    assert not _in_m_pow(2, 3)
    assert _in_m_pow(None, 99)


def test_krull_intersection_only_zero():
    # ∩_n m^n = {0}: every finite order eventually falls outside m^n;
    # only the zero element (infinite order) survives. The bench used to
    # assert a literal True here.
    from quant_fund.models.completion_ring import _in_cap_intersect

    assert _in_cap_intersect(None)
    assert not any(_in_cap_intersect(v) for v in range(1, 60))


def test_helpers():
    assert gr_piece(2, 2) == 3
    assert filtr_mult(2, 3) == 5


def test_bench():
    assert bench_completion_ring()["synthetic_completion_ring"] == 1.0
