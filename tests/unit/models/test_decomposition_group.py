"""Tests for decomposition_group — Frobenius order in cyclotomic fields."""

from __future__ import annotations

import threading


def test_mult_order_trivial_modulus_no_hang():
    """n=1: (Z/1Z)* is the trivial group — ord must be 1, not a hang."""
    from quant_fund.models.decomposition_group import mult_order

    out: list[int] = []
    t = threading.Thread(target=lambda: out.append(mult_order(5, 1)), daemon=True)
    t.start()
    t.join(5)
    assert out == [1]


def test_mult_order_nonpositive_modulus_no_hang():
    """n <= 0 has no unit group — ord is 0, not a hang."""
    from quant_fund.models.decomposition_group import mult_order

    out: list[int] = []
    t = threading.Thread(target=lambda: out.append(mult_order(5, -7)), daemon=True)
    t.start()
    t.join(5)
    assert out == [0]


def test_bench_decomposition_group():
    from quant_fund.models.decomposition_group import bench_decomposition_group

    assert bench_decomposition_group()["synthetic_decomposition_group"] == 1.0
