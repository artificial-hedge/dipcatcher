"""Tests for continued fractions and Pell equations."""

from __future__ import annotations

import pytest

from quant_fund.models.continued_fraction import (
    bench_continued_fraction,
    convergent,
    pell_min,
    sqrt_cf,
)


def test_sqrt_cf_classic() -> None:
    assert sqrt_cf(2) == (1, [2])
    assert sqrt_cf(4) == (2, [])  # perfect square -> empty period


def test_pell_solutions() -> None:
    assert pell_min(2) == (3, 2)
    assert pell_min(3) == (2, 1)
    assert pell_min(23) == (24, 5)


def test_pell_rejects_square_d() -> None:
    # Pell is defined for nonsquare d only; a square d used to return a
    # bogus (x,y) satisfying x^2 - d*y^2 = 0, never 1 — fail closed.
    for d in (0, 1, 4, 9, -3):
        with pytest.raises(ValueError):
            pell_min(d)


def test_convergent() -> None:
    # sqrt(2) convergents: C_0 = 1/1, C_1 = 3/2, C_2 = 7/5
    assert convergent(1, [2], 0) == (1, 1)
    assert convergent(1, [2], 1) == (3, 2)
    assert convergent(1, [2], 2) == (7, 5)


def test_bench_continued_fraction() -> None:
    out = bench_continued_fraction()
    assert out["synthetic_pell_ok"] == 1.0
    assert out["synthetic_conv_bound"] == 1.0
