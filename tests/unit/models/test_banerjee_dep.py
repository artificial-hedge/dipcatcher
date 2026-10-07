"""Tests for models/banerjee_dep.py — direction must cover the symmetric case."""

from __future__ import annotations

from fractions import Fraction


def test_direction_constant_write_affine_read() -> None:
    """Write A[5], read A[i_r + 3]: the dependence hits i_r = 2. The
    a==0/b!=0 case used to return None — reporting no dependence where
    one exists, which a conservative oracle may never do."""
    from quant_fund.models.banerjee_dep import direction

    assert direction(([0], 5), ([1], 3), [(0, 10)], 0) == Fraction(2)
    # rational (non-integer) distances are reported exactly too
    assert direction(([0], 5), ([2], 4), [(0, 10)], 0) == Fraction(1, 2)


def test_direction_affine_write_constant_read() -> None:
    from quant_fund.models.banerjee_dep import direction

    assert direction(([1], 0), ([0], 4), [(0, 10)], 0) == Fraction(4)
