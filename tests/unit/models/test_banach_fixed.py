"""Tests for models/banach_fixed.py — non-convergence must fail closed."""

from __future__ import annotations

import math

import pytest


def test_nonconvergent_raises() -> None:
    """A non-contractive map used to return the 10_000th iterate as if it
    were a fixed point. It must raise instead."""
    from quant_fund.models.banach_fixed import fixed_point

    with pytest.raises(ValueError):
        fixed_point(lambda x: x + 1.0, 0.0, maxit=50)
    with pytest.raises(ValueError):
        fixed_point(lambda x: 2.0 * x + 1.0, 0.5, maxit=50)


def test_convergent_still_returns() -> None:
    from quant_fund.models.banach_fixed import fixed_point

    fp = fixed_point(math.cos, 0.5)
    assert abs(math.cos(fp) - fp) < 1e-10
