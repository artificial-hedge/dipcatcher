"""Fail-closed checks for capital, cash buffers, and leverage scales."""

from __future__ import annotations

import math
from numbers import Real


def finite_non_negative(value: object, name: str) -> float:
    """Accept zero. Reject booleans, NaN, infinities, and negatives."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def finite_positive(value: object, name: str) -> float:
    """Reject booleans, NaN, infinities, zero, and negatives."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number
