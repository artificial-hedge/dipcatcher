"""Fail-closed numeric checks for bar prices, sizes, and thresholds."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def non_negative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def min_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def finite_series(values: object, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def non_negative_series(values: object, name: str) -> FloatArray:
    arr = finite_series(values, name)
    if arr.size and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def positive_series(values: object, name: str, *, message: str | None = None) -> FloatArray:
    arr = finite_series(values, name)
    if arr.size and np.any(arr <= 0.0):
        raise ValueError(message or f"{name} must be positive")
    return arr
