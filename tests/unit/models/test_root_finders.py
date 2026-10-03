"""Classical 1-D root finders and minimizers."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.root_finders import (
    bisection,
    brent_min,
    brent_root,
    golden_min,
    illinois,
    ridders,
    secant,
)

_FINDERS = {
    "bisection": bisection,
    "secant": secant,
    "illinois": illinois,
    "ridders": ridders,
    "brent": brent_root,
}

_CASES = [
    (lambda x: x**3 - 2 * x - 5, 2.0, 3.0, 2.0945514815423265),
    (lambda x: np.cos(x) - x, 0.0, 1.0, 0.7390851332151607),
    (lambda x: np.exp(-x) - x, 0.0, 2.0, 0.5671432904097839),
    (lambda x: x - 0.5 * np.sin(x) - 1.0, 0.0, 3.0, 1.4987011335205458),
]


@pytest.mark.parametrize("case", _CASES)
def test_all_finders_converge(case):
    f, a, b, xtrue = case
    for name, fn in _FINDERS.items():
        r = fn(f, a, b)
        assert abs(r["root"] - xtrue) < 1e-7, (name, r)
        assert abs(r["f"]) < 1e-6, (name, r)


def test_bracket_guard():
    def f(x: float) -> float:
        return x**2 + 1

    for fn in (bisection, illinois, ridders, brent_root):
        with pytest.raises(ValueError):
            fn(f, 0.0, 1.0)


def test_golden_and_brent_agree():
    def f(x: float) -> float:
        return float((x - 0.7) ** 2 + 0.5 * x**4)

    g = golden_min(f, -2.0, 3.0)
    b = brent_min(f, -2.0, 3.0)
    assert abs(g["x"] - b["x"]) < 1e-4
    assert abs(g["x"] - 0.5413511) < 1e-3


def test_brent_min_on_asymmetric_bowl():
    def f(x: float) -> float:
        return float(np.exp(x) - 3 * x)

    b = brent_min(f, -1.0, 3.0)
    # minimum at x = ln 3
    assert abs(b["x"] - np.log(3)) < 1e-4


def test_secant_no_bracket_needed():
    def f(x: float) -> float:
        return x**3

    r = secant(f, -1.0, 0.5)
    # triple root: secant converges linearly here
    assert abs(r["root"]) < 1e-3
