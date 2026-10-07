"""Probes: root-finder honesty — real iteration counts, fail-closed on
non-finite interior evaluations, secant must not return a diverged
iterate as 'root'."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import root_finders as rf


def test_iters_are_measured_not_zero() -> None:
    f = lambda x: x**3 - 2 * x - 5  # noqa: E731
    for name in ("bisection", "secant", "illinois", "ridders", "brent_root"):
        r = getattr(rf, name)(f, 2.0, 3.0)
        assert r["iters"] > 0, f"{name} reported iters=0"
    for name in ("golden_min", "brent_min"):
        r = getattr(rf, name)(lambda x: (x - 0.7) ** 2, -2.0, 3.0)
        assert r["iters"] > 0, f"{name} reported iters=0"


def test_secant_fails_closed_on_divergence() -> None:
    # 1/x with symmetric start: next iterate lands exactly on the pole.
    with pytest.raises(ValueError, match="secant diverged"):
        rf.secant(lambda x: 1.0 / x if x != 0 else np.inf, -0.1, 0.1)


def test_secant_no_root_from_diverged_iterate() -> None:
    # The old code assigned x1=x2 and returned inf as "root".
    r: dict[str, float] = {}
    try:
        r = rf.secant(lambda x: 1.0 / x if x != 0 else np.inf, -0.1, 0.1)
    except ValueError:
        return
    assert np.isfinite(r["root"]) and np.isfinite(r["f"])


def test_bisection_fails_closed_on_interior_pole() -> None:
    # Pole at 0.5, hit exactly on the first midpoint eval of [0, 1].
    f = lambda x: np.inf if x == 0.5 else x - 0.3  # noqa: E731
    with pytest.raises(ValueError, match="non-finite"):
        rf.bisection(f, 0.0, 1.0)


def test_ridders_fails_closed_on_interior_pole() -> None:
    f = lambda x: np.inf if x == 0.5 else x - 0.3  # noqa: E731
    with pytest.raises(ValueError, match="non-finite"):
        rf.ridders(f, 0.0, 1.0)


def test_illinois_fails_closed_on_interior_pole() -> None:
    # First false-position eval lands exactly on 0.8.
    f = lambda x: np.inf if x == 0.8 else x - 0.8  # noqa: E731
    with pytest.raises(ValueError, match="non-finite"):
        rf.illinois(f, 0.0, 1.0)


def test_brent_root_fails_closed_on_interior_pole() -> None:
    # Brent's first accepted interpolation step on x^3-0.3 over [0,1]
    # evaluates x=0.3 (the secant point); a pole there must raise.
    f = lambda x: np.inf if x == 0.3 else x**3 - 0.3  # noqa: E731
    with pytest.raises(ValueError, match="non-finite"):
        rf.brent_root(f, 0.0, 1.0)


def test_battery_still_converges() -> None:
    for name in ("bisection", "illinois", "ridders", "brent_root"):
        r = getattr(rf, name)(lambda x: np.cos(x) - x, 0.0, 1.0)
        assert abs(r["root"] - 0.7390851332151607) < 1e-9
