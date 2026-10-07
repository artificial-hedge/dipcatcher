"""Unit tests for quant_fund.models._ilp_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._ilp_synth import (
    ILP_A,
    ILP_B,
    ILP_C,
    ILP_UB,
    SC_A,
    SC_C,
    brute_force_ilp,
    brute_set_cover,
    simplex,
)


def test_fixture_dims_consistent() -> None:
    assert ILP_A.shape == (2, 2) and ILP_B.shape == (2,) and ILP_C.shape == (2,)
    assert SC_A.shape == (10, 8) and SC_C.shape == (8,)


def test_simplex_solves_fixture() -> None:
    x, v = simplex(ILP_C, ILP_A, ILP_B)
    assert (x >= -1e-9).all()
    assert (ILP_A @ x <= ILP_B + 1e-9).all()
    # LP relaxation upper-bounds the integer optimum
    assert v >= brute_force_ilp(ILP_C, ILP_A, ILP_B, ILP_UB) - 1e-9


def test_simplex_known_answer() -> None:
    # max 5x+8y s.t. x+y<=6, 5x+9y<=45: LP optimum at (2.25, 3.75) = 41.25
    x, v = simplex(ILP_C, ILP_A, ILP_B)
    np.testing.assert_allclose(x, [2.25, 3.75], atol=1e-9)
    np.testing.assert_allclose(v, 41.25, atol=1e-9)


def test_simplex_unbounded_raises() -> None:
    with pytest.raises(ValueError, match="unbounded"):
        simplex(np.array([1.0]), np.array([[-1.0]]), np.array([1.0]))


def test_simplex_rejects_negative_b() -> None:
    # slack basis starts infeasible; silently returning garbage is dishonest
    with pytest.raises(ValueError, match="negative b"):
        simplex(np.array([1.0, 1.0]), np.eye(2), np.array([-1.0, 2.0]))


def test_brute_force_ilp_optimum() -> None:
    v = brute_force_ilp(ILP_C, ILP_A, ILP_B, ILP_UB)
    # hand-checked integer optimum: x=0, y=5 -> 40
    assert v == pytest.approx(40.0)


def test_brute_set_cover_finds_cover() -> None:
    v = brute_set_cover(SC_C, SC_A)
    assert np.isfinite(v)
    # cover cost must be <= sum of all sets and > 0
    assert 0 < v <= SC_C.sum()
