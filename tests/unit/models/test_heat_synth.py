"""Unit tests for quant_fund.models._heat_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._heat_synth import (
    MU0,
    S0,
    SIG,
    T_,
    L,
    eval_error,
    grid,
    mc_paths,
    rel_l2,
    u0,
    u_exact,
)


def test_grid_spans_domain() -> None:
    x = grid()
    assert x.shape == (101,)
    assert x[0] == -L and x[-1] == L


def test_grid_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="nx"):
        grid(1)
    with pytest.raises(ValueError, match="nx"):
        grid(0)


def test_u_exact_matches_initial_at_t0() -> None:
    x = grid()
    np.testing.assert_allclose(u_exact(x, 0.0), u0(x), atol=1e-12)


def test_u_exact_spreads_gaussian() -> None:
    x = grid()
    # variance widens by SIG^2 * t: peak drops
    assert u_exact(x, T_).max() < u0(x).max()
    # mass conserved ~1
    assert abs(np.trapezoid(u_exact(x, T_), x) - 1.0) < 0.01


def test_u_exact_rejects_negative_time() -> None:
    # t<0 runs the heat equation backward (ill-posed); variance went
    # negative under a sqrt -> nan laundering. Must raise.
    x = grid()
    with pytest.raises(ValueError, match="t"):
        u_exact(x, -0.5)
    with pytest.raises(ValueError, match="t"):
        u_exact(x, np.inf)


def test_eval_error_zero_for_exact_solver() -> None:
    assert eval_error(u_exact) < 1e-12


def test_eval_error_positive_for_bad_solver() -> None:
    assert eval_error(lambda x, t: u0(x)) > 0.0


def test_rel_l2_rejects_zero_truth() -> None:
    with pytest.raises(ValueError, match="zero truth"):
        rel_l2(np.ones(5), np.zeros(5))


def test_mc_paths_deterministic_and_gaussian() -> None:
    a = mc_paths(4000, T_, seed=0)
    b = mc_paths(4000, T_, seed=0)
    np.testing.assert_array_equal(a, b)
    # terminal law: N(MU0, S0^2 + SIG^2*T_)
    var = S0**2 + SIG**2 * T_
    assert abs(a.mean() - MU0) < 0.05
    assert abs(a.var() - var) / var < 0.06


def test_mc_paths_rejects_negative_time() -> None:
    # sqrt(negative t) produces nan paths — laundering
    with pytest.raises(ValueError, match="t"):
        mc_paths(8, -1.0, seed=0)
    with pytest.raises(ValueError, match="n"):
        mc_paths(0, T_, seed=0)
