from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.glft_closed_form import (
    glft_bench,
    mc_value_check,
    quotes_from_theta,
    solve_theta_ode,
    theta_closed_form,
)

G, K, A, SIG, T, Q = 0.05, 1.5, 1.5, 0.02, 1.0, 3


def test_theta_terminal_zero() -> None:
    times, thetas = solve_theta_ode(G, K, A, SIG, T, Q, n_steps=200)
    np.testing.assert_allclose(thetas[-1], 0.0, atol=1e-12)
    assert times[-1] == pytest.approx(T)


def test_theta_symmetry() -> None:
    _, thetas = solve_theta_ode(G, K, A, SIG, T, Q, n_steps=400)
    np.testing.assert_allclose(thetas[0], thetas[0][::-1], atol=1e-10)


def test_inventory_skew_sign() -> None:
    """Long inventory ⇒ tighter ask (wants to sell) and wider bid."""
    _, thetas = solve_theta_ode(G, K, A, SIG, T, Q, n_steps=400)
    ask, bid = quotes_from_theta(thetas[0], G, K, Q)
    i_long = 2 * Q  # q = +Q
    i_short = 0  # q = −Q
    assert bid[i_long - 1] > bid[Q]  # near-boundary long: wider bid
    assert ask[i_long] < ask[Q - 1]  # monotone tighter as q grows
    assert np.isnan(ask[i_short]) and np.isnan(bid[i_long])


def test_closed_form_converges_at_small_xi() -> None:
    """Linearization error shrinks as k·σ²T·Q → 0 (the real axis)."""
    errs = []
    for sig in (0.05, 0.005):
        _, thetas = solve_theta_ode(G, K, A, sig, T, Q, n_steps=2000)
        tl = theta_closed_form(G, K, A, sig, T, Q)
        ao, bo = quotes_from_theta(thetas[0], G, K, Q)
        al, bl = quotes_from_theta(tl, G, K, Q)
        errs.append(np.nanmax(np.abs(al - ao)) + np.nanmax(np.abs(bl - bo)))
    assert errs[1] < errs[0]


def test_linearized_theta_matches_ode_small_sigma() -> None:
    _, thetas = solve_theta_ode(G, K, A, 0.005, T, Q, n_steps=2000)
    tl = theta_closed_form(G, K, A, 0.005, T, Q)
    np.testing.assert_allclose(thetas[0], tl, rtol=0.05, atol=2e-3)


def test_mc_value_consistency() -> None:
    out = mc_value_check(0.05, K, A, SIG, T, Q, n_paths=4000, n_steps=100, seed=3)
    # MC estimate within ~5 standard errors of the PDE value
    assert out["abs_diff"] < 5.0 * out["mc_se"] + 1e-4


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        solve_theta_ode(-1.0, K, A, SIG, T, Q)
    with pytest.raises(ValueError):
        theta_closed_form(G, 0.0, A, SIG, T, Q)
    with pytest.raises(ValueError):
        theta_closed_form(G, K, A, SIG, T, Q, t=T + 1)


def test_bench_structure() -> None:
    out = glft_bench(gammas=(0.05,), Ts=(0.02, 1.0), seed=3)
    assert out["schema"] == "glft_bench.v1"
    assert out["data_label"] == "SYNTHETIC"
    assert out["mc_check"]["abs_diff"] < 0.05
    cell = out["cells"][0]
    assert cell["xi"] > 0.0 and cell["quote_err_ticks_max"] >= 0.0
    # the short-horizon cell must be closer to truth than the long one
    assert out["cells"][0]["quote_err_ticks_max"] <= out["cells"][1]["quote_err_ticks_max"]
