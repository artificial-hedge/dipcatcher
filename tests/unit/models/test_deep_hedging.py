"""Tests for quant_fund.models.deep_hedging — Deep Hedging (Buehler et al. 2019).

References: Buehler, Gonon, Teichmann & Wood (2019, Quantitative Finance
19(10):1271–1291, arXiv:1802.03042); Föllmer & Schied (2016, §4.9, entropic
risk); Rockafellar & Uryasev (2000/2002, ES/CVaR); Markowitz (1952,
variance); Black & Scholes (1973) / Merton (1973, delta baseline); Hamilton
(1989, regime switching); Davis & Norman (1990, proportional costs).

All data here is SYNTHETIC (simulated GBM / regime-switch paths) —
correctness evidence for the algorithm, never market evidence; no
live-trading claims. Torch tests skip cleanly when the nn extra is absent.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from quant_fund.models import deep_hedging as dh


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="deep hedging training requires the nn extra (torch)"
)

S0 = 100.0
STRIKE = 100.0
SIGMA = 0.2
MATURITY = 0.5
STEPS = 24
DT = MATURITY / STEPS
COST = 1e-3  # 10 bp proportional friction
N_PATHS = 2048

RISK_CASES = [
    ("entropic", {"gamma": 0.1}),
    ("expected_shortfall", {"alpha": 0.9}),
    ("variance", {}),
]


def _market(seed: int, n_paths: int = N_PATHS) -> np.ndarray:
    return dh.simulate_gbm_paths(n_paths, STEPS, s0=S0, sigma=SIGMA, dt=DT, seed=seed)


# ---------------------------------------------------------------------------
# numpy core (always run, no torch needed)
# ---------------------------------------------------------------------------


def test_gbm_paths_seeded_shape_and_determinism() -> None:
    a = dh.simulate_gbm_paths(8, 5, s0=S0, sigma=SIGMA, dt=DT, seed=3)
    b = dh.simulate_gbm_paths(8, 5, s0=S0, sigma=SIGMA, dt=DT, seed=3)
    c = dh.simulate_gbm_paths(8, 5, s0=S0, sigma=SIGMA, dt=DT, seed=4)
    assert a.shape == (8, 6)
    assert np.all(a[:, 0] == S0) and np.all(np.isfinite(a)) and np.all(a > 0.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gbm_paths_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        dh.simulate_gbm_paths(0, 5)
    with pytest.raises(ValueError, match="n_steps"):
        dh.simulate_gbm_paths(4, 0)
    with pytest.raises(ValueError, match="sigma"):
        dh.simulate_gbm_paths(4, 5, sigma=0.0)
    with pytest.raises(ValueError, match="s0"):
        dh.simulate_gbm_paths(4, 5, s0=-1.0)
    with pytest.raises(ValueError, match="dt"):
        dh.simulate_gbm_paths(4, 5, dt=0.0)


def test_regime_switch_paths_vol_states_and_fail_closed() -> None:
    # With sticky probability 1.0 in both states the regime never switches, so
    # per-step log-increment dispersion must match the state's sigma.
    for regime, sig in ((0, 0.1), (1, 0.4)):
        paths = dh.simulate_regime_switch_paths(
            512,
            20,
            s0=S0,
            sigma_low=0.1,
            sigma_high=0.4,
            p_stay_low=1.0,
            p_stay_high=1.0,
            dt=DT,
            initial_regime=regime,
            seed=2,
        )
        assert paths.shape == (512, 21) and np.all(paths > 0.0)
        inc_std = float(np.log(paths[:, 1:] / paths[:, :-1]).std())
        assert abs(inc_std - sig * np.sqrt(DT)) < 0.25 * sig * np.sqrt(DT)
    with pytest.raises(ValueError, match="p_stay_low"):
        dh.simulate_regime_switch_paths(4, 5, p_stay_low=1.5)
    with pytest.raises(ValueError, match="initial_regime"):
        dh.simulate_regime_switch_paths(4, 5, initial_regime=2)
    with pytest.raises(ValueError, match="sigma_high"):
        dh.simulate_regime_switch_paths(4, 5, sigma_high=-0.1)


def test_regime_switch_paths_seeded_determinism() -> None:
    a = dh.simulate_regime_switch_paths(16, 12, seed=7)
    b = dh.simulate_regime_switch_paths(16, 12, seed=7)
    c = dh.simulate_regime_switch_paths(16, 12, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_european_payoff_and_fail_closed() -> None:
    paths = np.array([[100.0, 110.0, 90.0]])
    assert np.allclose(dh.european_payoff(paths, 100.0), [0.0])
    assert np.allclose(dh.european_payoff(paths, 100.0, option="put"), [10.0])
    with pytest.raises(ValueError, match="option"):
        dh.european_payoff(paths, 100.0, option="straddle")
    with pytest.raises(ValueError, match="strike"):
        dh.european_payoff(paths, 0.0)
    with pytest.raises(ValueError, match="ndim"):
        dh.european_payoff(np.array([100.0, 110.0]), 100.0)
    with pytest.raises(ValueError, match="at least one path"):
        dh.european_payoff(np.empty((0, 5)), 100.0)
    with pytest.raises(ValueError, match="strictly positive"):
        dh.european_payoff(np.array([[100.0, -1.0]]), 100.0)


def test_black_scholes_delta_bounds_and_fail_closed() -> None:
    paths = _market(1, n_paths=64)
    call = dh.black_scholes_delta(paths, strike=STRIKE, maturity=MATURITY, sigma=SIGMA)
    put = dh.black_scholes_delta(paths, strike=STRIKE, maturity=MATURITY, sigma=SIGMA, call=False)
    assert call.shape == (64, STEPS)
    assert np.all((call >= 0.0) & (call <= 1.0))  # saturates at 1 deep ITM near maturity
    assert np.allclose(put, call - 1.0)
    with pytest.raises(ValueError, match="maturity"):
        dh.black_scholes_delta(paths, strike=STRIKE, maturity=0.0, sigma=SIGMA)
    with pytest.raises(ValueError, match="sigma"):
        dh.black_scholes_delta(paths, strike=STRIKE, maturity=MATURITY, sigma=-0.2)


def test_hedged_loss_hand_computed_accounting() -> None:
    """Short one payoff of 10; long 1 unit over the first step only, 1% costs.

    Trades: +1 at t0 (cost 0.01*100), -1 at t1 (cost 0.01*110); liquidation
    cost 0 (flat at T). Trading P&L = 1*(110-100) = 10. P&L = -10 + 10 - 2.1.
    """
    paths = np.array([[100.0, 110.0, 105.0]])
    comp = dh.hedged_pnl_components(paths, np.array([[1.0, 0.0]]), np.array([10.0]), cost_rate=0.01)
    assert np.allclose(comp["costs"], [2.1])
    assert np.allclose(comp["trading_pnl"], [10.0])
    assert np.allclose(comp["pnl"], [-2.1])
    assert np.allclose(comp["loss"], [2.1])
    # Zero-cost buy-and-hold: loss = payoff - (S_T - S_0).
    loss = dh.hedged_loss(paths, dh.buy_and_hold_positions(paths), np.array([10.0]), cost_rate=0.0)
    assert np.allclose(loss, [10.0 - 5.0])


def test_hedged_loss_fail_closed() -> None:
    paths = _market(2, n_paths=4)
    payoff = dh.european_payoff(paths, STRIKE)
    good = np.zeros((4, STEPS))
    with pytest.raises(ValueError, match="non-negative"):
        dh.hedged_loss(paths, good, payoff, cost_rate=-1e-4)
    with pytest.raises(ValueError, match="positions must have shape"):
        dh.hedged_loss(paths, np.zeros((4, STEPS + 1)), payoff, cost_rate=COST)
    with pytest.raises(ValueError, match="payoff"):
        dh.hedged_loss(paths, good, payoff[:-1], cost_rate=COST)
    with pytest.raises(ValueError, match="finite"):
        dh.hedged_loss(paths, np.full((4, STEPS), np.nan), payoff, cost_rate=COST)


def test_risk_measure_known_values_and_fail_closed() -> None:
    # Entropic risk of a constant loss is that constant (translation invariance).
    const = np.full(64, 3.0)
    assert dh.risk_measure(const, "entropic", gamma=0.7) == pytest.approx(3.0, abs=1e-10)
    assert dh.risk_measure(np.array([1.0, 2.0, 3.0, 4.0]), "expected_shortfall", alpha=0.5) == 3.5
    assert dh.risk_measure(np.array([1.0, 2.0, 3.0]), "variance") == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="unknown risk kind"):
        dh.risk_measure(const, "sharpe")
    with pytest.raises(ValueError, match="gamma"):
        dh.risk_measure(const, "entropic")
    with pytest.raises(ValueError, match="gamma"):
        dh.risk_measure(const, "entropic", gamma=0.0)
    with pytest.raises(ValueError, match="alpha"):
        dh.risk_measure(const, "expected_shortfall", alpha=1.5)
    with pytest.raises(ValueError, match="non-empty"):
        dh.risk_measure(np.array([]), "variance")
    with pytest.raises(ValueError, match="at least two losses"):
        dh.risk_measure(np.array([1.0]), "variance")
    with pytest.raises(ValueError, match="finite"):
        dh.risk_measure(np.array([1.0, np.nan]), "variance")


def test_module_imports_without_torch_and_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module has no top-level torch import; torch entry points fail closed."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    spec = importlib.util.spec_from_file_location("_dh_no_torch", dh.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_dh_no_torch", probe)  # dataclasses resolves via sys.modules
    spec.loader.exec_module(probe)  # must import cleanly without torch
    # numpy core stays usable while torch is blocked
    paths = probe.simulate_gbm_paths(4, 3, seed=0)
    assert paths.shape == (4, 4)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.deep_hedge(paths, probe.european_payoff(paths, 100.0), cost_rate=COST)
    with pytest.raises(ImportError, match=r"'nn' extra"):
        probe.compare_hedges(paths, strike=100.0, maturity=MATURITY, sigma=SIGMA, cost_rate=COST)


# ---------------------------------------------------------------------------
# torch lane (skipped when the nn extra is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market() -> tuple[np.ndarray, np.ndarray]:
    """Seeded SYNTHETIC train/eval batches (independent seeds)."""
    return _market(11), _market(99)


@requires_torch
@pytest.mark.parametrize(("kind", "params"), RISK_CASES)
def test_deep_hedge_beats_unhedged_under_friction(
    market: tuple[np.ndarray, np.ndarray], kind: str, params: dict
) -> None:
    """Hedged risk <= unhedged risk with 10 bp friction, in and out of sample."""
    paths, eval_paths = market
    payoff = dh.european_payoff(paths, STRIKE)
    res = dh.deep_hedge(
        paths,
        payoff,
        cost_rate=COST,
        risk=kind,
        hidden=(16, 16),
        epochs=150,
        lr=8e-3,
        seed=5,
        **params,
    )
    unhedged = dh.hedged_loss(paths, np.zeros_like(res.positions), payoff, cost_rate=COST)
    assert res.risk_kind == kind
    assert res.positions.shape == (N_PATHS, STEPS)
    assert np.all(np.abs(res.positions) <= 1.0)  # bounded admissible strategy
    assert np.all(np.isfinite(res.loss_curve))
    assert res.loss_curve[-1] < res.loss_curve[0]  # training reduced the risk
    assert res.risk <= dh.risk_measure(unhedged, kind, **params)
    # Out-of-sample: frozen strategy on an independently seeded batch.
    oos_pos = res.strategy_positions(eval_paths)
    oos_payoff = dh.european_payoff(eval_paths, STRIKE)
    oos_hedged = dh.hedged_loss(eval_paths, oos_pos, oos_payoff, cost_rate=COST)
    oos_unhedged = dh.hedged_loss(eval_paths, np.zeros_like(oos_pos), oos_payoff, cost_rate=COST)
    assert dh.risk_measure(oos_hedged, kind, **params) <= dh.risk_measure(
        oos_unhedged, kind, **params
    )


@requires_torch
def test_deep_hedge_deterministic_given_seed(market: tuple[np.ndarray, np.ndarray]) -> None:
    paths, _ = market
    payoff = dh.european_payoff(paths, STRIKE)
    kwargs = dict(cost_rate=COST, risk="variance", hidden=(8,), epochs=40, lr=8e-3)
    a = dh.deep_hedge(paths, payoff, seed=5, **kwargs)
    b = dh.deep_hedge(paths, payoff, seed=5, **kwargs)
    c = dh.deep_hedge(paths, payoff, seed=6, **kwargs)
    assert np.array_equal(a.positions, b.positions)
    assert a.risk == b.risk
    assert np.array_equal(a.loss_curve, b.loss_curve)
    assert not np.array_equal(a.positions, c.positions)


@requires_torch
def test_deep_hedge_fail_closed(market: tuple[np.ndarray, np.ndarray]) -> None:
    paths, eval_paths = market
    payoff = dh.european_payoff(paths, STRIKE)
    with pytest.raises(ValueError, match="payoff"):
        dh.deep_hedge(paths, payoff[:-1], cost_rate=COST)
    with pytest.raises(ValueError, match="non-negative"):
        dh.deep_hedge(paths, payoff, cost_rate=-0.01)
    with pytest.raises(ValueError, match="epochs"):
        dh.deep_hedge(paths, payoff, cost_rate=COST, epochs=0)
    with pytest.raises(ValueError, match="hidden"):
        dh.deep_hedge(paths, payoff, cost_rate=COST, hidden=())
    with pytest.raises(ValueError, match="lr"):
        dh.deep_hedge(paths, payoff, cost_rate=COST, lr=0.0)
    with pytest.raises(ValueError, match="unknown risk kind"):
        dh.deep_hedge(paths, payoff, cost_rate=COST, risk="sortino")
    res = dh.deep_hedge(
        paths, payoff, cost_rate=COST, risk="variance", hidden=(8,), epochs=5, seed=0
    )
    with pytest.raises(ValueError, match="columns to match"):
        res.strategy_positions(eval_paths[:, :-1])


@requires_torch
def test_compare_hedges_metrics_and_ordering(market: tuple[np.ndarray, np.ndarray]) -> None:
    """Scorecard-style comparison: learned vs analytic delta / static / unhedged."""
    paths, eval_paths = market
    cmp = dh.compare_hedges(
        paths,
        strike=STRIKE,
        maturity=MATURITY,
        sigma=SIGMA,
        cost_rate=COST,
        risk="entropic",
        gamma=0.1,
        hidden=(16, 16),
        epochs=150,
        lr=8e-3,
        seed=5,
        eval_paths=eval_paths,
    )
    m = cmp.metrics
    assert cmp.risk_kind == "entropic"
    assert set(cmp.risk) == {"deep", "bs_delta", "buy_and_hold", "unhedged"}
    assert all(np.isfinite(v) for v in m.values())
    # The learned hedge cuts risk vs the unhedged short payoff and the static hedge.
    assert m["dh_hedged_risk"] <= m["dh_unhedged_risk"]
    assert m["dh_hedged_risk"] <= m["dh_buy_and_hold_risk"]
    # Sanity: it captures at least half of the analytic delta's hedging benefit
    # under the same friction (Buehler et al. 2019 recover near-delta hedges).
    benefit = m["dh_unhedged_risk"] - m["dh_bs_delta_friction_risk"]
    assert benefit > 0.0
    assert m["dh_friction_gap_vs_bs_delta"] < 0.5 * benefit
    assert m["dh_unhedged_risk"] > 0.0
    assert cmp.mean_cost["unhedged"] == 0.0
    assert cmp.mean_cost["bs_delta"] > 0.0
    with pytest.raises(ValueError, match="time points"):
        dh.compare_hedges(
            paths,
            strike=STRIKE,
            maturity=MATURITY,
            sigma=SIGMA,
            cost_rate=COST,
            risk="variance",
            epochs=2,
            eval_paths=eval_paths[:, :-1],
        )


@requires_torch
def test_regime_switch_paths_end_to_end() -> None:
    """Training on regime-switch SYNTHETIC paths also beats the unhedged book."""
    paths = dh.simulate_regime_switch_paths(
        1024, STEPS, s0=S0, sigma_low=0.1, sigma_high=0.4, dt=DT, seed=13
    )
    payoff = dh.european_payoff(paths, STRIKE)
    res = dh.deep_hedge(
        paths, payoff, cost_rate=COST, risk="variance", hidden=(16, 16), epochs=120, lr=8e-3, seed=4
    )
    unhedged = dh.hedged_loss(paths, np.zeros_like(res.positions), payoff, cost_rate=COST)
    assert res.risk <= dh.risk_measure(unhedged, "variance")
    assert np.all(np.abs(res.positions) <= 1.0)
