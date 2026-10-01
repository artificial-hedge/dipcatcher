"""Tests for microstructure/zi_lob_simulator.py — lane B4-i.

Zero-intelligence LOB simulator + classical market-making baselines (AS 2008,
GLFT 2012) + emergent microstructure diagnostics. Foundation for the Moret &
Lillo (2026, arXiv:2609.11614) RL market-maker lane. Everything is **labeled
SYNTHETIC** correctness validation — never market evidence, no live-trading
claim. Simulator-internal mark-to-market accounting is namespaced
``sim_internal_*`` and asserted here to never leak as a headline metric.

Fast: the expensive session/impact bundles are computed once in module-scoped
fixtures. Pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
    as_policy,
    avellaneda_stoikov_quotes,
    book_phase_metrics,
    glft_policy,
    glft_quotes,
    metaorder_impact_slope,
    order_flow_autocorrelation,
    regime_flow_diagnostics,
    run_mm_session,
    santa_fe_config,
)
from quant_fund.models.market_making import as_optimal_quotes

# ---------------------------------------------------------------------------
# Shared pins (deterministic; numbers verified against seeded runs)
# ---------------------------------------------------------------------------

SIGMA = 0.02
KAPPA = 1000.0  # per price unit (= 10 per tick at tick=0.01): tight, active quoting
AS_GAMMA = 0.002
GLFT_GAMMA = 1.0
GLFT_A = 1.0
CAP = 40
HORIZON = 3000.0
FLOW_STATES = (
    RegimeState("buy_pressure", 1.0, 0.94),
    RegimeState("sell_pressure", 1.0, 0.06),
)
STAY = (0.9975, 0.9975)  # sticky regimes on the MO clock

# Forbidden *headline* metric tokens (mirrors research.catalog registry). "pnl"
# is permitted ONLY under the simulator-internal diagnostic namespace.
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")


def _as_pol():
    return as_policy(gamma=AS_GAMMA, sigma=SIGMA, kappa=KAPPA, tick=0.01)


def _glft_pol():
    return glft_policy(gamma=GLFT_GAMMA, sigma=SIGMA, kappa=KAPPA, a_fill=GLFT_A, tick=0.01)


def _all_keys(obj: object) -> list[str]:
    """Recursively collect mapping keys (dicts only; walk list/tuple values)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


# ---------------------------------------------------------------------------
# Module-scoped fixtures for the expensive bundles (computed once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mm_results() -> dict[str, dict]:
    """AS/GLFT × stationary/regime session bundles at a pinned seed."""
    out: dict[str, dict] = {}
    for label, mk in (("as", _as_pol), ("glft", _glft_pol)):
        out[f"{label}_stat"] = run_mm_session(
            config=santa_fe_config(seed=7),
            policy=mk(),
            horizon=HORIZON,
            decision_interval=1.0,
            inventory_cap=CAP,
        )
        flow = MarkovRegimeFlow(FLOW_STATES, STAY, seed=11)
        out[f"{label}_reg"] = run_mm_session(
            config=santa_fe_config(seed=7),
            policy=mk(),
            horizon=HORIZON,
            decision_interval=1.0,
            inventory_cap=CAP,
            flow=flow,
        )
    return out


@pytest.fixture(scope="module")
def impact_result() -> dict:
    """Square-root impact slope in the volume-diffusion (ref-anchored) regime."""
    cfg = ZILobConfig(
        s0=100.0,
        tick=0.01,
        lam=0.06,
        mu=0.10,
        theta_cxl=0.02,
        band=40,
        density_exponent=1.0,
        anchor="ref",
        ref_halflife=100.0,
        seed=0,
    )
    return metaorder_impact_slope(
        config=cfg,
        sizes=(16, 32, 64, 128, 256),
        n_seeds=6,
        inject_rate=10.0 * cfg.mu,
        warmup=300.0,
        measure="transient",
    )


# ---------------------------------------------------------------------------
# Config / calibration
# ---------------------------------------------------------------------------


def test_santa_fe_calibration_matches_paper() -> None:
    cfg = santa_fe_config(seed=3)
    # Moret & Lillo (2026) Eq. 4: lambda=0.06, mu=0.10, theta_cxl=0.02 per second.
    assert cfg.lam == pytest.approx(0.06)
    assert cfg.mu == pytest.approx(0.10)
    assert cfg.theta_cxl == pytest.approx(0.02)
    assert cfg.seed == 3
    assert cfg.anchor == "touch"  # MM default
    assert cfg.density_exponent == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"s0": -1.0},
        {"tick": 0.0},
        {"lam": -0.1},
        {"mu": 0.0},
        {"theta_cxl": float("nan")},
        {"p_buy": 1.5},
        {"p_buy": -0.2},
        {"band": 0},
        {"density_exponent": -1.0},
        {"ref_halflife": -5.0},
        {"ref_fill_gain": -0.1},
        {"ref_fill_gain": float("nan")},
        {"anchor": "sideways"},
        {"init_levels": 2, "init_depth": 0},
        {"seed": 1.5},
    ],
)
def test_config_fail_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ZILobConfig(**kwargs)


# ---------------------------------------------------------------------------
# RegimeState / MarkovRegimeFlow
# ---------------------------------------------------------------------------


def test_regime_state_fail_closed() -> None:
    with pytest.raises(ValueError):
        RegimeState("", 1.0, 0.5)
    with pytest.raises(ValueError):
        RegimeState("x", 0.0, 0.5)
    with pytest.raises(ValueError):
        RegimeState("x", 1.0, 1.5)


def test_markov_flow_fail_closed() -> None:
    s = (RegimeState("a", 1.0, 0.6), RegimeState("b", 1.0, 0.4))
    with pytest.raises(ValueError):
        MarkovRegimeFlow((s[0],), (0.9,), seed=0)  # only one state
    with pytest.raises(ValueError):
        MarkovRegimeFlow(s, (0.9,), seed=0)  # one stay prob
    with pytest.raises(ValueError):
        MarkovRegimeFlow(s, (0.9, 1.5), seed=0)  # stay prob > 1
    with pytest.raises(ValueError):
        MarkovRegimeFlow(s, (0.9, 0.9), seed=0).expected_p_buy()  # before any MO


def test_markov_flow_sticky_and_deterministic() -> None:
    s = (RegimeState("a", 1.0, 0.9), RegimeState("b", 1.0, 0.1))
    f1 = MarkovRegimeFlow(s, (1.0, 1.0), seed=5)  # never switch
    for _ in range(100):
        f1.advance()
    assert f1.state_index == 0
    assert f1.transitions == []
    assert f1.expected_p_buy() == pytest.approx(0.9)

    # Same seed -> identical transition path.
    fa = MarkovRegimeFlow(s, (0.9, 0.9), seed=42)
    fb = MarkovRegimeFlow(s, (0.9, 0.9), seed=42)
    for _ in range(200):
        fa.advance()
        fb.advance()
    assert fa.transitions == fb.transitions
    assert fa.state_mo_counts == fb.state_mo_counts
    assert fa.n_mo == 200


# ---------------------------------------------------------------------------
# Simulator engine: grid, determinism, conservation, FIFO, marketable guard
# ---------------------------------------------------------------------------


def test_price_level_grid_fail_closed() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=1))
    assert sim.price_to_level(100.0) == 0
    assert sim.price_to_level(100.01) == 1
    assert sim.level_to_price(-2) == pytest.approx(99.98)
    with pytest.raises(ValueError):
        sim.price_to_level(100.005)  # off-grid
    with pytest.raises(ValueError):
        sim.price_to_level(-1.0)  # non-positive


def test_determinism_same_seed() -> None:
    a = ZILobSimulator(santa_fe_config(seed=5))
    a.run(200.0)
    b = ZILobSimulator(santa_fe_config(seed=5))
    b.run(200.0)
    ta = [(t.t, t.price, t.aggressor, t.level) for t in a.trades]
    tb = [(t.t, t.price, t.aggressor, t.level) for t in b.trades]
    assert ta == tb
    assert a.t == pytest.approx(b.t)
    assert a.event_counts() == b.event_counts()


def test_seed_sensitivity() -> None:
    a = ZILobSimulator(santa_fe_config(seed=1))
    a.run(300.0)
    b = ZILobSimulator(santa_fe_config(seed=2))
    b.run(300.0)
    ta = [(round(t.t, 9), t.price) for t in a.trades]
    tb = [(round(t.t, 9), t.price) for t in b.trades]
    assert ta != tb


def test_order_conservation() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=9))
    sim.run(500.0)
    ec = sim.event_counts()
    assert ec["n_orders_created"] == ec["n_fills"] + ec["n_cancellations"] + ec["resting"]
    assert ec["n_fills"] == ec["n_mo_arrivals"] - ec["n_mo_noop"]
    assert ec["n_orders_created"] > 0


def test_price_time_priority_fifo() -> None:
    """Oldest order at a level fills first; queue_position tracks depth."""
    sim = ZILobSimulator(santa_fe_config(seed=4))
    sim.run(50.0)
    bid_px = sim.best_bid
    assert bid_px is not None
    first = sim.submit_limit_order("buy", bid_px, "zi")  # joins back of best bid
    second = sim.submit_limit_order("buy", bid_px, "zi")
    assert sim.queue_position(second) == sim.queue_position(first) + 1
    ahead = sim.queue_position(first)
    assert ahead is not None and ahead >= 0
    # Drain everything ahead of `first` (front of the FIFO queue), then one more
    # to consume `first` itself; `second` (behind it) must survive.
    for _ in range(ahead + 1):
        sim.inject_market_order("sell")
    assert not sim.order_alive(first)
    assert sim.order_alive(second)


def test_marketable_limit_rejected() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=6))
    sim.run(50.0)
    ask = sim.best_ask
    bid = sim.best_bid
    assert ask is not None and bid is not None
    with pytest.raises(ValueError):
        sim.submit_limit_order("buy", ask)  # crosses -> marketable
    with pytest.raises(ValueError):
        sim.submit_limit_order("sell", bid)  # crosses -> marketable


def test_inject_market_order_consumes_best() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=8))
    sim.run(50.0)
    ask0 = sim.best_ask
    assert ask0 is not None
    trades = sim.inject_market_order("buy", qty=1)
    assert len(trades) == 1
    assert trades[0].aggressor == "buy"
    assert trades[0].price == pytest.approx(ask0)
    # Fail-closed on bad qty.
    with pytest.raises(ValueError):
        sim.inject_market_order("buy", qty=0)


def test_run_rejects_nonfinite_horizon() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=1))
    with pytest.raises(ValueError):
        sim.run(float("inf"))


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov quotes (composes models.market_making)
# ---------------------------------------------------------------------------


def test_as_quotes_compose_with_models_module() -> None:
    """Without tick snapping, must equal the models.market_making closed form."""
    mid, q, gamma, sigma, tau, kappa = 100.0, 5.0, 0.1, 0.02, 0.5, 1.5
    got = avellaneda_stoikov_quotes(mid, q, gamma=gamma, sigma=sigma, tau=tau, kappa=kappa)
    ref = as_optimal_quotes(mid, q, gamma, sigma, tau, kappa)
    assert got["reservation_price"] == pytest.approx(ref["reservation_price"])
    assert got["bid"] == pytest.approx(ref["bid"])
    assert got["ask"] == pytest.approx(ref["ask"])
    assert got["half_spread"] == pytest.approx(ref["half_spread"])
    assert got["skew"] == pytest.approx(ref["skew"])
    assert got["snapped"] is False
    # Long inventory -> reservation price below mid (skew down to sell).
    assert got["skew"] < 0.0


def test_as_quotes_tick_snapping_widens() -> None:
    mid, q, gamma, sigma, tau, kappa = 100.0, 3.0, 0.1, 0.02, 0.5, 1.5
    tick = 0.01
    raw = avellaneda_stoikov_quotes(mid, q, gamma=gamma, sigma=sigma, tau=tau, kappa=kappa)
    snap = avellaneda_stoikov_quotes(
        mid, q, gamma=gamma, sigma=sigma, tau=tau, kappa=kappa, tick=tick
    )
    assert snap["snapped"] is True
    # Bid floored, ask ceiled -> snapped spread >= raw spread, both on grid.
    assert snap["bid"] <= raw["bid"] + 1e-12
    assert snap["ask"] >= raw["ask"] - 1e-12
    assert (snap["ask"] - snap["bid"]) >= (raw["ask"] - raw["bid"]) - 1e-9
    assert abs(snap["bid"] / tick - round(snap["bid"] / tick)) < 1e-6
    assert abs(snap["ask"] / tick - round(snap["ask"] / tick)) < 1e-6


def test_as_quotes_fail_closed() -> None:
    with pytest.raises(ValueError):
        avellaneda_stoikov_quotes(100.0, 0.0, gamma=0.0, sigma=0.02, tau=0.5, kappa=1.5)
    with pytest.raises(ValueError):
        avellaneda_stoikov_quotes(100.0, 0.0, gamma=0.1, sigma=0.02, tau=0.5, kappa=1.5, tick=0.0)


# ---------------------------------------------------------------------------
# GLFT quotes (closed form; not in models.market_making)
# ---------------------------------------------------------------------------


def _glft_manual(mid, q, gamma, sigma, kappa, a):
    ratio = 1.0 + gamma / kappa
    base = math.log(ratio) / gamma
    root = math.sqrt((sigma * sigma * gamma / (2.0 * kappa * a)) * ratio ** (1.0 + kappa / gamma))
    return base, root, mid - base - ((2 * q + 1) / 2) * root, mid + base - ((2 * q - 1) / 2) * root


def test_glft_closed_form_matches_manual() -> None:
    mid, q, gamma, sigma, kappa, a = 100.0, 4.0, 1.0, 0.02, 1000.0, 1.0
    base, root, bid, ask = _glft_manual(mid, q, gamma, sigma, kappa, a)
    got = glft_quotes(mid, q, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a)
    assert got["base_half_spread"] == pytest.approx(base)
    assert got["inventory_skew_unit"] == pytest.approx(root)
    assert got["bid"] == pytest.approx(bid)
    assert got["ask"] == pytest.approx(ask)
    assert got["spread"] == pytest.approx(ask - bid)
    assert got["bid"] < got["ask"]
    assert got["snapped"] is False


def test_glft_symmetric_at_flat_inventory() -> None:
    mid, gamma, sigma, kappa, a = 100.0, 1.0, 0.02, 1000.0, 1.0
    got = glft_quotes(mid, 0.0, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a)
    assert (mid - got["bid"]) == pytest.approx(got["ask"] - mid)


def test_glft_inventory_skew_monotone() -> None:
    mid, gamma, sigma, kappa, a = 100.0, 1.0, 0.02, 1000.0, 1.0
    flat = glft_quotes(mid, 0.0, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a)
    long_ = glft_quotes(mid, 6.0, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a)
    short = glft_quotes(mid, -6.0, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a)
    # Long inventory -> both quotes shift down (eager to sell); short -> up.
    assert long_["ask"] < flat["ask"] < short["ask"]
    assert long_["bid"] < flat["bid"] < short["bid"]


def test_glft_tick_snapping_and_fail_closed() -> None:
    mid, q, gamma, sigma, kappa, a = 100.0, 2.0, 1.0, 0.02, 1000.0, 1.0
    snap = glft_quotes(mid, q, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a, tick=0.01)
    assert snap["snapped"] is True
    assert snap["bid"] < snap["ask"]
    with pytest.raises(ValueError):
        glft_quotes(mid, q, gamma=0.0, sigma=sigma, kappa=kappa, a_fill=a)
    with pytest.raises(ValueError):
        glft_quotes(mid, q, gamma=gamma, sigma=-1.0, kappa=kappa, a_fill=a)
    # Extreme short inventory with big skew drives the bid non-positive -> fail-closed.
    with pytest.raises(ValueError):
        glft_quotes(1.0, 1e6, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a)


# ---------------------------------------------------------------------------
# Diagnostics: autocorrelation, regime detection, phase metrics
# ---------------------------------------------------------------------------


def test_order_flow_autocorrelation_iid_near_zero() -> None:
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=4000)
    rho = order_flow_autocorrelation(signs, lag=1)
    assert abs(rho) < 0.05


def test_order_flow_autocorrelation_sticky_positive() -> None:
    # Persistent runs -> strongly positive lag-1 autocorrelation.
    signs = np.repeat([1.0, -1.0], 200).astype(float)
    assert order_flow_autocorrelation(signs, lag=1) > 0.5


def test_order_flow_autocorrelation_fail_closed() -> None:
    with pytest.raises(ValueError):
        order_flow_autocorrelation([1.0, 1.0, 1.0], lag=1)  # constant
    with pytest.raises(ValueError):
        order_flow_autocorrelation([1.0], lag=1)  # too short
    with pytest.raises(ValueError):
        order_flow_autocorrelation([1.0, -1.0, 1.0, -1.0], lag=0)


def test_regime_flow_diagnostics_separates() -> None:
    rng = np.random.default_rng(1)
    iid = rng.choice([-1.0, 1.0], size=1500)
    d_iid = regime_flow_diagnostics(iid, window=50)
    assert d_iid["regime_detected"] is False
    assert abs(d_iid["sign_autocorr_lag1"]) < 0.10

    sticky = np.repeat([1.0, -1.0], 300).astype(float)
    d_sticky = regime_flow_diagnostics(sticky, window=50)
    assert d_sticky["regime_detected"] is True
    assert d_sticky["sign_autocorr_lag1"] > 0.3
    assert d_sticky["label"] == "SYNTHETIC"


def test_regime_flow_diagnostics_fail_closed() -> None:
    with pytest.raises(ValueError):
        regime_flow_diagnostics([1.0, -1.0], window=50)  # too few
    with pytest.raises(ValueError):
        regime_flow_diagnostics([1.0] * 200, window=1)  # window < 2


def test_book_phase_metrics_orderly_vs_disordered() -> None:
    from quant_fund.microstructure.zi_lob_simulator import BookSample

    tight = [
        BookSample(t=float(i), mid=100.0, spread_ticks=1, bid_depth=8, ask_depth=8)
        for i in range(10)
    ]
    m_tight = book_phase_metrics(tight)
    assert m_tight["phase"] == "orderly_tight"
    assert m_tight["frac_spread_one_tick"] == pytest.approx(1.0)
    assert m_tight["mean_spread_ticks"] == pytest.approx(1.0)
    assert m_tight["label"] == "SYNTHETIC"

    wide = [
        BookSample(t=float(i), mid=100.0, spread_ticks=6, bid_depth=1, ask_depth=1)
        for i in range(10)
    ]
    m_wide = book_phase_metrics(wide)
    assert m_wide["phase"] == "disordered_wide"
    assert m_wide["mean_spread_ticks"] > m_tight["mean_spread_ticks"]

    with pytest.raises(ValueError):
        book_phase_metrics(tight[:1])


# ---------------------------------------------------------------------------
# Square-root impact (emergent; connects to execution.impact)
# ---------------------------------------------------------------------------


def test_sqrt_impact_slope_near_half(impact_result: dict) -> None:
    slope = impact_result["impact_slope"]
    assert impact_result["label"] == "SYNTHETIC"
    assert impact_result["measure"] == "transient"
    # Square-root law: slope ~= 0.5 with tolerance; clean log-log fit.
    assert 0.40 <= slope <= 0.60, f"impact slope {slope} not near 0.5"
    assert impact_result["impact_r2"] >= 0.90
    assert impact_result["n_runs"] == 6 * 2 * 5  # n_seeds × signs × sizes
    # Monotone increasing impact with size.
    imp = impact_result["mean_abs_impact_return"]
    assert all(b > a for a, b in zip(imp, imp[1:], strict=False))


def test_impact_slope_fail_closed() -> None:
    cfg = santa_fe_config(seed=0, band=10)
    with pytest.raises(ValueError):
        metaorder_impact_slope(config=cfg, sizes=(10, 5))  # not increasing
    with pytest.raises(ValueError):
        metaorder_impact_slope(config=cfg, sizes=(1,))  # too few
    with pytest.raises(ValueError):
        metaorder_impact_slope(config=cfg, n_seeds=0)
    with pytest.raises(ValueError):
        metaorder_impact_slope(config=cfg, measure="sideways")


# ---------------------------------------------------------------------------
# Market-making sessions: stationary-bounded vs regime-saturated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ("as", "glft"))
def test_stationary_inventory_bounded(mm_results: dict, label: str) -> None:
    r = mm_results[f"{label}_stat"]
    assert r["label"] == "SYNTHETIC"
    assert r["n_fills"] >= 50  # active session, completed without blowup
    assert r["max_abs_inventory"] <= 12  # bounded well inside the cap
    assert r["max_abs_inventory"] < CAP  # never saturates the cap
    assert math.isfinite(r["sim_internal_mtm_pnl_final"])
    fd = r["flow_diagnostics"]
    assert fd is not None
    assert fd["regime_detected"] is False  # stationary flow: no regime
    assert r["phase_metrics"]["phase"] == "orderly_tight"


@pytest.mark.parametrize("label", ("as", "glft"))
def test_regime_inventory_saturates(mm_results: dict, label: str) -> None:
    stat = mm_results[f"{label}_stat"]
    reg = mm_results[f"{label}_reg"]
    # Stationarily-calibrated policy saturates under persistent directional flow
    # (the Moret & Lillo motivation): inventory >> stationary, near the cap.
    assert reg["max_abs_inventory"] >= 15
    assert reg["max_abs_inventory"] >= 2.5 * max(stat["max_abs_inventory"], 1)
    fd = reg["flow_diagnostics"]
    assert fd is not None
    assert fd["regime_detected"] is True  # diagnostics detect the regime
    assert fd["sign_autocorr_lag1"] > 0.3
    assert fd["flow_bias_abs"] > 0.3


@pytest.mark.parametrize("label", ("as", "glft"))
def test_regime_internal_mtm_worse_than_stationary(mm_results: dict, label: str) -> None:
    """Simulator-internal diagnostic mirroring the paper's negative-PnL finding.

    This is NOT a headline/return claim — it is the synthetic engine's own
    mark-to-market under stress vs calm, namespaced sim_internal_*.
    """
    stat = mm_results[f"{label}_stat"]
    reg = mm_results[f"{label}_reg"]
    assert reg["sim_internal_mtm_pnl_final"] < stat["sim_internal_mtm_pnl_final"]


def test_session_queue_position_accounting(mm_results: dict) -> None:
    r = mm_results["glft_stat"]
    # Active session records per-order queue accounting at fill time.
    assert r["n_fills"] > 0
    assert r["mean_queue_ahead_at_fill"] >= 0.0
    assert r["mean_fill_wait_seconds"] >= 0.0
    assert r["n_mm_cancels"] > 0  # re-quoting cancels outstanding orders
    assert len(r["inventory_path"]) == len(r["inventory_path_times"])


def test_session_honesty_no_forbidden_headline_keys(mm_results: dict) -> None:
    for r in mm_results.values():
        keys = _all_keys(r)
        low = [k.lower() for k in keys]
        # No forbidden headline tokens anywhere.
        for tok in FORBIDDEN_HEADLINE_TOKENS:
            assert not any(tok in k for k in low), f"forbidden token {tok!r} in keys"
        # Any 'pnl' *metric* key is namespaced as a simulator-internal
        # diagnostic; the sole exception is the honesty flag live_pnl_claim
        # (asserted False below), which is a claim gate, not a metric.
        for k in low:
            if "pnl" in k:
                assert k == "live_pnl_claim" or k.startswith("sim_internal_"), (
                    f"pnl metric key not simulator-internal: {k}"
                )
        assert r["research_only"] is True
        assert r["live_pnl_claim"] is False
        assert r["claim"] == "simulator_internal_diagnostic_only"
        assert r["label"] == "SYNTHETIC"
        assert r["data_source"] == "SYNTHETIC_ZI_LOB_v1"


def test_ref_fill_gain_moves_reference_per_fill() -> None:
    """Each fill shifts _ref_ema by exactly ref_fill_gain in its direction."""
    cfg = replace(santa_fe_config(seed=4), anchor="ref", ref_halflife=0.0, ref_fill_gain=0.25)
    sim = ZILobSimulator(cfg)
    sim.run(200.0)
    net = sum(1 if t.aggressor == "buy" else -1 for t in sim.trades)
    assert sim._ref_ema == pytest.approx(0.25 * net)
    assert sim.n_fills > 0


def test_ref_fill_gain_zero_is_bit_identical() -> None:
    a = ZILobSimulator(santa_fe_config(seed=8))
    a.run(150.0)
    b = ZILobSimulator(replace(santa_fe_config(seed=8), ref_fill_gain=0.0))
    b.run(150.0)
    assert [(t.t, t.price, t.level) for t in a.trades] == [
        (t.t, t.price, t.level) for t in b.trades
    ]
    assert a._ref_ema == b._ref_ema


def test_session_fail_closed() -> None:
    cfg = santa_fe_config(seed=1)
    pol = _glft_pol()
    with pytest.raises(ValueError):
        run_mm_session(config=cfg, policy=pol, horizon=0.0)
    with pytest.raises(ValueError):
        run_mm_session(config=cfg, policy=pol, horizon=100.0, decision_interval=0.0)
    with pytest.raises(ValueError):
        run_mm_session(config=cfg, policy=pol, horizon=100.0, inventory_cap=0)
    with pytest.raises(TypeError):
        run_mm_session(config=cfg, policy="not-callable", horizon=100.0)  # type: ignore[arg-type]


def test_repost_frac_zero_is_bit_identical() -> None:
    a = ZILobSimulator(santa_fe_config(seed=11))
    a.run(300.0)
    b = ZILobSimulator(
        replace(santa_fe_config(seed=11), repost_frac=0.0, repost_window=500, repost_band=3)
    )
    b.run(300.0)
    assert [(t.price, t.level, t.aggressor) for t in a.trades] == [
        (t.price, t.level, t.aggressor) for t in b.trades
    ]


def test_repost_reseeds_emptied_levels() -> None:
    sim = ZILobSimulator(replace(santa_fe_config(seed=11), repost_frac=0.8, repost_window=500))
    sim.run(400.0)
    ec = sim.event_counts()
    # n_lo_reposts counts reposted rests, bounded by LO arrivals.
    assert 0 < ec["n_lo_reposts"] <= ec["n_lo_arrivals"]
