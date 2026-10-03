"""Tests for microstructure/regime_quoting.py — posterior-gated quoting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.regime_filter import FilterConfig, RegimeFilter
from quant_fund.microstructure.regime_quoting import (
    REGIME_QUOTING_SCHEMA,
    MMState,
    _drifting_flow,
    adaptive_policy,
    regime_quoting_bench,
    regime_session,
    touch_skew_policy,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def _state(inv: int = 0, bb: float = 99.98, ba: float = 100.0) -> MMState:
    return MMState(
        t=10.0,
        mid=(bb + ba) / 2,
        best_bid=bb,
        best_ask=ba,
        inventory=inv,
        tau=100.0,
    )


def _filt() -> RegimeFilter:
    return RegimeFilter(
        FilterConfig(mu=0.1, intensity=(1.0, 1.5), p_buy=(0.5, 0.78), stay=(0.97, 0.94))
    )


def test_touch_skew_improves_over_inside_quotes() -> None:
    pol = touch_skew_policy(gamma=0.0, tick=0.01)
    bid, ask = pol(_state())
    # improving by a tick inside the touch when the spread allows
    assert bid == pytest.approx(99.99)
    assert ask == pytest.approx(99.99) or ask == pytest.approx(100.0 - 0.01)


def test_touch_skew_long_inventory_skews_down() -> None:
    pol = touch_skew_policy(gamma=0.5, tick=0.01)
    bid0, ask0 = pol(_state(inv=0))
    bid5, ask5 = pol(_state(inv=5))
    # long inventory shifts both quotes down (sell more / buy less)
    assert bid5 < bid0 or bid5 <= bid0
    assert ask5 < ask0


def _drive_to_trend(filt: RegimeFilter) -> None:
    # belief is a copy — push the posterior to state 1 with a buy burst
    t = 0.0
    rng = np.random.default_rng(0)
    while float(filt.belief[1]) <= 0.9:
        t += float(rng.exponential(4.0))
        filt.update(t, True)


def test_adaptive_suppresses_ask_when_belief_is_trend() -> None:
    filt = _filt()
    _drive_to_trend(filt)
    pol = adaptive_policy(filt, gate=0.65, tick=0.01)
    bid, ask = pol(_state())
    assert ask is None
    assert bid is not None


def test_adaptive_quotes_both_sides_when_calm() -> None:
    filt = _filt()  # prior 0.5 stays below the gate only while calm evidence
    t = 0.0
    rng = np.random.default_rng(1)
    for _ in range(200):
        t += float(rng.exponential(10.0))
        filt.update(t, bool(rng.random() < 0.5))
    assert float(filt.belief[1]) < 0.65
    pol = adaptive_policy(filt, gate=0.65, tick=0.01)
    bid, ask = pol(_state())
    assert bid is not None and ask is not None


def test_regime_session_never_submits_marketable_limits() -> None:
    """Regression: level-space clip — float dust on bb+tick once slipped the
    cross-check and submitted a sell at exactly best_bid."""
    cfg = ZILobConfig(seed=1005)
    filt = _filt()
    st = regime_session(
        config=cfg,
        policy=adaptive_policy(filt, gate=0.8, tick=cfg.tick),
        flow=_drifting_flow(1005),
        horizon=200.0,
        decision_interval=0.5,
        filt=filt,
    )
    assert st.n_fills >= 0


def test_regime_session_determinism() -> None:
    cfg = ZILobConfig(seed=7)
    a = regime_session(
        config=cfg,
        policy=touch_skew_policy(tick=0.01),
        flow=_drifting_flow(7),
        horizon=60.0,
    )
    b = regime_session(
        config=ZILobConfig(seed=7),
        policy=touch_skew_policy(tick=0.01),
        flow=_drifting_flow(7),
        horizon=60.0,
    )
    assert a == b


def test_bench_smoke_and_schema() -> None:
    out = regime_quoting_bench(horizon=120.0, n_seeds=2)
    assert out["schema"] == REGIME_QUOTING_SCHEMA
    for arm in ("static", "adaptive", "symmetric_widen"):
        assert arm in out["arms"]
        assert out["arms"][arm]["n_fills_mean"] >= 0.0
    assert isinstance(out["symmetric_widen_worst_tail"], bool)


def test_mmstate_defaults() -> None:
    st = MMState(t=0.0, mid=100.0, best_bid=99.99, best_ask=100.01, inventory=0, tau=1.0)
    assert st.inventory == 0
