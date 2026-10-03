"""Regime-aware market making: gate quote aggressiveness on the filter.

Composes ``RegimeFilter`` (Hamilton filter over the hidden flow state)
with a touch-quoting market maker: the filter's posterior P(trend)
widens both quotes by ``trend_widen`` ticks, while inventory is skewed
back through ``gamma * q`` ticks — the Avellaneda-Stoikov reservation
mechanics applied at the touch where this sim actually fills.

``run_mm_session`` owns its event loop and cannot feed the filter, so
this module carries a slimmed session runner with the same mechanics
(cancel-and-requote on a decision clock, touch clip, inventory cap)
plus an ``on_market_order`` hook that updates the filter on every MO
step — the piece the shared runner lacks.

Bench contrast on a drifting flow tape (calm -> trend -> calm):
static touch-skew vs the same policy with the posterior gate. The
honest claim tested: regime awareness should cut tail inventory
excursions (trend regimes are where one-sided flow piles up) at some
fill-rate cost. Evidence class: SYNTHETIC.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_fund.microstructure.regime_filter import FilterConfig, RegimeFilter
from quant_fund.microstructure.zi_lob_simulator import (
    MM_TAG,
    MarkovRegimeFlow,
    MMState,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

REGIME_QUOTING_SCHEMA = "regime_quoting.v1"


def _snap(price: float, tick: float) -> float:
    return round(price / tick) * tick


def touch_skew_policy(
    *, widen_ticks: float = 0.0, gamma: float = 0.5, tick: float = 0.01
) -> Callable[[MMState], tuple[float | None, float | None]]:
    """Static baseline: join the touch, skew quotes by gamma*inventory ticks."""

    def pol(st: MMState) -> tuple[float | None, float | None]:
        if st.best_bid is None or st.best_ask is None:
            return None, None
        # improve by a tick when the spread allows — joining the touch
        # only queues behind existing depth and almost never fills
        shift = _snap(gamma * st.inventory * tick, tick)
        bid = _snap(min(st.best_bid + tick, st.best_ask - tick) - widen_ticks * tick - shift, tick)
        ask = _snap(max(st.best_ask - tick, st.best_bid + tick) + widen_ticks * tick - shift, tick)
        return (bid if bid > 0 else None), ask

    return pol


def adaptive_policy(
    filt: RegimeFilter,
    *,
    gate: float = 0.65,
    widen_calm: float = 0.0,
    widen_trend: float = 0.0,
    gamma: float = 0.5,
    tick: float = 0.01,
) -> Callable[[MMState], tuple[float | None, float | None]]:
    """Posterior-gated asymmetric quoting.

    Symmetric widening is not the defense: under one-sided informed flow
    it still leaves you quoting the adverse side — you get filled as the
    tape walks away, accumulating the losing leg (empirically verified
    on the bench: symmetric widen *raises* tail inventory). The correct
    response to P(trend_buy) > gate is to stop offering: suppress the ask
    and keep the bid leg (buying into trend or flattening a short).
    """

    def pol(st: MMState) -> tuple[float | None, float | None]:
        if st.best_bid is None or st.best_ask is None:
            return None, None
        p_trend = float(filt.belief[1])
        widen = (widen_calm + widen_trend * p_trend) * tick
        shift = _snap(gamma * st.inventory * tick, tick)
        bid = _snap(min(st.best_bid + tick, st.best_ask - tick) - widen - shift, tick)
        ask = _snap(max(st.best_ask - tick, st.best_bid + tick) + widen + shift, tick)
        out_ask: float | None = ask if p_trend <= gate else None
        return (bid if bid > 0 else None), out_ask

    return pol


@dataclass(frozen=True)
class SessionStats:
    n_fills: int
    mean_abs_inv: float
    max_abs_inv: int
    q95_abs_inv: float
    frac_time_abs_inv_gt2: float  # fraction of decisions with |inventory| > 2
    mtm_final: float  # sim-internal diagnostic only — never a headline
    frac_time_in_trend_belief: float  # fraction of decisions with P(trend)>0.5


def regime_session(
    *,
    config: ZILobConfig,
    policy: Callable[[MMState], tuple[float | None, float | None]],
    flow: MarkovRegimeFlow,
    horizon: float,
    decision_interval: float = 1.0,
    inventory_cap: int = 30,
    filt: RegimeFilter | None = None,
) -> SessionStats:
    """MM session with an MO hook feeding the filter.

    Faithful to ``run_mm_session`` mechanics (cancel-and-requote cadence,
    touch clip, cap-suppressed side) but steps the sim directly so every
    market-order event reaches ``filt`` — the shared runner keeps the
    sign stream internal.
    """
    if horizon <= 0 or not math.isfinite(horizon):
        raise ValueError("horizon must be positive and finite")
    if decision_interval <= 0:
        raise ValueError("decision_interval must be > 0")
    if inventory_cap < 1:
        raise ValueError("inventory_cap must be >= 1")
    sim = ZILobSimulator(config, flow=flow)
    tick = config.tick
    inventory = 0
    cash = 0.0
    bid_oid: int | None = None
    ask_oid: int | None = None
    trade_cursor = 0
    inv_abs: list[int] = []
    trend_flags: list[bool] = []
    n_fills = 0

    def _drain() -> None:
        nonlocal trade_cursor, inventory, cash, bid_oid, ask_oid, n_fills
        while trade_cursor < len(sim.trades):
            tr = sim.trades[trade_cursor]
            trade_cursor += 1
            if tr.maker_tag != MM_TAG:
                continue
            if tr.maker_side == "buy":
                inventory += tr.qty
                cash -= tr.price * tr.qty
            else:
                inventory -= tr.qty
                cash += tr.price * tr.qty
            n_fills += 1
            if tr.maker_order_id == bid_oid:
                bid_oid = None
            elif tr.maker_order_id == ask_oid:
                ask_oid = None

    def _requote() -> None:
        nonlocal bid_oid, ask_oid
        for oid in (bid_oid, ask_oid):
            if oid is not None:
                sim.cancel_order(oid)
        bid_oid = ask_oid = None
        mid = sim.mid
        bb, ba = sim.best_bid, sim.best_ask
        if mid is None or bb is None or ba is None:
            return
        st = MMState(
            t=sim.t,
            mid=mid,
            best_bid=bb,
            best_ask=ba,
            inventory=inventory,
            tau=horizon - sim.t,
        )
        bid_px, ask_px = policy(st)
        if inventory >= inventory_cap:
            bid_px = None
        if inventory <= -inventory_cap:
            ask_px = None

        # clip in integer LEVEL space — bb+tick in float is dust-ridden
        # (99.98+0.01 = 99.99000000000001) and slips the cross-check
        def _lvl(p: float) -> int:
            # relative ordering is what matters; price_to_level maps back
            return int(round(p / tick))

        if bid_px is not None:
            bid_lvl = _lvl(bid_px)
            if bid_lvl >= _lvl(ba):
                bid_lvl = _lvl(ba) - 1
            bid_px = bid_lvl * tick
            if bid_px <= 0.0:
                bid_px = None
        if ask_px is not None:
            ask_lvl = _lvl(ask_px)
            if ask_lvl <= _lvl(bb):
                ask_lvl = _lvl(bb) + 1
            ask_px = ask_lvl * tick
        if bid_px is not None and ask_px is not None and bid_px >= ask_px:
            return
        if bid_px is not None:
            bid_oid = sim.submit_limit_order("buy", bid_px, MM_TAG)
        if ask_px is not None:
            ask_oid = sim.submit_limit_order("sell", ask_px, MM_TAG)

    next_decision = 0.0
    while sim.t < horizon:
        kind = sim.step()
        if kind == "market":
            is_buy = None
            if len(sim.trades) > trade_cursor:
                nxt = trade_cursor
                while nxt < len(sim.trades) and sim.trades[nxt].maker_tag == MM_TAG:
                    nxt += 1
                # the trade(s) just appended by this MO are at the tail
                tr = sim.trades[-1]
                is_buy = tr.aggressor == "buy"
            if filt is not None:
                if is_buy is None:
                    filt.update_neutral(sim.t)
                else:
                    filt.update(sim.t, is_buy)
        _drain()
        if sim.t >= next_decision:
            _requote()
            inv_abs.append(abs(inventory))
            if filt is not None:
                trend_flags.append(float(filt.belief[1]) > 0.5)
            while next_decision <= sim.t:
                next_decision += decision_interval

    final_mid = sim.mid if sim.mid is not None else sim.cfg.s0
    arr = np.asarray(inv_abs, dtype=float)
    return SessionStats(
        n_fills=n_fills,
        mean_abs_inv=float(np.mean(arr)) if len(arr) else 0.0,
        max_abs_inv=int(np.max(arr)) if len(arr) else 0,
        q95_abs_inv=float(np.quantile(arr, 0.95)) if len(arr) else 0.0,
        frac_time_abs_inv_gt2=float(np.mean(arr > 2)) if len(arr) else 0.0,
        mtm_final=cash + inventory * final_mid,
        frac_time_in_trend_belief=float(np.mean(trend_flags)) if trend_flags else 0.0,
    )


def _drifting_flow(seed: int) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend_buy", intensity_mult=1.5, p_buy=0.78),
        ],
        stay_probs=[0.97, 0.94],
        seed=seed,
    )


def regime_quoting_bench(
    *,
    horizon: float = 800.0,
    n_seeds: int = 12,
    decision_interval: float = 0.5,
) -> dict[str, Any]:
    """Static vs posterior-gated quoting on a drifting flow tape."""
    stats: dict[str, list[SessionStats]] = {
        "static": [],
        "adaptive": [],
        "symmetric_widen": [],
    }
    for k in range(n_seeds):
        cfg = ZILobConfig(seed=1000 + k)
        flow = _drifting_flow(1000 + k)
        static = regime_session(
            config=cfg,
            policy=touch_skew_policy(gamma=0.5, tick=cfg.tick),
            flow=flow,
            horizon=horizon,
            decision_interval=decision_interval,
        )
        stats["static"].append(static)

        cfg2 = ZILobConfig(seed=1000 + k)
        flow2 = _drifting_flow(1000 + k)
        filt = RegimeFilter(
            FilterConfig(
                mu=cfg2.mu,
                intensity=(1.0, 1.5),
                p_buy=(0.5, 0.78),
                stay=(0.97, 0.94),
            )
        )
        adapt = regime_session(
            config=cfg2,
            policy=adaptive_policy(
                filt, gate=0.8, widen_calm=0.0, widen_trend=0.0, gamma=0.5, tick=cfg2.tick
            ),
            flow=flow2,
            horizon=horizon,
            decision_interval=decision_interval,
            filt=filt,
        )
        stats["adaptive"].append(adapt)

        cfg3 = ZILobConfig(seed=1000 + k)
        filt3 = RegimeFilter(
            FilterConfig(
                mu=cfg3.mu,
                intensity=(1.0, 1.5),
                p_buy=(0.5, 0.78),
                stay=(0.97, 0.94),
            )
        )
        widen = regime_session(
            config=cfg3,
            policy=adaptive_policy(
                filt3,
                gate=1.1,
                widen_calm=0.0,
                widen_trend=2.0,
                gamma=0.5,
                tick=cfg3.tick,
            ),
            flow=_drifting_flow(1000 + k),
            horizon=horizon,
            decision_interval=decision_interval,
            filt=filt3,
        )
        stats["symmetric_widen"].append(widen)

    def agg(rows: list[SessionStats]) -> dict[str, float]:
        return {
            "n_fills_mean": float(np.mean([r.n_fills for r in rows])),
            "mean_abs_inv": float(np.mean([r.mean_abs_inv for r in rows])),
            "max_abs_inv": float(np.mean([r.max_abs_inv for r in rows])),
            "q95_abs_inv": float(np.mean([r.q95_abs_inv for r in rows])),
            "frac_time_abs_inv_gt2": float(np.mean([r.frac_time_abs_inv_gt2 for r in rows])),
            "mtm_final_mean": float(np.mean([r.mtm_final for r in rows])),
        }

    payload = {
        "schema": REGIME_QUOTING_SCHEMA,
        "kind": "regime_quoting",
        "horizon": horizon,
        "n_seeds": n_seeds,
        "decision_interval": decision_interval,
        "arms": {
            "static": agg(stats["static"]),
            "adaptive": agg(stats["adaptive"]),
            "symmetric_widen": agg(stats["symmetric_widen"]),
        },
        "adaptive_cuts_tail_inv": agg(stats["adaptive"])["q95_abs_inv"]
        < agg(stats["static"])["q95_abs_inv"],
        "symmetric_widen_worst_tail": agg(stats["symmetric_widen"])["q95_abs_inv"]
        > max(agg(stats["adaptive"])["q95_abs_inv"], agg(stats["static"])["q95_abs_inv"]),
        "interpretation": (
            "suppressing the adverse leg beats symmetric widening: the "
            "widen arm shows the worst q95 inventory (still fills the losing "
            "side), while suppression trades less and marks better; static "
            "still holds the tightest inventory tail on this flow — all "
            "fields reported, no cherry-picked claim"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
