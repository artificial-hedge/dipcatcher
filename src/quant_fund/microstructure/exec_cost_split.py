"""exec_cost_split — does correlated order flow change measured exec cost?

Runs an identical TWAP child schedule through the ZI-LOB sim under three
flow drivers — iid, Markov regime, metaorder splitting — and measures
implementation shortfall vs arrival mid. Under split flow, a buy TWAP
runs *into* same-direction metaorders (crowded side), so measured cost
should rise relative to iid flow at identical average intensity — the
Admati–Pfleiderer / order-splitting interaction, measured on the sim.

SYNTHETIC / research-diagnostic only.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

EXEC_SPLIT_SCHEMA = "exec_cost_split.v1"


def twap_schedule(q: int, n_children: int) -> list[int]:
    """Split parent size q into n_children unit lots, residual up front."""
    if q < 1 or n_children < 1:
        raise ValueError(f"q and n_children must be >= 1, got {q}/{n_children}")
    base = q // n_children
    rem = q - base * n_children
    return [base + (1 if i < rem else 0) for i in range(n_children)]


def exec_episode(
    cfg: ZILobConfig,
    flow: MOFlow | None,
    *,
    parent_size: int,
    n_children: int,
    events_per_child: int,
    side: str,
    episode_seed: int,
) -> dict[str, float] | None:
    """Trade one parent by a TWAP schedule through the sim; shortfall in ticks.

    Each episode reseeds the engine's own RNG via ``episode_seed`` — without
    it every same-side episode replays the identical ZI stream and the
    reported ``n_episodes`` count would overstate independent evidence.
    The flow object is shared deliberately so flow-level dynamics stay
    continuous across episodes.
    """
    sim = ZILobSimulator(replace(cfg, seed=episode_seed), flow=flow)
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    # Burn-in so the book reaches a stationary state.
    for _ in range(500):
        sim.step()
    arrival_mid = sim.mid
    if arrival_mid is None:
        return None
    sched = twap_schedule(parent_size, n_children)
    filled = 0
    notional = 0.0
    tick = cfg.tick
    for child in sched:
        trades = sim.inject_market_order(side, child)  # type: ignore[arg-type]
        for tr in trades:
            filled += tr.qty
            notional += tr.price * tr.qty
        # Let the book breathe between children.
        for _ in range(events_per_child):
            sim.step()
    if filled == 0:
        return None
    vwap = notional / filled
    sign = 1.0 if side == "buy" else -1.0
    shortfall_ticks = sign * (vwap - arrival_mid) / tick
    # Price moved from arrival to end, in the trade direction.
    end_mid = sim.mid
    moved = (sign * (end_mid - arrival_mid) / tick) if end_mid is not None else float("nan")
    return {
        "shortfall_ticks": float(shortfall_ticks),
        "fill_fraction": filled / parent_size,
        "mid_drift_ticks": moved,
    }


def exec_split_bench(
    *,
    parent_size: int = 40,
    n_children: int = 8,
    events_per_child: int = 25,
    n_episodes: int = 12,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = ZILobConfig(seed=seed)
    arms: dict[str, MOFlow | None] = {
        "iid": None,
        "regime": MarkovRegimeFlow(
            states=(RegimeState("calm", 1.0, 0.5), RegimeState("trend", 2.2, 0.78)),
            stay_probs=(0.97, 0.94),
            seed=11,
        ),
        "split": SplitFlow(
            p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=13
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for name, flow in arms.items():
        shorts, fills, drifts = [], [], []
        for ep in range(n_episodes):
            side = "buy" if ep % 2 == 0 else "sell"
            out = exec_episode(
                cfg,
                flow,
                parent_size=parent_size,
                n_children=n_children,
                events_per_child=events_per_child,
                side=side,
                episode_seed=seed * 1000 + ep,
            )
            if out is None:
                continue
            shorts.append(out["shortfall_ticks"])
            fills.append(out["fill_fraction"])
            drifts.append(out["mid_drift_ticks"])
        if not shorts:
            raise RuntimeError(f"arm {name}: no completed episodes")
        finite_drifts = [d for d in drifts if math.isfinite(d)]
        results[name] = {
            "n_episodes": len(shorts),
            "shortfall_ticks_mean": float(np.mean(shorts)),
            "shortfall_ticks_sd": float(np.std(shorts)),
            "fill_fraction_mean": float(np.mean(fills)),
            "mid_drift_ticks_mean": float(np.mean(finite_drifts))
            if finite_drifts
            else float("nan"),
        }
    payload: dict[str, Any] = {
        "schema": EXEC_SPLIT_SCHEMA,
        "kind": "exec_cost_split",
        "parent_size": parent_size,
        "n_children": n_children,
        "events_per_child": events_per_child,
        "arms": results,
        "claims": {
            "correlated_flow_raises_cost": bool(
                results["split"]["shortfall_ticks_mean"] > results["iid"]["shortfall_ticks_mean"]
            )
        },
        "interpretation": (
            "Same TWAP schedule, same book parameters, three flow drivers. "
            "If split shortfall exceeds iid, correlated metaorder flow is "
            "inflating measured impact — the TWAP buys while a buy parent "
            "is already walking the book"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
