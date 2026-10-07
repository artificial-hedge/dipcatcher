"""Inventory-excursion geometry for MM policies on the ZI-LOB.

An MM policy's inventory path ``q_t`` is a stochastic walk steered by
the skew; its risk lives in the excursions: how deep it wanders and how
long it takes to come back. This lane measures that geometry:

- ``excursions(q)`` — decompose the path into excursions: maximal runs
  between consecutive times the |q| process touches/leaves bands
  (segment between successive sign-level crossings of |q| >= level).
- ``first_passage(q, level)`` — first t with |q_t| >= level (NaN if
  never); plus count of upcrossings.
- ``max_drawdown_excursion(q)`` — the deepest single excursion (max |q|
  within one excursion episode) and its duration.
- ``excursion_geometry_bench`` — sealed ``excursion_geometry.v1``:
  compares policies on excursion depth/duration/first-passage rates
  under identical CRN seeds (same arrival streams via
  ``MarkovRegimeFlow(seed=k)``), so differences are policy-attributable.
  Headline metrics are geometric (ticks of inventory, seconds), not
  P&L — honest under the proper-score contract.

SYNTHETIC only.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MMState,
    QuotePolicy,
    RegimeState,
    ZILobConfig,
    run_mm_session,
)
from quant_fund.utils.hashing import hash_bytes, sanitize_for_json
from quant_fund.utils.reproducibility import git_revision

EXCURSION_GEO_SCHEMA = "excursion_geometry.v1"


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


DEFAULT_TWO_STATE_FLOW = (
    RegimeState("calm", intensity_mult=1.0, p_buy=0.5),
    RegimeState("stress", intensity_mult=2.2, p_buy=0.55),
)
DEFAULT_STAY_PROBS = (0.985, 0.95)


def _check_path(q: NDArray[np.floating[Any]], times: NDArray[np.floating[Any]]) -> None:
    if q.ndim != 1 or times.ndim != 1 or q.size != times.size or q.size < 2:
        raise ValueError("q and times must be equal-length 1-D arrays with >= 2 points")
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(times)):
        raise ValueError("q/times must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")


@dataclass(frozen=True)
class Excursion:
    """One episode of |q| staying at or above ``level``."""

    start_i: int
    end_i: int
    peak_abs: float
    duration: float
    side: int  # sign of q at the peak


def excursions(
    q: NDArray[np.floating[Any]],
    times: NDArray[np.floating[Any]],
    level: float,
) -> list[Excursion]:
    """Episodes where |q_t| >= level, delimited by drops below it."""
    _check_path(q, times)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError(f"level must be positive and finite, got {level!r}")
    above = np.abs(q) >= level
    out: list[Excursion] = []
    i = 0
    n = len(q)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        seg = np.abs(q[i : j + 1])
        peak = float(seg.max())
        side = 1 if float(q[i + int(seg.argmax())]) >= 0.0 else -1
        out.append(
            Excursion(
                start_i=i,
                end_i=j,
                peak_abs=peak,
                duration=float(times[j] - times[i]),
                side=side,
            )
        )
        i = j + 1
    return out


def first_passage(
    q: NDArray[np.floating[Any]],
    times: NDArray[np.floating[Any]],
    level: float,
) -> float:
    """First time |q_t| >= level; NaN if never reached."""
    _check_path(q, times)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError(f"level must be positive and finite, got {level!r}")
    hit = np.nonzero(np.abs(q) >= level)[0]
    return float(times[hit[0]]) if hit.size else float("nan")


def excursion_stats(
    q: NDArray[np.floating[Any]],
    times: NDArray[np.floating[Any]],
    level: float,
) -> dict[str, float]:
    """Aggregate excursion geometry at a level."""
    _check_path(q, times)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError(f"level must be positive and finite, got {level!r}")
    eps = excursions(q, times, level)
    peaks = np.asarray([e.peak_abs for e in eps])
    durs = np.asarray([e.duration for e in eps])
    horizon = float(times[-1] - times[0])
    frac_above = float(np.mean(np.abs(q) >= level))
    return {
        "level": float(level),
        "n_excursions": float(len(eps)),
        "excursion_rate": len(eps) / horizon if horizon > 0 else float("nan"),
        "mean_peak_abs": float(peaks.mean()) if eps else float("nan"),
        "max_peak_abs": float(peaks.max()) if eps else 0.0,
        "mean_duration": float(durs.mean()) if eps else float("nan"),
        "frac_time_above": frac_above,
        "first_passage_t": first_passage(q, times, level),
    }


def _touch_skew_policy(*, gamma: float, sigma: float, tick: float) -> QuotePolicy:
    """Touch-quoting policy with AS reservation skew.

    Bids/asks are posted at the prevailing touch shifted by the
    inventory skew ``gamma * sigma**2 * q`` (the AS reservation-price
    displacement), tick-rounded. Both quotes shift together — it is a
    reservation-price shift, not a spread change. Higher gamma pulls the
    whole quote pair harder away from accumulated inventory, producing
    shallower excursions.
    """

    def policy(state: MMState) -> tuple[float | None, float | None]:
        if state.best_bid is None or state.best_ask is None:
            return (None, None)
        shift = round(gamma * sigma * sigma * state.inventory / tick) * tick
        # improve by a tick when the spread allows; else join the touch
        bid = min(state.best_bid + tick, state.best_ask - tick)
        ask = max(state.best_ask - tick, state.best_bid + tick)
        bid -= shift
        ask -= shift
        # snap to the tick grid: float drift (e.g. ba-tick landing an
        # ulp above bb) can slip past the clip's 1e-12 band and submit a
        # marketable order
        bid = round(bid / tick) * tick
        ask = round(ask / tick) * tick
        out_bid: float | None = bid if bid > 0.0 else None
        return (out_bid, ask)

    return policy


def _policy_geometry(
    config: ZILobConfig,
    policy: Any,
    *,
    horizon: float,
    inventory_cap: int,
    flow_states: tuple[RegimeState, ...],
    stay_probs: tuple[float, ...],
    seed: int,
    levels: tuple[int, ...],
) -> dict[str, Any]:
    flow = MarkovRegimeFlow(flow_states, stay_probs, seed=seed)
    cfg = dataclasses.replace(config, seed=seed)
    bundle = run_mm_session(
        config=cfg,
        policy=policy,
        horizon=horizon,
        decision_interval=1.0,
        flow=flow,
        inventory_cap=inventory_cap,
        sample_interval=1.0,
    )
    q = np.asarray(bundle["inventory_path"], dtype=np.float64)
    t = np.asarray(bundle["inventory_path_times"], dtype=np.float64)
    return {
        "levels": {str(lv): excursion_stats(q, t, float(lv)) for lv in levels},
        "max_abs_inventory": float(np.abs(q).max()),
        "mean_abs_inventory": float(np.abs(q).mean()),
        "cap_hits": float(np.sum(np.abs(q) >= inventory_cap)),
    }


def excursion_geometry_bench(
    *,
    horizon: float = 1500.0,
    inventory_cap: int = 10,
    levels: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> dict[str, Any]:
    """Sealed ``excursion_geometry.v1`` receipt.

    One fixed touch-skew policy runs the same ZI-LOB under two CRN flow
    regimes — symmetric and buy-biased stress — so the comparison
    isolates flow geometry, not policy differences: directional MO
    pressure stacks same-side fills into deeper |q| excursions.
    """
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    _pos_finite(horizon, "horizon")
    _pos_finite(inventory_cap, "inventory_cap")
    if not levels:
        raise ValueError("levels must be non-empty")
    for lv in levels:
        if lv < 1 or lv > inventory_cap:
            raise ValueError(f"level {lv} outside (0, inventory_cap]")
    base = santa_fe_config(seed=seed)
    policy = _touch_skew_policy(gamma=0.05, sigma=0.2, tick=base.tick)
    regimes = {
        "symmetric": (
            RegimeState("calm", 1.0, 0.5),
            RegimeState("stress", 3.0, 0.5),
        ),
        # persistent directional tape: the bias is on most of the time,
        # not only inside rare stress bursts
        "buy_biased": (
            RegimeState("calm", 1.0, 0.62),
            RegimeState("stress", 3.0, 0.78),
        ),
    }
    per_regime: dict[str, Any] = {}
    for name, states in regimes.items():
        per_regime[name] = _policy_geometry(
            base,
            policy,
            horizon=horizon,
            inventory_cap=inventory_cap,
            flow_states=states,
            stay_probs=(0.99, 0.96),
            seed=seed,
            levels=levels,
        )
    sym = per_regime["symmetric"]
    bia = per_regime["buy_biased"]
    payload: dict[str, Any] = {
        "schema": EXCURSION_GEO_SCHEMA,
        "kind": "excursion_geometry",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Inventory-excursion geometry on the synthetic ZI-LOB under "
            "CRN-paired flow regimes; never market evidence."
        ),
        "horizon": float(horizon),
        "inventory_cap": float(inventory_cap),
        "levels": [float(lv) for lv in levels],
        "per_regime": per_regime,
        # the honest contrast: biased tape lifts mean|q| and holds the
        # path above level longer (fewer, fatter excursions)
        "mean_abs_inventory_ratio": float(
            bia["mean_abs_inventory"] / max(sym["mean_abs_inventory"], 1e-12)
        ),
        "frac_time_above_ratio_level1": float(
            bia["levels"]["1"]["frac_time_above"]
            / max(sym["levels"]["1"]["frac_time_above"], 1e-12)
        )
        if "1" in bia["levels"] and "1" in sym["levels"]
        else float("nan"),
    }
    # Non-finite stats (quiet paths with no excursions, unhit levels)
    # seal as strict-JSON null, never literal NaN.
    payload = sanitize_for_json(payload)
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "DEFAULT_STAY_PROBS",
    "DEFAULT_TWO_STATE_FLOW",
    "EXCURSION_GEO_SCHEMA",
    "Excursion",
    "excursion_geometry_bench",
    "excursion_stats",
    "excursions",
    "first_passage",
]
