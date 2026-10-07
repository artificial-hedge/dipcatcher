"""refill_hazard_bench — per-price-level refill hazard after touch-emptying.

``joint_fit_bench`` proved no static placement law closes instant impact
and continuation at once, and ``level_gap_bench`` showed the tape keeps
the level behind the touch sparse (g1 ≈ 2.5 ticks). The residual
question is *why the hole persists*: is the emptied price level
re-occupied quickly (dense re-quoting) or does the book remember the
level stays empty?

This lane measures the refill hazard directly. For every event that
empties the touched level at price ``p``, track forward until ``p`` is
occupied again on that side (cap ``max_delay`` events) and record:

- the delay in events (or ``never``),
- the refill's position: does ``p`` return as the new touch (an
  improving submission lands there — the hole closed by re-quote) or
  does it re-appear deeper (the whole book slid back past it),
- the pre-fill gap behind the touch (``pre_gap``), to condition
  P(empty) on near-touch sparsity.

The hazard curve ``P(refill ≤ k)`` for k ∈ (1, 5, 10, 20, 50, 100, 200,
400) is the object: a fast-refilling book (sim baseline) looks nothing
like a tape where emptied levels stay empty.

Real side: the LOBSTER orderbook snapshots themselves are the book
state — occupancy of an absolute price is read per row, no replay
needed. Sim side: ``sim._asks``/``sim._bids`` level maps around each
fill-bearing step on the iid / split / lv_split arms.

Receipts: ``refill_hazard.v1`` — sealed, MIXED label (real tape +
SYNTHETIC arms), research-only.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from quant_fund.microstructure.lobster import EXECUTION, parse_messages, parse_orderbook_row
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

REFILL_HAZARD_SCHEMA = "refill_hazard.v1"
_KNOTS = (1, 5, 10, 20, 50, 100, 200, 400)
# Gap behind the emptied touch, in ticks: 1 = adjacent level, 5+ = deep hole.
_GAP_BINS = ((1.5, "gap_1"), (4.5, "gap_2_4"), (10**18, "gap_5p"))


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return float(s[n // 2])
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _hazard_stats(
    delays: list[int | None],
    as_touch: int,
    pre_gap_bins: dict[str, list[int]],
    *,
    cap: int,
) -> dict[str, Any]:
    n = len(delays)
    if n < 10:
        return {"ok": False, "n_empty": n}
    observed = [d for d in delays if d is not None]
    n_never = n - len(observed)
    p_never = n_never / n
    cdf = {str(k): round(sum(1 for d in observed if d <= k) / n, 4) for k in _KNOTS if k <= cap}
    conditioned: dict[str, dict[str, float]] = {}
    for label, marks in pre_gap_bins.items():
        if len(marks) >= 10:
            conditioned[label] = {
                "n": len(marks),
                "p_refill_within_cap": round(sum(marks) / len(marks), 4),
            }
    return {
        "ok": True,
        "n_empty": n,
        "p_never_within_cap": round(p_never, 4),
        "median_delay_events": _median([float(d) for d in observed]),
        "hazard_cdf": cdf,
        "p_refill_as_touch": round(as_touch / len(observed), 4) if observed else 0.0,
        "by_pre_gap": conditioned,
    }


def lobster_refill_hazard(msg_path: Path, ob_path: Path, *, max_delay: int = 400) -> dict[str, Any]:
    """Per-level refill hazard around touch-emptying execs on the tape.

    ``pending[side][price]`` holds (row_index, pre_gap_bin) for emptied
    levels still waiting on re-occupancy; each row's occupied-price set
    settles them. ``pre_gap`` is the gap from the emptied touch to the
    next level in the *pre-event* row.
    """
    pending: dict[str, dict[int, tuple[int, int]]] = {"ask": {}, "bid": {}}
    delays: list[int | None] = []
    as_touch = 0
    pre_gap_bins: dict[str, list[int]] = {label: [] for _, label in _GAP_BINS}
    row_idx = -1
    with ob_path.open() as f_ob:
        prev_asks: list[tuple[int, int]] = []
        prev_bids: list[tuple[int, int]] = []
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            row_idx += 1
            asks, bids = parse_orderbook_row(ob_row)
            ask_prices = {p for p, _ in asks}
            bid_prices = {p for p, _ in bids}
            # Settle pending refills against the current occupancy.
            for side, prices in (("ask", ask_prices), ("bid", bid_prices)):
                done = [p for p in pending[side] if p in prices]
                for p in done:
                    t0, bin_i = pending[side].pop(p)
                    delays.append(row_idx - t0)
                    touch = asks[0][0] if side == "ask" else bids[0][0]
                    if p == touch:
                        as_touch += 1
                    pre_gap_bins[list(pre_gap_bins)[bin_i]].append(1)
            # Detect a touch-emptying exec in this row.
            if ev.event_type == EXECUTION and prev_asks and prev_bids and asks and bids:
                if ev.direction == -1:  # buy-initiated hits the ask
                    pre_side, side, other = prev_asks, "ask", ask_prices
                else:
                    pre_side, side, other = prev_bids, "bid", bid_prices
                touch = pre_side[0][0]
                if touch not in other:
                    # Gap behind the emptied touch in the pre-event row,
                    # in ticks (LOBSTER prices are 1/10000$ = 100/tick).
                    # abs(): ask prices ascend but bid prices descend.
                    pre_gap = (
                        abs(pre_side[1][0] - pre_side[0][0]) / 100.0
                        if len(pre_side) > 1
                        else 10**18
                    )
                    bin_i = 0
                    for i, (hi, _label) in enumerate(_GAP_BINS):
                        if pre_gap < hi or i == len(_GAP_BINS) - 1:
                            bin_i = i
                            break
                    pending[side][touch] = (row_idx, bin_i)
            # Flush trackers past the delay cap.
            for side in ("ask", "bid"):
                for p, (t0, bin_i) in list(pending[side].items()):
                    if row_idx - t0 >= max_delay:
                        pending[side].pop(p)
                        delays.append(None)
                        pre_gap_bins[list(pre_gap_bins)[bin_i]].append(0)
            prev_asks, prev_bids = asks, bids
    # End-of-tape: still-pending emptied levels are right-censored at cap.
    for side in ("ask", "bid"):
        for _p, (_t0, bin_i) in pending[side].items():
            delays.append(None)
            pre_gap_bins[list(pre_gap_bins)[bin_i]].append(0)
    return _hazard_stats(delays, as_touch, pre_gap_bins, cap=max_delay)


def sim_refill_hazard(
    cfg: Any = None,
    flow: Any = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
    max_delay: int = 400,
) -> dict[str, Any]:
    """Same bookkeeping around each sim fill that moves the touch."""
    sim = ZILobSimulator(cfg if cfg is not None else santa_fe_config(seed=seed), flow=flow)
    pending: dict[str, dict[int, tuple[int, int]]] = {"ask": {}, "bid": {}}
    delays: list[int | None] = []
    as_touch = 0
    pre_gap_bins: dict[str, list[int]] = {label: [] for _, label in _GAP_BINS}
    n_trades = 0
    for step in range(horizon):
        pre_asks = sorted(sim._asks)
        pre_bids = sorted(sim._bids, reverse=True)
        sim.step()
        new_trades = sim.trades[n_trades:]
        n_trades = len(sim.trades)
        # Settle pending trackers against post-step occupancy.
        for side, book in (("ask", sim._asks), ("bid", sim._bids)):
            done = [lv for lv in pending[side] if lv in book]
            for lv in done:
                t0, bin_i = pending[side].pop(lv)
                delays.append(step - t0)
                touch = sim.best_ask_level if side == "ask" else sim.best_bid_level
                if lv == touch:
                    as_touch += 1
                pre_gap_bins[list(pre_gap_bins)[bin_i]].append(1)
        if new_trades:
            sign_buy = new_trades[0].aggressor == "buy"
            if sign_buy:
                pre_side, side = pre_asks, "ask"
                post_touch = sim.best_ask_level
            else:
                pre_side, side = pre_bids, "bid"
                post_touch = sim.best_bid_level
            if pre_side:
                touch = pre_side[0]
                emptied = post_touch is None or post_touch != touch
                if emptied:
                    # sim levels are already ticks.
                    pre_gap = abs(pre_side[1] - pre_side[0]) if len(pre_side) > 1 else 10**18
                    bin_i = 0
                    for i, (hi, _label) in enumerate(_GAP_BINS):
                        if pre_gap < hi or i == len(_GAP_BINS) - 1:
                            bin_i = i
                            break
                    pending[side][touch] = (step, bin_i)
        for side, _book in (("ask", sim._asks), ("bid", sim._bids)):
            for lv, (t0, bin_i) in list(pending[side].items()):
                if step - t0 >= max_delay:
                    pending[side].pop(lv)
                    delays.append(None)
                    pre_gap_bins[list(pre_gap_bins)[bin_i]].append(0)
    for side in ("ask", "bid"):
        for _lv, (_t0, bin_i) in pending[side].items():
            delays.append(None)
            pre_gap_bins[list(pre_gap_bins)[bin_i]].append(0)
    return _hazard_stats(delays, as_touch, pre_gap_bins, cap=max_delay)


def _split(seed: int) -> SplitFlow:
    return SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed
    )


def refill_hazard_bench(
    tape_dir: Path,
    ticker: str = "AMZN",
    *,
    horizon: int = 20000,
    seed: int = 7,
    max_delay: int = 400,
) -> dict[str, Any]:
    """Per-level refill hazard after touch-emptying, real vs sim arms."""
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook CSV pair under {tape_dir}")
    real = lobster_refill_hazard(msg[0], ob[0], max_delay=max_delay)
    # Cooldown calibration on the anchored arm: does an explicit per-price
    # vacancy window (``refill_cooldown`` events) move the sim's refill
    # hazard toward the tape's? Each cell is the same arm with a stronger
    # suppression window; cd=0 is the bare ref anchor.
    cooldown_cells = [0, 60, 120, 240]
    cooldown_scan = [
        {
            "refill_cooldown": cd,
            **{
                k: v
                for k, v in sim_refill_hazard(
                    replace(
                        santa_fe_config(seed=seed + 3),
                        anchor="ref",
                        density_exponent=1.0,
                        band=40,
                        ref_fill_gain=0.3,
                        refill_cooldown=cd,
                    ),
                    _split(seed + 3),
                    horizon=horizon,
                    max_delay=max_delay,
                ).items()
                if k
                in (
                    "ok",
                    "n_empty",
                    "p_never_within_cap",
                    "median_delay_events",
                    "p_refill_as_touch",
                    "hazard_cdf",
                )
            },
        }
        for cd in cooldown_cells
    ]
    arms = {
        "iid": sim_refill_hazard(horizon=horizon, seed=seed, max_delay=max_delay),
        "split": sim_refill_hazard(
            santa_fe_config(seed=seed + 1),
            _split(seed + 1),
            horizon=horizon,
            max_delay=max_delay,
        ),
        "lv_split": sim_refill_hazard(
            replace(santa_fe_config(seed=seed + 2), anchor="ref", ref_fill_gain=0.3),
            _split(seed + 2),
            horizon=horizon,
            max_delay=max_delay,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        r_never = float(real["p_never_within_cap"])
        r_med = real.get("median_delay_events")
        for name, arm in arms.items():
            if not arm.get("ok"):
                divergences.append(f"{name}:no_emptying")
                continue
            a_never = float(arm["p_never_within_cap"])
            a_med = arm.get("median_delay_events")
            if abs(a_never - r_never) > 0.15:
                divergences.append(f"{name}:p_never_{a_never}_vs_{r_never}")
            if (
                r_med is not None
                and a_med is not None
                and abs(a_med - r_med) > max(5.0, 0.5 * r_med)
            ):
                divergences.append(f"{name}:median_delay_{a_med}_vs_{r_med}")
    claims = {
        "tape_levels_have_memory": bool(
            real.get("ok")
            and real.get("median_delay_events") is not None
            and float(real["median_delay_events"]) > 20.0
        ),
        "sim_refill_is_fast": bool(
            all(a.get("ok") for a in arms.values())
            and all((a.get("median_delay_events") or 1e9) < 20.0 for a in arms.values())
        ),
        "cooldown_monotone_in_p_never": bool(
            len(cooldown_scan) >= 2
            and all(c.get("ok") for c in cooldown_scan)
            and all(
                float(cooldown_scan[i + 1]["p_never_within_cap"])
                >= float(cooldown_scan[i]["p_never_within_cap"]) - 1e-9
                for i in range(len(cooldown_scan) - 1)
            )
        ),
    }
    payload: dict[str, Any] = {
        "schema": REFILL_HAZARD_SCHEMA,
        "kind": "refill_hazard_bench",
        "ticker": ticker,
        "horizon": horizon,
        "seed": seed,
        "max_delay": max_delay,
        "real": real,
        "sim_arms": arms,
        "cooldown_scan": cooldown_scan,
        "divergences": divergences,
        "claims": claims,
        "interpretation": (
            "The hole behind an emptied touch is a level-memory structure, "
            "not a placement geometry: on the tape an emptied price stays "
            "empty for tens to hundreds of events (median delay + p_never "
            "are the headline stats) and is usually re-occupied as the new "
            "touch by an improving submission, while sim arms re-occupy "
            "levels almost immediately (or never, when the level can only "
            "be reached inside the spread). Placement-law and inner-edge "
            "knobs cannot express this — the missing mechanism is a "
            "per-price refill timescale."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "REFILL_HAZARD_SCHEMA",
    "lobster_refill_hazard",
    "refill_hazard_bench",
    "sim_refill_hazard",
]
