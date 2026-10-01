"""queue_fate — entry position determines outcome.

The queue-priority dogma: a resting order's fate (fill vs delete vs
still-open) is decided by where it entered the queue. `order_lifetime`
(#493) measured unconditional survival; `depth_consumption` (#507)
used `maker_queue_ahead_at_submit` but only on *filled* orders — a
survivorship lens. This lane conditions the full outcome distribution
on the entry position.

Entry classes (vs the same-side touch in the pre-event book row):

- ``improve``: priced strictly better than the touch → new queue front.
  On the real tape this is the inside-spread submit class (~10% of
  LOs, see marketable_limit.v1); in the sim it is a submit that lands
  ahead of the current best because the reference moved.
- ``join``: priced exactly at the touch → back of an existing queue.
  The displayed level size at submit is the exact queue ahead, in
  shares — the deepest position observable without level-3 data.
- ``behind``: priced worse than the touch → a deeper level queue.

Outcomes: ``filled`` (any EXECUTION event against the id; hidden-order
executions whose ids never appear as submissions are skipped),
``deleted`` (DELETE event, or remaining size cancelled to zero by
CANCEL_PARTIAL), ``open`` (still resting at tape end / sim horizon).
``filled`` dominates ``deleted`` when both occurred.

Sim arms classify identically using the pre-step best levels plus the
order's own ``queue_ahead`` count (sim sizes are unit lots, so queue
ahead is an order count, not shares).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    SUBMISSION,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CATEGORIES = ("improve", "join", "behind")
JOIN_AHEAD_BINS = (0, 100, 500, 2000)
JOIN_AHEAD_LABELS = ("0_99", "100_499", "500_1999", "2000_plus")


def _classify_submit(
    price: float, direction: int, top: tuple[float, float, float, float]
) -> tuple[str, float]:
    """Classify entry vs pre-event touch.

    ``top`` = (best_ask_px, best_ask_sz, best_bid_px, best_bid_sz) from the
    previous orderbook row. Returns (category, queue_ahead_shares).
    """
    best_ask, ask_sz, best_bid, bid_sz = top
    if direction == 1:  # buy side rests on the bid
        if price > best_bid:
            return "improve", 0.0
        if price == best_bid:
            return "join", bid_sz
        return "behind", (best_bid - price) / 100.0
    if price < best_ask:
        return "improve", 0.0
    if price == best_ask:
        return "join", ask_sz
    return "behind", (price - best_ask) / 100.0


def _fate_stats(
    cats: list[str],
    ahead: list[float],
    outcomes: list[str],
    times: list[float],
    otimes: list[float],
) -> dict[str, Any]:
    per_cat: dict[str, Any] = {}
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        n = len(idx)
        fill = float(np.mean([outcomes[i] in ("filled",) for i in idx]))
        delete = float(np.mean([outcomes[i] == "deleted" for i in idx]))
        tsub = [
            otimes[i] - times[i] for i in idx if outcomes[i] != "open" and np.isfinite(otimes[i])
        ]
        per_cat[cat] = {
            "n": n,
            "fill_rate": fill,
            "delete_rate": delete,
            "median_time_to_outcome_s": float(np.median(tsub)) if tsub else None,
        }
    join_idx = [i for i, c in enumerate(cats) if c == "join"]
    join_curve: dict[str, Any] = {}
    for lo, hi, label in zip(
        (0, JOIN_AHEAD_BINS[1], JOIN_AHEAD_BINS[2], JOIN_AHEAD_BINS[3]),
        (JOIN_AHEAD_BINS[1], JOIN_AHEAD_BINS[2], JOIN_AHEAD_BINS[3], float("inf")),
        JOIN_AHEAD_LABELS,
        strict=True,
    ):
        sel = [i for i in join_idx if lo <= ahead[i] < hi]
        if sel:
            join_curve[label] = {
                "n": len(sel),
                "fill_rate": float(np.mean([outcomes[i] == "filled" for i in sel])),
            }
    return {
        "ok": len(cats) >= 100,
        "n_submissions": len(cats),
        "per_category": per_cat,
        "join_fill_by_ahead": join_curve,
        "fill_rate_overall": float(np.mean([o == "filled" for o in outcomes]))
        if outcomes
        else None,
    }


def lobster_queue_fate(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    cats: list[str] = []
    ahead: list[float] = []
    times: list[float] = []
    otimes: list[float] = []
    outcomes: list[str] = []
    live: dict[int, int] = {}  # oid -> index into the arrays
    remaining: dict[int, float] = {}
    prev_top: tuple[float, float, float, float] | None = None
    with ob_path.open(newline="") as fo:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(fo), strict=True):
            if ev.event_type == SUBMISSION:
                if prev_top is not None:
                    cat, a = _classify_submit(float(ev.price), int(ev.direction), prev_top)
                    live[ev.order_id] = len(cats)
                    remaining[ev.order_id] = float(ev.size)
                    cats.append(cat)
                    ahead.append(a)
                    times.append(ev.time_s)
                    otimes.append(float("nan"))
                    outcomes.append("open")
            elif ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
                idx = live.get(ev.order_id)
                if idx is not None:
                    remaining[ev.order_id] = remaining.get(ev.order_id, 0.0) - ev.size
                    if not np.isfinite(otimes[idx]):
                        otimes[idx] = ev.time_s
                    outcomes[idx] = "filled"
                    if remaining[ev.order_id] <= 0:
                        live.pop(ev.order_id)
                        remaining.pop(ev.order_id)
            elif ev.event_type == CANCEL_PARTIAL:
                idx = live.get(ev.order_id)
                if idx is not None:
                    remaining[ev.order_id] = remaining.get(ev.order_id, 0.0) - ev.size
                    if remaining[ev.order_id] <= 0 and outcomes[idx] != "filled":
                        outcomes[idx] = "deleted"
                        otimes[idx] = ev.time_s
                        live.pop(ev.order_id)
                        remaining.pop(ev.order_id)
            elif ev.event_type == DELETE:
                idx = live.pop(ev.order_id, None)
                if idx is not None:
                    remaining.pop(ev.order_id, None)
                    otimes[idx] = ev.time_s
                    if outcomes[idx] != "filled":
                        outcomes[idx] = "deleted"
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                prev_top = (
                    float(asks[0][0]),
                    float(asks[0][1]),
                    float(bids[0][0]),
                    float(bids[0][1]),
                )
    return _fate_stats(cats, ahead, outcomes, times, otimes)


def sim_queue_fate(
    flow: Any | None = None,
    *,
    seed: int = 0,
    horizon: int = 30000,
) -> dict[str, Any]:
    cfg = ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    cats: list[str] = []
    ahead: list[float] = []
    times: list[float] = []
    otimes: list[float] = []
    outcomes: list[str] = []
    live: dict[int, int] = {}
    filled_oids: set[int] = set()
    n_trades_seen = 0
    for _ in range(horizon):
        ba_pre = sim.best_ask_level
        bb_pre = sim.best_bid_level
        before = set(sim._orders)
        sim.step()
        for oid, o in sim._orders.items():
            if oid in before:
                continue
            live[oid] = len(cats)
            if o.side == "sell":
                if ba_pre is not None and o.level < ba_pre:
                    cats.append("improve")
                elif ba_pre is not None and o.level == ba_pre:
                    cats.append("join")
                else:
                    cats.append("behind")
            else:
                if bb_pre is not None and o.level > bb_pre:
                    cats.append("improve")
                elif bb_pre is not None and o.level == bb_pre:
                    cats.append("join")
                else:
                    cats.append("behind")
            ahead.append(float(o.queue_ahead))
            times.append(float(sim.t))
            otimes.append(float("nan"))
            outcomes.append("open")
        for tr in sim.trades[n_trades_seen:]:
            filled_oids.add(tr.maker_order_id)
            idx = live.get(tr.maker_order_id)
            if idx is not None and not np.isfinite(otimes[idx]):
                otimes[idx] = tr.t
        n_trades_seen = len(sim.trades)
        for oid in before - set(sim._orders):
            idx = live.pop(oid, None)
            if idx is not None and outcomes[idx] == "open":
                outcomes[idx] = "deleted" if oid not in filled_oids else "filled"
                if not np.isfinite(otimes[idx]):
                    otimes[idx] = float(sim.t)
        for oid, idx in live.items():
            if oid in filled_oids and outcomes[idx] == "open":
                outcomes[idx] = "filled"
    for oid, idx in live.items():
        if oid in filled_oids:
            outcomes[idx] = "filled"
    return _fate_stats(cats, ahead, outcomes, times, otimes)


def queue_fate_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_queue_fate(msg_path, ob_path)
    arms = {
        "iid": sim_queue_fate(seed=seed),
        "regime": sim_queue_fate(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_queue_fate(
            SplitFlow(
                p_start=0.10,
                size_tail=1.2,
                k_min=10,
                k_max=600,
                intensity_mult=3.0,
                seed=seed + 2,
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    rj = real.get("per_category", {}).get("join")
    ri = real.get("per_category", {}).get("improve")
    if real.get("ok") and rj is not None:
        for name, arm in arms.items():
            aj = arm.get("per_category", {}).get("join")
            if aj is not None and abs(aj["fill_rate"] - rj["fill_rate"]) > 0.10:
                divergences.append(
                    f"{name}_joinfill_{aj['fill_rate']:.3f}_vs_{rj['fill_rate']:.3f}"
                )
            if ri is not None and aj is not None:
                real_grad = ri["fill_rate"] - rj["fill_rate"]
                arm_improve = arm.get("per_category", {}).get("improve")
                if arm_improve is not None:
                    arm_grad = arm_improve["fill_rate"] - aj["fill_rate"]
                    if real_grad > 0 and arm_grad < 0.5 * real_grad:
                        divergences.append(f"{name}_posgrad_{arm_grad:.3f}_vs_{real_grad:.3f}")
    payload: dict[str, Any] = {
        "kind": "queue_fate",
        "schema": "queue_fate.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "queue_position_determines_fate",
        "interpretation": (
            "Entry position vs the same-side touch: improve (strictly "
            "better price, new queue front), join (at the touch, queue "
            "ahead = displayed level size in shares on the real side / "
            "order count in the sim), behind (deeper level). Outcome "
            "filled = any EXECUTION on the order id; deleted = DELETE or "
            "cancelled-to-zero; open = still resting at the horizon. "
            "Hidden-order executions have no submission id and are "
            "skipped. Position is the strongest determinant of getting "
            "filled — this receipt measures the gradient on real tape "
            "and checks whether ZI flow reproduces it."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
