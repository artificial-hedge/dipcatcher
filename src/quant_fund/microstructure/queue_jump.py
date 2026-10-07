"""queue_jump — inside-spread placement vs queue length at the touch.

When a new limit order arrives at a long queue, does the trader join
the back or pay a tick to skip ahead? The queue-jump hypothesis
(Biais–Hillion–Spatt 1995; Cont–Kukanov): the longer the displayed
queue at the touch, the more expensive time-priority is, so impatient
liquidity steps inside the spread instead of joining it.

On the tape, each SUBMISSION (LOBSTER type 1) is scored at arrival,
pre-event:

- ``dist_ticks``: signed distance from the same-side touch in ticks;
  negative = better than the touch (inside or crossing)
- ``queue_at_touch``: displayed shares queued at the same-side touch
- ``inside_spread``: strictly between the touches (bid < p < ask)
- ``improves_touch``: better than the same-side touch — includes
  marketable submissions that cross the spread entirely

Headline: the share of submissions placed strictly inside the spread,
conditioned on queue-length bin (1-9, 10-49, 50-99, 100+ shares),
plus the unconditional share improving the touch.

The ZI-LOB sim cannot produce this statistic: its LO arrivals have no
placement choice (they land at a fixed band offset), so the sim arm is
reported honestly as ``mechanism_present: false`` — like hidden_depth's
hidden-orders arm.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    EXECUTION_HIDDEN,
    HALT,
    SUBMISSION,
    LobsterBook,
    parse_messages,
    parse_orderbook_row,
    resync_band,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

QUEUE_BINS = ((1, 9), (10, 49), (50, 99), (100, 10**9))
QUEUE_BIN_LABELS = ("1-9", "10-49", "50-99", "100+")


def _fin(x: float | None) -> float | None:
    """NaN/±inf → None so receipts stay JSON-clean."""
    if x is None:
        return None
    return x if np.isfinite(x) else None


def _queue_bin(queue_at_touch: int) -> str:
    for label, (lo, hi) in zip(QUEUE_BIN_LABELS, QUEUE_BINS, strict=True):
        if lo <= queue_at_touch <= hi:
            return label
    raise ValueError(f"queue_at_touch {queue_at_touch} out of range")


def _empty_bins() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "n_submissions": 0,
            "share_inside_spread": None,
            "share_improves_touch": None,
            "median_dist_ticks": None,
        }
        for label in QUEUE_BIN_LABELS
    }


def lobster_queue_jump(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Per-submission queue-jump stats on the real LOBSTER tape."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    book = LobsterBook()
    bin_dists: dict[str, list[float]] = {label: [] for label in QUEUE_BIN_LABELS}
    bin_inside: dict[str, int] = {label: 0 for label in QUEUE_BIN_LABELS}
    bin_improves: dict[str, int] = {label: 0 for label in QUEUE_BIN_LABELS}
    n_inside = n_improves = n_crossing = 0
    seeded = False
    with ob.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(f_ob), strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if not seeded:
                # Row 0 is the book state AFTER message 0: seeding from it
                # already includes event 0 — applying it would double-count
                # the first event.
                book.seed(asks_exp, bids_exp)
                seeded = True
                continue
            asks, bids = book.top("ask", 1), book.top("bid", 1)
            if ev.event_type == SUBMISSION and asks and bids:
                best_ask, best_bid = asks[0][0], bids[0][0]
                if ev.direction == 1:  # buy: distance behind bid touch
                    dist = (best_bid - ev.price) / 100.0
                    queue_at_touch = bids[0][1]
                    crossing = ev.price >= best_ask
                else:  # sell: distance behind ask touch
                    dist = (ev.price - best_ask) / 100.0
                    queue_at_touch = asks[0][1]
                    crossing = ev.price <= best_bid
                inside = best_bid < ev.price < best_ask
                improves = dist < 0.0
                label = _queue_bin(queue_at_touch)
                bin_dists[label].append(dist)
                bin_inside[label] += int(inside)
                bin_improves[label] += int(improves)
                n_inside += int(inside)
                n_improves += int(improves)
                n_crossing += int(crossing)
            book.apply(ev)
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", 10) != asks_exp or book.top("bid", 10) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
    n_subs = sum(len(v) for v in bin_dists.values())
    by_bin = _empty_bins()
    for label in QUEUE_BIN_LABELS:
        d = np.asarray(bin_dists[label])
        if not d.size:
            continue
        by_bin[label] = {
            "n_submissions": int(d.size),
            "share_inside_spread": round(float(bin_inside[label] / d.size), 4),
            "share_improves_touch": round(float(bin_improves[label] / d.size), 4),
            "median_dist_ticks": _fin(round(float(np.median(d)), 3)),
        }
    return {
        "n_submissions": n_subs,
        "n_inside_spread": n_inside,
        "n_improves_touch": n_improves,
        "n_crossing": n_crossing,
        "share_inside_spread": _fin(round(n_inside / n_subs, 4)) if n_subs else None,
        "share_improves_touch": _fin(round(n_improves / n_subs, 4)) if n_subs else None,
        "by_queue_bin": by_bin,
        "data_label": "REAL",
    }


def sim_queue_jump(horizon: int = 8000, seed: int = 7) -> dict[str, Any]:
    """Sim arm — no such mechanism. LO arrivals land at a fixed band
    offset; there is no queue-conditional placement choice to measure."""
    sim = ZILobSimulator(ZILobConfig(seed=seed))
    n_lo = 0
    # Seed orders exist before the first step; only ids appearing after a
    # step are submissions (order ids are monotonic in the engine).
    seen: set[int] = set(sim._orders)
    for _ in range(horizon):
        sim.step()
        for oid in sim._orders:
            if oid not in seen:
                seen.add(oid)
                n_lo += 1
    return {
        "n_submissions": n_lo,
        "mechanism_present": False,
        "reason": "sim LO arrivals have no placement choice (fixed band offset); "
        "queue-jump share is undefined, not zero",
        "data_label": "SYNTHETIC",
    }


def queue_jump_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Inside-spread share vs touch queue length, real vs sim. Sealed."""
    real = lobster_queue_jump(tape_dir, ticker)
    sim = sim_queue_jump(seed=seed)
    divergences = []
    bins = real.get("by_queue_bin", {})
    lo_share = bins.get("1-9", {}).get("share_inside_spread")
    hi_share = bins.get("100+", {}).get("share_inside_spread")
    if lo_share is not None and hi_share is not None and hi_share > lo_share:
        divergences.append(
            f"inside_share_rises_with_queue_{lo_share}_to_{hi_share};"
            "sim_has_no_queue_conditional_placement"
        )
    payload: dict[str, Any] = {
        "kind": "queue_jump",
        "schema": "queue_jump.v1",
        "ticker": ticker,
        "real": real,
        "sim": sim,
        "divergences": divergences,
        "claim": "queue_jump_measured",
        "interpretation": (
            "Queue-jump hypothesis: when the displayed queue at the touch "
            "is long, time-priority is expensive, so impatient liquidity "
            "steps inside the spread. share_inside_spread rising across "
            "queue bins is the signature. The sim arm lacks any placement "
            "choice, so the comparison is a pure mechanism gap."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
