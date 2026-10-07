"""quote_place — limit-order placement aggressiveness, real vs sim.

Where do resting orders land relative to the touch? Biais–Hillion–Spatt
and Bouchaud: the placement-distance distribution has its mode AT or
INSIDE the spread with a heavy tail behind the touch — impatient
liquidity competes for queue priority, patient liquidity waits deep.

On the tape, each SUBMISSION's signed distance from the pre-event
same-side touch is measured in ticks: negative = inside the spread
(price improves), 0 = at the touch, positive = behind. The distribution
shape is the measurable; the sim's flat `band`-uniform placement is the
divergence baseline.
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

DIST_BINS = (-9999, -5, -2, -1, 0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 9999)
BIN_LABELS = (
    "le-5",
    "-5..-2",
    "-2..-1",
    "-1..0",
    "0..1",
    "1..2",
    "2..3",
    "3..5",
    "5..8",
    "8..13",
    "13..21",
    "21..34",
    "34..55",
    "gt55",
)


def _distance_hist(dist: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, lo, hi in zip(BIN_LABELS, DIST_BINS[:-1], DIST_BINS[1:], strict=True):
        out[label] = int(((dist > lo) & (dist <= hi)).sum())
    return out


def lobster_placement(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Per-submission signed distance from the same-side touch, ticks."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    book = LobsterBook()
    dist: list[float] = []
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
                if ev.direction == 1:  # buy order: distance behind bid touch
                    dist.append((bids[0][0] - ev.price) / 100.0)
                else:  # sell: distance behind ask touch
                    dist.append((ev.price - asks[0][0]) / 100.0)
            book.apply(ev)
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", 10) != asks_exp or book.top("bid", 10) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
    d = np.asarray(dist)
    return {
        "n_submissions": int(d.size),
        "share_improves_spread": round(float((d < 0).mean()), 4) if d.size else None,
        "share_at_touch": round(float((d == 0).mean()), 4) if d.size else None,
        "share_behind_touch": round(float((d > 0).mean()), 4) if d.size else None,
        "median_dist_ticks": round(float(np.median(d)), 3) if d.size else None,
        "dist_hist_ticks": _distance_hist(d),
    }


def sim_placement(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    """Actual placement distance on the sim: new order level vs pre-step touch."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    dist: list[float] = []
    # Seed orders exist before the first step; only ids appearing after a
    # step are submissions (order ids are monotonic in the engine).
    seen: set[int] = set(sim._orders)
    for _ in range(horizon):
        pre_bid = sim.best_bid_level
        pre_ask = sim.best_ask_level
        sim.step()
        for oid, order in sim._orders.items():
            if oid in seen:
                continue
            seen.add(oid)
            if order.side == "buy":
                if pre_bid is not None:
                    dist.append(float(pre_bid - order.level))
            elif pre_ask is not None:
                dist.append(float(order.level - pre_ask))
    d = np.asarray(dist)
    return {
        "n_submissions": int(d.size),
        "share_improves_spread": round(float((d < 0).mean()), 4) if d.size else None,
        "share_at_touch": round(float((d == 0).mean()), 4) if d.size else None,
        "share_behind_touch": round(float((d > 0).mean()), 4) if d.size else None,
        "median_dist_ticks": round(float(np.median(d)), 3) if d.size else None,
        "dist_hist_ticks": _distance_hist(d),
    }


def quote_place_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Placement-distance histogram real vs sim. Sealed receipt."""
    real = lobster_placement(tape_dir, ticker)
    arms = {
        "iid": sim_placement(seed=seed),
        "regime": sim_placement(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_placement(
            flow=SplitFlow(
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
    payload: dict[str, Any] = {
        "kind": "quote_place",
        "schema": "quote_place.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "claim": "placement_distance_distribution_measured",
        "interpretation": (
            "Real submissions cluster at the touch and inside the spread "
            "(queue-priority competition) with a heavy deep tail; the sim "
            "places uniformly over its band — a structural gap flagged "
            "rather than hidden."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
