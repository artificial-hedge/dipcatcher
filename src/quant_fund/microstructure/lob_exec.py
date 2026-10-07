"""lob_exec — implementation shortfall on a real reconstructed tape.

Replays the LOBSTER tape through `LobsterBook` (96%-validated
reconstruction, `lobster.py`) and executes a synthetic parent order
against the reconstructed visible book: children walk the ask (for a
buy) consuming quoted depth price-time; cost = VWAP minus arrival mid.

Honest scope: this measures **visible-liquidity** shortfall — only the
displayed top-10 exists in the tape; hidden executions and off-display
depth are invisible and the resync band truncates deep levels. Labels
MIXED: real tape, synthetic order flow.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from quant_fund.microstructure.lobster import (
    EXECUTION_HIDDEN,
    HALT,
    LobsterBook,
    LobsterEvent,
    parse_messages,
    parse_orderbook_row,
    resync_band,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LOB_EXEC_SCHEMA = "lob_exec.v1"


def _walk_book(book: LobsterBook, side: str, qty: int) -> tuple[int, float, float]:
    """Consume up to qty units at the visible best levels.

    Returns (filled, notional_units, walk_cost_units) where walk cost is
    Σ (price_i − ref_price)·qty_i — the depth-consumption cost relative
    to the prevailing best quote, free of drift between children.
    """
    opp = "ask" if side == "buy" else "bid"
    levels = book.top(opp, 10)
    ref = levels[0][0] if levels else None
    filled = 0
    notional = 0.0
    walk = 0.0
    for price, size in levels:
        take = min(size, qty - filled)
        if take <= 0:
            break
        # consume from the book so later children see the thinner queue
        if opp == "ask":
            book.ask[price] = size - take
            if book.ask[price] == 0:
                del book.ask[price]
        else:
            book.bid[price] = size - take
            if book.bid[price] == 0:
                del book.bid[price]
        notional += price * take
        if ref is not None:
            walk += (price - ref) * take
        filled += take
        if filled >= qty:
            break
    # For a buy we pay >= the best ask: walk >= 0. For a sell, bid prices
    # are below the ref (best bid): walk <= 0 — sign-normalize to >= 0.
    return filled, notional, abs(walk)


def twap_children(parent: int, n_children: int) -> list[int]:
    base, rem = divmod(parent, n_children)
    return [base + (1 if i < rem else 0) for i in range(n_children)]


def exec_on_tape(
    events: list[LobsterEvent],
    snapshots: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
    *,
    start: int,
    parent_size: int,
    n_children: int,
    events_per_child: int,
    side: str,
    tick_units: float = 100.0,
) -> dict[str, Any] | None:
    """Execute one parent TWAP-style on the tape starting at event `start`."""
    book = LobsterBook()
    book.seed(*snapshots[0])
    # Row 0 is the book state AFTER message 0: the seed already includes
    # event 0 — replay from event 1 or the first event is double-counted.
    n_resync = 0
    for i in range(1, start):
        ev = events[i]
        book.apply(ev)
        if ev.event_type in (EXECUTION_HIDDEN, HALT):
            continue
        ae, be = snapshots[i]
        if book.top("ask", 10) != ae or book.top("bid", 10) != be:
            resync_band(book, ae, be)
            n_resync += 1
    bb, ba = book.top("bid", 1), book.top("ask", 1)
    if not bb or not ba:
        return None
    arrival_mid = (ba[0][0] + bb[0][0]) / 2
    filled = 0
    notional = 0.0
    walk_total = 0.0
    cursor = start
    for child in twap_children(parent_size, n_children):
        f, n, w = _walk_book(book, side, child)
        filled += f
        notional += n
        walk_total += w
        # breathe: replay events between children (book evolves)
        end = min(cursor + events_per_child, len(events))
        while cursor < end:
            ev = events[cursor]
            book.apply(ev)
            cursor += 1
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            # Row cursor-1 is the book state after the event just applied.
            if cursor - 1 < len(snapshots):
                ae, be = snapshots[cursor - 1]
                if book.top("ask", 10) != ae or book.top("bid", 10) != be:
                    resync_band(book, ae, be)
                    n_resync += 1
    if filled == 0:
        return None
    vwap = notional / filled
    sgn = 1.0 if side == "buy" else -1.0
    return {
        # Total implementation shortfall vs arrival mid — includes drift
        # between children (negative = price moved in our favor).
        "is_total_ticks": sgn * (vwap - arrival_mid) / tick_units,
        # Drift-free liquidity cost: per-child walk relative to the
        # prevailing best quote at fill time.
        "liquidity_cost_ticks": walk_total / filled / tick_units,
        "fill_fraction": filled / parent_size,
        "n_resyncs": n_resync,
        "events_consumed": cursor - start,
    }


def lob_exec_bench(
    tape_dir: Path, ticker: str = "AMZN", *, tick_units: float = 100.0
) -> dict[str, Any]:
    msg = next(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = next(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    rows = list(csv.reader(ob.open()))
    snapshots = [parse_orderbook_row(r) for r in rows]
    events = list(parse_messages(msg))
    n = min(len(events), len(snapshots))
    events, snapshots = events[:n], snapshots[:n]

    cells: dict[str, dict[str, Any]] = {}
    # Parents sized against typical visible depth (~few hundred shares).
    for parent_size, tag in ((50, "small"), (200, "medium"), (500, "large")):
        for n_children, sched in ((5, "twap5"), (20, "twap20")):
            outs = []
            # Spread episodes through the middle of the tape, both sides.
            for k in range(6):
                start = n // 4 + k * (n // 12)
                if start >= n - 2000:
                    break
                out = exec_on_tape(
                    events,
                    snapshots,
                    start=start,
                    parent_size=parent_size,
                    n_children=n_children,
                    events_per_child=200,
                    side="buy" if k % 2 == 0 else "sell",
                    tick_units=tick_units,
                )
                if out is not None:
                    outs.append(out)
            if not outs:
                cells[f"{tag}_{sched}"] = {"n_episodes": 0}
                continue
            shorts = [o["is_total_ticks"] for o in outs]
            liqs = [o["liquidity_cost_ticks"] for o in outs]
            cells[f"{tag}_{sched}"] = {
                "n_episodes": len(outs),
                "is_total_ticks_mean": float(sum(shorts) / len(shorts)),
                "is_total_ticks_min": float(min(shorts)),
                "is_total_ticks_max": float(max(shorts)),
                "liquidity_cost_ticks_mean": float(sum(liqs) / len(liqs)),
                "fill_fraction_mean": float(sum(o["fill_fraction"] for o in outs) / len(outs)),
                "resyncs_total": int(sum(o["n_resyncs"] for o in outs)),
            }
    payload: dict[str, Any] = {
        "schema": LOB_EXEC_SCHEMA,
        "kind": "lob_exec",
        "ticker": ticker,
        "cells": cells,
        "interpretation": (
            "TWAP shortfall on the reconstructed visible LOBSTER book, "
            "tick-normalized. liquidity_cost is drift-free (vs best quote "
            "at fill time); is_total vs arrival mid mixes in tape drift "
            "and can be negative. VISIBLE-liquidity only: top-10 tape, "
            "hidden executions invisible, resync band truncates depth"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
