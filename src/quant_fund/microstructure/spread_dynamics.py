"""spread_dynamics — spread width distribution over the tape day.

How wide is the book, and for how long? Track time-weighted spread
occupancy: fraction of the session spent at each spread width (in
ticks), mean/median spread, and the tight-market share (spread ≤ 2
ticks) — the liquidity regime a strategy actually faces.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    LobsterBook,
    parse_messages,
    parse_orderbook_row,
    resync_band,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SPREAD_BINS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 1 << 30)


def _spread_profile(spreads: list[float]) -> dict[str, Any]:
    """Time/event-weighted occupancy stats over a tick-spread series."""
    if not spreads:
        return {"ok": False, "reason": "no_spreads"}
    arr = np.asarray(spreads, dtype=float)
    bins = np.asarray(SPREAD_BINS, dtype=float)
    occ: dict[str, float] = {}
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (arr > lo) & (arr <= hi)
        label = f"{int(lo) + 1}-{int(hi) if hi < (1 << 30) else 'inf'}"
        occ[label] = round(float(mask.mean()), 4)
    return {
        "n_obs": int(arr.size),
        "mean_spread_ticks": round(float(arr.mean()), 3),
        "median_spread_ticks": round(float(np.median(arr)), 3),
        "p90_spread_ticks": round(float(np.percentile(arr, 90)), 3),
        "tight_share_le2": round(float((arr <= 2).mean()), 4),
        "spread_occupancy": occ,
    }


def lobster_spread_dynamics(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Event-sampled spread over the real tape (post-event book state)."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    book = LobsterBook()
    spreads: list[float] = []
    with ob.open() as fo:
        ob_rows = csv.reader(fo)
        first = True
        for ev, ob_row in zip(parse_messages(msg), ob_rows, strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if first:
                # Row 0 is the book state AFTER message 0: seeding from it
                # already includes event 0 — applying it would double-count
                # the first event.
                book.seed(asks_exp, bids_exp)
                first = False
                continue
            book.apply(ev)
            if book.top("ask", 10) != asks_exp or book.top("bid", 10) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
            asks = book.top("ask", 1)
            bids = book.top("bid", 1)
            if asks and bids:
                spreads.append((asks[0][0] - bids[0][0]) / 100.0)  # raw→ticks
    return _spread_profile(spreads)


def sim_spread_dynamics(horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed))
    spreads: list[float] = []
    for _ in range(horizon):
        sim.step()
        if sim.spread_ticks is not None:
            spreads.append(float(sim.spread_ticks))
    return _spread_profile(spreads)


def spread_dynamics_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Spread-occupancy comparison real vs sim. Sealed."""
    real = lobster_spread_dynamics(tape_dir, ticker)
    sim = sim_spread_dynamics(seed=seed)
    divergences = []
    if real.get("mean_spread_ticks") is not None and sim.get("mean_spread_ticks") is not None:
        ratio = real["mean_spread_ticks"] / sim["mean_spread_ticks"]
        if not math.isfinite(ratio):
            ratio = 0.0
        if abs(ratio - 1.0) > 0.5:
            divergences.append(f"mean_spread_ratio_{ratio:.2f}")
    payload: dict[str, Any] = {
        "kind": "spread_dynamics",
        "schema": "spread_dynamics.v1",
        "ticker": ticker,
        "real": real,
        "sim": sim,
        "divergences": divergences,
        "claim": "spread_occupancy_distribution_measured",
        "interpretation": (
            "The spread distribution is the liquidity regime surface: "
            "occupancy concentrates at 1-2 ticks in tight names; a fat "
            "right tail marks a wide-book stock like 2012-era AMZN."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
