"""depth_consumption — what share of displayed depth does a fill eat?

For each EXECUTION on the real tape, the pre-event orderbook row gives
the visible size at the executed level. `consumption = exec_size /
level_depth` measures sweep depth vs posted liquidity: real fills
typically take only part of the touch (icebergs refill the rest);
a share ≥ 1 marks a full level sweep. On the sim, the maker fill's
`queue_ahead` bookkeeping gives the same quantity at the touch.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, parse_messages, parse_orderbook_row
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _consume_stats(shares: list[float], sweep_frac: float = 1.0) -> dict[str, Any]:
    if not shares:
        return {"ok": False, "reason": "no_fills"}
    arr = np.asarray(shares, dtype=float)
    return {
        "n_fills": int(arr.size),
        "mean_consumption": round(float(arr.mean()), 4),
        "median_consumption": round(float(np.median(arr)), 4),
        "p90_consumption": round(float(np.percentile(arr, 90)), 4),
        "full_sweep_share": round(float((arr >= sweep_frac).mean()), 4),
        "over_sweep_share": round(float((arr > 1.0 + 1e-9).mean()), 4),
    }


def lobster_depth_consumption(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Exec size / visible size at that level (pre-event snapshot row)."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    shares: list[float] = []
    n_exec = 0
    with ob.open() as fo:
        prev_asks: list[tuple[int, int]] = []
        prev_bids: list[tuple[int, int]] = []
        for ev, ob_row in zip(parse_messages(msg), csv.reader(fo), strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if ev.event_type == EXECUTION and ev.direction == -1:
                # sell-direction resting order hit = buyer-initiated exec
                depth = dict(prev_asks).get(ev.price)
                if depth and depth > 0:
                    shares.append(ev.size / depth)
                n_exec += 1
            elif ev.event_type == EXECUTION and ev.direction == 1:
                depth = dict(prev_bids).get(ev.price)
                if depth and depth > 0:
                    shares.append(ev.size / depth)
                n_exec += 1
            prev_asks, prev_bids = asks_exp, bids_exp
    out = _consume_stats(shares)
    out["n_exec_total"] = n_exec
    out["n_with_depth"] = len(shares)
    return out


def sim_depth_consumption(horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Sim analog: fill qty vs the visible queue the maker joined.

    `maker_queue_ahead_at_submit` is the number of orders queued ahead
    of the resting maker when it submitted — the honest proxy for the
    visible depth at the maker's level. `qty / (queue_ahead + 1)` is
    the share of that queue the fill consumed.
    """
    sim = ZILobSimulator(ZILobConfig(seed=seed))
    for _ in range(horizon):
        sim.step()
    shares = [
        tr.qty / (tr.maker_queue_ahead_at_submit + 1)
        for tr in sim.trades
        if tr.maker_queue_ahead_at_submit >= 0
    ]
    out = _consume_stats(shares)
    out["mode"] = "qty_vs_maker_queue_at_submit"
    return out


def depth_consumption_bench(
    tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7
) -> dict[str, Any]:
    """Depth-consumption comparison real vs sim. Sealed."""
    real = lobster_depth_consumption(tape_dir, ticker)
    sim = sim_depth_consumption(seed=seed)
    divergences = []
    if real.get("full_sweep_share") is not None and sim.get("full_sweep_share") is not None:
        gap = real["full_sweep_share"] - sim["full_sweep_share"]
        if abs(gap) > 0.2:
            divergences.append(f"full_sweep_share_gap_{gap:+.2f}")
    payload: dict[str, Any] = {
        "kind": "depth_consumption",
        "schema": "depth_consumption.v1",
        "ticker": ticker,
        "real": real,
        "sim": sim,
        "divergences": divergences,
        "claim": "depth_consumption_share_measured",
        "interpretation": (
            "Low consumption share = fills nibble the touch (icebergs or "
            "large posted size); high share = sweeps. Over-sweep (>1) "
            "marks hidden liquidity also filling at that level."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
