"""price_improvement — where do hidden fills price relative to the touch?

LOBSTER EXECUTION_HIDDEN events (iceberg/hidden fills) execute at
prices that may sit *inside* the quoted spread — midpoint matches and
hidden liquidity improve the effective fill vs the visible touch.
For each hidden exec on the real tape, compute the signed distance
from the pre-event visible touch on the aggressed side, in ticks:
positive = price improvement vs crossing the visible book.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    EXECUTION_HIDDEN,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def lobster_price_improvement(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Per hidden exec: (touch_price - exec_price) for buys, signed.

    `direction` on an EXECUTION_HIDDEN is the *resting* (hidden) order's
    side: direction=-1 → hidden sell filled by an incoming buy → the
    improvement reference is the best ask; the exec price below it is
    improvement (buyer paid less than the visible touch).
    """
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    improvements: list[float] = []
    n_hidden = 0
    with ob.open() as fo:
        prev_asks: list[tuple[int, int]] = []
        prev_bids: list[tuple[int, int]] = []
        for ev, ob_row in zip(parse_messages(msg), csv.reader(fo), strict=True):
            if ev.event_type == EXECUTION_HIDDEN:
                n_hidden += 1
                if ev.direction == -1 and prev_asks:
                    ref = prev_asks[0][0]
                    improvements.append((ref - ev.price) / 100.0)
                elif ev.direction == 1 and prev_bids:
                    ref = prev_bids[0][0]
                    improvements.append((ev.price - ref) / 100.0)
            prev_asks, prev_bids = parse_orderbook_row(ob_row)
    if not improvements:
        return {"ok": False, "reason": "no_hidden_execs", "n_hidden": n_hidden}
    arr = np.asarray(improvements, dtype=float)
    return {
        "n_hidden": n_hidden,
        "n_priced": int(arr.size),
        "mean_improvement_ticks": round(float(arr.mean()), 4),
        "median_improvement_ticks": round(float(np.median(arr)), 4),
        "improved_share": round(float((arr > 0).mean()), 4),
        "at_touch_share": round(float(np.isclose(arr, 0.0).mean()), 4),
        "worse_share": round(float((arr < 0).mean()), 4),
        "p90_improvement_ticks": round(float(np.percentile(arr, 90)), 4),
    }


def price_improvement_bench(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Hidden-fill improvement stats on real tape. Sealed."""
    real = lobster_price_improvement(tape_dir, ticker)
    divergences = []
    if real.get("improved_share") is not None and real["improved_share"] > 0.1:
        divergences.append("sim_has_no_hidden_mechanism_to_price_improve")
    payload: dict[str, Any] = {
        "kind": "price_improvement",
        "schema": "price_improvement.v1",
        "ticker": ticker,
        "real": real,
        "divergences": divergences,
        "claim": "hidden_fill_price_improvement_measured",
        "interpretation": (
            "Positive improvement = hidden fills execute inside the "
            "spread, worth the quoted spread minus realized slippage; "
            "zero = they print at the touch (dark liquidity doesn't "
            "improve price, just adds size)."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "REAL"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
