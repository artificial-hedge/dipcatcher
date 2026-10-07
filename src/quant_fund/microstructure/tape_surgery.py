"""tape_surgery — counterfactual event ablation on the real tape.

Stylized-fact *attribution*: instead of measuring one more statistic on
the tape, remove a whole event class from the LOBSTER message stream and
re-measure the core fact set on the surviving stream through the same
replay engine. The delta between the ablated stream and the baseline is
the share of each stylized fact that event class *produces*.

Ablations:

- ``drop_liquidity_exits``: CANCEL_PARTIAL + DELETE — the cancel channel
  (~half the tape's events).
- ``drop_hidden_execs``: EXECUTION_HIDDEN — fills that never touch the
  visible book.
- ``drop_improve_submissions``: SUBMISSION placed strictly inside the
  prevailing spread (quote improvement events).

Cascading is honest: dropping a SUBMISSION tombstones its order_id, so
later cancels/execs referencing it are dropped too — the order never
existed in the counterfactual world.

Facts per modified stream come from our own ``LobsterBook`` replay
(seeded from the first orderbook row — LOBSTER row i is the book state
AFTER message i, so the first message is part of the seed, not replayed),
keeping baseline and ablations on the same measurement substrate: exec
share, cancel:exec ratio, mean spread (ticks), median touch depth
(shares), exec-sign lag-1 autocorrelation.

Receipt ``tape_surgery.v1``, data_label REAL (tape-forensic lane; no sim
arm — the sim's generators do not offer a removable event class).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    SUBMISSION,
    LobsterBook,
    LobsterEvent,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

TAPE_SURGERY_SCHEMA = "tape_surgery.v1"
_TICKS = 100  # raw price (dollars * 10^4) -> ticks

# (name, predicate(event, book_before)) — the book state visible BEFORE
# the event is applied
_ABLATIONS: list[tuple[str, Callable[[LobsterEvent, LobsterBook], bool]]] = [
    (
        "drop_liquidity_exits",
        lambda ev, bk: ev.event_type in (CANCEL_PARTIAL, DELETE),
    ),
    (
        "drop_hidden_execs",
        lambda ev, bk: ev.event_type == EXECUTION_HIDDEN,
    ),
    (
        "drop_improve_submissions",
        lambda ev, bk: ev.event_type == SUBMISSION and _inside_spread(ev, bk),
    ),
]


def _inside_spread(ev: LobsterEvent, bk: LobsterBook) -> bool:
    bid = bk.top("bid", 1)
    ask = bk.top("ask", 1)
    if not bid or not ask:
        return False
    return bool(bid[0][0] < ev.price < ask[0][0])


def lobster_tape_surgery(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Baseline + each ablation arm on the real tape."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    events = list(parse_messages(msg))
    if not events:
        raise ValueError("tape_surgery requires a non-empty message file")
    with ob.open() as f:
        seed_row = next(csv.reader(f))
    seed_asks, seed_bids = parse_orderbook_row(seed_row)

    arms: dict[str, Any] = {}
    for name, pred in [("baseline", None), *_ABLATIONS]:
        tombstoned: set[int] = set()
        if pred is not None:
            # first pass: find order_ids whose birth event is dropped.
            # The seed row already carries event 0's post-state, so
            # event 0 is unreplayable (no pre-state to predicate on or
            # to un-ring): every arm inherits it as the initial book.
            book = LobsterBook()
            book.seed(seed_asks, seed_bids)
            for ev in events[1:]:
                if ev.order_id in tombstoned:
                    continue
                if pred(ev, book):
                    tombstoned.add(ev.order_id)
                    continue
                book.apply(ev)
        # measurement replay with its own seeded book
        kept: list[LobsterEvent] = []
        if pred is None:
            kept = events
        else:
            seen = set(tombstoned)
            for ev in events:
                if ev.order_id in seen:
                    continue
                kept.append(ev)
        # measurement replay through a fresh seeded book
        arms[name] = _facts_seeded(kept, seed_asks, seed_bids, skip_apply_of=events[0])
        arms[name]["n_dropped"] = len(events) - len(kept)
    deltas: dict[str, Any] = {}
    base = arms["baseline"]
    for name in arms:
        if name == "baseline":
            continue
        d = {}
        for k, v in arms[name].items():
            b = base.get(k)
            if isinstance(v, (int, float)) and isinstance(b, (int, float)):
                d[k] = round(v - b, 4)
        deltas[name] = d
    return {"arms": arms, "attribution_deltas": deltas}


def _facts_seeded(
    events: list[LobsterEvent],
    seed_asks: list[tuple[int, int]],
    seed_bids: list[tuple[int, int]],
    skip_apply_of: LobsterEvent | None = None,
) -> dict[str, Any]:
    return _facts_with_seed(events, seed_asks, seed_bids, skip_apply_of=skip_apply_of)


def _facts_with_seed(
    events: list[LobsterEvent],
    seed_asks: list[tuple[int, int]],
    seed_bids: list[tuple[int, int]],
    skip_apply_of: LobsterEvent | None = None,
) -> dict[str, Any]:
    book = LobsterBook()
    book.seed(seed_asks, seed_bids)
    signs: list[int] = []
    spreads: list[int] = []
    depths: list[int] = []
    n_exec = n_cancel = 0
    for i, ev in enumerate(events):
        if ev.event_type == EXECUTION:
            n_exec += 1
            signs.append(-ev.direction)
        elif ev.event_type in (CANCEL_PARTIAL, DELETE):
            n_cancel += 1
        if ev is not skip_apply_of:
            # The seed row is this event's post-state; applying it a
            # second time would double-count its size/effect.
            book.apply(ev)
        if i % 8 == 0:
            bid = book.top("bid", 1)
            ask = book.top("ask", 1)
            if bid and ask and ask[0][0] > bid[0][0]:
                spreads.append(ask[0][0] - bid[0][0])
                depths.append(bid[0][1] + ask[0][1])
    s = np.asarray(signs, dtype=float)
    autoc = float(np.corrcoef(s[1:], s[:-1])[0, 1]) if s.size > 2 else float("nan")
    sp = np.asarray(spreads, dtype=float)
    dp = np.asarray(depths, dtype=float)
    n = len(events)
    return {
        "n_kept": n,
        "n_exec": n_exec,
        "n_cancel_delete": n_cancel,
        "exec_share": round(n_exec / n, 4) if n else None,
        "cancel_exec_ratio": round(n_cancel / n_exec, 2) if n_exec else None,
        "spread_mean_ticks": round(float(sp.mean()) / _TICKS, 3) if sp.size else None,
        "touch_depth_med_shares": float(np.median(dp)) if dp.size else None,
        "exec_sign_autocorr_lag1": round(autoc, 4) if np.isfinite(autoc) else None,
    }


def tape_surgery_bench(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Counterfactual ablation bench. Sealed."""
    out = lobster_tape_surgery(tape_dir, ticker)
    notable = []
    for name, d in out["attribution_deltas"].items():
        if "exec_sign_autocorr_lag1" in d and abs(d["exec_sign_autocorr_lag1"]) > 0.05:
            notable.append(f"{name}: sign_autocorr {d['exec_sign_autocorr_lag1']:+.3f}")
        if "spread_mean_ticks" in d and abs(d["spread_mean_ticks"]) > 1.0:
            notable.append(f"{name}: spread {d['spread_mean_ticks']:+.2f} ticks")
    payload: dict[str, Any] = {
        "kind": "tape_surgery",
        "schema": TAPE_SURGERY_SCHEMA,
        "ticker": ticker,
        "arms": out["arms"],
        "attribution_deltas": out["attribution_deltas"],
        "notable_attributions": notable,
        "claim": "event_class_attribution_measured",
        "interpretation": (
            "Counterfactual microstructure: each arm deletes one event "
            "class (with order-id cascading) and re-measures the fact set. "
            "The deltas attribute stylized facts to the classes that "
            "generate them — e.g. how much sign memory survives with the "
            "cancel channel removed."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "REAL"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
