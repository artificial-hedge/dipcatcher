"""lob_invariants — executable matching-engine invariant spec.

A limit-order book is a deterministic transition system; every legal
event preserves a small set of invariants. This module is the
executable specification of those invariants, checked over a
normalized event stream — the same spec audits the LOBSTER replay of
the real tape and the ZI-LOB simulator's synthesized stream.

Invariants checked per event (after applying the event):

- ``I1 nonneg_depth``: every price level and every tracked order keeps
  remaining size > 0.
- ``I2 uncrossed_at_rest``: when both sides are non-empty,
  ``max(bid) < min(ask)``. A crossed resting book should not persist.
- ``I3 order_conservation``: for each id, submitted size = executed +
  cancelled + remaining (checked implicitly by exact remaining-size
  accounting; an exec/cancel larger than the remainder is a violation).
- ``I4 id_validity``: exec/delete events reference a live order id;
  submit events reference a fresh id.
- ``I5 exec_price_truth``: a visible execution prints at the resting
  order's own price.
- ``I6 time_monotone``: event times are non-decreasing.
- ``I7 exec_at_best``: a *visible* execution consumes the best price
  on its side — the resting order's price equals the side's extreme
  just before the trade. On the real tape the extreme comes from the
  official orderbook row (authoritative), not the replayed book.
- ``I8 submit_uncrossed``: a resting submission priced at or through
  the opposite touch is marketable — it should not appear as a resting
  order. Real tape: ~0 by LOBSTER's recording convention; sim: its
  submit rule should never cross.

State-sensitive checks (I2, I7, I8) use the official orderbook row
when the stream provides one (``touch`` is the post-event official
top; the previous row's touch is the pre-event state). Pure event
replay without band resync drifts from the official book —
``replay_crossed_rate`` measures that drift as a diagnostic. On the
sim, where the replayed book IS the true book, any structural
violation is an engine bug.

The real tape's nonzero violation rates measure the shadow book the
10-level window cannot see; the sim's rates measure spec conformance.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

_KIND_BY_LOBSTER_TYPE = {
    SUBMISSION: "submit",
    CANCEL_PARTIAL: "cancel",
    DELETE: "delete",
    EXECUTION: "exec",
    EXECUTION_HIDDEN: "hidden_exec",
}


@dataclass(frozen=True)
class LobEvent:
    """Normalized book event; ``price``/``size`` are integer units of the
    stream's own tick convention (10^-4 dollars on LOBSTER, sim ticks
    ``level_to_price / tick``)."""

    t: float
    kind: str  # submit | cancel | delete | exec | hidden_exec
    order_id: int
    price: int
    size: int
    side: int  # +1 bid-side rest, -1 ask-side rest


def lobster_events(
    msg_path: Path, ob_path: Path | None = None
) -> Iterable[tuple[LobEvent, tuple[int, int] | None]]:
    """Yield (event, post-event official touch) pairs.

    ``touch`` = (best_ask_px, best_bid_px) from the same orderbook row;
    ``None`` when no book file is given or the row is empty.
    """
    if ob_path is None:
        for ev in parse_messages(msg_path):
            kind = _KIND_BY_LOBSTER_TYPE.get(ev.event_type)
            if kind is None:
                continue
            yield (
                LobEvent(
                    t=ev.time_s,
                    kind=kind,
                    order_id=ev.order_id,
                    price=ev.price,
                    size=ev.size,
                    side=ev.direction,
                ),
                None,
            )
        return
    with ob_path.open(newline="") as fo:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(fo), strict=True):
            kind = _KIND_BY_LOBSTER_TYPE.get(ev.event_type)
            if kind is None:
                continue
            asks, bids = parse_orderbook_row(ob_row)
            touch = (asks[0][0], bids[0][0]) if asks and bids else None
            yield (
                LobEvent(
                    t=ev.time_s,
                    kind=kind,
                    order_id=ev.order_id,
                    price=ev.price,
                    size=ev.size,
                    side=ev.direction,
                ),
                touch,
            )


def sim_events(sim: ZILobSimulator, tick: float, horizon: int) -> Iterable[tuple[LobEvent, None]]:
    """Synthesize a LOBSTER-shaped stream from a sim run via oid-diffing.

    The seeded opening book is emitted as submits at t=0 (the sim plants
    resting orders before step 0 — the analog of LOBSTER's pre-open
    book, which its message stream emits as submissions).
    """
    seeded = sorted(sim._orders.items(), key=lambda kv: kv[1].side)
    for oid, o in seeded:
        yield (
            LobEvent(
                t=0.0,
                kind="submit",
                order_id=oid,
                price=int(round(sim.level_to_price(o.level) / tick)),
                size=1,
                side=1 if o.side == "buy" else -1,
            ),
            None,
        )
    for _ in range(horizon):
        before = dict(sim._orders)
        n_tr = len(sim.trades)
        sim.step()
        for oid, o in sim._orders.items():
            if oid not in before:
                yield (
                    LobEvent(
                        t=float(sim.t),
                        kind="submit",
                        order_id=oid,
                        price=int(round(sim.level_to_price(o.level) / tick)),
                        size=1,
                        side=1 if o.side == "buy" else -1,
                    ),
                    None,
                )
        tr_by_maker = {tr.maker_order_id: tr for tr in sim.trades[n_tr:]}
        for oid in before.keys() - set(sim._orders):
            tr = tr_by_maker.get(oid)
            if tr is not None:
                yield (
                    LobEvent(
                        t=float(tr.t),
                        kind="exec",
                        order_id=oid,
                        price=int(round(tr.price / tick)),
                        size=int(tr.qty),
                        side=1 if tr.maker_side == "buy" else -1,
                    ),
                    None,
                )
            else:
                o = before[oid]
                yield (
                    LobEvent(
                        t=float(sim.t),
                        kind="delete",
                        order_id=oid,
                        price=int(round(sim.level_to_price(o.level) / tick)),
                        size=1,
                        side=1 if o.side == "buy" else -1,
                    ),
                    None,
                )


def check_book_trace(
    stream: Iterable[tuple[LobEvent, tuple[int, int] | None]],
) -> dict[str, Any]:
    """Run the invariant spec over (event, official touch) pairs.

    When ``touch`` is not None it is the post-event official top of
    book; state-sensitive checks (I7/I8) use the *previous* touch as
    the pre-event state and I2 is checked on official rows. Otherwise
    the event-replayed book is the state oracle.
    """
    violations: Counter[str] = Counter()
    examples: dict[str, Any] = {}
    orders: dict[int, list[int]] = {}  # oid -> [price, remaining]
    depth: dict[int, dict[int, int]] = {1: {}, -1: {}}
    prev_t: float | None = None
    prev_touch: tuple[int, int] | None = None
    n = 0

    def note(inv: str, idx: int, detail: str) -> None:
        violations[inv] += 1
        examples.setdefault(inv, f"event {idx}: {detail}")

    for ev, touch in stream:
        n += 1
        idx = n - 1
        if prev_t is not None and ev.t < prev_t:
            note("I6_time_monotone", idx, f"{ev.t} < {prev_t}")
        prev_t = ev.t
        if ev.size <= 0:
            note("I1_nonpositive_size", idx, f"size={ev.size}")
        if ev.kind == "submit":
            if ev.order_id in orders:
                note("I4_duplicate_submit", idx, f"oid={ev.order_id}")
            else:
                orders[ev.order_id] = [ev.price, ev.size]
                lv = depth[ev.side]
                lv[ev.price] = lv.get(ev.price, 0) + ev.size
                if prev_touch is not None:
                    opp_best = prev_touch[0] if ev.side == 1 else prev_touch[1]
                    crossed = ev.price >= opp_best if ev.side == 1 else ev.price <= opp_best
                    if crossed:
                        note("I8_submit_crossed", idx, f"px {ev.price} vs opp {opp_best}")
                elif depth[-ev.side]:
                    other = depth[-ev.side]
                    if ev.side == 1 and ev.price >= min(other):
                        note("I8_submit_crossed", idx, f"bid {ev.price} >= ask {min(other)}")
                    if ev.side == -1 and ev.price <= max(other):
                        note("I8_submit_crossed", idx, f"ask {ev.price} <= bid {max(other)}")
        elif ev.kind in ("exec", "hidden_exec", "cancel"):
            rec = orders.get(ev.order_id)
            if rec is None:
                note(f"I4_{ev.kind}_unknown_id", idx, f"oid={ev.order_id}")
            else:
                if ev.kind == "exec" and ev.price != rec[0]:
                    note("I5_exec_price_mismatch", idx, f"px {ev.price} != order {rec[0]}")
                if ev.kind == "exec":
                    pre_best: int | None
                    if prev_touch is not None:
                        pre_best = prev_touch[1] if ev.side == 1 else prev_touch[0]
                    else:
                        side_lv = depth[ev.side]
                        pre_best = (
                            (max(side_lv) if ev.side == 1 else min(side_lv)) if side_lv else None
                        )
                    if pre_best is not None and rec[0] != pre_best:
                        note("I7_exec_behind_best", idx, f"exec px {rec[0]}, best {pre_best}")
                if ev.size > rec[1]:
                    note("I3_size_overdraw", idx, f"{ev.size} > remaining {rec[1]}")
                    rec[1] = 0
                else:
                    rec[1] -= ev.size
                lv = depth[ev.side]
                lv[rec[0]] = lv.get(rec[0], 0) - ev.size
                if lv.get(rec[0], 0) <= 0:
                    if lv.get(rec[0], 0) < 0:
                        note("I1_negative_level", idx, f"level {rec[0]} -> {lv[rec[0]]}")
                    lv.pop(rec[0], None)
                if rec[1] <= 0:
                    orders.pop(ev.order_id)
        elif ev.kind == "delete":
            rec = orders.pop(ev.order_id, None)
            if rec is None:
                note("I4_delete_unknown_id", idx, f"oid={ev.order_id}")
            else:
                lv = depth[ev.side]
                lv[rec[0]] = lv.get(rec[0], 0) - rec[1]
                if lv.get(rec[0], 0) <= 0:
                    if lv.get(rec[0], 0) < 0:
                        note("I1_negative_level", idx, f"level {rec[0]} -> {lv[rec[0]]}")
                    lv.pop(rec[0], None)
        if touch is not None:
            if touch[1] >= touch[0]:
                note("I2_crossed_book", idx, f"official bid {touch[1]} >= ask {touch[0]}")
        elif depth[1] and depth[-1] and max(depth[1]) >= min(depth[-1]):
            note("I2_crossed_book", idx, f"bid {max(depth[1])} >= ask {min(depth[-1])}")
        if touch is not None:
            prev_touch = touch
    return {
        "n_events": n,
        "n_violations": int(sum(violations.values())),
        "violation_counts": dict(violations),
        "violation_rates": {k: v / max(n, 1) for k, v in violations.items()},
        "examples": examples,
        "n_open_orders": len(orders),
        "ok": True,
    }


def replay_crossed_rate(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    """Diagnostic: how often the pure event-replay book (no resync) is
    crossed vs the official rows. Measures replay drift, not tape truth."""
    n = 0
    crossed = 0
    bids_: dict[int, int] = {}
    asks_: dict[int, int] = {}
    orders: dict[int, tuple[int, int]] = {}
    for ev in parse_messages(msg_path):
        n += 1
        book = bids_ if ev.direction == 1 else asks_
        if ev.event_type == SUBMISSION:
            orders[ev.order_id] = (ev.price, ev.size)
            book[ev.price] = book.get(ev.price, 0) + ev.size
        elif ev.event_type in (EXECUTION, EXECUTION_HIDDEN, CANCEL_PARTIAL):
            rec = orders.get(ev.order_id)
            if rec is not None:
                rem = rec[1] - ev.size
                book[rec[0]] = book.get(rec[0], 0) - ev.size
                if book.get(rec[0], 0) <= 0:
                    book.pop(rec[0], None)
                if rem <= 0:
                    orders.pop(ev.order_id)
                else:
                    orders[ev.order_id] = (rec[0], rem)
        elif ev.event_type == DELETE:
            rec = orders.pop(ev.order_id, None)
            if rec is not None:
                book[rec[0]] = book.get(rec[0], 0) - rec[1]
                if book.get(rec[0], 0) <= 0:
                    book.pop(rec[0], None)
        if bids_ and asks_ and max(bids_) >= min(asks_):
            crossed += 1
    return {"n_events": n, "replay_crossed_events": crossed}


def csv_crossed_rows(ob_path: Path) -> dict[str, Any]:
    """I2 checked on the official orderbook rows (not our rebuild)."""
    n = 0
    crossed = 0
    with ob_path.open(newline="") as fo:
        for row in csv.reader(fo):
            asks, bids = parse_orderbook_row(row)
            if asks and bids:
                n += 1
                if bids[0][0] >= asks[0][0]:
                    crossed += 1
    return {"n_rows": n, "crossed_rows": crossed}


def lob_invariants_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = check_book_trace(lobster_events(msg_path, ob_path))
    real["csv"] = csv_crossed_rows(ob_path)
    real["replay_diag"] = replay_crossed_rate(msg_path, ob_path)
    arms: dict[str, Any] = {}
    for name, flow, s in (
        ("iid", None, seed),
        (
            "regime",
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed + 1,
        ),
        (
            "split",
            SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed + 2,
        ),
    ):
        cfg = ZILobConfig(seed=s)
        sim = ZILobSimulator(cfg, flow=flow)
        arms[name] = check_book_trace(sim_events(sim, cfg.tick, 20000))
    real_rates = real["violation_rates"]
    real_shadow = sum(v for k, v in real_rates.items() if k.startswith("I4"))
    divergences: list[str] = []
    for name, arm in arms.items():
        for inv in ("I7_exec_behind_best", "I2_crossed_book", "I8_submit_crossed"):
            r = real_rates.get(inv, 0.0)
            a = arm["violation_rates"].get(inv, 0.0)
            if abs(a - r) > 0.005:
                divergences.append(f"{name}_{inv}_{a:.4f}_vs_{r:.4f}")
        arm_shadow = sum(v for k, v in arm["violation_rates"].items() if k.startswith("I4"))
        if abs(real_shadow - arm_shadow) > 0.005:
            divergences.append(f"{name}_shadow_book_{arm_shadow:.4f}_vs_{real_shadow:.4f}")
        sim_structural = sum(
            v for k, v in arm["violation_counts"].items() if k.startswith(("I1", "I3", "I5"))
        )
        if sim_structural > 0:
            divergences.append(f"{name}_STRUCTURAL_VIOLATIONS_{sim_structural}")
    payload: dict[str, Any] = {
        "kind": "lob_invariants",
        "schema": "lob_invariants.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "matching_engine_invariants_checked_real_and_sim",
        "interpretation": (
            "Executable spec I1-I8 applied to identical event semantics "
            "on both engines. Real-side state checks (I7 exec-at-best, "
            "I8 submit-crossed, I2 uncrossed-at-rest) read the OFFICIAL "
            "orderbook rows as the state oracle; event bookkeeping "
            "checks (I3 conservation, I4 id validity, I5 exec price, "
            "I6 monotonic time) run on the message stream itself. "
            "replay_diag measures how crossed a bare event-replay book "
            "is without band resync — the drift the lobster replay "
            "module's resync absorbs. On the sim the replayed book is "
            "the truth, so ANY structural violation (I1/I3/I4/I5) is an "
            "engine bug, not a measurement gap."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
