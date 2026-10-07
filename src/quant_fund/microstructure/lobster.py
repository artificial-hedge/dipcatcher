"""lobster — LOBSTER message-file replay + reconstruction validation.

LOBSTER (limitorderbook.com) publishes the NASDAQ TotalView-ITCH
derived event tapes used across the microstructure literature
(Cont, Bouchaud, Huang, ...). This lane:

1. parses the (message, orderbook) CSV pair,
2. replays the message stream through our own book engine,
3. compares the reconstructed top-10 book against LOBSTER's own
   orderbook snapshot after *every* event, and
4. measures real-tape microstructure stats on the genuine tape.

Known data caveat — the orderbook reflects the full ITCH feed
(including event types the message file omits), so a level pushed
below the displayed band can be silently removed. The validator
therefore counts explicit resyncs of the visible band rather than
pretending the message file is complete; the headline claim is a
resync rate, not a perfect match.

The tape itself is not committed (license/size); the bench takes a
tape dir and records `tape_sha256`. Receipt `lobster_replay.v1`,
data_label records the real tape — NOT synthetic.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LOBSTER_SCHEMA = "lobster_replay.v1"

# LOBSTER event types
SUBMISSION = 1
CANCEL_PARTIAL = 2
DELETE = 3
EXECUTION = 4
EXECUTION_HIDDEN = 5
HALT = 7


@dataclass(frozen=True)
class LobsterEvent:
    time_s: float
    event_type: int
    order_id: int
    size: int
    price: int  # price * 10_000
    direction: int  # +1 buy-side order, -1 sell-side order


def parse_messages(path: Path) -> Iterator[LobsterEvent]:
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            yield LobsterEvent(
                time_s=float(row[0]),
                event_type=int(row[1]),
                order_id=int(row[2]),
                size=int(row[3]),
                price=int(row[4]),
                direction=int(row[5]),
            )


def parse_orderbook_row(row: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """LOBSTER orderbook CSV row → (asks, bids) top-N (price, size)."""
    vals = [float(x) for x in row]
    asks, bids = [], []
    for lvl in range(len(vals) // 4):
        ap, asz, bp, bsz = vals[4 * lvl : 4 * lvl + 4]
        if asz > 0:
            asks.append((int(ap), int(asz)))
        if bsz > 0:
            bids.append((int(bp), int(bsz)))
    return asks, bids


class LobsterBook:
    """Level book (price→size) + order registry (id→remaining).

    For ~0.01% of type-3 deletes the size field differs from the
    order's true remaining size, so a pure level book leaves phantom
    residue at that price forever. The registry tracks in-window
    submissions so deletes remove the exact remaining size; deletes of
    pre-window orders (untracked) fall back to the message size.
    The opening book is seeded from the first orderbook snapshot row.
    """

    def __init__(self) -> None:
        self.bid: dict[int, int] = {}
        self.ask: dict[int, int] = {}
        self.orders: dict[int, tuple[int, int]] = {}  # id -> (price, remaining)

    def _side(self, direction: int) -> dict[int, int]:
        return self.bid if direction == 1 else self.ask

    def seed(self, asks: list[tuple[int, int]], bids: list[tuple[int, int]]) -> None:
        for p, s in asks:
            if s > 0:
                self.ask[p] = s
        for p, s in bids:
            if s > 0:
                self.bid[p] = s

    def apply(self, ev: LobsterEvent) -> None:
        side = self._side(ev.direction)
        if ev.event_type == SUBMISSION:
            self.orders[ev.order_id] = (ev.price, ev.size)
            side[ev.price] = side.get(ev.price, 0) + ev.size
        elif ev.event_type == CANCEL_PARTIAL:
            self._subtract_raw(side, ev.price, ev.size)
            self._reduce_order(ev.order_id, ev.size)
        elif ev.event_type == DELETE:
            if ev.order_id in self.orders:
                px, rem = self.orders.pop(ev.order_id)
                self._subtract_raw(side, px, rem)
            else:
                self._subtract_raw(side, ev.price, ev.size)
        elif ev.event_type == EXECUTION:
            self._subtract_raw(side, ev.price, ev.size)
            self._reduce_order(ev.order_id, ev.size)
        # EXECUTION_HIDDEN / HALT: no visible-book effect

    def _reduce_order(self, order_id: int, qty: int) -> None:
        rec = self.orders.get(order_id)
        if rec is None:
            return
        px, rem = rec
        rem -= min(qty, rem)
        if rem <= 0:
            del self.orders[order_id]
        else:
            self.orders[order_id] = (px, rem)

    @staticmethod
    def _subtract_raw(side: dict[int, int], price: int, qty: int) -> None:
        remain = side.get(price, 0) - qty
        if remain > 0:
            side[price] = remain
        else:
            side.pop(price, None)

    def top(self, side: str, n: int) -> list[tuple[int, int]]:
        book = self.bid if side == "bid" else self.ask
        prices = sorted(book, reverse=(side == "bid"))[:n]
        return [(p, book[p]) for p in prices]


def resync_band(
    book: LobsterBook,
    asks_exp: list[tuple[int, int]],
    bids_exp: list[tuple[int, int]],
) -> None:
    """Adopt the displayed band; keep deeper tracked levels.

    Called when the reconstructed top-N differs from LOBSTER's
    snapshot: the visible band is replaced with the expected levels
    while our deeper-than-display state is kept.
    """
    if asks_exp:
        worst_ask = max(p for p, _ in asks_exp)
        book.ask = {p: s for p, s in book.ask.items() if p > worst_ask} | dict(asks_exp)
    if bids_exp:
        worst_bid = min(p for p, _ in bids_exp)
        book.bid = {p: s for p, s in book.bid.items() if p < worst_bid} | dict(bids_exp)


def validate_reconstruction(
    message_path: Path, orderbook_path: Path, n_levels: int = 10
) -> dict[str, Any]:
    """Replay messages; compare top-n book to LOBSTER's snapshot per event.

    On divergence the visible band resyncs and `n_resync_events` counts
    the silent-book-update episodes (events LOBSTER applies from the
    full ITCH feed but omits from its message file).
    """
    book = LobsterBook()
    n_events = 0
    n_compared = 0
    n_match = 0
    n_resync = 0
    n_size_only = 0
    cur_run = 0
    divergence_lens: list[int] = []
    first_mismatch: dict[str, Any] | None = None
    seeded = False
    with orderbook_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(message_path), csv.reader(f_ob), strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if not seeded:
                # Row 0 is the book state AFTER message 0: seeding from it
                # already includes event 0 — applying it would double-count
                # the first event.
                book.seed(asks_exp, bids_exp)
                seeded = True
                n_events += 1
                continue  # row 0 consumed as seed
            book.apply(ev)
            n_events += 1
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue  # no visible-book event; snapshot repeats
            n_compared += 1
            if book.top("ask", n_levels) == asks_exp and book.top("bid", n_levels) == bids_exp:
                n_match += 1
                if cur_run > 0:
                    divergence_lens.append(cur_run)
                    cur_run = 0
            else:
                n_resync += 1
                cur_run += 1
                if [p for p, _ in book.top("ask", n_levels)] == [p for p, _ in asks_exp] and [
                    p for p, _ in book.top("bid", n_levels)
                ] == [p for p, _ in bids_exp]:
                    n_size_only += 1  # same levels, different sizes
                if first_mismatch is None:
                    first_mismatch = {
                        "event_index": n_events,
                        "expected_ask": asks_exp,
                        "actual_ask": book.top("ask", n_levels),
                        "expected_bid": bids_exp,
                        "actual_bid": book.top("bid", n_levels),
                    }
                resync_band(book, asks_exp, bids_exp)
    if cur_run > 0:
        divergence_lens.append(cur_run)
    return {
        "n_events": n_events,
        "n_compared": n_compared,
        "n_match": n_match,
        "match_rate": n_match / max(1, n_compared),
        "n_resync_events": n_resync,
        "resync_rate": n_resync / max(1, n_compared),
        "n_size_only_resyncs": n_size_only,
        "n_structural_resyncs": n_resync - n_size_only,
        "n_divergence_runs": len(divergence_lens),
        "max_divergence_len": max(divergence_lens, default=0),
        "first_mismatch": first_mismatch,
    }


def tape_measurements(
    message_path: Path, orderbook_path: Path, n_levels: int = 10
) -> dict[str, Any]:
    """Real-tape microstructure stats off the replayed stream.

    Resyncs on divergence like validate_reconstruction so stats ride on
    LOBSTER's ground truth rather than a drifting reconstruction.
    """
    book = LobsterBook()
    spreads: list[int] = []
    mids: list[float] = []
    imbalances: list[float] = []
    signs: list[int] = []
    depth_ask: list[list[int]] = []
    depth_bid: list[list[int]] = []
    seeded = False
    stride = 137  # ~2000 samples over the day
    with orderbook_path.open() as f_ob:
        for i, (ev, ob_row) in enumerate(
            zip(parse_messages(message_path), csv.reader(f_ob), strict=True)
        ):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if not seeded:
                # Row 0 is the book state AFTER message 0: seeding from it
                # already includes event 0 — applying it would double-count
                # the first event.
                book.seed(asks_exp, bids_exp)
                seeded = True
                if ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
                    signs.append(-ev.direction)
                continue
            book.apply(ev)
            if ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
                # MO prints include hidden-liquidity fills (type 5); record
                # before the no-visible-book skip so the sign stream stays
                # the full aggressor tape. Exec dir = resting side.
                signs.append(-ev.direction)
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", n_levels) != asks_exp or book.top("bid", n_levels) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
            asks, bids = book.top("ask", 1), book.top("bid", 1)
            if not asks or not bids:
                continue
            ba, bb = asks[0], bids[0]
            spreads.append(ba[0] - bb[0])
            mids.append((ba[0] + bb[0]) / 2)
            imbalances.append(ba[1] / (ba[1] + bb[1]))
            if i % stride == 0:
                depth_ask.append([s for _, s in book.top("ask", 5)])
                depth_bid.append([s for _, s in book.top("bid", 5)])
    spreads_a = np.asarray(spreads)
    signs_a = np.asarray(signs)
    lag1 = (
        float(np.dot(signs_a[:-1], signs_a[1:]) / signs_a.size)
        if signs_a.size > 2
        else float("nan")
    )
    dmid = np.diff(mids)
    imb = np.asarray(imbalances[:-1])
    if imb.size > 2 and float((dmid * dmid).sum()) > 0:
        x, y = imb - imb.mean(), dmid
        denom = float((x * x).sum())
        r2 = float((x * y).sum() ** 2 / (denom * (y * y).sum())) if denom > 0 else 0.0
    else:
        r2 = 0.0

    def _level_means(rows: list[list[int]], m: int) -> list[float]:
        if not rows:
            return []
        arr = np.zeros((len(rows), m))
        for i, r in enumerate(rows):
            arr[i, : len(r)] = r
        return [float(x) for x in arr.mean(axis=0)]

    return {
        "n_book_events": int(spreads_a.size),
        "n_executions": int(signs_a.size),
        "spread_ticks_median": float(np.median(spreads_a)),
        "spread_ticks_p95": float(np.percentile(spreads_a, 95)),
        "sign_lag1_autocorr": lag1,
        "imbalance_move_r2": r2,
        "mid_move_std": float(np.std(dmid)),
        "mean_depth_ask_l1_5": _level_means(depth_ask, 5),
        "mean_depth_bid_l1_5": _level_means(depth_bid, 5),
    }


def lobster_replay_bench(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Full lane: reconstruction validity + real-tape measurements."""
    msg = next(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = next(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    recon = validate_reconstruction(msg, ob)
    stats = tape_measurements(msg, ob)
    tape_sha = hash_bytes(msg.read_bytes())
    payload: dict[str, Any] = {
        "schema": LOBSTER_SCHEMA,
        "kind": "lobster_replay",
        "ticker": ticker,
        "tape_sha256": tape_sha,
        "tape_files": [msg.name, ob.name],
        "reconstruction": recon,
        "measurements": stats,
        "engine_valid_on_emitted": bool(
            recon["match_rate"] > 0.90 and recon["max_divergence_len"] <= 5
        ),
        "interpretation": (
            "Replays the official LOBSTER sample through our book engine "
            "and compares the top-10 book to LOBSTER's own snapshot after "
            "every event. The orderbook reflects the full ITCH feed "
            "including event types the message file omits, so "
            "resync_rate bounds the silent-update fraction; match_rate "
            "is exact agreement on emitted events"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = f"LOBSTER-{ticker}-2012-06-21-sample"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
