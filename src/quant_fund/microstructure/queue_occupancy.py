"""queue_occupancy — share-of-queue and position-in-queue conditioning.

For a resting limit order, where it sits in its price-level FIFO queue
decides whether it trades or is cancelled. This lane reconstructs
per-(side, price) arrival-ordered queues from the LOBSTER message
stream and measures, over every in-window resting order:

  a. fill probability by share-of-queue decile (order size / level size
     at submission),
  b. fill probability by queue-position-rank bucket (entries ahead,
     0 = front),
  c. queue-ahead burn attribution for filled orders: of the shares that
     stood ahead, how much departed via executions vs cancels — does
     the front of the queue die by trading or by fleeing,
  d. median dwell at rank 0 before the order's own exit.

Reconstruction rules:

  - SUBMISSION appends to its (side, price) queue; rank / shares-ahead /
    share-of-queue are recorded at that instant.
  - DELETE removes the addressed order; CANCEL_PARTIAL shrinks it in
    place and the order keeps its position.
  - EXECUTION removes the addressed resting order's shares (the event's
    order id is the tape's own victim record — LOBSTER does not follow
    strict FIFO: ~40% of execs on AMZN hit victims 1+ slots deep, e.g.
    iceberg/reserve fills reported under the displayed oid). When the
    victim oid is untracked the exec consumes the ghost/front region
    instead — the FIFO-prior fallback for unidentifiable victims.
    ``victim_at_front_rate`` reports how often the tape's victim really
    was at rank 0. A parallel registry tracks per-order remaining size
    by event-addressed accounting and is compared to queue bookkeeping
    at each exit.
  - Pre-window orders (events on ids never submitted in-window) are the
    ghost population: they predate every in-window order, so they live
    in the front region. The orderbook row-0 seed installs one ghost
    block per displayed level; unknown-id events materialize further
    ghost departures lazily. Ghost blocks count as one rank slot; their
    shares count fully in shares-ahead. When a ghost is discovered
    late, recorded ahead-shares are a lower bound — honest, documented.
  - The reconstructed level totals are resynced against the orderbook
    snapshot (top-10 band): a ghost block absorbs untracked residue so
    displayed-level totals hug LOBSTER's ground truth; adjustments are
    bookkeeping, never burn events.

Sim arm: the ZI-LOB simulator exposes true FIFO per-level deques
(``sim._orders`` order registry, ``sim.trades`` maker ids at the front).
The identical recorder is fed by a per-step registry diff: new ids are
submissions, maker ids are executions, vanished-without-trade ids are
deletes. Sim sizes are unit lots (share-of-queue = 1/level-depth).
``mechanism_present`` is True — the sim's queue identity is complete.

Receipt ``queue_occupancy.v1``, data_label MIXED (real tape + sim arms).
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    SUBMISSION,
    LobsterEvent,
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

QUEUE_OCCUPANCY_SCHEMA = "queue_occupancy.v1"

# Rank buckets: entries ahead of the order at submission (0 = front).
RANK_EDGES = (0, 1, 2, 3, 6, 11, 26, 101)
RANK_LABELS = ("front", "1", "2", "3_5", "6_10", "11_25", "26_100", "101p")

_GHOST_CAUSE_EXEC = "exec"
_GHOST_CAUSE_CANCEL = "cancel"


@dataclass(eq=False)
class _QueueEntry:
    """One FIFO slot at a (side, price) level.

    ``oid=None`` marks the seeded aggregate ghost block (row-0 depth of
    pre-window orders, submitted before the tape opens so ids are
    unrecoverable). ``exec_lost`` / ``cancel_lost`` tally this entry's
    departures by cause for diagnostics.
    """

    oid: int | None
    remaining: int
    ghost: bool
    exec_lost: int = 0
    cancel_lost: int = 0
    rec: _OrderRecord | None = None


@dataclass(eq=False)
class _OrderRecord:
    """A tracked in-window submission and its queue-conditioned outcome."""

    oid: int
    direction: int
    price: int
    size: int
    t_submit: float
    rank: int
    shares_ahead: int
    ghost_ahead_shares: int
    share_of_queue: float
    entry: _QueueEntry | None = None
    t_rank0: float | None = None
    t_exit: float | None = None
    outcome: str = "open"
    exec_received: int = 0
    burn_exec: int = 0
    burn_cancel: int = 0


class QueueOccupancyRecorder:
    """Reconstructs FIFO per-level queues and per-order fate.

    Drive it with ``submit`` / ``execute`` / ``cancel_partial`` /
    ``delete`` (plus ``seed_ghost`` for pre-window depth). Burn
    accounting is direct: shares departing an entry at position ``i``
    accumulate onto every tracked entry at positions ``> i`` — position
    order IS the ahead relation and behind-entries never jump ahead,
    so no per-order snapshots are needed. Executions are id-directed
    (the tape names the victim; see module docstring).
    """

    def __init__(self) -> None:
        self._queues: dict[tuple[int, int], deque[_QueueEntry]] = {}
        self._live: dict[int, _OrderRecord] = {}
        # Event-addressed bookkeeping, parallel to queue bookkeeping:
        # remaining size per order id as the raw events declare it.
        self._registry: dict[int, int] = {}
        # Tracked (non-ghost) share totals per level, kept incrementally
        # so the per-event resync against the displayed band is O(1).
        self._trk: dict[tuple[int, int], int] = {}
        self._records: list[_OrderRecord] = []
        self._n_exec_events = 0
        self._n_exec_front_match = 0
        self._n_exec_untracked = 0
        self._n_cancel_untracked = 0
        self._n_delete_untracked = 0
        self._ghost_overrun_shares = 0
        self._exec_residual_shares = 0
        self._victim_pos_hist: dict[int, int] = {}
        self._n_exit_registry_checked = 0
        self._n_exit_registry_match = 0
        self._n_resync = 0
        self._resync_shares = 0
        self._resync_overage = 0
        self._n_evicted_silent = 0
        self._evicted_silent_shares = 0
        self._max_queue_len = 0

    # -- event intake ------------------------------------------------------

    def seed_ghost(self, direction: int, price: int, size: int) -> None:
        """Install/extend the aggregate ghost block at a level's front."""
        if size <= 0:
            return
        q = self._queues.setdefault((direction, price), deque())
        g = self._ghost(q)
        if g is None:
            g = _QueueEntry(oid=None, remaining=size, ghost=True)
            q.appendleft(g)
        else:
            g.remaining += size

    def submit(self, t: float, direction: int, price: int, oid: int, size: int) -> None:
        q = self._queues.setdefault((direction, price), deque())
        g = self._ghost(q)
        ghost_now = g.remaining if g is not None else 0
        ahead_total = self._trk.get((direction, price), 0) + ghost_now
        rec = _OrderRecord(
            oid=oid,
            direction=direction,
            price=price,
            size=size,
            t_submit=t,
            rank=len(q),
            shares_ahead=ahead_total,
            ghost_ahead_shares=ghost_now,
            share_of_queue=size / (ahead_total + size),
        )
        entry = _QueueEntry(oid=oid, remaining=size, ghost=False)
        entry.rec = rec
        rec.entry = entry
        q.append(entry)
        self._live[oid] = rec
        self._registry[oid] = size
        self._trk[(direction, price)] = self._trk.get((direction, price), 0) + size
        self._records.append(rec)
        self._max_queue_len = max(self._max_queue_len, len(q))
        if len(q) == 1:
            rec.t_rank0 = t

    def execute(self, t: float, direction: int, price: int, oid_hint: int, size: int) -> None:
        """Apply ``size`` executed shares of the addressed resting order.

        The victim is the order id the tape names. Fidelity counters
        record how often that victim sits at rank 0 (strict-FIFO rate)
        and the position histogram when it does not.
        """
        self._n_exec_events += 1
        if oid_hint in self._registry:
            self._registry[oid_hint] = max(0, self._registry[oid_hint] - size)
        rec = self._live.get(oid_hint)
        if rec is None or rec.entry is None:
            # The victim is a pre-window/hidden order we never tracked:
            # ghost-region departure, never a fake fill of a tracked order.
            self._n_exec_untracked += 1
            self._ghost_depart(t, direction, price, size, _GHOST_CAUSE_EXEC)
            return
        q = self._queues[(rec.direction, rec.price)]
        e = rec.entry
        pos = self._index(q, e)
        if pos == 0:
            self._n_exec_front_match += 1
        elif pos is not None:
            self._victim_pos_hist[pos] = self._victim_pos_hist.get(pos, 0) + 1
        take = min(size, e.remaining)
        e.remaining -= take
        e.exec_lost += take
        rec.exec_received += take
        self._trk[(rec.direction, rec.price)] -= take
        if pos is not None:
            self._burn_behind(q, pos, take, _GHOST_CAUSE_EXEC)
        residual = size - take
        if residual > 0:
            # Executed shares beyond the victim's displayed remainder —
            # reserve/iceberg slices trading under the same oid. They
            # stood at-or-ahead of the victim's slot: burn for everyone
            # still behind it.
            self._exec_residual_shares += residual
            if pos is not None:
                self._burn_behind(q, pos, residual, _GHOST_CAUSE_EXEC)
        if e.remaining <= 0:
            if pos is not None:
                del q[pos]
            self._exit(t, rec, _GHOST_CAUSE_EXEC)
            self._stamp_front(t, q)

    def cancel_partial(self, t: float, direction: int, price: int, oid: int, size: int) -> None:
        if oid in self._registry:
            self._registry[oid] = max(0, self._registry[oid] - size)
        rec = self._live.get(oid)
        if rec is None or rec.entry is None:
            self._n_cancel_untracked += 1
            self._ghost_depart(t, direction, price, size, _GHOST_CAUSE_CANCEL)
            return
        e = rec.entry
        take = min(size, e.remaining)
        e.remaining -= take
        e.cancel_lost += take
        self._trk[(rec.direction, rec.price)] -= take
        q = self._queues[(rec.direction, rec.price)]
        idx = self._index(q, e)
        if idx is not None:
            self._burn_behind(q, idx, take, _GHOST_CAUSE_CANCEL)
        if e.remaining <= 0:
            if idx is not None:
                del q[idx]
            self._exit(t, rec, _GHOST_CAUSE_CANCEL)
            self._stamp_front(t, q)

    def delete(self, t: float, direction: int, price: int, oid: int, size: int) -> None:
        rec = self._live.get(oid)
        if rec is None or rec.entry is None:
            self._n_delete_untracked += 1
            self._ghost_depart(t, direction, price, size, _GHOST_CAUSE_CANCEL)
            return
        e = rec.entry
        e.cancel_lost += e.remaining
        self._trk[(rec.direction, rec.price)] -= e.remaining
        q = self._queues[(rec.direction, rec.price)]
        idx = self._index(q, e)
        if idx is not None:
            self._burn_behind(q, idx, e.remaining, _GHOST_CAUSE_CANCEL)
            del q[idx]
        self._exit(t, rec, _GHOST_CAUSE_CANCEL)
        self._stamp_front(t, q)

    def resync_level(self, t: float, direction: int, price: int, target_total: int) -> None:
        """Drag a displayed level's total to LOBSTER's ground truth.

        Ghost depth absorbs the silent-ITCH residue: ghost remaining is
        set to ``target - tracked_sum``. When tracked orders alone exceed
        the target the difference is drift we cannot honestly undo —
        counted, never invented into cancellations.
        """
        q = self._queues.get((direction, price))
        tracked = self._trk.get((direction, price), 0)
        if q is None:
            if target_total > 0:
                self.seed_ghost(direction, price, target_total)
                self._n_resync += 1
                self._resync_shares += target_total
            return
        g = self._ghost(q)
        ghost_now = g.remaining if g is not None else 0
        want = max(0, target_total - tracked)
        # Silent departures/adjustments are bookkeeping corrections, never
        # burn events — their cause (cancel vs exec) is unobservable, so
        # counting them would bias the exec-vs-cancel attribution.
        if want != ghost_now:
            self._resync_shares += abs(want - ghost_now)
            self._n_resync += 1
        if want == 0 and g is not None:
            q.popleft()
            self._stamp_front(t, q)
        elif want > 0:
            if g is None:
                q.appendleft(_QueueEntry(oid=None, remaining=want, ghost=True))
            else:
                g.remaining = want
        if target_total < tracked:
            # LOBSTER's book dropped shares our message stream never
            # exited — silent removals are a known tape quirk (see
            # lobster.py). Retire the surplus from the front: the
            # stalest orders are the departed ones empirically.
            surplus = tracked - target_total
            self._resync_overage += surplus
            self._evict_silent(t, direction, price, surplus)

    def level_tracked_total(self, direction: int, price: int) -> int:
        return self._trk.get((direction, price), 0)

    # -- internals ---------------------------------------------------------

    def _evict_silent(self, t: float, direction: int, price: int, surplus: int) -> None:
        """Retire ``surplus`` shares of tracked depth, front of queue first.

        Departures LOBSTER's orderbook records but the message file omits:
        exit outcome ``evicted_silent`` so the order counts in neither
        fill nor cancel statistics. Not a burn event for orders behind —
        the cause is unobservable.
        """
        key = (direction, price)
        q = self._queues[key]
        left = surplus
        while left > 0 and q:
            e = q[0]
            take = min(e.remaining, left)
            e.remaining -= take
            left -= take
            if not e.ghost:
                self._trk[key] -= take
                self._evicted_silent_shares += take
            if e.remaining <= 0:
                q.popleft()
                if e.rec is not None:
                    rec = e.rec
                    rec.t_exit = t
                    rec.outcome = "evicted_silent"
                    self._n_evicted_silent += 1
                    self._live.pop(rec.oid, None)
                    self._registry.pop(rec.oid, None)
        self._stamp_front(t, q)

    @staticmethod
    def _ghost(q: deque[_QueueEntry]) -> _QueueEntry | None:
        return q[0] if q and q[0].ghost else None

    @staticmethod
    def _index(q: deque[_QueueEntry], e: _QueueEntry) -> int | None:
        for i, x in enumerate(q):
            if x is e:
                return i
        return None

    @staticmethod
    def _burn_behind(q: deque[_QueueEntry], idx: int, shares: int, cause: str) -> None:
        if shares <= 0:
            return
        for j in range(idx + 1, len(q)):
            r = q[j].rec
            if r is None:
                continue
            if cause == _GHOST_CAUSE_EXEC:
                r.burn_exec += shares
            else:
                r.burn_cancel += shares

    def _ghost_depart(self, t: float, direction: int, price: int, size: int, cause: str) -> None:
        """Departure shares of an untracked (pre-window) order at a level.

        Ghosts occupy the front region: shares leaving them stood ahead
        of every live tracked order at that level. With no ghost block
        recorded the departures are attributed directly to burn counters;
        with one, the block's own size shrinks (overrun counted).
        """
        q = self._queues.get((direction, price))
        if q is None:
            return
        g = self._ghost(q)
        if g is not None:
            take = min(size, g.remaining)
            self._ghost_overrun_shares += size - take
            if cause == _GHOST_CAUSE_EXEC:
                g.exec_lost += take
            else:
                g.cancel_lost += take
            # The full departed size stood ahead of every live tracked
            # order at the level, even when our ghost estimate ran low.
            self._burn_behind(q, 0, size, cause)
            g.remaining -= take
            if g.remaining <= 0:
                q.popleft()
                self._stamp_front(t, q)
        else:
            # Materialize-and-depart: shares were ahead of all tracked
            # orders here; attribute burn to every live one at the level.
            for e in q:
                if e.rec is not None:
                    if cause == _GHOST_CAUSE_EXEC:
                        e.rec.burn_exec += size
                    else:
                        e.rec.burn_cancel += size

    def _exit(self, t: float, rec: _OrderRecord, cause: str) -> None:
        rec.t_exit = t
        if cause == _GHOST_CAUSE_EXEC or rec.exec_received >= rec.size:
            rec.outcome = "filled"
        elif rec.exec_received > 0:
            rec.outcome = "partial_filled"
        else:
            rec.outcome = "deleted"
        self._live.pop(rec.oid, None)
        reg_rem = self._registry.pop(rec.oid, None)
        if reg_rem is not None and rec.entry is not None:
            self._n_exit_registry_checked += 1
            if reg_rem == rec.entry.remaining:
                self._n_exit_registry_match += 1

    @staticmethod
    def _stamp_front(t: float, q: deque[_QueueEntry]) -> None:
        if q and q[0].rec is not None and q[0].rec.t_rank0 is None:
            q[0].rec.t_rank0 = t

    # -- aggregates --------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        recs = self._records
        n = len(recs)
        n_filled = sum(1 for r in recs if r.outcome == "filled")
        n_partial = sum(1 for r in recs if r.outcome == "partial_filled")
        n_deleted = sum(1 for r in recs if r.outcome == "deleted")
        n_evicted = sum(1 for r in recs if r.outcome == "evicted_silent")
        n_open = n - n_filled - n_partial - n_deleted - n_evicted

        def fill_rate(sel: list[_OrderRecord]) -> float:
            return float(np.mean([r.outcome == "filled" for r in sel])) if sel else 0.0

        shares = np.asarray([r.share_of_queue for r in recs])
        share_deciles: dict[str, Any] = {}
        if n >= 20:
            edges = np.quantile(shares, np.linspace(0.0, 1.0, 11))
            for i in range(10):
                lo, hi = float(edges[i]), float(edges[i + 1])
                if i == 9:
                    sel = [r for r in recs if lo <= r.share_of_queue <= hi]
                else:
                    sel = [r for r in recs if lo <= r.share_of_queue < hi]
                share_deciles[f"d{i}"] = {
                    "lo": round(lo, 6),
                    "hi": round(hi, 6),
                    "n": len(sel),
                    "fill_rate": fill_rate(sel),
                }

        rank_buckets: dict[str, Any] = {}
        for i, (lo, label) in enumerate(zip(RANK_EDGES, RANK_LABELS, strict=True)):
            hi_rank = RANK_EDGES[i + 1] if i + 1 < len(RANK_EDGES) else None
            sel = [r for r in recs if lo <= r.rank and (hi_rank is None or r.rank < hi_rank)]
            rank_buckets[label] = {
                "n": len(sel),
                "fill_rate": fill_rate(sel),
                "median_shares_ahead": float(np.median([r.shares_ahead for r in sel]))
                if sel
                else None,
            }

        filled_with_ahead = [r for r in recs if r.outcome == "filled" and r.shares_ahead > 0]
        burn_exec = sum(r.burn_exec for r in filled_with_ahead)
        burn_cancel = sum(r.burn_cancel for r in filled_with_ahead)
        burn_total = burn_exec + burn_cancel
        per_order_share = [
            r.burn_exec / (r.burn_exec + r.burn_cancel)
            for r in filled_with_ahead
            if r.burn_exec + r.burn_cancel > 0
        ]
        # How much of what stood ahead of a filled order did we observe
        # departing? The complement is shares still standing at its fill
        # (mid-queue victims trade with the queue ahead intact) plus
        # resync-absorbed / late-ghost depth.
        burn_coverage = [(r.burn_exec + r.burn_cancel) / r.shares_ahead for r in filled_with_ahead]

        dwell = [
            r.t_exit - r.t_rank0 for r in recs if r.t_rank0 is not None and r.t_exit is not None
        ]
        dwell_filled = [
            r.t_exit - r.t_rank0
            for r in recs
            if r.t_rank0 is not None and r.t_exit is not None and r.outcome == "filled"
        ]
        dwell_deleted = [
            r.t_exit - r.t_rank0
            for r in recs
            if r.t_rank0 is not None and r.t_exit is not None and r.outcome == "deleted"
        ]

        def _q(vals: list[float]) -> dict[str, Any]:
            if not vals:
                return {"n": 0}
            arr = np.asarray(vals)
            return {
                "n": int(arr.size),
                "median": round(float(np.median(arr)), 6),
                "p10": round(float(np.percentile(arr, 10)), 6),
                "p90": round(float(np.percentile(arr, 90)), 6),
            }

        n_tracked_exec_exits = sum(1 for r in recs if r.outcome == "filled" and r.exec_received > 0)
        return {
            "n_orders": n,
            "n_filled": n_filled,
            "n_partial_filled": n_partial,
            "n_deleted": n_deleted,
            "n_evicted_silent": n_evicted,
            "n_open_at_end": n_open,
            "fill_rate": round(n_filled / max(1, n), 6),
            "fill_or_partial_rate": round((n_filled + n_partial) / max(1, n), 6),
            "share_deciles": share_deciles,
            "rank_buckets": rank_buckets,
            "queue_ahead_burn": {
                "n_filled_with_ahead": len(filled_with_ahead),
                "exec_burn_shares": burn_exec,
                "cancel_burn_shares": burn_cancel,
                "exec_burn_share": round(burn_exec / burn_total, 6) if burn_total > 0 else None,
                "median_per_order_exec_share": round(float(np.median(per_order_share)), 6)
                if per_order_share
                else None,
                "median_burn_coverage": round(float(np.median(burn_coverage)), 6)
                if burn_coverage
                else None,
                "still_standing_shares": sum(
                    r.shares_ahead - r.burn_exec - r.burn_cancel for r in filled_with_ahead
                ),
            },
            "rank0_dwell_s": {
                "all_exits": _q(dwell),
                "filled_exits": _q(dwell_filled),
                "deleted_exits": _q(dwell_deleted),
                "n_reached_front": sum(1 for r in recs if r.t_rank0 is not None),
            },
            "median_rank_at_submit": {
                "filled": float(np.median([r.rank for r in recs if r.outcome == "filled"]))
                if n_filled
                else None,
                "deleted": float(np.median([r.rank for r in recs if r.outcome == "deleted"]))
                if n_deleted
                else None,
                "open": float(np.median([r.rank for r in recs if r.outcome == "open"]))
                if n_open
                else None,
            },
            "median_share_at_submit": {
                "filled": round(
                    float(np.median([r.share_of_queue for r in recs if r.outcome == "filled"])),
                    6,
                )
                if n_filled
                else None,
                "deleted": round(
                    float(np.median([r.share_of_queue for r in recs if r.outcome == "deleted"])),
                    6,
                )
                if n_deleted
                else None,
            },
            "fidelity": {
                "n_exec_events": self._n_exec_events,
                "victim_at_front_rate": round(
                    self._n_exec_front_match / max(1, self._n_exec_events), 6
                ),
                "victim_position_hist": {
                    str(k): v for k, v in sorted(self._victim_pos_hist.items())
                },
                "n_exec_untracked_oid": self._n_exec_untracked,
                "n_cancel_untracked_oid": self._n_cancel_untracked,
                "n_delete_untracked_oid": self._n_delete_untracked,
                "ghost_overrun_shares": self._ghost_overrun_shares,
                "exec_residual_shares": self._exec_residual_shares,
                "n_resync_events": self._n_resync,
                "resync_shares_adjusted": self._resync_shares,
                "resync_tracked_overage": self._resync_overage,
                "n_evicted_silent_orders": self._n_evicted_silent,
                "evicted_silent_shares": self._evicted_silent_shares,
                "registry_exit_match_rate": round(
                    self._n_exit_registry_match / max(1, self._n_exit_registry_checked),
                    6,
                ),
                "max_queue_len_entries": self._max_queue_len,
                "n_tracked_exec_exits": n_tracked_exec_exits,
            },
            "mechanism_present": True,
        }


def lobster_queue_occupancy(msg_path: Path, ob_path: Path, n_levels: int = 10) -> dict[str, Any]:
    """Run the queue-occupancy measurement on the real LOBSTER tape.

    Orderbook row ``i`` is the book state after message ``i`` (verified
    empirically on this tape: level changes land on the same row index
    as their message). Message 1 is applied first, then row 0's
    displayed levels resync — everything the snapshot shows that the
    stream has not submitted becomes ghost depth for pre-window orders.
    Every later event resyncs against its own row, so silent book
    updates (a known LOBSTER quirk) are absorbed by the ghost region
    instead of corrupting the reconstruction.
    """
    rec = QueueOccupancyRecorder()
    with ob_path.open(newline="") as f_ob:
        ob_rows = csv.reader(f_ob)
        msgs = iter(parse_messages(msg_path))
        try:
            seed_row = next(ob_rows)
            ev0 = next(msgs)
        except StopIteration:  # pragma: no cover - empty tape guard
            return {"ok": False, "reason": "empty_orderbook"}
        _apply(rec, ev0)
        asks0, bids0 = parse_orderbook_row(seed_row)
        for px, sz in asks0[:n_levels]:
            rec.resync_level(ev0.time_s, -1, px, sz)
        for px, sz in bids0[:n_levels]:
            rec.resync_level(ev0.time_s, 1, px, sz)
        # The lead means zip now pairs message k with orderbook row
        # k-1 — the state right after that message. Resync each event
        # against its own row's displayed band.
        for ev, ob_row in zip(msgs, ob_rows, strict=True):
            _apply(rec, ev)
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            for px, sz in asks_exp[:n_levels]:
                rec.resync_level(ev.time_s, -1, px, sz)
            for px, sz in bids_exp[:n_levels]:
                rec.resync_level(ev.time_s, 1, px, sz)
    out = rec.finalize()
    out["ok"] = out["n_orders"] >= 100
    return out


def _apply(rec: QueueOccupancyRecorder, ev: LobsterEvent) -> None:
    if ev.event_type == SUBMISSION:
        rec.submit(ev.time_s, ev.direction, ev.price, ev.order_id, ev.size)
    elif ev.event_type == CANCEL_PARTIAL:
        rec.cancel_partial(ev.time_s, ev.direction, ev.price, ev.order_id, ev.size)
    elif ev.event_type == DELETE:
        rec.delete(ev.time_s, ev.direction, ev.price, ev.order_id, ev.size)
    elif ev.event_type == EXECUTION:
        rec.execute(ev.time_s, ev.direction, ev.price, ev.order_id, ev.size)
    # EXECUTION_HIDDEN / HALT / other: no visible-queue effect


def sim_queue_occupancy(
    flow: MarkovRegimeFlow | SplitFlow | None = None,
    *,
    seed: int = 0,
    horizon: int = 30000,
) -> dict[str, Any]:
    """Identical measurement driven off the ZI-LOB sim's order registry.

    Each step emits at most one book event; the registry diff before/after
    gives submissions (new ids), executions (``sim.trades`` maker ids —
    the sim consumes the level front, matching the tape rule) and
    deletes (vanished without a trade). The sim's seeded book installs
    ghost blocks so seed-order deaths classify like pre-window tape
    orders. Sizes are unit lots.
    """
    # SplitFlow implements the flow duck-type (current()/advance()) without
    # subclassing MarkovRegimeFlow; the simulator only consumes that surface.
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=cast(MarkovRegimeFlow, flow))
    rec = QueueOccupancyRecorder()
    for level, dq in sim._bids.items():
        rec.seed_ghost(1, level, len(dq))
    for level, dq in sim._asks.items():
        rec.seed_ghost(-1, level, len(dq))
    n_trades_seen = 0
    filled_ever: set[int] = set()
    n_ahead_check = 0
    n_ahead_match = 0
    for _ in range(horizon):
        before = {oid: (o.side, o.level) for oid, o in sim._orders.items()}
        sim.step()
        for oid, o in sim._orders.items():
            if oid not in before:
                rec.submit(sim.t, 1 if o.side == "buy" else -1, o.level, oid, 1)
                # Ground-truth cross-check: the sim's own queue_ahead
                # count must equal our reconstructed shares ahead
                # (unit lots => one share per queue slot).
                n_ahead_check += 1
                if rec._records[-1].shares_ahead == o.queue_ahead:
                    n_ahead_match += 1
        for tr in sim.trades[n_trades_seen:]:
            filled_ever.add(tr.maker_order_id)
            rec.execute(
                tr.t,
                1 if tr.maker_side == "buy" else -1,
                tr.level,
                tr.maker_order_id,
                tr.qty,
            )
        n_trades_seen = len(sim.trades)
        sim.trades.clear()
        n_trades_seen = 0
        after = set(sim._orders)
        for oid in before.keys() - after:
            if oid in filled_ever:
                continue
            side, level = before[oid]
            rec.delete(sim.t, 1 if side == "buy" else -1, level, oid, 1)
    out = rec.finalize()
    out["ok"] = out["n_orders"] >= 100
    out["n_events"] = sim.n_events
    out["n_trades"] = len(filled_ever)
    out["fidelity"]["shares_ahead_selfcheck_rate"] = round(n_ahead_match / max(1, n_ahead_check), 6)
    return out


def queue_occupancy_bench(
    tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7, horizon: int = 30000
) -> dict[str, Any]:
    """Queue-occupancy measurement, real tape + ZI-LOB sim arms. Sealed."""
    msg_path = next(tape_dir.glob(f"{ticker}_*_message_*.csv"), None)
    ob_path = next(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"), None)
    if msg_path is None or ob_path is None:
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_queue_occupancy(msg_path, ob_path)
    arms = {
        "iid": sim_queue_occupancy(seed=seed, horizon=horizon),
        "regime": sim_queue_occupancy(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
            horizon=horizon,
        ),
        "split": sim_queue_occupancy(
            SplitFlow(
                p_start=0.10,
                size_tail=1.2,
                k_min=10,
                k_max=600,
                intensity_mult=3.0,
                seed=seed + 2,
            ),
            seed=seed + 2,
            horizon=horizon,
        ),
    }
    divergences: list[str] = []
    real_burn = real.get("queue_ahead_burn", {}).get("exec_burn_share")
    for name, arm in arms.items():
        arm_burn = arm.get("queue_ahead_burn", {}).get("exec_burn_share")
        if real_burn is not None and arm_burn is not None and abs(arm_burn - real_burn) > 0.15:
            divergences.append(f"{name}_execburn_{arm_burn:.3f}_vs_{real_burn:.3f}")
        real_front = real.get("rank_buckets", {}).get("front", {}).get("fill_rate")
        arm_front = arm.get("rank_buckets", {}).get("front", {}).get("fill_rate")
        real_back = real.get("rank_buckets", {}).get("101p", {}).get("fill_rate")
        arm_back = arm.get("rank_buckets", {}).get("101p", {}).get("fill_rate")
        if (
            real_front is not None
            and arm_front is not None
            and real_back is not None
            and arm_back is not None
        ):
            real_grad = real_front - real_back
            arm_grad = arm_front - arm_back
            if real_grad > 0.05 and arm_grad < 0.5 * real_grad:
                divergences.append(f"{name}_rankgrad_{arm_grad:.3f}_vs_{real_grad:.3f}")
    payload: dict[str, Any] = {
        "kind": "queue_occupancy",
        "schema": QUEUE_OCCUPANCY_SCHEMA,
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "queue_position_and_share_condition_fills",
        "interpretation": (
            "Per-(side, price) FIFO queues rebuilt from the message "
            "stream: submissions append, deletes remove, partial cancels "
            "shrink in place, executions decrement the named resting "
            "order — the tape's own victim record; LOBSTER deviates from "
            "strict FIFO (victim_at_front_rate < 1: iceberg/reserve fills "
            "reported under a displayed oid, same-ms sweeps), so the "
            "addressed order is authoritative and front-consumption is "
            "the fallback for untracked victims. Ghost depth — row-0 "
            "seed plus lazily materialized untracked-id events — covers "
            "pre-window orders ahead of every in-window order; displayed "
            "levels resync to the orderbook snapshot so silent ITCH "
            "updates land on the ghost region (bookkeeping, never burn). "
            "Burn attribution counts, per filled order, shares that "
            "departed ahead of it via executions vs cancels; rank-0 "
            "dwell is exit-time minus first-front time. Sim arms drive "
            "the identical recorder off the ZI-LOB order registry (unit "
            "lots, so share = 1/level-depth)."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
