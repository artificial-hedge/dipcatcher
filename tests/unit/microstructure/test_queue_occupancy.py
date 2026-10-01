"""Tests for microstructure/queue_occupancy.py.

Recorder-level tests drive synthetic event streams through
``QueueOccupancyRecorder`` directly; the tape test builds a
LOBSTER-format (message, orderbook) pair where orderbook row i is the
book state after message i — the convention verified on the real tape.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from quant_fund.microstructure.lobster import LobsterBook, LobsterEvent
from quant_fund.microstructure.queue_occupancy import (
    QUEUE_OCCUPANCY_SCHEMA,
    QueueOccupancyRecorder,
    lobster_queue_occupancy,
    queue_occupancy_bench,
    sim_queue_occupancy,
)
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _rec() -> QueueOccupancyRecorder:
    return QueueOccupancyRecorder()


def _records(rec: QueueOccupancyRecorder) -> dict[int, object]:
    return {r.oid: r for r in rec._records}


def _ev(t: int, ty: int, oid: int, sz: int, px: int, d: int) -> LobsterEvent:
    return LobsterEvent(float(t), ty, oid, sz, px, d)


def _write_pair(
    msg_path: Path,
    ob_path: Path,
    events: list[LobsterEvent],
    *,
    seed_asks: list[tuple[int, int]] | None = None,
    seed_bids: list[tuple[int, int]] | None = None,
    levels: int = 2,
) -> None:
    """Write message CSV + orderbook CSV; row i = post-event-i book."""
    book = LobsterBook()
    book.seed(seed_asks or [], seed_bids or [])
    rows: list[list[str]] = []
    for ev in events:
        book.apply(ev)
        asks = book.top("ask", levels)
        bids = book.top("bid", levels)
        row: list[str] = []
        for lvl in range(levels):
            a = asks[lvl] if lvl < len(asks) else (0, 0)
            b = bids[lvl] if lvl < len(bids) else (0, 0)
            row += [str(a[0]), str(a[1]), str(b[0]), str(b[1])]
        rows.append(row)
    with msg_path.open("w", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            w.writerow([ev.time_s, ev.event_type, ev.order_id, ev.size, ev.price, ev.direction])
    with ob_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------
# recorder: submission state
# ---------------------------------------------------------------------


def test_submit_records_rank_ahead_share() -> None:
    r = _rec()
    r.submit(0.0, 1, 5000, 1, 100)
    r.submit(1.0, 1, 5000, 2, 50)
    r.submit(2.0, 1, 5000, 3, 30)
    recs = _records(r)
    a, b, c = recs[1], recs[2], recs[3]
    assert a.rank == 0 and a.shares_ahead == 0 and a.share_of_queue == 1.0
    assert a.t_rank0 == 0.0
    assert b.rank == 1 and b.shares_ahead == 100
    assert math.isclose(b.share_of_queue, 50 / 150)
    assert c.rank == 2 and c.shares_ahead == 150
    assert math.isclose(c.share_of_queue, 30 / 180)


def test_ghost_depth_counts_as_one_rank_slot() -> None:
    r = _rec()
    r.seed_ghost(1, 5000, 300)  # pre-window depth at the level
    r.submit(0.0, 1, 5000, 7, 50)
    recs = _records(r)
    assert recs[7].rank == 1  # ghost block occupies the front slot
    assert recs[7].shares_ahead == 300
    assert recs[7].ghost_ahead_shares == 300
    assert math.isclose(recs[7].share_of_queue, 50 / 350)


# ---------------------------------------------------------------------
# recorder: exits and burn attribution
# ---------------------------------------------------------------------


def test_exec_front_exit_and_burn_behind() -> None:
    r = _rec()
    r.submit(0.0, -1, 6000, 1, 100)
    r.submit(1.0, -1, 6000, 2, 50)
    r.submit(2.0, -1, 6000, 3, 30)
    r.execute(3.0, -1, 6000, 1, 100)  # front order fully executed
    recs = _records(r)
    assert recs[1].outcome == "filled" and recs[1].t_exit == 3.0
    # the 100 departed ahead-shares burn as execs for everyone behind
    assert recs[2].burn_exec == 100 and recs[2].burn_cancel == 0
    assert recs[3].burn_exec == 100
    # b promoted to front at the exec timestamp
    assert recs[2].t_rank0 == 3.0


def test_exec_mid_queue_victim_leaves_front_alive() -> None:
    """Id-directed execs: a deeper victim fills while the front stands."""
    r = _rec()
    r.submit(0.0, -1, 6000, 1, 100)
    r.submit(1.0, -1, 6000, 2, 50)
    r.execute(2.0, -1, 6000, 2, 50)  # victim is order 2 at rank 1
    recs = _records(r)
    assert recs[2].outcome == "filled"
    assert recs[1].outcome == "open"  # front order never traded
    assert recs[2].burn_exec == 0  # nothing departed ahead of it
    out = r.finalize()
    assert out["fidelity"]["victim_at_front_rate"] == 0.0
    assert out["fidelity"]["victim_position_hist"] == {"1": 1}


def test_exec_partial_then_partial_exit() -> None:
    r = _rec()
    r.submit(0.0, -1, 6000, 1, 100)
    r.execute(1.0, -1, 6000, 1, 40)  # partial fill: keeps position
    r.delete(2.0, -1, 6000, 1, 60)
    recs = _records(r)
    assert recs[1].outcome == "partial_filled"
    assert recs[1].exec_received == 40


def test_delete_removes_and_burns_cancel() -> None:
    r = _rec()
    r.submit(0.0, -1, 6000, 1, 100)
    r.submit(1.0, -1, 6000, 2, 50)
    r.delete(2.0, -1, 6000, 1, 100)
    recs = _records(r)
    assert recs[1].outcome == "deleted"
    assert recs[2].burn_cancel == 100 and recs[2].burn_exec == 0
    assert recs[2].t_rank0 == 2.0


def test_cancel_partial_shrinks_in_place() -> None:
    r = _rec()
    r.submit(0.0, -1, 6000, 1, 100)
    r.submit(1.0, -1, 6000, 2, 50)
    r.cancel_partial(2.0, -1, 6000, 1, 60)
    recs = _records(r)
    assert recs[1].outcome == "open"
    assert recs[1].entry is not None and recs[1].entry.remaining == 40
    # the 60 departed ahead-shares burn as cancels for the order behind
    assert recs[2].burn_cancel == 60
    # order keeps its slot: still ahead of order 2
    assert recs[1].rank == 0
    r.cancel_partial(3.0, -1, 6000, 1, 40)  # down to zero removes
    assert recs[1].outcome == "deleted"


def test_untracked_events_burn_ghost_then_live() -> None:
    r = _rec()
    r.seed_ghost(1, 5000, 300)
    r.submit(0.0, 1, 5000, 7, 50)
    r.delete(1.0, 1, 5000, 999, 100)  # pre-window order leaves
    recs = _records(r)
    assert recs[7].burn_cancel == 100
    # ghost 300 -> 200; execs on an unknown oid then drain it
    r.execute(2.0, 1, 5000, 888, 250)
    assert recs[7].burn_exec == 250  # full departed size stood ahead
    assert r._ghost_overrun_shares == 50  # ghost only held 200
    out = r.finalize()
    assert out["fidelity"]["n_exec_untracked_oid"] == 1
    assert out["fidelity"]["n_delete_untracked_oid"] == 1
    assert recs[7].outcome == "open"
    assert recs[7].t_rank0 == 2.0  # ghost popped -> order promoted


# ---------------------------------------------------------------------
# recorder: resync (silent book updates)
# ---------------------------------------------------------------------


def test_resync_grows_ghost_for_untracked_depth() -> None:
    r = _rec()
    r.submit(0.0, 1, 5000, 1, 50)
    r.resync_level(1.0, 1, 5000, 250)  # displayed 250, tracked 50
    r.submit(2.0, 1, 5000, 2, 50)
    recs = _records(r)
    # a ghost block of 200 materialized ahead of both live orders
    assert recs[2].rank == 2 and recs[2].shares_ahead == 250
    out = r.finalize()
    assert out["fidelity"]["n_resync_events"] == 1


def test_resync_evicts_silent_front_first() -> None:
    r = _rec()
    r.submit(0.0, 1, 5000, 1, 100)
    r.submit(1.0, 1, 5000, 2, 100)
    r.submit(2.0, 1, 5000, 3, 50)
    r.resync_level(3.0, 1, 5000, 150)  # displayed 150, tracked 250
    recs = _records(r)
    assert recs[1].outcome == "evicted_silent"  # stalest order retired
    assert recs[2].outcome == "open" and recs[3].outcome == "open"
    # no burn fabricated: cause unobservable
    assert recs[2].burn_exec == 0 and recs[2].burn_cancel == 0
    r.resync_level(4.0, 1, 5000, 0)  # level empties silently
    assert recs[2].outcome == "evicted_silent"
    assert recs[3].outcome == "evicted_silent"
    out = r.finalize()
    assert out["n_evicted_silent"] == 3
    assert out["n_filled"] == 0 and out["n_deleted"] == 0
    assert out["fidelity"]["evicted_silent_shares"] == 250


def test_resync_shrinks_ghost_to_target() -> None:
    r = _rec()
    r.seed_ghost(1, 5000, 300)
    r.submit(0.0, 1, 5000, 1, 50)
    r.resync_level(1.0, 1, 5000, 150)  # tracked 50 -> ghost 100
    q = r._queues[(1, 5000)]
    assert q[0].ghost and q[0].remaining == 100


# ---------------------------------------------------------------------
# recorder: aggregates
# ---------------------------------------------------------------------


def test_finalize_aggregates_shape() -> None:
    r = _rec()
    # 40 orders: half at share 1.0 (own level), half queued behind 100
    for i in range(20):
        r.submit(float(i), 1, 5000 + i * 100, 1000 + i, 10)  # rank 0
    for i in range(20):
        r.submit(100.0 + i, -1, 7000, 2000 + i, 10)  # queues at one level
    # fill the first five rank-0 orders, delete the next five
    for i in range(5):
        r.execute(200.0 + i, 1, 5000 + i * 100, 1000 + i, 10)
    for i in range(5, 10):
        r.delete(300.0 + i, 1, 5000 + i * 100, 1000 + i, 10)
    # fill two of the queued ask orders mid-queue (id-directed)
    r.execute(400.0, -1, 7000, 2005, 10)
    r.execute(401.0, -1, 7000, 2010, 10)
    out = r.finalize()
    assert out["n_orders"] == 40
    assert out["n_filled"] == 7 and out["n_deleted"] == 5
    assert out["mechanism_present"] is True
    assert len(out["share_deciles"]) == 10
    # rank-0 bucket: the 20 own-level bids plus the first ask
    assert out["rank_buckets"]["front"]["n"] == 21
    assert math.isclose(out["rank_buckets"]["front"]["fill_rate"], 5 / 21)
    assert out["rank_buckets"]["3_5"]["n"] == 3
    assert out["rank_buckets"]["11_25"]["n"] == 9
    burn = out["queue_ahead_burn"]
    assert burn["n_filled_with_ahead"] == 2  # only the mid-queue fills
    # order 2005's 10 shares departed (exec) ahead of order 2010
    assert burn["exec_burn_shares"] == 10 and burn["exec_burn_share"] == 1.0
    # 50 ahead of 2005 still standing; 90 of 100 ahead of 2010 standing
    assert burn["still_standing_shares"] == 140
    assert out["fidelity"]["registry_exit_match_rate"] == 1.0


def test_rank0_dwell_measured() -> None:
    r = _rec()
    r.submit(0.0, 1, 5000, 1, 100)
    r.submit(1.0, 1, 5000, 2, 50)
    r.delete(5.0, 1, 5000, 1, 100)  # order 2 reaches front at t=5
    r.delete(8.0, 1, 5000, 2, 50)  # exits at t=8: dwell 3
    out = r.finalize()
    d = out["rank0_dwell_s"]["all_exits"]
    assert d["n"] == 2
    assert math.isclose(d["median"], (5.0 - 0.0 + 3.0) / 2)


# ---------------------------------------------------------------------
# tape arm: synthetic (message, orderbook) pair
# ---------------------------------------------------------------------


def test_lobster_pair_basic(tmp_path: Path) -> None:
    events = [
        _ev(0, 1, 1, 100, 5000, -1),  # submit ask 100 @5000
        _ev(1, 1, 2, 50, 5000, -1),  # submit ask 50 @5000 behind it
        _ev(2, 4, 1, 100, 5000, -1),  # exec order 1 -> filled
        _ev(3, 3, 2, 50, 5000, -1),  # delete order 2
        _ev(4, 1, 3, 200, 4900, 1),  # submit bid 200 @4900
        _ev(5, 2, 3, 50, 4900, 1),  # partial cancel bid to 150
        _ev(6, 3, 3, 150, 4900, 1),  # delete bid
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    out = lobster_queue_occupancy(msg, ob, n_levels=2)
    assert out["n_orders"] == 3
    assert out["n_filled"] == 1 and out["n_deleted"] == 2
    assert out["fidelity"]["victim_at_front_rate"] == 1.0
    assert out["fidelity"]["registry_exit_match_rate"] == 1.0
    burn = out["queue_ahead_burn"]
    assert burn["n_filled_with_ahead"] == 0  # order 1 sat at rank 0
    assert out["n_evicted_silent"] == 0


def test_lobster_pair_seed_ghost(tmp_path: Path) -> None:
    """Pre-window depth in row 0 becomes ghost ahead of in-window orders."""
    events = [
        _ev(0, 1, 1, 50, 5000, -1),  # in-window submit; pre-seed -> rank 0
        _ev(1, 1, 2, 30, 5000, -1),  # submits behind ghost(300) + order 1
        _ev(2, 3, 999, 300, 5000, -1),  # untracked (seed) order deletes
        _ev(3, 3, 1, 50, 5000, -1),
        _ev(4, 3, 2, 30, 5000, -1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events, seed_asks=[(5000, 300)])
    out = lobster_queue_occupancy(msg, ob, n_levels=2)
    assert out["n_orders"] == 2
    assert out["n_deleted"] == 2
    assert out["fidelity"]["n_delete_untracked_oid"] == 1
    rk = out["rank_buckets"]
    # order 1 at rank 0 (submitted before row-0 resync seeded ghosts);
    # order 2 at rank 2 (ghost block + order 1 ahead)
    assert rk["front"]["n"] == 1 and rk["2"]["n"] == 1


def test_lobster_pair_silent_drop_evicts(tmp_path: Path) -> None:
    """Orderbook row drops a live order with no event -> evicted_silent."""
    events = [
        _ev(0, 1, 1, 100, 5000, -1),
        _ev(1, 1, 2, 50, 5000, -1),
        _ev(2, 5, 0, 10, 5000, -1),  # hidden exec: no book effect
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    # corrupt row 1: level shrinks to 50 though no delete arrived
    lines = ob.read_text().strip().split("\n")
    parts = lines[1].split(",")
    parts[1] = "50"  # ask size 150 -> 50
    lines[1] = ",".join(parts)
    # keep row 2 consistent (level stays 50)
    parts = lines[2].split(",")
    parts[1] = "50"
    lines[2] = ",".join(parts)
    ob.write_text("\n".join(lines) + "\n")
    out = lobster_queue_occupancy(msg, ob, n_levels=2)
    assert out["n_evicted_silent"] == 1  # front order retired silently
    assert out["n_filled"] == 0 and out["n_deleted"] == 0
    assert out["fidelity"]["evicted_silent_shares"] == 100


# ---------------------------------------------------------------------
# sim arm
# ---------------------------------------------------------------------


def test_sim_arm_end_to_end() -> None:
    out = sim_queue_occupancy(seed=3, horizon=2000)
    assert out["ok"] and out["mechanism_present"] is True
    assert out["n_orders"] >= 100
    # the sim consumes the true front: victims are at rank 0 nearly
    # always (untracked seed-order fills dilute the rate slightly)
    assert out["fidelity"]["victim_at_front_rate"] >= 0.9
    # reconstructed shares-ahead must equal the engine's queue_ahead
    assert out["fidelity"]["shares_ahead_selfcheck_rate"] == 1.0
    assert out["fidelity"]["registry_exit_match_rate"] == 1.0
    assert out["queue_ahead_burn"]["exec_burn_share"] is not None


# ---------------------------------------------------------------------
# sealed receipt
# ---------------------------------------------------------------------


def _flat_keys(obj: object, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.update(_flat_keys(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            keys.update(_flat_keys(v, f"{prefix}{i}."))
    else:
        keys.add(prefix.rstrip("."))
    return keys


def _key_tokens(key: str) -> set[str]:
    out: set[str] = set()
    for segment in key.split("."):
        out.update(t for t in segment.lower().replace("-", "_").split("_") if t)
    return out


def test_bench_seals_queue_occupancy_v1(tmp_path: Path) -> None:
    events = [
        _ev(0, 1, 1, 100, 5000, -1),
        _ev(1, 1, 2, 50, 5000, -1),
        _ev(2, 4, 1, 100, 5000, -1),
        _ev(3, 3, 2, 50, 5000, -1),
        _ev(4, 1, 3, 200, 4900, 1),
        _ev(5, 3, 3, 200, 4900, 1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    payload = queue_occupancy_bench(tmp_path, "AMZN", seed=3, horizon=1500)
    assert payload["schema"] == QUEUE_OCCUPANCY_SCHEMA
    assert payload["kind"] == "queue_occupancy"
    assert payload["data_label"] == "MIXED"
    assert payload["research_only"] is True
    # sealed: hash over the payload before receipt_sha256 was added
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    assert payload["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))
    # honesty contract: no forbidden headline metric keys anywhere
    bad = {k for k in _flat_keys(payload) if _key_tokens(k) & FORBIDDEN_RESEARCH_METRIC_KEYS}
    assert not bad
    assert set(payload["sim_arms"]) == {"iid", "regime", "split"}
    assert isinstance(payload["divergences"], list)
