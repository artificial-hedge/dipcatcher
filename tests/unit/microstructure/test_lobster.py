"""Tests for microstructure/lobster.py — LOBSTER replay validation.

Builds a synthetic LOBSTER-format (message, orderbook) pair by
construction: apply each event to a reference book, dump the top-N
snapshot as the orderbook row — so a correct replay matches 100%.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.lobster import (
    LOBSTER_SCHEMA,
    LobsterBook,
    LobsterEvent,
    lobster_replay_bench,
    parse_orderbook_row,
    resync_band,
    validate_reconstruction,
)


def _ev(t: int, ty: int, oid: int, sz: int, px: int, d: int) -> LobsterEvent:
    return LobsterEvent(float(t), ty, oid, sz, px, d)


def _write_pair(msg_path: Path, ob_path: Path, events: list[LobsterEvent], levels: int = 2) -> None:
    """Write message CSV + orderbook CSV where each row = post-event book."""
    book = LobsterBook()
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


def test_book_apply_submission_and_delete() -> None:
    b = LobsterBook()
    b.apply(_ev(0, 1, 1, 100, 5000, -1))  # sell 100 @ 5000
    b.apply(_ev(1, 1, 2, 50, 4900, 1))  # buy 50 @ 4900
    assert b.top("ask", 1) == [(5000, 100)]
    assert b.top("bid", 1) == [(4900, 50)]
    b.apply(_ev(2, 3, 1, 100, 5000, -1))  # full delete
    assert b.top("ask", 1) == []


def test_partial_cancel_reduces_level() -> None:
    b = LobsterBook()
    b.apply(_ev(0, 1, 1, 100, 5000, -1))
    b.apply(_ev(1, 2, 1, 40, 5000, -1))
    assert b.top("ask", 1) == [(5000, 60)]
    assert b.orders[1] == (5000, 60)


def test_untracked_delete_falls_back_to_size() -> None:
    b = LobsterBook()
    b.seed([(5000, 100)], [(4900, 50)])
    b.apply(_ev(1, 3, 999, 30, 5000, -1))  # pre-window order, unknown id
    assert b.top("ask", 1) == [(5000, 70)]


def test_resync_band_adopts_display() -> None:
    b = LobsterBook()
    b.seed([(5000, 100), (6000, 10)], [(4900, 50)])
    resync_band(b, [(5100, 20)], [(4800, 30)])
    assert b.top("ask", 5) == [(5100, 20), (6000, 10)]  # deep kept
    assert b.top("bid", 5) == [(4800, 30)]


def test_parse_orderbook_row() -> None:
    asks, bids = parse_orderbook_row(["100", "5", "90", "3", "200", "7", "80", "2"])
    assert asks == [(100, 5), (200, 7)]
    assert bids == [(90, 3), (80, 2)]


def test_replay_synthetic_pair_exact(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 10, 5000, -1),  # hidden exec: row0 = seed book state
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 50, 4900, 1),
        _ev(3, 4, 2, 20, 4900, 1),  # exec on bid
        _ev(4, 3, 1, 100, 5000, -1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    out = validate_reconstruction(msg, ob, n_levels=2)
    # hidden execs aren't compared; every visible event must match
    assert out["match_rate"] == 1.0
    assert out["n_resync_events"] == 0


def test_replay_detects_divergence(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 10, 5000, -1),
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 50, 4900, 1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    # corrupt the last orderbook row: drop the bid
    lines = ob.read_text().strip().split("\n")
    parts = lines[-1].split(",")
    parts[2], parts[3] = "1", "1"  # bid price/size garbage
    lines[-1] = ",".join(parts)
    ob.write_text("\n".join(lines) + "\n")
    out = validate_reconstruction(msg, ob, n_levels=2)
    assert out["n_resync_events"] == 1
    assert out["n_match"] == out["n_compared"] - 1


def test_bench_schema(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 10, 5000, -1),
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 50, 4900, 1),
        _ev(3, 4, 2, 10, 4900, 1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    out = lobster_replay_bench(tmp_path, "AMZN")
    assert out["schema"] == LOBSTER_SCHEMA
    assert out["data_label"].startswith("LOBSTER-AMZN")
    assert len(out["tape_sha256"]) == 64
    assert "resync_rate" in out["reconstruction"]
    with pytest.raises(StopIteration):
        lobster_replay_bench(tmp_path, "ZZZZ")
