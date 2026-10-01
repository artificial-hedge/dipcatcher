"""Tests for microstructure/lob_exec.py."""

from __future__ import annotations

import csv
from pathlib import Path

from quant_fund.microstructure.lob_exec import (
    LOB_EXEC_SCHEMA,
    _walk_book,
    exec_on_tape,
    lob_exec_bench,
    twap_children,
)
from quant_fund.microstructure.lobster import LobsterBook, LobsterEvent


def _ev(t: int, ty: int, oid: int, sz: int, px: int, d: int) -> LobsterEvent:
    return LobsterEvent(float(t), ty, oid, sz, px, d)


def _pair(tmp_path: Path, events: list[LobsterEvent], levels: int = 2) -> tuple[Path, Path]:
    book = LobsterBook()
    rows: list[list[str]] = []
    for ev in events:
        book.apply(ev)
        asks, bids = book.top("ask", levels), book.top("bid", levels)
        row: list[str] = []
        for lvl in range(levels):
            a = asks[lvl] if lvl < len(asks) else (0, 0)
            b = bids[lvl] if lvl < len(bids) else (0, 0)
            row += [str(a[0]), str(a[1]), str(b[0]), str(b[1])]
        rows.append(row)
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            w.writerow([ev.time_s, ev.event_type, ev.order_id, ev.size, ev.price, ev.direction])
    with ob.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return msg, ob


def test_twap_children() -> None:
    assert twap_children(10, 3) == [4, 3, 3]
    assert sum(twap_children(101, 7)) == 101


def test_walk_book_consumption() -> None:
    b = LobsterBook()
    b.seed([(5000, 100), (5100, 50)], [(4900, 80)])
    filled, notional, walk = _walk_book(b, "buy", 120)
    assert filled == 120
    assert notional == 5000 * 100 + 5100 * 20
    assert walk == (5100 - 5000) * 20
    assert b.ask[5000] == 0 if 5000 in b.ask else True
    assert 5100 in b.ask and b.ask[5100] == 30


def test_exec_on_tape_basic(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 1, 5000, -1),
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 100, 5100, -1),
        _ev(3, 1, 3, 100, 4900, 1),
        _ev(4, 1, 4, 100, 4800, 1),
        _ev(5, 1, 5, 100, 5000, -1),
    ]
    msg, ob = _pair(tmp_path, events)
    from quant_fund.microstructure.lobster import parse_messages, parse_orderbook_row

    evs = list(parse_messages(msg))
    snaps = [parse_orderbook_row(r) for r in csv.reader(ob.open())]
    out = exec_on_tape(
        evs,
        snaps,
        start=5,  # after asks@5000,5100 and bids@4900,4800 are resting
        parent_size=10,
        n_children=2,
        events_per_child=1,
        side="buy",
        tick_units=100.0,
    )
    assert out is not None
    assert out["fill_fraction"] == 1.0
    assert out["liquidity_cost_ticks"] >= 0


def test_bench_schema(tmp_path: Path) -> None:
    events = (
        [_ev(0, 5, 0, 1, 5000, -1)]
        + [_ev(i, 1, i, 100, 5000 + (i % 3) * 100, -1) for i in range(1, 30)]
        + [_ev(i, 1, i + 100, 100, 4900 - (i % 3) * 50, 1) for i in range(30, 60)]
    )
    _pair(tmp_path, events)
    import quant_fund.microstructure.lob_exec as m

    orig = m.exec_on_tape

    def fast(*a, **kw):
        kw["events_per_child"] = 2
        return orig(*a, **kw)

    m.exec_on_tape = fast
    try:
        out = lob_exec_bench(tmp_path, "AMZN")
    finally:
        m.exec_on_tape = orig
    assert out["schema"] == LOB_EXEC_SCHEMA
    assert out["data_label"] == "MIXED"
    assert "receipt_sha256" in out
