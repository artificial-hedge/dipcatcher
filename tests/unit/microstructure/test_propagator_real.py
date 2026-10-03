from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from quant_fund.microstructure.lobster import LobsterBook, LobsterEvent
from quant_fund.microstructure.propagator_real import (
    _propagator,
    propagator_lobster,
    propagator_real_bench,
    propagator_sim,
)


def _ev(t: int, ty: int, oid: int, sz: int, px: int, d: int) -> LobsterEvent:
    return LobsterEvent(float(t), ty, oid, sz, px, d)


def _write_pair(msg_path: Path, ob_path: Path, events: list[LobsterEvent], levels: int = 2) -> None:
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
    with msg_path.open("w", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            w.writerow([ev.time_s, ev.event_type, ev.order_id, ev.size, ev.price, ev.direction])
    with ob_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_propagator_permanent_impact_stays_positive() -> None:
    mids = [float(i) for i in range(50)]
    out = _propagator(mids, list(range(49)), [1] * 49)
    assert out["response_ticks"]["1"] == pytest.approx(1.0)
    assert out["response_ticks"]["8"] == pytest.approx(8.0)


def test_propagator_sign_symmetry() -> None:
    mids = [float(i) for i in range(10)]
    buy = _propagator(mids, list(range(6)), [1] * 6)["response_ticks"]["1"]
    sell = _propagator(mids, list(range(6)), [-1] * 6)["response_ticks"]["1"]
    assert buy == -sell


def test_propagator_sim_smoke() -> None:
    out = propagator_sim(horizon=2000, seed=3)
    assert out["n_events"] >= 1500
    assert out["n_trades"] > 10
    assert set(out["response_ticks"]) == {"1", "2", "4", "8", "16", "32", "64", "128", "256"}


def test_propagator_lobster_on_tiny_tape(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 10, 5000, -1),  # hidden exec: row0 seeds the book
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 50, 4900, 1),
        _ev(3, 4, 2, 20, 4900, 1),  # buy-side exec on the bid
        _ev(4, 4, 1, 30, 5000, -1),  # sell-side exec on the ask
    ]
    msg, ob = tmp_path / "m.csv", tmp_path / "o.csv"
    _write_pair(msg, ob, events)
    out = propagator_lobster(msg, ob)
    assert out["n_trades"] == 2
    assert out["n_events"] >= 3
    assert all(v is None or math.isfinite(v) for v in out["response_ticks"].values())


def test_bench_requires_real_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        propagator_real_bench(tmp_path)
