"""Tests for microstructure/sim_real_ledger.py — sim-vs-real scorecard."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.lobster import LobsterBook, LobsterEvent
from quant_fund.microstructure.sim_real_ledger import (
    SIM_REAL_SCHEMA,
    _sign_lag1,
    measure_lobster,
    sim_real_ledger,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def _ev(t: int, ty: int, oid: int, sz: int, px: int, d: int) -> LobsterEvent:
    return LobsterEvent(float(t), ty, oid, sz, px, d)


def _write_pair(msg: Path, ob: Path, events: list[LobsterEvent], levels: int = 2) -> None:
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
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            w.writerow([ev.time_s, ev.event_type, ev.order_id, ev.size, ev.price, ev.direction])
    with ob.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_sign_lag1() -> None:
    import numpy as np

    # n-normalized: dot/n, not (n-1)/n — same convention as flow_memory
    assert _sign_lag1(np.ones(10)) == pytest.approx(0.9)
    alt = np.array([1.0, -1.0] * 5)
    assert _sign_lag1(alt) < 0


def test_measure_sim_smoke() -> None:
    from quant_fund.microstructure.sim_real_ledger import measure_sim

    out = measure_sim(ZILobConfig(seed=3), None, horizon=400)
    assert out["n_events"] == 400
    assert out["mo_fraction"] > 0
    assert out["spread_ticks_median"] > 0
    assert out["units"] == "ticks"


def test_measure_lobster_synth(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 1, 5000, -1),
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 50, 4900, 1),
        _ev(3, 4, 2, 10, 4900, 1),
        _ev(4, 1, 3, 20, 5100, -1),
        _ev(5, 4, 1, 30, 5000, -1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    out = measure_lobster(msg, ob, tick_units=100.0)
    assert out["n_trades"] == 2
    assert out["n_events"] == 5  # first row consumed as seed
    assert out["mo_fraction"] == 2 / 5
    assert out["spread_ticks_median"] > 0


def test_ledger_schema(tmp_path: Path) -> None:
    events = [
        _ev(0, 5, 0, 1, 5000, -1),
        _ev(1, 1, 1, 100, 5000, -1),
        _ev(2, 1, 2, 50, 4900, 1),
        _ev(3, 4, 2, 10, 4900, 1),
    ]
    msg = tmp_path / "AMZN_x_message_2.csv"
    ob = tmp_path / "AMZN_x_orderbook_2.csv"
    _write_pair(msg, ob, events)
    import quant_fund.microstructure.sim_real_ledger as m

    orig = m.measure_sim

    def fast(cfg, flow, *, horizon=20000):  # shrink sim arm for the test
        return orig(cfg, flow, horizon=300)

    m.measure_sim = fast
    try:
        out = sim_real_ledger(tmp_path, "AMZN")
    finally:
        m.measure_sim = orig
    assert out["schema"] == SIM_REAL_SCHEMA
    assert out["data_label"] == "MIXED"
    assert "sim_calm" in out["table"]["sign_lag1"]
    assert "real" in out["table"]["sign_lag1"]
    assert len(out["receipt_sha256"]) == 64
