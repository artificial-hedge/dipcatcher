from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.lobster import LobsterEvent
from quant_fund.microstructure.order_lifetime import (
    _hist,
    lobster_lifetimes,
    order_lifetime_bench,
    sim_fill_delays,
)


def _ev(t: float, ty: int, oid: int, sz: int, px: int, d: int) -> LobsterEvent:
    return LobsterEvent(t, ty, oid, sz, px, d)


def _write_msg(path: Path, events: list[LobsterEvent]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            w.writerow([ev.time_s, ev.event_type, ev.order_id, ev.size, ev.price, ev.direction])


def test_lifetimes_exec_cancel_delete(tmp_path: Path) -> None:
    events = [
        _ev(0.0, 1, 1, 100, 5000, -1),  # submit sell id1
        _ev(0.0, 1, 2, 50, 4900, 1),  # submit buy id2
        _ev(2.0, 4, 1, 100, 5000, -1),  # exec consumes id1 fully -> executed, life 2s
        _ev(5.0, 2, 2, 50, 4900, 1),  # partial cancel of id2 fully -> canceled, life 5s
        _ev(6.0, 1, 3, 10, 5100, -1),
        _ev(9.0, 3, 3, 10, 5100, -1),  # delete id3 -> deleted, life 3s
    ]
    msg = tmp_path / "m.csv"
    _write_msg(msg, events)
    out = lobster_lifetimes(msg)
    assert out["n_executed"] == 1
    assert out["n_canceled"] == 1
    assert out["n_deleted"] == 1
    assert out["lifetime_s_p50_executed"] == pytest.approx(2.0)
    assert out["lifetime_s_p50_deleted"] == pytest.approx(3.0)
    assert out["exec_share_of_resolved_volume"] == pytest.approx(0.6667)


def test_lifetimes_censored_order(tmp_path: Path) -> None:
    events = [_ev(0.0, 1, 7, 100, 5000, 1)]
    msg = tmp_path / "m.csv"
    _write_msg(msg, events)
    out = lobster_lifetimes(msg)
    assert out["n_open_at_end_censored"] == 1
    assert out["n_orders_tracked"] == 1


def test_hist_covers_all() -> None:
    a = np.asarray([0.0005, 0.005, 0.05, 0.5, 5.0, 50.0, 500.0, 5000.0])
    h = _hist(a, (0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 600.0))
    assert sum(h.values()) == 8


def test_sim_fill_delays() -> None:
    out = sim_fill_delays(horizon=1500, seed=3)
    assert out["n_fills"] > 10
    assert out["fill_delay_p50_s"] is not None


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        order_lifetime_bench(tmp_path)
