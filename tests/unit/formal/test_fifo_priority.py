"""Tests for fifo_priority: queue-rank audit of fills."""

from __future__ import annotations

import csv

import pytest

from quant_fund.formal.fifo_priority import (
    fifo_priority_bench,
    lobster_fifo,
    sim_fifo,
)


def _row(w, t, et, oid, size, price, direction):
    w.writerow([t, et, oid, size, price, direction])


def _ob(w, bid_px, bid_sz):
    """orderbook row: ask 4100x500; bid level at bid_px x bid_sz."""
    w.writerow([4100, 500, bid_px, bid_sz])


def test_fifo_clean(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        w, wo = csv.writer(fm), csv.writer(fo)
        _row(w, 34200.0, 1, 1, 100, 4000, 1)
        _ob(wo, 4000, 100)
        _row(w, 34200.1, 1, 2, 100, 4000, 1)
        _ob(wo, 4000, 200)
        _row(w, 34200.2, 4, 1, 100, 4000, -1)  # front order fully consumed
        _ob(wo, 4000, 100)
        _row(w, 34200.3, 4, 2, 50, 4000, -1)  # next in line -> rank 0
        _ob(wo, 4000, 50)
    out = lobster_fifo(msg, ob)
    assert out["n_fills"] == 2
    assert out["n_verified_fills"] == 2
    assert out["rank0_share_verified"] == 1.0
    assert out["rank_violation_share"] == 0.0


def test_fifo_violation_detected(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        w, wo = csv.writer(fm), csv.writer(fo)
        _row(w, 34200.0, 1, 1, 100, 4000, 1)
        _ob(wo, 4000, 100)
        _row(w, 34200.1, 1, 2, 100, 4000, 1)
        _ob(wo, 4000, 200)
        _row(w, 34200.2, 4, 2, 100, 4000, -1)  # back of queue fills first
        _ob(wo, 4000, 100)
    out = lobster_fifo(msg, ob)
    assert out["rank_violation_share"] == 1.0
    assert out["max_rank_verified"] == 1


def test_fifo_unclean_when_official_disagrees(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        w, wo = csv.writer(fm), csv.writer(fo)
        _row(w, 34200.0, 1, 1, 100, 4000, 1)
        _ob(wo, 4000, 100)
        _row(w, 34200.1, 1, 2, 100, 4000, 1)
        _ob(wo, 4000, 999)  # official says 999, replay says 200 -> unverifiable
        _row(w, 34200.2, 4, 2, 100, 4000, -1)
        _ob(wo, 4000, 899)
    out = lobster_fifo(msg, ob)
    assert out["n_unclean_fills"] == 1
    assert out["rank_violation_share"] is None


def test_fifo_shadow_fill(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        w, wo = csv.writer(fm), csv.writer(fo)
        _row(w, 34200.0, 4, 99, 10, 4000, 1)  # exec on never-seen id
        _ob(wo, 4000, 4000)
    out = lobster_fifo(msg, ob)
    assert out["n_shadow_fills"] == 1
    assert out["shadow_share"] == 1.0


def test_sim_fifo_clean() -> None:
    out = sim_fifo(horizon=8000, seed=3)
    assert out["ok"]
    assert out["rank0_share_verified"] == 1.0


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        fifo_priority_bench(tmp_path)
