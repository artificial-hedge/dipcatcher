from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.queue_jump import (
    _queue_bin,
    lobster_queue_jump,
    queue_jump_bench,
    sim_queue_jump,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

MSG = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
OB = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"


def _write_crafted(tmp_path: Path) -> None:
    """Six events; row_i is the post-event book snapshot (LOBSTER layout).

    Row 0 seeds the book (and its event is applied, not scored).
    Scored submissions, pre-event state:
    - ev1 buy@3100: bid 3000x510 (bin 100+) -> inside spread
    - ev2 sell@4000: ask 4000x500 (bin 100+) -> at touch
    - ev3 sell@3900: ask 4000x505 (bin 100+) -> inside spread
    - ev4 buy@4000:  bid 3100x10  (bin 10-49) -> crossing (improves, not inside)
    - ev5 buy@3050:  bid 3100x10  (bin 10-49) -> behind touch
    """
    msg = tmp_path / MSG
    ob = tmp_path / OB
    events = [
        [34200.0, 1, 1, 10, 3000, 1],  # seed event: buy@3000 on seeded book
        [34201.0, 1, 2, 10, 3100, 1],
        [34202.0, 1, 3, 5, 4000, -1],
        [34203.0, 1, 4, 5, 3900, -1],
        [34204.0, 1, 5, 5, 4000, 1],
        [34205.0, 1, 6, 1, 3050, 1],
    ]
    rows = [
        [4000, 500, 3000, 500],
        [4000, 500, 3100, 10, 0, 0, 3000, 510],
        [4000, 505, 3100, 10, 0, 0, 3000, 510],
        [3900, 5, 3100, 10, 4000, 505, 3000, 510],
        [4000, 505, 3100, 10, 0, 0, 3000, 510],
        [4000, 505, 3100, 10, 0, 0, 3050, 1, 0, 0, 3000, 510],
    ]
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        for ev, row in zip(events, rows, strict=True):
            wm.writerow(ev)
            wo.writerow(row)


def test_queue_bin_boundaries() -> None:
    assert _queue_bin(1) == "1-9"
    assert _queue_bin(9) == "1-9"
    assert _queue_bin(10) == "10-49"
    assert _queue_bin(49) == "10-49"
    assert _queue_bin(50) == "50-99"
    assert _queue_bin(99) == "50-99"
    assert _queue_bin(100) == "100+"
    assert _queue_bin(999_999) == "100+"
    with pytest.raises(ValueError, match="out of range"):
        _queue_bin(0)


def test_lobster_queue_jump_csv(tmp_path: Path) -> None:
    _write_crafted(tmp_path)
    out = lobster_queue_jump(tmp_path)
    assert out["n_submissions"] == 5
    assert out["n_inside_spread"] == 2  # ev1, ev3
    assert out["n_improves_touch"] == 3  # ev1, ev3, ev4(crossing)
    assert out["n_crossing"] == 1  # ev4
    b100 = out["by_queue_bin"]["100+"]
    assert b100["n_submissions"] == 3  # ev1..ev3
    assert b100["share_inside_spread"] == 0.6667
    assert b100["share_improves_touch"] == 0.6667
    b1049 = out["by_queue_bin"]["10-49"]
    assert b1049["n_submissions"] == 2  # ev4, ev5
    assert b1049["share_inside_spread"] == 0.0
    assert b1049["share_improves_touch"] == 0.5  # crossing improves but isn't inside
    assert b1049["median_dist_ticks"] == -4.25  # median(-9, +0.5)


def test_sim_arm_mechanism_absent() -> None:
    out = sim_queue_jump(horizon=2000, seed=3)
    assert out["mechanism_present"] is False
    assert out["data_label"] == "SYNTHETIC"
    assert out["n_submissions"] > 0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        queue_jump_bench(tmp_path)


def test_bench_receipt_sealed(tmp_path: Path) -> None:
    _write_crafted(tmp_path)
    out = queue_jump_bench(tmp_path)
    assert out["schema"] == "queue_jump.v1"
    assert out["kind"] == "queue_jump"
    assert out["data_label"] == "MIXED"
    assert out["real"]["data_label"] == "REAL"
    assert out["sim"]["mechanism_present"] is False
    assert out["divergences"] == []  # 1-9 bin empty -> no rising-share claim
    sha = out["receipt_sha256"]
    assert isinstance(sha, str) and len(sha) == 64
    body = {k: v for k, v in out.items() if k != "receipt_sha256"}
    assert hash_bytes(canonical_json_bytes(body)) == sha
