"""Tests for microstructure/tape_surgery.py."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.tape_surgery import (
    lobster_tape_surgery,
    tape_surgery_bench,
)

OB_ROW = "2001000,5,2000800,7" + ",0,0" * 9  # ask 200.10 x5, bid 200.08 x7


def _write_tape(tmp: Path, rows: list[list[str]]) -> Path:
    msg = tmp / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with ob.open("w", newline="") as f:
        for _ in rows:
            csv.writer(f).writerow(OB_ROW.split(","))
    return tmp


def test_missing_tape_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tape_surgery_bench(tmp_path)


def test_cancel_ablation_removes_all_exits(tmp_path):
    rows = [
        ["1.0", "1", "1", "10", "2000800", "1"],  # buy limit at bid
        ["1.1", "1", "2", "10", "2001000", "-1"],  # sell limit at ask
        ["1.2", "3", "1", "10", "2000800", "1"],  # delete order 1
        ["1.3", "4", "2", "5", "2001000", "-1"],  # exec order 2 (buy aggression)
        ["1.4", "4", "2", "5", "2001000", "-1"],  # exec order 2 again
    ]
    _write_tape(tmp_path, rows)
    out = lobster_tape_surgery(tmp_path)
    base = out["arms"]["baseline"]
    cut = out["arms"]["drop_liquidity_exits"]
    assert base["n_kept"] == 5
    assert base["n_cancel_delete"] == 1 and base["n_exec"] == 2
    assert cut["n_cancel_delete"] == 0
    assert cut["n_exec"] == 2  # execs unaffected by dropping the delete
    assert out["attribution_deltas"]["drop_liquidity_exits"]["n_cancel_delete"] == -1


def test_improve_ablation_cascades(tmp_path):
    rows = [
        ["1.0", "1", "1", "10", "2000800", "1"],  # at bid
        ["1.1", "1", "2", "10", "2000900", "1"],  # inside spread -> dropped
        ["1.2", "2", "2", "4", "2000900", "1"],  # partial cancel of dropped order
        ["1.3", "4", "1", "5", "2000800", "1"],  # sell aggression hits order 1
    ]
    _write_tape(tmp_path, rows)
    out = lobster_tape_surgery(tmp_path)
    arm = out["arms"]["drop_improve_submissions"]
    assert arm["n_kept"] == 2  # submission AND its cancel both dropped
    assert arm["n_exec"] == 1


def test_hidden_exec_ablation(tmp_path):
    rows = [
        ["1.0", "1", "1", "10", "2000800", "1"],
        ["1.1", "5", "9", "7", "2000900", "-1"],  # hidden exec
        ["1.2", "4", "1", "5", "2000800", "1"],  # visible exec
    ]
    _write_tape(tmp_path, rows)
    out = lobster_tape_surgery(tmp_path)
    assert out["arms"]["drop_hidden_execs"]["n_kept"] == 2


def test_bench_sealed_and_deterministic(tmp_path):
    rows = [
        ["1.0", "1", "1", "10", "2000800", "1"],
        ["1.1", "1", "2", "10", "2001000", "-1"],
        ["1.2", "4", "1", "5", "2000800", "1"],
        ["1.3", "3", "2", "10", "2001000", "-1"],
    ]
    _write_tape(tmp_path, rows)
    a = tape_surgery_bench(tmp_path)
    b = tape_surgery_bench(tmp_path)
    assert a["schema"] == "tape_surgery.v1"
    assert a["data_label"] == "REAL"
    assert a["receipt_sha256"] == b["receipt_sha256"]
    assert len(a["receipt_sha256"]) == 64
