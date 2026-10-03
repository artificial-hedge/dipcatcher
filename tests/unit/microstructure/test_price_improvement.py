from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.price_improvement import (
    lobster_price_improvement,
    price_improvement_bench,
)


def _write_pair(tmp_path: Path, msg_rows: list, ob_rows: list) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as f:
        csv.writer(f).writerows(msg_rows)
    with ob.open("w", newline="") as f:
        csv.writer(f).writerows(ob_rows)


def test_hidden_inside_spread_improves(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        [
            [34200.0, 1, 1, 100, 2500, 1],
            [34200.5, 5, 9, 50, 3950, -1],  # hidden sell filled at 3950
            [34201.0, 5, 8, 40, 4000, -1],  # hidden sell at touch
        ],
        [
            [4000, 200, 3000, 100],  # seed ask=4000
            [4000, 200, 3000, 100],  # pre-event ask=4000
            [4000, 100, 3000, 100],
        ],
    )
    out = lobster_price_improvement(tmp_path)
    assert out["n_hidden"] == 2
    # (4000-3950)/100 = 0.5 ticks improvement; then 0
    assert out["mean_improvement_ticks"] == pytest.approx(0.25)
    assert out["improved_share"] == pytest.approx(0.5)
    assert out["at_touch_share"] == pytest.approx(0.5)


def test_no_hidden_execs(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        [[34200.0, 1, 1, 100, 2500, 1]],
        [[4000, 200, 3000, 100]],
    )
    out = lobster_price_improvement(tmp_path)
    assert out["ok"] is False


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        price_improvement_bench(tmp_path)
