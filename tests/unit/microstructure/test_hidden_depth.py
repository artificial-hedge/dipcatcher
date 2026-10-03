from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.hidden_depth import (
    _size_moments,
    hidden_depth_bench,
    lobster_hidden_share,
    sim_hidden_share,
)


def _write_msg(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_hidden_share_counts_both_types(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    _write_msg(
        msg,
        [
            [34200.0, 4, 1, 100, 5000, -1],  # visible exec
            [34200.1, 4, 2, 50, 5001, -1],
            [34200.2, 5, 3, 300, 5002, -1],  # hidden exec
            [34200.3, 5, 4, 150, 5003, 1],
        ],
    )
    out = lobster_hidden_share(tmp_path)
    assert out["n_execs"] == 4
    assert out["n_hidden_execs"] == 2
    assert out["hidden_trade_share"] == pytest.approx(0.5)
    assert out["hidden_volume_share"] == pytest.approx(450 / 600)
    assert out["hidden"]["median_size"] == pytest.approx(225.0)


def test_hidden_share_no_execs(tmp_path: Path) -> None:
    _write_msg(
        tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv",
        [[34200.0, 1, 1, 100, 5000, -1]],
    )
    out = lobster_hidden_share(tmp_path)
    assert out["ok"] is False


def test_sim_reports_mechanism_absent() -> None:
    out = sim_hidden_share(horizon=1500, seed=3)
    assert out["mechanism_present"] is False
    assert out["hidden_trade_share"] == 0.0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hidden_depth_bench(tmp_path)


def test_size_moments_empty() -> None:
    assert _size_moments([]) == {"n": 0}
