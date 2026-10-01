from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.event_matrix import (
    event_matrix_bench,
    lobster_event_matrix,
    sim_event_matrix,
)


def test_lobster_event_matrix_counts(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([34200.0, 1, 1, 100, 4000, 1])  # sub buy
        w.writerow([34200.5, 1, 2, 100, 4010, -1])  # sub sell
        w.writerow([34201.0, 3, 1, 100, 4000, 1])  # delete
        w.writerow([34202.0, 4, 3, 50, 4010, -1])  # exec
        w.writerow([34203.0, 5, 4, 30, 4020, 1])  # hidden exec
    out = lobster_event_matrix(tmp_path)
    assert out["n_events"] == 5
    assert out["span_s"] == pytest.approx(3.0)
    assert out["events_per_s"]["sub"] == pytest.approx(0.6667)
    assert out["events_per_s"]["exec_hidden"] == pytest.approx(0.3333)
    assert out["mix_share"]["sub"] == pytest.approx(0.4)


def test_sim_event_matrix_runs() -> None:
    out = sim_event_matrix(horizon=1500, seed=3)
    assert out["n_events"] == 1500
    assert out["events_per_s"]["exec"] > 0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        event_matrix_bench(tmp_path)
