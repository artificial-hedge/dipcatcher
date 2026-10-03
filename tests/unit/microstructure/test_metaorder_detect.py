from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.metaorder_detect import (
    cluster_metaorders,
    lobster_metaorders,
    metaorder_bench,
    sim_metaorders,
)


def test_cluster_breaks_on_sign_and_gap() -> None:
    times = [0.0, 0.2, 0.4, 5.0, 5.2, 5.4, 5.6]
    signs = [1, 1, 1, -1, -1, -1, -1]
    meta = cluster_metaorders(times, signs, gap_s=1.0, min_len=3)
    assert len(meta) == 2
    assert meta[0]["n_fills"] == 3 and meta[1]["sign"] == -1


def test_cluster_min_len_filters() -> None:
    times = [0.0, 0.2, 2.0, 2.2]
    signs = [1, 1, 1, 1]
    meta = cluster_metaorders(times, signs, gap_s=1.0, min_len=3)
    assert len(meta) == 0  # two runs of 2 — below min_len


def test_lobster_metaorders_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        for i in range(4):
            w.writerow([34200.0 + i * 0.5, 4, i + 1, 50, 5000, -1])
        w.writerow([34210.0, 4, 9, 50, 5000, 1])
    out = lobster_metaorders(tmp_path, gap_s=1.0, min_len=3)
    assert out["n_metaorders"] == 1
    assert out["clustered_fill_share"] == pytest.approx(0.8)


def test_sim_metaorders_scores_detector() -> None:
    out = sim_metaorders(horizon=6000, seed=3, p_start=0.25)
    assert out["n_true_children"] > 0
    assert out["precision"] is not None
    assert 0.0 <= out["precision"] <= 1.0
    assert 0.0 <= out["recall"] <= 1.0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        metaorder_bench(tmp_path)
