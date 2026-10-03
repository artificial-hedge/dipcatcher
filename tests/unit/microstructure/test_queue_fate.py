"""Tests for queue_fate: entry-position-conditioned outcomes."""

from __future__ import annotations

import csv

import pytest

from quant_fund.microstructure.queue_fate import (
    _classify_submit,
    _fate_stats,
    lobster_queue_fate,
    queue_fate_bench,
    sim_queue_fate,
)


def test_classify_submit() -> None:
    top = (4100.0, 500.0, 4000.0, 300.0)  # ask_px, ask_sz, bid_px, bid_sz
    assert _classify_submit(4050.0, 1, top) == ("improve", 0.0)
    assert _classify_submit(4000.0, 1, top) == ("join", 300.0)
    cat, d = _classify_submit(3900.0, 1, top)
    assert cat == "behind" and d == 1.0
    assert _classify_submit(4050.0, -1, top) == ("improve", 0.0)
    assert _classify_submit(4100.0, -1, top) == ("join", 500.0)
    cat, d = _classify_submit(4200.0, -1, top)
    assert cat == "behind" and d == 1.0


def test_fate_stats_fill_gradient() -> None:
    cats = ["improve"] * 200 + ["join"] * 200 + ["behind"] * 200
    ahead = [0.0] * 200 + [50.0] * 200 + [10.0] * 200
    outcomes = ["filled"] * 160 + ["deleted"] * 40
    outcomes += ["filled"] * 80 + ["deleted"] * 120
    outcomes += ["deleted"] * 190 + ["open"] * 10
    times = [0.0] * 600
    otimes = [1.0] * 600
    out = _fate_stats(cats, ahead, outcomes, times, otimes)
    assert out["ok"]
    assert out["per_category"]["improve"]["fill_rate"] == pytest.approx(0.8)
    assert out["per_category"]["join"]["fill_rate"] == pytest.approx(0.4)
    assert out["per_category"]["behind"]["fill_rate"] == pytest.approx(0.0)
    assert out["per_category"]["improve"]["median_time_to_outcome_s"] == pytest.approx(1.0)


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        oid = 0
        for _ in range(60):
            oid += 1
            t += 0.01
            wm.writerow([t, 1, oid, 10, 4000, 1])  # join bid
            wo.writerow([4100, 500, 4000, 500])
            t += 0.01
            wm.writerow([t, 4, oid, 10, 4000, 1])  # executed (resting side = buy)
            wo.writerow([4100, 500, 4000, 500])
        for _ in range(60):
            oid += 1
            t += 0.01
            wm.writerow([t, 1, oid, 10, 3900, 1])  # behind
            wo.writerow([4100, 500, 4000, 500])
            t += 0.01
            wm.writerow([t, 3, oid, 0, 0, 1])  # deleted
            wo.writerow([4100, 500, 4000, 500])
    out = lobster_queue_fate(msg, ob)
    assert out["ok"]
    assert out["per_category"]["join"]["fill_rate"] == pytest.approx(1.0)
    assert out["per_category"]["behind"]["fill_rate"] == pytest.approx(0.0)


def test_sim_runs() -> None:
    out = sim_queue_fate(horizon=8000, seed=3)
    assert out["ok"]
    assert 0.0 <= out["fill_rate_overall"] <= 1.0
    assert "improve" in out["per_category"] or "join" in out["per_category"]


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        queue_fate_bench(tmp_path)
