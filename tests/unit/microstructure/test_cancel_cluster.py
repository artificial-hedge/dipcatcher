from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.cancel_cluster import (
    _lift,
    cancel_cluster_bench,
    lobster_cancel_cluster,
    sim_cancel_cluster,
)


def test_lift_clustered_cancels() -> None:
    # execs at t=0,10,...,200; cancels always 0.1s after each exec
    execs = [float(t) for t in range(0, 200, 10)]
    cxls = [t + 0.1 for t in execs] * 3
    cxls.sort()
    out = _lift(execs, cxls, 0.0, 210.0)
    assert out.get("ok", True) is not False
    assert out["lift_0.5s"] > 1.0  # cancels cluster right after execs


def test_lift_homogeneous_is_one() -> None:
    rng = np.random.default_rng(0)
    execs = np.sort(rng.uniform(0, 1000, 100)).tolist()
    cxls = np.sort(rng.uniform(0, 1000, 400)).tolist()
    out = _lift(execs, cxls, 0.0, 1000.0)
    assert out["lift_10s"] == pytest.approx(1.0, abs=0.4)


def test_lift_too_few() -> None:
    out = _lift([1.0, 2.0], [1.5, 2.5], 0.0, 10.0)
    assert out["ok"] is False


def test_lobster_cancel_cluster_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    rows = []
    for i in range(100):
        rows.append([34200.0 + i, 1, 10_000 + i, 50, 5000, -1])  # submissions
        if i % 10 == 0:
            rows.append([34200.0 + i + 0.5, 4, 10_000 + i, 50, 5000, -1])  # exec
            rows.append([34200.0 + i + 0.6, 3, 20_000 + i, 10, 5000, -1])  # delete
    with msg.open("w", newline="") as f:
        for r in rows:
            csv.writer(f).writerow(r)
    out = lobster_cancel_cluster(tmp_path)
    assert out["n_execs"] == 10
    assert out["n_cancels"] == 10


def test_sim_cancel_cluster_runs() -> None:
    out = sim_cancel_cluster(horizon=1500, seed=3)
    assert out["n_execs"] > 10
    assert "lift_10s" in out["both_sides"] or out["both_sides"]["ok"] is False


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cancel_cluster_bench(tmp_path)
