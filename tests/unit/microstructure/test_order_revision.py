from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.lobster import DELETE, SUBMISSION, LobsterEvent
from quant_fund.microstructure.order_revision import (
    _link_revisions,
    lobster_revision_stats,
    revision_bench,
    sim_revision_stats,
)


def test_link_revisions_nearest() -> None:
    cancels = [LobsterEvent(1.0, DELETE, 1, 10, 1000, 1)]
    submits = [
        LobsterEvent(1.05, SUBMISSION, 2, 10, 1003, 1),  # +3 units, closer
        LobsterEvent(1.04, SUBMISSION, 3, 10, 1010, 1),  # earlier but farther
        LobsterEvent(1.02, SUBMISSION, 4, 10, 1000, -1),  # wrong side
    ]
    lats, steps, dirs = _link_revisions(cancels, submits, 0.5)
    assert dirs.size == 1
    assert steps[0] == pytest.approx(3.0)
    assert lats[0] == pytest.approx(0.05)


def test_link_revisions_window_cutoff() -> None:
    cancels = [LobsterEvent(1.0, DELETE, 1, 10, 1000, 1)]
    submits = [LobsterEvent(1.6, SUBMISSION, 2, 10, 1000, 1)]  # outside w=0.5
    _, _, dirs = _link_revisions(cancels, submits, 0.5)
    assert dirs.size == 0


def test_lobster_revision_stats_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        # 80 deletes at distinct prices, each followed by a +1-tick resubmit
        t = 34200.0
        oid = 0
        for i in range(80):
            t += 0.01
            oid += 1
            w.writerow([t, 3, oid, 10, 4000 + i * 100, 1])  # delete
            t += 0.002
            oid += 1
            w.writerow([t, 1, oid, 10, 4000 + i * 100 + 100, 1])  # resubmit +1 tick
    out = lobster_revision_stats(msg)
    assert out["ok"] and out["revision_rate"] > 0.9
    assert out["step_ticks_abs_median"] == pytest.approx(1.0)
    assert out["toward_mid_share"] == pytest.approx(1.0)  # buys re-quoting up


def test_sim_null_rate_small() -> None:
    out = sim_revision_stats(horizon=8000, seed=5)
    assert out["ok"]
    # accidental-link null should sit well below real-tape rates
    assert out["revision_rate"] < 0.6


def test_too_few_cancels(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    msg.write_text("34200.0,1,1,10,4000,1\n")
    out = lobster_revision_stats(msg)
    assert out["ok"] is False


def test_bench_missing_tape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        revision_bench(tmp_path)


def test_returns_ndarray() -> None:
    cancels = [LobsterEvent(0.0, DELETE, 1, 1, 100, 1)]
    submits = [LobsterEvent(0.1, SUBMISSION, 2, 1, 100, 1)]
    lats, steps, dirs = _link_revisions(cancels, submits, 0.5)
    assert isinstance(lats, np.ndarray) and isinstance(dirs, np.ndarray)
