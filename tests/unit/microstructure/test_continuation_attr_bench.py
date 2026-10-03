"""continuation_attr bench — channel attribution of the k200 drift."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.continuation_attr_bench as m
from quant_fund.microstructure.continuation_attr_bench import (
    CONTINUATION_ATTR_SCHEMA,
    _attr_totals,
    continuation_attr_bench,
    lobster_attr,
    sim_attr,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_file

_EX = 4
_SUB = 1


def _write_tape(d: Path, *, execs: int = 8) -> tuple[Path, Path]:
    """Minimal tape: seeded book, then execs with a mid uptick after."""
    rows = [
        [0.10, _SUB, 1, 50, 9900, -1],
        [0.11, _SUB, 2, 50, 10100, 1],
    ]
    for i in range(execs):
        rows.append([0.2 + 0.001 * i, _EX, 1, 5, 10100, 1])
        rows.append([0.3 + 0.001 * i, _SUB, 10 + i, 10, 10050, -1])
    msg = d / "m.csv"
    with msg.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    ob = d / "o.csv"
    with ob.open("w", newline="") as fh:
        for r in rows:
            ask = 10100 + (50 if r[1] == _EX else 0)
            csv.writer(fh).writerow([ask, 30, 9900, 30])
    return msg, ob


def test_attr_totals_sums_once_per_anchor() -> None:
    # Two anchors 5 events apart; event 7's Δmid belongs to both windows.
    records = [
        (10, 1.0, 12, "fill", "hit", 2.0),
        (10, 1.0, 30, "lo", "unhit", 1.0),
        (12, -1.0, 30, "lo", "unhit", 1.0),
        (12, -1.0, 60, "cxl", "hit", 0.5),
    ]
    out = _attr_totals(records)
    assert out["n_anchor_fills"] == 2
    assert out["k200_per_channel_ticks"]["fill"] == pytest.approx(1.0)
    assert out["k200_per_channel_ticks"]["lo"] == pytest.approx(0.0)
    assert out["windows"]["1_10"]["per_channel_ticks"]["fill"]["total"] == pytest.approx(2.0)


def test_lobster_attr_shape(tmp_path: Path) -> None:
    msg, ob = _write_tape(tmp_path)
    out = lobster_attr(msg, ob)
    assert out["ok"] and out["n_events"] > 0
    assert set(out["windows"]) == {"1_10", "10_50", "50_200"}
    # The fixture's exec rows move the ask — attribution must be nonzero.
    assert out["k200_signed_ticks"] != 0.0


def test_sim_attr_shape() -> None:
    out = sim_attr(santa_fe_config(seed=7), None, 4000)
    assert out["n_anchor_fills"] > 0
    assert set(out["k200_per_channel_ticks"]) == {"fill", "lo", "cxl", "none"}


def test_bench_seals(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pane = {
        "ok": True,
        "n_events": 100,
        "n_anchor_fills": 50,
        "windows": {},
        "k200_per_channel_ticks": {"fill": 2.5, "lo": 2.8, "cxl": -0.7, "none": 0.0},
        "k200_signed_ticks": 4.6,
        "positive_channel_shares": {"fill": 0.47, "lo": 0.53, "cxl": -0.13, "none": 0.0},
        "positive_share_sum": 1.0,
    }
    sim_pane = {
        "ok": True,
        "n_events": 100,
        "n_anchor_fills": 40,
        "windows": {},
        "k200_per_channel_ticks": {"fill": 4.9, "lo": -2.5, "cxl": -0.2, "none": 0.0},
        "k200_signed_ticks": 2.2,
        "positive_channel_shares": {"fill": 1.0, "lo": -0.5, "cxl": -0.04, "none": 0.0},
        "positive_share_sum": 1.0,
    }
    monkeypatch.setattr(m, "lobster_attr", lambda *a: fake_pane)
    monkeypatch.setattr(m, "sim_attr", lambda *a: sim_pane)
    out = continuation_attr_bench(tmp_path / "m.csv", tmp_path / "o.csv", horizon=100)
    assert out["schema"] == CONTINUATION_ATTR_SCHEMA
    assert all(out["claims"].values())
    assert out["data_label"] == "MIXED"
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_missing_tape_falls_back_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        m,
        "sim_attr",
        lambda *a: {
            "ok": True,
            "n_events": 10,
            "n_anchor_fills": 2,
            "windows": {},
            "k200_per_channel_ticks": {"fill": 0.1, "lo": 0.1, "cxl": 0.0, "none": 0.0},
            "k200_signed_ticks": 0.2,
            "positive_channel_shares": {"fill": 0.5, "lo": 0.5, "cxl": 0.0, "none": 0.0},
            "positive_share_sum": 1.0,
        },
    )
    out = continuation_attr_bench(None, None, horizon=10)
    assert out["data_label"] == "SYNTHETIC"
    assert out["claims"]["drift_attributed"]
