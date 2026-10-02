"""aftermath_flow bench — post-fill channel decomposition + relief knob."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.aftermath_flow_bench as m
from quant_fund.microstructure.aftermath_flow_bench import (
    AFTERMATH_SCHEMA,
    _FlowSim,
    aftermath_flow_bench,
    lobster_aftermath,
    sim_aftermath,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.research.receipt_v2 import verify_receipt_file

_EX = 4
_SUB = 1


def _write_tape(d: Path) -> None:
    """Minimal LOBSTER tape: seeded book + fills so anchoring works."""
    rows = [
        [0.10, _SUB, 1, 50, 9900, -1],
        [0.11, _SUB, 2, 50, 10100, 1],
        [0.12, _SUB, 3, 30, 9800, -1],
        [0.13, _SUB, 4, 30, 10200, 1],
    ]
    oid = 5
    for i in range(6):
        rows.append([0.2 + 0.001 * i, _EX, 1, 5, 9900, -1])
        rows.append([0.3 + 0.001 * i, _SUB, oid, 10, 9900 + i % 3, -1])
        oid += 1
        rows.append([0.4 + 0.001 * i, 2, 2, 5, 10100, 1])
    msg = d / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    ob = d / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with ob.open("w", newline="") as fh:
        for _ in rows:
            csv.writer(fh).writerow([10100, 30, 9900, 30])


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def test_relief_bit_identical_at_zero() -> None:
    a = ZILobSimulator(santa_fe_config(seed=3))
    b = ZILobSimulator(dataclasses.replace(santa_fe_config(seed=3), cxl_unhit_relief=0.0))
    _drive(a, 1500)
    _drive(b, 1500)
    assert a.n_events == b.n_events and a.n_fills == b.n_fills


def test_relief_moves_cancel_mass_hit_side() -> None:
    """relief=1.0: every post-fill-window cancel lands on the hit side."""
    cfg = ZILobConfig(
        cxl_unhit_relief=1.0,
        cxl_unhit_window=50,
        theta_cxl=0.4,
        mu=2.0,
        seed=11,
    )
    sim = _FlowSim(cfg, None)
    _drive(sim, 30000)
    # last fill's resting side before each logged event
    last_hit: str | None = None
    hit_cxl = unhit_cxl = 0
    fill_ev = -(10**9)
    for kind, side, _lvl, ev in sim.flow_log:
        if kind == "fill":
            last_hit = side
            fill_ev = ev
            continue
        if kind != "cxl" or last_hit is None or ev - fill_ev > 50:
            continue
        if side == last_hit:
            hit_cxl += 1
        else:
            unhit_cxl += 1
    # Without relief the post-fill window is unhit-heavy (the hit book is
    # often empty after a sweep); relief=1.0 must flip that to hit-side.
    assert hit_cxl > unhit_cxl


def test_relief_knobs_validated() -> None:
    with pytest.raises(ValueError, match="cxl_unhit_relief"):
        ZILobConfig(cxl_unhit_relief=1.5)
    with pytest.raises(ValueError, match="cxl_unhit_window"):
        ZILobConfig(cxl_unhit_window=-1)


def test_hit_retreat_marker_carries_both_deadlines() -> None:
    cfg = ZILobConfig(
        hit_narrow_dist=2,
        hit_narrow_window=10,
        cxl_unhit_relief=0.5,
        cxl_unhit_window=80,
        mu=3.0,
        seed=5,
    )
    sim = ZILobSimulator(cfg)
    _drive(sim, 20000)
    if sim._hit_retreat is not None:  # noqa: SLF001
        assert len(sim._hit_retreat) == 8  # noqa: SLF001


def test_lobster_aftermath_shape(tmp_path: Path) -> None:
    _write_tape(tmp_path)
    out = lobster_aftermath(
        tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv",
        tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
    )
    assert out["ok"]
    assert out["windows"]["1_10"]["n_anchor_fills"] > 0
    assert set(out["windows"]) == {"1_10", "10_50", "50_200"}


def test_sim_aftermath_shape() -> None:
    out = sim_aftermath(santa_fe_config(seed=7), None, 4000)
    assert set(out["windows"]) == {"1_10", "10_50", "50_200"}


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aftermath_flow_bench(tmp_path, "AMZN", horizon=2000)


def test_bench_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tape(tmp_path)

    def _fake_sim(cfg: Any, flow: Any, horizon: int) -> dict[str, Any]:
        cells = {
            w: {
                "hit_net_per_fill": -1.0,
                "unhit_net_per_fill": 0.9,
                "hit_channel_rates": {"add": 1.5, "cxl": 1.0, "fill": 0.5},
                "unhit_channel_rates": {"add": 2.5, "cxl": 1.0, "fill": 0.1},
            }
            for w in ("1_10", "10_50", "50_200")
        }
        return {"ok": True, "n_fills": 10, "windows": cells}

    monkeypatch.setattr(m, "sim_aftermath", _fake_sim)
    out = aftermath_flow_bench(tmp_path, "AMZN", horizon=100)
    assert out["schema"] == AFTERMATH_SCHEMA
    assert out["claims"]["sim_rate_channel_achieved"]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]
