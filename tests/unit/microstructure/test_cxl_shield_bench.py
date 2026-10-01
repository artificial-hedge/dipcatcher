"""cxl_shield bench — joint channel+price score of the shield knobs."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

import quant_fund.microstructure.cxl_shield_bench as m
from quant_fund.microstructure.aftermath_flow_bench import _FlowSim
from quant_fund.microstructure.cxl_shield_bench import (
    _GRID,
    CXLSHIELD_SCHEMA,
    _shield_cell,
    cxl_shield_bench,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.research.receipt_v2 import verify_receipt_file


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def test_damp_bit_identical_at_zero() -> None:
    a = ZILobSimulator(santa_fe_config(seed=3))
    b = ZILobSimulator(dataclasses.replace(santa_fe_config(seed=3), cxl_unhit_damp=0.0))
    _drive(a, 1500)
    _drive(b, 1500)
    assert a.n_events == b.n_events and a.n_fills == b.n_fills


def test_damp_suppresses_unhit_cancels() -> None:
    """damp=1.0 with a window: in-window cxl events on the unhit side
    are suppressed — the unhit-side cxl count drops vs the baseline."""
    base = _FlowSim(ZILobConfig(theta_cxl=0.4, mu=2.0, seed=11), None)
    damped = _FlowSim(
        ZILobConfig(
            theta_cxl=0.4,
            mu=2.0,
            seed=11,
            cxl_unhit_damp=1.0,
            cxl_unhit_window=50,
        ),
        None,
    )
    _drive(base, 30000)
    _drive(damped, 30000)

    def _unhit_cxls(sim: _FlowSim) -> int:
        last_hit: str | None = None
        fill_ev = -(10**9)
        n = 0
        for kind, side, _lvl, ev in sim.flow_log:
            if kind == "fill":
                last_hit = side
                fill_ev = ev
                continue
            if kind == "cxl" and last_hit is not None and ev - fill_ev <= 50 and side != last_hit:
                n += 1
        return n

    assert _unhit_cxls(damped) < _unhit_cxls(base)


def test_damp_only_fires_inside_window() -> None:
    """Past the window the marker is dead — suppression stops."""
    sim = _FlowSim(
        ZILobConfig(
            theta_cxl=0.4,
            mu=2.0,
            seed=5,
            cxl_unhit_damp=1.0,
            cxl_unhit_window=5,
        ),
        None,
    )
    _drive(sim, 30000)
    last_hit: str | None = None
    fill_ev = -(10**9)
    late_unhit = 0
    for kind, side, _lvl, ev in sim.flow_log:
        if kind == "fill":
            last_hit = side
            fill_ev = ev
            continue
        if kind == "cxl" and last_hit is not None and ev - fill_ev > 5 and side != last_hit:
            late_unhit += 1
    assert late_unhit > 0


def test_damp_knobs_validated() -> None:
    with pytest.raises(ValueError, match="cxl_unhit_damp"):
        ZILobConfig(cxl_unhit_damp=1.5)
    with pytest.raises(ValueError, match="cxl_unhit_window"):
        ZILobConfig(cxl_unhit_damp=0.5, cxl_unhit_window=-1)


def test_shield_cell_shape() -> None:
    cell = _shield_cell(0.0, 0.3, 100, horizon=3000, seed=7)
    assert set(cell["windows"]) == {"1_10", "10_50", "50_200"}
    assert cell["n_fills"] > 0
    assert cell["instant_signed_ticks"] is not None


def test_bench_seals(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_cell(
        relief: float, damp: float, window: int, *, horizon: int, seed: int
    ) -> dict[str, Any]:
        ok_channel = damp > 0 and relief == 0.0
        return {
            "relief": relief,
            "damp": damp,
            "window": window,
            "n_fills": 10,
            "unhit_cxl_per_add": 0.60 if ok_channel else 0.9,
            "hit_cxl_per_fill": 1.6,
            "unhit_net_per_fill": 0.9,
            "instant_signed_ticks": 0.9 if ok_channel else 1.3,
            "k200_ticks": 3.0,
            "windows": {},
        }

    monkeypatch.setattr(m, "_shield_cell", _fake_cell)
    out = cxl_shield_bench(horizon=100, seed=1)
    assert out["schema"] == CXLSHIELD_SCHEMA
    assert out["claims"]["joint_closure_exists"]
    assert out["joint_closure_cells"]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_grid_starts_at_zero_cell() -> None:
    assert _GRID[0] == (0.0, 0.0, 0)
