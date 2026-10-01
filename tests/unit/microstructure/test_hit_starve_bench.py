"""hit_starve bench — hit-side refill suppression mechanism probe."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.hit_starve_bench as m
from quant_fund.microstructure.hit_starve_bench import HIT_STARVE_SCHEMA, hit_starve_bench
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_file


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def test_starve_bit_identical_at_zero() -> None:
    c0 = santa_fe_config(seed=11)
    c1 = dataclasses.replace(
        santa_fe_config(seed=11), hit_refill_damp=0.0, hit_refill_band=5, hit_refill_window=50
    )
    s0, s1 = ZILobSimulator(c0), ZILobSimulator(c1)
    _drive(s0, 3000)
    _drive(s1, 3000)
    assert s0.n_fills == s1.n_fills
    assert s0.n_lo_arrivals == s1.n_lo_arrivals
    assert s0.n_events == s1.n_events


def test_starve_suppresses_hit_side_arrivals() -> None:
    cfg = dataclasses.replace(
        santa_fe_config(seed=11), hit_refill_damp=0.9, hit_refill_band=5, hit_refill_window=50
    )
    sim = ZILobSimulator(cfg)
    _drive(sim, 3000)
    assert sim.n_lo_suppressed > 0
    assert sim.n_fills > 0


def test_starve_window_scales_suppression() -> None:
    short = dataclasses.replace(
        santa_fe_config(seed=11), hit_refill_damp=1.0, hit_refill_band=5, hit_refill_window=5
    )
    long = dataclasses.replace(
        santa_fe_config(seed=11), hit_refill_damp=1.0, hit_refill_band=5, hit_refill_window=200
    )
    s5, s200 = ZILobSimulator(short), ZILobSimulator(long)
    _drive(s5, 3000)
    _drive(s200, 3000)
    assert 0 < s5.n_lo_suppressed < s200.n_lo_suppressed


def test_starve_knobs_validated() -> None:
    with pytest.raises(ValueError, match="hit_refill_damp"):
        dataclasses.replace(santa_fe_config(seed=0), hit_refill_damp=-0.1)
    with pytest.raises(ValueError, match="hit_refill_band"):
        dataclasses.replace(santa_fe_config(seed=0), hit_refill_band=-1)
    with pytest.raises(ValueError, match="hit_refill_window"):
        dataclasses.replace(santa_fe_config(seed=0), hit_refill_window=-3)


def test_bench_seals(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    base_cell = {
        "hit_refill_damp": 0.0,
        "hit_refill_band": 0,
        "hit_refill_window": 0,
        "cxl_unhit_damp": 0.0,
        "cxl_unhit_window": 0,
        "n_fills": 10,
        "n_lo_suppressed": 5,
        "instant_signed_ticks": 0.9,
        "k200_ticks": 3.0,
        "k200_per_channel_ticks": {"fill": 4.8, "lo": -2.5, "cxl": -0.2, "none": 0.0},
        "lo_channel_ticks": -2.5,
        "attr_windows": {},
    }
    starve_cell = dict(
        base_cell,
        hit_refill_damp=0.5,
        hit_refill_band=5,
        hit_refill_window=50,
        n_lo_suppressed=100,
        instant_signed_ticks=0.95,
        k200_ticks=2.8,
        k200_per_channel_ticks={"fill": 6.0, "lo": -3.8, "cxl": -0.25, "none": 0.0},
        lo_channel_ticks=-3.8,
    )
    grid = [base_cell] + [dict(starve_cell) for _ in m._GRID[1:]]
    it = iter(grid)
    monkeypatch.setattr(m, "_starve_cell", lambda *a, **k: next(it))
    out = hit_starve_bench(horizon=50, seed=3)
    assert out["schema"] == HIT_STARVE_SCHEMA
    assert out["claims"]["starve_grid_evaluated"]
    assert out["claims"]["starve_deepens_lo_gap"]
    assert out["claims"]["joint_closure_absent"]
    assert out["data_label"] == "SYNTHETIC"
    p = Path(tmp_path) / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_grid_starts_at_zero_cell() -> None:
    assert m._GRID[0] == (0.0, 0, 0, 0.0, 0)
