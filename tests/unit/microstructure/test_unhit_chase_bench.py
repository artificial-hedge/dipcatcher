"""unhit_chase bench — unhit-side post-fill touch-chase mechanism."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.unhit_chase_bench as m
from quant_fund.microstructure.unhit_chase_bench import (
    UNHIT_CHASE_SCHEMA,
    unhit_chase_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_file


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def test_chase_bit_identical_at_zero() -> None:
    c0 = santa_fe_config(seed=11)
    c1 = dataclasses.replace(santa_fe_config(seed=11), unhit_imp_frac=0.0, unhit_imp_window=100)
    s0, s1 = ZILobSimulator(c0), ZILobSimulator(c1)
    _drive(s0, 3000)
    _drive(s1, 3000)
    assert s0.n_fills == s1.n_fills
    assert s0.n_lo_arrivals == s1.n_lo_arrivals
    assert s0.n_events == s1.n_events
    assert s0.n_lo_improve == s1.n_lo_improve


def test_unhit_chase_levels() -> None:
    cfg = dataclasses.replace(santa_fe_config(seed=11), unhit_imp_frac=1.0, unhit_imp_window=500)
    sim = ZILobSimulator(cfg)
    _drive(sim, 2000)
    sim.inject_market_order("buy")
    # Marker armed with hit="sell"; unhit side = "buy" chases inside spread.
    assert sim._hit_retreat is not None and sim._hit_retreat[0] == "sell"
    bb, ba = sim.best_bid_level, sim.best_ask_level
    assert bb is not None and ba is not None
    level = sim._unhit_chase("buy")
    assert level is not None and bb <= level < ba
    assert sim._unhit_chase("sell") is None  # hit side never rerouted


def test_chase_fires_post_fill() -> None:
    cfg = dataclasses.replace(santa_fe_config(seed=11), unhit_imp_frac=0.7, unhit_imp_window=200)
    sim = ZILobSimulator(cfg)
    _drive(sim, 4000)
    assert sim.n_fills > 0
    assert sim.n_lo_improve > 0


def test_chase_never_crosses_spread() -> None:
    cfg = dataclasses.replace(santa_fe_config(seed=11), unhit_imp_frac=1.0, unhit_imp_window=500)
    sim = ZILobSimulator(cfg)
    for _ in range(4000):
        sim.step()
        ba, bb = sim.best_ask_level, sim.best_bid_level
        if ba is not None and bb is not None:
            assert bb < ba


def test_chase_knobs_validated() -> None:
    with pytest.raises(ValueError, match="unhit_imp_frac"):
        dataclasses.replace(santa_fe_config(seed=0), unhit_imp_frac=-0.1)
    with pytest.raises(ValueError, match="unhit_imp_frac"):
        dataclasses.replace(santa_fe_config(seed=0), unhit_imp_frac=1.1)
    with pytest.raises(ValueError, match="unhit_imp_window"):
        dataclasses.replace(santa_fe_config(seed=0), unhit_imp_window=-3)


def test_bench_seals(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    base_cell = {
        "unhit_imp_frac": 0.0,
        "unhit_imp_window": 0,
        "cxl_unhit_damp": 0.0,
        "cxl_unhit_window": 0,
        "n_fills": 10,
        "n_lo_improve": 5,
        "n_lo_suppressed": 5,
        "instant_signed_ticks": 0.9,
        "k200_ticks": 3.0,
        "k200_per_channel_ticks": {"fill": 4.8, "lo": -2.5, "cxl": -0.2, "none": 0.0},
        "lo_channel_ticks": -2.5,
        "fill_channel_ticks": 4.8,
        "attr_windows": {},
    }
    chase_cell = dict(
        base_cell,
        unhit_imp_frac=0.5,
        unhit_imp_window=200,
        n_lo_improve=100,
        instant_signed_ticks=0.95,
        k200_ticks=4.2,
        k200_per_channel_ticks={"fill": 4.6, "lo": 2.7, "cxl": -0.2, "none": 0.0},
        lo_channel_ticks=2.7,
        fill_channel_ticks=4.6,
    )
    grid = [base_cell] + [dict(chase_cell) for _ in m._GRID[1:]]
    it = iter(grid)
    monkeypatch.setattr(m, "_chase_cell", lambda *a, **k: next(it))
    out = unhit_chase_bench(horizon=50, seed=3)
    assert out["schema"] == UNHIT_CHASE_SCHEMA
    assert out["claims"]["chase_grid_evaluated"]
    assert out["claims"]["lo_channel_lifts"]
    assert out["claims"]["lo_channel_in_tol_exists"]
    assert out["claims"]["joint_closure_exists"]
    assert out["claims"]["fill_channel_not_worsened"]
    assert out["data_label"] == "SYNTHETIC"
    p = Path(tmp_path) / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_grid_starts_at_zero_cell() -> None:
    assert m._GRID[0] == (0.0, 0, 0.0, 0)
