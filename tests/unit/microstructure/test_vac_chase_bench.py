"""vac_chase bench — vacancy-coupled post-fill chase."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.vac_chase_bench as m
from quant_fund.microstructure.vac_chase_bench import (
    VAC_CHASE_SCHEMA,
    vac_chase_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.research.receipt_v2 import verify_receipt_file


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def test_vac_chase_bit_identical_at_zero() -> None:
    c0 = santa_fe_config(seed=11)
    c1 = dataclasses.replace(santa_fe_config(seed=11), vac_chase_frac=0.0, vac_chase_window=200)
    s0, s1 = ZILobSimulator(c0), ZILobSimulator(c1)
    _drive(s0, 3000)
    _drive(s1, 3000)
    assert s0.n_fills == s1.n_fills
    assert s0.n_lo_arrivals == s1.n_lo_arrivals
    assert s0.n_events == s1.n_events
    assert s0.n_lo_improve == s1.n_lo_improve


def test_vac_chase_fires_only_while_vacant() -> None:
    cfg = dataclasses.replace(santa_fe_config(seed=11), vac_chase_frac=1.0, vac_chase_window=300)
    sim = ZILobSimulator(cfg)
    assert sim._vac_chase("buy") is None  # no vacancy yet
    _drive(sim, 2000)
    # Buy injections until an ask level empties → sell-side vacancy.
    for _ in range(50):
        sim.inject_market_order("buy")
        if sim._vacancy:
            break
    assert sim._vacancy
    bb, ba = sim.best_bid_level, sim.best_ask_level
    assert bb is not None and ba is not None
    level = sim._vac_chase("buy")
    assert level is not None and bb <= level < ba


def test_vac_chase_knobs_validated() -> None:
    with pytest.raises(ValueError, match="vac_chase_frac"):
        dataclasses.replace(santa_fe_config(seed=0), vac_chase_frac=-0.1)
    with pytest.raises(ValueError, match="vac_chase_frac"):
        dataclasses.replace(santa_fe_config(seed=0), vac_chase_frac=1.1)
    with pytest.raises(ValueError, match="vac_chase_window"):
        dataclasses.replace(santa_fe_config(seed=0), vac_chase_window=-3)


def _cell(cd: int, frac: float, **over: Any) -> dict[str, Any]:
    cell = {
        "refill_cooldown": cd,
        "vac_chase_frac": frac,
        "vac_chase_window": 200,
        "cxl_unhit_damp": 0.0,
        "cxl_unhit_window": 0,
        "n_fills": 10,
        "n_lo_improve": 5,
        "instant_signed_ticks": 0.7,
        "k200_ticks": 3.0,
        "k200_per_channel_ticks": {"fill": 4.0, "lo": -2.0, "cxl": -0.2, "none": 0.0},
        "lo_channel_ticks": -2.0,
        "fill_channel_ticks": 4.0,
        "cxl_channel_ticks": -0.2,
        "attr_windows": {},
    }
    cell.update(over)
    return cell


def test_bench_seals_and_scores(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fabricate a grid where one stacked cell lands every channel in tol
    # and stacked cooldown+chase cells preserve the instant gain.
    grid = []
    for i, (cd, f, _vw, _cd2, _cw) in enumerate(m._GRID):
        if i == 5:
            grid.append(
                _cell(
                    cd,
                    f,
                    instant_signed_ticks=0.90,
                    k200_ticks=4.4,
                    lo_channel_ticks=2.9,
                    fill_channel_ticks=2.6,
                    cxl_channel_ticks=-0.7,
                )
            )
        elif cd > 0 and f == 0.0:
            grid.append(_cell(cd, f, instant_signed_ticks=0.91))
        elif f > 0.0 and cd > 0:
            grid.append(_cell(cd, f, instant_signed_ticks=0.85, lo_channel_ticks=1.5))
        elif f > 0.0:
            grid.append(_cell(cd, f, lo_channel_ticks=2.5))
        else:
            grid.append(_cell(cd, f))
    it = iter(grid)
    monkeypatch.setattr(m, "_vac_cell", lambda *a, **k: next(it))
    out = vac_chase_bench(horizon=50, seed=3)
    assert out["schema"] == VAC_CHASE_SCHEMA
    assert out["claims"]["grid_evaluated"]
    assert out["claims"]["chase_still_lifts_lo"]
    assert out["claims"]["vacancy_coupling_preserves_instant"]
    assert out["claims"]["joint_closure_exists"]
    assert out["n_cells_all_in_tol"] == 1
    assert out["data_label"] == "SYNTHETIC"
    p = Path(tmp_path) / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_bench_honest_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_vac_cell", lambda *a, **k: _cell(0, 0.0))
    out = vac_chase_bench(horizon=50, seed=3)
    assert out["claims"]["joint_closure_exists"] is False
    assert out["claims"]["chase_still_lifts_lo"] is False
    assert out["n_cells_all_in_tol"] == 0


def test_grid_starts_at_zero_cell() -> None:
    assert m._GRID[0] == (0, 0.0, 0, 0.0, 0)
