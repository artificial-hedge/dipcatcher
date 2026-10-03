"""closure_stack bench — joint instant×chase closure capstone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import quant_fund.microstructure.closure_stack_bench as m
from quant_fund.microstructure.closure_stack_bench import (
    CLOSURE_STACK_SCHEMA,
    closure_stack_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_file


def _cell(cd: int, frac: float, **over: Any) -> dict[str, Any]:
    cell = {
        "refill_cooldown": cd,
        "unhit_imp_frac": frac,
        "unhit_imp_window": 200,
        "cxl_unhit_damp": 0.0,
        "cxl_unhit_window": 0,
        "n_fills": 10,
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


def test_grid_starts_at_zero_cell() -> None:
    assert m._GRID[0] == (0, 0.0, 0, 0.0, 0)


def test_bench_seals_and_scores(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fabricate a grid where one stacked cell lands every channel in tol.
    grid = []
    for i, (cd, f, iw, _cd2, _cw) in enumerate(m._GRID):
        if i == 6:
            grid.append(
                _cell(
                    cd,
                    f,
                    refill_cooldown=cd,
                    unhit_imp_window=iw,
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
            grid.append(_cell(cd, f, instant_signed_ticks=0.5, lo_channel_ticks=-0.5))
        elif f > 0.0:
            grid.append(_cell(cd, f, lo_channel_ticks=2.9))
        else:
            grid.append(_cell(cd, f))
    it = iter(grid)
    monkeypatch.setattr(m, "_stack_cell", lambda *a, **k: next(it))
    out = closure_stack_bench(horizon=50, seed=3)
    assert out["schema"] == CLOSURE_STACK_SCHEMA
    assert out["claims"]["stack_grid_evaluated"]
    assert out["claims"]["cooldown_lifts_instant"]
    assert out["claims"]["joint_closure_exists"]
    assert out["n_cells_all_in_tol"] == 1
    assert out["best_cell"]["n_in_tol"] == 5
    assert out["data_label"] == "SYNTHETIC"
    p = Path(tmp_path) / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_bench_honest_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    # No cell in tol → joint_closure_exists must be false.
    monkeypatch.setattr(m, "_stack_cell", lambda *a, **k: _cell(0, 0.0))
    out = closure_stack_bench(horizon=50, seed=3)
    assert out["claims"]["joint_closure_exists"] is False
    assert out["n_cells_all_in_tol"] == 0
