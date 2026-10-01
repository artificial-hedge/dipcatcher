"""shield_decay bench — kernel-shape discrimination for the shield."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

import quant_fund.microstructure.shield_decay_bench as m
from quant_fund.microstructure.aftermath_flow_bench import _FlowSim
from quant_fund.microstructure.shield_decay_bench import (
    _GRID,
    SHIELD_DECAY_SCHEMA,
    shield_decay_bench,
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


def test_decay_bit_identical_at_zero() -> None:
    a = ZILobSimulator(santa_fe_config(seed=3))
    b = ZILobSimulator(dataclasses.replace(santa_fe_config(seed=3), cxl_unhit_damp_decay=0.0))
    _drive(a, 1500)
    _drive(b, 1500)
    assert a.n_events == b.n_events and a.n_fills == b.n_fills


def test_decay_weakens_suppression_over_time() -> None:
    """Decayed suppression: late in-window cancels on the unhit side
    survive more often than under flat damp."""
    kw: dict[str, Any] = {
        "theta_cxl": 0.4,
        "mu": 2.0,
        "seed": 11,
        "cxl_unhit_damp": 0.8,
        "cxl_unhit_window": 60,
    }
    flat = _FlowSim(ZILobConfig(**kw), None)
    decayed = _FlowSim(ZILobConfig(**kw, cxl_unhit_damp_decay=12.0), None)
    _drive(flat, 30000)
    _drive(decayed, 30000)

    def _late_unhit_cxls(sim: _FlowSim) -> int:
        last_hit: str | None = None
        fill_ev = -(10**9)
        n = 0
        for kind, side, _lvl, ev in sim.flow_log:
            if kind == "fill":
                last_hit = side
                fill_ev = ev
                continue
            if (
                kind == "cxl"
                and last_hit is not None
                and 30 <= ev - fill_ev <= 60
                and side != last_hit
            ):
                n += 1
        return n

    assert _late_unhit_cxls(decayed) > _late_unhit_cxls(flat)


def test_decay_knob_validated() -> None:
    with pytest.raises(ValueError, match="cxl_unhit_damp must be a probability"):
        ZILobConfig(cxl_unhit_damp=-0.1)
    with pytest.raises(ValueError, match="cxl_unhit_damp_decay"):
        ZILobConfig(cxl_unhit_damp=0.5, cxl_unhit_damp_decay=-1.0)


def test_bench_seals(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_cell(
        relief: float,
        damp: float,
        window: int,
        *,
        horizon: int,
        seed: int,
        damp_decay: float = 0.0,
    ) -> dict[str, Any]:
        good_shape = (damp_decay == 0.0 and 0 < window <= 25 and damp >= 0.4) or damp_decay > 0.0
        ratios = (
            {"1_10": 0.55, "10_50": 0.85, "50_200": 0.92}
            if good_shape
            else {"1_10": 0.55, "10_50": 0.70, "50_200": 0.80}
        )
        win = {
            w: {
                "unhit_channel_rates": {"add": 10.0, "cxl": 10.0 * ratios[w]},
                "hit_channel_rates": {"add": 8.0, "cxl": 8.0, "fill": 1.0},
                "unhit_net_per_fill": 0.9,
            }
            for w in ratios
        }
        return {
            "relief": relief,
            "damp": damp,
            "damp_decay": damp_decay,
            "window": window,
            "n_fills": 10,
            "unhit_cxl_per_add": 0.6,
            "hit_cxl_per_fill": 1.6,
            "unhit_net_per_fill": 0.9,
            "instant_signed_ticks": 0.85 if good_shape else 0.7,
            "k200_ticks": 3.0,
            "windows": win,
        }

    monkeypatch.setattr(m, "_shield_cell", _fake_cell)
    out = shield_decay_bench(horizon=100, seed=1)
    assert out["schema"] == SHIELD_DECAY_SCHEMA
    assert out["claims"]["short_shield_matches"]
    assert out["claims"]["decay_kernel_matches"]
    assert out["claims"]["long_flat_overshields_late"]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    assert verify_receipt_file(p)["valid"]


def test_grid_starts_at_zero_cell() -> None:
    assert _GRID[0] == (0.0, 0.0, 0)
