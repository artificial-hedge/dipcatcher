"""Tests for reload_gate_bench (iceberg_reload_mode gate)."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.microstructure.place_law_bench import _calibrated
from quant_fund.microstructure.reload_gate_bench import (
    _CELLS,
    RELOAD_GATE_SCHEMA,
    reload_gate_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator


def test_reload_mode_validates() -> None:
    with pytest.raises(ValueError, match="iceberg_reload_mode"):
        _calibrated(3, {"iceberg_reload_mode": "bogus"})


def test_per_unit_default_bit_identical() -> None:
    """reload_mode default leaves the trajectory untouched."""
    base = dict(_DEEP, iceberg_reload=0.5)
    a = _sim_crown("a", dict(base), horizon=1500, seed=9, collect_counts=True)
    b = _sim_crown(
        "b",
        dict(base, iceberg_reload_mode="per_unit"),
        horizon=1500,
        seed=9,
        collect_counts=True,
    )
    a.pop("regime")
    b.pop("regime")
    assert a == b


def test_residual_gate_blocks_cleared_level() -> None:
    """Deterministic probe: a fill that clears its level refills under
    per_unit and stays empty under residual."""
    for mode, survives in (("per_unit", True), ("residual", False)):
        cfg = _calibrated(3, {"iceberg_reload": 1.0, "iceberg_reload_mode": mode})
        sim = ZILobSimulator(cfg)
        sim._asks.clear()  # noqa: SLF001 — mechanism probe
        sim._bids.clear()  # noqa: SLF001
        sim._orders.clear()  # noqa: SLF001
        sim._rest("buy", 90, "manual")  # noqa: SLF001
        sim._rest("sell", 100, "manual")  # noqa: SLF001 — lone ask touch
        tr = sim._consume_best("buy")  # noqa: SLF001
        assert tr is not None and tr.level == 100
        assert (100 in sim._asks) is survives  # noqa: SLF001


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = reload_gate_bench(horizon=3000, seed=5)
    assert out["schema"] == RELOAD_GATE_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "residual_reopens_empty",
        "hidden_survives_gate",
        "gate_decouples_hidden_from_empty",
        "joint_gate_cell",
    }
    assert out["receipt_sha256"]


def test_sim_accepts_mode() -> None:
    cfg = _calibrated(3, {"iceberg_reload": 0.3, "iceberg_reload_mode": "residual"})
    sim = ZILobSimulator(cfg)
    for _ in range(50):
        sim.step()
    assert sim.n_fills >= 0
