"""Tests for ice_budget_bench (finite per-level iceberg reserve)."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.microstructure.ice_budget_bench import (
    _CELLS,
    ICE_BUDGET_SCHEMA,
    ice_budget_bench,
)
from quant_fund.microstructure.place_law_bench import _calibrated
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator


def test_budget_validates() -> None:
    with pytest.raises(ValueError, match="iceberg_budget"):
        _calibrated(3, {"iceberg_budget": -1})


def test_budget_default_bit_identical() -> None:
    """budget=0 leaves the trajectory untouched."""
    base = dict(_DEEP, iceberg_reload=0.5, iceberg_reload_mode="residual")
    a = _sim_crown("a", dict(base), horizon=1500, seed=9, collect_counts=True)
    b = _sim_crown(
        "b",
        dict(base, iceberg_budget=0),
        horizon=1500,
        seed=9,
        collect_counts=True,
    )
    a.pop("regime")
    b.pop("regime")
    assert a == b


def test_budget_exhausts_per_level() -> None:
    """Deterministic probe: after ``iceberg_budget`` refills the level
    dies for real, even though unlimited mode would keep refilling."""
    budget = 2
    cfg = _calibrated(
        3,
        {
            "iceberg_reload": 1.0,
            "iceberg_reload_mode": "per_unit",
            "iceberg_budget": budget,
        },
    )
    sim = ZILobSimulator(cfg)
    sim._asks.clear()  # noqa: SLF001 — mechanism probe
    sim._bids.clear()  # noqa: SLF001
    sim._orders.clear()  # noqa: SLF001
    sim._rest("buy", 90, "manual")  # noqa: SLF001
    sim._rest("sell", 100, "manual")  # noqa: SLF001 — lone ask touch
    # budget + the original visible unit = budget+1 fills before death.
    for _ in range(budget):
        tr = sim._consume_best("buy")  # noqa: SLF001
        assert tr is not None and tr.level == 100
        assert 100 in sim._asks  # noqa: SLF001 — refill spent
    tr = sim._consume_best("buy")  # noqa: SLF001
    assert tr is not None and tr.level == 100
    assert 100 not in sim._asks  # noqa: SLF001 — reserve exhausted


def test_budget_caps_hidden_fills() -> None:
    """Over a run, hidden fills are bounded by budget x levels-touched —
    so a tiny budget must strictly reduce hidden fills vs unlimited."""
    base = dict(
        _DEEP,
        iceberg_reload=0.8,
        iceberg_reload_mode="per_unit",
        theta_cxl=1e-9,
        touch_pull=0.0,
    )
    unlim = _sim_crown("u", dict(base), horizon=3000, seed=11, collect_counts=True)
    capped = _sim_crown(
        "c", dict(base, iceberg_budget=1), horizon=3000, seed=11, collect_counts=True
    )
    assert capped["n_hidden_fills"] < unlim["n_hidden_fills"]


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = ice_budget_bench(horizon=3000, seed=5)
    assert out["schema"] == ICE_BUDGET_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "budget_reopens_empty",
        "hidden_survives_budget",
        "budget_decouples_hidden_from_empty",
        "joint_budget_cell",
    }
    assert out["receipt_sha256"]
