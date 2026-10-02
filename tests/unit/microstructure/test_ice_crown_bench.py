"""Tests for ice_crown_bench (crown × iceberg_reload composition)."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.microstructure.ice_crown_bench import (
    _CELLS,
    ICE_CROWN_SCHEMA,
    ice_crown_bench,
)
from quant_fund.microstructure.place_law_bench import _calibrated


def test_iceberg_reload_validates() -> None:
    with pytest.raises(ValueError, match="iceberg_reload"):
        _calibrated(3, {"iceberg_reload": 1.5})


def test_hidden_fill_share_collected() -> None:
    c = _sim_crown(
        "ice",
        dict(_DEEP, iceberg_reload=0.6),
        horizon=800,
        seed=5,
        collect_counts=True,
    )
    assert "n_hidden_fills" in c
    assert "hidden_fill_share" in c


def test_not_collected_by_default() -> None:
    c = _sim_crown("plain", dict(_DEEP), horizon=200, seed=5)
    assert "n_hidden_fills" not in c


def test_reload_zero_no_hidden_fills() -> None:
    c = _sim_crown(
        "ice0",
        dict(_DEEP, iceberg_reload=0.0),
        horizon=800,
        seed=5,
        collect_counts=True,
    )
    assert c["n_hidden_fills"] == 0
    assert c["hidden_fill_share"] == 0.0


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = ice_crown_bench(horizon=3000, seed=5)
    assert out["schema"] == ICE_CROWN_SCHEMA
    assert out["kind"] == "sim_bench"
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "hidden_share_reaches_tape_band",
        "hidden_off_visible",
        "ice_keeps_channel_alive",
        "joint_ice_cell",
    }
    assert all("hidden_fill_share" in c for c in out["cells"])
    assert out["receipt_sha256"]
