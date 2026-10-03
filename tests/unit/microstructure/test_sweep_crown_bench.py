"""Tests for sweep_crown_bench (crown × iceberg × MO-tail grid)."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.microstructure.sweep_crown_bench import (
    _CELLS,
    SWEEP_CROWN_SCHEMA,
    sweep_crown_bench,
)


def test_collect_counts_emits_sweep_stats() -> None:
    c = _sim_crown("x", dict(_DEEP), horizon=800, seed=5, collect_counts=True)
    for k in (
        "n_hidden_fills",
        "hidden_fill_share",
        "n_mo_units",
        "n_mo_arrivals",
        "sweep_p_ge2",
        "sweep_max_levels",
    ):
        assert k in c


def test_unit_mo_never_sweeps() -> None:
    c = _sim_crown(
        "u",
        dict(_DEEP, mo_size_pmf=((1, 1.0),)),
        horizon=800,
        seed=5,
        collect_counts=True,
    )
    assert c["sweep_p_ge2"] == 0.0
    assert c["sweep_max_levels"] == 1


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = sweep_crown_bench(horizon=3000, seed=5)
    assert out["schema"] == SWEEP_CROWN_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "mopmf_overshoots_tape",
        "sweep_footprint_in_band",
        "empty_not_sweep_limited",
        "joint_sweep_cell",
    }
    assert out["receipt_sha256"]
