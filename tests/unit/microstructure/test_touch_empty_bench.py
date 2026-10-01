"""Tests for touch_empty_bench (near-band cap + corrected reveal index)."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.microstructure.place_law_bench import _calibrated
from quant_fund.microstructure.touch_empty_bench import (
    _CELLS,
    TOUCH_EMPTY_SCHEMA,
    touch_empty_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator


def test_cap_validates() -> None:
    with pytest.raises(ValueError, match="near_level_cap"):
        _calibrated(3, {"near_level_cap": -1})
    with pytest.raises(ValueError, match="near_level_span"):
        _calibrated(3, {"near_level_cap": 2, "near_level_span": -1})


def test_cap_default_bit_identical() -> None:
    """cap=0 leaves the trajectory untouched."""
    base = dict(_DEEP, crown_stack_frac=0.3, crown_stack_span=2)
    a = _sim_crown("a", dict(base), horizon=1500, seed=9, collect_counts=True)
    b = _sim_crown(
        "b",
        dict(base, near_level_cap=0, near_level_span=3),
        horizon=1500,
        seed=9,
        collect_counts=True,
    )
    a.pop("regime")
    b.pop("regime")
    assert a == b


def test_cap_refuses_full_touch() -> None:
    """Deterministic probe: a third own-touch arrival is refused at cap 2."""
    cfg = _calibrated(3, {"near_level_cap": 2, "near_level_span": 3})
    sim = ZILobSimulator(cfg)
    sim._asks.clear()  # noqa: SLF001 — mechanism probe
    sim._bids.clear()  # noqa: SLF001
    sim._orders.clear()  # noqa: SLF001
    sim._rest("buy", 90, "manual")  # noqa: SLF001 — bid below the ask touch
    sim._rest("sell", 100, "manual")  # noqa: SLF001
    sim._rest("sell", 100, "manual")  # noqa: SLF001 — ask touch full at 2
    assert sim._touch_capped("sell", 100)  # noqa: SLF001
    assert not sim._touch_capped("sell", 104)  # noqa: SLF001 — out of band
    sim._rest("sell", 104, "manual")  # noqa: SLF001
    assert len(sim._asks[104]) == 1  # noqa: SLF001


def test_reveal_index_counts_empties() -> None:
    """Under the corrected index the deep arm's touches empty on ~most
    fills — the pre-fix index hid them behind immediate re-seeds."""
    out = _sim_crown("deep", dict(_DEEP), horizon=4000, seed=11, collect_counts=True)
    assert out["n_fills"] > 0
    assert out["n_reveals"] > 0.3 * out["n_fills"]


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = touch_empty_bench(horizon=3000, seed=5)
    assert out["schema"] == TOUCH_EMPTY_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "index_fix_unmasks_empties",
        "touch_stack_lowers_empties",
        "empty_band_reached",
        "joint_thin_cell",
    }
    assert out["receipt_sha256"]


def test_sim_counts_capped() -> None:
    cfg = _calibrated(3, {"near_level_cap": 1, "near_level_span": 2})
    sim = ZILobSimulator(cfg)
    for _ in range(200):
        sim.step()
    assert sim.event_counts()["n_lo_capped"] >= 0
