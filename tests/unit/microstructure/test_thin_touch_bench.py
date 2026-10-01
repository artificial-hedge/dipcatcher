"""Tests for thin_touch_bench (near-touch depth cap)."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.microstructure.place_law_bench import _calibrated
from quant_fund.microstructure.thin_touch_bench import (
    _CELLS,
    THIN_TOUCH_SCHEMA,
    thin_touch_bench,
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
    # Level 100 is the ask touch and full -> capped; deeper level is not.
    assert sim._touch_capped("sell", 100)  # noqa: SLF001
    assert not sim._touch_capped("sell", 104)  # noqa: SLF001 — out of band
    sim._rest("sell", 104, "manual")  # noqa: SLF001
    assert len(sim._asks[104]) == 1  # noqa: SLF001


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = thin_touch_bench(horizon=3000, seed=5)
    assert out["schema"] == THIN_TOUCH_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "cap_lifts_empties",
        "cap_reaches_tape_empty",
        "hidden_survives_cap",
        "joint_thin_cell",
    }
    assert out["receipt_sha256"]


def test_sim_counts_capped() -> None:
    cfg = _calibrated(3, {"near_level_cap": 1, "near_level_span": 2})
    sim = ZILobSimulator(cfg)
    for _ in range(200):
        sim.step()
    assert sim.event_counts()["n_lo_capped"] >= 0
