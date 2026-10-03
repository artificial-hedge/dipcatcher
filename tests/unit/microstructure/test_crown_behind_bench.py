"""Tests for crown_behind_bench + the crown_offset sim knob."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_behind_bench import (
    _CELLS,
    CROWN_BEHIND_SCHEMA,
    crown_behind_bench,
)
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator


def test_crown_offset_validates() -> None:
    with pytest.raises(ValueError, match="crown_offset"):
        _calibrated(3, {"crown_stack_frac": 0.5, "crown_offset": -1})
    with pytest.raises(ValueError, match="crown_offset"):
        _calibrated(3, {"crown_stack_frac": 0.5, "crown_offset": True})


def test_offset_shifts_band_behind_touch() -> None:
    """Offset-1 crown orders must never land on the touch itself."""
    sim = ZILobSimulator(
        _calibrated(3, {"crown_stack_frac": 0.9, "crown_stack_span": 2, "crown_offset": 1}),
        _split(3.0, 4),
    )
    for _ in range(2000):
        sim.step()
    assert sim.n_lo_crown > 0


def test_offset_dead_when_frac_zero() -> None:
    def run(extra: dict[str, float] | None) -> list[float | None]:
        sim = ZILobSimulator(_calibrated(3, extra), _split(3.0, 4))
        mids: list[float | None] = []
        for _ in range(3000):
            sim.step()
            bb, ba = sim.best_bid_level, sim.best_ask_level
            mids.append(None if bb is None or ba is None else 0.5 * (bb + ba))
        return mids

    assert run({"crown_stack_frac": 0.0, "crown_offset": 5}) == run(None)


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = crown_behind_bench(horizon=3000, seed=5)
    assert out["schema"] == CROWN_BEHIND_SCHEMA
    assert out["kind"] == "sim_bench"
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "behind_restores_empty",
        "behind_crown_seeds",
        "reveal_approaches_tape",
        "joint_crown_cell",
    }
    assert out["receipt_sha256"]
