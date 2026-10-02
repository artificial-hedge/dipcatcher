"""Tests for crown_size_bench + the crown_cap / crown_size_pmf knobs."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_size_bench import (
    _CELLS,
    CROWN_SIZE_SCHEMA,
    crown_size_bench,
)
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator


def test_crown_cap_and_size_validate() -> None:
    with pytest.raises(ValueError, match="crown_cap"):
        _calibrated(3, {"crown_stack_frac": 0.5, "crown_cap": -1})
    with pytest.raises(ValueError, match="crown_cap"):
        _calibrated(3, {"crown_stack_frac": 0.5, "crown_cap": True})
    with pytest.raises(ValueError, match="crown_size_pmf"):
        _calibrated(3, {"crown_stack_frac": 0.5, "crown_size_pmf": ((0, 1.0),)})


def test_cap_and_size_dead_at_zero_frac() -> None:
    def run(extra: dict[str, object] | None) -> list[float | None]:
        sim = ZILobSimulator(_calibrated(3, extra), _split(3.0, 4))
        mids: list[float | None] = []
        for _ in range(3000):
            sim.step()
            bb, ba = sim.best_bid_level, sim.best_ask_level
            mids.append(None if bb is None or ba is None else 0.5 * (bb + ba))
        return mids

    dead = {
        "crown_stack_frac": 0.0,
        "crown_offset": 4,
        "crown_cap": 2,
        "crown_size_pmf": ((9, 1.0),),
    }
    assert run(dead) == run(None)


def test_crown_fires_under_cap_and_size() -> None:
    sim = ZILobSimulator(
        _calibrated(
            3,
            {
                "crown_stack_frac": 0.6,
                "crown_stack_span": 2,
                "crown_offset": 1,
                "crown_cap": 12,
                "crown_size_pmf": ((3, 0.5), (9, 0.5)),
            },
        ),
        _split(3.0, 4),
    )
    for _ in range(3000):
        sim.step()
    assert sim.n_lo_crown > 0


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = crown_size_bench(horizon=3000, seed=5)
    assert out["schema"] == CROWN_SIZE_SCHEMA
    assert out["kind"] == "sim_bench"
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "cap_binds_share",
        "sized_floods_share",
        "sized_empty_suppressed",
        "joint_unreachable",
    }
    assert out["receipt_sha256"]
