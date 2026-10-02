"""Tests for crown_join_bench + the crown_stack_* sim knobs."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import _DEEP
from quant_fund.microstructure.crown_join_bench import (
    _CELLS,
    CROWN_JOIN_SCHEMA,
    crown_join_bench,
)
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator


def test_zero_default_bit_identical() -> None:
    def run(extra: dict[str, float] | None) -> list[float | None]:
        sim = ZILobSimulator(_calibrated(3, extra), _split(3.0, 4))
        mids: list[float | None] = []
        for _ in range(3000):
            sim.step()
            bb, ba = sim.best_bid_level, sim.best_ask_level
            mids.append(None if bb is None or ba is None else 0.5 * (bb + ba))
        return mids

    assert run(None) == run({"crown_stack_frac": 0.0, "crown_stack_span": 0})


def test_crown_fires_and_validates() -> None:
    with pytest.raises(ValueError, match="crown_stack_frac"):
        _calibrated(3, {"crown_stack_frac": 1.5})
    with pytest.raises(ValueError, match="crown_stack_span"):
        _calibrated(3, {"crown_stack_frac": 0.5, "crown_stack_span": -1})
    sim = ZILobSimulator(
        _calibrated(3, {"crown_stack_frac": 0.4, "crown_stack_span": 3}),
        _split(3.0, 4),
    )
    for _ in range(3000):
        sim.step()
    assert sim.n_lo_crown > 0
    assert sim.event_counts()["n_lo_crown"] == sim.n_lo_crown


def test_wide_crown_narrows_spread() -> None:
    from quant_fund.microstructure.crown_density_bench import _sim_crown

    ref = _sim_crown("wide", dict(_DEEP), horizon=3000, seed=5)
    cr = _sim_crown(
        "wide_crown",
        dict(_DEEP, crown_stack_frac=0.3, crown_stack_span=3),
        horizon=3000,
        seed=5,
    )
    assert cr["crown_share_of_visible"] is not None
    assert ref["crown_share_of_visible"] is not None
    assert cr["crown_share_of_visible"] > ref["crown_share_of_visible"]
    assert (cr["spread_mean"] or 1e9) < (ref["spread_mean"] or 0.0)


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = crown_join_bench(horizon=3000, seed=5)
    assert out["schema"] == CROWN_JOIN_SCHEMA
    assert out["kind"] == "sim_bench"
    assert out["data_label"] == "SYNTHETIC"
    assert out["research_only"] is True
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "crown_seeded",
        "spread_compresses",
        "crown_kills_reveal",
        "narrow_crown_survives",
    }
    assert out["receipt_sha256"]
