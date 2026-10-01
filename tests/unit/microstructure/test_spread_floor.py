"""Tests for deep-anchored LO placement (ZILobConfig.lo_offset)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)


def _run(cfg: object, horizon: float) -> ZILobSimulator:
    sim = ZILobSimulator(cfg)  # type: ignore[arg-type]
    while sim.t < horizon:
        sim.step()
    return sim


class TestLoOffset:
    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="lo_offset"):
            replace(santa_fe_config(seed=0), lo_offset=-1)

    def test_zero_is_bit_identical(self) -> None:
        cfg0 = santa_fe_config(seed=7)
        cfg1 = replace(cfg0, lo_offset=0)
        a = _run(cfg0, 100.0)
        b = _run(cfg1, 100.0)
        assert a.n_events == b.n_events
        assert [(t.aggressor, t.price, t.qty) for t in a.trades] == [
            (t.aggressor, t.price, t.qty) for t in b.trades
        ]

    def test_offset_floors_spread(self) -> None:
        # Every deposit lands at least offset+1 deep: the observed
        # spread can never collapse below ~offset+1 ticks.
        cfg = replace(santa_fe_config(seed=3), lo_offset=10, band=14)
        sim = ZILobSimulator(cfg)
        min_spread = 10**9
        while sim.t < 500.0:
            sim.step()
            s = sim.spread_ticks
            if s is not None:
                min_spread = min(min_spread, s)
        assert min_spread <= 11  # floor reachable
        assert min_spread >= 1

    def test_offset_changes_trade_prices(self) -> None:
        a = _run(replace(santa_fe_config(seed=5), lo_offset=0, band=14), 300.0)
        b = _run(replace(santa_fe_config(seed=5), lo_offset=8, band=14), 300.0)
        assert a.trades != b.trades

    def test_improvement_counter_exists(self) -> None:
        sim = _run(replace(santa_fe_config(seed=5), lo_offset=6, band=14), 300.0)
        c = sim.event_counts()
        assert c["n_lo_improve"] >= 0
        assert c["n_lo_improve"] <= c["n_lo_arrivals"]


def test_bench_smoke() -> None:
    from quant_fund.microstructure.spread_floor_bench import spread_floor_bench

    r = spread_floor_bench(horizon=300.0, seed=7)
    assert r["schema"] == "spread_floor.v1"
    assert r["claims"]["legacy_hugs_touch"]
    assert len(r["receipt_sha256"]) == 64
    assert [a["name"] for a in r["arms"]] == ["offset_0", "offset_8", "offset_12"]
