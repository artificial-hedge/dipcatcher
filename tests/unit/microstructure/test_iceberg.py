"""Tests for iceberg reload liquidity (ZILobConfig.iceberg_reload)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)


def _run(cfg: ZILobConfig, horizon: float) -> ZILobSimulator:
    sim = ZILobSimulator(cfg)
    while sim.t < horizon:
        sim.step()
    return sim


class TestIcebergReload:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="iceberg_reload"):
            replace(santa_fe_config(seed=0), iceberg_reload=1.5)

    def test_zero_is_bit_identical(self) -> None:
        a = _run(santa_fe_config(seed=9), 100.0)
        b = _run(replace(santa_fe_config(seed=9), iceberg_reload=0.0), 100.0)
        assert [(t.aggressor, t.price, t.qty) for t in a.trades] == [
            (t.aggressor, t.price, t.qty) for t in b.trades
        ]
        assert a.n_hidden_fills == 0

    def test_reload_produces_iceberg_fills(self) -> None:
        sim = _run(replace(santa_fe_config(seed=9), iceberg_reload=0.5), 500.0)
        assert sim.n_hidden_fills > 0
        assert any(t.maker_tag == "iceberg" for t in sim.trades)

    def test_hidden_fill_share_bounded_by_fills(self) -> None:
        sim = _run(replace(santa_fe_config(seed=9), iceberg_reload=0.3), 500.0)
        assert 0 < sim.n_hidden_fills <= sim.n_fills

    def test_reloaded_orders_rest_at_same_level(self) -> None:
        # Direct probe: consume a seeded level twice; the reload keeps
        # the same level alive.
        sim = ZILobSimulator(
            replace(
                santa_fe_config(seed=4),
                init_levels=1,
                init_depth=1,
                iceberg_reload=1.0,
            )
        )
        bb0 = sim.best_bid_level
        assert bb0 is not None
        fills = sim.inject_market_order("sell", qty=1)
        assert fills and fills[0].level == bb0
        # With reload=1.0 the level must still exist.
        assert sim.best_bid_level == bb0
        # And the surviving order is tagged iceberg.
        bid_oids = sim._bids[bb0]
        assert sim._orders[bid_oids[0]].tag == "iceberg"


def test_bench_smoke() -> None:
    from quant_fund.microstructure.iceberg_bench import iceberg_bench

    r = iceberg_bench(horizon=300.0, seed=11)
    assert r["schema"] == "iceberg.v1"
    assert r["claims"]["legacy_has_no_hidden_fills"]
    assert len(r["receipt_sha256"]) == 64
    assert [a["name"] for a in r["arms"]] == ["reload_0.0", "reload_0.25"]
