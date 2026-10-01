"""Tests for cxl_touch_bias — touch-concentrated cancels."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)


def _run(cfg: ZILobConfig, horizon: float = 150.0) -> ZILobSimulator:
    sim = ZILobSimulator(cfg)
    while sim.t < horizon:
        sim.step()
    return sim


class TestCxlTouchBias:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="cxl_touch_bias"):
            replace(santa_fe_config(seed=0), cxl_touch_bias=1.5)
        with pytest.raises(ValueError, match="cxl_touch_bias"):
            replace(santa_fe_config(seed=0), cxl_touch_bias=-0.1)

    def test_zero_is_bit_identical(self) -> None:
        a = _run(santa_fe_config(seed=6))
        b = _run(replace(santa_fe_config(seed=6), cxl_touch_bias=0.0))
        assert [(t.aggressor, t.price, t.qty) for t in a.trades] == [
            (t.aggressor, t.price, t.qty) for t in b.trades
        ]
        assert a.n_cancellations == b.n_cancellations

    def test_bias_fires_and_counts(self) -> None:
        sim = _run(replace(santa_fe_config(seed=6), cxl_touch_bias=0.5), 300.0)
        assert sim.n_cxl_touch > 0
        assert sim.cxl_dist[0] == sim.n_cxl_touch or sim.cxl_dist[0] >= sim.n_cxl_touch
        assert sum(sim.cxl_dist) == sim.n_cancellations
        assert sim.event_counts()["n_cxl_touch"] == sim.n_cxl_touch

    def test_uniform_cancels_record_distance(self) -> None:
        sim = _run(santa_fe_config(seed=6), 300.0)
        assert sum(sim.cxl_dist) == sim.n_cancellations > 0
        # Uniform cancels land off the touch too.
        assert sum(sim.cxl_dist[1:]) > 0


class TestCxlDistDecay:
    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="cxl_dist_decay"):
            replace(santa_fe_config(seed=0), cxl_dist_decay=-1.0)

    def test_zero_bit_identical(self) -> None:
        a = _run(santa_fe_config(seed=6))
        b = _run(replace(santa_fe_config(seed=6), cxl_touch_bias=0.0, cxl_dist_decay=0.0))
        assert a.n_cancellations == b.n_cancellations
        assert a.cxl_dist == b.cxl_dist

    def test_decay_concentrates_near_touch(self) -> None:
        sim = _run(
            replace(santa_fe_config(seed=6), cxl_touch_bias=0.9, cxl_dist_decay=2.0),
            300.0,
        )
        near = sum(sim.cxl_dist[:4])
        assert near / sim.n_cancellations > 0.5
        assert sum(sim.cxl_dist) == sim.n_cancellations

    def test_decay_tracks_age(self) -> None:
        sim = _run(
            replace(santa_fe_config(seed=6), cxl_touch_bias=0.9, cxl_dist_decay=2.0),
            200.0,
        )
        assert len(sim.cxl_ages) == sim.n_cancellations


class TestCxlRequote:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="cxl_requote"):
            replace(santa_fe_config(seed=0), cxl_requote=1.5)
        with pytest.raises(ValueError, match="cxl_requote"):
            replace(santa_fe_config(seed=0), cxl_requote=-0.1)

    def test_zero_bit_identical(self) -> None:
        a = _run(replace(santa_fe_config(seed=6), cxl_touch_bias=0.5))
        b = _run(replace(santa_fe_config(seed=6), cxl_touch_bias=0.5, cxl_requote=0.0))
        assert [(t.aggressor, t.price, t.qty) for t in a.trades] == [
            (t.aggressor, t.price, t.qty) for t in b.trades
        ]
        assert a.n_cancellations == b.n_cancellations
        assert b.n_requotes == 0

    def test_requote_preserves_depth(self) -> None:
        base = _run(replace(santa_fe_config(seed=6), cxl_touch_bias=0.9), 300.0)
        churn = _run(
            replace(santa_fe_config(seed=6), cxl_touch_bias=0.9, cxl_requote=0.9),
            300.0,
        )
        assert churn.n_requotes > 0
        assert churn.event_counts()["n_requotes"] == churn.n_requotes
        base_depth = sum(len(d) for d in base._bids.values()) + sum(
            len(d) for d in base._asks.values()
        )
        churn_depth = sum(len(d) for d in churn._bids.values()) + sum(
            len(d) for d in churn._asks.values()
        )
        assert churn_depth >= base_depth


def test_bench_smoke() -> None:
    from quant_fund.microstructure.cancel_gradient_bench import (
        cancel_gradient_bench,
    )

    p = cancel_gradient_bench(400.0, seed=3)
    assert p["schema"] == "cancel_gradient_bench.v1"
    assert p["data_label"] == "MIXED"
    assert p["research_only"] is True
    assert len(p["arms"]) == 3
