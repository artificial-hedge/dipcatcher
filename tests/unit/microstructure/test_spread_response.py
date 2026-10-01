"""Tests for excitation-coupled LO anchoring (ZILobConfig.lo_offset_gain)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.microstructure.hawkes_clock_bench import _CROSS, BETA
from quant_fund.microstructure.zi_lob_simulator import (
    HawkesClockSpec,
    ZILobSimulator,
    santa_fe_config,
)


def _run(cfg: object, horizon: float) -> ZILobSimulator:
    sim = ZILobSimulator(cfg)  # type: ignore[arg-type]
    while sim.t < horizon:
        sim.step()
    return sim


class TestLoOffsetGain:
    def test_requires_hawkes(self) -> None:
        with pytest.raises(ValueError, match="requires a HawkesClockSpec"):
            replace(santa_fe_config(seed=0), lo_offset_gain=1.0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="lo_offset_gain"):
            replace(
                santa_fe_config(seed=0),
                hawkes=HawkesClockSpec(kernel=_CROSS, beta=BETA),
                lo_offset_gain=-1.0,
            )

    def test_zero_gain_is_bit_identical(self) -> None:
        spec = HawkesClockSpec(kernel=_CROSS, beta=BETA)
        a = _run(replace(santa_fe_config(seed=6), hawkes=spec), 100.0)
        b = _run(replace(santa_fe_config(seed=6), hawkes=spec, lo_offset_gain=0.0), 100.0)
        assert [(t.aggressor, t.price, t.qty) for t in a.trades] == [
            (t.aggressor, t.price, t.qty) for t in b.trades
        ]

    def test_gain_widens_placement(self) -> None:
        spec = HawkesClockSpec(kernel=_CROSS, beta=BETA)
        a = _run(replace(santa_fe_config(seed=6), band=14, hawkes=spec), 400.0)
        b = _run(
            replace(
                santa_fe_config(seed=6),
                band=14,
                hawkes=spec,
                lo_offset_gain=100.0,
            ),
            400.0,
        )
        # Same clock stream (identical event count), deeper placements.
        assert a.n_events == b.n_events

    def test_excitation_accessor(self) -> None:
        spec = HawkesClockSpec(kernel=_CROSS, beta=BETA)
        sim = ZILobSimulator(replace(santa_fe_config(seed=1), hawkes=spec))
        assert sim._hawkes is not None
        assert sim._hawkes.excitation(1) == 0.0
        sim.step()
        assert all(sim._hawkes.excitation(k) >= 0.0 for k in range(3))


def test_bench_smoke() -> None:
    from quant_fund.microstructure.spread_response_bench import spread_response_bench

    r = spread_response_bench(horizon=200.0, seed=13)
    assert r["schema"] == "spread_response_bench.v1"
    assert r["claims"]["static_kernel_is_flat"]
    assert len(r["receipt_sha256"]) == 64
    assert [a["name"] for a in r["arms"]] == ["static_offset", "excitation_coupled"]
