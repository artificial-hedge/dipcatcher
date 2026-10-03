"""Tests for the Markov-modulated baseline (RateRegimeSpec + rate_regimes)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    HawkesClockSpec,
    RateRegimeSpec,
    ZILobSimulator,
    santa_fe_config,
)

_REGIME = RateRegimeSpec(
    scales=((0.55, 0.45, 0.55), (2.5, 6.0, 4.0)),
    stay_probs=(0.99, 0.99),
    start=0,
)
_ZERO_K = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _events(sim: ZILobSimulator, horizon: float) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    while sim.t < horizon:
        out.append((sim.t, sim.step()))
    return out


class TestRateRegimeSpec:
    def test_rejects_empty_scales(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RateRegimeSpec((), ())  # type: ignore[arg-type]

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="share a length"):
            RateRegimeSpec(((1.0, 1.0, 1.0),), (0.9, 0.9))

    def test_rejects_bad_scale_shape(self) -> None:
        with pytest.raises(ValueError, match="3-vector"):
            RateRegimeSpec(((1.0, 1.0),), (0.9,))  # type: ignore[arg-type]

    def test_rejects_negative_scale(self) -> None:
        with pytest.raises(ValueError, match="scales"):
            RateRegimeSpec(((-1.0, 1.0, 1.0),), (0.9,))

    def test_rejects_stay_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="stay_probs"):
            RateRegimeSpec(((1.0, 1.0, 1.0),), (1.5,))

    def test_rejects_bad_start(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            RateRegimeSpec(((1.0, 1.0, 1.0),), (0.9,), start=1)

    def test_config_type_check(self) -> None:
        with pytest.raises(TypeError, match="RateRegimeSpec"):
            replace(santa_fe_config(seed=0), rate_regimes="x")  # type: ignore[arg-type]

    def test_normalizes_scales(self) -> None:
        s = RateRegimeSpec(((1, 2, 3),), (0.5,))  # type: ignore[arg-type]
        assert s.scales == ((1.0, 2.0, 3.0),)


class TestRegimeClock:
    def test_single_state_unit_scale_bit_identical(self) -> None:
        """One-state unit-scale spec consumes zero draws -> identical."""
        spec = RateRegimeSpec(((1.0, 1.0, 1.0),), (0.9,))
        ev_r = _events(
            ZILobSimulator(replace(santa_fe_config(seed=3), rate_regimes=spec)),
            200.0,
        )
        ev_p = _events(ZILobSimulator(santa_fe_config(seed=3)), 200.0)
        assert ev_r == ev_p

    def test_single_state_scaled_shares_rng_stream(self) -> None:
        """Scaled single state keeps the same draw stream: kind picks are
        identical (uniform scaling preserves proportions), only dt's
        rescale — so the kind sequence is a prefix-match."""
        spec = RateRegimeSpec(((2.0, 2.0, 2.0),), (1.0,))
        ev_r = _events(
            ZILobSimulator(replace(santa_fe_config(seed=3), rate_regimes=spec)),
            200.0,
        )
        ev_p = _events(ZILobSimulator(santa_fe_config(seed=3)), 100.0)
        kinds_r = [k for _, k in ev_r]
        kinds_p = [k for _, k in ev_p]
        assert kinds_r[: len(kinds_p)] == kinds_p
        # doubled rates -> twice the event count over the same horizon
        assert len(ev_r) > len(ev_p) * 1.5

    def test_absorbing_state_is_frozen(self) -> None:
        from quant_fund.microstructure.zi_lob_simulator import RateRegimeFlow

        spec = RateRegimeSpec(((2.0, 2.0, 2.0),), (1.0,))
        flow = RateRegimeFlow(spec, np.random.default_rng(0))
        for _ in range(50):
            flow.advance()
        assert flow.state == 0
        assert flow.n_transitions == 0

    def test_transition_count_and_state_visibility(self) -> None:
        sim = ZILobSimulator(replace(santa_fe_config(seed=5), rate_regimes=_REGIME))
        _events(sim, 2000.0)
        c = sim.event_counts()
        assert c["n_regime_transitions"] > 0
        assert c["n_regime_transitions"] < sim.n_events

    def test_determinism(self) -> None:
        cfg = replace(santa_fe_config(seed=9), rate_regimes=_REGIME)
        assert _events(ZILobSimulator(cfg), 300.0) == _events(ZILobSimulator(cfg), 300.0)

    def test_hot_state_lifts_event_rate(self) -> None:
        hot = RateRegimeSpec(
            scales=((1.0, 1.0, 1.0), (5.0, 5.0, 5.0)),
            stay_probs=(0.5, 0.98),
            start=1,
        )
        sim = ZILobSimulator(replace(santa_fe_config(seed=1), rate_regimes=hot))
        _events(sim, 500.0)
        sim_p = ZILobSimulator(santa_fe_config(seed=1))
        _events(sim_p, 500.0)
        assert sim.n_events > sim_p.n_events * 1.5

    def test_compose_with_hawkes(self) -> None:
        spec = HawkesClockSpec(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), 4.0)
        cfg = replace(
            santa_fe_config(seed=4),
            rate_regimes=_REGIME,
            hawkes=spec,
        )
        sim = ZILobSimulator(cfg)
        ev = _events(sim, 500.0)
        c = sim.event_counts()
        assert c["n_hawkes_proposals"] > 0
        assert c["n_regime_transitions"] >= 0
        assert len(ev) == sim.n_events

    def test_uniform_transition_kernel_reaches_all_states(self) -> None:
        spec = RateRegimeSpec(
            scales=((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)),
            stay_probs=(0.9, 0.9, 0.9),
        )
        from quant_fund.microstructure.zi_lob_simulator import RateRegimeFlow

        rng = np.random.default_rng(0)
        flow = RateRegimeFlow(spec, rng)
        seen = {flow.state}
        for _ in range(200):
            flow.advance()
            seen.add(flow.state)
        assert seen == {0, 1, 2}


def test_bench_smoke() -> None:
    from quant_fund.microstructure.regime_clock_bench import regime_clock_bench

    r = regime_clock_bench(horizon=300.0, seed=7)
    assert r["schema"] == "regime_clock.v1"
    assert r["claims"]["poisson_stream_is_memoryless"]
    assert len(r["receipt_sha256"]) == 64
    names = {a["name"] for a in r["arms"]}
    assert names == {"poisson", "regime", "regime_hawkes"}
