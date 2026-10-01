"""Tests for the HawkesClock event clock (HawkesClockSpec + ZILobConfig.hawkes)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    HawkesClock,
    HawkesClockSpec,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)

BETA = 4.0
_ZERO = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_KINDS = {"limit": 0, "market": 1, "cancel": 2}


def _spec(entries: dict[tuple[int, int], float], beta: float = BETA) -> HawkesClockSpec:
    k = [list(r) for r in _ZERO]
    for (i, j), eta in entries.items():
        k[i][j] = eta * beta
    return HawkesClockSpec(tuple(tuple(r) for r in k), beta)


def _with_hawkes(spec: HawkesClockSpec, seed: int) -> ZILobConfig:
    from dataclasses import replace

    return replace(santa_fe_config(seed=seed), hawkes=spec)


def _events(sim: ZILobSimulator, horizon: float) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    while sim.t < horizon:
        out.append((sim.t, _KINDS[sim.step()]))
    return out


def _b_gaps(times: list[float]) -> float:
    g = np.diff(np.asarray(times, dtype=float))
    mu, sd = float(g.mean()), float(g.std())
    return (sd - mu) / (sd + mu)


class TestHawkesClockSpec:
    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            HawkesClockSpec(((1.0, 0.0), (0.0, 1.0)), 1.0)  # type: ignore[arg-type]

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            HawkesClockSpec(((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0)

    def test_rejects_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            HawkesClockSpec(((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0)

    def test_rejects_supercritical(self) -> None:
        with pytest.raises(ValueError, match="sub-critical"):
            _spec({(1, 1): 1.5})  # branching ratio > 1 on a self-loop explodes

    def test_rejects_bad_beta(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            HawkesClockSpec(_ZERO, 0.0)

    def test_feedforward_strong_edges_stay_subcritical(self) -> None:
        spec = _spec({(1, 1): 0.5, (1, 2): 2.0, (2, 2): 0.4})
        assert spec.beta == BETA

    def test_config_rejects_foreign_hawkes(self) -> None:
        with pytest.raises(TypeError, match="HawkesClockSpec"):
            ZILobConfig(hawkes="not-a-spec")  # type: ignore[arg-type]


class TestBitIdentity:
    def test_zero_kernel_is_bit_identical_to_poisson(self) -> None:
        """A zero kernel consumes the same draws as the Poisson clock."""
        a = ZILobSimulator(santa_fe_config(seed=11))
        b = ZILobSimulator(_with_hawkes(HawkesClockSpec(_ZERO, BETA), seed=11))
        ea = _events(a, 300.0)
        eb = _events(b, 300.0)
        assert ea == eb
        strip = {k: v for k, v in a.event_counts().items() if not k.startswith("n_hawkes")}
        assert strip == {k: v for k, v in b.event_counts().items() if not k.startswith("n_hawkes")}
        assert [(t.t, t.price, t.qty, t.aggressor) for t in a.trades] == [
            (t.t, t.price, t.qty, t.aggressor) for t in b.trades
        ]

    def test_unset_hawkes_is_deterministic(self) -> None:
        a = ZILobSimulator(santa_fe_config(seed=3))
        b = ZILobSimulator(santa_fe_config(seed=3))
        assert _events(a, 100.0) == _events(b, 100.0)


class TestDynamics:
    def test_self_excitation_raises_burstiness(self) -> None:
        tp = [t for t, _ in _events(ZILobSimulator(santa_fe_config(seed=5)), 800.0)]
        sim_h = ZILobSimulator(_with_hawkes(_spec({(0, 0): 0.5, (1, 1): 0.6, (2, 2): 0.5}), seed=5))
        th = [t for t, _ in _events(sim_h, 800.0)]
        assert _b_gaps(th) > _b_gaps(tp) + 0.05

    def test_cross_excitation_produces_cancel_retreat(self) -> None:
        spec = _spec({(1, 2): 2.0, (1, 1): 0.5, (2, 2): 0.4, (1, 0): 0.3, (2, 0): 0.15})
        sim = ZILobSimulator(_with_hawkes(spec, seed=9))
        ev = _events(sim, 800.0)
        mo = [t for t, k in ev if k == 1]
        cx = np.asarray([t for t, k in ev if k == 2], dtype=float)
        in_win = sum(
            int(np.searchsorted(cx, t + 0.5) - np.searchsorted(cx, t, "right")) for t in mo
        )
        baseline = len(cx) / ev[-1][0]
        lift = in_win / (len(mo) * 0.5) / baseline
        assert lift > 1.5

    def test_determinism(self) -> None:
        spec = _spec({(1, 1): 0.5, (1, 2): 1.0})
        a = ZILobSimulator(_with_hawkes(spec, seed=17))
        b = ZILobSimulator(_with_hawkes(spec, seed=17))
        assert _events(a, 200.0) == _events(b, 200.0)

    def test_counters(self) -> None:
        sim = ZILobSimulator(_with_hawkes(_spec({(1, 1): 0.5}), seed=2))
        _events(sim, 200.0)
        c = sim.event_counts()
        assert c["n_hawkes_proposals"] == sim.n_events + c["n_hawkes_rejected"]

    def test_poisson_counters_zero(self) -> None:
        sim = ZILobSimulator(santa_fe_config(seed=2))
        _events(sim, 100.0)
        assert sim.event_counts()["n_hawkes_proposals"] == 0


class TestClockDirect:
    def test_step_returns_valid_kind(self) -> None:
        rng = np.random.default_rng(0)
        clock = HawkesClock(_spec({(1, 1): 0.5}), rng)
        for _ in range(50):
            dt, kind = clock.step((1.0, 0.5, 0.5))
            assert dt > 0 and kind in (0, 1, 2)


def test_bench_smoke() -> None:
    from quant_fund.microstructure.hawkes_clock_bench import hawkes_clock_bench

    r = hawkes_clock_bench(horizon=200.0, seed=7)
    assert r["schema"] == "hawkes_clock.v1"
    assert r["claims"]["poisson_stream_is_memoryless"]
    assert len(r["receipt_sha256"]) == 64
    names = {a["name"] for a in r["arms"]}
    assert names == {"poisson", "hawkes_self", "hawkes_cross", "hawkes_powerlaw"}


class TestMultiTimescale:
    def test_rejects_mismatched_banks(self) -> None:
        with pytest.raises(ValueError, match="non-empty length"):
            HawkesClockSpec(_ZERO, 1.0, rates=(4.0, 1.0), bank_weights=(1.0,))

    def test_rejects_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            HawkesClockSpec(_ZERO, 1.0, rates=(4.0, 1.0), bank_weights=(0.9, 0.9))

    def test_rejects_nonpositive_rate(self) -> None:
        with pytest.raises(ValueError, match="rates"):
            HawkesClockSpec(_ZERO, 1.0, rates=(4.0, 0.0), bank_weights=(0.5, 0.5))

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="bank_weights"):
            HawkesClockSpec(_ZERO, 1.0, rates=(4.0, 1.0), bank_weights=(1.5, -0.5))

    def test_subcritical_uses_effective_h(self) -> None:
        # kernel/beta alone is sub-critical (rho = 0.5), but with most
        # weight on the slow bank H blows past it -> must still raise.
        k = ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="sub-critical"):
            HawkesClockSpec(k, 4.0, rates=(100.0, 0.05), bank_weights=(0.05, 0.95))

    def test_branching_matrix_scales_by_h(self) -> None:
        rates = (4.0, 1.0)
        weights = (0.5, 0.5)
        h = sum(w / r for w, r in zip(weights, rates, strict=True))
        spec = HawkesClockSpec(
            ((h, 0, 0), (0, 0, 0), (0, 0, 0)),  # jump alpha = h
            4.0,
            rates=rates,
            bank_weights=weights,
        )
        b = np.asarray(spec.branching_matrix())
        assert b.shape == (3, 3)
        assert b[0, 0] == pytest.approx(h * h)  # alpha * H
        # Single-rate: branching ratio reduces to alpha / beta.
        spec2 = HawkesClockSpec(((2.0, 0, 0), (0, 0, 0), (0, 0, 0)), 4.0)
        assert spec2.branching_matrix()[0][0] == pytest.approx(0.5)

    def test_zero_kernel_multi_bank_is_poisson_identical(self) -> None:
        spec = HawkesClockSpec(_ZERO, BETA, rates=(16.0, 4.0, 1.0), bank_weights=(0.5, 0.25, 0.25))
        ev_pl = _events(ZILobSimulator(_with_hawkes(spec, seed=11)), 200.0)
        ev_p = _events(ZILobSimulator(santa_fe_config(seed=11)), 200.0)
        assert ev_pl == ev_p

    def test_multi_bank_determinism(self) -> None:
        spec = HawkesClockSpec(
            ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0)),
            BETA,
            rates=(16.0, 4.0, 1.0),
            bank_weights=(0.5, 0.3, 0.2),
        )
        a = _events(ZILobSimulator(_with_hawkes(spec, seed=3)), 300.0)
        b = _events(ZILobSimulator(_with_hawkes(spec, seed=3)), 300.0)
        assert a == b

    def test_multi_bank_excites(self) -> None:
        spec = HawkesClockSpec(
            # eta = alpha * H = 8 * (0.7/16 + 0.3/4) = 0.95 (sub-critical).
            ((0.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 0.0)),
            BETA,
            rates=(16.0, 4.0),
            bank_weights=(0.7, 0.3),
        )
        n_pl = len(_events(ZILobSimulator(_with_hawkes(spec, seed=5)), 200.0))
        n_p = len(_events(ZILobSimulator(santa_fe_config(seed=5)), 200.0))
        assert n_pl > n_p * 1.2
