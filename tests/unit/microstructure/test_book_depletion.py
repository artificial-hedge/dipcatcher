"""Tests for microstructure/book_depletion.py — empirical level-depth dynamics.

Birth-death level dynamics on the ZI (Santa Fe) LOB: event-clock depth panel,
per-level arrival/cancel/consumption rates, the constant-rate CTMC mean
depletion time (gambler's-ruin absorption), and the sealed
``book_depletion.v1`` SYNTHETIC bench receipt. Everything here is a labeled
SYNTHETIC correctness diagnostic — never market evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.book_depletion import (
    BOOK_DEPLETION_KIND,
    arrival_cancel_rates,
    book_depletion_bench,
    collect_depth_panel,
    depletion_time_ctmc,
    level_depletion_spells,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

# ---------------------------------------------------------------------------
# CTMC hitting-time KATs (gambler's-ruin closed form E[T_q] = q/(mu - lam))
# ---------------------------------------------------------------------------


def test_depletion_time_ctmc_pure_death_lam0() -> None:
    """lam=0 is the pure-death chain: q0 Exp(mu) holding times -> q0/mu."""
    assert depletion_time_ctmc(5, 0.0, 0.1) == pytest.approx(50.0)
    assert depletion_time_ctmc(1, 0.0, 2.0) == pytest.approx(0.5)
    assert depletion_time_ctmc(3, 0.0, 0.06) == pytest.approx(50.0)


def test_depletion_time_ctmc_subcritical_drift_form() -> None:
    """lam < mu: the geometric-series closed form telescopes to q0/(mu - lam)."""
    assert depletion_time_ctmc(6, 0.04, 0.12) == pytest.approx(75.0)
    assert depletion_time_ctmc(1, 0.06, 0.10) == pytest.approx(25.0)
    # Hitting time is linear in the start depth (skip-free chain to the left).
    e1 = depletion_time_ctmc(1, 0.03, 0.09)
    e4 = depletion_time_ctmc(4, 0.03, 0.09)
    assert e4 == pytest.approx(4.0 * e1)


def test_depletion_time_ctmc_critical_and_supercritical_inf() -> None:
    """lam >= mu: absorption is not a.s. finite — the honest answer is +inf."""
    assert math.isinf(depletion_time_ctmc(4, 0.10, 0.10))  # null recurrent
    assert math.isinf(depletion_time_ctmc(2, 0.20, 0.10))


def test_depletion_time_ctmc_monte_carlo_agreement() -> None:
    """The closed form must match a Monte Carlo of the embedded chain.

    An independent single-path Gillespie sim (not the module's vectorized
    helper) on a strongly subcritical chain; ~120k paths pin the mean inside
    a 5% relative band.
    """
    q0, lam, mu = 4, 0.05, 0.15
    rng = np.random.default_rng(20260)
    n_paths = 120_000
    times = np.empty(n_paths, dtype=np.float64)
    rate = lam + mu
    p_up = lam / rate
    for i in range(n_paths):
        q = q0
        t = 0.0
        while q > 0:
            t += rng.exponential(1.0 / rate)
            q += 1 if rng.random() < p_up else -1
        times[i] = t
    assert float(times.mean()) == pytest.approx(depletion_time_ctmc(q0, lam, mu), rel=0.05)


def test_depletion_time_ctmc_fail_closed() -> None:
    with pytest.raises(ValueError):
        depletion_time_ctmc(0, 0.0, 0.1)  # empty start depth
    with pytest.raises(ValueError):
        depletion_time_ctmc(-2, 0.0, 0.1)
    with pytest.raises(ValueError):
        depletion_time_ctmc(1, -0.01, 0.1)  # negative birth rate
    with pytest.raises(ValueError):
        depletion_time_ctmc(1, 0.0, 0.0)  # non-positive death rate
    with pytest.raises(ValueError):
        depletion_time_ctmc(1, 0.0, -0.1)
    with pytest.raises(ValueError):
        depletion_time_ctmc(1, float("nan"), 0.1)


# ---------------------------------------------------------------------------
# Panel collector
# ---------------------------------------------------------------------------


def test_collect_depth_panel_smoke() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=11))
    n_events, depth = 300, 3
    panel = collect_depth_panel(sim, sample_interval=n_events, depth=depth)
    assert panel["t"].shape == (n_events + 1,)
    assert panel["bid_depth"].shape == (n_events + 1, depth)
    assert panel["ask_depth"].shape == (n_events + 1, depth)
    # Event-clock rows: non-decreasing sim time, kinds limited to the sim's.
    assert np.all(np.diff(panel["t"]) >= 0.0)
    assert set(np.unique(panel["kind"])) <= {"init", "limit", "market", "cancel"}
    assert panel["n_events"][-1] == sim.n_events
    # Level anchors froze at the initial touch; depths are unit counts >= 0.
    assert (panel["bid_levels"] == panel["best_bid_level"][0] - np.arange(depth)).all()
    assert (panel["ask_levels"] == panel["best_ask_level"][0] + np.arange(depth)).all()
    assert (panel["bid_depth"] >= 0).all() and (panel["ask_depth"] >= 0).all()
    # Final row is the live sim state at each tracked absolute level.
    for i, lvl in enumerate(panel["bid_levels"]):
        assert panel["bid_depth"][-1, i] == sim.depth_at("buy", int(lvl))
    for i, lvl in enumerate(panel["ask_levels"]):
        assert panel["ask_depth"][-1, i] == sim.depth_at("sell", int(lvl))


def test_collect_depth_panel_fail_closed() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=3))
    with pytest.raises(ValueError):
        collect_depth_panel(sim, sample_interval=0, depth=2)
    with pytest.raises(ValueError):
        collect_depth_panel(sim, sample_interval=10, depth=0)
    with pytest.raises(ValueError):
        collect_depth_panel(sim, sample_interval=10, depth=-1)
    with pytest.raises(TypeError):
        collect_depth_panel("not a sim", sample_interval=10, depth=2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rate estimation
# ---------------------------------------------------------------------------


def test_arrival_cancel_rates_sanity() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=23))
    panel = collect_depth_panel(sim, sample_interval=2_000, depth=4)
    rates = arrival_cancel_rates(panel)
    assert rates["n_events"] == 2_000
    assert rates["window_s"] > 0.0
    for side in ("bid", "ask"):
        s = rates[side]
        path = panel[f"{side}_depth"]
        # Exact per-level conservation identity: tracked deltas decompose
        # completely into births (+1) and deaths (-1).
        net = path[-1] - path[0]
        assert (s["n_birth"] - s["n_death"] == net).all()
        assert (s["n_death_cancel"] + s["n_death_market"] == s["n_death"]).all()
        assert (s["alive_s"] >= 0.0).all() and (s["in_band_s"] >= 0.0).all()
        assert (s["birth_rate"] >= 0.0).all()
        # The level-0 touch is hit by both MOs and cancels; positive exposure.
        assert s["alive_s"][0] > 0.0
        assert s["death_rate"][0] > 0.0
        # Santa Fe calibration: lam=0.06/s per level per side while in band —
        # the in-band estimate must land within a generous calibration band.
        in_band = s["birth_rate_in_band"]
        if np.isfinite(in_band[0]):
            assert 0.005 < in_band[0] < 0.5


def test_arrival_cancel_rates_fail_closed() -> None:
    with pytest.raises(TypeError):
        arrival_cancel_rates(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        arrival_cancel_rates({})  # empty panel
    sim = ZILobSimulator(santa_fe_config(seed=5))
    panel = collect_depth_panel(sim, sample_interval=8, depth=2)
    one_row = {k: (v[:1] if isinstance(v, np.ndarray) and v.ndim else v) for k, v in panel.items()}
    with pytest.raises(ValueError):
        arrival_cancel_rates(one_row)  # single row -> zero window
    # Forged panel violating the unit-event invariant must raise.
    forged = dict(panel)
    bad_depth = panel["bid_depth"].copy()
    bad_depth[1, 0] += 3
    forged["bid_depth"] = bad_depth
    with pytest.raises(ValueError):
        arrival_cancel_rates(forged)


def test_level_depletion_spells_complete_and_censored() -> None:
    sim = ZILobSimulator(santa_fe_config(seed=17))
    panel = collect_depth_panel(sim, sample_interval=1_500, depth=2)
    spells = level_depletion_spells(panel)
    total = spells["n_complete"] + spells["n_left_censored"] + spells["n_right_censored"]
    assert spells["n_complete"] >= 0 and total >= 2  # levels open populated
    for side in ("bid", "ask"):
        for per_level in spells[side]:
            for sp in per_level:
                assert sp["q_start"] >= 1  # unit births -> starts are q=1
                assert sp["duration_s"] > 0.0


# ---------------------------------------------------------------------------
# Sealed bench receipt
# ---------------------------------------------------------------------------


def test_book_depletion_bench_schema_and_determinism() -> None:
    r1 = book_depletion_bench(n_events=1_500, depth=3, mc_paths=32, seed=41)
    r2 = book_depletion_bench(n_events=1_500, depth=3, mc_paths=32, seed=41)
    assert r1 == r2  # bit-identical sealed receipt under the same seed
    assert r1["schema"] == BOOK_DEPLETION_KIND == "book_depletion.v1"
    assert r1["kind"] == "book_depletion.v1"
    assert r1["label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    assert r1["live_pnl_claim"] is False
    assert r1["claim"] == "simulator_internal_diagnostic_only"
    # Seal integrity: receipt_sha256 binds the canonical body.
    unsigned = {k: v for k, v in r1.items() if k != "receipt_sha256"}
    assert r1["receipt_sha256"] == hash_bytes(canonical_json_bytes(unsigned))
    # Receipt structure: per-level rows on both sides, agreement + symmetry.
    assert len(r1["per_level"]["bid"]) == 3 and len(r1["per_level"]["ask"]) == 3
    row0 = r1["per_level"]["bid"][0]
    for key in (
        "birth_rate",
        "death_rate",
        "cancel_rate",
        "market_rate_at_best",
        "n_birth",
        "n_death",
        "n_spells",
        "obs_mean_depletion_s",
        "mc_mean_depletion_s",
        "analytic_depletion_s",
        "rel_err_mc",
    ):
        assert key in row0
    assert r1["agreement"]["n_levels_scored"] >= 1
    assert r1["agreement"]["n_levels"] == 6
    assert math.isfinite(r1["agreement"]["mc_rel_err_mean"])
    assert "sides_symmetric" in r1["symmetry"]
    assert r1["event_counts"]["n_events"] >= 1_500
    # A different seed yields a different sealed receipt.
    r3 = book_depletion_bench(n_events=1_500, depth=3, mc_paths=32, seed=42)
    assert r3["receipt_sha256"] != r1["receipt_sha256"]


def test_book_depletion_bench_no_forbidden_metric_keys() -> None:
    receipt = book_depletion_bench(n_events=400, depth=2, mc_paths=8, seed=7)

    def _keys(obj: object) -> list[str]:
        out: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.append(str(k))
                out.extend(_keys(v))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                out.extend(_keys(item))
        return out

    for key in _keys(receipt):
        for token in ("sharpe", "sortino", "calmar", "nav"):
            assert token not in key.lower()
        # "pnl" may appear only inside the mandatory honesty flag key.
        if "pnl" in key.lower():
            assert key == "live_pnl_claim"


def test_book_depletion_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        book_depletion_bench(n_events=0)
    with pytest.raises(ValueError):
        book_depletion_bench(depth=0)
    with pytest.raises(ValueError):
        book_depletion_bench(mc_paths=0)
    with pytest.raises(ValueError):
        book_depletion_bench(mc_step_cap=0)
    with pytest.raises(ValueError):
        book_depletion_bench(sym_tol=0.0)
    with pytest.raises(ValueError):
        book_depletion_bench(seed="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        book_depletion_bench(config="nope")  # type: ignore[arg-type]


def test_collect_depth_panel_seed_determinism() -> None:
    a = collect_depth_panel(ZILobSimulator(ZILobConfig(seed=99)), sample_interval=200, depth=2)
    b = collect_depth_panel(ZILobSimulator(ZILobConfig(seed=99)), sample_interval=200, depth=2)
    assert np.array_equal(a["bid_depth"], b["bid_depth"])
    assert np.array_equal(a["kind"], b["kind"])
    assert np.allclose(a["t"], b["t"])
