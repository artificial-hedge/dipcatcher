"""Adversarial probes for the microstructure stats/models lane.

Each test pins a defect the lane audit fixed: seal-gated claims reads,
EXECUTION_HIDDEN inclusion, LOBSTER event-0 semantics, zero-fill
market steps, vacuous-true claims, seed-book oid attribution, and the
GLFT/Hawkes likelihood constants. Probes are seeded and SYNTHETIC.
"""

from __future__ import annotations

import csv
import json
import math
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.microstructure import event_granger, hmm_learn, vol_clock
from quant_fund.microstructure.glft_closed_form import _theta_rhs, theta_closed_form
from quant_fund.microstructure.hawkes_calibrate import (
    fit_hawkes_exp,
    hawkes_exp_loglik,
    zi_mo_times,
)
from quant_fund.microstructure.hawkes_real import mo_times_lobster
from quant_fund.microstructure.intraday_shape import lobster_intraday
from quant_fund.microstructure.lobster import (
    EXECUTION,
    EXECUTION_HIDDEN,
    HALT,
    SUBMISSION,
)
from quant_fund.microstructure.price_clustering import sim_cluster_stats
from quant_fund.microstructure.round_lot import lobster_round_lot
from quant_fund.microstructure.streak_stats import lobster_streaks
from quant_fund.microstructure.sweep_width import sweep_width_bench
from quant_fund.microstructure.wave23_map import _LANES, wave23_map
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig
from quant_fund.research.receipt_v2 import seal_receipt

_TICKER = "AMZN"
_MSG = f"{_TICKER}_2012-06-21_34200000_57600000_message_10.csv"
_OB = f"{_TICKER}_2012-06-21_34200000_57600000_orderbook_10.csv"


def _write_tape(
    d: Path,
    rows: list[list[float]],
    ob_rows: list[list[int]] | None = None,
) -> tuple[Path, Path]:
    msg = d / _MSG
    with msg.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    ob = d / _OB
    with ob.open("w", newline="") as fh:
        # orderbook rows pair 1:1 with message rows; 10-level format is
        # (ask_px, ask_sz, bid_px, bid_sz) x 10
        default = [10100, 30, 9900, 30] + [0, 0, 0, 0] * 9
        for i in range(len(rows)):
            csv.writer(fh).writerow(ob_rows[i] if ob_rows else default)
    return msg, ob


# ---------------------------------------------------------------------------
# Hawkes horizon honesty
# ---------------------------------------------------------------------------


def test_loglik_silent_tail_costs_likelihood() -> None:
    """Compensator must integrate over [0, T], not stop at the last event."""
    t = np.asarray([1.0, 2.0, 3.0])
    short = hawkes_exp_loglik(t, 0.5, 0.3, 2.0, horizon=3.0)
    long = hawkes_exp_loglik(t, 0.5, 0.3, 2.0, horizon=30.0)
    assert math.isfinite(short) and math.isfinite(long)
    assert long < short  # a longer silent tail is less likely under same params


def test_loglik_rejects_horizon_before_last_event() -> None:
    t = np.asarray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="horizon"):
        hawkes_exp_loglik(t, 0.5, 0.3, 2.0, horizon=2.5)


def test_fit_honors_observation_window() -> None:
    """Same events, bigger window -> smaller fitted baseline mu."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.5, 10.0, size=60))
    fit_short = fit_hawkes_exp(t, horizon=10.0)
    fit_long = fit_hawkes_exp(t, horizon=100.0)
    assert fit_long.mu < fit_short.mu


# ---------------------------------------------------------------------------
# Zero-fill market steps must not fabricate MO arrivals
# ---------------------------------------------------------------------------


class _ZeroFillSim:
    """step() -> "market" but only every third call appends a trade."""

    def __init__(self, *_a: Any, **_k: Any) -> None:
        self.t = 0.0
        self.trades: list[Any] = []
        self.n = 0

    def step(self) -> str:
        self.t += 1.0
        self.n += 1
        if self.n % 3 == 0:
            self.trades.append(types.SimpleNamespace(aggressor="buy"))
        return "market"

    @property
    def mid(self) -> float:
        return 100.0


class _BurstyFillSim(_ZeroFillSim):
    """All steps claim "market"; fills only in the middle third."""

    def step(self) -> str:
        self.t += 1.0
        self.n += 1
        if 30 < self.n <= 60 and self.n % 3 == 0:
            self.trades.append(types.SimpleNamespace(aggressor="buy"))
        return "market"


def test_zi_mo_times_counts_fills_not_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.microstructure.zi_lob_simulator as zsim

    sims: list[_ZeroFillSim] = []

    def factory(*a: Any, **k: Any) -> _ZeroFillSim:
        s = _ZeroFillSim()
        sims.append(s)
        return s

    monkeypatch.setattr(zsim, "ZILobSimulator", factory)
    times = zi_mo_times(object(), 30.0)
    assert times.tolist() == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]


def test_vol_clock_event_clock_skips_zero_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vol_clock, "ZILobSimulator", _ZeroFillSim)
    tape = vol_clock.collect_tape(config=object(), horizon=30.0)  # type: ignore[arg-type]
    assert tape.n_mo == 10  # 30 market steps, only every third fills


def test_hmm_discretize_zero_fill_windows_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hmm_learn, "ZILobSimulator", _BurstyFillSim)
    syms = hmm_learn.discretize_tape(
        config=object(),  # type: ignore[arg-type]
        horizon=90.0,
        flow=None,  # type: ignore[arg-type]
        window_s=30.0,
    )
    # Windows [0,30) quiet, [30,60) all buy fills, [60,90) quiet: with
    # zero-fill steps counted every window saturates; fill-only counting
    # must light up exactly the middle window (2*hi_rate + hi_buy = 3).
    assert syms.tolist() == [0, 3, 0]


# ---------------------------------------------------------------------------
# EXECUTION_HIDDEN inclusion
# ---------------------------------------------------------------------------


def test_mo_times_lobster_counts_hidden_fills(tmp_path: Path) -> None:
    msg, _ = _write_tape(
        tmp_path,
        [
            [34200.00, SUBMISSION, 1, 50, 9900, -1],
            [34200.01, SUBMISSION, 2, 50, 10100, 1],
            [34200.02, EXECUTION, 2, 50, 10100, 1],
            [34200.03, EXECUTION_HIDDEN, 9, 50, 10100, 1],
        ],
    )
    times = mo_times_lobster(msg)
    assert times.size == 2  # visible + hidden, both are MO arrivals


def test_round_lot_includes_hidden_fills(tmp_path: Path) -> None:
    _write_tape(
        tmp_path,
        [
            [34200.00, SUBMISSION, 1, 50, 9900, -1],
            [34200.01, SUBMISSION, 2, 100, 10100, 1],
            [34200.02, EXECUTION, 2, 40, 10100, 1],
            [34200.03, EXECUTION_HIDDEN, 9, 100, 10100, 1],
        ],
    )
    out = lobster_round_lot(tmp_path, _TICKER)
    assert out["n_trades"] == 2


def test_streaks_include_hidden_fills(tmp_path: Path) -> None:
    _write_tape(
        tmp_path,
        [
            [34200.00, SUBMISSION, 1, 50, 9900, -1],
            [34200.01, SUBMISSION, 2, 100, 10100, 1],
            [34200.02, EXECUTION, 2, 40, 10100, 1],
            [34200.03, EXECUTION_HIDDEN, 1, 20, 9900, -1],
        ],
    )
    out = lobster_streaks(tmp_path, _TICKER)
    assert out["n_execs"] == 2


def test_granger_exec_channel_folds_hidden() -> None:
    n = 64
    counts = {
        "submit": np.zeros(n),
        "cancel_partial": np.zeros(n),
        "delete": np.zeros(n),
        "exec": np.zeros(n),
        "exec_hidden": np.zeros(n),
        "halt": np.zeros(n),
    }
    # hidden fills arrive exactly one bin after submissions
    counts["submit"][::4] = 1.0
    counts["exec_hidden"][1::4] = 1.0
    pairs = event_granger._pairwise(counts)  # noqa: SLF001
    assert pairs["submit->exec"]["peak_corr_lead"] > 0.9
    # the dead-channel bug: with no visible execs the pair must still score
    assert pairs["exec->submit"]["peak_corr"] > 0.5


# ---------------------------------------------------------------------------
# LOBSTER event-0 semantics + event accounting
# ---------------------------------------------------------------------------


def test_intraday_seed_row_not_double_applied(tmp_path: Path) -> None:
    """Row i is the book state AFTER message i: seeding from row 0 already
    includes event 0 — applying it too would double-count the first event.
    A type-5 hidden exec and a HALT must both be *counted* (they are real
    events), with the halt excluded from spread sampling."""
    _write_tape(
        tmp_path,
        [
            [34200.00, SUBMISSION, 1, 50, 9900, -1],  # event 0: seed row
            [34200.10, SUBMISSION, 2, 50, 10100, 1],
            [34200.20, EXECUTION_HIDDEN, 9, 50, 10100, 1],
            [34200.30, HALT, 0, 0, 0, 0],
            [34200.40, SUBMISSION, 3, 30, 10110, 1],
        ],
    )
    out = lobster_intraday(tmp_path, _TICKER, n_bins=13)
    # 4 counted events (seed row is state, not an event)
    assert sum(out["events_per_bin"]) == 4
    # hidden exec counted as an execution
    assert sum(out["execs_per_bin"]) == 1


def test_seed_book_oids_not_attributed_to_first_event() -> None:
    """The seeded book is sim state; only event-minted orders count."""
    cfg = ZILobConfig(init_levels=3, init_depth=5, mu=0.5, lam=2.0, theta_cxl=1e-9, seed=3)
    out = sim_cluster_stats(config=cfg, horizon=0.2)
    # 0.2 sim-seconds: only a handful of events fire (~5), but the seeded
    # book rests 30 orders. With the empty-initial-diff bug the first
    # step's diff sweeps in all ~30 seed oids as if the first event
    # placed them.
    assert out["n_submits"] <= 2 * out["n_events"] + 5


# ---------------------------------------------------------------------------
# Sealed-receipt gating in synthesis maps
# ---------------------------------------------------------------------------


def _fake_receipt(fname: str, *, sealed: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": "x.v1",
        "kind": "x",
        "claims": {"winner": "planted"},
        "cells": [{"all_in_tol": True}],
    }
    if sealed:
        return seal_receipt(body)
    body["receipt_sha256"] = "0" * 64  # tampered / forged seal
    return body


def test_wave23_map_ignores_unsealed_claims(tmp_path: Path) -> None:
    # only the first lane present, with a forged seal: its claims must not
    # count as evidence even though the file parses.
    lane_file = _LANES[0][0]
    (tmp_path / lane_file).write_text(json.dumps(_fake_receipt(lane_file, sealed=False)))
    out = wave23_map(tmp_path)
    lane = next(e for e in out["lanes"] if e["receipt"] == lane_file)
    assert lane["sealed"] is False
    assert lane["claims"] is None
    assert lane["n_cells_all_in_tol"] == 0
    assert out["claims"]["all_lanes_present"] is False


def test_wave23_map_reads_sealed_claims(tmp_path: Path) -> None:
    lane_file = _LANES[0][0]
    (tmp_path / lane_file).write_text(json.dumps(_fake_receipt(lane_file, sealed=True)))
    out = wave23_map(tmp_path)
    lane = next(e for e in out["lanes"] if e["receipt"] == lane_file)
    assert lane["sealed"] is True
    assert lane["claims"] == {"winner": "planted"}
    assert lane["n_cells_all_in_tol"] == 1


# ---------------------------------------------------------------------------
# Receipt kind/schema convention
# ---------------------------------------------------------------------------


def test_sweep_width_receipt_convention(tmp_path: Path) -> None:
    _write_tape(
        tmp_path,
        [
            [34200.00, SUBMISSION, 1, 50, 9900, -1],
            [34200.01, SUBMISSION, 2, 50, 10100, 1],
            [34200.02, EXECUTION, 2, 50, 10100, 1],
        ],
    )
    payload = sweep_width_bench(tmp_path)
    assert payload["kind"] == "sweep_width"
    assert payload["schema"] == "sweep_width.v1"
    # seal must cover the emitted body
    assert (
        payload["receipt_sha256"]
        == seal_receipt({k: v for k, v in payload.items() if k != "receipt_sha256"})[
            "receipt_sha256"
        ]
    )


# ---------------------------------------------------------------------------
# GLFT/Hawkes constants
# ---------------------------------------------------------------------------


def test_glft_forcing_has_no_gamma_divisor() -> None:
    """γ·θ̇ = (γσ²/2)q² − ν[...] ⇒ θ̇ forcing term is σ²q²/2, not σ²q²/(2γ)."""
    q_idx = np.arange(-3, 4)
    th = np.zeros(7)
    for gamma in (0.25, 0.5, 1.0, 4.0):
        rhs = _theta_rhs(th, q_idx, sigma=0.02, gamma=gamma, k=1.5, nu=0.0)
        expected = (0.02**2 / 2.0) * q_idx.astype(float) ** 2
        # nu=0 ⇒ the only term is the γ-free forcing
        np.testing.assert_allclose(rhs, expected, rtol=0, atol=1e-15)


def test_glft_theta_gamma_invariant_when_nu_negligible() -> None:
    """With A→0 the ν terms vanish and θ_q ≈ (σ²q²/2)τ — γ-free.

    The old code's extra 1/γ on the forcing would make θ scale as 1/γ
    (a 16x spread across this γ range); the corrected forcing is γ-free.
    """
    kw: dict[str, Any] = {"k": 1.5, "A": 1e-9, "sigma": 0.02, "T": 0.05, "q_max": 4}
    th_lo = theta_closed_form(gamma=0.25, **kw)
    th_hi = theta_closed_form(gamma=4.0, **kw)
    np.testing.assert_allclose(th_lo, th_hi, rtol=1e-4, atol=1e-9)
    # shape is the expected quadratic: |θ| ∝ q² (θ_±Q ≈ 16·θ_±1)
    assert abs(th_lo[0] / th_lo[3]) == pytest.approx(16.0, rel=0.02)
