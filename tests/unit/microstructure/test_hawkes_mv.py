from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.hawkes_mv import (
    MARKS,
    _merge_streams,
    hawkes_mv_bench,
    hawkes_mv_fit,
    hawkes_mv_loglik,
    hawkes_mv_simulate,
    lobster_marks,
    spectral_radius,
    write_lobster_tape,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

TAPE_NAME = "AMZN_2012-06-21_34200000_57600000_message_10.csv"


def _planted_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2x2 excitation block on the MO marks; other marks are quiet baselines."""
    mu = np.array([0.003, 0.003, 0.0005, 0.0005, 0.0005, 0.0005])
    alpha = np.zeros((6, 6))
    alpha[0, 0] = 0.004
    alpha[0, 1] = 0.003
    alpha[1, 1] = 0.004
    alpha[1, 0] = 0.002
    beta = np.full(6, 0.01)
    return mu, alpha, beta


def _synthetic_tape(tmp_path: Path, seed: int = 3) -> Path:
    mu = np.full(6, 0.002)
    alpha = np.full((6, 6), 0.001)
    alpha[0, 0] = 0.004
    alpha[1, 1] = 0.004
    beta = np.full(6, 0.01)
    streams = hawkes_mv_simulate(mu, alpha, beta, 8e5, seed=seed)
    path = tmp_path / TAPE_NAME
    write_lobster_tape(streams, path)
    return path


def test_hawkes_mv_fit_recovers_planted_excitation() -> None:
    mu, alpha, beta = _planted_params()
    assert spectral_radius(alpha / beta[None, :]) < 1.0
    streams = hawkes_mv_simulate(mu, alpha, beta, 1.2e6, seed=42)
    assert sum(v.size for v in streams.values()) >= 20000
    fit = hawkes_mv_fit(streams)
    assert fit.stationary
    assert fit.rho < 1.0
    # spec: alpha error < 0.15 on the planted block (observed ~0.002)
    err = float(np.abs(fit.alpha - alpha).max())
    assert err < 0.15
    # rho should land near the planted branching spectral radius
    rho_true = spectral_radius(alpha / beta[None, :])
    assert abs(fit.rho - rho_true) < 0.1


def test_hawkes_mv_supercritical_simulate_refused() -> None:
    mu = np.full(6, 0.001)
    alpha = np.full((6, 6), 0.01)  # B = 1.0 everywhere -> rho >> 1
    beta = np.full(6, 0.005)
    assert spectral_radius(alpha / beta[None, :]) > 1.0
    with pytest.raises(ValueError):
        hawkes_mv_simulate(mu, alpha, beta, 1000.0, seed=1)


def test_hawkes_mv_fit_enforces_rho_cap() -> None:
    # clustered streams that tempt a supercritical fit still report rho < 1
    rng = np.random.default_rng(9)
    streams = {}
    for i, name in enumerate(MARKS):
        bursts = np.sort(rng.uniform(0, 2000, 60))
        repeats = np.concatenate([bursts + off for off in (0.05, 0.1, 0.2, 0.4)])
        streams[name] = np.sort(repeats + rng.uniform(0, 0.02, repeats.size) * i)
    fit = hawkes_mv_fit(streams, rho_cap=0.9)
    assert fit.rho < 0.9 + 1e-9


def test_hawkes_mv_loglik_matches_direct_sum() -> None:
    rng = np.random.default_rng(0)
    streams = {MARKS[i]: np.sort(rng.uniform(0, 4000, 150)) for i in range(3)}
    for i in range(3, 6):
        streams[MARKS[i]] = np.sort(rng.uniform(0, 4000, 80))
    mu = np.full(6, 0.02)
    alpha = np.full((6, 6), 0.0005)
    beta = np.full(6, 0.004)
    horizon = 4000.0
    fast = hawkes_mv_loglik(streams, mu, alpha, beta, horizon)
    times, mi = _merge_streams(streams)
    tl, ml = times.tolist(), mi.tolist()
    direct = 0.0
    for i, ti in enumerate(tl):
        lam = mu[ml[i]]
        for k in range(i):
            lam += alpha[ml[i], ml[k]] * math.exp(-beta[ml[k]] * (ti - tl[k]))
        direct += math.log(lam)
    for m in range(6):
        direct -= mu[m] * horizon
        for k, tk in enumerate(tl):
            direct -= (
                alpha[m, ml[k]] * (1.0 - math.exp(-beta[ml[k]] * (horizon - tk))) / beta[ml[k]]
            )
    assert fast == pytest.approx(direct, rel=1e-9, abs=1e-6)


def test_hawkes_mv_poisson_null_rho_small() -> None:
    rng = np.random.default_rng(11)
    streams = {name: np.sort(rng.uniform(0, 2e5, 2500)) for name in MARKS}
    fit = hawkes_mv_fit(streams)
    assert fit.stationary
    assert fit.rho < 0.3


def test_lobster_marks_roundtrip(tmp_path: Path) -> None:
    path = _synthetic_tape(tmp_path)
    marks = lobster_marks(path)
    assert set(marks) == set(MARKS)
    for name in MARKS:
        assert marks[name].size > 0
        assert np.all(np.diff(marks[name]) >= 0)


def test_hawkes_mv_bench_synthetic_e2e(tmp_path: Path) -> None:
    _synthetic_tape(tmp_path)
    payload = hawkes_mv_bench(tmp_path, ticker="AMZN")
    assert payload["kind"] == "hawkes_mv"
    assert payload["schema"] == "hawkes_mv.v1"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False
    assert payload["claim"]["n_probes"] == payload["claim"]["n_passed"]
    assert payload["claim"]["ok"] is True
    # sealed-receipt contract: hash over the canonical payload minus the seal
    sealed = payload.pop("receipt_sha256")
    assert sealed == hash_bytes(canonical_json_bytes(payload))
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    assert forbidden.isdisjoint(payload.keys())


def test_hawkes_mv_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hawkes_mv_bench(tmp_path)
