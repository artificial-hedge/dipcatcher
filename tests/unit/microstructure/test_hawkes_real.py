from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.hawkes_real import (
    hawkes_fit,
    hawkes_real_bench,
    mo_times_sim,
)


def _ogata_sample(eta: float, beta: float, mu: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times: list[float] = []
    t = 0.0
    lam = mu
    while len(times) < n:
        t += float(rng.exponential(1.0 / max(lam, 1e-9)))
        r = rng.random() * lam
        hist = sum(eta * beta * math.exp(-beta * (t - ti)) for ti in times[-200:])
        lam_now = mu + hist
        if r <= lam_now:
            times.append(t)
            lam = lam_now + eta * beta
        else:
            lam = lam_now
        if t > 1e7:
            break
    return np.asarray(times)


def test_hawkes_fit_recovers_planted_eta() -> None:
    times = _ogata_sample(eta=0.7, beta=10.0, mu=0.5, n=1500, seed=5)
    out = hawkes_fit(times, float(times.max()))
    assert out["ok"]
    assert 0.4 <= out["branching_ratio_eta"] <= 1.1
    assert out["loglik_gain_vs_poisson"] > 0


def test_hawkes_fit_poisson_data_small_eta() -> None:
    rng = np.random.default_rng(11)
    times = np.sort(rng.uniform(0, 10000, 2000))
    out = hawkes_fit(times, 10000.0)
    assert out["ok"]
    assert out["branching_ratio_eta"] < 0.3


def test_hawkes_fit_fails_closed_on_few_events() -> None:
    out = hawkes_fit(np.asarray([1.0, 2.0, 3.0]), 10.0)
    assert not out["ok"]


def test_mo_times_sim_shape() -> None:
    ts, hz = mo_times_sim(horizon=1500, seed=3)
    assert ts.size > 10
    assert hz > 0
    assert np.all(np.diff(ts) >= 0)


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hawkes_real_bench(tmp_path)
