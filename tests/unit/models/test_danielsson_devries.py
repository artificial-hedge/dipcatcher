"""Tests for danielsson_devries — tail-simulation VaR."""

import numpy as np
import pytest
from scipy import stats as _stats

from quant_fund.models.danielsson_devries import (
    bench_danielsson_devries,
    dd_var,
    hill_gamma,
    optimal_k,
    pot_var,
    synth_dd,
)


def test_xi_recovery_student_t() -> None:
    x = synth_dd(seed=1)
    r = pot_var(x, 0.01)
    assert 0.15 < r["xi"] < 0.55


def test_var_near_theory() -> None:
    x = synth_dd(seed=2)
    r = dd_var(x, 0.01, n_sim=100, seed=2)
    true_q = _stats.t.ppf(0.99, 3.0)
    assert abs(r["var"] - true_q) / true_q < 0.45


def test_beats_gaussian() -> None:
    x = synth_dd(seed=3)
    r = dd_var(x, 0.01, n_sim=100, seed=3)
    true_q = _stats.t.ppf(0.99, 3.0)
    gauss_q = _stats.norm.ppf(0.99)
    assert abs(r["var"] - true_q) < abs(gauss_q - true_q)


def test_sim_interval_brackets() -> None:
    x = synth_dd(seed=4)
    r = dd_var(x, 0.01, n_sim=100, seed=4)
    assert r["var_lo"] < r["var"] < r["var_hi"]


def test_optimal_k_sane() -> None:
    x = synth_dd(seed=5)
    k = optimal_k(x, seed=5, n_boot=40)
    assert 15 <= k <= 1000


def test_fail_closed() -> None:
    x = synth_dd(seed=6)
    srt = np.sort(x)[::-1]
    with pytest.raises(ValueError):
        hill_gamma(srt, 5)
    with pytest.raises(ValueError):
        pot_var(x[:100], 0.01)
    with pytest.raises(ValueError):
        pot_var(x, 0.5)
    with pytest.raises(ValueError):
        dd_var(np.full(500, np.nan), 0.01)
    with pytest.raises(ValueError):
        pot_var(-np.abs(x), 0.01)


def test_determinism() -> None:
    x = synth_dd(seed=7)
    a = dd_var(x, 0.01, n_sim=60, seed=7)
    b = dd_var(x, 0.01, n_sim=60, seed=7)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_danielsson_devries()
    for k in (
        "synthetic_xi_hat",
        "synthetic_var_hat",
        "synthetic_var_true",
        "synthetic_var_gauss",
        "synthetic_k_opt",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
