"""Tests for muller_watson — low-frequency inference."""

import numpy as np
import pytest

from quant_fund.models.muller_watson import (
    bench_muller_watson,
    lf_predictive_test,
    low_freq_correlation,
    synth_mw,
)


def test_shared_driver_high_correlation() -> None:
    x, y = synth_mw(seed=1, shared=True)
    r = low_freq_correlation(x, y, q=16)
    assert r["rho_lf"] > 0.5
    assert r["rho_lo"] > 0.0


def test_independent_pair_near_zero() -> None:
    x, y = synth_mw(seed=2, shared=False)
    r = low_freq_correlation(x, y, q=16)
    assert abs(r["rho_lf"]) < 0.45
    assert r["rho_lo"] < 0.0 < r["rho_hi"]


def test_predictive_beta_recovers() -> None:
    x, y = synth_mw(seed=3, shared=True)
    r = lf_predictive_test(x, y, q=16)
    assert r["beta_lf"] > 0.5
    assert r["p_lf"] < 0.01


def test_cosine_transform_scaling() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(400)
    r = low_freq_correlation(x, rng.standard_normal(400), q=10)
    # white noise S(0) ~ 1 → mean X_j^2 ~ O(1)
    assert 0.2 < r["s0_x"] < 5.0


def test_fail_closed() -> None:
    x, y = synth_mw(seed=5)
    with pytest.raises(ValueError):
        low_freq_correlation(x[:40], y[:40])
    with pytest.raises(ValueError):
        low_freq_correlation(x, y[:100])
    with pytest.raises(ValueError):
        low_freq_correlation(np.full(400, np.nan), y[:400])
    with pytest.raises(ValueError):
        low_freq_correlation(x, y, q=2)


def test_determinism() -> None:
    x, y = synth_mw(seed=6)
    a = low_freq_correlation(x, y, q=16)
    b = low_freq_correlation(x, y, q=16)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_muller_watson()
    for k in (
        "synthetic_rho_shared",
        "synthetic_rho_indep",
        "synthetic_p_beta",
        "synthetic_beta_lf",
        "synthetic_ur_stat",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
