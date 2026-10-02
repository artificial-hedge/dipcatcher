"""Lin CCC and Bland-Altman tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lin_ccc_blandaltman import (
    bench_lin_ccc,
    bland_altman,
    lin_ccc,
)


def test_ccc_perfect_agreement():
    x = np.linspace(0, 10, 50)
    y = x + np.linspace(-0.01, 0.01, 50)
    out = lin_ccc(x, y)
    assert out["rho_c"] > 0.99


def test_ccc_detects_scale_bias():
    rng = np.random.default_rng(0)
    x = rng.uniform(5, 20, 80)
    y = 1.5 * x + rng.normal(0, 0.5, 80)
    out = lin_ccc(x, y)
    pearson = out["pearson_r"]
    assert out["rho_c"] < pearson - 0.05
    assert out["rho_c_lo"] < out["rho_c"] < out["rho_c_hi"]


def test_ba_recovers_bias():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 100)
    y = x + 2.5 + rng.normal(0, 0.4, 100)
    out = bland_altman(x, y)
    assert abs(out["bias"] + 2.5) < 0.2
    assert out["loa_lo"] < out["bias"] < out["loa_hi"]
    assert out["bias_lo"] <= out["bias"] <= out["bias_hi"]


def test_ba_loa_width_consistent_with_sd():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 5, 80)
    y = x + rng.normal(0, 0.3, 80)
    out = bland_altman(x, y)
    assert abs(out["loa_hi"] - out["loa_lo"] - 2 * 1.96 * out["sd_diff"]) < 1e-9


def test_input_validation():
    with pytest.raises(ValueError):
        lin_ccc(np.arange(3.0), np.arange(3.0))
    with pytest.raises(ValueError):
        bland_altman(np.array([1.0, np.nan, 2.0, 3.0, 4.0]), np.arange(5.0))


def test_bench_passes():
    out = bench_lin_ccc()
    assert out["synthetic_ccc_discriminates"] == 1.0
    assert out["synthetic_ccc_good"] > out["synthetic_ccc_bad"]
    assert out["synthetic_ba_bias_err"] < 1e-6
    assert out["synthetic_score"] == 1.0
