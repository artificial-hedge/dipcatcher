"""Unit tests for quant_fund.models.tvp_var."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.tvp_var import (
    bench_tvp_var,
    synth_tvp,
    tvp_regression,
)


def test_tracks_mid_sample_break() -> None:
    d = synth_tvp(seed=1)
    out = tvp_regression(d["y"], d["x"])
    path = np.asarray(out["_path"])
    n = path.shape[0]
    pre = float(np.mean(path[: n // 3, 1]))
    post = float(np.mean(path[2 * n // 3 :, 1]))
    assert post - pre > 0.6


def test_constant_beta_recovers_ols() -> None:
    rng = np.random.default_rng(4)
    n = 300
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = 0.2 + 1.1 * x[:, 1] + rng.normal(0.0, 0.2, n)
    out = tvp_regression(y, x, lam=0.001)
    assert abs(out["beta_final_1"] - 1.1) < 0.3


def test_schema() -> None:
    d = synth_tvp(seed=2)
    out = tvp_regression(d["y"], d["x"])
    assert "_path" in out
    assert set(out) - {"_path"} == {
        "mse_innovation",
        "beta_final_0",
        "beta_final_1",
        "beta_early_0",
        "beta_early_1",
        "beta_mid_1",
        "state_drift",
        "break_magnitude",
        "loglik_per_obs",
    }


def test_determinism() -> None:
    d = synth_tvp(seed=5)
    a = tvp_regression(d["y"], d["x"])
    b = tvp_regression(d["y"], d["x"])
    assert np.array_equal(np.asarray(a["_path"]), np.asarray(b["_path"]))
    assert {k: v for k, v in a.items() if k != "_path"} == {
        k: v for k, v in b.items() if k != "_path"
    }


def test_validation() -> None:
    with pytest.raises(ValueError):
        tvp_regression(np.ones(10), np.ones((10, 1)))  # too short
    with pytest.raises(ValueError):
        tvp_regression(np.ones(50), np.ones(50))  # x not 2d
    with pytest.raises(ValueError):
        tvp_regression(np.full(50, np.nan), np.ones((50, 1)))
    with pytest.raises(ValueError):
        tvp_regression(np.random.default_rng(0).normal(size=50), np.ones((50, 1)), lam=0.0)


def test_bench_keys_and_pass() -> None:
    out = bench_tvp_var()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_pre_beta1",
        "synthetic_post_beta1",
        "synthetic_break_jump",
        "synthetic_mse",
        "synthetic_beta0_drift",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
