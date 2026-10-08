"""Tests for sheppard_heavy — HEAVY(P) realized-measure model."""

import numpy as np
import pytest

from quant_fund.models.sheppard_heavy import (
    bench_sheppard_heavy,
    fit_heavy,
    synth_heavy,
)


def test_tracks_true_variance() -> None:
    d = synth_heavy(seed=1)
    r = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    s2_hat = np.asarray(r["sigma2"])
    s2_t = np.asarray(d["s2_true"])
    rel = np.sqrt(np.mean(((s2_hat[100:] - s2_t[100:]) / s2_t[100:]) ** 2))
    assert rel < 0.5


def test_persistence_interior() -> None:
    d = synth_heavy(seed=2)
    r = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    assert 0.3 < r["persistence_r"] < 0.999
    assert r["alpha_r"] >= 0.0
    assert r["beta_r"] >= 0.0


def test_variance_path_positive() -> None:
    d = synth_heavy(seed=3)
    r = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    assert np.all(np.asarray(r["sigma2"]) > 0.0)
    assert np.all(np.asarray(r["mu2"]) > 0.0)


def test_regime_shift_captured() -> None:
    d = synth_heavy(seed=4)
    r = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    s2_hat = np.asarray(r["sigma2"])
    # post-break mean variance should exceed pre-break
    n = s2_hat.size
    assert s2_hat[int(0.7 * n) :].mean() > s2_hat[int(0.2 * n) : int(0.45 * n)].mean()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_heavy(np.ones(50), np.ones(50))
    with pytest.raises(ValueError):
        fit_heavy(np.ones(200), -np.ones(200))
    with pytest.raises(ValueError):
        fit_heavy(np.ones(200), np.ones(100))
    with pytest.raises(ValueError):
        fit_heavy(np.full(200, np.nan), np.ones(200))


def test_determinism() -> None:
    d = synth_heavy(seed=6)
    a = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    b = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    np.testing.assert_array_equal(a["sigma2"], b["sigma2"])
    assert a["alpha_r"] == b["alpha_r"]


def test_bench_schema_and_score() -> None:
    r = bench_sheppard_heavy()
    for k in (
        "synthetic_rel_rmse",
        "synthetic_innov_corr",
        "synthetic_persistence_r",
        "synthetic_alpha_r",
        "synthetic_beta_r",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
