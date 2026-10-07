"""Tests for beveridge_nelson — permanent/transitory decomposition."""

import numpy as np
import pytest

from quant_fund.models.beveridge_nelson import (
    bench_beveridge_nelson,
    beveridge_nelson,
    synth_bn,
)


def test_permanent_tracks_martingale() -> None:
    d = synth_bn(seed=1)
    r = beveridge_nelson(np.asarray(d["y"]), p=3)
    perm_err = np.sqrt(np.mean((np.asarray(r["permanent"]) - np.asarray(d["perm_true"])[3:]) ** 2))
    assert perm_err / np.std(np.asarray(d["perm_true"])[3:]) < 0.5


def test_cycle_correlates_with_truth() -> None:
    d = synth_bn(seed=2)
    r = beveridge_nelson(np.asarray(d["y"]), p=3)
    corr = np.corrcoef(np.asarray(r["cycle"]), np.asarray(d["cyc_true"])[3:])[0, 1]
    assert corr > 0.6


def test_decomposition_adds_up() -> None:
    d = synth_bn(seed=3)
    r = beveridge_nelson(np.asarray(d["y"]), p=3)
    np.testing.assert_allclose(
        np.asarray(r["permanent"]) + np.asarray(r["cycle"]),
        np.asarray(r["y"]),
        atol=1e-10,
    )


def test_cycle_is_stationary_shaped() -> None:
    d = synth_bn(seed=4)
    r = beveridge_nelson(np.asarray(d["y"]), p=3)
    cyc = np.asarray(r["cycle"])
    assert abs(np.mean(cyc)) < np.std(cyc)
    assert r["max_root"] < 1.0


def test_drift_recovery() -> None:
    d = synth_bn(seed=5, mu=0.01)
    r = beveridge_nelson(np.asarray(d["y"]), p=3)
    assert abs(float(r["mu"]) - 0.01) < 0.01


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        beveridge_nelson(np.ones(20))
    with pytest.raises(ValueError):
        beveridge_nelson(np.array([np.nan] * 100))
    with pytest.raises(ValueError):
        beveridge_nelson(np.random.default_rng(0).standard_normal(200), p=0)


def test_determinism() -> None:
    d = synth_bn(seed=7)
    a = beveridge_nelson(np.asarray(d["y"]), p=3)
    b = beveridge_nelson(np.asarray(d["y"]), p=3)
    np.testing.assert_array_equal(a["permanent"], b["permanent"])
    np.testing.assert_array_equal(a["cycle"], b["cycle"])


def test_bench_schema_and_score() -> None:
    r = bench_beveridge_nelson()
    for k in (
        "synthetic_rel_err",
        "synthetic_cyc_corr",
        "synthetic_mu_hat",
        "synthetic_mu_true",
        "synthetic_max_root",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
