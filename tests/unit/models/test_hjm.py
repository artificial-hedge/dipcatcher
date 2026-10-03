"""Unit tests for quant_fund.models.hjm."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hjm import bench_hjm, hjm_simulate, synth_hjm


def test_simulated_price_brackets_analytic() -> None:
    d = synth_hjm(seed=1)
    out = hjm_simulate(d["f0"], d["maturities"], n_paths=1500, seed=7)
    assert out["bond_price_p5"] < out["bond_price_init"] < out["bond_price_p95"]
    assert abs(out["bond_price_mean"] - out["bond_price_init"]) < 0.2


def test_no_arb_drift_is_positive() -> None:
    d = synth_hjm(seed=2)
    out = hjm_simulate(d["f0"], d["maturities"], n_paths=200, seed=3)
    assert out["drift_magnitude"] >= 0.0


def test_schema() -> None:
    d = synth_hjm(seed=2)
    out = hjm_simulate(d["f0"], d["maturities"], n_paths=200, seed=3)
    assert set(out) == {
        "bond_price_mean",
        "bond_price_p5",
        "bond_price_p95",
        "bond_price_init",
        "f_mean_mid",
        "f_std_mid",
        "drift_magnitude",
        "_f_path",
    }


def test_determinism() -> None:
    d = synth_hjm(seed=5)
    a = hjm_simulate(d["f0"], d["maturities"], n_paths=200, seed=11)
    b = hjm_simulate(d["f0"], d["maturities"], n_paths=200, seed=11)
    assert np.array_equal(np.asarray(a["_f_path"]), np.asarray(b["_f_path"]))
    assert {k: v for k, v in a.items() if k != "_f_path"} == {
        k: v for k, v in b.items() if k != "_f_path"
    }


def test_validation() -> None:
    d = synth_hjm(seed=1)
    with pytest.raises(ValueError):
        hjm_simulate(d["f0"], np.ones(4))  # shape mismatch
    with pytest.raises(ValueError):
        hjm_simulate(np.full(10, np.nan), np.linspace(0, 5, 10))
    with pytest.raises(ValueError):
        hjm_simulate(d["f0"], d["maturities"], n_paths=5)
    with pytest.raises(ValueError):
        hjm_simulate(d["f0"], d["maturities"], sigma0=-0.1)
    with pytest.raises(ValueError):
        hjm_simulate(d["f0"], d["maturities"][::-1])  # non-increasing


def test_bench_keys_and_pass() -> None:
    out = bench_hjm()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_price_mean",
        "synthetic_price_init",
        "synthetic_price_p5",
        "synthetic_price_p95",
        "synthetic_f_std",
        "synthetic_drift",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
