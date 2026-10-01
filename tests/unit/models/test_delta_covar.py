"""Unit tests for quant_fund.models.delta_covar."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.delta_covar import (
    bench_delta_covar,
    delta_covar,
    synth_covar,
)


def test_negative_delta_under_tail_dependence() -> None:
    d = synth_covar(seed=1)
    out = delta_covar(d["r_sys"], d["r1"])
    assert out["delta_covar"] < 0.0
    assert out["var_inst"] < 0.0


def test_heavy_institution_dominates() -> None:
    d = synth_covar(seed=3)
    a = delta_covar(d["r_sys"], d["r1"])
    b = delta_covar(d["r_sys"], d["r2"])
    assert a["delta_covar"] < b["delta_covar"]


def test_schema() -> None:
    d = synth_covar(seed=2)
    out = delta_covar(d["r_sys"], d["r1"])
    assert set(out) == {
        "var_inst",
        "covar",
        "covar_median",
        "delta_covar",
        "beta_tau",
    }


def test_determinism() -> None:
    d = synth_covar(seed=5)
    a = delta_covar(d["r_sys"], d["r1"])
    b = delta_covar(d["r_sys"], d["r1"])
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        delta_covar(np.ones(10), np.ones(10))  # too short
    with pytest.raises(ValueError):
        delta_covar(np.ones(100), np.ones(50))  # length mismatch
    with pytest.raises(ValueError):
        delta_covar(np.full(100, np.nan), np.ones(100))
    with pytest.raises(ValueError):
        delta_covar(
            np.random.default_rng(0).normal(size=100),
            np.random.default_rng(1).normal(size=100),
            tau=0.6,
        )


def test_bench_keys_and_pass() -> None:
    out = bench_delta_covar()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_dcovar_heavy",
        "synthetic_dcovar_light",
        "synthetic_covar",
        "synthetic_var_inst",
        "synthetic_beta_tau",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
