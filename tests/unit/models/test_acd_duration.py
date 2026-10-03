"""Unit tests for quant_fund.models.acd_duration."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.acd_duration import (
    acd_fit,
    bench_acd_duration,
    synth_acd,
)


def test_recovers_persistence() -> None:
    d = synth_acd(seed=1)
    out = acd_fit(d["durations"])
    assert out["persistence"] > 0.7
    assert out["alpha"] > 0.0
    assert out["beta"] > 0.0


def test_standardized_residuals_white() -> None:
    d = synth_acd(seed=3)
    out = acd_fit(d["durations"])
    assert abs(out["eps_acf1"]) < 0.15
    assert abs(out["eps_mean"] - 1.0) < 0.1


def test_schema() -> None:
    d = synth_acd(seed=2)
    out = acd_fit(d["durations"])
    assert set(out) == {
        "omega",
        "alpha",
        "beta",
        "persistence",
        "mean_duration",
        "eps_acf1",
        "eps_mean",
        "nll",
    }


def test_determinism() -> None:
    d = synth_acd(seed=5)
    a = acd_fit(d["durations"])
    b = acd_fit(d["durations"])
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        acd_fit(np.ones(30))
    with pytest.raises(ValueError):
        acd_fit(-np.ones(100))
    with pytest.raises(ValueError):
        acd_fit(np.full(100, np.nan))
    with pytest.raises(ValueError):
        synth_acd(n=10)


def test_bench_keys_and_pass() -> None:
    out = bench_acd_duration()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_persistence",
        "synthetic_alpha",
        "synthetic_beta",
        "synthetic_eps_acf1",
        "synthetic_eps_mean",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
