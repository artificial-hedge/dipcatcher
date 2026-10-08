"""Tests for cochrane_piazzesi — return-forecasting bond factor."""

import numpy as np
import pytest

from quant_fund.models.cochrane_piazzesi import (
    bench_cochrane_piazzesi,
    cp_factor,
    cp_restricted,
    synth_cp,
)


def test_factor_captures_predictability() -> None:
    f, rx, _, _ = synth_cp(seed=1)
    r = cp_factor(f, rx.mean(axis=1))
    assert r["r2"] > 0.6


def test_tent_shape() -> None:
    f, rx, _, _ = synth_cp(seed=2)
    g = np.asarray(cp_factor(f, rx.mean(axis=1))["gamma"])
    assert g[2] > g[0] and g[2] > g[4]


def test_one_factor_restriction() -> None:
    f, rx, _, _ = synth_cp(seed=3)
    g = np.asarray(cp_factor(f, rx.mean(axis=1))["gamma"])
    r = cp_restricted(f, rx, g)
    assert r["r2_restricted"] > 0.5


def test_null_unpredictable() -> None:
    _, _, f0, rx0 = synth_cp(seed=4)
    r = cp_factor(f0, rx0.mean(axis=1))
    assert r["r2"] < 0.15


def test_fail_closed() -> None:
    f, rx, _, _ = synth_cp(seed=5)
    with pytest.raises(ValueError):
        cp_factor(f[:30], rx[:30].mean(axis=1))
    with pytest.raises(ValueError):
        cp_factor(np.full((100, 5), np.nan), np.ones(100))
    with pytest.raises(ValueError):
        cp_factor(f, np.zeros(rx.shape[0]))
    with pytest.raises(ValueError):
        cp_restricted(f, rx, np.ones(3))


def test_determinism() -> None:
    f, rx, _, _ = synth_cp(seed=6)
    assert cp_factor(f, rx.mean(axis=1))["r2"] == cp_factor(f, rx.mean(axis=1))["r2"]


def test_bench_schema_and_score() -> None:
    r = bench_cochrane_piazzesi()
    for k in (
        "synthetic_r2_full",
        "synthetic_r2_restricted",
        "synthetic_gamma3_peak",
        "synthetic_r2_null",
        "synthetic_r2_null_restr",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
