"""Unit tests for quant_fund.models.affine_term."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.affine_term import (
    affine_term,
    bench_affine_term,
    synth_ats,
)


def test_recovers_yield_loading() -> None:
    d = synth_ats(seed=1)
    out = affine_term(d["short_rate"], d["long_rate"])
    b_true = float(np.asarray(d["b_true"])[0])
    assert abs(out["b_hat"] - b_true) < 0.1


def test_eh_slope_below_one() -> None:
    d = synth_ats(seed=3)
    out = affine_term(d["short_rate"], d["long_rate"])
    assert out["eh_slope"] < 1.0


def test_schema() -> None:
    d = synth_ats(seed=2)
    out = affine_term(d["short_rate"], d["long_rate"])
    assert set(out) == {
        "kappa_p",
        "theta_p",
        "sigma_r",
        "b_hat",
        "kappa_q",
        "term_premium",
        "eh_slope",
        "spread_mean",
        "n",
    }


def test_determinism() -> None:
    d = synth_ats(seed=5)
    a = affine_term(d["short_rate"], d["long_rate"])
    b = affine_term(d["short_rate"], d["long_rate"])
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        affine_term(np.ones(30), np.ones(30))
    with pytest.raises(ValueError):
        affine_term(np.ones(100), np.ones(50))
    with pytest.raises(ValueError):
        affine_term(np.full(100, np.nan), np.ones(100))
    with pytest.raises(ValueError):
        affine_term(
            np.random.default_rng(0).normal(size=100),
            np.random.default_rng(1).normal(size=100),
            tau=0.0,
        )


def test_bench_keys_and_pass() -> None:
    out = bench_affine_term()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_b_hat",
        "synthetic_kappa_p",
        "synthetic_term_premium",
        "synthetic_eh_slope",
        "synthetic_spread_mean",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
