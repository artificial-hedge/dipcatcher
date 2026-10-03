"""Unit tests for quant_fund.models.meta_analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.meta_analysis import (
    bench_meta_analysis,
    meta_analysis,
    synth_meta,
)


def test_re_covers_common_effect() -> None:
    d = synth_meta(seed=1)
    out = meta_analysis(d["effects"], d["se"])
    assert abs(out["theta_re"] - 0.4) < 0.15


def test_i2_positive_with_heterogeneity() -> None:
    d = synth_meta(seed=3, hetero_sd=0.2)
    out = meta_analysis(d["effects"], d["se"])
    assert out["i2"] > 0.0
    assert out["tau2"] > 0.0


def test_egger_flags_small_study_bias() -> None:
    d = synth_meta(seed=5, bias_small=3.0)
    out = meta_analysis(d["effects"], d["se"])
    assert abs(out["egger_t"]) > 2.0


def test_schema() -> None:
    d = synth_meta(seed=2)
    out = meta_analysis(d["effects"], d["se"])
    assert set(out) == {
        "theta_fe",
        "theta_re",
        "se_re",
        "q_stat",
        "tau2",
        "i2",
        "egger_intercept",
        "egger_t",
        "egger_slope",
        "k",
    }


def test_determinism() -> None:
    d = synth_meta(seed=5)
    a = meta_analysis(d["effects"], d["se"])
    b = meta_analysis(d["effects"], d["se"])
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        meta_analysis(np.ones(3), np.ones(3))  # too few studies
    with pytest.raises(ValueError):
        meta_analysis(np.ones(8), np.zeros(8))  # non-positive se
    with pytest.raises(ValueError):
        meta_analysis(np.full(8, np.nan), np.full(8, 0.1))
    with pytest.raises(ValueError):
        meta_analysis(np.ones(8), np.ones(4))  # length mismatch


def test_bench_keys_and_pass() -> None:
    out = bench_meta_analysis()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_theta_re",
        "synthetic_tau2",
        "synthetic_i2",
        "synthetic_egger_t_biased",
        "synthetic_egger_t_null",
        "synthetic_q_stat",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
