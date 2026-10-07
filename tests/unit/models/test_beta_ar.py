"""Unit tests for quant_fund.models.beta_ar."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.beta_ar import (
    bench_beta_ar,
    beta_ar_fit,
    synth_beta_ar,
)


def test_recovers_phi() -> None:
    y = synth_beta_ar(seed=1)
    out = beta_ar_fit(y)
    assert 0.3 < out["phi"] < 0.9


def test_precision_positive_and_sizeable() -> None:
    y = synth_beta_ar(seed=2)
    out = beta_ar_fit(y)
    assert out["nu"] > 5.0


def test_onestep_beats_marginal() -> None:
    y = synth_beta_ar(seed=3)
    out = beta_ar_fit(y)
    rmse_marg = float(np.sqrt(np.mean((y[1:] - np.mean(y)) ** 2)))
    assert out["rmse_os"] < rmse_marg


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        beta_ar_fit(np.linspace(0.1, 0.9, 10))
    with pytest.raises(ValueError):
        beta_ar_fit(np.concatenate([np.linspace(0.1, 0.9, 100), [1.5]]))


def test_bench_contract() -> None:
    out = bench_beta_ar()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_beta_rmse_os"] < out["synthetic_beta_rmse_marg"]
