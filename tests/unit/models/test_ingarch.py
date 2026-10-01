"""Unit tests for quant_fund.models.ingarch."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ingarch import (
    bench_ingarch,
    ingarch_fit,
    synth_ingarch,
)


def test_recovers_persistence() -> None:
    y = synth_ingarch(seed=1)
    out = ingarch_fit(y)
    assert 0.5 < out["persistence"] < 0.95


def test_uncond_mean_matches_sample() -> None:
    y = synth_ingarch(seed=2)
    out = ingarch_fit(y)
    assert abs(out["uncond_mean"] - float(np.mean(y))) / float(np.mean(y)) < 0.5


def test_onestep_beats_marginal() -> None:
    y = synth_ingarch(seed=3)
    out = ingarch_fit(y)
    rmse_marg = float(np.sqrt(np.mean((y[1:] - np.mean(y)) ** 2)))
    assert out["rmse_os"] < rmse_marg


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        ingarch_fit(np.ones(20))
    with pytest.raises(ValueError):
        ingarch_fit(-np.ones(100))


def test_bench_contract() -> None:
    out = bench_ingarch()
    assert out["score"] == 1.0
    assert out["synthetic_ingarch_rmse_os"] < out["synthetic_ingarch_rmse_marg"]
