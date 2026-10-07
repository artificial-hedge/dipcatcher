"""Unit tests for quant_fund.models.log_concave."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.log_concave import (
    bench_log_concave,
    is_log_concave,
    log_concave_mle,
    synth_log_concave,
)


def test_gaussian_fit_concave() -> None:
    g, _ = synth_log_concave(seed=1)
    out = log_concave_mle(g)
    assert is_log_concave(np.asarray(out["phi"]), np.asarray(out["knots"]))


def test_mass_normalized() -> None:
    g, _ = synth_log_concave(seed=2)
    out = log_concave_mle(g)
    assert abs(float(out["mass"][0]) - 1.0) < 1e-6


def test_mode_near_zero() -> None:
    g, _ = synth_log_concave(seed=3)
    out = log_concave_mle(g)
    knots = np.asarray(out["knots"])
    phi = np.asarray(out["phi"])
    assert abs(float(knots[int(np.argmax(phi))])) < 0.8


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        log_concave_mle(np.ones(10))
    with pytest.raises(ValueError):
        log_concave_mle(np.full(200, 3.0))


def test_bench_contract() -> None:
    out = bench_log_concave()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_lc_concave"] == 1.0
