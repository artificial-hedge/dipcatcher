"""Tests for design_effects — Kish DEFF diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.design_effects import (
    bench_design_effects,
    design_effect,
    trimmed_deff_curve,
)


def test_equal_weights_deff_one():
    out = design_effect(np.ones(100))
    assert out["deff"] == pytest.approx(1.0)
    assert out["n_eff"] == pytest.approx(100.0)


def test_variable_weights_deff_gt_one():
    w = np.exp(0.9 * np.random.default_rng(0).standard_normal(300))
    out = design_effect(w)
    assert out["deff"] > 1.0
    assert out["n_eff"] < 300


def test_deff_closed_form():
    w = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    out = design_effect(w)
    # deff = n sum w^2 / (sum w)^2 = 8*20/144
    assert out["deff"] == pytest.approx(8 * 20 / 144)


def test_trimming_reduces_deff():
    w = np.exp(1.2 * np.random.default_rng(1).standard_normal(400))
    curve = trimmed_deff_curve(w)
    assert curve[0] < design_effect(w)["deff"]


def test_fail_closed_nonpositive():
    with pytest.raises(ValueError):
        design_effect(np.array([1.0, -1.0, 1.0, 1.0, 1.0]))


def test_bench():
    out = bench_design_effects()
    assert out["synthetic_score"] == 1.0
