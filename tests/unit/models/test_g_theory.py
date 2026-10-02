"""Tests for g_theory — generalizability coefficients."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.g_theory import bench_g_theory, d_study, g_study


def test_person_variance_dominant():
    rng = np.random.default_rng(0)
    person = rng.standard_normal(60) * 2
    x = person[:, None] + 0.4 * rng.standard_normal((60, 3))
    out = g_study(x)
    assert out["g_coefficient"] > 0.85
    assert out["sigma_person"] > out["sigma_rater"]


def test_noise_low_g():
    rng = np.random.default_rng(1)
    out = g_study(rng.standard_normal((60, 3)))
    assert out["g_coefficient"] < 0.4


def test_d_study_improves():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(60)[:, None] + 0.5 * rng.standard_normal((60, 3))
    g = g_study(x)
    d = d_study(x, n_raters=10)
    assert d["g_coefficient"] >= g["g_coefficient"]


def test_phi_below_g_when_rater_effect():
    rng = np.random.default_rng(3)
    x = (
        rng.standard_normal(50)[:, None]
        + np.array([0.0, 0.8, -0.4])[None, :]
        + 0.3 * rng.standard_normal((50, 3))
    )
    out = g_study(x)
    assert out["phi_coefficient"] <= out["g_coefficient"] + 1e-9


def test_fail_closed_shape():
    with pytest.raises(ValueError):
        g_study(np.ones((30, 1)))


def test_bench():
    out = bench_g_theory()
    assert out["score"] == 1.0
