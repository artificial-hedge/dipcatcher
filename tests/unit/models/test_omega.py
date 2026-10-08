"""Tests for omega — McDonald's composite reliability."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.omega import bench_omega, omega_total


def test_congeneric_high_omega():
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(400)
    lam = np.linspace(0.4, 1.0, 6)
    x = theta[:, None] * lam[None, :] + rng.standard_normal((400, 6)) * np.sqrt(1 - lam**2)[None, :]
    out = omega_total(x)
    assert out["omega"] > 0.75


def test_noise_low_omega():
    rng = np.random.default_rng(1)
    out = omega_total(rng.standard_normal((400, 6)))
    assert out["omega"] < 0.4


def test_resid_rms_reported():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)[:, None] + 0.3 * rng.standard_normal((200, 5))
    out = omega_total(x)
    assert np.isfinite(out["resid_rms"])
    assert out["resid_rms"] >= 0


def test_fail_closed_two_items():
    with pytest.raises(ValueError):
        omega_total(np.ones((50, 2)))


def test_bench():
    out = bench_omega()
    assert out["synthetic_score"] == 1.0
