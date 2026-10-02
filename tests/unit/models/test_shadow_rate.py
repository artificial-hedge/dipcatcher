"""Unit tests for quant_fund.models.shadow_rate."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.shadow_rate import (
    _affine_loadings,
    bench_shadow_rate,
    ekf_shadow,
)


def test_loadings_shape() -> None:
    f = np.array([[0.95]])
    q = np.array([[0.01]])
    delta = np.array([0.01, 1.0])
    a, b = _affine_loadings(f, q, delta, 0.0, 6)
    assert a.shape == (7,) and b.shape == (7, 1)
    assert b[0, 0] == 0.0 and b[1, 0] == pytest.approx(1.0)


def test_loadings_monotone() -> None:
    f = np.array([[0.9]])
    q = np.array([[0.01]])
    delta = np.array([0.01, 0.5])
    a, b = _affine_loadings(f, q, delta, 0.0, 10)
    assert np.all(np.diff(b[:, 0]) >= 0.0)


def test_ekf_recovers_state() -> None:
    rng = np.random.default_rng(3)
    t_n = 200
    x = np.zeros(t_n)
    for i in range(1, t_n):
        x[i] = 0.95 * x[i - 1] + 0.1 * rng.standard_normal()
    f = np.array([[0.95]])
    q = np.array([[0.01]])
    delta = np.array([0.005, 1.0])
    r_obs = np.maximum(delta[0] + x, 0.0) + 0.001 * rng.standard_normal(t_n)
    y2 = 0.01 + 0.8 * x + 0.001 * rng.standard_normal(t_n)
    y = np.column_stack([r_obs, y2])
    h = np.array([[0.01, 0.8]])
    r_mat = np.diag([1e-6, 1e-6])
    xs, shadow = ekf_shadow(y, f, q, h, r_mat, delta, 0.0)
    corr = np.corrcoef(xs[:, 0], x)[0, 1]
    assert corr > 0.9


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        ekf_shadow(
            np.zeros((3, 2)),
            np.eye(1),
            np.eye(1),
            np.array([[0.0, 0.5]]),
            np.eye(2),
            np.array([0.0, 1.0]),
            0.0,
        )


def test_bench_score() -> None:
    out = bench_shadow_rate()
    assert out["score"] == 1.0
    assert out["synthetic_sr_state_corr"] > 0.9
