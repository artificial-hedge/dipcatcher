"""Unit tests for quant_fund.models.higham_corr."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.higham_corr import (
    bench_higham,
    higham_nearcorr,
    newton_nearcorr,
)


def _broken() -> np.ndarray:
    rng = np.random.default_rng(11)
    q = rng.standard_normal((6, 6))
    c = np.corrcoef(q)
    m = 0.6 * c + 0.4 * rng.uniform(-0.5, 0.5, (6, 6))
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)
    m[:3, 3:] *= 3.0
    m[3:, :3] *= 3.0
    np.clip(m, -1.5, 1.5, out=m)
    np.fill_diagonal(m, 1.0)
    return m


def test_output_is_valid_correlation() -> None:
    a = _broken()
    for fn in (higham_nearcorr, newton_nearcorr):
        x = fn(a)
        assert np.allclose(np.diag(x), 1.0, atol=1e-8)
        assert np.allclose(x, x.T, atol=1e-10)
        assert np.linalg.eigvalsh(x).min() > -1e-7


def test_methods_agree_on_minimum() -> None:
    a = _broken()
    d_h = float(np.linalg.norm(higham_nearcorr(a) - a))
    d_q = float(np.linalg.norm(newton_nearcorr(a) - a))
    assert abs(d_h - d_q) < 0.02 * (1.0 + d_h)


def test_identity_preserved() -> None:
    eye = np.eye(5)
    assert np.linalg.norm(higham_nearcorr(eye) - eye) < 1e-8
    assert np.linalg.norm(newton_nearcorr(eye) - eye) < 1e-8


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        higham_nearcorr(np.ones((3, 4)))
    with pytest.raises(ValueError):
        higham_nearcorr(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        newton_nearcorr(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_bench_contract() -> None:
    out = bench_higham()
    assert out["score"] == 1.0
    assert out["synthetic_nc_dist_higham"] <= out["synthetic_nc_dist_altproj"] + 1e-9
