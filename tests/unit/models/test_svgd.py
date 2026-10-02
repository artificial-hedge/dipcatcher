"""Unit tests for quant_fund.models.svgd."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.svgd import _median_bandwidth, bench_svgd, svgd


def _gauss_target(mu: float = 0.0, sd: float = 1.0):
    def score(x: np.ndarray) -> np.ndarray:
        return -(x - mu) / (sd * sd)

    return score


def test_median_bandwidth_positive() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 2))
    h = _median_bandwidth(x)
    assert h > 0.0


def test_svgd_moves_particles_toward_mode() -> None:
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((200, 1)) * 4.0 - 6.0
    xf = svgd(x0, _gauss_target(0.0, 1.0), n_iter=500, eta=1.0)
    assert abs(xf.mean()) < abs(x0.mean())


def test_svgd_deterministic() -> None:
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((80, 2))
    a = svgd(x0.copy(), _gauss_target(0.5, 1.2), n_iter=50, eta=0.5)
    b = svgd(x0.copy(), _gauss_target(0.5, 1.2), n_iter=50, eta=0.5)
    np.testing.assert_array_equal(a, b)


def test_svgd_rejects_bad_config() -> None:
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((20, 1))
    with pytest.raises(ValueError):
        svgd(x0[:1], _gauss_target(), n_iter=10)  # needs >=2 particles
    with pytest.raises(ValueError):
        svgd(x0, _gauss_target(), n_iter=0)
    with pytest.raises(ValueError):
        svgd(x0, _gauss_target(), n_iter=10, eta=0.0)


def test_svgd_gaussian_moments() -> None:
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((300, 1)) * 3.0
    mu, sd = 1.0, 0.7
    xf = svgd(x0, _gauss_target(mu, sd), n_iter=500, eta=1.0)
    assert xf.mean() == pytest.approx(mu, abs=0.15)
    assert xf.std() == pytest.approx(sd, rel=0.4)


def test_bench_svgd_score() -> None:
    out = bench_svgd()
    assert out["score"] == pytest.approx(1.0)
    assert 0.35 < out["synthetic_svgd_right_frac"] < 0.65
