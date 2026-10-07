"""Tests for research/energy_score.py — multivariate calibration."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.energy_score import (
    ENERGY_SCORE_SCHEMA,
    energy_score,
    energy_score_bench,
    gaussian_copula_samples,
    variogram_score,
)


def test_energy_score_perfect_prediction() -> None:
    # samples concentrated at y → ES → E‖X−y‖ − ½E‖X−X'‖ ≈ 0
    y = np.array([1.0, -1.0])
    s = np.tile(y, (200, 1))
    assert energy_score(y, s) == pytest.approx(0.0, abs=1e-12)


def test_energy_score_known_value() -> None:
    # X ≡ 0, y = e1 → term1 = 1, term2 = 0 → ES = 1
    s = np.zeros((50, 2))
    y = np.array([1.0, 0.0])
    assert energy_score(y, s) == pytest.approx(1.0)


def test_variogram_score_perfect() -> None:
    y = np.array([0.0, 1.0, 2.0])
    s = np.tile(y, (30, 1))
    assert variogram_score(y, s) == pytest.approx(0.0, abs=1e-12)


def test_variogram_score_known_value() -> None:
    # d=2, p=1: y=(0,1) → |y1-y2|=1; samples all-zero → E|x1-x2|=0
    y = np.array([0.0, 1.0])
    s = np.zeros((4, 2))
    # VS = Σ_ij (|yi-yj| - E|xi-xj|)^2 = 1 + 1 = 2 on the two off-diagonals
    assert variogram_score(y, s, p=1.0) == pytest.approx(2.0)


def test_copula_preserves_marginals() -> None:
    rng = np.random.default_rng(0)
    marginals = [np.sort(rng.standard_normal(400)) + j for j in range(3)]
    out = gaussian_copula_samples(marginals, 0.5, rng)
    assert out.shape == (400, 3)
    for j in range(3):
        assert np.allclose(np.sort(out[:, j]), marginals[j])


def test_copula_induces_correlation() -> None:
    rng = np.random.default_rng(1)
    marginals = [rng.standard_normal(2000) for _ in range(4)]
    out = gaussian_copula_samples(marginals, 0.7, rng)
    c = np.corrcoef(out.T)
    assert np.all(np.diag(c) == pytest.approx(1.0))
    off = c[~np.eye(4, dtype=bool)]
    assert off.mean() == pytest.approx(0.7, abs=0.08)


def test_copula_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="same length|share length"):
        gaussian_copula_samples([np.zeros(10), np.zeros(12)], 0.5, rng)
    with pytest.raises(ValueError, match="rho"):
        gaussian_copula_samples([np.zeros(10)] * 3, 2.0, rng)


def test_bench_smoke_and_schema() -> None:
    out = energy_score_bench(n_reps=5, m=64, d=4)
    assert out["schema"] == ENERGY_SCORE_SCHEMA
    for arm in ("correct", "misspecified", "independent"):
        assert "energy_score" in out["arms"][arm]
        assert "variogram_score" in out["arms"][arm]
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64


def test_empty_sample_set_fails_closed() -> None:
    """m = 0 means over an empty set silently yields nan — the score must
    fail closed instead."""
    y = np.zeros(3)
    empty = np.empty((0, 3))
    with pytest.raises(ValueError):
        energy_score(y, empty)
    with pytest.raises(ValueError):
        variogram_score(y, empty)
