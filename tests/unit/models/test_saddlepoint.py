"""Unit tests for quant_fund.models.saddlepoint."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gamma, norm

from quant_fund.models.saddlepoint import (
    bench_saddlepoint,
    lugannani_rice,
    saddlepoint_pdf,
)


def test_normal_tail_exact() -> None:
    # For a Gaussian CGF the Lugannani-Rice tail is exact.
    tail = float(lugannani_rice(np.array([1.5]), "normal", 0.1, 0.2, 4)[0])
    exact = float(norm.sf(1.5, 0.1, np.sqrt(0.2 / 4)))
    assert tail == pytest.approx(exact, abs=1e-10)


def test_gamma_tail_matches_exact() -> None:
    tail = float(lugannani_rice(np.array([3.0]), "gamma", 2.0, 1.0, 5)[0])
    exact = float(gamma.sf(3.0 * 5, a=2.0 * 5, scale=1.0))
    assert tail == pytest.approx(exact, rel=0.02)


def test_pdf_integrates_to_one() -> None:
    grid = np.linspace(0.1, 3.0, 400)
    p = saddlepoint_pdf(grid, "gamma", 1.5, 0.8, 3)
    assert np.all(p >= 0.0)
    mass = float(np.trapezoid(p, grid))
    assert 0.8 < mass < 1.05


def test_pdf_mode_matches_exact_gamma_mean() -> None:
    # Mean of n Gamma(k1, rate k2) is Gamma(n k1, rate n k2);
    # its mode is (n k1 - 1)/(n k2) = 0.7 here.
    grid = np.linspace(0.1, 4.0, 800)
    p = saddlepoint_pdf(grid, "gamma", 1.2, 1.0, 2)
    mode = grid[int(np.argmax(p))]
    assert mode == pytest.approx(0.7, abs=0.05)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        lugannani_rice(np.array([np.nan]), "normal", 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        lugannani_rice(np.array([1.0]), "bogus", 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        saddlepoint_pdf(np.array([np.nan, 1.0]), "gamma", 1.0, 1.0, 2)


def test_bench_score() -> None:
    out = bench_saddlepoint()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_sp_err_gamma"] < 1e-3
