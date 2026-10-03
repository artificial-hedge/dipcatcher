"""Marginal-homogeneity tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.marginal_homogeneity import (
    bench_marginal_homogeneity,
    bhapkar,
    bowker_symmetry,
    mcnemar,
    stuart_maxwell,
)


def test_mcnemar_strong_discordance():
    tab = np.array([[50, 30], [5, 45]])
    out = mcnemar(tab)
    assert out["p"] < 0.01
    assert out["b"] == 30.0
    assert out["c"] == 5.0


def test_mcnemar_balanced():
    tab = np.array([[60, 20], [20, 60]])
    out = mcnemar(tab)
    assert out["p"] > 0.3


def test_bowker_symmetric_table():
    tab = np.array([[30, 10, 8], [10, 25, 9], [8, 9, 20]])
    out = bowker_symmetry(tab)
    assert out["p"] > 0.5


def test_bowker_asymmetric():
    tab = np.array([[30, 40, 5], [5, 25, 8], [3, 4, 20]])
    out = bowker_symmetry(tab)
    assert out["p"] < 0.05


def test_stuart_maxwell_detects_shift():
    tab = np.array([[30, 30, 5], [10, 35, 5], [5, 10, 20]])
    out = stuart_maxwell(tab)
    assert out["p"] < 0.05


def test_bhapkar_agrees_with_sm():
    tab = np.array([[30, 30, 5], [10, 35, 5], [5, 10, 20]])
    sm = stuart_maxwell(tab)
    bh = bhapkar(tab)
    assert abs(sm["p"] - bh["p"]) < 1e-6


def test_input_validation():
    with pytest.raises(ValueError):
        mcnemar(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    with pytest.raises(ValueError):
        bowker_symmetry(np.array([[1, 2], [3]]))


def test_bench_passes():
    out = bench_marginal_homogeneity()
    assert out["synthetic_mcnemar_p"] < 0.1
    assert out["synthetic_bowker_p"] > 0.3
    assert out["synthetic_sm_p"] < 0.05
    assert out["synthetic_bhapkar_p"] < 0.05
    assert out["synthetic_score"] == 1.0
