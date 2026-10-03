"""Correspondence analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.correspondence_analysis import (
    bench_ca,
    correspondence_analysis,
    expected_independence,
)


def test_inertia_matches_chi2():
    rng = np.random.default_rng(0)
    table = rng.integers(10, 100, (5, 4)).astype(float)
    ca = correspondence_analysis(table, n_dim=2)
    exp = expected_independence(table)
    chi2 = float(((table - exp) ** 2 / exp).sum())
    assert abs(ca["total_inertia"] * table.sum() - chi2) / chi2 < 1e-9


def test_independent_table_small_inertia():
    rng = np.random.default_rng(1)
    r = rng.uniform(1, 2, (5, 1))
    c = rng.uniform(1, 2, (1, 4))
    table = rng.poisson(500 * r @ c).astype(float) + 1000 * r @ c
    ca = correspondence_analysis(table, n_dim=1)
    dep = rng.multinomial(
        5000,
        np.exp(2 * np.outer(np.arange(5) - 2, np.arange(4) - 1.5)).ravel()
        / np.exp(2 * np.outer(np.arange(5) - 2, np.arange(4) - 1.5)).sum(),
    ).reshape(5, 4)
    ca_dep = correspondence_analysis(dep, n_dim=1)
    assert ca_dep["total_inertia"] > 10 * ca["total_inertia"]


def test_strong_axis_dominates():
    rng = np.random.default_rng(2)
    axis = np.outer(np.linspace(-1, 1, 6), np.linspace(-1, 1, 5))
    logits = 4.0 * axis
    probs = np.exp(logits) / np.exp(logits).sum()
    table = rng.multinomial(4000, probs.ravel()).reshape(6, 5)
    ca = correspondence_analysis(table, n_dim=2)
    assert np.asarray(ca["inertia_share"])[0] > 0.5


def test_coord_shapes():
    rng = np.random.default_rng(3)
    table = rng.integers(5, 50, (6, 5)).astype(float)
    ca = correspondence_analysis(table, n_dim=2)
    assert np.asarray(ca["row_coords"]).shape == (6, 2)
    assert np.asarray(ca["col_coords"]).shape == (5, 2)


def test_input_validation():
    with pytest.raises(ValueError):
        correspondence_analysis(np.ones((2, 2)))
    with pytest.raises(ValueError):
        correspondence_analysis(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        expected_independence(np.zeros((3, 3)))


def test_bench_passes():
    out = bench_ca()
    assert out["synthetic_first_inertia_share"] > 0.5
    assert out["synthetic_chi2_rel_err"] < 1e-6
    assert out["synthetic_score"] == 1.0
