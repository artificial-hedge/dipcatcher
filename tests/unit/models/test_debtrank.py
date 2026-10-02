"""Unit tests for quant_fund.models.debtrank."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.debtrank import (
    bench_debtrank,
    debtrank,
    debtrank_profile,
    in_strength,
)


def test_debtrank_shape_validation() -> None:
    w = np.zeros((3, 3))
    sizes = np.ones(3)
    with pytest.raises(ValueError):
        debtrank(w, sizes, np.ones(4))
    with pytest.raises(ValueError):
        debtrank(w, np.ones(4), np.ones(3))
    with pytest.raises(ValueError):
        debtrank(w, sizes, np.array([0.0, 1.5, 0.0]))


def test_no_edges_no_propagation() -> None:
    w = np.zeros((3, 3))
    sizes = np.array([1.0, 1.0, 1.0])
    shock = np.array([1.0, 0.0, 0.0])
    r, h = debtrank(w, sizes, shock)
    assert r == pytest.approx(0.0)
    np.testing.assert_array_equal(h, shock)


def test_propagation_adds_distress() -> None:
    w = np.zeros((2, 2))
    w[0, 1] = 0.5  # node 0 exposed to node 1
    sizes = np.ones(2)
    shock = np.array([0.0, 1.0])
    r, h = debtrank(w, sizes, shock)
    assert h[0] == pytest.approx(0.5)
    assert r == pytest.approx(0.25)


def test_distress_bounded_by_one() -> None:
    w = np.array([[0.0, 0.9], [0.9, 0.0]])
    sizes = np.ones(2)
    _, h = debtrank(w, sizes, np.ones(2) * 0.6)
    assert np.all(h <= 1.0)


def test_profile_matches_single_shocks() -> None:
    w = np.zeros((3, 3))
    w[1, 0] = 0.5
    w[2, 0] = 0.5
    sizes = np.array([1.0, 2.0, 2.0])
    prof = debtrank_profile(w, sizes)
    for i in range(3):
        shock = np.zeros(3)
        shock[i] = 1.0
        r, _ = debtrank(w, sizes, shock)
        assert prof[i] == pytest.approx(r)


def test_in_strength_column_sums() -> None:
    w = np.array([[0.0, 0.2], [0.5, 0.0]])
    ist = in_strength(w)
    np.testing.assert_allclose(ist, [0.5, 0.2])


def test_bench_debtrank_score() -> None:
    out = bench_debtrank()
    assert out["score"] == pytest.approx(1.0)
    # hub (rank 0) outranks the in-strength leader (rank 7).
    assert out["synthetic_dr_hub_rank"] == pytest.approx(0.0)
    assert out["synthetic_ist_leader_rank"] == pytest.approx(7.0)
