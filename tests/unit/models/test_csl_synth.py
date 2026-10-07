"""Unit tests for quant_fund.models._csl_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._csl_synth import (
    corr_baseline_order,
    lingam_sem,
    order_err,
    skeleton,
    skeleton_f1,
)


def test_lingam_sem_deterministic_and_dag() -> None:
    X1, B1, o1 = lingam_sem(3, n=200)
    X2, B2, o2 = lingam_sem(3, n=200)
    assert np.array_equal(X1, X2) and np.array_equal(B1, B2) and o1 == o2
    assert sorted(o1) == list(range(6))
    assert (np.abs(B1) > 0).sum() == 7


def test_lingam_rejects_too_many_edges() -> None:
    # sibling _cs_synth.sem_data raises on this; csl silently clamped before
    with pytest.raises(ValueError, match="acyclic pairs"):
        lingam_sem(0, d=4, edges=99)


def test_lingam_rejects_unknown_noise() -> None:
    with pytest.raises(ValueError, match="noise family"):
        lingam_sem(0, n=50, noise="bogus")


def test_lingam_noise_families_produce_data() -> None:
    for noise in ("uniform", "laplace", "student", "gauss"):
        X, _B, _o = lingam_sem(1, n=100, noise=noise)
        assert X.shape == (100, 6)
        assert np.isfinite(X).all()


def test_skeleton_f1_bounds_and_perfect() -> None:
    B = np.zeros((4, 4))
    B[0, 1] = 0.8
    assert skeleton(B)[0, 1] == 1.0 and skeleton(B)[1, 0] == 1.0
    assert skeleton_f1(B, B) == 1.0
    Bh = np.zeros((4, 4))
    Bh[1, 0] = 0.5  # reversed edge: same skeleton
    assert skeleton_f1(B, Bh) == 1.0
    empty = np.zeros((4, 4))
    assert skeleton_f1(B, empty) == 0.0


def test_order_err_extremes() -> None:
    o = [0, 1, 2, 3]
    assert order_err(o, o) == 0.0
    assert order_err(o, o[::-1]) == 1.0


def test_corr_baseline_order_is_permutation() -> None:
    X, _B, _o = lingam_sem(2, n=300)
    ob = corr_baseline_order(X)
    assert sorted(ob) == list(range(6))
