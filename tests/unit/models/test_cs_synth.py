"""Unit tests for quant_fund.models._cs_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd


def test_sem_data_deterministic_and_dag() -> None:
    X1, B1 = sem_data(3, n=200)
    X2, B2 = sem_data(3, n=200)
    assert np.array_equal(X1, X2) and np.array_equal(B1, B2)
    # B_true must be acyclic: there exists a topological order —
    # verify no self-loops and at least one source/sink structure
    assert np.all(np.diag(B1) == 0)
    assert (np.abs(B1) > 0).sum() == 7


def test_sem_edges_hard_cap() -> None:
    with pytest.raises(ValueError, match="acyclic pairs"):
        sem_data(0, d=4, edges=99)


def test_shd_zero_for_identical() -> None:
    B = np.zeros((4, 4))
    B[0, 1] = 0.8
    B[2, 3] = -1.0
    assert shd(B, B) == 0
    # one missing edge
    Bh = B.copy()
    Bh[0, 1] = 0.0
    assert shd(B, Bh) == 1


def test_corr_baseline_edge_budget() -> None:
    X, _B = sem_data(1, n=500)
    Bb = corr_baseline(X, 7)
    assert (np.abs(Bb) > 0).sum() <= 7
    assert np.all(np.diag(Bb) == 0)
