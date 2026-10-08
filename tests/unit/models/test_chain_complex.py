"""Adversarial probes for chain_complex homology benches."""

from __future__ import annotations

import numpy as np

import quant_fund.models.chain_complex as cc
from quant_fund.models.chain_complex import bench_chain_complex, homology_dim


def test_h0_check_uses_real_d0(monkeypatch) -> None:
    # The bench must verify H0 = dim C0 - rank(d1) = 1 for a connected
    # triangle through a REAL (0,3) boundary map — the old code evaluated
    # homology_dim(np.eye(0), d1) == -2 and then vacuously passed with
    # `or True`.
    calls: list[tuple[tuple[int, ...], int]] = []
    real = homology_dim

    def spy(b_in: np.ndarray, b_out: np.ndarray) -> int:
        calls.append((tuple(b_in.shape), int(real(b_in, b_out))))
        return calls[-1][1]

    monkeypatch.setattr(cc, "homology_dim", spy)
    assert cc._bench_chain_complex() == 1.0
    assert any(shape == (0, 3) and result == 1 for shape, result in calls)


def test_homology_dim_triangle() -> None:
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=float)
    assert homology_dim(np.zeros((0, 3)), d1) == 1  # connected H0
    assert homology_dim(d1, np.zeros((3, 0))) == 1  # open triangle H1
    d2f = np.array([[1], [1], [-1]], dtype=float)
    assert homology_dim(d1, d2f) == 0  # filled triangle kills H1


def test_bench_passes() -> None:
    assert bench_chain_complex()["synthetic_chain_complex"] == 1.0
