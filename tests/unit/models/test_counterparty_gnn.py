"""Tests for counterparty_gnn — contagion GNN bench."""

from __future__ import annotations

import numpy as np

from quant_fund.models.counterparty_gnn import auc_score, bench_counterparty_gnn


def test_auc_tied_scores_midranks() -> None:
    # tied scores must share midranks: all-tied gives exactly 0.5. The old
    # code assigned arbitrary distinct ranks, so the baseline AUC (tiled
    # features -> massive ties) was order-dependent noise, not a score.
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert auc_score(y, np.full(4, 0.5)) == 0.5


def test_auc_partial_ties() -> None:
    # 0.2 pair ties (ranks 1.5 each), 0.9 takes rank 3:
    # pos ranks {1.5, 3} -> (4.5 - 3) / 2 = 0.75.
    s = np.array([0.2, 0.2, 0.9])
    y = np.array([0.0, 1.0, 1.0])
    assert auc_score(y, s) == 0.75


def test_auc_no_ties_unchanged() -> None:
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    # sorted ranks: 0.1->1, 0.35->2, 0.4->3, 0.8->4; pos ranks {2,4}
    # -> (6 - 3) / 4 = 0.75 (one inversion: neg 0.4 outranks pos 0.35)
    assert auc_score(y, s) == 0.75


def test_bench_counterparty_gnn() -> None:
    out = bench_counterparty_gnn()
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
