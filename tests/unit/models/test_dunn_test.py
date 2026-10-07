"""Tests for dunn_test — pairwise rank post-hoc."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dunn_test import bench_dunn_test, dunn_test


def test_shifted_pairs_rejected():
    rng = np.random.default_rng(0)
    out = dunn_test(
        rng.standard_normal(60), rng.standard_normal(60) + 1.4, rng.standard_normal(60) + 1.4
    )
    p_adj = np.asarray(out["p_adj"])
    pairs = np.asarray(out["pairs"], dtype=int)
    p_ab = float(p_adj[np.where((pairs == [0, 1]).all(axis=1))[0][0]])
    assert p_ab < 0.05


def test_null_pair_held():
    rng = np.random.default_rng(1)
    out = dunn_test(
        rng.standard_normal(60) + 1.4, rng.standard_normal(60) + 1.4, rng.standard_normal(60)
    )
    p_adj = np.asarray(out["p_adj"])
    pairs = np.asarray(out["pairs"], dtype=int)
    p_bc = float(p_adj[np.where((pairs == [0, 1]).all(axis=1))[0][0]])
    assert p_bc > 0.05


def test_bonferroni_adjust():
    rng = np.random.default_rng(2)
    out = dunn_test(rng.standard_normal(40), rng.standard_normal(40) + 1.2, adjust="bonferroni")
    p_adj = np.asarray(out["p_adj"])
    p_raw = np.asarray(out["p_raw"])
    assert (p_adj >= p_raw).all()


def test_fail_closed_adjust():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        dunn_test(rng.standard_normal(20), rng.standard_normal(20), adjust="tukey")


def test_bench():
    out = bench_dunn_test()
    assert out["synthetic_score"] == 1.0
