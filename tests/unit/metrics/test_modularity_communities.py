"""Tests for metrics/modularity_communities.py (wave 26)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.modularity_communities import (
    adjusted_rand,
    bench_modularity_communities,
    correlation_network,
    louvain,
    modularity,
    nmi,
    planted_partition,
)


class TestModularity:
    def test_one_community_zero(self):
        w, _ = planted_partition(20, 2, 0.5, 0.1, seed=0)
        q = modularity(w, np.zeros(20, dtype=np.int64))
        assert abs(q) < 1e-9

    def test_true_partition_positive(self):
        w, truth = planted_partition(40, 4, 0.6, 0.05, seed=1)
        q = modularity(w, truth)
        assert q > 0.2

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            modularity(np.ones((3, 4)), np.zeros(3, dtype=int))
        with pytest.raises(ValueError):
            modularity(np.ones((3, 3)), np.zeros(3, dtype=int))  # non-hollow
        with pytest.raises(ValueError):
            modularity(-np.ones((4, 4)) * 0.1, np.zeros(4, dtype=int))  # negative
        w = np.zeros((5, 5))
        with pytest.raises(ValueError):
            modularity(w, np.zeros(5, dtype=int))  # empty graph


class TestLouvain:
    def test_recovers_planted(self):
        w, truth = planted_partition(60, 4, 0.5, 0.05, seed=3)
        labels, q, nc = louvain(w, seed=0)
        assert adjusted_rand(truth, labels) > 0.6
        assert nc >= 2

    def test_monotone_over_random(self):
        w, truth = planted_partition(50, 3, 0.5, 0.08, seed=4)
        _, q_louvain, _ = louvain(w, seed=0)
        rng = np.random.default_rng(0)
        q_rand = max(modularity(w, rng.integers(0, 3, 50)) for _ in range(20))
        assert q_louvain >= q_rand

    def test_determinism(self):
        w, _ = planted_partition(30, 3, 0.4, 0.1, seed=5)
        l1, q1, _ = louvain(w, seed=7)
        l2, q2, _ = louvain(w, seed=7)
        assert np.array_equal(l1, l2) and q1 == q2

    def test_two_cliques(self):
        w = np.zeros((8, 8))
        w[:4, :4] = 1.0
        w[4:, 4:] = 1.0
        np.fill_diagonal(w, 0.0)
        labels, q, nc = louvain(w, seed=0)
        assert nc == 2
        assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            louvain(np.ones((2, 2)))  # non-hollow
        with pytest.raises(ValueError):
            louvain(np.zeros((1, 1)))


class TestCorrelationNetwork:
    def test_clip_and_hollow(self):
        c = np.array([[1.0, 0.8, -0.3], [0.8, 1.0, 0.2], [-0.3, 0.2, 1.0]])
        w = correlation_network(c)
        assert w[0, 2] == 0.0 and w[0, 1] == 0.8
        assert np.diag(w).max() == 0.0

    def test_threshold(self):
        c = np.array([[1.0, 0.8, 0.4], [0.8, 1.0, 0.1], [0.4, 0.1, 1.0]])
        w = correlation_network(c, threshold=0.5)
        assert w[0, 1] > 0 and w[0, 2] == 0.0 and w[1, 2] == 0.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            correlation_network(np.array([[1.0, 0.5], [0.2, 1.0]]))  # asymmetric
        with pytest.raises(ValueError):
            correlation_network(np.eye(3), threshold=2.0)


class TestPlanted:
    def test_properties(self):
        w, labels = planted_partition(30, 3, 0.5, 0.05, seed=0)
        assert w.shape == (30, 30) and labels.shape == (30,)
        assert np.allclose(w, w.T)
        assert np.diag(w).max() == 0.0
        assert len(np.unique(labels)) == 3

    def test_invalid(self):
        with pytest.raises(ValueError):
            planted_partition(10, 2, 0.1, 0.5)  # p_out > p_in
        with pytest.raises(ValueError):
            planted_partition(4, 8, 0.5, 0.1)


class TestScores:
    def test_ari_perfect(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand(a, a) == pytest.approx(1.0)

    def test_ari_random_low(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 3, 200)
        b = rng.integers(0, 3, 200)
        assert adjusted_rand(a, b) < 0.1

    def test_nmi_perfect(self):
        a = np.array([0, 0, 1, 1])
        assert nmi(a, a) == pytest.approx(1.0)

    def test_nmi_independent(self):
        # orthogonal balanced splits carry no mutual information
        a2 = np.array([0] * 50 + [1] * 50)
        b2 = np.array([0, 1] * 50)
        assert nmi(a2, b2) < 0.1


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_modularity_communities(seed=17)
        assert blob
        assert all(np.isfinite(v) for v in blob.values())
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in blob:
            assert forbidden.isdisjoint(k.lower().split("_"))

    def test_bench_determinism(self):
        assert bench_modularity_communities(seed=19) == bench_modularity_communities(seed=19)
