"""Tests for metrics/causal_discovery.py — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.causal_discovery import (
    apply_meek_rules,
    bench_causal_discovery,
    ges_fit,
    lingam_pairwise,
    orient_vstructures,
    partial_corr,
    pc_fit,
    pc_skeleton,
    synth_dag,
)


def _chain_data(n: int = 600, seed: int = 0) -> np.ndarray:
    """X0 -> X1 -> X2 linear-Gaussian chain."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + 0.3 * rng.standard_normal(n)
    x2 = 0.8 * x1 + 0.3 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2])


def _collider_data(n: int = 600, seed: int = 1) -> np.ndarray:
    """X0 -> X2 <- X1 collider."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = 0.7 * x0 + 0.7 * x1 + 0.2 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2])


class TestPartialCorr:
    def test_marginal_independent(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.standard_normal((500, 3))
        r, p = partial_corr(0, 1, (), data)
        assert abs(r) < 0.15
        assert p > 0.05

    def test_chain_xz_marginal_dependent_conditional_independent(self) -> None:
        data = _chain_data()
        _r0, p_marg = partial_corr(0, 2, (), data)
        _r1, p_cond = partial_corr(0, 2, (1,), data)
        assert p_marg < 0.01
        assert p_cond > 0.05

    def test_invalid(self) -> None:
        data = np.zeros((50, 3))
        with pytest.raises(ValueError):
            partial_corr(0, 0, (), data)
        with pytest.raises(ValueError):
            partial_corr(0, 9, (), data)
        with pytest.raises(ValueError):
            partial_corr(0, 1, (0,), data)


class TestSkeleton:
    def test_chain_removes_marginal_edge(self) -> None:
        data = _chain_data()
        adj, sepsets, _ = pc_skeleton(data, alpha=0.05)
        assert adj[0, 1] and adj[1, 2]
        assert not adj[0, 2]
        assert 1 in sepsets.get((0, 2), ())

    def test_collider_keeps_spouses(self) -> None:
        data = _collider_data()
        adj, _, _ = pc_skeleton(data, alpha=0.05)
        assert adj[0, 2] and adj[1, 2]
        assert not adj[0, 1]

    def test_stable_order_independent(self) -> None:
        data = _chain_data()
        a1, s1, _ = pc_skeleton(data, alpha=0.05)
        a2, s2, _ = pc_skeleton(data, alpha=0.05)
        assert np.array_equal(a1, a2)
        assert s1 == s2

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            pc_skeleton(np.zeros((50, 3)), alpha=1.5)


class TestOrientation:
    def test_collider_oriented(self) -> None:
        data = _collider_data()
        adj, sepsets, _ = pc_skeleton(data, alpha=0.05)
        cp = orient_vstructures(adj, sepsets)
        assert cp[0, 2] == 1.0 and cp[1, 2] == 1.0

    def test_chain_not_collider(self) -> None:
        data = _chain_data()
        adj, sepsets, _ = pc_skeleton(data, alpha=0.05)
        cp = orient_vstructures(adj, sepsets)
        # 0-1-2 unshielded triple but 1 IS in sepset(0,2) → no collider
        assert not (cp[0, 1] == 1.0 and cp[2, 1] == 1.0)

    def test_meek_rule1(self) -> None:
        # c->a-b, c not adj b → a->b
        cp = np.zeros((3, 3))
        cp[0, 1] = 1.0  # 0 -> 1
        cp[1, 2] = cp[2, 1] = -1.0  # 1 — 2 undirected
        cp[0, 2] = cp[2, 0] = 0.0  # 0 not adjacent 2
        out = apply_meek_rules(cp)
        assert out[1, 2] == 1.0

    def test_meek_converges(self) -> None:
        cp = -np.ones((4, 4)) + 2 * np.eye(4)
        np.fill_diagonal(cp, 0.0)
        out = apply_meek_rules(cp, max_rounds=10)
        assert out.shape == (4, 4)


class TestPipelines:
    def test_pc_fit_shapes(self) -> None:
        data = _chain_data()
        res = pc_fit(data)
        assert res.skeleton.shape == (3, 3)
        assert res.cpdag.shape == (3, 3)
        assert res.n_tests > 0

    def test_ges_finds_edges(self) -> None:
        data = _chain_data()
        adj = ges_fit(data)
        assert adj.sum() >= 2  # should find the two chain edges

    def test_ges_acyclic(self) -> None:
        data = _collider_data()
        adj = ges_fit(data)
        # check no directed cycle via matrix powers
        reach = adj.astype(int)
        for _ in range(adj.shape[0]):
            reach = reach @ adj.astype(int) + reach
        assert np.all(np.diag(reach) == 0)

    def test_lingam_direction(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.uniform(-1, 1, 400)
        y = 0.9 * x + rng.uniform(-1, 1, 400)
        r = lingam_pairwise(x, y)
        assert r["direction"] == 1.0
        r_rev = lingam_pairwise(y, x)
        assert r_rev["direction"] == -1.0

    def test_lingam_invalid(self) -> None:
        with pytest.raises(ValueError):
            lingam_pairwise(np.ones(10), np.ones(10))


class TestSynth:
    def test_dag_acyclic(self) -> None:
        x, adj = synth_dag(n=200, p=6, seed=5)
        assert x.shape == (200, 6)
        # adj upper triangular => acyclic
        assert np.all(~np.tril(adj))

    def test_deterministic(self) -> None:
        x1, a1 = synth_dag(n=100, p=4, seed=3)
        x2, a2 = synth_dag(n=100, p=4, seed=3)
        assert np.array_equal(x1, x2) and np.array_equal(a1, a2)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            synth_dag(n=10)
        with pytest.raises(ValueError):
            synth_dag(n=100, edge_prob=0.0)


class TestBench:
    def test_bench_keys_finite(self) -> None:
        blob = bench_causal_discovery()
        assert len(blob) >= 7
        assert all(np.isfinite(v) for v in blob.values())
        for key in blob:
            assert key.startswith("synthetic_")
            assert {"sharpe", "sortino", "calmar", "pnl", "nav"}.isdisjoint(key.lower().split("_"))

    def test_bench_quality(self) -> None:
        blob = bench_causal_discovery()
        assert blob["synthetic_skeleton_f1"] > 0.4
        assert blob["synthetic_lingam_dir_acc"] > 0.7
        assert blob["synthetic_ges_bic_gain"] > 0
        assert blob["synthetic_determinism"] == 1.0
