"""Tests for metrics/sga_uq.py — SGA graph-complexity UQ (arXiv:2609.28582)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.sga_uq import (
    ForecastEnsemble,
    bench_sga_uq,
    build_slice_dag,
    graph_complexity,
    inverse_cdf_sample,
    kde_entropy,
    kde_log_density,
    node_uncertainties,
    seasonal_tau,
    sga_uq_score,
    slice_rollouts,
)


def _hist(n: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, 0.4, n)) + 5.0 * np.sin(2 * np.pi * np.arange(n) / 7.0)


def _rollouts(k: int = 16, h: int = 12, seed: int = 1, spread: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(h, dtype=float)
    base = np.sin(2 * np.pi * t / 7.0)
    return base[None, :] + rng.normal(0.0, spread, (k, h)) * (t[None, :] / h + 0.2)


class TestInverseCdfSample:
    def test_shape_and_finite(self) -> None:
        rng = np.random.default_rng(0)
        lv = np.linspace(0.05, 0.95, 9)
        qm = np.sort(np.tile(np.linspace(-1, 1, 9)[:, None], (1, 6)), axis=0)
        out = inverse_cdf_sample(lv, qm, 32, rng)
        assert out.shape == (32, 6)
        assert np.all(np.isfinite(out))

    def test_samples_lie_inside_quantile_envelope(self) -> None:
        rng = np.random.default_rng(3)
        lv = np.linspace(0.05, 0.95, 9)
        qm = np.tile(np.linspace(-2, 2, 9)[:, None], (1, 4))
        out = inverse_cdf_sample(lv, qm, 200, rng)
        assert out.min() >= qm.min() - 1e-9
        assert out.max() <= qm.max() + 1e-9

    def test_joint_quantile_level_shared_across_steps(self) -> None:
        # Monotone cross-step quantile grid: a drawn u must produce monotone
        # rank-coherent trajectories (the paper's horizon-level joint map).
        rng = np.random.default_rng(5)
        lv = np.linspace(0.05, 0.95, 9)
        ramp = np.linspace(0.0, 3.0, 8)
        qm = lv[:, None] * 4.0 - 2.0 + ramp[None, :]  # shift grows with step
        qm = np.sort(qm, axis=0)
        out = inverse_cdf_sample(lv, qm, 64, rng)
        # High draws sit high at every step -> within-trajectory rank corr 1.
        for k in range(64):
            ranks = np.argsort(np.argsort(out[k]))
            assert np.all(np.diff(out[k][np.argsort(ramp)]) >= -1e-9)
            assert ranks.size == 8

    def test_rejects_bad_inputs(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            inverse_cdf_sample(np.array([0.5]), np.ones((2, 3)), 4, rng)
        with pytest.raises(ValueError):
            inverse_cdf_sample(np.array([0.2, 0.8]), np.ones((2, 3)), 1, rng)
        with pytest.raises(ValueError):
            inverse_cdf_sample(np.array([0.5, 0.5]), np.ones((2, 3)), 4, rng)


class TestSlicing:
    def test_slice_partition(self) -> None:
        ro = _rollouts(k=4, h=12)
        slices = slice_rollouts(ro, 3)
        assert len(slices) == 4
        assert len(slices[0]) == 4
        np.testing.assert_allclose(slices[1][2], ro[1, 6:9])

    def test_remainder_dropped(self) -> None:
        ro = _rollouts(k=2, h=10)
        slices = slice_rollouts(ro, 4)
        assert len(slices[0]) == 2  # steps 8..9 dropped

    def test_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            slice_rollouts(_rollouts(h=4), 5)
        with pytest.raises(ValueError):
            slice_rollouts(np.zeros((0, 4)), 2)


class TestSeasonalTau:
    def test_tau_matches_seasonal_scaled_error(self) -> None:
        x = _hist()
        tau = seasonal_tau(x, season=7, lam=1.0)
        manual = float(np.mean(np.abs(x[:-7] - x[7:])))
        assert tau == pytest.approx(manual)

    def test_lam_scales_linearly(self) -> None:
        x = _hist()
        assert seasonal_tau(x, 7, 2.0) == pytest.approx(2.0 * seasonal_tau(x, 7, 1.0))

    def test_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            seasonal_tau(np.ones(3), season=7)
        with pytest.raises(ValueError):
            seasonal_tau(_hist(), season=0)
        with pytest.raises(ValueError):
            seasonal_tau(_hist(), season=7, lam=0.0)


class TestDag:
    def test_forks_survive_alignment(self) -> None:
        # Two well-separated branches -> last layer keeps 2 nodes.
        rng = np.random.default_rng(0)
        t = np.arange(12, dtype=float)
        up = t * 2.0 + rng.normal(0, 0.05, (8, 12))
        dn = -t + rng.normal(0, 0.05, (8, 12))
        ro = np.vstack([up, dn])
        dag = build_slice_dag(ro, slice_len=3, tau=0.5)
        assert dag.depth == 4
        assert dag.nodes[-1][0] or dag.nodes[-1][1]
        assert len(dag.nodes[-1]) == 2

    def test_tight_ensemble_collapses_to_one_node(self) -> None:
        ro = _rollouts(k=16, h=8, spread=0.02)
        dag = build_slice_dag(ro, slice_len=2, tau=5.0)
        assert all(len(layer) == 1 for layer in dag.nodes)

    def test_edges_track_temporal_dependency(self) -> None:
        ro = _rollouts(k=6, h=6, spread=0.4)
        dag = build_slice_dag(ro, slice_len=2, tau=1e9)  # merge everything
        # Single chain: every deeper layer's sole node has exactly one pred.
        for j in range(1, dag.depth):
            assert dag.preds[j][0] == [0]
        assert dag.edge_count() == dag.depth

    def test_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            build_slice_dag(_rollouts(h=6), 3, tau=0.0)


class TestUncertaintyAndComplexity:
    def test_kde_entropy_grows_with_spread(self) -> None:
        rng = np.random.default_rng(0)
        narrow = kde_entropy(rng.normal(0, 0.1, 50))
        wide = kde_entropy(rng.normal(0, 2.0, 50))
        assert wide > narrow

    def test_kde_log_density_peaks_at_center(self) -> None:
        v = np.linspace(-1, 1, 21)
        lp = kde_log_density(v, np.array([0.0, 3.0]))
        assert lp[0] > lp[1]

    def test_centrality_accumulates_through_dag(self) -> None:
        ro = np.vstack(
            [
                _rollouts(k=4, h=6, seed=1, spread=1.5),
                _rollouts(k=4, h=6, seed=2, spread=1.5) + 3.0,
            ]
        )
        dag = build_slice_dag(ro, slice_len=2, tau=0.8)
        nu = node_uncertainties(dag, ro, 2)
        gc = graph_complexity(dag, nu, alpha=0.1)
        naive = sum(u for layer in nu for u in layer)
        # Alpha-centrality adds the predecessor accumulation term.
        assert gc >= naive
        assert np.isfinite(gc)

    def test_deeper_dag_scores_higher_than_flat(self) -> None:
        ro = _rollouts(k=8, h=12, spread=1.0)
        tau = seasonal_tau(_hist(), 7)
        dag_deep = build_slice_dag(ro, slice_len=2, tau=tau)
        dag_flat = build_slice_dag(ro, slice_len=6, tau=tau)
        nu_d = node_uncertainties(dag_deep, ro, 2)
        nu_f = node_uncertainties(dag_flat, ro, 6)
        assert graph_complexity(dag_deep, nu_d) > graph_complexity(dag_flat, nu_f)


class TestEndToEnd:
    def test_sga_uq_score_keys(self) -> None:
        hist = _hist()
        ro = _rollouts(k=12, h=12)
        out = sga_uq_score(hist, ro, slice_len=3, season=7)
        for k in (
            "graph_complexity",
            "tau",
            "n_nodes",
            "n_edges",
            "max_layer_width",
            "mean_node_uncertainty",
        ):
            assert k in out and np.isfinite(out[k])
        assert out["n_nodes"] >= out["max_layer_width"]

    def test_forecast_ensemble_quantile_grid(self) -> None:
        ens = ForecastEnsemble(horizon=10, season=7)
        lv, qm = ens.predict_quantiles(_hist(), branch_weight=0.7)
        assert lv.shape[0] == qm.shape[0]
        assert qm.shape[1] == 10
        # Mixture grid is nondecreasing in level per step.
        assert np.all(np.diff(qm, axis=0) >= -1e-9)

    def test_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            sga_uq_score(np.ones(3), _rollouts(h=8), 2, season=7)
        with pytest.raises(ValueError):
            sga_uq_score(_hist(), _rollouts(h=8), 2, season=0)


class TestBench:
    @pytest.fixture(scope="class")
    def blob(self) -> dict[str, float]:
        return bench_sga_uq(seed=20261108)

    def test_clean_and_finite(self, blob: dict[str, float]) -> None:
        assert blob
        assert all(np.isfinite(v) for v in blob.values())
        assert all(k.startswith("synthetic_") for k in blob)

    def test_ranking_beats_baselines(self, blob: dict[str, float]) -> None:
        # Paper's headline: SGA ranks predictive error better than marginal
        # spread baselines (ensemble std, residual std).
        assert blob["synthetic_spearman_sga"] > 0.3
        assert blob["synthetic_sga_beats_ensemble_std"] == 1.0
        assert blob["synthetic_sga_beats_resid_std"] == 1.0

    def test_ambiguity_and_scaling(self, blob: dict[str, float]) -> None:
        assert blob["synthetic_gc_ambiguous_gt_clear"] == 1.0
        assert blob["synthetic_gc_scaling_direction"] == 1.0
        assert blob["synthetic_gc_tight"] < blob["synthetic_gc_wide"]

    def test_error_lift_and_coverage(self, blob: dict[str, float]) -> None:
        assert blob["synthetic_high_gc_error_lift"] >= 0.6
        assert 0.5 <= blob["synthetic_conformal_90_coverage_indist"] <= 1.0
        assert blob["synthetic_conformal_90_coverage_shift"] > 0.4

    def test_deterministic(self, blob: dict[str, float]) -> None:
        assert bench_sga_uq(seed=20261108) == blob
