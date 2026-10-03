"""Tests for TDA sliding-window regime detection (models/tda_persistence.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.tda_persistence import (
    _pairwise_dists,
    _vr_pairs,
    bottleneck_distance,
    diagram_landscape_l2,
    h0_persistence,
    h1_persistence,
    landscape_l2,
    landscape_norm,
    persistence_entropy,
    persistence_landscape,
    persistence_summary,
    takens_embedding,
    tda_regime_scores,
    vietoris_rips,
)


def _circle(n: int = 16, r: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    cloud = np.column_stack([r * np.cos(t), r * np.sin(t)])
    if noise > 0.0:
        cloud = cloud + noise * np.random.default_rng(seed).standard_normal(cloud.shape)
    return cloud


class TestTakens:
    def test_shape_and_rows(self):
        x = np.arange(10.0)
        emb = takens_embedding(x, dim=3, delay=2)
        assert emb.shape == (10 - 2 * 2, 3)
        np.testing.assert_allclose(emb[0], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(emb[-1], [5.0, 7.0, 9.0])

    def test_delay_one(self):
        emb = takens_embedding(np.arange(6.0), dim=2)
        np.testing.assert_allclose(emb[0], [0.0, 1.0])
        np.testing.assert_allclose(emb[-1], [4.0, 5.0])

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            takens_embedding(np.array([1.0, np.nan, 2.0]), dim=2)
        with pytest.raises(ValueError):
            takens_embedding(np.ones((5, 2)), dim=2)
        with pytest.raises(ValueError):
            takens_embedding(np.arange(10.0), dim=1)
        with pytest.raises(ValueError):
            takens_embedding(np.arange(10.0), dim=2, delay=0)
        with pytest.raises(ValueError):
            takens_embedding(np.arange(5.0), dim=5, delay=3)  # too few points


class TestH0UnionFind:
    def test_known_line_points(self):
        # Points 0, 1, 10 on a line: merges at 1 and 9.
        dgm = h0_persistence(np.array([[0.0], [1.0], [10.0]]))
        np.testing.assert_allclose(np.sort(dgm[:, 1]), [1.0, 9.0])
        assert np.all(dgm[:, 0] == 0.0)

    def test_two_points(self):
        dgm = h0_persistence(np.array([[0.0], [3.0]]))
        np.testing.assert_allclose(dgm, [[0.0, 3.0]])

    def test_n_minus_one_bars(self):
        pts = np.random.default_rng(0).standard_normal((12, 3))
        dgm = h0_persistence(pts)
        assert dgm.shape == (11, 2)
        assert np.all(dgm[:, 1] > dgm[:, 0])

    def test_matches_boundary_reduction(self):
        # The general Z2 boundary reduction must reproduce union-find H0.
        pts = np.random.default_rng(1).standard_normal((10, 2))
        pairs, _, f_sorted, dims = _vr_pairs(_pairwise_dists(pts), None)
        h0_red = sorted(f_sorted[j] for piv, j in pairs if dims[piv] == 0 and dims[j] == 1)
        h0_uf = sorted(h0_persistence(pts)[:, 1].tolist())
        np.testing.assert_allclose(h0_red, h0_uf)

    def test_r_max_splits_components(self):
        # Two clusters far apart; bounding r_max inside the gap drops the
        # bridging edge, so the bounded complex keeps both clusters.
        pts = np.array([[0.0], [1.0], [10.0], [11.0]])
        dgm = h0_persistence(pts, r_max=5.0)
        np.testing.assert_allclose(np.sort(dgm[:, 1]), [1.0, 1.0])


class TestH1BoundaryReduction:
    def test_circle_one_long_bar(self):
        dgm = h1_persistence(_circle(16))
        assert dgm.shape[0] >= 1
        pers = dgm[:, 1] - dgm[:, 0]
        top = dgm[int(np.argmax(pers))]
        # Unit circle: loop born near edge spacing, dies near the diameter.
        assert top[1] - top[0] > 0.8

    def test_blob_no_long_bar(self):
        pts = np.random.default_rng(2).standard_normal((14, 2))
        dgm = h1_persistence(pts)
        pers = dgm[:, 1] - dgm[:, 0]
        assert pers.max() < 0.8

    def test_known_triangle(self):
        # Equilateral triangle: one loop born at edge length 1, filled at 1.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])
        dgm = h1_persistence(pts)
        assert dgm.shape == (0, 2)  # filled immediately -> zero-length pruned
        dgm_e = h1_persistence(pts, include_essential=True)
        assert dgm_e.shape == (0, 2)

    def test_square_essential_when_unbounded_without_triangles(self):
        # Bounding r_max below the diagonal leaves a 4-cycle with no filling
        # triangle -> one essential H1 class born at the side length.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        dgm = h1_persistence(pts, r_max=1.0, include_essential=True)
        assert dgm.shape == (1, 2)
        assert dgm[0, 0] == pytest.approx(1.0)
        assert math.isinf(dgm[0, 1])
        # Without include_essential the loop is invisible.
        assert h1_persistence(pts, r_max=1.0).shape == (0, 2)

    def test_square_with_diagonal_filled(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        dgm = h1_persistence(pts)
        # One loop, born at side 1, killed at diagonal sqrt(2).
        np.testing.assert_allclose(dgm, [[1.0, math.sqrt(2.0)]], atol=1e-10)

    def test_vietoris_rips_both_dims(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        out = vietoris_rips(pts)
        assert set(out) == {"h0", "h1"}
        np.testing.assert_allclose(np.sort(out["h0"][:, 1]), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(out["h1"], [[1.0, math.sqrt(2.0)]])

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            h1_persistence(np.array([[0.0]]))  # one point
        with pytest.raises(ValueError):
            h1_persistence(np.array([[0.0, np.nan], [1.0, 0.0]]))
        with pytest.raises(ValueError):
            h1_persistence(np.zeros((4, 2)), r_max=-1.0)


class TestLandscape:
    def test_single_tent_analytic_l2(self):
        # T(t) = min(t-b, d-t)+ on [b,d]: int T^2 = (d-b)^3 / 12.
        dgm = np.array([[1.0, 5.0]])
        land = persistence_landscape(dgm, n_levels=1, n_grid=1001)
        expected = math.sqrt((4.0**3) / 12.0)
        assert landscape_norm(land, p=2.0) == pytest.approx(expected, rel=0.02)

    def test_tent_shape(self):
        dgm = np.array([[0.0, 2.0]])
        land = persistence_landscape(dgm, n_levels=2, n_grid=201)
        mid = int(np.argmin(np.abs(land.grid - 1.0)))
        assert land.values[0, mid] == pytest.approx(1.0, abs=0.02)
        assert land.values[1].max() == 0.0  # level 2 empty for one point

    def test_second_level_picks_smaller_tent(self):
        dgm = np.array([[0.0, 4.0], [1.0, 2.0]])
        land = persistence_landscape(dgm, n_levels=2, n_grid=401)
        t15 = int(np.argmin(np.abs(land.grid - 1.5)))
        assert land.values[1, t15] == pytest.approx(0.5, abs=0.02)

    def test_empty_diagram(self):
        land = persistence_landscape(np.empty((0, 2)), n_levels=3, n_grid=50)
        assert land.values.shape == (3, 50)
        assert np.all(land.values == 0.0)

    def test_norm_levels_and_linf(self):
        land = persistence_landscape(np.array([[0.0, 2.0]]), n_levels=1, n_grid=101)
        assert landscape_norm(land, p=math.inf) == pytest.approx(1.0, abs=0.02)
        assert landscape_norm(land, p=1.0) == pytest.approx(1.0, abs=0.03)

    def test_l2_self_zero_and_symmetric(self):
        d1 = np.array([[0.0, 2.0], [1.0, 3.0]])
        d2 = np.array([[0.2, 1.8]])
        assert diagram_landscape_l2(d1, d1, n_grid=64) == pytest.approx(0.0, abs=1e-9)
        assert diagram_landscape_l2(d1, d2, n_grid=64) == pytest.approx(
            diagram_landscape_l2(d2, d1, n_grid=64)
        )

    def test_l2_grid_mismatch_fails(self):
        a = persistence_landscape(np.array([[0.0, 2.0]]), n_grid=50, t_min=0, t_max=2)
        b = persistence_landscape(np.array([[0.0, 2.0]]), n_grid=60, t_min=0, t_max=2)
        with pytest.raises(ValueError):
            landscape_l2(a, b)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            persistence_landscape(np.array([[3.0, 1.0]]))  # death < birth
        with pytest.raises(ValueError):
            persistence_landscape(np.array([[0.0, math.inf]]))  # essential bar
        with pytest.raises(ValueError):
            landscape_norm(persistence_landscape(np.array([[0.0, 2.0]])), p=0.5)


class TestBottleneck:
    def test_identity_zero(self):
        d = np.array([[0.0, 2.0], [0.5, 3.5]])
        assert bottleneck_distance(d, d) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric(self):
        a = np.array([[0.0, 2.0], [1.0, 3.0]])
        b = np.array([[0.3, 1.7], [0.0, 4.0]])
        assert bottleneck_distance(a, b) == pytest.approx(bottleneck_distance(b, a))

    def test_planted_shift_exact(self):
        # Uniform shift by delta below half the min persistence: d_B = delta.
        rng = np.random.default_rng(3)
        births = rng.uniform(0.0, 0.5, 6)
        deaths = births + rng.uniform(2.0, 3.0, 6)
        a = np.column_stack([births, deaths])
        assert bottleneck_distance(a, a + 0.25) == pytest.approx(0.25)

    def test_empty_and_diagonal(self):
        assert bottleneck_distance(np.empty((0, 2)), np.empty((0, 2))) == 0.0
        d = np.array([[1.0, 3.0]])
        assert bottleneck_distance(np.empty((0, 2)), d) == pytest.approx(1.0)

    def test_birth_death_move(self):
        # Moving one point's death by eps costs eps (L-infinity cost).
        a = np.array([[0.0, 2.0]])
        b = np.array([[0.0, 2.2]])
        assert bottleneck_distance(a, b) == pytest.approx(0.2, abs=1e-12)

    def test_triangle_inequality(self):
        rng = np.random.default_rng(4)
        a = np.column_stack([rng.uniform(0, 1, 5), rng.uniform(2, 4, 5)])
        b = np.column_stack([rng.uniform(0, 1, 5), rng.uniform(2, 4, 5)])
        c = np.column_stack([rng.uniform(0, 1, 5), rng.uniform(2, 4, 5)])
        d_ac = bottleneck_distance(a, c)
        assert d_ac <= bottleneck_distance(a, b) + bottleneck_distance(b, c) + 1e-9

    def test_greedy_upper_bounds_exact(self):
        rng = np.random.default_rng(5)
        for _ in range(10):
            a = np.column_stack([rng.uniform(0, 1, 6), rng.uniform(2, 4, 6)])
            b = np.column_stack([rng.uniform(0, 1, 6), rng.uniform(2, 4, 6)])
            exact = bottleneck_distance(a, b)
            greedy = bottleneck_distance(a, b, method="greedy")
            assert greedy >= exact - 1e-9

    def test_greedy_exact_agree_on_easy_pairs(self):
        a = np.array([[0.0, 3.0]])
        b = np.array([[0.1, 3.2]])
        assert bottleneck_distance(a, b, "greedy") == pytest.approx(bottleneck_distance(a, b))

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            bottleneck_distance(np.array([[0.0, 2.0]]), np.array([[0.0, 2.0]]), "bogus")
        with pytest.raises(ValueError):
            bottleneck_distance(np.array([[0.0, -2.0]]), np.empty((0, 2)))


class TestEntropyAndSummary:
    def test_entropy_known(self):
        # Two equal lifetimes -> log 2.
        d = np.array([[0.0, 1.0], [2.0, 3.0]])
        assert persistence_entropy(d) == pytest.approx(math.log(2.0))
        assert persistence_entropy(np.empty((0, 2))) == 0.0

    def test_summary_keys_and_values(self):
        d = np.array([[0.0, 2.0], [0.5, 1.0]])
        s = persistence_summary(d, n_grid=100)
        assert s["n_points"] == 2.0
        assert s["total_persistence"] == pytest.approx(2.5)
        assert s["max_persistence"] == pytest.approx(2.0)
        assert s["landscape_l2"] > 0.0


class TestRegimeScores:
    def _drift_series(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n1, n2 = 160, 160
        e = rng.standard_normal(n1 + n2)
        x1 = 0.5 * e[:n1]
        tt = np.arange(n2)
        x2 = 1.5 * np.sin(2.0 * math.pi * tt / 24.0) + 0.15 * e[n1:]
        return np.concatenate([x1, x2])

    def test_reference_mode_detects_boundary(self):
        x = self._drift_series()
        res = tda_regime_scores(x, window=32, emb_dim=2, delay=4, stride=4, dims=(1,), n_ref=8)
        assert res.scores.shape == res.starts.shape
        pos = res.starts >= 160.0
        assert res.scores[pos].mean() > res.scores[~pos].mean()

    def test_consecutive_mode_alignment(self):
        x = self._drift_series()
        res = tda_regime_scores(x, window=32, emb_dim=2, delay=4, stride=4, dims=(0,))
        assert res.scores.shape == res.starts.shape
        assert len(res.scores) == len(np.arange(0, 320 - 32 + 1, 4)) - 1

    def test_fail_closed(self):
        x = np.arange(100.0)
        with pytest.raises(ValueError):
            tda_regime_scores(x, window=2, emb_dim=2)
        with pytest.raises(ValueError):
            tda_regime_scores(x, window=32, emb_dim=2, stride=0)
        with pytest.raises(ValueError):
            tda_regime_scores(x, window=32, emb_dim=2, dims=(2,))
        with pytest.raises(ValueError):
            tda_regime_scores(x, window=100, emb_dim=2)  # only 1 window
        with pytest.raises(ValueError):
            tda_regime_scores(x, window=32, emb_dim=2, n_ref=99)


class TestBench:
    def test_keys_prefixed_and_float(self):
        from quant_fund.models.tda_persistence import bench_tda_persistence

        out = bench_tda_persistence(5)
        assert out
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)

    def test_deterministic(self):
        from quant_fund.models.tda_persistence import bench_tda_persistence

        assert bench_tda_persistence(9) == bench_tda_persistence(9)

    def test_headline_ranges(self):
        from quant_fund.models.tda_persistence import bench_tda_persistence

        out = bench_tda_persistence(7)
        assert out["synthetic_bottleneck_self"] < 1e-9
        assert out["synthetic_bottleneck_err"] < 1e-6
        assert out["synthetic_landscape_sep"] > 0.0
        assert out["synthetic_regime_auc"] > 0.9
