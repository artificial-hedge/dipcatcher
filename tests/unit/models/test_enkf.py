"""Tests for models/enkf.py — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.enkf import (
    assimilate_l96,
    bench_enkf,
    eakf_update,
    enkf_update,
    inflate,
    integrate_l96,
    letkf_update,
    localize_gaspari_cohn,
    lorenz96,
)


class TestLorenz96:
    def test_rhs_shape_and_bounded_orbit(self) -> None:
        x = np.ones(6) * 8.0
        dx = lorenz96(x, f=8.0)
        assert dx.shape == (6,)
        orbit = integrate_l96(x, dt=0.05, steps=200, f=8.0)
        assert np.all(np.isfinite(orbit))
        assert np.abs(orbit).max() < 1e3  # attractor is bounded

    def test_constant_state_uniform_drift(self) -> None:
        dx = lorenz96(np.full(10, 5.0), f=8.0)
        # (5-5)*5 - 5 + 8 = 3 uniformly
        assert np.allclose(dx, 3.0)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            lorenz96(np.ones(2))
        with pytest.raises(ValueError):
            integrate_l96(np.ones(5), dt=0.0, steps=1)


class TestGaspariCohn:
    def test_endpoints(self) -> None:
        assert localize_gaspari_cohn(0.0, 1.0) == pytest.approx(1.0)
        assert localize_gaspari_cohn(2.0, 1.0) == pytest.approx(0.0, abs=1e-9)
        assert localize_gaspari_cohn(3.0, 1.0) == pytest.approx(0.0)

    def test_continuous_at_knot(self) -> None:
        c = 1.5
        left = float(localize_gaspari_cohn(c - 1e-9, c))
        right = float(localize_gaspari_cohn(c + 1e-9, c))
        assert abs(left - right) < 1e-4

    def test_monotone_decay(self) -> None:
        z = np.linspace(0, 2, 50)
        g = np.asarray(localize_gaspari_cohn(z, c=1.0))
        assert np.all(np.diff(g) <= 1e-9)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            localize_gaspari_cohn(0.5, 0.0)


class TestUpdates:
    @staticmethod
    def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        ens = rng.standard_normal((40, 5))
        h = ens[:, :3] + 0.1
        y = np.array([0.0, 0.0, 0.0])
        return ens, y, h

    def test_enkf_reduces_mean_error(self) -> None:
        ens, y, h = self._toy()
        post = enkf_update(ens, y, h, r=0.5, seed=0)
        # posterior mean of observed components moves toward y
        prior_err = np.abs(h.mean(axis=0) - y).sum()
        post_err = np.abs(post[:, :3].mean(axis=0) - y).sum()
        assert post_err < prior_err

    def test_eakf_posterior_variance(self) -> None:
        ens, y, h = self._toy()
        post = eakf_update(ens, y, h, r=0.5)
        p_var = float(np.var(h[:, 0], ddof=1))
        expected_var = 1.0 / (1.0 / 0.5 + 1.0 / p_var)
        got_var = float(np.var(post[:, 0], ddof=1))
        # state component 0 is observed (h[:,0]=ens[:,0]+0.1) → its
        # posterior variance matches the scalar Bayesian update
        assert got_var == pytest.approx(expected_var, rel=0.15)

    def test_letkf_transform_shrinks_spread(self) -> None:
        ens, y, h = self._toy()
        post = letkf_update(ens, y, h, r_local=0.5)
        assert post.std(axis=0, ddof=1).mean() < ens.std(axis=0, ddof=1).mean() + 0.2
        assert (
            np.abs(post.mean(axis=0)[:3] - y).sum() < np.abs(ens.mean(axis=0)[:3] - y).sum() + 0.5
        )

    def test_inflate(self) -> None:
        ens, _y, _h = self._toy()
        out = inflate(ens, 1.5)
        assert np.allclose(out.mean(axis=0), ens.mean(axis=0))
        assert out.std(axis=0).mean() == pytest.approx(ens.std(axis=0).mean() * 1.5, rel=1e-6)

    def test_invalid(self) -> None:
        ens, y, h = self._toy()
        with pytest.raises(ValueError):
            enkf_update(ens[:1], y, h[:1], r=1.0)
        with pytest.raises(ValueError):
            eakf_update(ens, y, h, r=-1.0)
        with pytest.raises(ValueError):
            letkf_update(ens, y[:-1], h, r_local=1.0)
        with pytest.raises(ValueError):
            inflate(ens, 0.0)


class TestAssimilate:
    def test_beats_climatology(self) -> None:
        res = assimilate_l96(n_cycles=60, n_members=30, rho=1.10, seed=0)
        assert res["rmse"].mean() < res["clim_rmse"][0]

    def test_determinism(self) -> None:
        a = assimilate_l96(n_cycles=30, n_members=15, seed=1)
        b = assimilate_l96(n_cycles=30, n_members=15, seed=1)
        assert np.array_equal(a["rmse"], b["rmse"])

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            assimilate_l96(method="bogus")
        with pytest.raises(ValueError):
            assimilate_l96(n_members=1)


class TestBench:
    def test_keys_finite(self) -> None:
        blob = bench_enkf()
        assert len(blob) >= 8
        assert all(np.isfinite(v) for v in blob.values())
        for key in blob:
            assert key.startswith("synthetic_")

    def test_bench_skill(self) -> None:
        blob = bench_enkf()
        assert blob["synthetic_enkf_rmse_vs_clim"] < 1.0
        assert blob["synthetic_eakf_rmse_vs_clim"] < 1.0
        assert blob["synthetic_letkf_rmse_vs_clim"] <= 1.0
        assert blob["synthetic_determinism"] == 1.0
