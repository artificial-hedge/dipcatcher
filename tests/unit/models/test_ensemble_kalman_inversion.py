"""Tests for models/ensemble_kalman_inversion.py — EKI."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ensemble_kalman_inversion import (
    bench_ensemble_kalman_inversion,
    eki_update,
    ensemble_kalman_inversion,
    sv_moment_forward,
    sv_moment_obs,
    synth_inverse_problem,
    tikhonov_eki,
)


@pytest.fixture(scope="module")
def lin_prob() -> dict[str, np.ndarray]:
    return synth_inverse_problem(20, 4, seed=0)


@pytest.fixture(scope="module")
def lin_setup(lin_prob):
    a_mat = lin_prob["a"]

    def fwd(th):
        return a_mat @ th

    def prior(j, r):
        return r.normal(0.0, 2.0, (j, 4))

    return fwd, prior


class TestGuards:
    def test_bad_inputs(self, lin_setup):
        fwd, prior = lin_setup
        with pytest.raises(ValueError):
            eki_update(np.ones((3, 4)), np.zeros(20), fwd, np.eye(20), np.random.default_rng(0))
        with pytest.raises(ValueError):
            eki_update(np.ones((10, 4)), np.zeros(20), fwd, np.eye(20), "not_rng")
        with pytest.raises(ValueError):
            eki_update(
                np.ones((10, 4)),
                np.zeros(20),
                fwd,
                np.eye(20),
                np.random.default_rng(0),
                inflation=0.0,
            )
        with pytest.raises(ValueError):
            eki_update(
                np.ones((10, 4)),
                np.zeros(20),
                fwd,
                -np.eye(20),
                np.random.default_rng(0),
            )
        with pytest.raises(ValueError):
            ensemble_kalman_inversion(np.zeros(5), prior, fwd, n_ens=3)
        with pytest.raises(ValueError):
            synth_inverse_problem(4, 10)

    def test_nonfinite_forward_fails(self, lin_setup):
        _, prior = lin_setup

        def bad(th):
            return np.full(20, np.nan)

        with pytest.raises(ValueError):
            ensemble_kalman_inversion(np.zeros(20), prior, bad, n_ens=8, n_iter=2)


class TestUpdate:
    def test_update_moves_toward_obs(self, lin_prob, lin_setup):
        fwd, _ = lin_setup
        rng = np.random.default_rng(0)
        th0 = rng.normal(0.0, 2.0, (40, 4))
        th1, g0 = eki_update(th0, lin_prob["y"], fwd, lin_prob["cov_obs"], rng)
        assert th1.shape == th0.shape
        # misfit shrinks after one step
        m0 = np.linalg.norm(g0 - lin_prob["y"][None, :], axis=1).mean()
        g1 = np.stack([fwd(th1[j]) for j in range(40)])
        m1 = np.linalg.norm(g1 - lin_prob["y"][None, :], axis=1).mean()
        assert m1 < m0

    def test_perturb_free_variant(self, lin_prob, lin_setup):
        fwd, _ = lin_setup
        rng = np.random.default_rng(0)
        th0 = rng.normal(0.0, 2.0, (40, 4))
        a, _ = eki_update(th0, lin_prob["y"], fwd, lin_prob["cov_obs"], rng, perturb=True)
        rng2 = np.random.default_rng(0)
        b, _ = eki_update(th0, lin_prob["y"], fwd, lin_prob["cov_obs"], rng2, perturb=False)
        assert not np.array_equal(a, b)


class TestLoop:
    def test_converges_on_linear(self, lin_prob, lin_setup):
        fwd, prior = lin_setup
        res = ensemble_kalman_inversion(
            lin_prob["y"],
            prior,
            fwd,
            n_ens=60,
            n_iter=25,
            seed=0,
            cov_obs=lin_prob["cov_obs"],
        )
        rel = np.linalg.norm(res.theta_mean - lin_prob["theta_true"]) / np.linalg.norm(
            lin_prob["theta_true"]
        )
        assert rel < 0.1
        assert res.misfit_path[-1] < res.misfit_path[0]
        assert res.spread_path[-1] < res.spread_path[0]

    def test_deterministic(self, lin_prob, lin_setup):
        fwd, prior = lin_setup
        kw = dict(n_ens=40, n_iter=10, seed=3, cov_obs=lin_prob["cov_obs"])
        a = ensemble_kalman_inversion(lin_prob["y"], prior, fwd, **kw)
        b = ensemble_kalman_inversion(lin_prob["y"], prior, fwd, **kw)
        np.testing.assert_array_equal(a.theta_post, b.theta_post)

    def test_tikhonov_recovers(self, lin_prob, lin_setup):
        fwd, prior = lin_setup
        res = tikhonov_eki(
            lin_prob["y"],
            prior,
            fwd,
            prior_mean=np.zeros(4),
            prior_cov=np.eye(4) * 4.0,
            n_ens=60,
            n_iter=25,
            seed=0,
        )
        rel = np.linalg.norm(res.theta_mean - lin_prob["theta_true"]) / np.linalg.norm(
            lin_prob["theta_true"]
        )
        assert rel < 0.15

    def test_inflation_preserves_spread(self, lin_prob, lin_setup):
        fwd, prior = lin_setup
        kw = dict(n_ens=40, n_iter=15, seed=1, cov_obs=lin_prob["cov_obs"])
        base = ensemble_kalman_inversion(lin_prob["y"], prior, fwd, **kw)
        infl = ensemble_kalman_inversion(lin_prob["y"], prior, fwd, inflation=1.05, **kw)
        assert infl.spread_path[-1] > base.spread_path[-1]


class TestSVForward:
    def test_shape_and_finite(self):
        out = sv_moment_forward(np.array([3.0, -1.5]), n_obs=400, seed=0)
        assert out.shape == (3,) and np.isfinite(out).all()

    def test_persistent_vol_higher_acf(self):
        hi = sv_moment_forward(np.array([4.0, -1.5]), n_obs=800, seed=0)
        lo = sv_moment_forward(np.array([1.0, -1.5]), n_obs=800, seed=0)
        assert hi[0] > lo[0]  # higher φ → higher return autocorrelation

    def test_moment_obs(self):
        obs = sv_moment_obs(0.97, 0.25, seed=0)
        assert obs.shape == (3,)
        assert obs[0] > 0  # persistent SV → positive lag-1 acf


class TestBench:
    def test_keys_finite(self):
        blob = bench_ensemble_kalman_inversion(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_no_forbidden_tokens(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_ensemble_kalman_inversion(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_science(self):
        blob = bench_ensemble_kalman_inversion(seed=0)
        assert blob["synthetic_param_relerr"] < 0.15
        assert blob["synthetic_misfit_reduction"] > 5.0
        assert blob["synthetic_spread_collapse"] < 0.2
        assert blob["synthetic_monotone_misfit_frac"] > 0.7
        assert blob["synthetic_determinism"] == 1.0
