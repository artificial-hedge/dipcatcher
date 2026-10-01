"""Tests for models/durbin_koopman.py — DK simulation/disturbance smoothers."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.durbin_koopman import (
    bench_durbin_koopman,
    disturbance_smooth,
    draw_posterior_paths,
    kalman_filter,
    rts_smooth,
    simulation_smoother,
    synth_ssm,
)


@pytest.fixture(scope="module")
def sim() -> dict:
    return synth_ssm(200, 4, 0.9, seed=0)


@pytest.fixture(scope="module")
def mats(sim) -> tuple:
    return (sim["z"], sim["trans"], sim["q"], sim["r"], sim["a1"], sim["p1"])


class TestGuards:
    def test_bad_shapes(self, mats):
        y = np.ones((50, 3))
        z, trans, q, r, a1, p1 = mats
        with pytest.raises(ValueError):
            kalman_filter(np.ones((3, 3)), z, trans, q, r, a1, p1)
        with pytest.raises(ValueError):
            kalman_filter(y, np.ones((3, 2)), trans, q, r, a1, p1)
        with pytest.raises(ValueError):
            kalman_filter(y, z[:, :1], np.ones((2, 2)), q, r, np.zeros(2), np.eye(2))
        with pytest.raises(ValueError):
            kalman_filter(y, z, trans, q, -np.ones(4), a1, p1)
        with pytest.raises(ValueError):
            kalman_filter(y, z, trans, q, np.full((4, 4), 0.3), a1, p1)
        with pytest.raises(ValueError):
            kalman_filter(np.full((50, 4), np.nan), z, trans, q, r, a1, p1)
        with pytest.raises(ValueError):
            simulation_smoother(y, z, trans, q, r, a1, p1, n_draws=1)
        with pytest.raises(ValueError):
            synth_ssm(10, 2, 0.9, 0)


class TestFilter:
    def test_shapes(self, sim, mats):
        kf = kalman_filter(sim["y"], *mats)
        assert np.asarray(kf["a_filt"]).shape == (200, 1)
        assert np.asarray(kf["p_filt"]).shape == (200, 1, 1)
        assert float(kf["loglik"]) < 0

    def test_recovers_factor(self, sim, mats):
        kf = kalman_filter(sim["y"], *mats)
        corr = abs(np.corrcoef(np.asarray(kf["a_filt"])[:, 0], sim["f_true"])[0, 1])
        assert corr > 0.8

    def test_masked_rows(self, sim, mats):
        y = sim["y"].copy()
        y[::7] = np.nan
        kf = kalman_filter(y, *mats)
        assert np.isfinite(np.asarray(kf["a_filt"])).all()

    def test_full_nan_row_skipped(self, mats):
        rng = np.random.default_rng(0)
        y = rng.standard_normal((40, 3))
        y[10] = np.nan
        kf = kalman_filter(
            y,
            np.ones((3, 1)),
            np.asarray([[0.9]]),
            np.asarray([[0.19]]),
            np.full(3, 0.5),
            np.zeros(1),
            np.eye(1),
        )
        # filtered state at a fully-missing row equals its prediction
        assert np.asarray(kf["a_filt"])[10] == pytest.approx(np.asarray(kf["a_pred"])[10])


class TestRTS:
    def test_smoother_beats_filter(self, sim, mats):
        kf = kalman_filter(sim["y"], *mats)
        sm = rts_smooth(sim["y"], *mats)
        corr_f = abs(np.corrcoef(np.asarray(kf["a_filt"])[:, 0], sim["f_true"])[0, 1])
        corr_s = abs(np.corrcoef(np.asarray(sm["a_s"])[:, 0], sim["f_true"])[0, 1])
        assert corr_s >= corr_f - 1e-9

    def test_endpoint_agrees_with_filter(self, sim, mats):
        kf = kalman_filter(sim["y"], *mats)
        sm = rts_smooth(sim["y"], *mats, _kf=kf)
        np.testing.assert_allclose(np.asarray(sm["a_s"])[-1], np.asarray(kf["a_filt"])[-1])


class TestDisturbance:
    def test_shapes(self, sim, mats):
        dist = disturbance_smooth(sim["y"], *mats)
        assert np.asarray(dist["eta_hat"]).shape == (200, 1)
        assert np.asarray(dist["eps_hat"]).shape == (200, 4)

    def test_eta_hat_correlates_with_truth(self, sim, mats):
        # disturbance recovery is weak but positive for a persistent factor
        dist = disturbance_smooth(sim["y"], *mats)
        corr = abs(np.corrcoef(np.asarray(dist["eta_hat"])[1:-1, 0], sim["eta_true"][1:-1])[0, 1])
        assert corr > 0.1

    def test_eps_hat_is_obs_residual(self, sim, mats):
        # eps_hat should agree with y - Z a_s in projection
        dist = disturbance_smooth(sim["y"], *mats)
        sm = rts_smooth(sim["y"], *mats)
        resid = sim["y"] - np.asarray(sm["a_s"]) @ sim["z"].T
        eps = np.asarray(dist["eps_hat"])
        corr = np.corrcoef(resid.ravel(), eps.ravel())[0, 1]
        assert corr > 0.3


class TestSimulationSmoother:
    def test_draw_shapes(self, sim, mats):
        draws = simulation_smoother(sim["y"], *mats, n_draws=20, seed=0)
        assert draws.shape == (20, 200, 1)

    def test_draws_match_smoother_moments(self, sim, mats):
        sm = rts_smooth(sim["y"], *mats)
        a_s = np.asarray(sm["a_s"])[:, 0]
        p_s = np.asarray(sm["p_s"])[:, 0, 0]
        draws = simulation_smoother(sim["y"], *mats, n_draws=300, seed=1)
        mean_err = np.abs(draws.mean(axis=0)[:, 0] - a_s).max()
        sd_ratio = (draws.std(axis=0)[:, 0] / np.sqrt(p_s)).mean()
        assert mean_err < 0.15
        assert abs(sd_ratio - 1.0) < 0.15

    def test_draw_calibration(self, sim, mats):
        draws = simulation_smoother(sim["y"], *mats, n_draws=200, seed=2)
        inside = (
            np.abs(sim["f_true"] - draws.mean(axis=0)[:, 0]) <= 1.96 * draws.std(axis=0)[:, 0]
        ).mean()
        assert 0.8 <= inside <= 1.0

    def test_deterministic(self, sim, mats):
        d1 = simulation_smoother(sim["y"], *mats, n_draws=10, seed=5)
        d2 = simulation_smoother(sim["y"], *mats, n_draws=10, seed=5)
        np.testing.assert_array_equal(d1, d2)

    def test_draw_posterior_paths_bundle(self, sim, mats):
        out = draw_posterior_paths(sim["y"], *mats, n_draws=40, seed=3)
        assert out["draws"].shape == (40, 200, 1)
        assert out["draw_mean"].shape == (200, 1)
        assert out["draw_cov"].shape == (200, 1, 1)


class TestSynth:
    def test_shapes(self):
        s = synth_ssm(120, 3, 0.85, 7)
        assert s["y"].shape == (120, 3)
        assert s["f_true"].shape == (120,)

    def test_factor_autocorr(self):
        s = synth_ssm(500, 2, 0.9, 4)
        ac = np.corrcoef(s["f_true"][:-1], s["f_true"][1:])[0, 1]
        assert ac > 0.8


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_durbin_koopman(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_bench_contract_no_forbidden(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_durbin_koopman(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_bench_science(self):
        blob = bench_durbin_koopman(seed=0)
        assert blob["synthetic_smooth_corr"] > 0.9
        assert blob["synthetic_smooth_ge_filter"] > 0.5
        assert blob["synthetic_draw_mean_corr"] > 0.95
        assert blob["synthetic_draw_cov_calib"] > 0.8
        assert blob["synthetic_determinism"] == 1.0
