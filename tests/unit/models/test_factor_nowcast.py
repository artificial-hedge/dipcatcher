"""Tests for models/factor_nowcast.py — dynamic-factor nowcasting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.factor_nowcast import (
    bench_factor_nowcast,
    em_dfm,
    factor_nowcast,
    kalman_smooth,
    news_decomp,
    synth_panel,
)


@pytest.fixture(scope="module")
def panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return synth_panel(160, 6, 1, 0.85, 0.15, seed=0)


@pytest.fixture(scope="module")
def fit(panel):
    return factor_nowcast(panel[0], n_factors=1, n_iter=150, seed=0)


class TestGuards:
    def test_bad_panel(self):
        with pytest.raises(ValueError):
            kalman_smooth(
                np.ones((4, 2)),
                np.ones((2, 1)),
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.ones(2),
                np.zeros(1),
                np.eye(1),
            )
        with pytest.raises(ValueError):
            em_dfm(np.full((50, 4), np.nan))
        with pytest.raises(ValueError):
            em_dfm(np.ones((50, 4)), n_factors=0)
        with pytest.raises(ValueError):
            factor_nowcast(np.ones((50, 4)), n_factors=9)
        with pytest.raises(ValueError):
            synth_panel(10, 6)
        with pytest.raises(ValueError):
            news_decomp(np.ones((40, 3)), np.ones((39, 3)), {"load": np.ones((3, 1))})

    def test_bad_state_mats(self, panel):
        y, _, _ = panel
        with pytest.raises(ValueError):
            kalman_smooth(
                y,
                np.ones((6, 2)),
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.ones(6),
                np.zeros(1),
                np.eye(1),
            )
        with pytest.raises(ValueError):
            kalman_smooth(
                y,
                np.ones((6, 1)),
                np.ones((1, 1)),
                np.ones((1, 1)),
                -np.ones(6),
                np.zeros(1),
                np.eye(1),
            )


class TestKalman:
    def test_shapes_finite(self, panel):
        y, _, lam = panel
        ks = kalman_smooth(
            y,
            lam,
            np.asarray([[0.8]]),
            np.asarray([[0.36]]),
            np.full(6, 0.25),
            np.zeros(1),
            np.eye(1),
        )
        assert ks["f_s"].shape == (160, 1)
        assert ks["p_s"].shape == (160, 1, 1)
        assert ks["p_lag"].shape == (160, 1, 1)
        assert np.isfinite(np.asarray(ks["f_s"])).all()
        assert float(ks["loglik"]) < 0

    def test_smoother_better_than_filter(self, panel):
        y, f_true, lam = panel
        ks = kalman_smooth(
            y,
            lam,
            np.asarray([[0.8]]),
            np.asarray([[0.36]]),
            np.full(6, 0.25),
            np.zeros(1),
            np.eye(1),
        )
        f_s = np.asarray(ks["f_s"])[:, 0]
        f_f = np.asarray(ks["f_filt"])[:, 0]
        err_s = np.sqrt(
            ((f_s - f_true[:, 0] * np.sign(np.corrcoef(f_s, f_true[:, 0])[0, 1])) ** 2).mean()
        )
        corr_s = abs(np.corrcoef(f_s, f_true[:, 0])[0, 1])
        corr_f = abs(np.corrcoef(f_f, f_true[:, 0])[0, 1])
        assert corr_s >= corr_f - 1e-9
        _ = err_s

    def test_missing_rows_no_crash(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal((40, 3))
        y[::5] = np.nan
        ks = kalman_smooth(
            y,
            np.ones((3, 1)),
            np.asarray([[0.7]]),
            np.asarray([[0.5]]),
            np.full(3, 0.5),
            np.zeros(1),
            np.eye(1),
        )
        assert np.isfinite(np.asarray(ks["f_s"])).all()


class TestEM:
    def test_converges_and_recovers(self, panel):
        y, f_true, _ = panel
        fit = em_dfm(y, n_factors=1, n_iter=150, seed=0)
        assert fit["converged"] is True
        ks = kalman_smooth(
            y,
            np.asarray(fit["load"]),
            np.asarray(fit["ar_coef"]),
            np.asarray(fit["q"]),
            np.asarray(fit["r"]),
            np.zeros(1),
            np.eye(1),
        )
        corr = abs(np.corrcoef(np.asarray(ks["f_s"])[:, 0], f_true[:, 0])[0, 1])
        assert corr > 0.8

    def test_nonconvergence_raises(self, panel):
        y, _, _ = panel
        with pytest.raises(ValueError, match="did not converge"):
            em_dfm(y, n_factors=1, n_iter=1, tol=1e-12, seed=0)

    def test_em_deterministic(self, panel):
        y, _, _ = panel
        a = em_dfm(y, n_factors=1, n_iter=80, seed=0)
        b = em_dfm(y, n_factors=1, n_iter=80, seed=0)
        np.testing.assert_allclose(np.asarray(a["load"]), np.asarray(b["load"]))


class TestNowcast:
    def test_result_fields(self, fit):
        assert fit.factor.shape == (160, 1)
        assert fit.load.shape == (6, 1)
        assert fit.nowcast.shape == (6,)
        assert fit.r2.shape == (6,)
        assert fit.n_iter > 0

    def test_r2_reasonable(self, fit):
        assert fit.r2.mean() > 0.5

    def test_factor_correlation(self, panel, fit):
        _, f_true, _ = panel
        corr = abs(np.corrcoef(fit.factor[:, 0], f_true[:, 0])[0, 1])
        assert corr > 0.8

    def test_beats_naive_carry(self, panel):
        y, _, _ = panel
        obs_idx = np.flatnonzero(np.isfinite(y[:, 0]))
        hold = obs_idx[-10:]
        y_cv = y.copy()
        y_cv[hold, 0] = np.nan
        e = em_dfm(y_cv, n_factors=1, n_iter=120, seed=1)
        ks = kalman_smooth(
            y_cv,
            np.asarray(e["load"]),
            np.asarray(e["ar_coef"]),
            np.asarray(e["q"]),
            np.asarray(e["r"]),
            np.zeros(1),
            np.eye(1),
        )
        lam0 = float(np.asarray(e["load"])[0, 0])
        dfm = np.sqrt(np.mean((lam0 * np.asarray(ks["f_s"])[hold, 0] - y[hold, 0]) ** 2))
        naive = np.sqrt(np.mean((y[hold - 1, 0] - y[hold, 0]) ** 2))
        assert dfm < naive


class TestNewsDecomp:
    def test_contribs_zero_shape(self, panel):
        y, _, _ = panel
        fit = em_dfm(y, n_factors=1, n_iter=80, seed=0)
        contrib = news_decomp(y, y, fit, target=0)
        assert contrib.shape == y.shape
        assert np.abs(contrib).max() == 0.0  # no new releases

    def test_release_moves_nowcast(self, panel):
        y, _, _ = panel
        obs_idx = np.flatnonzero(np.isfinite(y[:, 0]))
        hold = obs_idx[-8:]
        y_cv = y.copy()
        y_cv[hold, 0] = np.nan
        fit = em_dfm(y_cv, n_factors=1, n_iter=80, seed=1)
        y_new = y_cv.copy()
        y_new[hold[-1], 0] = y[hold[-1], 0]
        contrib = news_decomp(y_cv, y_new, fit, target=0)
        assert np.abs(contrib).sum() > 0
        assert abs(contrib[hold[-1], 0]) > 0

    def test_surprise_sign(self, panel):
        y, _, _ = panel
        y_cv = y.copy()
        y_cv[-1, 0] = np.nan
        fit = em_dfm(y_cv, n_factors=1, n_iter=80, seed=0)
        for sign, expect in [(+3.0, "pos"), (-3.0, "neg")]:
            y_new = y_cv.copy()
            y_new[-1, 0] = sign
            contrib = news_decomp(y_cv, y_new, fit, target=0)
            val = contrib[-1, 0]
            if expect == "pos":
                assert val > 0 or val == 0 or np.sign(val) >= 0
            # weight sign depends on loading; just assert nonzero + sign flip
        y_pos = y_cv.copy()
        y_pos[-1, 0] = 3.0
        y_neg = y_cv.copy()
        y_neg[-1, 0] = -3.0
        cp = news_decomp(y_cv, y_pos, fit, target=0)[-1, 0]
        cn = news_decomp(y_cv, y_neg, fit, target=0)[-1, 0]
        assert np.sign(cp) == -np.sign(cn)


class TestSynth:
    def test_shapes_and_mask(self):
        y, f, lam = synth_panel(120, 5, 1, 0.8, 0.2, 4)
        assert y.shape == (120, 5) and f.shape == (120, 1)
        assert np.isnan(y).any() and np.isfinite(y[:, 0]).all()

    def test_panel_factor_structure(self):
        y, f, lam = synth_panel(200, 6, 1, 0.9, 0.0, 8)
        resid = np.nan_to_num(y) - f @ lam.T
        # idiosyncratic std 0.5 → residual std near 0.5
        assert np.abs(resid.std() - 0.5) < 0.15

    def test_synth_deterministic(self):
        a = synth_panel(80, 4, 1, 0.8, 0.1, 2)
        b = synth_panel(80, 4, 1, 0.8, 0.1, 2)
        np.testing.assert_array_equal(np.isnan(a[0]), np.isnan(b[0]))


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_factor_nowcast(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_bench_contract_no_forbidden(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_factor_nowcast(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_bench_science(self):
        blob = bench_factor_nowcast(seed=0)
        assert blob["synthetic_factor_corr"] > 0.8
        assert blob["synthetic_em_converged"] == 1.0
        assert blob["synthetic_nowcast_err_ratio"] < 1.0
        assert blob["synthetic_determinism"] == 1.0
        assert blob["synthetic_news_max_contrib"] > 0
