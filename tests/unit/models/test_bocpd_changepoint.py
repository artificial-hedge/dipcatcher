"""Tests for bocpd_changepoint (Adams & MacKay 2007, arXiv:0710.3742)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.bocpd_changepoint import (
    BocpdResult,
    NormalGammaLearner,
    bench_bocpd_changepoint,
    changepoint_maxima,
    constant_hazard,
    detected_change_points,
    expected_run_length,
    run_bocpd,
    synthetic_multi_regime,
)


class TestHazard:
    def test_constant_hazard(self):
        h = constant_hazard(10, 50.0)
        assert h.shape == (10,) and np.allclose(h, 0.02)

    def test_hazard_validates(self):
        with pytest.raises(ValueError):
            constant_hazard(1, 50.0)
        with pytest.raises(ValueError):
            constant_hazard(10, -1.0)


class TestLearner:
    def test_prior_predictive_is_student_t(self):
        lr = NormalGammaLearner(mu0=0.0, kappa0=2.0, alpha0=3.0, beta0=4.0)
        scale = math.sqrt(4.0 * 3.0 / (3.0 * 2.0))
        assert lr.predict(0.0)[0] == pytest.approx(stats.t.pdf(0.0, 6.0, 0.0, scale))

    def test_update_grows_and_matches_conjugate(self):
        lr = NormalGammaLearner(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        lr.update(2.0)
        # after one obs: kappa 2, mu = (1*0 + 2)/2 = 1, alpha 1.5,
        # beta = 1 + 0.5*1*(2-0)^2/2 = 2
        assert lr.kappa[1] == 2.0 and lr.mu[1] == 1.0
        assert lr.alpha[1] == 1.5 and lr.beta[1] == 2.0
        assert lr.mu[0] == 0.0  # slot 0 stays the prior

    def test_invalid_hyperparams_raise(self):
        with pytest.raises(ValueError):
            NormalGammaLearner(kappa0=0.0)
        with pytest.raises(ValueError):
            NormalGammaLearner(alpha0=-1.0)
        with pytest.raises(ValueError):
            NormalGammaLearner(beta0=0.0)


class TestRunBocpd:
    def test_posterior_rows_normalized(self):
        x, _ = synthetic_multi_regime(3, n_seg=3, seg_len=20)
        res = run_bocpd(x, constant_hazard(200, 100))
        rows = res.run_length_posterior
        tot = rows.sum(axis=1)
        np.testing.assert_allclose(tot, 1.0, atol=1e-6)

    def test_result_shape_and_types(self):
        x, _ = synthetic_multi_regime(4, n_seg=2, seg_len=25)
        res = run_bocpd(x, constant_hazard(200, 80), max_run=150)
        assert isinstance(res, BocpdResult)
        assert res.n_obs == x.size
        assert res.run_length_posterior.shape == (x.size, 150)
        assert np.all(res.map_run_length >= 0)
        assert np.all(res.change_posterior >= 0)

    def test_expected_run_length_grows_within_segment(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0.0, 0.3, 60)
        res = run_bocpd(x, constant_hazard(200, 500))
        erl = expected_run_length(res)
        # deep inside one regime the expected run length should ramp up
        assert erl[-1] > erl[5]

    def test_reset_at_planted_change(self):
        x, cps = synthetic_multi_regime(5, n_seg=3, seg_len=40)
        res = run_bocpd(x, constant_hazard(200, 120))
        found = detected_change_points(res, min_gap=10)
        # each true cp detected within 8 steps
        hits = sum(any(abs(f - c) <= 8 for f in found) for c in cps)
        assert hits >= cps.size - 1

    def test_fail_closed_inputs(self):
        with pytest.raises(ValueError):
            run_bocpd(np.array([1.0, 2.0]), constant_hazard(10, 5))
        with pytest.raises(ValueError):
            run_bocpd(np.array([np.nan] * 10), constant_hazard(10, 5))
        with pytest.raises(ValueError):
            run_bocpd(np.arange(20.0), np.array([0.5, 2.0]))  # hazard >1

    def test_determinism(self):
        x, _ = synthetic_multi_regime(9, n_seg=2, seg_len=20)
        a = run_bocpd(x, constant_hazard(100, 80))
        b = run_bocpd(x, constant_hazard(100, 80))
        np.testing.assert_array_equal(a.run_length_posterior, b.run_length_posterior)


class TestChangePointExtraction:
    def test_maxima_spacing(self):
        x, cps = synthetic_multi_regime(6, n_seg=4, seg_len=40)
        res = run_bocpd(x, constant_hazard(200, 120))
        picks = changepoint_maxima(res, min_gap=10, threshold=0.01)
        assert all(b - a >= 10 for a, b in zip(picks[:-1], picks[1:], strict=True))
        assert len(picks) >= 2

    def test_detected_points_near_truth(self):
        x, cps = synthetic_multi_regime(8, n_seg=3, seg_len=40)
        res = run_bocpd(x, constant_hazard(200, 120))
        found = detected_change_points(res)
        assert found and min(abs(f - c) for f in found for c in cps) <= 8


class TestSyntheticFixture:
    def test_changepoint_locations(self):
        x, cps = synthetic_multi_regime(2, n_seg=4, seg_len=30)
        assert x.size == 120
        np.testing.assert_array_equal(cps, [30.0, 60.0, 90.0])

    def test_fixture_validates(self):
        with pytest.raises(ValueError):
            synthetic_multi_regime(1, n_seg=1)
        with pytest.raises(ValueError):
            synthetic_multi_regime(1, seg_len=5)


class TestBench:
    @pytest.fixture(scope="class")
    def blob(self):
        return bench_bocpd_changepoint(seed=42)

    def test_detection_quality(self, blob):
        assert blob["synthetic_precision"] >= 0.5
        assert blob["synthetic_recall"] >= 0.5
        assert blob["synthetic_detection_delay_mean"] < 10.0

    def test_predictive_ordering(self, blob):
        # oracle (knows boundaries) >= bocpd > no-change filter
        assert blob["synthetic_oracle_gap"] > 0.0
        assert blob["synthetic_beats_nochange"] == 1.0
        assert blob["synthetic_logpred_bocpd"] > blob["synthetic_logpred_nochange"]

    def test_determinism(self):
        assert bench_bocpd_changepoint(seed=9) == bench_bocpd_changepoint(seed=9)
