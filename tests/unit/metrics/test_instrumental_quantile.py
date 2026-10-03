"""Tests for metrics/instrumental_quantile.py — Chernozhukov–Hansen IVQR."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.instrumental_quantile import (
    anderson_rubin_ivq,
    bench_instrumental_quantile,
    first_stage_f,
    ivqr_ci,
    ivqr_fit,
    ivqr_objective,
    synth_iv_data,
)


@pytest.fixture(scope="module")
def dgp() -> dict[str, np.ndarray]:
    return synth_iv_data(300, 1.0, 1.0, 0.6, seed=0)


@pytest.fixture(scope="module")
def grid() -> np.ndarray:
    return np.linspace(-1.5, 3.5, 41)


class TestGuards:
    def test_bad_inputs(self, dgp):
        with pytest.raises(ValueError):
            ivqr_objective(np.ones(5), dgp["x_endog"], dgp["z_inst"], None, 0.5, 1.0)
        with pytest.raises(ValueError):
            ivqr_objective(dgp["y"], dgp["x_endog"], dgp["z_inst"], None, 1.5, 1.0)
        with pytest.raises(ValueError):
            ivqr_objective(dgp["y"], np.ones((300, 2)), dgp["z_inst"], None, 0.5, 1.0)
        with pytest.raises(ValueError):
            ivqr_objective(dgp["y"], dgp["x_endog"], dgp["z_inst"][:50], None, 0.5, 1.0)
        with pytest.raises(ValueError):
            synth_iv_data(10, 1.0, 1.0)
        with pytest.raises(ValueError):
            synth_iv_data(100, 1.0, 0.0)
        with pytest.raises(ValueError):
            synth_iv_data(100, 1.0, 1.0, 1.5)
        with pytest.raises(ValueError):
            ivqr_fit(dgp["y"], dgp["x_endog"], dgp["z_inst"], None, 0.5, np.array([1.0]))
        with pytest.raises(ValueError):
            ivqr_ci(dgp["y"], dgp["x_endog"], dgp["z_inst"], None, 0.5, level=1.5)


class TestObjective:
    def test_nonnegative_and_minimized_near_truth(self, dgp, grid):
        objs = [
            ivqr_objective(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, b)
            for b in grid
        ]
        objs_arr = np.asarray(objs)
        assert (objs_arr >= 0).all()
        assert grid[int(np.argmin(objs_arr))] == pytest.approx(1.0, abs=0.25)

    def test_true_beta_beats_false(self, dgp):
        o_true = ivqr_objective(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, 1.0)
        o_false = ivqr_objective(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, 3.0)
        assert o_true < o_false


class TestFit:
    def test_recovers_effect_median(self, dgp, grid):
        fit = ivqr_fit(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, grid)
        assert fit["beta"] == pytest.approx(1.0, abs=0.2)
        # naive QR is biased upward under endogeneity
        assert abs(fit["naive_beta"] - 1.0) > abs(fit["beta"] - 1.0)

    def test_deterministic(self, dgp, grid):
        a = ivqr_fit(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, grid)
        b = ivqr_fit(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, grid)
        assert a["beta"] == b["beta"]


class TestAR:
    def test_accepts_true_null(self, dgp):
        ar = anderson_rubin_ivq(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, 1.0)
        assert ar["p_value"] > 0.05
        assert ar["stat"] >= 0

    def test_rejects_false_null(self, dgp):
        ar = anderson_rubin_ivq(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, 3.0)
        assert ar["p_value"] < 0.05

    def test_robust_under_weak_iv(self):
        weak = synth_iv_data(250, 1.0, 0.05, 0.6, seed=1)
        ar = anderson_rubin_ivq(
            weak["y"], weak["x_endog"], weak["z_inst"], weak["x_exog"], 0.5, 1.0
        )
        # AR stays valid: not a spurious rejection under the true null
        assert ar["p_value"] > 0.01


class TestCI:
    def test_covers_truth(self, dgp, grid):
        ci = ivqr_ci(dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"], 0.5, 0.95, grid)
        assert ci["lo"] <= 1.0 <= ci["hi"]
        assert ci["lo"] <= ci["beta"] <= ci["hi"]


class TestFirstStage:
    def test_strong_iv(self, dgp):
        assert first_stage_f(dgp["x_endog"], dgp["z_inst"], dgp["x_exog"]) > 10

    def test_weak_iv_flagged(self):
        weak = synth_iv_data(200, 1.0, 0.08, seed=3)
        assert first_stage_f(weak["x_endog"], weak["z_inst"], weak["x_exog"]) < 10


class TestBench:
    def test_keys_finite(self):
        blob = bench_instrumental_quantile(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_no_forbidden_tokens(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_instrumental_quantile(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_science(self):
        blob = bench_instrumental_quantile(seed=0)
        assert abs(blob["synthetic_ivqr_bias_median"]) < abs(blob["synthetic_naive_qr_bias_median"])
        assert blob["synthetic_ar_size"] < 0.3
        assert blob["synthetic_ar_power"] > 0.8
        assert blob["synthetic_first_stage_f"] > 10
        assert blob["synthetic_weak_iv_flag"] == 1.0
        assert blob["synthetic_ci_covers_truth"] == 1.0
        assert blob["synthetic_determinism"] == 1.0
