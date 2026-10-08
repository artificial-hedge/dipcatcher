"""Tests for parametric VaR/ES."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quant_fund.metrics.risk_parametric import (
    cornish_fisher_es,
    cornish_fisher_var,
    fit_student_t,
    parametric_var_es,
    student_t_var_es,
)


class TestCornishFisher:
    def test_gaussian_reduces_to_z(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.01, 0.02, 5000)
        out = cornish_fisher_var(x, 0.99)
        # Near-Gaussian data: z_cf ≈ z.
        assert abs(out["z_cf"] - out["z"]) < 0.2
        gauss_var = x.mean() + x.std(ddof=1) * out["z"]
        assert abs(out["var"] - gauss_var) / abs(gauss_var) < 0.15

    def test_heavy_tail_increases_var(self):
        rng = np.random.default_rng(1)
        x = rng.standard_t(3.0, 5000) * 0.02
        cf = cornish_fisher_var(x, 0.99)
        # Excess kurtosis positive -> z_cf > z (for z ~ 2.33 region).
        assert cf["excess_kurtosis"] > 0.5
        assert cf["z_cf"] > cf["z"]

    def test_es_exceeds_var(self):
        rng = np.random.default_rng(2)
        x = rng.standard_t(4.0, 4000) * 0.01
        out = cornish_fisher_es(x, 0.95)
        assert out["es"] > out["var"]


class TestStudentT:
    def test_fit_recovers_df(self):
        rng = np.random.default_rng(3)
        x = rng.standard_t(5.0, 5000) * 0.02
        fit = fit_student_t(x)
        assert 2.5 < fit["nu"] < 15.0
        assert abs(fit["sigma"] - 0.02 * np.sqrt(5.0 / 3.0)) < 0.01

    def test_fit_accepts_valid_lower_bound_solution(self):
        rng = np.random.default_rng(3)
        x = np.r_[np.zeros(5), rng.normal(size=25)]
        fit = fit_student_t(x)
        assert fit["nu"] >= 2.01
        assert np.isfinite(fit["sigma"])

    def test_var_es_t(self):
        rng = np.random.default_rng(4)
        x = rng.standard_t(4.0, 5000) * 0.015
        out = student_t_var_es(x, 0.99)
        assert out["es"] > out["var"] > 0.0
        # Empirical check: ES close to empirical tail mean.
        emp_var = np.quantile(x, 0.99)
        emp_es = x[x >= emp_var].mean()
        assert abs(out["es"] - emp_es) / emp_es < 0.35

    def test_t_var_exceeds_gaussian_on_heavy(self):
        rng = np.random.default_rng(5)
        x = rng.standard_t(3.0, 5000) * 0.01
        t_out = parametric_var_es(x, 0.99, "student_t")
        g_out = parametric_var_es(x, 0.99, "gaussian")
        assert t_out["es"] > g_out["es"]

    @pytest.mark.parametrize(
        ("success", "fun", "jac"),
        [
            (False, 1.0, np.zeros(3)),
            (True, np.nan, np.zeros(3)),
            (True, 1.0, np.array([0.0, np.nan, 0.0])),
        ],
    )
    def test_fit_rejects_failed_or_nonfinite_optimization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        success: bool,
        fun: float,
        jac: np.ndarray,
    ) -> None:
        result = SimpleNamespace(success=success, x=np.array([8.0, 0.0, -1.0]), fun=fun, jac=jac)
        monkeypatch.setattr(
            "quant_fund.metrics.risk_parametric.opt.minimize",
            lambda *args, **kwargs: result,
        )
        with pytest.raises(ValueError, match="did not converge"):
            fit_student_t(np.linspace(-1.0, 1.0, 100))


class TestDispatch:
    def test_dispatch_and_failclosed(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0.0, 0.02, 500)
        for m in ("gaussian", "student_t", "cornish_fisher"):
            out = parametric_var_es(x, 0.95, m)
            assert out["es"] >= out["var"]
        with pytest.raises(ValueError):
            parametric_var_es(x, 0.95, "bogus")
        with pytest.raises(ValueError):
            parametric_var_es(x, 1.5)
        with pytest.raises(ValueError):
            cornish_fisher_var(np.arange(10.0))
        with pytest.raises(ValueError):
            cornish_fisher_var(np.zeros(100))
