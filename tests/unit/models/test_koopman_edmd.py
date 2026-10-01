"""EDMD/DMD machinery: spectrum recovery, forecast, drift, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.koopman_edmd import (
    auc_from_scores,
    bench_koopman_edmd,
    delay_embed,
    edmd_fit,
    eigenfunction_drift,
    hankel_dmd,
    koopman_forecast,
    observables,
    synth_linear_oscillator,
    synth_logistic_map,
    synth_regime_switching_ar,
    synth_van_der_pol,
)


class TestDelayEmbed:
    def test_shape_and_content(self):
        x = np.arange(10.0)
        z = delay_embed(x, dim=3)
        assert z.shape == (8, 3)
        np.testing.assert_allclose(z[0], [2.0, 1.0, 0.0])
        np.testing.assert_allclose(z[1], [3.0, 2.0, 1.0])

    def test_delay_stride(self):
        x = np.arange(20.0)
        z = delay_embed(x, dim=3, delay=2)
        np.testing.assert_allclose(z[0], [4.0, 2.0, 0.0])

    def test_deterministic(self):
        x = np.random.default_rng(0).standard_normal(50)
        np.testing.assert_allclose(delay_embed(x, 4), delay_embed(x, 4))

    def test_fail_closed_too_short(self):
        with pytest.raises(ValueError):
            delay_embed(np.arange(4.0), dim=8)

    def test_fail_closed_bad_params(self):
        x = np.arange(20.0)
        with pytest.raises(ValueError):
            delay_embed(x, dim=1)
        with pytest.raises(ValueError):
            delay_embed(x, dim=3, delay=0)


class TestObservables:
    def test_delays_identity(self):
        z = np.arange(12.0).reshape(4, 3)
        np.testing.assert_allclose(observables(z, "delays"), z)

    def test_poly_adds_features(self):
        z = np.array([[1.0, 2.0]])
        out = observables(z, "poly", degree=3)
        assert out.shape[1] == 4  # 2 delays + x^2 + x^3
        np.testing.assert_allclose(out[0, 2:], [4.0, 8.0])

    def test_fourier_adds_sincos(self):
        z = np.array([[0.5, 1.0]])
        out = observables(z, "fourier")
        assert out.shape[1] == 4
        np.testing.assert_allclose(out[0, 2:], np.sin(1.0) and [np.sin(1.0), np.cos(1.0)])

    def test_unknown_dictionary(self):
        with pytest.raises(ValueError):
            observables(np.ones((2, 2)), "nope")


class TestEDMDLinearSystem:
    def test_operator_shape(self):
        x = synth_linear_oscillator(500)
        res = hankel_dmd(x, dim=8)
        assert res.operator.shape == (8, 8)
        assert res.eigenvalues.shape == (8,)

    def test_eigenvalue_modulus_decay(self):
        decay = 0.01
        x = synth_linear_oscillator(2000, decay=decay, seed=1)
        res = hankel_dmd(x, dim=12)
        top = np.max(np.abs(res.eigenvalues))
        assert abs(top - np.exp(-decay)) < 0.05

    def test_eigenvalue_angle_frequency(self):
        x = synth_linear_oscillator(2000, omega=0.4, decay=0.0, seed=2)
        res = hankel_dmd(x, dim=12)
        angs = np.abs(np.angle(res.eigenvalues))
        cand = angs[angs > 0.05]
        assert cand.size > 0
        assert abs(np.min(cand) - 0.4) < 0.05

    def test_residual_small_on_linear(self):
        x = synth_linear_oscillator(800, seed=3)
        res = hankel_dmd(x, dim=10)
        assert res.residual < 0.2

    def test_implied_timescales_finite_positive(self):
        x = synth_linear_oscillator(500, decay=0.02, seed=4)
        res = hankel_dmd(x, dim=8)
        tau = res.implied_timescales
        assert tau.shape == (8,)
        assert np.any(np.isfinite(tau))

    def test_fail_closed_degenerate(self):
        with pytest.raises(ValueError):
            edmd_fit(np.ones((2, 3)))
        with pytest.raises(ValueError):
            edmd_fit(np.full((20, 4), np.nan))


class TestEDMDvsDMD:
    def test_agreement_delay_observables(self):
        x = synth_logistic_map(400, seed=5)
        z = delay_embed(x, dim=6)
        r1 = edmd_fit(z[:300], dictionary="delays")
        r2 = hankel_dmd(x[:306], dim=6)
        d = np.linalg.norm(np.sort(np.abs(r1.eigenvalues)) - np.sort(np.abs(r2.eigenvalues)))
        assert d < 0.5


class TestForecast:
    def test_forecast_shape(self):
        x = synth_linear_oscillator(400)
        z = delay_embed(x, dim=6)
        res = edmd_fit(z)
        fc = koopman_forecast(res, z[-1], steps=10)
        assert fc.shape == (10, 6)

    def test_forecast_beats_persistence_linear(self):
        x = synth_linear_oscillator(1000, decay=0.005, seed=6)
        z = delay_embed(x, dim=10)
        res = edmd_fit(z[:800])
        fc = koopman_forecast(res, z[799], steps=30)
        true = z[800:830, 0]
        err_model = np.sum((true - fc[:, 0]) ** 2)
        err_pers = np.sum((true - z[799, 0]) ** 2)
        assert err_model < err_pers

    def test_forecast_bad_steps(self):
        x = synth_linear_oscillator(300)
        res = hankel_dmd(x, dim=6)
        z = delay_embed(x, dim=6)
        with pytest.raises(ValueError):
            koopman_forecast(res, z[-1], steps=0)


class TestDriftDetection:
    def test_switching_exceeds_constant(self):
        sw = synth_regime_switching_ar(3000, seed=7)
        base = synth_regime_switching_ar(3000, p=0.0, seed=8)
        d_sw = eigenfunction_drift(sw, window=300, step=150)
        d_base = eigenfunction_drift(base, window=300, step=150)
        assert d_sw.mean() > d_base.mean()

    def test_auc_above_chance(self):
        sw = synth_regime_switching_ar(4000, seed=9)
        base = synth_regime_switching_ar(4000, p=0.0, seed=10)
        d_sw = eigenfunction_drift(sw, window=400, step=200)
        d_base = eigenfunction_drift(base, window=400, step=200)
        pos = d_sw[d_sw >= np.quantile(d_sw, 0.75)]
        auc = auc_from_scores(pos, d_base)
        assert auc > 0.5

    def test_fail_closed_small_window(self):
        with pytest.raises(ValueError):
            eigenfunction_drift(np.random.default_rng(0).standard_normal(100), window=8)


class TestGenerators:
    def test_logistic_bounded(self):
        x = synth_logistic_map(200, seed=11)
        assert np.all((x >= 0) & (x <= 1))

    def test_vdp_bounded_oscillation(self):
        x = synth_van_der_pol(2000, seed=12)
        assert np.abs(x).max() < 10.0
        assert np.std(x) > 0.5  # genuinely oscillating

    def test_regime_switching_differs(self):
        a = synth_regime_switching_ar(500, p=0.1, seed=13)
        b = synth_regime_switching_ar(500, p=0.0, seed=13)
        assert not np.allclose(a, b)

    def test_generators_deterministic(self):
        np.testing.assert_allclose(synth_logistic_map(50, seed=14), synth_logistic_map(50, seed=14))
        np.testing.assert_allclose(synth_van_der_pol(50, seed=14), synth_van_der_pol(50, seed=14))


class TestAUC:
    def test_perfect_separation(self):
        assert auc_from_scores(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0

    def test_inverted(self):
        assert auc_from_scores(np.array([0.0]), np.array([1.0])) == 0.0

    def test_ties(self):
        assert auc_from_scores(np.array([1.0]), np.array([1.0])) == 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            auc_from_scores(np.array([]), np.array([1.0]))


class TestBench:
    def test_bench_keys_finite(self):
        out = bench_koopman_edmd(20260131)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_bench_deterministic(self):
        a = bench_koopman_edmd(20260131)
        b = bench_koopman_edmd(20260131)
        assert a == b

    def test_bench_quality_thresholds(self):
        out = bench_koopman_edmd(20260131)
        assert out["synthetic_eig_modulus_err"] < 0.1
        assert out["synthetic_determinism_delta"] == 0.0
        assert out["synthetic_forecast_beats_persistence"] == 1.0
