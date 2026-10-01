"""Fourier-Hermite: basis, coefficients, density, pricing, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.fourier_hermite import (
    bench_fourier_hermite,
    he_prob,
    hermite_call_price,
    hermite_coefficients,
    hermite_density,
    hermite_moments,
    hermite_put_price,
    repair_density,
    synth_mixture_samples,
)


class TestBasis:
    def test_values(self):
        z = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(he_prob(0, z), [1, 1, 1])
        np.testing.assert_allclose(he_prob(1, z), z)
        np.testing.assert_allclose(he_prob(2, z), z**2 - 1)
        np.testing.assert_allclose(he_prob(3, z), z**3 - 3 * z)
        np.testing.assert_allclose(he_prob(4, z), z**4 - 6 * z**2 + 3)

    def test_recurrence(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal(50)
        for n in range(1, 8):
            np.testing.assert_allclose(
                he_prob(n + 1, z), z * he_prob(n, z) - n * he_prob(n - 1, z), rtol=1e-10
            )

    def test_orthogonality(self):
        z = np.linspace(-6, 6, 4000)
        w = norm.pdf(z)
        for m, n in ((3, 4), (5, 2)):
            ip = np.trapezoid(he_prob(m, z) * he_prob(n, z) * w, z)
            assert abs(ip) < 1e-6
        ip33 = np.trapezoid(he_prob(3, z) ** 2 * w, z)
        assert abs(ip33 - 6.0) < 0.05  # E[He_3^2] = 3!

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            he_prob(-1, np.array([0.0]))
        with pytest.raises(ValueError):
            he_prob(30, np.array([0.0]))


class TestCoefficients:
    def test_gaussian(self):
        rng = np.random.default_rng(1)
        a = hermite_coefficients(rng.standard_normal(5000), order=6)
        assert abs(a[0] - 1.0) < 0.05
        # He_n sampling noise grows with n! — loose absolute band
        for n in range(1, 7):
            assert abs(a[n]) < 0.4

    def test_skewed_captures_a3(self):
        rng = np.random.default_rng(2)
        x = rng.gamma(2.0, 1.0, 8000)
        a = hermite_coefficients(x, order=6)
        assert a[3] > 0.5  # right-skewed

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            hermite_coefficients(np.arange(10.0), order=4)
        with pytest.raises(ValueError):
            hermite_coefficients(np.ones(100), order=4)  # sd=0
        with pytest.raises(ValueError):
            hermite_coefficients(np.random.default_rng(0).standard_normal(100), order=2)


class TestDensity:
    def test_gaussian_recovers(self):
        a = np.zeros(5)
        a[0] = 1.0
        z = np.linspace(-4, 4, 200)
        d = hermite_density(z, a)
        np.testing.assert_allclose(d, norm.pdf(z), atol=1e-10)

    def test_mass_conserved_after_repair(self):
        x = synth_mixture_samples(3000, seed=3)
        a = hermite_coefficients(x, order=6)
        zg = np.linspace(-6, 6, 800)
        d = hermite_density(zg, a)
        rep, _neg = repair_density(zg, d)
        assert abs(np.trapezoid(rep, zg) - 1.0) < 1e-6

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            repair_density(np.arange(5.0), np.ones(4))
        with pytest.raises(ValueError):
            repair_density(np.arange(5.0), -np.ones(5))
        with pytest.raises(ValueError):
            hermite_density(np.arange(3.0), np.ones(4), sigma=0.0)


class TestMoments:
    def test_skew_exkurt(self):
        x = synth_mixture_samples(6000, seed=4)
        a = hermite_coefficients(x, order=6)
        m = hermite_moments(a)
        z = (x - x.mean()) / x.std()
        assert abs(m["skew"] - float(np.mean(z**3))) < 0.1
        assert abs(m["exkurt"] - float(np.mean(z**4) - 3.0)) < 0.5

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            hermite_moments(np.ones(4))


class TestPricing:
    def test_gaussian_matches_black_std_form(self):
        # a=0 beyond a0: E[(Z-k)+] = phi(k) - k*Phi_bar(k)
        a = np.zeros(6)
        a[0] = 1.0
        for k in (-1.0, 0.0, 1.5):
            c = hermite_call_price(a, k)
            ref = float(norm.pdf(k) - k * norm.sf(k))
            assert abs(c - ref) < 1e-8

    def test_put_call_parity(self):
        x = synth_mixture_samples(5000, seed=5)
        a = hermite_coefficients(x, order=8)
        mu, sd = float(x.mean()), float(x.std())
        for k in (0.0, 1.0):
            c = hermite_call_price(a, k, mu=mu, sigma=sd)
            p = hermite_put_price(a, k, mu=mu, sigma=sd)
            assert abs((c - p) - (mu - k)) < 1e-8

    def test_mixture_price_accuracy(self):
        rng = np.random.default_rng(6)
        n = 30000
        pick = rng.random(n) < 0.65
        x = np.where(pick, rng.normal(0.0, 1.0, n), rng.normal(1.8, 1.4, n))
        a = hermite_coefficients(x, order=8)
        mu, sd = float(x.mean()), float(x.std())
        for k in (-0.5, 0.5, 1.5):
            emp = float(np.mean(np.maximum(x - k, 0.0)))
            est = hermite_call_price(a, k, mu=mu, sigma=sd)
            assert abs(est - emp) / emp < 0.15

    def test_fail_closed(self):
        a = np.zeros(4)
        a[0] = 1.0
        with pytest.raises(ValueError):
            hermite_call_price(a, 0.0, sigma=-1.0)


class TestBench:
    def test_keys_finite(self):
        out = bench_fourier_hermite(20260201)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_fourier_hermite(20260201) == bench_fourier_hermite(20260201)

    def test_quality(self):
        out = bench_fourier_hermite(20260201)
        assert out["synthetic_skew_err"] < 0.2
        assert out["synthetic_call_relerr_mean"] < 0.25
        assert out["synthetic_determinism"] == 1.0
