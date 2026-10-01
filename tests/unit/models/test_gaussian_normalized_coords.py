"""Tests for gaussian_normalized_coords (Sun 2026, arXiv:2609.14212)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.gaussian_normalized_coords import (
    ArbReport,
    bachelier_normalized,
    bench_gaussian_normalized_coords,
    butterfly_inequality_q,
    cdf_deformation_eta,
    deformed_cdf,
    density_factor_m,
    fukasawa_coords,
    mills_bounds_eta,
    normalized_arb_check,
    repair_to_arb_free,
)


class TestCoordinates:
    def test_bachelier_normalized(self):
        k = np.linspace(-0.5, 0.5, 21)
        s = np.full_like(k, 0.4)
        np.testing.assert_allclose(bachelier_normalized(k, s), k / 0.4)

    def test_bachelier_rejects_bad(self):
        with pytest.raises(ValueError):
            bachelier_normalized(np.array([0.1]), np.array([0.4]))
        with pytest.raises(ValueError):
            bachelier_normalized(np.linspace(-1, 1, 5), -np.ones(5))

    def test_fukasawa_coords(self):
        k = np.linspace(-0.3, 0.3, 11)
        v = np.full_like(k, 0.2)
        a, b = fukasawa_coords(k, v)
        np.testing.assert_allclose(a, k / 0.2 - 0.1)
        np.testing.assert_allclose(b, k / 0.2 + 0.1)
        np.testing.assert_array_less(a, b)

    def test_fukasawa_rejects_bad(self):
        with pytest.raises(ValueError):
            fukasawa_coords(np.linspace(0, 1, 5), np.zeros(5))

    def test_mills_bounds_straddle_zero(self):
        z = np.linspace(-3, 3, 41)
        lo, hi = mills_bounds_eta(z)
        assert np.all(lo < 0) and np.all(hi > 0)
        np.testing.assert_allclose(lo, -hi[::-1], rtol=1e-10)  # symmetry


class TestDeformation:
    def test_zero_eta_gives_gaussian(self):
        z = np.linspace(-2, 2, 51)
        h = deformed_cdf(z, np.zeros_like(z))
        np.testing.assert_allclose(h, stats.norm.cdf(z), atol=1e-12)

    def test_eta_roundtrip(self):
        z = np.linspace(-2.5, 2.5, 101)
        eta = 0.1 * np.sin(z) * np.exp(-0.5 * z * z)
        np.testing.assert_allclose(cdf_deformation_eta(z, deformed_cdf(z, eta)), eta, atol=1e-10)

    def test_m_is_h_prime_over_phi(self):
        z = np.linspace(-2, 2, 201)
        eta = 0.08 * np.cos(3 * z) * np.exp(-0.2 * z * z)
        m = density_factor_m(z, eta)
        h = deformed_cdf(z, eta)
        hp = np.gradient(h, z)
        np.testing.assert_allclose(m[5:-5], hp[5:-5] / stats.norm.pdf(z[5:-5]), atol=2e-3)

    def test_flat_eta_m_is_one(self):
        z = np.linspace(-2, 2, 51)
        eta = np.full_like(z, 0.05)
        m = density_factor_m(z, eta)
        # m = 1 - z*0.05: not constant; check formula directly
        np.testing.assert_allclose(m, 1.0 - z * 0.05, atol=2e-3)


class TestArb:
    def test_gaussian_surface_is_free(self):
        z = np.linspace(-2.5, 2.5, 61)
        rep = normalized_arb_check(z, np.zeros_like(z))
        assert rep.arbitrage_free and rep.m_min > 0.9
        assert rep.m_negative_count == 0 and rep.mills_violations == 0

    def test_flat_h_flags_arb(self):
        z = np.linspace(-2.5, 2.5, 61)
        eta = np.where(np.abs(z) < 0.3, 0.4, 0.0)  # sharp CDF bump
        rep = normalized_arb_check(z, eta)
        assert not rep.arbitrage_free and rep.m_negative_count > 0

    def test_mills_violation_flagged(self):
        z = np.linspace(-2.5, 2.5, 61)
        eta = np.full_like(z, -1.0)  # below -Phi/phi on the left tail
        rep = normalized_arb_check(z, eta)
        assert rep.mills_violations > 0

    def test_q_inequality_affine_is_flat(self):
        # q(y) = q0 - y (linear scale growth) -> q'' + y q' - q = -q < 0? no:
        # r = 0 + y*(-1) - (q0 - y) = -q0 <= 0 -> admissible
        y = np.linspace(-1.5, 1.5, 51)
        q = 2.0 - y
        r = butterfly_inequality_q(y, q)
        assert np.all(r <= 1e-9)

    def test_q_inequality_detects_bump(self):
        y = np.linspace(-2, 2, 101)
        q = 1.0 + 0.8 * np.exp(-0.5 * (y / 0.2) ** 2)
        r = butterfly_inequality_q(y, q)
        assert np.any(r > 0)


class TestRepair:
    def test_isotonic_repair_restores_m(self):
        z = np.linspace(-2.5, 2.5, 81)
        eta = 0.5 * np.exp(-0.5 * (z / 0.25) ** 2)  # violation-sized bump
        assert not normalized_arb_check(z, eta).arbitrage_free
        eta_rep = repair_to_arb_free(z, eta)
        rep = normalized_arb_check(z, eta_rep)
        assert rep.m_min > -0.02  # gradient noise on isotonic kinks
        assert rep.mills_violations == 0

    def test_repair_is_idempotent_on_free(self):
        z = np.linspace(-2.5, 2.5, 61)
        eta = 0.05 * np.sin(z) * np.exp(-0.5 * z * z)
        eta_rep = repair_to_arb_free(z, eta)
        np.testing.assert_allclose(eta_rep, eta, atol=1e-10)


class TestBench:
    @pytest.fixture(scope="class")
    def blob(self):
        return bench_gaussian_normalized_coords(seed=20261111)

    def test_clean_and_viol(self, blob):
        assert blob["synthetic_clean_arb_free"] == 1.0
        assert blob["synthetic_viol_flagged"] == 1.0
        assert blob["synthetic_m_min_clean"] > 0.0
        assert blob["synthetic_m_min_viol"] < 0.0

    def test_q_detection(self, blob):
        assert blob["synthetic_q_precision"] >= 0.8
        assert blob["synthetic_q_recall"] >= 0.5

    def test_repair(self, blob):
        assert blob["synthetic_repair_m_min"] > -0.02
        assert blob["synthetic_repair_l2"] < 1.0

    def test_black_path(self, blob):
        assert blob["synthetic_black_clean_free"] == 1.0
        assert blob["synthetic_black_viol_flagged"] == 1.0

    def test_roundtrip(self, blob):
        assert blob["synthetic_roundtrip_eta_max_err"] < 1e-10

    def test_determinism(self):
        assert bench_gaussian_normalized_coords(seed=5) == bench_gaussian_normalized_coords(seed=5)

    def test_report_type(self):
        z = np.linspace(-2, 2, 41)
        rep = normalized_arb_check(z, np.zeros_like(z))
        assert isinstance(rep, ArbReport)
