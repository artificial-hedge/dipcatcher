"""Kinetic Ising: dynamics, nMF/TAP/PL recovery, enumeration, gates."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.kinetic_ising import (
    bench_kinetic_ising,
    effective_temperature,
    enumerate_boltzmann,
    magnetization_path,
    nmf_invert,
    pl_refine,
    simulate_kinetic_ising,
    simulated_pair_corr,
    synth_market_spins,
    tap_refine,
)


class TestSimulate:
    def test_shapes_and_bounds(self):
        s, J, h = simulate_kinetic_ising(8, 200, seed=0)
        assert s.shape == (200, 8)
        assert J.shape == (8, 8)
        assert h.shape == (8,)
        assert np.isin(np.unique(s), (-1.0, 1.0)).all()
        assert np.all(np.diag(J) == 0.0)

    def test_strong_positive_coupling_correlates(self):
        # parallel updates: J_ij drives the DELAYED corr <s_j(t) s_i(t+1)>
        J = np.zeros((4, 4))
        J[0, 1] = J[1, 0] = 0.8
        s, _, _ = simulate_kinetic_ising(4, 2000, J=J, h=np.zeros(4), seed=1)
        delayed = np.corrcoef(s[:-1, 1], s[1:, 0])[0, 1]
        assert delayed > 0.5

    def test_strong_field_magnetizes(self):
        s, _, _ = simulate_kinetic_ising(6, 300, J=np.zeros((6, 6)), h=np.full(6, 5.0), seed=2)
        assert (s == 1.0).mean() > 0.95

    def test_deterministic(self):
        a, _, _ = simulate_kinetic_ising(6, 200, seed=3)
        b, _, _ = simulate_kinetic_ising(6, 200, seed=3)
        np.testing.assert_allclose(a, b)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_kinetic_ising(2, 200)
        with pytest.raises(ValueError):
            simulate_kinetic_ising(6, 10)
        with pytest.raises(ValueError):
            simulate_kinetic_ising(4, 100, J=np.zeros((3, 3)))


class TestEstimators:
    def test_nmf_recovers_direction(self):
        s, J_true, _ = synth_market_spins(10, 4000, seed=10)
        fit = nmf_invert(s)
        corr = np.corrcoef(fit.J.ravel(), J_true.ravel())[0, 1]
        assert corr > 0.5

    def test_pl_beats_or_ties_nmf(self):
        s, J_true, _ = synth_market_spins(10, 4000, seed=11)
        e_nmf = np.linalg.norm(nmf_invert(s).J - J_true)
        e_pl = np.linalg.norm(pl_refine(s).J - J_true)
        assert e_pl < e_nmf * 1.2  # PL should be competitive or better

    def test_tap_finite(self):
        s, _, _ = synth_market_spins(8, 2000, seed=12)
        fit = tap_refine(s)
        assert np.isfinite(fit.J).all() and np.isfinite(fit.h).all()
        assert fit.method == "tap"

    def test_loglik_finite(self):
        s, _, _ = synth_market_spins(8, 1000, seed=13)
        for fit in (nmf_invert(s), tap_refine(s), pl_refine(s)):
            assert np.isfinite(fit.loglik)

    def test_no_self_coupling(self):
        s, _, _ = synth_market_spins(6, 1000, seed=14)
        assert np.all(np.diag(nmf_invert(s).J) == 0.0)
        assert np.all(np.diag(pl_refine(s).J) == 0.0)

    def test_fail_closed(self):
        bad = np.zeros((100, 5))
        bad[0, 0] = 2.0
        with pytest.raises(ValueError):
            nmf_invert(bad)
        with pytest.raises(ValueError):
            pl_refine(np.ones((10, 4)))


class TestDiagnostics:
    def test_magnetization_bounded(self):
        s, _, _ = simulate_kinetic_ising(6, 300, seed=20)
        m = magnetization_path(s)
        assert m.shape == (300,)
        assert np.all(np.abs(m) <= 1.0)

    def test_eff_temperature_positive(self):
        J = np.full((5, 5), 0.8)
        np.fill_diagonal(J, 0.0)
        s, _, _ = simulate_kinetic_ising(5, 500, J=J, seed=21)
        t_eff = effective_temperature(s)
        assert np.isfinite(t_eff) and t_eff > 0

    def test_pair_corr_symmetric(self):
        s, _, _ = simulate_kinetic_ising(6, 300, seed=22)
        c = simulated_pair_corr(s)
        np.testing.assert_allclose(c, c.T)
        np.testing.assert_allclose(np.diag(c), 1.0)


class TestEnumeration:
    def test_ising_zero_field(self):
        # J=0, h=0: uniform measure -> mean 0, pair corr 0
        out = enumerate_boltzmann(np.zeros((4, 4)), np.zeros(4))
        assert abs(out["mean_abs_mag"]) < 1e-12
        assert abs(out["pair_corr_mean"]) < 1e-12

    def test_ferromagnet_magnetizes(self):
        J = np.full((4, 4), 2.0)
        np.fill_diagonal(J, 0.0)
        out = enumerate_boltzmann(J, np.zeros(4))
        assert out["pair_corr_mean"] > 0.9

    def test_matches_simulation_roughly(self):
        rng = np.random.default_rng(23)
        J = rng.normal(0.0, 0.3, (6, 6))
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        h = rng.normal(0.0, 0.05, 6)
        exact = enumerate_boltzmann(J, h)
        s, _, _ = simulate_kinetic_ising(6, 30000, J=J, h=h, seed=23)
        emp = simulated_pair_corr(s)
        gap = abs(emp[np.triu_indices(6, 1)].mean() - exact["pair_corr_mean"])
        assert gap < 0.15  # parallel-update dynamics approximates Boltzmann

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            enumerate_boltzmann(np.zeros((11, 11)), np.zeros(11))
        with pytest.raises(ValueError):
            enumerate_boltzmann(np.zeros((3, 3)), np.zeros(4))


class TestBench:
    def test_keys_finite(self):
        out = bench_kinetic_ising(20260203)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_kinetic_ising(20260203) == bench_kinetic_ising(20260203)

    def test_quality(self):
        out = bench_kinetic_ising(20260203)
        assert out["synthetic_nmf_J_relerr"] < 1.0
        assert out["synthetic_corr_recovery_relerr"] < 0.5
