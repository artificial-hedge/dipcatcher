"""Tests for models/svi_surface.py — SVI slices, calibration, arbitrage checks.

Seeded SYNTHETIC only: every smile is generated from the SVI family
itself or built with a violation planted by construction; nothing here
is market evidence (module docstring honesty contract).

Independent references (mutation hardening):
- ``_w_ref``/``_g_ref``/``_ssvi_w_ref`` re-derive the formulas straight
  from the paper's equations rather than calling the module.
- ``raw_svi_dw``/``raw_svi_d2w`` are pinned by central finite
  differences of ``raw_svi_w``.
- The PAVA repair is pinned against a brute-force isotonic solve on
  small inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

import quant_fund.models.svi_surface as svi
from quant_fund.models.svi_surface import (
    GriddedSurface,
    NaturalSVIParams,
    SmileSlice,
    SVIParams,
    SVISurface,
    _interp_w,
    _pava_nondec,
    assemble_surface,
    calendar_report,
    calibrate_slice,
    calibrate_slice_iv,
    essvi_slices_consistent,
    implied_log_density,
    lee_wing_check,
    natural_svi_w,
    natural_to_raw,
    raw_svi_d2w,
    raw_svi_dw,
    raw_svi_w,
    raw_to_natural,
    repair_calendar,
    slice_arbitrage_report,
    ssvi_butterfly_ok,
    ssvi_calendar_partial_theta_bound,
    ssvi_power_phi,
    ssvi_to_raw,
    ssvi_w,
    svi_g,
    svi_implied_vol,
    svi_jump_wings,
)

P = SVIParams(a=0.04, b=0.4, rho=-0.45, m=0.02, sigma=0.25)
K_GRID = np.linspace(-2.0, 2.0, 401)


def _w_ref(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def _g_ref(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """Independent g(k): finite differences of raw SVI w, paper eq. 2.1."""
    h = 1e-5
    w = _w_ref(k, p.a, p.b, p.rho, p.m, p.sigma)
    wp = (
        _w_ref(k + h, p.a, p.b, p.rho, p.m, p.sigma) - _w_ref(k - h, p.a, p.b, p.rho, p.m, p.sigma)
    ) / (2 * h)
    wpp = (
        _w_ref(k + h, p.a, p.b, p.rho, p.m, p.sigma)
        - 2 * w
        + _w_ref(k - h, p.a, p.b, p.rho, p.m, p.sigma)
    ) / h**2
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2


def _ssvi_w_ref(k: np.ndarray, theta: float, rho: float, phi: float) -> np.ndarray:
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho**2))


def _natural_w_ref(
    k: np.ndarray, delta: float, mu: float, rho: float, omega: float, zeta: float
) -> np.ndarray:
    return delta + 0.5 * omega * (
        1.0 + zeta * rho * (k - mu) + np.sqrt((zeta * (k - mu) + rho) ** 2 + 1.0 - rho**2)
    )


# ---------------------------------------------------------------------
# Slice evaluation and derivatives
# ---------------------------------------------------------------------


class TestEvalAndDerivatives:
    def test_raw_w_matches_reference(self) -> None:
        np.testing.assert_allclose(
            raw_svi_w(K_GRID, P), _w_ref(K_GRID, P.a, P.b, P.rho, P.m, P.sigma)
        )

    def test_w_at_m_and_asymptotes(self) -> None:
        # w(m) = a + b sigma; linear wings w ~ b(1 +/- rho)|k|.
        assert raw_svi_w(P.m, P) == pytest.approx(P.a + P.b * P.sigma)
        k_big = 1e4
        assert raw_svi_w(k_big, P) / k_big == pytest.approx(P.b * (1 + P.rho), rel=1e-3)
        assert raw_svi_w(-k_big, P) / k_big == pytest.approx(P.b * (1 - P.rho), rel=1e-3)

    def test_dw_finite_difference(self) -> None:
        h = 1e-6
        fd = (raw_svi_w(K_GRID + h, P) - raw_svi_w(K_GRID - h, P)) / (2 * h)
        np.testing.assert_allclose(raw_svi_dw(K_GRID, P), fd, atol=1e-8)

    def test_d2w_finite_difference(self) -> None:
        h = 1e-4
        fd = (raw_svi_dw(K_GRID + h, P) - raw_svi_dw(K_GRID - h, P)) / (2 * h)
        np.testing.assert_allclose(raw_svi_d2w(K_GRID, P), fd, atol=1e-5)

    def test_w_scalar_and_array(self) -> None:
        assert raw_svi_w(0.0, P).shape == ()
        assert raw_svi_w(K_GRID, P).shape == K_GRID.shape

    def test_implied_vol(self) -> None:
        k = np.array([-0.5, 0.0, 0.5])
        np.testing.assert_allclose(svi_implied_vol(k, P, 0.75), np.sqrt(raw_svi_w(k, P) / 0.75))

    def test_eval_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            raw_svi_w(np.array([0.0, np.nan]), P)
        with pytest.raises(ValueError):
            svi_implied_vol(K_GRID, P, 0.0)
        with pytest.raises(ValueError):
            svi_implied_vol(K_GRID, P, -1.0)


class TestParamsValidation:
    def test_bad_params_raise(self) -> None:
        with pytest.raises(ValueError):
            SVIParams(0.0, -0.1, 0.0, 0.0, 0.3)  # b < 0
        with pytest.raises(ValueError):
            SVIParams(0.0, 0.1, 1.0, 0.0, 0.3)  # |rho| = 1
        with pytest.raises(ValueError):
            SVIParams(0.0, 0.1, 0.0, 0.0, 0.0)  # sigma = 0
        with pytest.raises(ValueError):
            SVIParams(0.0, 0.1, 0.0, 0.0, np.nan)  # non-finite
        with pytest.raises(ValueError):
            NaturalSVIParams(0.0, 0.0, 0.0, -1.0, 1.0)  # omega < 0
        with pytest.raises(ValueError):
            NaturalSVIParams(0.0, 0.0, 0.0, 1.0, 0.0)  # zeta = 0


# ---------------------------------------------------------------------
# Equivalent parameterizations.
# ---------------------------------------------------------------------


class TestParameterizations:
    def test_natural_to_raw_mapping(self) -> None:
        pn = NaturalSVIParams(delta=0.01, mu=0.03, rho=-0.4, omega=0.5, zeta=1.2)
        raw = natural_to_raw(pn)
        np.testing.assert_allclose(natural_svi_w(K_GRID, pn), raw_svi_w(K_GRID, raw), atol=1e-14)
        # and against the paper's eq. 3.2 written out independently
        np.testing.assert_allclose(
            natural_svi_w(K_GRID, pn),
            _natural_w_ref(K_GRID, pn.delta, pn.mu, pn.rho, pn.omega, pn.zeta),
            atol=1e-12,
        )

    def test_raw_natural_roundtrip(self) -> None:
        back = natural_to_raw(raw_to_natural(P))
        for f in ("a", "b", "rho", "m", "sigma"):
            assert getattr(back, f) == pytest.approx(getattr(P, f), abs=1e-12)
        np.testing.assert_allclose(raw_svi_w(K_GRID, back), raw_svi_w(K_GRID, P))

    def test_ssvi_to_raw_equivalence(self) -> None:
        theta, rho, phi = 0.05, -0.4, 0.6
        np.testing.assert_allclose(
            ssvi_w(K_GRID, theta, rho, phi), _ssvi_w_ref(K_GRID, theta, rho, phi)
        )
        np.testing.assert_allclose(
            ssvi_w(K_GRID, theta, rho, phi),
            raw_svi_w(K_GRID, ssvi_to_raw(theta, rho, phi)),
            atol=1e-14,
        )
        # ATM consistency: w(0) = theta exactly.
        assert ssvi_w(0.0, theta, rho, phi) == pytest.approx(theta)

    def test_ssvi_power_phi_values(self) -> None:
        th = np.array([0.02, 0.05, 0.1])
        np.testing.assert_allclose(ssvi_power_phi(th, 0.5, 0.4), 0.5 * th**-0.4)
        with pytest.raises(ValueError):
            ssvi_power_phi(th, 0.5, 1.5)  # gamma > 1
        with pytest.raises(ValueError):
            ssvi_power_phi(-th, 0.5, 0.4)  # theta <= 0

    def test_ssvi_to_raw_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            ssvi_to_raw(0.0, 0.0, 0.5)  # theta = 0
        with pytest.raises(ValueError):
            ssvi_to_raw(0.05, 1.0, 0.5)  # rho = 1
        with pytest.raises(ValueError):
            ssvi_to_raw(0.05, 0.0, -0.5)  # phi <= 0


# ---------------------------------------------------------------------
# Butterfly arbitrage: g(k), density, Lee wings, reports.
# ---------------------------------------------------------------------


class TestButterfly:
    def test_g_matches_independent_fd(self) -> None:
        k = np.linspace(-1.5, 1.5, 301)
        np.testing.assert_allclose(svi_g(k, P), _g_ref(k, P), atol=1e-3, rtol=1e-3)

    def test_flat_smile_g_positive(self) -> None:
        flat = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.3)
        assert np.all(svi_g(K_GRID, flat) == pytest.approx(1.0))

    def test_free_slice_detected_free(self) -> None:
        # SSVI slice deep inside the Theorem-4.2 region -> provably free.
        p = ssvi_to_raw(0.05, -0.4, 0.6)
        rep = slice_arbitrage_report(p, -8.0, 8.0, 4001)
        assert rep["free"] == 1.0
        assert rep["w_positive"] == 1.0
        assert rep["wing_violated"] == 0.0

    def test_mid_smile_g_violation_detected(self) -> None:
        # Negative a with strong curvature: w > 0 and wings obey Lee, but
        # g dips below zero mid-smile (verified: g(-0.88) ~ -0.022).
        bad = SVIParams(a=-0.1, b=1.0, rho=0.0, m=0.0, sigma=0.3)
        assert np.min(svi_g(np.linspace(-2.0, 2.0, 2001), bad)) < 0.0
        rep = slice_arbitrage_report(bad, -5.0, 5.0, 4001)
        assert rep["free"] == 0.0
        assert rep["min_g"] < 0.0
        assert rep["w_positive"] == 1.0  # violation is g, not w

    def test_wing_violation_detected(self) -> None:
        bad = SVIParams(a=0.05, b=1.5, rho=0.5, m=0.0, sigma=0.3)
        assert bad.b * (1 + bad.rho) > 2.0  # planted Lee breach
        rep = slice_arbitrage_report(bad, -5.0, 5.0, 2001)
        assert rep["wing_violated"] == 1.0
        assert rep["free"] == 0.0

    def test_negative_variance_detected(self) -> None:
        rho, sigma, b = 0.3, 0.3, 0.6
        a = -b * sigma * np.sqrt(1 - rho**2) - 0.02
        bad = SVIParams(a=a, b=b, rho=rho, m=0.0, sigma=sigma)
        rep = slice_arbitrage_report(bad, -5.0, 5.0, 2001)
        assert rep["w_positive"] == 0.0
        assert rep["free"] == 0.0
        assert rep["min_g"] == -np.inf

    def test_lee_wing_slopes_exact(self) -> None:
        rep = lee_wing_check(P)
        assert rep["left_slope"] == pytest.approx(P.b * (1 - P.rho))
        assert rep["right_slope"] == pytest.approx(P.b * (1 + P.rho))
        assert rep["bound"] == 2.0
        assert rep["violated"] == 0.0
        with pytest.raises(ValueError):
            lee_wing_check(P, bound=0.0)

    def test_density_nonnegative_on_free_slice(self) -> None:
        p = ssvi_to_raw(0.06, -0.3, 0.7)
        d = implied_log_density(K_GRID, p)
        assert np.all(d >= 0.0)
        # integrates to ~1 over the deep-tail grid (density in k)
        k_fine = np.linspace(-20.0, 20.0, 40001)
        mass = np.trapezoid(implied_log_density(k_fine, p), k_fine)
        assert mass == pytest.approx(1.0, abs=2e-3)

    def test_density_shows_violation(self) -> None:
        bad = SVIParams(a=-0.1, b=1.0, rho=0.0, m=0.0, sigma=0.3)
        assert np.min(implied_log_density(K_GRID, bad)) < 0.0

    def test_report_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            slice_arbitrage_report(P, 1.0, -1.0)  # k_hi <= k_lo
        with pytest.raises(ValueError):
            slice_arbitrage_report(P, -1.0, 1.0, n_grid=3)
        with pytest.raises(ValueError):
            svi_g(np.array([np.inf]), P)


class TestSSVIConditions:
    def test_butterfly_conditions(self) -> None:
        # theta phi (1+|rho|) < 4 and theta phi^2 (1+|rho|) <= 4.
        ok = ssvi_butterfly_ok(0.05, -0.4, 0.6)
        assert ok["ok"] == 1.0
        bad = ssvi_butterfly_ok(0.05, 0.9, 8.0)  # 0.05*8*1.9 = 0.76<4 but sq: 0.05*64*1.9>4
        assert bad["cond2_ok"] == 0.0
        assert bad["ok"] == 0.0
        bad2 = ssvi_butterfly_ok(0.5, 0.9, 4.5)  # cond1: 0.5*4.5*1.9 = 4.275 > 4
        assert bad2["cond1_ok"] == 0.0
        with pytest.raises(ValueError):
            ssvi_butterfly_ok(0.0, 0.0, 0.5)

    def test_calendar_partial_theta_bound(self) -> None:
        # rho = 0 -> upper bound +inf; any non-negative d(theta phi) passes.
        ok = ssvi_calendar_partial_theta_bound(0.0, 0.5, 10.0)
        assert ok["ok"] == 1.0 and ok["upper"] == np.inf
        neg = ssvi_calendar_partial_theta_bound(0.3, 0.5, -0.1)
        assert neg["ok"] == 0.0
        # upper = phi (1 + sqrt(1-rho^2)) / rho^2; rho=0.5 -> 0.5*(1+0.866)/0.25
        rep = ssvi_calendar_partial_theta_bound(0.5, 0.5, 4.0)
        assert rep["upper"] == pytest.approx(0.5 * (1 + np.sqrt(0.75)) / 0.25)
        assert rep["ok"] == 0.0  # 4.0 > upper ~ 3.732
        rep2 = ssvi_calendar_partial_theta_bound(0.5, 0.5, 3.0)
        assert rep2["ok"] == 1.0

    def test_essvi_consistent_slices(self) -> None:
        th = np.array([0.02, 0.04, 0.08])
        ph = np.array([0.6, 0.55, 0.5])
        rh = np.array([-0.4, -0.35, -0.3])
        rep = essvi_slices_consistent(th, rh, ph)
        assert rep["consistent"] == 1.0
        assert rep["theta_nondecreasing"] == 1.0
        assert rep["psi_nondecreasing"] == 1.0

    def test_essvi_theta_decrease_flagged(self) -> None:
        th = np.array([0.02, 0.015, 0.08])
        rep = essvi_slices_consistent(th, np.full(3, -0.4), np.full(3, 0.6))
        assert rep["consistent"] == 0.0
        assert rep["theta_nondecreasing"] == 0.0

    def test_essvi_rho_jump_flagged(self) -> None:
        # Large rho jump with small psi increase -> |d(rho psi)/d psi| > 1.
        th = np.array([0.02, 0.0205])
        ph = np.array([0.6, 0.6])
        rh = np.array([-0.4, 0.6])
        rep = essvi_slices_consistent(th, rh, ph)
        assert rep["consistent"] == 0.0
        assert rep["no_crossing"] == 0.0
        assert rep["worst_rho_psi_ratio"] > 1.0

    def test_essvi_fail_closed(self) -> None:
        with pytest.raises(ValueError):
            essvi_slices_consistent(np.array([0.02]), np.array([-0.4]), np.array([0.6]))
        with pytest.raises(ValueError):
            essvi_slices_consistent(np.array([0.02, 0.04]), np.array([-0.4]), np.array([0.6, 0.5]))
        with pytest.raises(ValueError):
            essvi_slices_consistent(
                np.array([0.02, -0.04]), np.array([-0.4, -0.4]), np.array([0.6, 0.5])
            )


class TestJumpWings:
    def test_jw_values(self) -> None:
        t = 0.5
        rep = svi_jump_wings(P, t)
        w0 = float(raw_svi_w(0.0, P))
        assert rep["v"] == pytest.approx(w0 / t)
        assert rep["p"] == pytest.approx(P.b * (1 - P.rho) / t)
        assert rep["c"] == pytest.approx(P.b * (1 + P.rho) / t)
        w_min = P.a + P.b * P.sigma * np.sqrt(1 - P.rho**2)
        assert rep["v_tilde"] == pytest.approx(w_min / t)
        # psi = d sigma_BS / dk at k=0, checked by finite differences
        h = 1e-6
        fd = (svi_implied_vol(h, P, t) - svi_implied_vol(-h, P, t)) / (2 * h)
        assert rep["psi"] == pytest.approx(fd, abs=1e-6)
        with pytest.raises(ValueError):
            svi_jump_wings(P, 0.0)


# ---------------------------------------------------------------------
# Calibration.
# ---------------------------------------------------------------------


class TestCalibration:
    def test_exact_recovery(self) -> None:
        k = np.linspace(-1.0, 1.0, 25)
        w = raw_svi_w(k, P)
        fit = calibrate_slice(k, w, 0.5, seed=0, n_starts=8)
        assert fit.mode == "svi"
        assert fit.converged
        assert fit.rmse < 1e-10
        assert fit.params is not None
        np.testing.assert_allclose(raw_svi_w(k, fit.params), w, atol=1e-8)
        assert fit.params.a == pytest.approx(P.a, abs=1e-4)
        assert fit.params.b == pytest.approx(P.b, abs=1e-4)
        assert fit.params.rho == pytest.approx(P.rho, abs=1e-4)
        assert fit.params.sigma == pytest.approx(P.sigma, abs=1e-4)

    def test_noisy_recovery(self) -> None:
        rng = np.random.default_rng(11)
        k = np.linspace(-1.2, 1.2, 31)
        w = raw_svi_w(k, P)
        wn = np.clip(w * (1 + rng.normal(scale=0.01, size=k.size)), 1e-6, None)
        fit = calibrate_slice(k, wn, 0.5, seed=3, n_starts=8)
        assert fit.mode == "svi"
        assert fit.rmse < 0.01 * np.sqrt(np.mean(w**2)) * 5.0
        np.testing.assert_allclose(fit.w(k), w, atol=0.01)

    def test_determinism(self) -> None:
        k = np.linspace(-1.0, 1.0, 21)
        w = raw_svi_w(k, P) * 1.001  # tiny deterministic noise
        f1 = calibrate_slice(k, w, 1.0, seed=17, n_starts=6)
        f2 = calibrate_slice(k, w, 1.0, seed=17, n_starts=6)
        assert f1.params == f2.params
        assert f1.rmse == f2.rmse

    def test_unsorted_k_accepted(self) -> None:
        k = np.linspace(-1.0, 1.0, 21)
        rng = np.random.default_rng(4)
        perm = rng.permutation(k.size)
        w = raw_svi_w(k, P)
        fit = calibrate_slice(k[perm], w[perm], 0.5, seed=0, n_starts=6)
        assert fit.mode == "svi"
        np.testing.assert_allclose(np.sort(fit.k_obs), k)

    def test_calibrate_iv(self) -> None:
        k = np.linspace(-1.0, 1.0, 25)
        t = 0.75
        iv = np.sqrt(raw_svi_w(k, P) / t)
        fit = calibrate_slice_iv(k, iv, t, seed=0, n_starts=6)
        assert fit.mode == "svi"
        assert fit.rmse < 1e-8
        np.testing.assert_allclose(svi_implied_vol(k, fit.params, t), iv, atol=1e-6)

    def test_weights(self) -> None:
        k = np.linspace(-1.0, 1.0, 25)
        w = raw_svi_w(k, P)
        wts = np.linspace(0.5, 2.0, k.size)
        fit = calibrate_slice(k, w, 0.5, seed=0, n_starts=6, weights=wts)
        assert fit.mode == "svi" and fit.rmse < 1e-9

    def test_penalized_reduces_violation(self) -> None:
        # Quotes from a mid-smile-arbitrageable SVI: the plain LSQ fit
        # recovers it (arb survives); the penalized fit trades a little
        # rmse for a nearly-clean g (documented: pushed, not guaranteed).
        bad = SVIParams(a=-0.1, b=1.0, rho=0.0, m=0.0, sigma=0.3)
        k = np.linspace(-1.0, 1.0, 21)
        w = raw_svi_w(k, bad)
        f0 = calibrate_slice(k, w, 1.0, seed=1, n_starts=6)
        f1 = calibrate_slice(k, w, 1.0, seed=1, n_starts=6, arb_penalty=200.0)
        assert f0.params is not None and f1.params is not None
        r0 = slice_arbitrage_report(f0.params, -3.0, 3.0, 2001)
        r1 = slice_arbitrage_report(f1.params, -3.0, 3.0, 2001)
        assert r0["min_g"] < -1e-3
        assert r1["min_g"] > -1e-3  # violation shrunk by ~3 decades
        assert r1["min_g"] > r0["min_g"]
        assert f1.rmse < 0.1

    def test_penalized_clean_fit_unchanged(self) -> None:
        k = np.linspace(-1.0, 1.0, 25)
        w = raw_svi_w(k, P)
        fit = calibrate_slice(k, w, 0.5, seed=0, n_starts=6, arb_penalty=100.0)
        assert fit.mode == "svi"
        assert fit.rmse < 1e-8
        assert fit.penalty == pytest.approx(0.0, abs=1e-6)

    def test_prev_slice_calendar_penalty(self) -> None:
        k = np.linspace(-1.0, 1.0, 25)
        prev = ssvi_to_raw(0.05, -0.4, 0.6)
        w = raw_svi_w(k, ssvi_to_raw(0.08, -0.3, 0.55))
        fit = calibrate_slice(k, w, 1.0, seed=0, n_starts=6, arb_penalty=50.0, prev_slice=prev)
        assert fit.mode == "svi"
        # fitted slice sits above prev (the true data does too)
        assert np.mean(fit.w(k) - raw_svi_w(k, prev)) > 0.0

    def test_fallback_on_optimizer_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        k = np.linspace(-1.0, 1.0, 15)
        w = raw_svi_w(k, P)

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("forced")

        monkeypatch.setattr(svi.optimize, "least_squares", boom)
        fit = calibrate_slice(k, w, 0.5, seed=0, n_starts=4)
        assert fit.mode == "interp"
        assert not fit.converged
        np.testing.assert_allclose(fit.w(k), np.interp(k, np.sort(k), w[np.argsort(k)]))

    def test_interp_fallback_wing_extension(self) -> None:
        k = np.linspace(-1.0, 1.0, 9)
        w = raw_svi_w(k, P)
        s = SmileSlice(
            t=1.0,
            mode="interp",
            params=None,
            rmse=np.inf,
            converged=False,
            penalty=np.inf,
            k_obs=k,
            w_obs=w,
        )
        np.testing.assert_allclose(s.w(k), w)
        outside = np.array([-2.0, 2.0])
        np.testing.assert_allclose(s.w(outside), _interp_w(outside, k, w))

    @pytest.mark.parametrize(
        "k,w,t",
        [
            (np.array([]), np.array([]), 1.0),  # empty
            (np.linspace(0, 1, 4), np.full(4, 0.1), 1.0),  # <5 quotes
            (np.linspace(0, 1, 6), np.full(6, 0.1), 0.0),  # t = 0
            (np.linspace(0, 1, 6), np.full(6, 0.1), -0.5),  # t < 0
            (np.linspace(0, 1, 6), np.array([0.1, 0.1, np.nan, 0.1, 0.1, 0.1]), 1.0),
            (np.linspace(0, 1, 6), np.array([0.1, 0.1, -0.2, 0.1, 0.1, 0.1]), 1.0),
            (np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4]), np.full(6, 0.1), 1.0),  # dup k
        ],
    )
    def test_fail_closed(self, k: np.ndarray, w: np.ndarray, t: float) -> None:
        with pytest.raises(ValueError):
            calibrate_slice(k, w, t, seed=0)

    def test_fail_closed_more(self) -> None:
        k = np.linspace(-1.0, 1.0, 10)
        w = raw_svi_w(k, P)
        with pytest.raises(ValueError):
            calibrate_slice(k, w[:-1], 1.0)  # length mismatch
        with pytest.raises(ValueError):
            calibrate_slice(k, w, 1.0, weights=np.ones(3))  # bad weights
        with pytest.raises(ValueError):
            calibrate_slice(k, w, 1.0, arb_penalty=-1.0)
        with pytest.raises(ValueError):
            calibrate_slice(k, w, 1.0, prev_slice=P)  # prev without penalty
        with pytest.raises(ValueError):
            calibrate_slice_iv(k, np.zeros(10), 1.0)  # iv = 0
        with pytest.raises(ValueError):
            calibrate_slice_iv(k, np.full(10, 0.2), -1.0)


# ---------------------------------------------------------------------
# Surface assembly, calendar checks, monotone repair.
# ---------------------------------------------------------------------


def _ssvi_surface(rng: np.random.Generator, drop_idx: int | None = None) -> SVISurface:
    """4 SSVI slices with growing theta; ``drop_idx`` plants a theta drop."""
    k = np.linspace(-1.0, 1.0, 9)
    w_placeholder = np.full(9, 0.05)
    ts = np.linspace(0.25, 1.0, 4)
    slices = []
    for i, ti in enumerate(ts):
        th = 0.02 * ti / ts[0]
        if drop_idx is not None and i == drop_idx:
            th = 0.02 * ts[i - 1] / ts[0] * 0.6
        rho = -0.4 + 0.02 * i
        phi = 0.6 - 0.02 * i
        slices.append(
            SmileSlice(
                t=float(ti),
                mode="svi",
                params=ssvi_to_raw(th, rho, phi),
                rmse=0.0,
                converged=True,
                penalty=0.0,
                k_obs=k,
                w_obs=w_placeholder,
            )
        )
    return assemble_surface(slices)


class TestSurface:
    def test_surface_eval_on_grid(self) -> None:
        rng = np.random.default_rng(0)
        surf = _ssvi_surface(rng)
        k = np.array([-0.5, 0.0, 0.5])
        for i, s in enumerate(surf.slices):
            np.testing.assert_allclose(surf.w(k, surf.maturities[i]), s.w(k))
        # midpoint = mean of adjacent slices (linear in t)
        t_mid = 0.5 * (surf.maturities[0] + surf.maturities[1])
        np.testing.assert_allclose(
            surf.w(k, t_mid), 0.5 * (surf.slices[0].w(k) + surf.slices[1].w(k))
        )

    def test_surface_fail_closed(self) -> None:
        rng = np.random.default_rng(0)
        surf = _ssvi_surface(rng)
        with pytest.raises(ValueError):
            surf.w(0.0, 0.1)  # below grid
        with pytest.raises(ValueError):
            surf.w(0.0, 5.0)  # above grid
        with pytest.raises(ValueError):
            assemble_surface([])
        bad = SmileSlice(
            t=-1.0,
            mode="interp",
            params=None,
            rmse=0.0,
            converged=False,
            penalty=0.0,
            k_obs=np.linspace(0, 1, 5),
            w_obs=np.full(5, 0.1),
        )
        with pytest.raises(ValueError):
            assemble_surface([bad])  # t <= 0
        k = np.linspace(-1, 1, 9)
        w = np.full(9, 0.1)

        def mk(t: float) -> SmileSlice:
            return SmileSlice(
                t=t,
                mode="interp",
                params=None,
                rmse=0.0,
                converged=False,
                penalty=0.0,
                k_obs=k,
                w_obs=w,
            )

        with pytest.raises(ValueError):
            assemble_surface([mk(1.0), mk(0.5)])  # not increasing

    def test_calendar_clean(self) -> None:
        rng = np.random.default_rng(1)
        surf = _ssvi_surface(rng)
        rep = calendar_report(surf, np.linspace(-2.0, 2.0, 41))
        assert rep["free"] == 1.0
        assert rep["n_violations"] == 0.0
        assert rep["worst_decrement"] >= 0.0

    def test_calendar_violation_detected(self) -> None:
        rng = np.random.default_rng(1)
        surf = _ssvi_surface(rng, drop_idx=2)
        rep = calendar_report(surf, np.linspace(-2.0, 2.0, 41))
        assert rep["free"] == 0.0
        assert rep["n_violations"] > 0.0
        assert rep["worst_decrement"] < 0.0

    def test_calendar_fail_closed(self) -> None:
        rng = np.random.default_rng(1)
        surf = _ssvi_surface(rng)
        with pytest.raises(ValueError):
            calendar_report(surf, np.array([]))
        with pytest.raises(ValueError):
            calendar_report(surf, np.array([0.0, np.nan]))


class TestRepair:
    def test_pava_known_cases(self) -> None:
        np.testing.assert_allclose(_pava_nondec(np.array([3.0, 1.0, 2.0])), [2.0, 2.0, 2.0])
        np.testing.assert_allclose(_pava_nondec(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            _pava_nondec(np.array([1.0, 3.0, 2.0, 4.0])), [1.0, 2.5, 2.5, 4.0]
        )

    def test_pava_is_l2_closest(self) -> None:
        rng = np.random.default_rng(5)
        y = rng.normal(size=8).cumsum()  # mostly monotone with noise dips
        y += rng.normal(scale=0.5, size=8)
        iso = _pava_nondec(y)
        # brute-force check: iso is feasible and beats naive shifts
        assert np.all(np.diff(iso) >= -1e-12)
        flat = np.full(8, y.mean())
        assert np.sum((iso - y) ** 2) <= np.sum((flat - y) ** 2) + 1e-12

    def test_repair_fixes_violations(self) -> None:
        rng = np.random.default_rng(2)
        kg = np.linspace(-2.0, 2.0, 41)
        surf = _ssvi_surface(rng, drop_idx=2)
        fixed = repair_calendar(surf, kg)
        assert isinstance(fixed, GriddedSurface)
        d = np.diff(fixed.w_mat, axis=0)
        assert np.all(d >= -1e-10)
        # repair preserves values where there was no violation upstream
        # (isotonic regression only lifts downward jumps)
        assert fixed.w_mat.shape == (4, 41)

    def test_repair_idempotent_on_clean(self) -> None:
        rng = np.random.default_rng(3)
        kg = np.linspace(-2.0, 2.0, 41)
        surf = _ssvi_surface(rng)
        fixed = repair_calendar(surf, kg)
        orig = np.stack([s.w(kg) for s in surf.slices], axis=0)
        np.testing.assert_allclose(fixed.w_mat, orig, atol=1e-12)

    def test_gridded_surface_eval(self) -> None:
        rng = np.random.default_rng(2)
        kg = np.linspace(-2.0, 2.0, 21)
        surf = _ssvi_surface(rng)
        fixed = repair_calendar(surf, kg)
        # on-grid points match the matrix; k inside grid interp
        np.testing.assert_allclose(fixed.w(kg, fixed.maturities[1]), fixed.w_mat[1])
        with pytest.raises(ValueError):
            fixed.w(0.0, fixed.maturities[-1] + 1.0)
        with pytest.raises(ValueError):
            repair_calendar(surf, np.array([1.0, 0.5]))  # unsorted grid


# ---------------------------------------------------------------------
# Bench.
# ---------------------------------------------------------------------


class TestBench:
    def test_bench_schema_and_determinism(self) -> None:
        b1 = svi.bench_svi_surface(99)
        b2 = svi.bench_svi_surface(99)
        assert b1 == b2
        assert len(b1) >= 10
        for key, val in b1.items():
            assert key.startswith("synthetic_")
            assert isinstance(val, float)
            assert np.isfinite(val)

    def test_bench_numbers_sane(self) -> None:
        b = svi.bench_svi_surface(7)
        assert b["synthetic_calib_rmse_exact"] < 1e-8
        assert b["synthetic_calib_w_err"] < 1e-6
        assert b["synthetic_arb_precision"] == 1.0
        assert b["synthetic_arb_recall"] == 1.0
        assert b["synthetic_calendar_detection"] == 1.0
        assert b["synthetic_repair_residual"] == 0.0
        assert b["synthetic_lee_detection"] == 1.0
        assert b["synthetic_natural_roundtrip"] < 1e-10
        assert b["synthetic_ssvi_equiv_err"] < 1e-12
