"""Fukasawa first-order implied-variance representation -- SYNTHETIC tests.

arXiv:2609.13961, Theorem 2.5: w_hat(k, T) = E[A_T | S_T = S0 e^k] + o(v r)
for bounded standardized log-strikes z = (k + v/2) / sqrt(v). Seeded
Monte-Carlo correctness material only -- deterministic-limit reductions,
closed-form equalities (BS put under conditional clock D_T, eq. 53
normalization, the eq. 37 first-order sqrt(T) slope), the three regime
convergence tables under common random numbers, fail-closed edges,
determinism, and the flat bench dict. Never market evidence; no PnL-style
headline metrics anywhere.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models import fukasawa_iv as F
from quant_fund.models.iv_approx import brenner_subrahmanyam_iv
from quant_fund.models.options import bs_price
from quant_fund.models.realized import realized_variance

SEED = 2609
S0 = 1.0


def _det_paths(v: float = 0.04, T: float = 0.5, n_paths: int = 4000) -> F.FactorSVPaths:
    """Deterministic-variance model: a = v T per path (a_scale = 0)."""
    return F.simulate_lognormal_factor_sv(
        v_star=v,
        volvol=1.0,
        a_scale=0.0,
        rho=-0.7,
        maturity=T,
        n_steps=64,
        n_paths=n_paths,
        seed=SEED,
    )


def _sv_paths(
    a_scale: float = 0.5, rho: float = -0.7, n_paths: int = 20000, seed: int = SEED
) -> F.FactorSVPaths:
    return F.simulate_lognormal_factor_sv(
        v_star=0.04,
        volvol=1.0,
        a_scale=a_scale,
        rho=rho,
        maturity=0.5,
        n_steps=96,
        n_paths=n_paths,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Black-Scholes put under total variance (paper eq. 2)
# ---------------------------------------------------------------------------


class TestBsPutTotalVariance:
    def test_matches_bs_price_scalar(self):
        s, K, T, q = 1.0, 0.95, 0.5, 0.08
        sigma = math.sqrt(q / T)
        got = F.bs_put_total_variance(s, K, q)
        ref = bs_price(s, K, T, sigma, r=0.0, call=False)
        assert got == pytest.approx(ref, rel=1e-12)

    def test_q0_returns_intrinsic(self):
        assert F.bs_put_total_variance(0.9, 1.0, 0.0) == pytest.approx(0.1)
        assert F.bs_put_total_variance(1.1, 1.0, 0.0) == pytest.approx(0.0)

    def test_vectorized_and_monotone_in_q(self):
        s = np.array([0.9, 1.0, 1.1])
        q = np.array([0.02, 0.05, 0.10])
        p = F.bs_put_total_variance(s, 1.0, q)
        assert p.shape == (3,)
        row = F.bs_put_total_variance(1.0, 1.0, q)
        assert np.all(np.diff(row) > 0.0)  # put price increases with variance

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            F.bs_put_total_variance(0.0, 1.0, 0.05)
        with pytest.raises(ValueError):
            F.bs_put_total_variance(1.0, -1.0, 0.05)
        with pytest.raises(ValueError):
            F.bs_put_total_variance(1.0, 1.0, -0.01)
        with pytest.raises(ValueError):
            F.bs_put_total_variance(np.nan, 1.0, 0.05)


# ---------------------------------------------------------------------------
# Model put price + implied total variance
# ---------------------------------------------------------------------------


class TestPutPriceAndImpliedVariance:
    def test_deterministic_vol_price_is_bs(self):
        # a = 0: J random but P_BS(s0 e^{J - C/2}, K, D) still MC-averages
        # the same put as a deterministic-vol model only through the RB
        # identity; with C = rho^2 A = 0 it degenerates. Use rho = 0 where
        # J = 0 exactly and the RB price is the closed form.
        paths = F.simulate_lognormal_factor_sv(
            v_star=0.04,
            volvol=1.0,
            a_scale=0.0,
            rho=0.0,
            maturity=0.5,
            n_steps=32,
            n_paths=4000,
            seed=SEED,
        )
        price, se = F.model_put_price(paths, -0.1, S0)
        ref = F.bs_put_total_variance(S0, S0 * math.exp(-0.1), 0.04 * 0.5)
        assert price == pytest.approx(ref, rel=1e-12)
        assert se == pytest.approx(0.0, abs=1e-15)

    def test_rao_blackwell_vs_payoff_agree(self):
        paths = _sv_paths(a_scale=0.4, n_paths=20000)
        p_rb, se_rb = F.model_put_price(paths, -0.10, S0, method="rao_blackwell")
        p_nv, se_nv = F.model_put_price(paths, -0.10, S0, method="payoff")
        assert abs(p_rb - p_nv) < 4.0 * math.hypot(se_rb, se_nv) + 2e-5
        assert se_rb < se_nv  # Rao-Blackwellization reduces MC variance

    def test_implied_total_variance_roundtrip(self):
        T, k, v = 0.5, -0.1, 0.05
        price = float(F.bs_put_total_variance(S0, S0 * math.exp(k), v * T))
        w = F.implied_total_variance(price, S0, k, T)
        assert w == pytest.approx(v * T, rel=1e-6)

    def test_implied_vol_vs_atm_approx(self):
        # Brenner-Subrahmanyam ATM approx vs exact inversion at k ~ 0.
        paths = _sv_paths(a_scale=0.3, n_paths=20000)
        price, _ = F.model_put_price(paths, 0.0, S0)
        w = F.implied_total_variance(price, S0, 0.0, paths.maturity)
        approx_sigma = brenner_subrahmanyam_iv(price, S0, paths.maturity)
        exact_sigma = math.sqrt(w / paths.maturity)
        assert exact_sigma == pytest.approx(approx_sigma, rel=0.06)

    def test_fail_closed(self):
        paths = _sv_paths(n_paths=1000)
        with pytest.raises(ValueError):
            F.model_put_price(paths, -0.1, S0, method="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            F.model_put_price(paths, math.nan, S0)
        with pytest.raises(ValueError):
            F.implied_total_variance(-0.5, S0, 0.0, 0.5)
        with pytest.raises(ValueError):
            F.implied_total_variance(2.0, S0, 0.0, 0.5)  # above put bound K e^k
        with pytest.raises(ValueError):
            F.implied_total_variance(0.1, S0, 0.0, 0.0)  # T = 0
        with pytest.raises(TypeError):
            F.model_put_price({"S": np.zeros(5)}, -0.1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Conditional-QV estimator (paper eqs. 51-53)
# ---------------------------------------------------------------------------


class TestConditionalQV:
    def test_deterministic_vol_reduction_exact(self):
        # A deterministic => E[A | S_T = K] = A for every k (eq. 53).
        paths = _det_paths(v=0.04, T=0.5)
        cq = F.conditional_qv(paths, np.linspace(-0.25, 0.25, 11))
        np.testing.assert_allclose(cq.m, 0.04 * 0.5, rtol=1e-10)
        assert cq.v_mean == pytest.approx(0.04 * 0.5, rel=1e-12)

    def test_density_integrates_to_one(self):
        paths = _sv_paths(a_scale=0.4)
        sq_v = math.sqrt(paths.v_mean)
        kk = np.linspace(-3.5 * sq_v, 3.5 * sq_v, 401)
        cq = F.conditional_qv(paths, kk)
        assert np.trapezoid(cq.density, kk) == pytest.approx(1.0, abs=0.02)

    def test_normalization_identity(self):
        # int m(k) E[h(k)] dk = E[A] -- exact identity of eq. 53.
        paths = _sv_paths(a_scale=0.4)
        sq_v = math.sqrt(paths.v_mean)
        kk = np.linspace(-4.0 * sq_v, 4.0 * sq_v, 801)
        cq = F.conditional_qv(paths, kk)
        lhs = np.trapezoid(cq.m * cq.density, kk)
        assert lhs == pytest.approx(cq.v_mean, rel=2e-3)

    def test_leverage_sign_skew(self):
        # rho < 0: down moves carry the vol factor -> conditional QV is
        # higher at negative k than at +|k| (leverage skew).
        paths = _sv_paths(a_scale=0.6, rho=-0.7, n_paths=30000)
        cq = F.conditional_qv(paths, np.array([-0.15, 0.15]))
        assert cq.m[0] > cq.m[1]

    def test_shapes_and_se(self):
        paths = _sv_paths()
        kk = np.linspace(-0.2, 0.2, 7)
        cq = F.conditional_qv(paths, kk)
        assert cq.k.shape == cq.m.shape == cq.density.shape == cq.m_se.shape == (7,)
        assert np.all(cq.m > 0.0) and np.all(cq.density > 0.0)
        assert np.all(np.isfinite(cq.m_se)) and np.all(cq.m_se >= 0.0)
        assert cq.n_paths == paths.n_paths

    def test_fail_closed_clock_consistency(self):
        # a != c + d: not the J/N decomposition of a local martingale.
        n = 64
        a = np.full(n, 0.02)
        with pytest.raises(ValueError, match="martingale|clocks"):
            F.conditional_qv_arrays(a, np.zeros(n), np.full(n, 0.005), np.full(n, 0.01), [0.0])

    def test_fail_closed_inputs(self):
        n = 8
        a = np.full(n, 0.02)
        j = np.zeros(n)
        c = np.full(n, 0.01)
        d = np.full(n, 0.01)
        with pytest.raises(ValueError):
            F.conditional_qv_arrays(-a, j, c, d, [0.0])
        with pytest.raises(ValueError):
            F.conditional_qv_arrays(a, j, c, np.zeros(n), [0.0])  # D <= 0
        with pytest.raises(ValueError):
            F.conditional_qv_arrays(a, j, -c, d, [0.0])
        with pytest.raises(ValueError):
            F.conditional_qv_arrays(a * np.nan, j, c, d, [0.0])
        with pytest.raises(ValueError):
            F.conditional_qv_arrays(a[:2], j[:2], c[:2], d[:2], [0.0])
        with pytest.raises(ValueError):
            F.conditional_qv_arrays(a[:-1], j, c, d, [0.0])  # length mismatch
        paths = _sv_paths(n_paths=2000)
        with pytest.raises(TypeError):
            F.conditional_qv({"a_total": a}, [0.0])  # type: ignore[arg-type]
        # standardized strike beyond the boundedness guard
        with pytest.raises(ValueError, match="standardized"):
            F.conditional_qv(paths, [50.0 * math.sqrt(paths.v_mean)])

    def test_underflow_density_fails(self):
        # k inside z_max yet past the simulated support: E[h] underflows.
        paths = _sv_paths(a_scale=0.05, n_paths=2000)
        with pytest.raises(ValueError):
            F.conditional_qv(paths, [2.0])  # ~10 stdev of log-returns


# ---------------------------------------------------------------------------
# Standardized strike + fukasawa_residual
# ---------------------------------------------------------------------------


class TestResidual:
    def test_standardized_strike_formula(self):
        v = 0.02
        k = np.array([-0.1, 0.0, 0.1])
        z = F.standardized_strike(k, v)
        np.testing.assert_allclose(z, (k + v / 2.0) / math.sqrt(v))
        with pytest.raises(ValueError):
            F.standardized_strike(k, 0.0)
        with pytest.raises(ValueError):
            F.standardized_strike(k, -0.02)

    def test_deterministic_residual_is_zero(self):
        # a = 0, rho = 0: J = 0, price deterministic, m = v T exactly.
        paths = F.simulate_lognormal_factor_sv(
            v_star=0.04,
            volvol=1.0,
            a_scale=0.0,
            rho=0.0,
            maturity=0.5,
            n_steps=32,
            n_paths=4000,
            seed=SEED,
        )
        est = F.fukasawa_residual(paths, -0.1)
        assert est.m == pytest.approx(0.02, rel=1e-10)
        assert abs(est.residual) < 5e-4

    def test_residual_field_consistency(self):
        paths = _sv_paths(a_scale=0.4)
        est = F.fukasawa_residual(paths, -0.12)
        assert est.residual == pytest.approx(est.w_hat - est.m, rel=1e-12)
        assert est.z == pytest.approx((est.k + est.v_mean / 2.0) / math.sqrt(est.v_mean))
        assert est.residual_se >= max(est.m_se, 0.0)

    def test_residual_scales_with_volvol(self):
        # Larger vol-of-vol => larger |residual| at fixed k (paper: err = O(a)).
        est_small = F.fukasawa_residual(_sv_paths(a_scale=0.2, n_paths=40000), -0.12)
        est_big = F.fukasawa_residual(_sv_paths(a_scale=0.9, n_paths=40000), -0.12)
        assert abs(est_big.residual) > abs(est_small.residual)

    def test_fail_closed(self):
        paths = _sv_paths(n_paths=2000)
        with pytest.raises(TypeError):
            F.fukasawa_residual(np.zeros(10), 0.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            F.fukasawa_residual(paths, 0.0, s0=0.0)
        with pytest.raises(ValueError, match="standardized"):
            F.fukasawa_residual(paths, 30.0)


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------


class TestSimulators:
    def test_lognormal_martingale_and_clock(self):
        paths = _sv_paths(a_scale=0.5)
        np.testing.assert_allclose(paths.a_total, paths.c_total + paths.d_total, rtol=1e-10)
        # martingale: E[S_T / S0] = E[exp(log_s)] = 1 within MC se
        s_t = np.exp(paths.log_s)
        se = float(np.std(s_t, ddof=1) / math.sqrt(paths.n_paths))
        assert abs(np.mean(s_t) - 1.0) < 5.0 * se

    def test_lognormal_deterministic_closed_form(self):
        # a = 0 with drift: A = v_* (e^{drift T} - 1)/drift exactly.
        drift, v, T = 0.3, 0.04, 0.5
        n_steps = 32
        paths = F.simulate_lognormal_factor_sv(
            v_star=v,
            volvol=1.0,
            a_scale=0.0,
            drift=drift,
            rho=-0.5,
            maturity=T,
            n_steps=n_steps,
            n_paths=4000,
            seed=SEED,
        )
        # A is the left-point grid sum of the deterministic vol path.
        dt = T / n_steps
        exact = float(np.sum(v * np.exp(drift * np.arange(n_steps) * dt)) * dt)
        np.testing.assert_allclose(paths.a_total, exact, rtol=1e-10)
        # ...and within discretization error of the continuum integral.
        continuum = v * (math.exp(drift * T) - 1.0) / drift
        assert exact == pytest.approx(continuum, rel=5e-3)

    def test_ou_stationary_mean_and_realized_crosscheck(self):
        paths = F.simulate_ou_factor_sv(
            v_star=0.04,
            beta=0.6,
            lam=1.0,
            nu=1.0,
            accel=32.0,
            rho=-0.5,
            maturity=0.5,
            n_steps=512,
            n_paths=20000,
            seed=SEED,
            return_log_path=True,
        )
        v_bar = F.ou_vol_stat_mean(0.04, 0.6, 1.0, 1.0)
        assert paths.v_mean / 0.5 == pytest.approx(v_bar, rel=0.05)
        # realized QV of the log-price path tracks integrated variance A
        assert paths.log_s_path is not None
        rets = np.diff(paths.log_s_path, axis=1)
        rv = np.array([realized_variance(r) for r in rets])
        assert float(np.mean(rv)) == pytest.approx(paths.v_mean, rel=0.1)

    def test_heston_adapter_martingale(self):
        paths = F.heston_factor_paths(
            spot=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            v0=0.04,
            maturity=0.5,
            n_steps=128,
            n_paths=8000,
            seed=SEED,
        )
        np.testing.assert_allclose(paths.a_total, paths.c_total + paths.d_total)
        s_t = np.exp(paths.log_s)
        se = float(np.std(s_t, ddof=1) / math.sqrt(paths.n_paths))
        assert abs(np.mean(s_t) - 1.0) < 5.0 * se
        # E[A] ~ theta * T for stationary init v0 = theta
        assert paths.v_mean == pytest.approx(0.04 * 0.5, rel=0.1)

    def test_simulator_fail_closed(self):
        with pytest.raises(ValueError):
            F.simulate_lognormal_factor_sv(
                v_star=0.04,
                volvol=1.0,
                a_scale=-0.1,
                rho=-0.5,
                maturity=0.5,
                n_steps=16,
                n_paths=16,
                seed=SEED,
            )
        with pytest.raises(ValueError):
            F.simulate_lognormal_factor_sv(
                v_star=0.0,
                volvol=1.0,
                a_scale=0.1,
                rho=-0.5,
                maturity=0.5,
                n_steps=16,
                n_paths=16,
                seed=SEED,
            )
        with pytest.raises(ValueError):
            F.simulate_lognormal_factor_sv(
                v_star=0.04,
                volvol=1.0,
                a_scale=0.1,
                rho=1.0,
                maturity=0.5,
                n_steps=16,
                n_paths=16,
                seed=SEED,
            )
        with pytest.raises(ValueError):
            F.simulate_lognormal_factor_sv(
                v_star=0.04,
                volvol=1.0,
                a_scale=0.1,
                rho=-0.5,
                maturity=0.5,
                n_steps=3,
                n_paths=16,
                seed=SEED,
            )
        with pytest.raises(ValueError):
            F.simulate_ou_factor_sv(
                v_star=0.04,
                beta=0.5,
                lam=1.0,
                nu=1.0,
                accel=-1.0,
                rho=-0.5,
                maturity=0.5,
                n_steps=16,
                n_paths=16,
                seed=SEED,
            )
        with pytest.raises(ValueError):
            F.heston_factor_paths(
                spot=1.0,
                kappa=2.0,
                theta=0.04,
                xi=0.0,
                rho=-0.5,
                v0=0.04,
                maturity=0.5,
                n_steps=16,
                n_paths=16,
                seed=SEED,
            )
        with pytest.raises(ValueError):
            F.simulate_lognormal_factor_sv(
                v_star=0.04,
                volvol=1.0,
                a_scale=0.1,
                rho=-0.5,
                maturity=0.5,
                n_steps=16,
                n_paths=16,
                seed=-1,
            )

    def test_factor_sv_paths_validation(self):
        n = 8
        good = dict(
            a_total=np.full(n, 0.02),
            j_total=np.zeros(n),
            c_total=np.full(n, 0.01),
            d_total=np.full(n, 0.01),
            log_s=np.zeros(n),
            maturity=0.5,
        )
        F.FactorSVPaths(**good)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            F.FactorSVPaths(**{**good, "a_total": np.full(n, -0.02)})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            F.FactorSVPaths(**{**good, "d_total": np.zeros(n)})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            F.FactorSVPaths(**{**good, "j_total": np.zeros(n - 1)})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            F.FactorSVPaths(
                **{**good, "log_s_path": np.zeros((n + 1, 5))}  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError):
            F.FactorSVPaths(
                **{**good, "log_s_path": np.full((n, 5), np.nan)}  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Regime convergence tables (common-random-number paired anchors)
# ---------------------------------------------------------------------------


class TestSmallVolVolRegime:
    def test_scaled_residual_decreases(self):
        # Sec. 2.5 eq. 41: |paired|/a is the o(a) remainder per unit a.
        out = F.small_volvol_convergence(
            [0.5, 0.25, 0.125],
            k=0.10,
            n_paths=40000,
            n_steps=96,
            seed=SEED,
        )
        sc = np.asarray(out["scaled_residual"])
        assert sc[0] > sc[1] > sc[2]
        assert sc[-1] < 0.6 * sc[0]

    def test_anchor_near_exact_limit(self):
        out = F.small_volvol_convergence(
            [0.5, 0.25],
            k=0.10,
            n_paths=40000,
            n_steps=96,
            seed=SEED,
        )
        # a = 0 anchor: true representation error is 0; |anchor res| is
        # pure MC inversion noise, small vs the raw residuals' scale.
        assert abs(out["anchor_residual"]) < 5e-3

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            F.small_volvol_convergence([0.0], k=0.1)
        with pytest.raises(ValueError):
            F.small_volvol_convergence([], k=0.1)
        with pytest.raises(ValueError):
            F.small_volvol_convergence([0.5, np.nan], k=0.1)


class TestFastMeanRevertingRegime:
    def test_scaled_residual_decreases(self):
        # Sec. 2.6 eq. 49: |paired| * sqrt(n) shrinks (o(n^{-1/2})).
        out = F.fast_mean_reverting_convergence(
            [1.0, 4.0, 16.0],
            k=0.10,
            n_paths=40000,
            n_steps=1024,
            seed=SEED,
        )
        sc = np.asarray(out["scaled_residual"])
        assert sc[0] > sc[-1]
        assert sc[0] > np.mean(sc[1:])

    def test_v_bar_and_anchor(self):
        out = F.fast_mean_reverting_convergence(
            [4.0],
            k=0.10,
            n_paths=20000,
            n_steps=512,
            seed=SEED,
        )
        assert out["v_bar"] == pytest.approx(F.ou_vol_stat_mean(0.04, 0.6, 1.0, 1.0))
        assert out["v_bar"] == pytest.approx(0.0420, abs=5e-4)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            F.fast_mean_reverting_convergence([0.5], k=0.1)
        with pytest.raises(ValueError):
            F.fast_mean_reverting_convergence([4.0], k=0.1, v_star=0.0)


class TestShortMaturityRegime:
    def test_slope_m_matches_closed_form(self):
        # Eq. 37 (H = 1/2): E[A/T | S = e^{x sqrt T}] -> v0 +
        # sigma_U rho x sqrt(v0)/2 * sqrt(T) -- m's slope is the exact
        # closed form at affordable MC sizes.
        out = F.short_maturity_convergence(
            [0.08, 0.04, 0.02, 0.01],
            x=0.4,
            n_paths=160000,
            n_steps=256,
            seed=SEED,
        )
        slope_m = np.asarray(out["slope_m"])
        np.testing.assert_allclose(slope_m, out["slope_theory"], rtol=0.12)

    def test_slope_sig2_agrees_first_order(self):
        # Theorem: sigma_hat^2 and E[A/T|S] share the leading sqrt(T)
        # coefficient (the residual is sub-leading).
        out = F.short_maturity_convergence(
            [0.08, 0.04, 0.02, 0.01],
            x=0.4,
            n_paths=160000,
            n_steps=256,
            seed=SEED,
        )
        slope_s = np.asarray(out["slope_sig2"])
        np.testing.assert_allclose(slope_s, out["slope_theory"], rtol=0.25)

    def test_remainder_subleading(self):
        # |res_ann| stays far below the shared first-order term
        # |slope| * sqrt(T): the quantitative o(sqrt(T)) statement.
        out = F.short_maturity_convergence(
            [0.08, 0.04, 0.02, 0.01],
            x=0.4,
            n_paths=160000,
            n_steps=256,
            seed=SEED,
        )
        tt = np.asarray(out["maturity"])
        bound = 0.25 * abs(out["slope_theory"]) * np.sqrt(tt)
        assert np.all(np.abs(np.asarray(out["residual"])) < bound)
        # ...and consistent with zero given the MC standard errors.
        res = np.asarray(out["residual"])
        se = np.asarray(out["residual_se"])
        assert np.all(np.abs(res) < 3.0 * se)

    def test_grid_snap_and_fail_closed(self):
        with pytest.raises(ValueError):
            F.short_maturity_convergence([0.08, 0.0], x=0.4)
        with pytest.raises(ValueError):
            F.short_maturity_convergence(
                [0.08, 0.001], x=0.4, n_steps=64
            )  # smallest row < _MIN_STEPS
        with pytest.raises(ValueError):
            F.short_maturity_convergence([0.08], x=0.4, rho=1.5)
        with pytest.raises(ValueError):
            F.short_maturity_convergence([0.08], x=np.nan)


# ---------------------------------------------------------------------------
# Determinism + bench
# ---------------------------------------------------------------------------


class TestDeterminismAndBench:
    def test_bit_identical_same_seed(self):
        p1 = _sv_paths(seed=SEED)
        p2 = _sv_paths(seed=SEED)
        for name in ("a_total", "j_total", "c_total", "d_total", "log_s"):
            np.testing.assert_array_equal(getattr(p1, name), getattr(p2, name))
        e1 = F.fukasawa_residual(p1, -0.1)
        e2 = F.fukasawa_residual(p2, -0.1)
        assert e1.residual == e2.residual

    def test_different_seed_differs(self):
        p1 = _sv_paths(seed=SEED)
        p2 = _sv_paths(seed=SEED + 1)
        assert not np.array_equal(p1.log_s, p2.log_s)

    def test_bench_flat_dict(self):
        out = F.bench_fukasawa_iv(n_paths=8000, n_steps=64)
        assert isinstance(out, dict)
        for key, val in out.items():
            assert isinstance(val, float), key
            assert math.isfinite(val), key
            low = key.lower()
            for bad in ("sharpe", "sortino", "calmar", "pnl", "nav", "return"):
                assert bad not in low, key
        assert out["synthetic"] == 1.0
        assert out["synthetic_deterministic_vol_abs_resid"] < 5e-4
        assert (
            out["synthetic_small_volvol_scaled_resid_a25"]
            < out["synthetic_small_volvol_scaled_resid_a50"]
        )
        assert out["synthetic_normalization_gap"] < 5e-3
        assert out["synthetic_runtime_seconds"] > 0.0
        assert F.SYNTHETIC_LABEL.lower().startswith("synthetic")
