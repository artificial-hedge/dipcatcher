"""SYNTHETIC tests for conformal OCE risk-averse decision making.

Pins the OCE closed forms against the repo metrics (CVaR vs
``expected_shortfall_srm``, entropic vs ``entropic_risk_measure``,
mean-variance vs ``mean + beta * var``), checks the CVaR prediction-set
theorem of arXiv:2608.28179 Sec. III-B by running the reserve-scan route
and the exact set-based atom route on the same seeded discrete
distribution, cross-checks the fixed-reserve policy with a cvxpy LP, and
Monte-Carlo-verifies the high-probability calibration guarantee
(Sec. IV, Eq. 19-20), including graceful degradation as the calibration
set shrinks and fail-closed refusal when nothing certifies.

Every random draw uses a seeded ``np.random.default_rng`` — deterministic.
All data is SYNTHETIC correctness evidence, never market evidence.  Proper
scores only (CVaR / entropic / mean-variance OCE of losses); no
Sharpe/P&L content.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.conformal_oce import (
    OCE,
    CalibrationResult,
    KnownPolicyResult,
    PredictionSetResult,
    bench_oce_calibration,
    conformal_oce_calibration,
    cvar_prediction_set_policy,
    known_distribution_policy,
    optimal_policy_lp,
    weighted_cvar,
)
from quant_fund.metrics.entropic_risk import entropic_risk_measure
from quant_fund.metrics.spectral_risk import expected_shortfall_srm

Array = NDArray[np.float64]

SEED = 20260930
ALPHA = 0.3


def _losses(seed: int = SEED, n: int = 400) -> Array:
    """SYNTHETIC loss sample (normal, positive-is-loss convention)."""
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(loc=0.4, scale=1.0, size=n), dtype=np.float64)


def _env(seed: int = 11) -> tuple[Array, Array, Array]:
    """SYNTHETIC discrete decision environment (paper Sec. V-A analogue).

    4 beam-offset actions, 5 quantised angle-error states, 3 observation
    groups with bimodal conditional laws.  Seeded; probabilities are
    normalised irrational-ish floats so the CVaR optimum is unique.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, 5)
    offs = np.linspace(-1.1, 1.1, 4)
    mismatch = np.abs(offs[:, None] - u[None, :])
    loss = np.clip(0.4 * mismatch + rng.uniform(0.0, 0.06, size=mismatch.shape), 0.0, 0.95)
    modes = np.stack(
        [
            np.exp(-((u + 0.6) ** 2) / 0.2),
            np.exp(-((u - 0.6) ** 2) / 0.2),
        ]
    )
    ws = np.array([0.7, 0.45, 0.25])
    cond = np.stack([w * modes[0] + (1.0 - w) * modes[1] for w in ws])
    cond = np.asarray(cond / cond.sum(axis=1, keepdims=True), dtype=np.float64)
    marg = np.array([0.4, 0.35, 0.25])
    return np.asarray(loss, dtype=np.float64), cond, marg


@pytest.fixture(scope="module")
def bench_row() -> dict[str, float | str]:
    """One shared SYNTHETIC Monte-Carlo bench run (module-scoped, ~2-3 s)."""
    return bench_oce_calibration()


# ------------------------------------------------------- OCE closed-form pins


class TestOCEClosedFormPins:
    def test_cvar_pins_expected_shortfall_srm(self) -> None:
        """OCE with rho(z)=z+/alpha IS the repo expected shortfall (alpha flipped).

        Paper convention: alpha is the tail mass; repo convention:
        ``expected_shortfall_srm(alpha=...)`` is the confidence level.
        """
        losses = _losses()
        for alpha in (0.05, 0.1, 0.25, 0.5):
            oce = OCE(kind="cvar", alpha=alpha).oce(losses)
            srm = expected_shortfall_srm(losses, alpha=1.0 - alpha)
            assert oce == pytest.approx(srm, abs=1e-12)

    def test_cvar_pins_numerical_dual(self) -> None:
        losses = _losses()
        for alpha in (0.1, 0.3):
            closed = OCE(kind="cvar", alpha=alpha).oce(losses)
            numeric = OCE(kind="cvar", alpha=alpha).oce_numerical(losses)
            assert closed == pytest.approx(numeric, abs=1e-6)

    def test_entropic_pins_entropic_risk_measure(self) -> None:
        """OCE with rho(z)=(e^{theta z}-1)/theta IS the repo entropic risk."""
        losses = _losses()
        for theta in (0.5, 1.0, 2.0):
            oce = OCE(kind="entropic", theta=theta).oce(losses)
            repo = entropic_risk_measure(losses, theta)
            assert oce == pytest.approx(repo, abs=1e-12)
            numeric = OCE(kind="entropic", theta=theta).oce_numerical(losses)
            assert oce == pytest.approx(numeric, abs=1e-6)

    def test_entropic_stationarity_equation_9(self) -> None:
        """E[rho'(L - t*)] = 1 at the entropic optimum (paper Eq. 9)."""
        losses = _losses()
        theta = 0.7
        oce = OCE(kind="entropic", theta=theta)
        t_star = oce.oce(losses)  # closed form: t* equals the OCE value
        assert float(np.mean(oce.rho_prime(losses - t_star))) == pytest.approx(1.0, abs=1e-10)

    def test_mean_variance_pins_second_moment(self) -> None:
        losses = _losses()
        for beta in (0.0, 0.5, 2.0):
            oce_val = OCE(kind="mean_variance", beta=beta).oce(losses)
            assert oce_val == pytest.approx(float(losses.mean() + beta * losses.var()), abs=1e-12)
            numeric = OCE(kind="mean_variance", beta=beta).oce_numerical(losses)
            assert oce_val == pytest.approx(numeric, abs=1e-6)
        # stationarity: E[1 + 2 beta (L - t*)] = 1 exactly at t* = mean(L)
        mv = OCE(kind="mean_variance", beta=0.5)
        assert float(np.mean(mv.rho_prime(losses - losses.mean()))) == pytest.approx(1.0, abs=1e-12)

    def test_scvar_zero_temperature_limit_is_cvar(self) -> None:
        """sCVaR (paper Sec. II) recovers CVaR as tau -> 0+."""
        losses = _losses()
        alpha = 0.2
        cvar = OCE(kind="cvar", alpha=alpha).oce(losses)
        scvar = OCE(kind="scvar", alpha=alpha, tau=1e-4).oce(losses)
        assert scvar == pytest.approx(cvar, abs=1e-9)

    def test_scvar_monotone_in_tau_and_above_cvar(self) -> None:
        losses = _losses()
        alpha = 0.2
        cvar = OCE(kind="cvar", alpha=alpha).oce(losses)
        vals = [OCE(kind="scvar", alpha=alpha, tau=t).oce(losses) for t in (1e-4, 0.05, 0.2)]
        assert vals[0] <= vals[1] + 1e-12 <= vals[2] + 1e-12
        assert all(v >= cvar - 1e-12 for v in vals)


class TestOCEFailClosed:
    def test_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            OCE(kind="sharpe")  # type: ignore[arg-type]

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2, np.nan])
    def test_bad_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            OCE(kind="cvar", alpha=alpha)

    def test_bad_tau_theta_beta(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            OCE(kind="scvar", tau=0.0)
        with pytest.raises(ValueError, match="theta"):
            OCE(kind="entropic", theta=-1.0)
        with pytest.raises(ValueError, match="beta"):
            OCE(kind="mean_variance", beta=-0.5)

    def test_nonfinite_penalty_input(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            OCE(kind="cvar").rho(np.array([0.0, np.nan]))

    def test_entropic_overflow(self) -> None:
        with pytest.raises(ValueError, match="overflow"):
            OCE(kind="entropic", theta=2.0).rho(np.array([800.0]))
        with pytest.raises(ValueError, match="overflow"):
            OCE(kind="entropic", theta=2.0).rho_prime(np.array([800.0]))

    def test_oce_bad_sample(self) -> None:
        with pytest.raises(ValueError, match="observations"):
            OCE(kind="cvar").oce(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="observations"):
            OCE(kind="cvar").oce(np.full(10, np.inf))
        with pytest.raises(ValueError, match="non-empty"):
            OCE(kind="cvar").oce_numerical(np.array([]))


# ------------------------------------------------------------- weighted CVaR


class TestWeightedCvar:
    def test_uniform_weights_match_empirical(self) -> None:
        losses = _losses(n=201)
        probs = np.full(losses.size, 1.0 / losses.size)
        var_w, cvar_w = weighted_cvar(losses, probs, ALPHA)
        ordered = np.sort(losses)
        k = int(np.ceil(losses.size * (1.0 - ALPHA)))
        assert var_w == pytest.approx(float(ordered[k - 1]), abs=1e-12)
        assert cvar_w == pytest.approx(OCE(kind="cvar", alpha=ALPHA).oce(losses), abs=1e-12)

    def test_monotone_in_alpha(self) -> None:
        losses = _losses(n=201)
        probs = np.full(losses.size, 1.0 / losses.size)
        cvars = [weighted_cvar(losses, probs, a)[1] for a in (0.1, 0.25, 0.5, 0.8)]
        assert all(x >= y - 1e-12 for x, y in zip(cvars, cvars[1:], strict=False))

    def test_fail_closed(self) -> None:
        losses = _losses(n=10)
        with pytest.raises(ValueError, match="non-negative"):
            weighted_cvar(losses, np.full(10, -0.1), ALPHA)
        with pytest.raises(ValueError, match="sum to one"):
            weighted_cvar(losses, np.full(10, 0.05), ALPHA)
        with pytest.raises(ValueError, match="aligned"):
            weighted_cvar(losses, np.full(9, 1.0 / 9), ALPHA)
        with pytest.raises(ValueError, match="alpha"):
            weighted_cvar(losses, np.full(10, 0.1), 1.5)


# --------------------------------------- known-distribution theorem (Sec. III)


class TestKnownPolicyTheorem:
    def test_routes_agree_for_cvar(self) -> None:
        """Theorem check: reserve-scan route == prediction-set route (Sec. III-B)."""
        loss, cond, marg = _env()
        r1 = known_distribution_policy(loss, cond, marg, OCE(kind="cvar", alpha=ALPHA))
        r2 = cvar_prediction_set_policy(loss, cond, marg, ALPHA)
        assert isinstance(r1, KnownPolicyResult)
        assert isinstance(r2, PredictionSetResult)
        assert np.array_equal(r1.actions, r2.actions)
        assert r1.reserve == pytest.approx(r2.reserve, abs=1e-9)
        assert r1.risk == pytest.approx(r2.risk, abs=1e-12)
        # t* is the VaR of the deployed loss (paper Eq. 14 / Eq. 3)
        assert r2.var_alpha == pytest.approx(r2.reserve, abs=1e-12)
        # and the risk is exactly the population CVaR of the deployed loss
        assert r2.cvar_weighted == pytest.approx(r2.risk, abs=1e-12)
        assert r1.risk == pytest.approx(
            weighted_cvar(r1.loss_values, r1.loss_probs, ALPHA)[1], abs=1e-12
        )

    def test_hard_coverage_condition(self) -> None:
        """Eq. 14 in the discrete case: exceedance <= alpha, coverage >= 1-alpha."""
        loss, cond, marg = _env()
        r2 = cvar_prediction_set_policy(loss, cond, marg, ALPHA)
        assert r2.exceedance <= ALPHA + 1e-12
        assert r2.coverage >= 1.0 - ALPHA - 1e-12
        assert r2.coverage == pytest.approx(1.0 - r2.exceedance, abs=1e-12)

    def test_prediction_sets_match_reserve(self) -> None:
        """C*(x) = {y : ell(a*(x), y) <= t*} with marginal coverage 1 - alpha."""
        loss, cond, marg = _env()
        r2 = cvar_prediction_set_policy(loss, cond, marg, ALPHA)
        expected_sets = loss[r2.actions, :] <= r2.reserve
        assert np.array_equal(r2.prediction_sets, expected_sets)
        mass_in = float(np.sum((marg[:, None] * cond) * r2.prediction_sets))
        assert mass_in == pytest.approx(r2.coverage, abs=1e-12)
        assert mass_in >= 1.0 - ALPHA - 1e-12

    def test_lp_cross_check_at_reserve(self) -> None:
        """cvxpy LP over randomised policies == deterministic argmin route."""
        loss, cond, marg = _env()
        oce = OCE(kind="cvar", alpha=ALPHA)
        r1 = known_distribution_policy(loss, cond, marg, oce)
        lp_obj, pi = optimal_policy_lp(loss, cond, marg, oce, r1.reserve)
        pen = oce.rho(loss - r1.reserve)
        cost = np.asarray(cond @ pen.T, dtype=float)
        greedy_obj = float(cost[np.arange(cost.shape[0]), np.argmin(cost, axis=1)] @ marg)
        assert lp_obj == pytest.approx(greedy_obj, abs=1e-8)
        assert pi.shape == cond.shape[:1] + loss.shape[:1]
        assert np.all(pi >= -1e-9)
        assert np.allclose(pi.sum(axis=1), marg, atol=1e-8)

    def test_smooth_penalties_satisfy_stationarity(self) -> None:
        loss, cond, marg = _env()
        for oce in (
            OCE(kind="entropic", theta=0.8),
            OCE(kind="mean_variance", beta=0.5),
            OCE(kind="scvar", alpha=ALPHA, tau=0.05),
        ):
            r = known_distribution_policy(loss, cond, marg, oce)
            assert abs(r.stationarity) <= 1e-6, oce.kind

    def test_optimal_policy_beats_fixed_actions(self) -> None:
        loss, cond, marg = _env()
        r1 = known_distribution_policy(loss, cond, marg, OCE(kind="cvar", alpha=ALPHA))
        for a0 in range(loss.shape[0]):
            acts = np.full(marg.size, a0, dtype=np.int64)
            vals = np.asarray(loss[acts, :], dtype=float).ravel()
            probs = np.asarray(marg[:, None] * cond, dtype=float).ravel()
            fixed_cvar = weighted_cvar(vals, probs, ALPHA)[1]
            assert r1.risk <= fixed_cvar + 1e-9

    def test_determinism(self) -> None:
        loss, cond, marg = _env()
        a = known_distribution_policy(loss, cond, marg, OCE(kind="cvar", alpha=ALPHA))
        b = known_distribution_policy(loss, cond, marg, OCE(kind="cvar", alpha=ALPHA))
        assert np.array_equal(a.actions, b.actions)
        assert a.reserve == b.reserve
        assert a.risk == b.risk

    def test_fail_closed(self) -> None:
        loss, cond, marg = _env()
        bad_cond = cond.copy()
        bad_cond[0] *= 0.5
        with pytest.raises(ValueError, match="sum to one"):
            known_distribution_policy(loss, bad_cond, marg, OCE(kind="cvar", alpha=ALPHA))
        with pytest.raises(ValueError, match="marginal"):
            known_distribution_policy(loss, cond, marg[:-1], OCE(kind="cvar", alpha=ALPHA))
        with pytest.raises(ValueError, match="grid"):
            known_distribution_policy(loss, cond, marg, OCE(kind="cvar", alpha=ALPHA), grid=4)
        with pytest.raises(ValueError, match="alpha"):
            cvar_prediction_set_policy(loss, cond, marg, 1.2)
        with pytest.raises(ValueError, match="reserve"):
            optimal_policy_lp(loss, cond, marg, OCE(kind="cvar", alpha=ALPHA), np.nan)


class TestPredictionSetNesting:
    """Cheap link to the multi-level extension (arXiv:2609.11524).

    That paper shows multi-outage-level design is equivalent to optimising
    over NESTED prediction sets; the single-level sets produced here must
    nest accordingly: smaller tail alpha => larger reserve => smaller set.
    """

    def test_reserves_and_risks_are_monotone_in_alpha(self) -> None:
        loss, cond, marg = _env()
        alphas = (0.1, 0.2, 0.3, 0.45)
        results = [cvar_prediction_set_policy(loss, cond, marg, a) for a in alphas]
        reserves = [r.reserve for r in results]
        risks = [r.risk for r in results]
        assert all(x >= y - 1e-12 for x, y in zip(reserves, reserves[1:], strict=False))
        assert all(x >= y - 1e-12 for x, y in zip(risks, risks[1:], strict=False))
        for a, r in zip(alphas, results, strict=True):
            assert r.coverage >= 1.0 - a - 1e-12


# --------------------------------------- data-driven calibration (Sec. IV)


def _calib_setup(seed: int, n: int) -> tuple[Array, Array, Array, Array]:
    """SYNTHETIC calibration draw: groups, misspecified model rows, states."""
    loss, cond, marg = _env(seed)
    rng = np.random.default_rng(seed + 5)
    groups = rng.integers(0, cond.shape[0], size=n)
    model_probs = np.asarray(0.7 * cond[groups] + 0.3 / cond.shape[1], dtype=np.float64)
    cum = np.cumsum(cond[groups], axis=1)
    states = (rng.random((n, 1)) > cum).sum(axis=1)
    return loss, model_probs, states, np.linspace(0.0, float(loss.max()), 21)


class TestCalibration:
    def test_certifies_with_generous_epsilon(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 300)
        res = conformal_oce_calibration(
            loss, mp, states, OCE(kind="cvar", alpha=ALPHA), grid, epsilon=2.0, delta=0.05
        )
        assert isinstance(res, CalibrationResult)
        assert res.certified
        assert res.t_hat is not None and float(res.t_hat) in set(grid.tolist())
        assert res.ucb is not None and res.ucb <= 2.0 + 1e-12
        assert res.actions is not None and res.actions.shape == (300,)
        assert int(res.actions.min()) >= 0 and int(res.actions.max()) < loss.shape[0]

    def test_ucb_identity_and_radius_formula(self) -> None:
        """UCB_n(t) = t + mean penalty + range * sqrt(log(|T|/delta)/(2n)) — Eq. 19."""
        loss, mp, states, grid = _calib_setup(SEED, 250)
        delta = 0.05
        res = conformal_oce_calibration(
            loss, mp, states, OCE(kind="cvar", alpha=ALPHA), grid, epsilon=2.0, delta=delta
        )
        assert np.allclose(
            res.ucb_curve, res.grid + res.mean_penalty_curve + res.radius_curve, atol=1e-15
        )
        expected_radius = (res.penalty_hi_curve - res.penalty_lo_curve) * np.sqrt(
            np.log(grid.size / delta) / (2.0 * 250)
        )
        assert np.allclose(res.radius_curve, expected_radius, atol=1e-15)
        assert np.all(res.radius_curve >= 0.0)

    def test_per_reserve_radius_tightens_uniform_bound(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 250)
        oce = OCE(kind="cvar", alpha=ALPHA)
        res = conformal_oce_calibration(loss, mp, states, oce, grid, epsilon=2.0, delta=0.05)
        uniform = conformal_oce_calibration(
            loss,
            mp,
            states,
            oce,
            grid,
            epsilon=2.0,
            delta=0.05,
            penalty_range=(0.0, float(loss.max()) / ALPHA),
        )
        assert np.allclose(uniform.radius_curve, uniform.radius_curve[0], atol=1e-15)
        assert np.all(res.radius_curve <= uniform.radius_curve[0] + 1e-15)

    def test_t_hat_is_ucb_argmin_over_feasible(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 300)
        eps = 0.62
        res = conformal_oce_calibration(
            loss, mp, states, OCE(kind="cvar", alpha=ALPHA), grid, epsilon=eps, delta=0.1
        )
        feasible = np.flatnonzero(res.ucb_curve <= eps)
        assert res.certified == bool(feasible.size)
        if feasible.size:
            assert res.t_hat == pytest.approx(
                float(grid[int(feasible[np.argmin(res.ucb_curve[feasible])])]), abs=1e-15
            )

    def test_fail_closed_when_nothing_certifies(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 300)
        res = conformal_oce_calibration(
            loss, mp, states, OCE(kind="cvar", alpha=ALPHA), grid, epsilon=-1.0, delta=0.1
        )
        assert not res.certified
        assert res.t_hat is None and res.ucb is None and res.actions is None
        # diagnostics survive the refusal
        assert res.ucb_curve.shape == grid.shape

    def test_declared_range_violation_raises(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 100)
        with pytest.raises(ValueError, match="Hoeffding range"):
            conformal_oce_calibration(
                loss,
                mp,
                states,
                OCE(kind="cvar", alpha=ALPHA),
                grid,
                epsilon=2.0,
                delta=0.1,
                penalty_range=(0.0, 0.001),
            )

    def test_validation_fail_closed(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 60)
        oce = OCE(kind="cvar", alpha=ALPHA)
        with pytest.raises(ValueError, match="delta"):
            conformal_oce_calibration(loss, mp, states, oce, grid, epsilon=1.0, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            conformal_oce_calibration(loss, mp, states, oce, grid, epsilon=1.0, delta=1.0)
        with pytest.raises(ValueError, match="epsilon"):
            conformal_oce_calibration(loss, mp, states, oce, grid, epsilon=np.nan, delta=0.1)
        with pytest.raises(ValueError, match="reserve_grid"):
            conformal_oce_calibration(loss, mp, states, oce, np.array([]), epsilon=1.0, delta=0.1)
        with pytest.raises(ValueError, match=r"\[0, K\)"):
            conformal_oce_calibration(
                loss, mp, states + loss.shape[1], oce, grid, epsilon=1.0, delta=0.1
            )
        with pytest.raises(ValueError, match="integer"):
            conformal_oce_calibration(
                loss, mp, states.astype(float), oce, grid, epsilon=1.0, delta=0.1
            )
        with pytest.raises(ValueError, match="length"):
            conformal_oce_calibration(loss, mp, states[:-1], oce, grid, epsilon=1.0, delta=0.1)
        bad_mp = mp.copy()
        bad_mp[0] *= 0.5
        with pytest.raises(ValueError, match="sum to one"):
            conformal_oce_calibration(loss, bad_mp, states, oce, grid, epsilon=1.0, delta=0.1)
        with pytest.raises(ValueError, match=r"\(n, K\)"):
            conformal_oce_calibration(loss, mp[:, :-1], states, oce, grid, epsilon=1.0, delta=0.1)

    def test_determinism(self) -> None:
        loss, mp, states, grid = _calib_setup(SEED, 200)
        oce = OCE(kind="cvar", alpha=ALPHA)
        a = conformal_oce_calibration(loss, mp, states, oce, grid, epsilon=0.7, delta=0.1)
        b = conformal_oce_calibration(loss, mp, states, oce, grid, epsilon=0.7, delta=0.1)
        assert a.certified == b.certified
        assert a.t_hat == b.t_hat
        assert np.array_equal(a.ucb_curve, b.ucb_curve)
        assert np.array_equal(a.actions, b.actions)


# ------------------------- SYNTHETIC Monte-Carlo of the Eq. 20 guarantee


BENCH_KEYS = {
    "dgp",
    "claim",
    "seed",
    "alpha",
    "delta",
    "epsilon",
    "trials",
    "n_cal",
    "n_cal_small",
    "cert_rate",
    "cert_rate_small",
    "violation_rate",
    "violation_rate_certified",
    "violation_rate_small",
    "baseline_violation_rate",
    "plugin_cert_rate",
    "plugin_violation_rate",
    "hoeffding_radius",
    "hoeffding_radius_small",
    "radius_ratio",
    "mean_ucb_gap",
}


class TestCalibrationMonteCarlo:
    def test_bench_keys_and_honesty_labels(self, bench_row: dict[str, float | str]) -> None:
        assert set(bench_row) == BENCH_KEYS
        assert bench_row["claim"] == "research_metric_only"
        assert str(bench_row["dgp"]).startswith("synthetic")
        for k, v in bench_row.items():
            if k not in ("dgp", "claim"):
                assert np.isfinite(float(v)), k

    def test_high_probability_oce_control(self, bench_row: dict[str, float | str]) -> None:
        """Eq. 20: violation rate of CERTIFIED policies <= delta + tol."""
        delta = float(bench_row["delta"])
        assert float(bench_row["cert_rate"]) >= 0.2  # guarantee is exercised, not vacuous
        assert float(bench_row["violation_rate"]) <= delta + 0.03
        assert float(bench_row["violation_rate_certified"]) <= delta + 0.03
        assert float(bench_row["mean_ucb_gap"]) >= 0.0  # UCB dominates true risk

    def test_control_is_not_vacuous(self, bench_row: dict[str, float | str]) -> None:
        """The uncertified greedy baseline violates often; the gate does work."""
        assert float(bench_row["baseline_violation_rate"]) >= 0.2
        assert float(bench_row["baseline_violation_rate"]) > float(bench_row["violation_rate"])
        # plug-in ablation certifies recklessly (no Hoeffding margin)
        assert float(bench_row["plugin_cert_rate"]) >= float(bench_row["cert_rate"])

    def test_graceful_degradation_as_calibration_shrinks(
        self, bench_row: dict[str, float | str]
    ) -> None:
        """Smaller n => wider radius => fewer certificates, never violations."""
        assert float(bench_row["hoeffding_radius_small"]) > float(bench_row["hoeffding_radius"])
        assert float(bench_row["radius_ratio"]) == pytest.approx(
            float(np.sqrt(float(bench_row["n_cal"]) / float(bench_row["n_cal_small"]))),
            abs=1e-12,
        )
        assert float(bench_row["cert_rate_small"]) <= float(bench_row["cert_rate"]) + 1e-12
        assert float(bench_row["violation_rate_small"]) <= float(bench_row["delta"]) + 0.03

    def test_bench_determinism(self) -> None:
        a = bench_oce_calibration(trials=40, n_cal=500, n_cal_small=125)
        b = bench_oce_calibration(trials=40, n_cal=500, n_cal_small=125)
        assert a == b
