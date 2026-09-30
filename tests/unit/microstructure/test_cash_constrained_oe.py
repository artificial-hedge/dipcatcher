"""SYNTHETIC validation of cash-constrained multi-asset optimal execution.

Reproduces the mechanism experiments of Hashimoto & Stillman (2026,
arXiv:2609.27786) — tighter expected-cash budgets shift schedules toward
sell-first execution and cut peak cash drawdown at comparable implementation
shortfall — on seeded synthetic parameters.  All numbers here are
correctness tests of the QCQP formulation, never market evidence.

Paper corrections encoded below (see module docstring of
``quant_fund.execution.cash_constrained_oe``):
- The Table-1 budgets ``c_k = 5 + 0.1*k`` are strictly infeasible (terminal
  expected-cash floor 10.4 > cap 7); we assert the fail-closed RuntimeError.
- Sell-first asymmetry needs imbalanced cash legs, so the canonical config
  here uses heterogeneous prices/liquidity (cf. the paper's Experiment 2)
  instead of the perfectly symmetric Table-1 setup.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.execution.cash_constrained_oe import (
    build_qcqp,
    is_components,
    peak_cash_drawdown,
    schedule_shape,
    simulate_execution,
    solve_multiasset_oe,
    unconstrained_oe,
)

# Canonical SYNTHETIC configuration (deterministic convex program; the only
# randomness in this file is the seeded Monte-Carlo in TestSimulation).
# Buy leg: dearer-per-cash-unit? No — 15 shares @ 70 (notional 1050), risky
# (sigma 0.3) and cheap temporary impact -> unconstrained AC front-loads it.
# Sell leg: 20 shares @ 55 (notional 1100), calm and expensive temporary
# impact -> unconstrained AC spreads it out.  Cash budgets c_k = beta * k.
N_PERIODS = 20
TAU = 1.0
RISK_AVERSION = 0.30
ORDERS = np.array([15.0, -20.0])
P0 = np.array([70.0, 55.0])
PERM = np.diag([0.02, 0.02])
TEMP = np.diag([0.01, 0.10])
COV = np.diag([0.09, 0.0025])
BETAS = (8.0, 3.0, 1.0, 0.0)  # loose -> tight
SEED = 20260918
N_PATHS = 4000
BUY_NOTIONAL = float(np.sum(P0 * np.clip(ORDERS, 0.0, None)))  # 1050.0


def _budgets(beta: float) -> np.ndarray:
    return beta * np.arange(1, N_PERIODS + 1, dtype=np.float64)


def _solve(beta: float | None = None):
    return solve_multiasset_oe(
        ORDERS,
        P0,
        perm_impact=PERM,
        temp_impact=TEMP,
        cov=COV,
        n_periods=N_PERIODS,
        tau=TAU,
        risk_aversion=RISK_AVERSION,
        cash_budgets=None if beta is None else _budgets(beta),
    )


@pytest.fixture(scope="module")
def solutions() -> dict:
    out: dict = {
        "unconstrained": unconstrained_oe(
            ORDERS,
            P0,
            perm_impact=PERM,
            temp_impact=TEMP,
            cov=COV,
            n_periods=N_PERIODS,
            tau=TAU,
            risk_aversion=RISK_AVERSION,
        )
    }
    for beta in BETAS:
        out[beta] = _solve(beta)
    return out


def _cum_fraction(trades: np.ndarray, asset: int, upto: int) -> float:
    total = float(np.abs(trades[:, asset]).sum())
    return float(np.abs(trades[:upto, asset]).sum()) / total


class TestInputValidation:
    def test_rejects_asymmetric_perm_impact(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            _solve_with(perm=np.array([[0.02, 0.03], [0.01, 0.02]]))

    def test_rejects_indefinite_perm_impact(self) -> None:
        with pytest.raises(ValueError, match="positive semidefinite"):
            _solve_with(perm=np.array([[0.05, 0.0], [0.0, -0.01]]))

    def test_rejects_indefinite_temp_impact(self) -> None:
        with pytest.raises(ValueError, match="positive semidefinite"):
            _solve_with(temp=np.array([[0.01, 0.2], [0.2, 0.01]]))

    def test_rejects_indefinite_cov(self) -> None:
        with pytest.raises(ValueError, match="positive semidefinite"):
            _solve_with(cov=np.array([[0.09, 0.5], [0.5, 0.0025]]))

    def test_rejects_matrix_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"\(2, 2\)"):
            _solve_with(perm=np.eye(3))

    def test_rejects_nonfinite_matrix(self) -> None:
        bad = np.array([[0.02, np.nan], [np.nan, 0.02]])
        with pytest.raises(ValueError, match="finite"):
            _solve_with(perm=bad)

    @pytest.mark.parametrize("lam", [-0.1, np.nan, np.inf])
    def test_rejects_bad_risk_aversion(self, lam: float) -> None:
        with pytest.raises(ValueError, match="risk_aversion"):
            _solve_with(risk_aversion=lam)

    @pytest.mark.parametrize("tau", [0.0, -1.0, np.nan])
    def test_rejects_bad_tau(self, tau: float) -> None:
        with pytest.raises(ValueError, match="tau"):
            _solve_with(tau=tau)

    @pytest.mark.parametrize("p0", [np.array([0.0, 55.0]), np.array([-70.0, 55.0])])
    def test_rejects_nonpositive_p0(self, p0: np.ndarray) -> None:
        with pytest.raises(ValueError, match="p0"):
            solve_multiasset_oe(
                ORDERS,
                p0,
                perm_impact=PERM,
                temp_impact=TEMP,
                cov=COV,
                n_periods=N_PERIODS,
            )

    def test_rejects_budget_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _solve_with(cash_budgets=np.zeros(N_PERIODS - 1))

    @pytest.mark.parametrize("bad", [np.nan, -np.inf])
    def test_rejects_nan_or_minus_inf_budget(self, bad: float) -> None:
        budgets = _budgets(1.0)
        budgets[3] = bad
        with pytest.raises(ValueError, match="finite or \\+inf"):
            _solve_with(cash_budgets=budgets)

    @pytest.mark.parametrize("k", [0, -3, True])
    def test_rejects_bad_n_periods(self, k: int) -> None:
        with pytest.raises(ValueError, match="n_periods"):
            _solve_with(n_periods=k)

    def test_rejects_oversized_problem(self) -> None:
        n = 100
        with pytest.raises(ValueError, match="problem size"):
            solve_multiasset_oe(
                np.zeros(n),
                np.ones(n),
                perm_impact=np.eye(n),
                temp_impact=np.eye(n),
                cov=np.eye(n),
                n_periods=n,
            )

    def test_infeasible_paper_budget_raises(self) -> None:
        # Hashimoto & Stillman Table 1 with beta = 0.1: c_20 = 7 but the
        # terminal expected cash floor is 2*(g/2*(Q^2+Q^2/K)+h*Q^2/K) = 10.4
        # for ANY schedule -> strictly infeasible.  Fail closed, never return
        # a budget-violating schedule (the paper's SCS run did not flag this).
        budgets = 5.0 + 0.1 * np.arange(1, N_PERIODS + 1)
        with pytest.raises(RuntimeError, match="infeasible"):
            solve_multiasset_oe(
                np.array([20.0, -20.0]),
                np.array([50.0, 50.0]),
                perm_impact=np.diag([0.02, 0.02]),
                temp_impact=np.diag([0.05, 0.05]),
                cov=np.diag([0.01, 0.01]),
                n_periods=N_PERIODS,
                tau=TAU,
                risk_aversion=RISK_AVERSION,
                cash_budgets=budgets,
            )


def _solve_with(**overrides):
    """Solve the canonical config with individual arguments overridden."""
    kwargs = {
        "perm_impact": overrides.pop("perm", PERM),
        "temp_impact": overrides.pop("temp", TEMP),
        "cov": overrides.pop("cov", COV),
        "n_periods": overrides.pop("n_periods", N_PERIODS),
        "tau": overrides.pop("tau", TAU),
        "risk_aversion": overrides.pop("risk_aversion", RISK_AVERSION),
        "cash_budgets": overrides.pop("cash_budgets", None),
    }
    orders = overrides.pop("orders", ORDERS)
    p0 = overrides.pop("p0", P0)
    assert not overrides
    return solve_multiasset_oe(orders, p0, **kwargs)


class TestConvexityAndClosedForm:
    def test_objective_matrix_is_psd(self) -> None:
        form = build_qcqp(
            ORDERS,
            perm_impact=PERM,
            temp_impact=TEMP,
            cov=COV,
            n_periods=N_PERIODS,
            tau=TAU,
            risk_aversion=RISK_AVERSION,
        )
        eig = np.linalg.eigvalsh(form.objective_matrix)
        assert float(eig.min()) >= 0.0
        assert form.objective_matrix.shape == (2 * N_PERIODS, 2 * N_PERIODS)

    def test_unconstrained_matches_kkt_closed_form(self) -> None:
        # Classical multi-asset AC baseline: with H positive definite the
        # equality-constrained QP has the exact KKT solution
        # x* = H^{-1} C' (C H^{-1} C')^{-1} b.
        form = build_qcqp(
            ORDERS,
            perm_impact=PERM,
            temp_impact=TEMP,
            cov=COV,
            n_periods=N_PERIODS,
            tau=TAU,
            risk_aversion=RISK_AVERSION,
        )
        h_inv = np.linalg.inv(form.objective_matrix)
        c = form.inventory_matrix
        x_star = h_inv @ c.T @ np.linalg.solve(c @ h_inv @ c.T, form.inventory_rhs)
        baseline = _solve(None)
        np.testing.assert_allclose(baseline.trades, x_star.reshape(2, N_PERIODS).T, atol=1e-8)

    def test_infinite_budget_recovers_unconstrained_exactly(self, solutions: dict) -> None:
        inf_budget = solve_multiasset_oe(
            ORDERS,
            P0,
            perm_impact=PERM,
            temp_impact=TEMP,
            cov=COV,
            n_periods=N_PERIODS,
            tau=TAU,
            risk_aversion=RISK_AVERSION,
            cash_budgets=np.full(N_PERIODS, np.inf),
        )
        np.testing.assert_allclose(inf_budget.trades, solutions["unconstrained"].trades, atol=1e-12)
        assert abs(inf_budget.expected_is - solutions["unconstrained"].expected_is) < 1e-12
        assert abs(inf_budget.objective - solutions["unconstrained"].objective) < 1e-12

    def test_inventory_constraint_holds(self, solutions: dict) -> None:
        for key, res in solutions.items():
            np.testing.assert_allclose(res.trades.sum(axis=0), ORDERS, atol=1e-8, err_msg=key)

    def test_cash_constraints_respected(self, solutions: dict) -> None:
        for beta in BETAS:
            res = solutions[beta]
            caps = _budgets(beta)
            assert np.all(res.expected_cash_path <= caps + 1e-6 * max(1.0, caps.max()))

    def test_zero_risk_zero_perm_impact_gives_twap(self) -> None:
        # Gamma = 0, lambda = 0: minimising (eta/tau) * sum Q_k^2 subject to
        # sum Q_k = Q gives exactly even slices (TWAP).
        res = solve_multiasset_oe(
            np.array([10.0]),
            np.array([100.0]),
            perm_impact=np.zeros((1, 1)),
            temp_impact=np.array([[0.05]]),
            cov=np.array([[0.04]]),
            n_periods=5,
            tau=1.0,
            risk_aversion=0.0,
        )
        np.testing.assert_allclose(res.trades.ravel(), np.full(5, 2.0), atol=1e-8)

    def test_zero_orders_is_trivial(self) -> None:
        res = _solve_with(orders=np.zeros(2), cash_budgets=np.zeros(N_PERIODS))
        # Degenerate optimum: solver epsilon-level trades are acceptable.
        np.testing.assert_allclose(res.trades, np.zeros((N_PERIODS, 2)), atol=1e-4)
        assert res.peak_cash_drawdown == pytest.approx(0.0, abs=1e-9)
        assert np.isnan(res.peak_cash_drawdown_frac)  # no buy leg -> undefined


class TestSellFirstMechanism:
    def test_tight_budget_advances_sells_and_postpones_buys(self, solutions: dict) -> None:
        # Paper RQ1: tighter budgets postpone buys and advance sells.
        shapes = {key: schedule_shape(solutions[key].trades) for key in solutions}
        sell_seq = [shapes["unconstrained"]["sell_first_half_fraction"]]
        buy_seq = [shapes["unconstrained"]["buy_first_half_fraction"]]
        for beta in BETAS:  # loose -> tight
            sell_seq.append(shapes[beta]["sell_first_half_fraction"])
            buy_seq.append(shapes[beta]["buy_first_half_fraction"])
        for looser, tighter in zip(sell_seq, sell_seq[1:], strict=False):
            assert tighter > looser + 1e-3
        for looser, tighter in zip(buy_seq, buy_seq[1:], strict=False):
            assert tighter < looser - 1e-3

    def test_tightest_budget_sign_pattern(self, solutions: dict) -> None:
        tight = solutions[0.0]
        buy_trades, sell_trades = tight.trades[:, 0], tight.trades[:, 1]
        # Sign pattern: the buy asset only buys, the sell asset only sells.
        assert np.all(buy_trades >= -1e-9)
        assert np.all(sell_trades <= 1e-9)
        assert buy_trades[0] > 0.0
        assert sell_trades[0] < 0.0
        # Sell-first in the cash ledger: every period's cumulative sell
        # notional covers the cumulative buy notional (E[M_k] <= 0 budget).
        buy_notional_cum = np.cumsum(buy_trades) * P0[0]
        sell_notional_cum = -np.cumsum(sell_trades) * P0[1]
        assert np.all(sell_notional_cum >= buy_notional_cum - 1e-6)
        assert np.all(tight.expected_cash_path <= 1e-9)

    def test_first_period_sells_exceed_buys_under_tight_budget(self, solutions: dict) -> None:
        tight, loose = solutions[0.0], solutions["unconstrained"]
        # Per-period quantities: tight budget makes the first trade sell-heavy.
        assert abs(tight.trades[0, 1]) > tight.trades[0, 0]
        assert abs(tight.trades[0, 1]) > 3.0 * abs(loose.trades[0, 1])
        assert tight.trades[0, 0] < 0.6 * loose.trades[0, 0]

    def test_sell_completion_front_loaded_vs_unconstrained(self, solutions: dict) -> None:
        quarter = N_PERIODS // 4
        tight, loose = solutions[0.0], solutions["unconstrained"]
        sell_frac_tight = _cum_fraction(tight.trades, 1, quarter)
        sell_frac_loose = _cum_fraction(loose.trades, 1, quarter)
        assert sell_frac_tight >= 2.0 * sell_frac_loose
        # and the buy leg is postponed relative to the unconstrained baseline
        assert _cum_fraction(tight.trades, 0, quarter) < _cum_fraction(loose.trades, 0, quarter)


class TestDrawdownAndShortfallTradeoff:
    def test_peak_drawdown_monotone_as_budget_tightens(self, solutions: dict) -> None:
        peaks = [solutions[key].peak_cash_drawdown for key in ["unconstrained", *BETAS]]
        for looser, tighter in zip(peaks, peaks[1:], strict=False):
            assert tighter <= looser + 1e-9
        # The headline tradeoff: drawdown collapses (>10x from unconstrained
        # to the loosest tested budget; ~0 at the tightest).
        assert peaks[0] > 5.0 * peaks[1]
        assert solutions[0.0].peak_cash_drawdown <= 1e-9

    def test_peak_drawdown_bounded_by_budget_caps(self, solutions: dict) -> None:
        for beta in BETAS:
            cap = float(_budgets(beta).max())
            assert solutions[beta].peak_cash_drawdown <= cap + 1e-6

    def test_is_comparable_while_drawdown_drops(self, solutions: dict) -> None:
        base = solutions["unconstrained"].expected_is
        tightest = solutions[0.0].expected_is
        # Nested feasible sets: expected IS can only rise as budgets tighten,
        # and stays comparable (paper: comparable IS, much lower drawdown).
        assert tightest >= base - 1e-6
        assert tightest <= 1.5 * base
        assert (
            solutions[0.0].peak_cash_drawdown < 0.01 * solutions["unconstrained"].peak_cash_drawdown
        )

    def test_expected_is_monotone_under_nested_budgets(self, solutions: dict) -> None:
        seq = [solutions[key].expected_is for key in ["unconstrained", *BETAS]]
        for looser, tighter in zip(seq, seq[1:], strict=False):
            assert tighter >= looser - 1e-6

    def test_peak_drawdown_fraction_normalised_by_buy_notional(self, solutions: dict) -> None:
        for key, res in solutions.items():
            expected = res.peak_cash_drawdown / BUY_NOTIONAL
            assert res.peak_cash_drawdown_frac == pytest.approx(expected, abs=1e-12), key


class TestMetrics:
    def test_is_components_matches_result(self, solutions: dict) -> None:
        res = solutions[3.0]
        comp = is_components(res.trades, perm_impact=PERM, temp_impact=TEMP, cov=COV, tau=TAU)
        assert set(comp) == {
            "temporary_impact_cost",
            "permanent_impact_cost",
            "expected_is",
            "variance_is",
        }
        assert comp["expected_is"] == pytest.approx(
            comp["temporary_impact_cost"] + comp["permanent_impact_cost"], abs=1e-12
        )
        assert comp["expected_is"] == pytest.approx(res.expected_is, abs=1e-9)
        assert comp["variance_is"] == pytest.approx(res.variance_is, abs=1e-9)
        assert comp["variance_is"] >= 0.0

    def test_objective_is_is_plus_lambda_variance(self, solutions: dict) -> None:
        res = solutions[1.0]
        assert res.objective == pytest.approx(
            res.expected_is + RISK_AVERSION * res.variance_is, abs=1e-9
        )

    def test_peak_cash_drawdown_helper(self) -> None:
        assert peak_cash_drawdown(np.array([-5.0, -1.0, -3.0])) == 0.0
        assert peak_cash_drawdown(np.array([-5.0, 2.0, -3.0])) == 2.0
        with pytest.raises(ValueError, match="finite"):
            peak_cash_drawdown(np.array([1.0, np.nan]))
        with pytest.raises(ValueError, match="non-empty"):
            peak_cash_drawdown(np.array([]))

    def test_schedule_shape_ranges_and_centroids(self, solutions: dict) -> None:
        for _key, res in solutions.items():
            shape = schedule_shape(res.trades)
            assert 0.0 <= float(shape["buy_first_half_fraction"]) <= 1.0
            assert 0.0 <= float(shape["sell_first_half_fraction"]) <= 1.0
            centroid = np.asarray(shape["timing_centroid"])
            assert centroid.shape == (2,)
            assert np.all(centroid >= 1.0) and np.all(centroid <= N_PERIODS)
            assert shape["n_periods"] == N_PERIODS
            # Tighter budgets pull the sell centroid earlier and push the buy
            # centroid later (sell-first reshaping).
        tight_c = np.asarray(schedule_shape(solutions[0.0].trades)["timing_centroid"])
        loose_c = np.asarray(schedule_shape(solutions["unconstrained"].trades)["timing_centroid"])
        assert tight_c[1] < loose_c[1] - 0.5
        assert tight_c[0] > loose_c[0] + 0.5

    def test_schedule_shape_rejects_degenerate(self) -> None:
        with pytest.raises(ValueError, match="all-zero"):
            schedule_shape(np.zeros((5, 2)))
        with pytest.raises(ValueError, match="finite"):
            schedule_shape(np.array([[np.nan]]))


class TestSimulationConsistency:
    """Seeded SYNTHETIC Monte-Carlo checks of the closed-form moments."""

    @pytest.fixture(scope="class")
    def sim(self, solutions: dict) -> dict:
        res = solutions[3.0]
        return {
            "result": res,
            **simulate_execution(
                res.trades,
                P0,
                perm_impact=PERM,
                temp_impact=TEMP,
                cov=COV,
                tau=TAU,
                n_paths=N_PATHS,
                seed=SEED,
            ),
        }

    def test_simulated_is_mean_matches_closed_form(self, sim: dict) -> None:
        res, realized = sim["result"], sim["realized_is"]
        tol = 5.0 * float(realized.std()) / np.sqrt(N_PATHS) + 1e-8
        assert abs(float(realized.mean()) - res.expected_is) <= tol

    def test_simulated_is_variance_matches_closed_form(self, sim: dict) -> None:
        res, realized = sim["result"], sim["realized_is"]
        sample_var = float(realized.var(ddof=1))
        tol = 5.0 * sample_var * np.sqrt(2.0 / N_PATHS) + 1e-8
        assert abs(sample_var - res.variance_is) <= tol

    def test_simulated_cash_path_mean_matches_closed_form(self, sim: dict) -> None:
        res, paths = sim["result"], sim["cash_paths"]
        assert paths.shape == (N_PATHS, N_PERIODS)
        tol = 5.0 * paths.std(axis=0) / np.sqrt(N_PATHS) + 1e-8
        assert np.all(np.abs(paths.mean(axis=0) - res.expected_cash_path) <= tol)

    def test_simulated_peaks_respect_constrained_scale(self, sim: dict) -> None:
        # Realized peaks concentrate near the expected path peak for the
        # constrained schedule (budget 3*k keeps net spend small early).
        peaks = sim["peak_drawdowns"]
        assert np.all(peaks >= 0.0)
        assert float(peaks.mean()) < 0.5 * sim["result"].peak_cash_drawdown + 50.0


class TestDeterminism:
    def test_solve_is_bitwise_deterministic(self) -> None:
        a, b = _solve(1.0), _solve(1.0)
        assert np.array_equal(a.trades, b.trades)
        assert np.array_equal(a.expected_cash_path, b.expected_cash_path)
        assert a.expected_is == b.expected_is

    def test_solve_independent_of_global_seed(self) -> None:
        np.random.seed(1)
        a = _solve(3.0)
        np.random.seed(999)
        b = _solve(3.0)
        assert np.array_equal(a.trades, b.trades)

    def test_simulate_execution_seeded_reproducible(self) -> None:
        trades = _solve(None).trades
        kw = {
            "perm_impact": PERM,
            "temp_impact": TEMP,
            "cov": COV,
            "tau": TAU,
            "n_paths": 200,
        }
        s1 = simulate_execution(trades, P0, seed=SEED, **kw)
        s2 = simulate_execution(trades, P0, seed=SEED, **kw)
        s3 = simulate_execution(trades, P0, seed=SEED + 1, **kw)
        assert np.array_equal(s1["realized_is"], s2["realized_is"])
        assert np.array_equal(s1["cash_paths"], s2["cash_paths"])
        assert not np.array_equal(s1["realized_is"], s3["realized_is"])

    def test_simulate_execution_rejects_bad_inputs(self) -> None:
        trades = _solve(None).trades
        with pytest.raises(ValueError, match="n_paths"):
            simulate_execution(trades, P0, perm_impact=PERM, temp_impact=TEMP, cov=COV, n_paths=0)
        with pytest.raises(ValueError, match="seed"):
            simulate_execution(
                trades,
                P0,
                perm_impact=PERM,
                temp_impact=TEMP,
                cov=COV,
                seed=1.5,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="trades"):
            simulate_execution(np.zeros((0, 2)), P0, perm_impact=PERM, temp_impact=TEMP, cov=COV)


class TestSingleAsset:
    def test_buy_only_pacing_respects_budget(self) -> None:
        # Buy-only program: cash caps force slower early pacing than the
        # unconstrained front-loaded schedule, still fully executed.
        orders = np.array([10.0])
        p0 = np.array([100.0])
        perm = np.array([[0.02]])
        temp = np.array([[0.05]])
        cov = np.array([[0.04]])
        free = solve_multiasset_oe(
            orders,
            p0,
            perm_impact=perm,
            temp_impact=temp,
            cov=cov,
            n_periods=10,
            tau=1.0,
            risk_aversion=0.5,
        )
        # ~1 unit/period plus impact slack (terminal cash has an impact floor).
        budgets = 110.0 * np.arange(1, 11, dtype=np.float64)
        capped = solve_multiasset_oe(
            orders,
            p0,
            perm_impact=perm,
            temp_impact=temp,
            cov=cov,
            n_periods=10,
            tau=1.0,
            risk_aversion=0.5,
            cash_budgets=budgets,
        )
        np.testing.assert_allclose(capped.trades.sum(axis=0), orders, atol=1e-8)
        assert np.all(capped.expected_cash_path <= budgets + 1e-6)
        assert float(np.cumsum(capped.trades)[0]) < float(np.cumsum(free.trades)[0])
        # The unconstrained optimum minimises the same objective over a
        # superset of the feasible region (nested-feasibility guarantee).
        assert capped.objective >= free.objective - 1e-6
