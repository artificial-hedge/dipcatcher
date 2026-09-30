"""Tests for models/american_lsm.py — Longstaff-Schwartz MC + AB04 dual bound.

Seeded SYNTHETIC: the LSM primal, the Andersen-Broadie dual upper bound and
the four benchmarks are verified against known answers — the BS European
closed form, the repo's BAW approximation (models/american_baw.py), the
deep-ITM immediate-exercise boundary, and the shrinkage of the primal-dual
gap in the sub-simulation budget (Longstaff-Schwartz 2001; Andersen-Broadie
2004; Rogers 2002; Haugh-Kogan 2004).  No live option positions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.american_baw import baw_american
from quant_fund.models.american_lsm import (
    LSMPolicy,
    american_lsm_benchmarks,
    basis_size,
    bs_american_put_bench,
    call_payoff,
    deep_itm_boundary_bench,
    dual_upper_bound,
    european_limit_bench,
    lsm_american_price,
    max_call_2asset_bench,
    max_call_payoff,
    polynomial_basis,
    put_payoff,
    simulate_gbm,
)

pytestmark = pytest.mark.synthetic

SEED = 20260929
S0 = 100.0
K = 100.0
T = 1.0
R = 0.05
Q = 0.0
SIGMA = 0.2


def _grid(n_dates: int) -> tuple[float, ...]:
    return tuple(float(x) for x in np.linspace(0.0, T, n_dates))


def _bs_put(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return float(k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------


class TestPolynomialBasis:
    def test_shape_1d_degree3(self) -> None:
        states = np.array([[90.0], [100.0], [110.0]])
        basis = polynomial_basis(states, 3, 100.0)
        assert basis.shape == (3, 4)
        assert basis_size(1, 3) == 4
        np.testing.assert_allclose(basis[:, 0], 1.0)

    def test_shape_2d_degree3(self) -> None:
        states = np.tile(np.array([[90.0, 110.0]]), (5, 1))
        basis = polynomial_basis(states, 3, 100.0)
        assert basis.shape == (5, 10)
        assert basis_size(2, 3) == 10

    def test_monomial_content_degree2(self) -> None:
        states = np.array([[100.0], [200.0]])
        basis = polynomial_basis(states, 2, 100.0)
        expected = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]])
        np.testing.assert_allclose(basis, expected)

    def test_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="2-d"):
            polynomial_basis(np.array([1.0, 2.0]), 3)
        with pytest.raises(ValueError, match="degree"):
            polynomial_basis(np.array([[1.0]]), 0)
        with pytest.raises(ValueError, match="scale"):
            polynomial_basis(np.array([[1.0]]), 3, 0.0)


# ---------------------------------------------------------------------------
# Payoffs and GBM simulation
# ---------------------------------------------------------------------------


class TestPayoffsAndGBM:
    def test_payoff_values(self) -> None:
        states = np.array([[90.0, 120.0], [110.0, 80.0]])
        np.testing.assert_allclose(put_payoff(100.0)(states, 0.5), [10.0, 0.0])
        np.testing.assert_allclose(call_payoff(100.0)(states, 0.5), [0.0, 10.0])
        np.testing.assert_allclose(max_call_payoff(100.0)(states, 0.5), [20.0, 10.0])
        with pytest.raises(ValueError, match="strike"):
            put_payoff(0.0)
        with pytest.raises(ValueError, match="strike"):
            max_call_payoff(-1.0)

    def test_shape_and_t0(self) -> None:
        paths = simulate_gbm(S0, R, Q, SIGMA, _grid(5), 128, SEED)
        assert paths.shape == (128, 5, 1)
        np.testing.assert_allclose(paths[:, 0, 0], S0)

    def test_determinism_and_seed_sensitivity(self) -> None:
        a = simulate_gbm(S0, R, Q, SIGMA, _grid(4), 256, SEED)
        b = simulate_gbm(S0, R, Q, SIGMA, _grid(4), 256, SEED)
        c = simulate_gbm(S0, R, Q, SIGMA, _grid(4), 256, SEED + 1)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_discounted_mean_is_spot(self) -> None:
        # Risk-neutral martingale property: E[e^{-rT} S_T] = S0 (q = 0).
        paths = simulate_gbm(S0, R, Q, SIGMA, (0.0, T), 50_000, SEED)
        st = paths[:, -1, 0]
        discounted = float(st.mean()) * math.exp(-R * T)
        se = float(st.std(ddof=1)) * math.exp(-R * T) / math.sqrt(st.size)
        assert abs(discounted - S0) <= 5.0 * se + 1e-9

    def test_correlation_structure(self) -> None:
        corr = ((1.0, 0.9), (0.9, 1.0))
        paths = simulate_gbm((S0, S0), R, Q, (SIGMA, SIGMA), (0.0, T), 20_000, SEED, corr)
        log_ret = np.log(paths[:, -1, :] / S0)
        empirical = float(np.corrcoef(log_ret[:, 0], log_ret[:, 1])[0, 1])
        assert empirical > 0.85

    def test_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="dates\\[0\\]"):
            simulate_gbm(S0, R, Q, SIGMA, (0.5, 1.0), 64, SEED)
        with pytest.raises(ValueError, match="strictly increasing"):
            simulate_gbm(S0, R, Q, SIGMA, (0.0, 0.5, 0.25), 64, SEED)
        with pytest.raises(ValueError, match="positive"):
            simulate_gbm(-S0, R, Q, SIGMA, _grid(3), 64, SEED)
        with pytest.raises(ValueError, match="sigma"):
            simulate_gbm(S0, R, Q, 0.0, _grid(3), 64, SEED)
        with pytest.raises(ValueError, match="n_paths"):
            simulate_gbm(S0, R, Q, SIGMA, _grid(3), 0, SEED)
        with pytest.raises(ValueError, match="positive definite"):
            simulate_gbm((S0, S0), R, Q, SIGMA, _grid(3), 64, SEED, ((1.0, 2.0), (2.0, 1.0)))
        with pytest.raises(ValueError, match="shape"):
            simulate_gbm(S0, R, Q, SIGMA, _grid(3), 64, SEED, ((1.0, 0.5), (0.5, 1.0)))
        with pytest.raises(ValueError, match="seed"):
            simulate_gbm(S0, R, Q, SIGMA, _grid(3), 64, -1)


# ---------------------------------------------------------------------------
# Fail-closed validation of the pricers
# ---------------------------------------------------------------------------


class TestValidation:
    def test_lsm_rejects_bad_inputs(self) -> None:
        grid = _grid(5)
        payoff = put_payoff(K)
        with pytest.raises(ValueError, match="n_paths"):
            lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=8, seed=SEED)
        with pytest.raises(ValueError, match="degree"):
            lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=512, degree=0, seed=SEED)
        with pytest.raises(ValueError, match="seed"):
            lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=512, seed=True)
        with pytest.raises(ValueError, match="seed"):
            lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=512, seed=2**32)
        with pytest.raises(ValueError, match="basis size"):
            lsm_american_price(
                (S0, S0), R, Q, SIGMA, grid, max_call_payoff(K), n_paths=64, seed=SEED
            )
        with pytest.raises(ValueError, match="basis_scale"):
            lsm_american_price(
                S0, R, Q, SIGMA, grid, payoff, n_paths=512, basis_scale=0.0, seed=SEED
            )

    def test_lsm_rejects_bad_payoff(self) -> None:
        grid = _grid(5)
        with pytest.raises(ValueError, match="callable"):
            lsm_american_price(S0, R, Q, SIGMA, grid, 3.14, n_paths=512, seed=SEED)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="\\(n,\\) array"):
            lsm_american_price(
                S0, R, Q, SIGMA, grid, lambda states, t: states, n_paths=512, seed=SEED
            )
        with pytest.raises(ValueError, match="finite"):
            lsm_american_price(
                S0,
                R,
                Q,
                SIGMA,
                grid,
                lambda states, t: np.full(states.shape[0], np.nan),
                n_paths=512,
                seed=SEED,
            )

    def test_dual_rejects_bad_inputs(self) -> None:
        grid = _grid(5)
        payoff = put_payoff(K)
        lsm = lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=512, seed=SEED)
        with pytest.raises(ValueError, match="n_sub"):
            dual_upper_bound(
                S0, R, Q, SIGMA, grid, payoff, lsm.policy, n_paths=64, n_sub=4, seed=SEED
            )
        with pytest.raises(ValueError, match="n_paths"):
            dual_upper_bound(
                S0, R, Q, SIGMA, grid, payoff, lsm.policy, n_paths=8, n_sub=16, seed=SEED
            )
        bad_policy = "not-a-policy"
        with pytest.raises(ValueError, match="policy"):
            dual_upper_bound(
                S0, R, Q, SIGMA, grid, payoff, bad_policy, n_paths=64, n_sub=16, seed=SEED
            )
        with pytest.raises(ValueError, match="must match"):
            dual_upper_bound(
                S0, R, Q, SIGMA, _grid(13), payoff, lsm.policy, n_paths=64, n_sub=16, seed=SEED
            )

    def test_dual_requires_interior_regression_coverage(self) -> None:
        grid = _grid(5)
        empty_policy = LSMPolicy(
            dates=grid,
            degree=3,
            basis_scale=K,
            coefficients=(None, None, None, None, None),
            exercise_at_zero=False,
            zero_continuation=0.0,
            dual_coefficients=(None, None, None, None, None),
            continuation_caps=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="interior date"):
            dual_upper_bound(
                S0, R, Q, SIGMA, grid, put_payoff(K), empty_policy, n_paths=32, n_sub=8, seed=SEED
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_lsm_same_seed_bitwise(self) -> None:
        grid = _grid(13)
        payoff = put_payoff(K)
        a = lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=2_000, seed=SEED)
        b = lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=2_000, seed=SEED)
        assert a.in_sample_price == b.in_sample_price
        assert a.oos_price == b.oos_price
        assert a.t0_exercise == b.t0_exercise

    def test_lsm_seed_changes_result(self) -> None:
        grid = _grid(13)
        payoff = put_payoff(K)
        a = lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=2_000, seed=SEED)
        b = lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=2_000, seed=SEED + 1)
        assert a.oos_price != b.oos_price

    def test_dual_same_seed_bitwise_and_budget_documented(self) -> None:
        grid = _grid(5)
        payoff = put_payoff(K)
        lsm = lsm_american_price(S0, R, Q, SIGMA, grid, payoff, n_paths=2_000, seed=SEED)
        d1 = dual_upper_bound(
            S0, R, Q, SIGMA, grid, payoff, lsm.policy, n_paths=500, n_sub=16, seed=SEED
        )
        d2 = dual_upper_bound(
            S0, R, Q, SIGMA, grid, payoff, lsm.policy, n_paths=500, n_sub=16, seed=SEED
        )
        assert d1.upper_bound == d2.upper_bound
        assert d1.n_sub_paths_total == 16 * 500 * (len(grid) - 1)


# ---------------------------------------------------------------------------
# European limit (benchmark c)
# ---------------------------------------------------------------------------


class TestEuropeanLimit:
    def test_european_call_limit_bench(self) -> None:
        bench = european_limit_bench(seed=SEED)
        assert bench["label"] == "SYNTHETIC"
        assert bench["within_mc_error"] is True
        assert bench["dates_regressed"] == []
        assert bench["abs_diff_vs_closed_form"] <= bench["mc_error_tolerance"]

    def test_european_put_limit_matches_closed_form(self) -> None:
        payoff = put_payoff(K)
        res = lsm_american_price(
            S0, R, Q, SIGMA, (0.0, T), payoff, n_paths=20_000, n_oos_paths=20_000, seed=SEED
        )
        bs = _bs_put(S0, K, T, R, Q, SIGMA)
        assert abs(res.oos_price - bs) <= 4.0 * res.oos_stderr + 0.01
        assert res.dates_regressed == ()


# ---------------------------------------------------------------------------
# BS American put: LSM vs BAW, primal-dual bracket (benchmark a)
# ---------------------------------------------------------------------------


class TestBSAmericanPutVsBAW:
    @pytest.fixture(scope="class")
    def bench(self) -> dict:
        return bs_american_put_bench(seed=SEED)

    def test_baw_agreement_within_documented_tolerance(self, bench: dict) -> None:
        assert bench["label"] == "SYNTHETIC"
        assert bench["baw_agreement_ok"] is True
        # |LSM_oos - BAW| small: BAW is an approximation, LSM has MC error and
        # a monthly Bermudan grid; documented tolerance = 3.5 se + 0.03.
        assert bench["abs_lsm_oos_vs_baw"] < 0.10
        assert bench["baw_reference_price"] == pytest.approx(
            baw_american(S0, K, T, R, Q, SIGMA, "put"), rel=1e-12
        )

    def test_lsm_inside_dual_bracket(self, bench: dict) -> None:
        assert bench["lower_le_upper"] is True
        assert bench["baw_within_dual_bracket"] is True
        lower = bench["lsm_oos_lower_price"]
        upper = bench["dual_upper_bound"]
        baw = bench["baw_reference_price"]
        assert lower <= baw <= upper
        assert 0.0 < bench["duality_gap"] <= 0.08 * lower

    def test_in_sample_high_bias_vs_oos(self, bench: dict) -> None:
        # Standard LSM bias structure: in-sample (look-ahead) >= out-of-sample.
        assert bench["lsm_in_sample_price"] >= bench["lsm_oos_lower_price"]

    def test_early_exercise_premium_over_european(self, bench: dict) -> None:
        assert bench["premium_nonneg_ok"] is True
        assert bench["early_exercise_premium_vs_european"] > 0.0


# ---------------------------------------------------------------------------
# Immediate-exercise boundary (benchmark b)
# ---------------------------------------------------------------------------


class TestDeepITMBoundary:
    def test_deep_itm_put_exercised_at_zero(self) -> None:
        bench = deep_itm_boundary_bench(seed=SEED)
        assert bench["label"] == "SYNTHETIC"
        assert bench["continuation_below_intrinsic"] is True
        assert bench["exercised_at_zero"] is True
        assert bench["t0_exercise_fraction_oos"] == 1.0
        assert bench["price_equals_intrinsic"] is True
        assert bench["lsm_oos_price"] == pytest.approx(100.0, abs=1e-10)
        assert bench["baw_equals_intrinsic"] is True

    def test_direct_lsm_t0_exercise_is_exact(self) -> None:
        res = lsm_american_price(
            100.0,
            R,
            Q,
            SIGMA,
            _grid(13),
            put_payoff(200.0),
            n_paths=8_000,
            n_oos_paths=8_000,
            basis_scale=200.0,
            seed=SEED,
            exercise_at_zero=True,
        )
        assert res.t0_exercise is True
        # Every path stops at t=0: price is the deterministic intrinsic value.
        assert res.oos_price == 200.0 - 100.0
        assert res.oos_stderr == 0.0
        assert res.in_sample_price == 100.0


# ---------------------------------------------------------------------------
# Andersen-Broadie duality: lower <= upper, gap shrinks with n_sub
# ---------------------------------------------------------------------------


class TestDualityGapShrinkage:
    @pytest.fixture(scope="class")
    def setup(self) -> tuple[float, list]:
        grid = _grid(5)  # quarterly Bermudan
        payoff = put_payoff(K)
        lsm = lsm_american_price(
            S0, R, Q, SIGMA, grid, payoff, n_paths=20_000, n_oos_paths=20_000, seed=SEED
        )
        uppers = [
            dual_upper_bound(
                S0, R, Q, SIGMA, grid, payoff, lsm.policy, n_paths=3_000, n_sub=m, seed=SEED
            )
            for m in (16, 64, 256)
        ]
        return lsm.oos_price, uppers

    def test_duality_lower_le_upper_at_every_budget(self, setup: tuple[float, list]) -> None:
        lower, uppers = setup
        for dual in uppers:
            assert lower <= dual.upper_bound
            assert dual.stderr > 0.0

    def test_upper_bound_decreases_with_sub_sims(self, setup: tuple[float, list]) -> None:
        _, uppers = setup
        u16, u64, u256 = (d.upper_bound for d in uppers)
        # Jensen bias of the pathwise max is positive and O(1/n_sub); streams
        # are prefix-nested, so the paired bounds must tighten with budget.
        assert u16 >= u64 + 0.05
        assert u64 >= u256 + 0.01

    def test_gap_shrinks(self, setup: tuple[float, list]) -> None:
        lower, uppers = setup
        gap16 = uppers[0].upper_bound - lower
        gap256 = uppers[2].upper_bound - lower
        assert gap16 > 0.0
        assert gap256 <= 0.5 * gap16


# ---------------------------------------------------------------------------
# 2-asset American max-call (AB04 headline use case, benchmark d)
# ---------------------------------------------------------------------------


class TestMaxCall2Asset:
    @pytest.fixture(scope="class")
    def bench(self) -> dict:
        return max_call_2asset_bench(seed=SEED)

    def test_duality_and_structure(self, bench: dict) -> None:
        assert bench["label"] == "SYNTHETIC"
        assert bench["lower_le_upper"] is True
        assert bench["n_basis_terms"] == 10  # documented 2-d degree-3 tensor basis
        assert 0.0 <= bench["duality_gap"] <= 0.05 * bench["lsm_oos_lower_price"]

    def test_policy_value_below_upper(self, bench: dict) -> None:
        assert bench["dual_policy_value"] <= bench["dual_upper_bound"] + 3.0 * bench["dual_stderr"]

    def test_early_exercise_premium_not_negative(self, bench: dict) -> None:
        assert bench["premium_nonneg_ok"] is True


# ---------------------------------------------------------------------------
# Aggregate benchmark bundle
# ---------------------------------------------------------------------------


class TestBenchmarkAggregate:
    def test_fast_bundle_all_flags(self) -> None:
        res = american_lsm_benchmarks(seed=SEED, fast=True)
        assert res["label"] == "SYNTHETIC"
        assert res["seed"] == SEED
        benches = res["benchmarks"]
        assert set(benches) == {
            "bs_american_put",
            "deep_itm_boundary",
            "european_limit",
            "max_call_2asset",
        }
        for name, bench in benches.items():
            assert bench["label"] == "SYNTHETIC", name
            assert "references" in bench and "params" in bench, name
            for key, value in bench.items():
                if key.endswith("_ok"):
                    assert value is True, f"{name}.{key}"
        assert benches["deep_itm_boundary"]["exercised_at_zero"] is True
        assert benches["deep_itm_boundary"]["price_equals_intrinsic"] is True
        assert benches["european_limit"]["within_mc_error"] is True
        assert benches["bs_american_put"]["lower_le_upper"] is True
        assert benches["max_call_2asset"]["lower_le_upper"] is True
