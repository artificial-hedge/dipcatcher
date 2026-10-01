"""Tests for microstructure/impulse_mm.py — Hawkes-excited impulse MM lane.

Bivariate Hawkes MO intensity (recursive O(1) form + binned convolution),
the Gueant-style Riccati-ODE solver in ``(t, q, lambda+, lambda-)`` (backward
finite-difference shooting), the ``HawkesFlow`` ``RegimeState`` adapter, and
the sealed ``hawkes_mm.v1`` bench receipt. Everything is **labeled
SYNTHETIC** correctness validation — never market evidence, no live-trading
claim. ``sim_internal_*`` keys are simulator-internal diagnostics asserted
here to never leak as headline metrics.

Fast: the two solver runs and the session bundle are module-scoped fixtures.
Pure numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.impulse_mm import (
    BivariateHawkes,
    BivariateHawkesParams,
    HawkesFlow,
    ImpulseMMConfig,
    ImpulseMMState,
    hawkes_intensity_path,
    hawkes_mm_bench,
    hawkes_mm_policy,
    run_impulse_mm_session,
    solve_impulse_mm,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig
from quant_fund.models.market_making import as_optimal_quotes
from quant_fund.utils.receipt import seal_errors

# ---------------------------------------------------------------------------
# Shared pins
# ---------------------------------------------------------------------------

GAMMA = 0.05
SIGMA = 0.05
KAPPA = 500.0
A_FILL = 0.8
Q_MAX = 4
HORIZON = 40.0
TAU_EVAL = 20.0
Q_POINTS = (-2, -1, 0, 1, 2)

# Forbidden *headline* metric tokens (mirrors research.catalog registry). "pnl"
# is permitted ONLY under the simulator-internal diagnostic namespace.
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")


def _hawkes(alpha: float = 0.0, beta: float = 0.8) -> BivariateHawkesParams:
    """Symmetric bivariate Hawkes; ``alpha`` scales both diagonal channels."""
    return BivariateHawkesParams(
        mu_plus=0.10,
        mu_minus=0.10,
        alpha_pp=alpha,
        alpha_pm=0.2 * alpha,
        alpha_mp=0.2 * alpha,
        alpha_mm=alpha,
        beta=beta,
    )


def _config(alpha: float, *, gamma: float = GAMMA, n_lam: int = 9) -> ImpulseMMConfig:
    return ImpulseMMConfig(
        gamma=gamma,
        sigma=SIGMA,
        kappa=KAPPA,
        a_fill=A_FILL,
        q_max=Q_MAX,
        hawkes=_hawkes(alpha),
        horizon=HORIZON,
        n_t=120,
        n_lam=n_lam,
        lam_max=1.6,
    )


def _all_keys(obj: object) -> list[str]:
    """Recursively collect mapping keys (dicts only; walk list/tuple values)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


def _as_residual(sol, gamma: float) -> float:
    """Max |quote - AS closed form| over the pinned (q, mu) evaluation set."""
    worst = 0.0
    for q in Q_POINTS:
        as_q = as_optimal_quotes(100.0, float(q), gamma, SIGMA, TAU_EVAL, KAPPA)
        db, da = sol.offsets(TAU_EVAL, q, 0.10, 0.10)
        worst = max(
            worst, abs(100.0 - db - float(as_q["bid"])), abs(100.0 + da - float(as_q["ask"]))
        )
    return worst


@pytest.fixture(scope="module")
def sol_poisson():
    return solve_impulse_mm(_config(0.0))


@pytest.fixture(scope="module")
def sol_hawkes():
    return solve_impulse_mm(_config(0.45, n_lam=15))


@pytest.fixture(scope="module")
def session(sol_hawkes):
    flow = HawkesFlow(excitation=0.35, decay_rho=0.90, p_buy=0.5)
    return run_impulse_mm_session(
        config=ZILobConfig(
            s0=100.0,
            tick=0.01,
            lam=0.06,
            mu=0.10,
            theta_cxl=0.02,
            p_buy=0.5,
            band=5,
            init_levels=3,
            init_depth=5,
            seed=3,
        ),
        solution=sol_hawkes,
        horizon=600.0,
        decision_interval=1.0,
        inventory_cap=Q_MAX,
        sample_interval=25.0,
        flow=flow,
        estimator=BivariateHawkes(_hawkes(0.45)),
    )


@pytest.fixture(scope="module")
def receipt():
    return hawkes_mm_bench(seed=0, quick=True)


# ---------------------------------------------------------------------------
# Hawkes parameters: validation + stationary moments
# ---------------------------------------------------------------------------


class TestBivariateHawkesParams:
    def test_stationary_mean_kat(self) -> None:
        # (I - A)^{-1} mu with symmetric alpha: 0.1 / (1 - (0.45 + 0.09)) = 0.2173...
        p = BivariateHawkesParams(
            mu_plus=0.10,
            mu_minus=0.10,
            alpha_pp=0.45,
            alpha_pm=0.09,
            alpha_mp=0.09,
            alpha_mm=0.45,
            beta=0.8,
        )
        mp, mm = p.stationary_mean
        assert mp == pytest.approx(mm)
        assert mp == pytest.approx(0.10 / (1.0 - 0.54))
        assert p.branching_ratio == pytest.approx(0.54)

    def test_jump_vectors(self) -> None:
        p = _hawkes(0.45)
        jp, jm = p.jump_buy
        assert jp == pytest.approx(0.45 * 0.8)
        assert jm == pytest.approx(0.09 * 0.8)
        jp, jm = p.jump_sell
        assert jp == pytest.approx(0.09 * 0.8)
        assert jm == pytest.approx(0.45 * 0.8)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha_pp": -0.1},
            {"alpha_mp": -0.1},
            {"beta": 0.0},
            {"beta": -1.0},
            {"beta": float("nan")},
            {"mu_plus": 0.0},
            {"mu_minus": -0.2},
        ],
    )
    def test_fail_closed_params(self, kwargs: dict) -> None:
        base = dict(
            mu_plus=0.10,
            mu_minus=0.10,
            alpha_pp=0.3,
            alpha_pm=0.05,
            alpha_mp=0.05,
            alpha_mm=0.3,
            beta=0.8,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            BivariateHawkesParams(**base)  # type: ignore[arg-type]

    def test_explosive_branching_rejected(self) -> None:
        with pytest.raises(ValueError, match="spectral radius"):
            BivariateHawkesParams(
                mu_plus=0.10,
                mu_minus=0.10,
                alpha_pp=0.9,
                alpha_pm=0.5,
                alpha_mp=0.5,
                alpha_mm=0.9,
                beta=0.8,
            )

    def test_branching_boundary_rejected(self) -> None:
        with pytest.raises(ValueError, match="spectral radius"):
            BivariateHawkesParams(
                mu_plus=0.10,
                mu_minus=0.10,
                alpha_pp=1.0,
                alpha_pm=0.0,
                alpha_mp=0.0,
                alpha_mm=1.0,
                beta=0.8,
            )


# ---------------------------------------------------------------------------
# Intensity recursion (KAT) + convolution path
# ---------------------------------------------------------------------------


class TestIntensityRecursion:
    def test_post_event_jump_and_decay(self) -> None:
        p = _hawkes(0.45)
        h = BivariateHawkes(p)
        # Buy MO at t=0.5: pre-jump = baseline (decayed to mu), then jump.
        pre_p, pre_m = h.observe(0.5, "buy")
        assert pre_p == pytest.approx(0.10)
        assert pre_m == pytest.approx(0.10)
        assert h.lam_plus == pytest.approx(0.10 + 0.45 * 0.8)
        assert h.lam_minus == pytest.approx(0.10 + 0.09 * 0.8)
        # Decay to t=1.5 (one full second): excess multiplies by e^{-beta}.
        lp, lm = h.advance_to(1.5)
        decay = math.exp(-0.8)
        assert lp == pytest.approx(0.10 + 0.36 * decay)
        assert lm == pytest.approx(0.10 + 0.072 * decay)

    def test_cross_excitation_ordering(self) -> None:
        p = _hawkes(0.45)
        h = BivariateHawkes(p)
        h.observe(0.0, "sell")
        # A sell event excites the sell channel most, buy channel by cross.
        assert h.lam_minus > h.lam_plus > p.mu_plus

    def test_o1_recursion_matches_bruteforce(self) -> None:
        # Recursive form must equal the explicit decay-sum over all events.
        p = _hawkes(0.45)
        events = [(0.3, "buy"), (0.7, "sell"), (1.1, "buy"), (2.0, "buy")]
        h = BivariateHawkes(p)
        for t, side in events:
            h.observe(t, side)
        lp, lm = h.advance_to(4.0)
        sb = sum(math.exp(-p.beta * (4.0 - t)) for t, s in events if s == "buy")
        ss = sum(math.exp(-p.beta * (4.0 - t)) for t, s in events if s == "sell")
        assert lp == pytest.approx(p.mu_plus + p.beta * (p.alpha_pp * sb + p.alpha_pm * ss))
        assert lm == pytest.approx(p.mu_minus + p.beta * (p.alpha_mp * sb + p.alpha_mm * ss))

    def test_backward_time_rejected(self) -> None:
        h = BivariateHawkes(_hawkes(0.3))
        h.observe(1.0, "buy")
        with pytest.raises(ValueError):
            h.advance_to(0.5)
        with pytest.raises(ValueError):
            h.observe(1.0, "sideways")

    def test_convolution_impulse_kat(self) -> None:
        p = _hawkes(0.45)
        times = np.arange(0.0, 10.0, 0.005)
        path = hawkes_intensity_path(times, np.array([2.0]), np.array([]), p)
        lam = path["lam_plus"]
        i0 = int(round(2.0 / 0.005))
        # At the event bin: lam+ = mu + alpha_pp * beta e^{-beta * 0} = mu + a*beta.
        assert lam[i0] == pytest.approx(0.10 + 0.45 * 0.8, abs=1e-9)
        # One half-life later: excess halves.
        hl = math.log(2.0) / p.beta
        i1 = int(round((2.0 + hl) / 0.005))
        assert lam[i1] - 0.10 == pytest.approx(0.5 * (lam[i0] - 0.10), rel=0.05)
        # Sell channel responds only through the cross kernel.
        lm = path["lam_minus"]
        assert lm[i0] == pytest.approx(0.10 + 0.09 * 0.8, abs=1e-9)

    def test_convolution_rejects_nonuniform_grid(self) -> None:
        with pytest.raises(ValueError):
            hawkes_intensity_path(
                np.array([0.0, 0.1, 0.5]), np.array([]), np.array([]), _hawkes(0.3)
            )


# ---------------------------------------------------------------------------
# HawkesFlow: the RegimeState-contract adapter
# ---------------------------------------------------------------------------


class TestHawkesFlow:
    def test_contract_fields_and_recursion_kat(self) -> None:
        f = HawkesFlow(excitation=0.35, decay_rho=0.90, p_buy=0.55)
        st = f.current()
        assert st.name == "hawkes_excited"
        assert st.intensity_mult == 1.0
        assert st.p_buy == 0.55
        f.advance()
        # I_1 = 1 + eps (kick happens after one geometric decay of excess 0)
        assert f.current().intensity_mult == pytest.approx(1.0 + 0.35)
        f.advance()
        assert f.current().intensity_mult == pytest.approx(1.0 + 0.35 * 0.9 + 0.35)
        assert f.n_mo == 2

    def test_bound_and_determinism(self) -> None:
        bound = 1.0 + 0.35 / 0.10
        f1 = HawkesFlow(excitation=0.35, decay_rho=0.90, p_buy=0.5)
        f2 = HawkesFlow(excitation=0.35, decay_rho=0.90, p_buy=0.5)
        for _ in range(500):
            f1.advance()
            f2.advance()
        assert f1.current().intensity_mult == f2.current().intensity_mult
        assert f1.current().intensity_mult <= bound + 1e-9
        assert f1.current().intensity_mult == pytest.approx(bound, rel=1e-9)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"excitation": -0.1},
            {"decay_rho": 0.0},
            {"decay_rho": 1.0},
            {"decay_rho": float("nan")},
            {"p_buy": 1.5},
            {"p_buy": -0.1},
            {"name": ""},
        ],
    )
    def test_fail_closed(self, kwargs: dict) -> None:
        base = {"excitation": 0.3, "decay_rho": 0.9, "p_buy": 0.5}
        base.update(kwargs)
        with pytest.raises(ValueError):
            HawkesFlow(**base)


# ---------------------------------------------------------------------------
# Riccati solver: Poisson limit, excitation response, walls, validation
# ---------------------------------------------------------------------------


class TestSolveImpulseMM:
    def test_poisson_limit_recovers_as(self, sol_poisson) -> None:
        # The exact Riccati offsets approach the AS closed form (the AS
        # formula is the first-order-in-gamma approximation of the same HJB);
        # at gamma=0.05 the honest residual is ~2e-3, ~60% of the AS
        # half-spread — bounded by one full half-spread as a structural KAT.
        half = float(as_optimal_quotes(100.0, 0.0, GAMMA, SIGMA, TAU_EVAL, KAPPA)["half_spread"])
        assert _as_residual(sol_poisson, GAMMA) < half

    def test_poisson_limit_gamma_contraction(self) -> None:
        # Convergence is monotone in gamma: the residual must shrink as
        # gamma -> 0 (observed ~gamma^2; asserted conservatively at >4x).
        sol_hi = solve_impulse_mm(_config(0.0, gamma=0.05))
        sol_lo = solve_impulse_mm(_config(0.0, gamma=0.01))
        res_hi = _as_residual(sol_hi, 0.05)
        res_lo = _as_residual(sol_lo, 0.01)
        assert res_lo < 1.5e-4  # pinned: ~5e-5 observed
        assert res_lo < 0.25 * res_hi

    def test_as_limit_monotone_in_alpha(self, sol_poisson) -> None:
        # Hawkes -> Poisson: |offset - AS| is minimized at alpha = 0 and
        # grows with excitation strength.
        resid = [_as_residual(sol_poisson, GAMMA)]
        for a in (0.15, 0.30):
            resid.append(_as_residual(solve_impulse_mm(_config(a)), GAMMA))
        assert resid[0] < resid[1] < resid[2]

    def test_skew_direction(self, sol_poisson) -> None:
        # Long inventory widens the bid and tightens the ask (reservation skew).
        db0, da0 = sol_poisson.offsets(TAU_EVAL, 0, 0.10, 0.10)
        db2, da2 = sol_poisson.offsets(TAU_EVAL, 2, 0.10, 0.10)
        assert db2 > db0
        assert da2 < da0
        assert db0 == pytest.approx(da0)  # symmetric at flat inventory

    def test_excitation_response(self, sol_hawkes) -> None:
        # Higher buy-side intensity deepens the ask offset (high arrival rate
        # supports posting deeper) and tightens the bid.
        db_lo, da_lo = sol_hawkes.offsets(TAU_EVAL, 0, 0.10, 0.10)
        db_hi, da_hi = sol_hawkes.offsets(TAU_EVAL, 0, 1.2, 0.10)
        assert da_hi > da_lo
        assert db_hi < db_lo

    def test_inventory_wall_offsets(self, sol_hawkes) -> None:
        db, da = sol_hawkes.offsets(TAU_EVAL, -Q_MAX, 0.5, 0.5)
        assert da == float("inf") and math.isfinite(db)
        db, da = sol_hawkes.offsets(TAU_EVAL, Q_MAX, 0.5, 0.5)
        assert db == float("inf") and math.isfinite(da)

    def test_determinism(self) -> None:
        a = solve_impulse_mm(_config(0.3))
        b = solve_impulse_mm(_config(0.3))
        np.testing.assert_array_equal(a.theta, b.theta)
        np.testing.assert_array_equal(a.delta_ask, b.delta_ask)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gamma": 0.0},
            {"sigma": -1.0},
            {"kappa": 0.0},
            {"a_fill": 0.0},
            {"a_fill": 1.5},
            {"q_max": 0},
            {"horizon": -1.0},
            {"n_t": 4},
            {"n_lam": 3},
            {"lam_max": 0.05},  # below baseline mu
            {"lam_max": 0.15},  # above mu but below the stationary mean
            {"neg_clip": -1.0},
        ],
    )
    def test_fail_closed_config(self, kwargs: dict) -> None:
        base = dict(
            gamma=GAMMA,
            sigma=SIGMA,
            kappa=KAPPA,
            a_fill=A_FILL,
            q_max=Q_MAX,
            hawkes=_hawkes(0.3),
            horizon=HORIZON,
            n_t=60,
            n_lam=7,
            lam_max=1.6,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            ImpulseMMConfig(**base)  # type: ignore[arg-type]

    def test_offsets_fail_closed(self, sol_hawkes) -> None:
        with pytest.raises(ValueError):
            sol_hawkes.offsets(-1.0, 0, 0.1, 0.1)
        with pytest.raises(ValueError):
            sol_hawkes.offsets(TAU_EVAL, Q_MAX + 1, 0.1, 0.1)
        with pytest.raises(ValueError):
            sol_hawkes.offsets(TAU_EVAL, 0, -0.1, 0.1)
        with pytest.raises(TypeError):
            solve_impulse_mm("nope")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            hawkes_mm_policy("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Policy adapter
# ---------------------------------------------------------------------------


class TestHawkesMMPolicy:
    def test_wall_suppresses_side(self, sol_hawkes) -> None:
        pol = hawkes_mm_policy(sol_hawkes, tick=0.01)
        st = ImpulseMMState(
            t=0.0,
            mid=100.0,
            best_bid=99.9,
            best_ask=100.1,
            inventory=-Q_MAX,
            tau=TAU_EVAL,
            lam_plus=0.5,
            lam_minus=0.5,
        )
        bid, ask = pol(st)
        assert bid is not None and ask is None

    def test_no_mid_no_quotes(self, sol_hawkes) -> None:
        pol = hawkes_mm_policy(sol_hawkes)
        st = ImpulseMMState(
            t=0.0,
            mid=None,
            best_bid=None,
            best_ask=None,
            inventory=0,
            tau=TAU_EVAL,
            lam_plus=0.1,
            lam_minus=0.1,
        )
        assert pol(st) == (None, None)


# ---------------------------------------------------------------------------
# Session runner + sealed bench receipt
# ---------------------------------------------------------------------------


class TestSessionAndReceipt:
    def test_session_inventory_cap(self, session) -> None:
        assert session["max_abs_inventory"] <= Q_MAX
        assert session["inventory_cap"] == Q_MAX
        assert session["n_decisions"] > 0

    def test_session_estimator_and_flow_report(self, session) -> None:
        assert session["flow_report"]["kind"] == "hawkes_flow"
        assert session["flow_report"]["intensity_mult_final"] <= (
            session["flow_report"]["intensity_mult_bound"] + 1e-9
        )
        lp = np.asarray(session["lam_plus_path"], dtype=np.float64)
        assert lp.size > 0 and np.all(np.isfinite(lp))
        # Excitation is observable: the filter must deviate from baseline.
        assert lp.max() > 0.10 + 1e-6

    def test_session_honesty_labels(self, session) -> None:
        assert session["label"] == "SYNTHETIC"
        assert session["research_only"] is True
        assert session["live_pnl_claim"] is False
        assert session["claim"] == "simulator_internal_diagnostic_only"
        for key in _all_keys(session):
            low = key.lower()
            for tok in FORBIDDEN_HEADLINE_TOKENS:
                assert tok not in low, f"forbidden headline token {tok} in {key}"
            # Any 'pnl' *metric* key is namespaced as a simulator-internal
            # diagnostic; the sole exception is the honesty flag live_pnl_claim.
            if "pnl" in low:
                assert low == "live_pnl_claim" or low.startswith("sim_internal_"), key

    def test_session_determinism(self, sol_hawkes) -> None:
        def _run() -> dict:
            return run_impulse_mm_session(
                config=ZILobConfig(
                    s0=100.0,
                    tick=0.01,
                    lam=0.06,
                    mu=0.10,
                    theta_cxl=0.02,
                    p_buy=0.5,
                    band=5,
                    init_levels=3,
                    init_depth=5,
                    seed=3,
                ),
                solution=sol_hawkes,
                horizon=120.0,
                decision_interval=1.0,
                inventory_cap=Q_MAX,
                flow=HawkesFlow(excitation=0.35, decay_rho=0.90, p_buy=0.5),
                estimator=BivariateHawkes(_hawkes(0.45)),
            )

        a, b = _run(), _run()
        assert a["n_fills"] == b["n_fills"]
        assert a["sim_internal_mtm_pnl_final"] == b["sim_internal_mtm_pnl_final"]

    def test_receipt_seal_and_labels(self, receipt) -> None:
        assert receipt["kind"] == "hawkes_mm.v1"
        assert receipt["label"] == "SYNTHETIC"
        assert receipt["live_pnl_claim"] is False
        assert seal_errors(receipt) == []
        assert isinstance(receipt["receipt_sha256"], str)
        for key in _all_keys(receipt):
            low = key.lower()
            for tok in FORBIDDEN_HEADLINE_TOKENS:
                assert tok not in low, f"forbidden headline token {tok} in {key}"

    def test_receipt_blocks(self, receipt) -> None:
        ir = receipt["impulse_response"]
        assert abs(ir["impulse_half_life_residual"]) < 5e-3
        assert ir["impulse_peak_excess"] > 0.0
        pl = receipt["poisson_limit"]
        half = float(
            as_optimal_quotes(
                100.0,
                0.0,
                receipt["mm_params"]["gamma"],
                receipt["mm_params"]["sigma"],
                pl["tau_eval"],
                receipt["mm_params"]["kappa"],
            )["half_spread"]
        )
        assert 0.0 <= pl["max_abs_offset_residual"] < half
        ex = receipt["excitation"]
        assert np.all(np.isfinite(np.asarray(ex["spread_levels"])))

    def test_receipt_deterministic_hash(self) -> None:
        a, b = hawkes_mm_bench(seed=0, quick=True), hawkes_mm_bench(seed=0, quick=True)
        assert a["receipt_sha256"] == b["receipt_sha256"]
