"""Tests for quant_fund.models.xva — CVA / FVA / MVA Monte Carlo suite.

References: Gregory (2020), *The XVA Challenge* 2nd ed.; Pykhtin & Zhu
(2007), GSI RM 2(1) — CVA = (1-R) sum dPD(t) EE(t); Hull & White (2012),
"The FVA Debate", Risk — one-sided borrowing-cost FVA; Andersen, Choudhury
& Xing (2019), JPM — capital-charged MVA over margin-period-of-risk
windows; Li (2000) — Gaussian-copula default dependence for WWR.

All data here is SYNTHETIC (seeded simulated paths on a fixture book) —
engine-correctness evidence, never market evidence; no live counterparty
or trading claims.  Closed-form anchors: deterministic-exposure CVA =
(1-R) PD EE exactly; CVA monotone in hazard/LGD; FVA -> 0 as the funding
spread -> 0; MVA -> 0 as alpha -> 0; copula WWR CVA exceeds the
common-random-numbers rho = 0 baseline on a planted wrong-way scenario
(and reverses under the opposite sign).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.gaussian_copula_default import conditional_default_prob
from quant_fund.models.options import bs_price
from quant_fund.models.pair_vine_copula import cvine_structure, vine_sample
from quant_fund.models.xva import (
    ExposureSimulation,
    SyntheticBook,
    bench_xva,
    compute_cva,
    compute_fva,
    compute_mva,
    compute_xva_suite,
    cumulative_hazard,
    default_prob_increments,
    exposure_profile,
    simulate_exposure,
    survival_prob,
)
from quant_fund.research.catalog.registry import FORBIDDEN_RESEARCH_METRIC_KEYS

Array = np.ndarray


def _f(out: dict, key: str) -> float:
    val = out[key]
    assert isinstance(val, float)
    return val


def _a(out: dict, key: str) -> Array:
    val = out[key]
    assert isinstance(val, np.ndarray)
    return val


@pytest.fixture(scope="module")
def sim_small() -> ExposureSimulation:
    return simulate_exposure(SyntheticBook(), n_paths=3000, n_steps=120, seed=11)


@pytest.fixture(scope="module")
def sim_big() -> ExposureSimulation:
    return simulate_exposure(SyntheticBook(), n_paths=8000, n_steps=120, seed=12)


def _det_sim(ee: float, t_end: float, n: int = 200) -> ExposureSimulation:
    """Deterministic-exposure anchor sim: constant V = ee, zero rates."""
    times = np.linspace(0.0, t_end, n + 1)
    return ExposureSimulation.from_arrays(times, np.full((1, n + 1), ee), np.ones((1, n + 1)))


# ---------------------------------------------------------------- book


def test_book_derived_defaults_and_initial_values(sim_small: ExposureSimulation) -> None:
    book = sim_small.book
    assert book is not None
    assert book.horizon_resolved == pytest.approx(5.0)
    assert 0.0 < book.swap_fixed_rate < 0.2
    assert book.fx_fwd_strike_price > 0.0
    # par swap and initial-forward FX leg start at (numerically) zero value
    assert np.max(np.abs(book and sim_small.components["swap"][:, 0])) < 1e-4
    assert np.max(np.abs(sim_small.components["fx_forward"][:, 0])) < 1e-4
    # equity call at t=0 equals the repo's scalar BSM pricer (q = 0 default)
    expected = book.eq_opt_notional * bs_price(
        book.eq0, book.eq_opt_strike, book.eq_opt_maturity, book.sigma_eq, book.r0
    )
    assert sim_small.components["equity_call"][0, 0] == pytest.approx(expected, rel=1e-8)


def test_components_sum_to_portfolio_value(sim_small: ExposureSimulation) -> None:
    comp = sim_small.components
    rebuilt = comp["swap"] + comp["fx_forward"] + comp["equity_call"]
    assert np.allclose(sim_small.values, rebuilt, rtol=1e-13, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_fx": 0.0},
        {"sigma_eq": -0.1},
        {"kappa_r": 0.0},
        {"fx0": 0.0},
        {"swap_notional": -1.0},
        {"eq_opt_strike": 0.0},
        {"swap_freq": 3},
        {"swap_maturity": 2.3, "swap_freq": 2},
        {"swap_maturity": 7.0, "horizon": 5.0},
        {"horizon": -1.0},
        {"rho_rate_fx": 1.0},
        {"rho_rate_fx": 0.95, "rho_rate_eq": 0.95, "rho_fx_eq": -0.95},
        {"swap_fixed": float("nan")},
        {"fx_fwd_strike": -2.0},
    ],
)
def test_book_fail_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SyntheticBook(**kwargs)


# ---------------------------------------------------------------- simulate


def test_simulate_shape_finiteness_and_discounts(sim_small: ExposureSimulation) -> None:
    assert sim_small.values.shape == (3000, 121)
    assert sim_small.discounts.shape == sim_small.values.shape
    assert np.isfinite(sim_small.values).all()
    assert (sim_small.discounts > 0.0).all()
    assert np.all(sim_small.discounts[:, 0] == 1.0)
    assert sim_small.times[0] == 0.0 and sim_small.times[-1] == pytest.approx(5.0)
    assert sim_small.source == "SYNTHETIC"
    assert set(sim_small.factor_terminals) == {"rate", "fx", "equity"}


def test_simulate_determinism() -> None:
    book = SyntheticBook()
    a = simulate_exposure(book, n_paths=500, n_steps=36, seed=7)
    b = simulate_exposure(book, n_paths=500, n_steps=36, seed=7)
    c = simulate_exposure(book, n_paths=500, n_steps=36, seed=8)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.discounts, b.discounts)
    assert np.array_equal(a.factor_terminals["rate"], b.factor_terminals["rate"])
    assert not np.array_equal(a.values, c.values)


def test_simulate_fail_closed() -> None:
    book = SyntheticBook()
    with pytest.raises(ValueError, match="n_paths"):
        simulate_exposure(book, n_paths=1, n_steps=10)
    with pytest.raises(ValueError, match="n_steps"):
        simulate_exposure(book, n_paths=10, n_steps=1)
    with pytest.raises(ValueError, match="SyntheticBook"):
        simulate_exposure("not-a-book")  # type: ignore[arg-type]


def test_from_arrays_fail_closed() -> None:
    times = np.linspace(0.0, 1.0, 5)
    good_v = np.ones((2, 5))
    with pytest.raises(ValueError):
        ExposureSimulation.from_arrays(times, np.ones((2, 4)), np.ones((2, 4)))
    with pytest.raises(ValueError):
        ExposureSimulation.from_arrays(times, np.full((2, 5), np.nan), np.ones((2, 5)))
    with pytest.raises(ValueError):
        ExposureSimulation.from_arrays(times, good_v, np.zeros((2, 5)))
    with pytest.raises(ValueError):
        ExposureSimulation.from_arrays(np.linspace(0.5, 1.5, 5), good_v, np.ones((2, 5)))
    with pytest.raises(ValueError):
        ExposureSimulation.from_arrays(np.array([0.0, 1.0, 0.5, 2.0, 3.0]), good_v, np.ones((2, 5)))


# ---------------------------------------------------------------- exposure


def test_exposure_profile_curves(sim_small: ExposureSimulation) -> None:
    prof = exposure_profile(sim_small)
    n_times = sim_small.n_times
    for key in ("times", "ee", "epe", "ene", "q95", "q99", "discounted_ee", "mean_value"):
        arr = _a(prof, key)
        assert arr.shape == (n_times,), key
        assert np.isfinite(arr).all(), key
    assert (_a(prof, "ee") >= 0.0).all()
    assert (_a(prof, "ene") >= 0.0).all()
    assert (_a(prof, "discounted_ee") >= 0.0).all()
    assert (_a(prof, "q99") >= _a(prof, "q95") - 1e-12).all()
    # EPE is the running time-average of EE (Gregory 2020), epe(0) = ee(0)
    ee = _a(prof, "ee")
    epe = _a(prof, "epe")
    times = _a(prof, "times")
    assert epe[0] == pytest.approx(ee[0])
    expected_final = np.sum(0.5 * (ee[1:] + ee[:-1]) * np.diff(times)) / times[-1]
    assert epe[-1] == pytest.approx(expected_final, rel=1e-12)
    # EE(0) is consistent with the stored paths
    assert ee[0] == pytest.approx(float(np.mean(np.maximum(sim_small.values[:, 0], 0.0))))


# ---------------------------------------------------------------- hazard


def test_hazard_curve_utilities() -> None:
    times = np.linspace(0.0, 5.0, 11)
    lam = cumulative_hazard(times, 0.02)
    assert lam == pytest.approx(0.02 * times)
    surv = survival_prob(times, 0.02)
    assert surv[0] == 1.0 and np.all(np.diff(surv) < 0.0)
    inc = default_prob_increments(times, 0.02)
    assert inc.shape == (10,)
    assert float(inc.sum()) == pytest.approx(1.0 - surv[-1], abs=1e-15)
    assert (inc >= 0.0).all()
    # term hazard: piecewise-constant intensities accumulate additively
    term = np.full(10, 0.01)
    term[5:] = 0.03
    lam_t = cumulative_hazard(times, term)
    assert lam_t[-1] == pytest.approx(0.01 * 2.5 + 0.03 * 2.5)


def test_hazard_fail_closed() -> None:
    times = np.linspace(0.0, 5.0, 11)
    with pytest.raises(ValueError, match="hazard"):
        cumulative_hazard(times, -0.01)
    with pytest.raises(ValueError, match="term hazard"):
        cumulative_hazard(times, np.full(9, 0.01))
    with pytest.raises(ValueError, match="term hazard"):
        cumulative_hazard(times, np.full(10, -0.5))
    with pytest.raises(ValueError, match="times"):
        cumulative_hazard(np.array([1.0, 2.0]), 0.01)


# ------------------------------------------------------------------- CVA


def test_cva_closed_form_deterministic_exposure() -> None:
    # CVA of a deterministic exposure with zero rates = (1-R) PD EE exactly
    recovery, ee, lam, t_end = 0.4, 100.0, 0.02, 10.0
    out = compute_cva(_det_sim(ee, t_end), recovery=recovery, hazard=lam)
    expected = (1.0 - recovery) * ee * (1.0 - math.exp(-lam * t_end))
    assert _f(out, "cva") == pytest.approx(expected, rel=1e-12)
    # term-hazard variant telescopes to the cumulative hazard exactly
    times = np.linspace(0.0, t_end, 201)
    term = 0.01 + 0.005 * np.arange(times.size - 1) / (times.size - 1)
    out_t = compute_cva(_det_sim(ee, t_end), recovery=recovery, hazard=term)
    expected_t = (
        (1.0 - recovery) * ee * (1.0 - math.exp(-float(cumulative_hazard(times, term)[-1])))
    )
    assert _f(out_t, "cva") == pytest.approx(expected_t, rel=1e-12)
    # increments telescope to the total
    assert float(_a(out, "cva_increments").sum()) == pytest.approx(_f(out, "cva"), rel=1e-12)


def test_cva_zero_hazard_is_zero(sim_small: ExposureSimulation) -> None:
    out = compute_cva(sim_small, hazard=0.0)
    assert _f(out, "cva") == 0.0
    out_w = compute_cva(sim_small, hazard=0.0, mode="copula_wwr", rho_wwr=0.3, wwr_sign=-1)
    assert _f(out_w, "cva") == 0.0


def test_cva_monotone_in_hazard_and_recovery(sim_small: ExposureSimulation) -> None:
    cvas = [_f(compute_cva(sim_small, hazard=h), "cva") for h in (0.005, 0.01, 0.02, 0.04)]
    assert all(b > a for a, b in zip(cvas, cvas[1:], strict=False))
    by_rec = [_f(compute_cva(sim_small, hazard=0.02, recovery=r), "cva") for r in (0.2, 0.4, 0.6)]
    assert all(b < a for a, b in zip(by_rec, by_rec[1:], strict=False))
    # term-hazard bump: pointwise larger curve -> larger CVA
    n = sim_small.n_times - 1
    base = np.full(n, 0.01)
    bumped = np.full(n, 0.015)
    assert _f(compute_cva(sim_small, hazard=bumped), "cva") > _f(
        compute_cva(sim_small, hazard=base), "cva"
    )


def test_cva_fail_closed(sim_small: ExposureSimulation) -> None:
    with pytest.raises(ValueError, match="recovery"):
        compute_cva(sim_small, recovery=1.0)
    with pytest.raises(ValueError, match="recovery"):
        compute_cva(sim_small, recovery=-0.1)
    with pytest.raises(ValueError, match="hazard"):
        compute_cva(sim_small, hazard=-0.01)
    with pytest.raises(ValueError, match="mode"):
        compute_cva(sim_small, mode="hybrid")
    with pytest.raises(ValueError, match="rho_wwr"):
        compute_cva(sim_small, mode="copula_wwr", rho_wwr=1.0)
    with pytest.raises(ValueError, match="wwr_sign"):
        compute_cva(sim_small, mode="copula_wwr", rho_wwr=0.5, wwr_sign=0)
    with pytest.raises(ValueError, match="wwr_factor"):
        compute_cva(sim_small, mode="copula_wwr", rho_wwr=0.5, wwr_factor="credit")


def test_cva_wwr_requires_factor_simulated_book() -> None:
    det = _det_sim(100.0, 5.0)
    with pytest.raises(ValueError, match="factor"):
        compute_cva(det, mode="copula_wwr", rho_wwr=0.5)


def test_cva_wwr_zero_rho_matches_independent(sim_big: ExposureSimulation) -> None:
    # marginal default law is the hazard curve for any rho, so the rho = 0
    # copula MC reproduces the deterministic Pykhtin-Zhu CVA within MC error
    hazard, recovery = 0.04, 0.4
    ind = _f(compute_cva(sim_big, hazard=hazard, recovery=recovery), "cva")
    mc0 = _f(
        compute_cva(
            sim_big,
            hazard=hazard,
            recovery=recovery,
            mode="copula_wwr",
            rho_wwr=0.0,
            wwr_sign=-1,
            seed=21,
        ),
        "cva",
    )
    assert ind > 0.0
    assert abs(mc0 - ind) / ind < 0.12


def test_cva_wwr_contrast_planted_scenario(sim_big: ExposureSimulation) -> None:
    # planted WWR: the counterparty (our option writer) defaults when equity
    # RALLIES (wwr_sign=-1 on the equity latent) — exactly when the long call
    # is deep in the money, the classic wrong-way setup.  Common random
    # numbers: identical seed for the copula epsilons across the three runs.
    hazard, recovery, rho, seed = 0.04, 0.4, 0.6, 21
    kw = {
        "hazard": hazard,
        "recovery": recovery,
        "mode": "copula_wwr",
        "rho_wwr": rho,
        "wwr_factor": "equity",
    }
    wwr = compute_cva(sim_big, **kw, wwr_sign=-1, seed=seed)
    rwr = compute_cva(sim_big, **kw, wwr_sign=1, seed=seed)
    base = compute_cva(
        sim_big,
        hazard=hazard,
        recovery=recovery,
        mode="copula_wwr",
        rho_wwr=0.0,
        wwr_factor="equity",
        wwr_sign=-1,
        seed=seed,
    )
    c_wwr, c_base, c_rwr = _f(wwr, "cva"), _f(base, "cva"), _f(rwr, "cva")
    # classic WWR result: copula-coupled CVA exceeds the independent baseline
    assert c_wwr > 1.2 * c_base
    # mirrored right-way-risk scenario reverses the ordering
    assert c_rwr < 0.8 * c_base
    assert c_wwr > c_rwr
    # marginal preservation: realised default frequency ~ hazard-curve PD
    pd_T = 1.0 - math.exp(-hazard * float(sim_big.times[-1]))
    assert _f(wwr, "default_rate_horizon") == pytest.approx(pd_T, abs=0.02)
    assert _f(wwr, "pd_horizon") == pytest.approx(pd_T, rel=1e-12)
    assert _f(base, "default_rate_horizon") == pytest.approx(pd_T, abs=0.02)


def test_cva_wwr_conditional_pd_matches_repo_copula(sim_big: ExposureSimulation) -> None:
    # the copula scheme's conditional default probabilities must equal the
    # repo's one-factor Gaussian-copula machinery (Li 2000 convention)
    rho, seed = 0.6, 21
    out = compute_cva(
        sim_big,
        hazard=0.04,
        recovery=0.4,
        mode="copula_wwr",
        rho_wwr=rho,
        wwr_sign=-1,
        seed=seed,
    )
    pd_T = _f(out, "pd_horizon")
    nodes = _a(out, "wwr_m_nodes")
    cond = _a(out, "wwr_conditional_pd")
    manual = np.array(
        [float(conditional_default_prob(np.array([pd_T]), rho, float(m))[0]) for m in nodes]
    )
    assert np.allclose(cond, manual, atol=1e-12)
    # empirical check: per-path default frequency binned on the signed latent
    m = _a(out, "wwr_latent_signed_m")
    tau = _a(out, "wwr_default_times")
    defaulted = np.isfinite(tau) & (tau <= float(sim_big.times[-1]))
    for node in (-1.0, 0.0, 1.0):
        sel = np.abs(m - node) <= 0.25
        assert int(sel.sum()) >= 100
        emp = float(np.mean(defaulted[sel]))
        exp = float(conditional_default_prob(np.array([pd_T]), rho, node)[0])
        assert abs(emp - exp) < 0.05


def test_cva_wwr_coupling_matches_vine_copula_machinery(sim_big: ExposureSimulation) -> None:
    # xva's WWR latent uses the Li (2000) LOADING convention (as in
    # gaussian_copula_default): L = sqrt(rho) M + sqrt(1-rho) eps, i.e. a
    # Gaussian pair-copula with CORRELATION sqrt(rho) — exactly the gaussian
    # h-inverse in models/vine_copula.py.  Cross-check both directions.
    rho = 0.6
    # high hazard -> virtually every path defaults inside the horizon, so the
    # latent is recoverable from tau without censoring: Phi(L) = 1 - e^{-lam tau}
    out = compute_cva(
        sim_big,
        hazard=2.0,
        recovery=0.4,
        mode="copula_wwr",
        rho_wwr=rho,
        wwr_factor="equity",
        wwr_sign=-1,
        seed=21,
    )
    tau = _a(out, "wwr_default_times")
    m = _a(out, "wwr_latent_signed_m")
    ok = np.isfinite(tau)
    assert float(ok.mean()) > 0.99
    latent = np.asarray(stats.norm.ppf(-np.expm1(-2.0 * tau[ok])), dtype=float)
    assert np.corrcoef(m[ok], latent)[0, 1] == pytest.approx(math.sqrt(rho), abs=0.04)
    # the repo's vine sampler with a gaussian pair-copula at correlation
    # sqrt(rho) reproduces the same latent coupling
    vm = cvine_structure(2)
    vm.families[(0, 0)] = "gaussian"
    vm.params[(0, 0)] = {"rho": math.sqrt(rho)}
    u = vine_sample(vm, 20_000, seed=3)
    z = np.asarray(stats.norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12)), dtype=float)
    assert np.corrcoef(z[:, 0], z[:, 1])[0, 1] == pytest.approx(math.sqrt(rho), abs=0.02)


def test_cva_wwr_determinism_and_increments(sim_small: ExposureSimulation) -> None:
    kw = {"hazard": 0.03, "mode": "copula_wwr", "rho_wwr": 0.4, "wwr_sign": -1}
    a = compute_cva(sim_small, **kw, seed=99)
    b = compute_cva(sim_small, **kw, seed=99)
    c = compute_cva(sim_small, **kw, seed=100)
    assert _f(a, "cva") == _f(b, "cva")
    assert np.array_equal(_a(a, "wwr_default_times"), _a(b, "wwr_default_times"))
    assert _f(a, "cva") != _f(c, "cva")
    assert float(_a(a, "cva_increments").sum()) == pytest.approx(_f(a, "cva"), rel=1e-9, abs=1e-9)


# ------------------------------------------------------------------- FVA


def test_fva_zero_spread_vanishes(sim_small: ExposureSimulation) -> None:
    out = compute_fva(sim_small, funding_spread=0.0)
    assert _f(out, "fva") == 0.0
    assert _f(out, "fva_funding_cost") == 0.0
    assert "hull_white_2012" in str(out["convention"])


def test_fva_linear_and_monotone_in_spread(sim_small: ExposureSimulation) -> None:
    f1 = _f(compute_fva(sim_small, funding_spread=0.003), "fva")
    f2 = _f(compute_fva(sim_small, funding_spread=0.006), "fva")
    f3 = _f(compute_fva(sim_small, funding_spread=0.010), "fva")
    assert f1 > 0.0
    assert f2 == pytest.approx(2.0 * f1, rel=1e-12)
    assert f3 > f2 > f1


def test_fva_full_collateral_zero_and_broadcast(sim_small: ExposureSimulation) -> None:
    # VM-style full collateralisation leaves no funding gap
    out = compute_fva(sim_small, funding_spread=0.005, collateral=sim_small.values)
    assert _f(out, "fva") == 0.0
    # (n_times,) schedule broadcasts across paths; zeros == uncollateralised
    zeros_sched = np.zeros(sim_small.n_times)
    a = _f(compute_fva(sim_small, funding_spread=0.005, collateral=zeros_sched), "fva")
    b = _f(compute_fva(sim_small, funding_spread=0.005), "fva")
    assert a == pytest.approx(b, rel=1e-15)


def test_fva_benefit_flag_nets(sim_small: ExposureSimulation) -> None:
    plain = compute_fva(sim_small, funding_spread=0.005)
    netted = compute_fva(sim_small, funding_spread=0.005, include_funding_benefit=True)
    cost = _f(plain, "fva_funding_cost")
    benefit = _f(netted, "fva_funding_benefit")
    assert _f(plain, "fva") == pytest.approx(cost, rel=1e-15)
    assert benefit > 0.0  # the book does go out-of-the-money on some paths
    assert _f(netted, "fva") == pytest.approx(cost - benefit, rel=1e-12)
    assert _f(netted, "fva") < cost
    assert "two_sided" in str(netted["convention"])


def test_fva_gap_curve_and_fail_closed(sim_small: ExposureSimulation) -> None:
    out = compute_fva(sim_small, funding_spread=0.005)
    gap = _a(out, "funding_gap_curve")
    assert gap.shape == (sim_small.n_times,)
    assert (gap >= 0.0).all() and np.isfinite(gap).all()
    with pytest.raises(ValueError, match="funding_spread"):
        compute_fva(sim_small, funding_spread=-0.001)
    with pytest.raises(ValueError, match="collateral"):
        compute_fva(sim_small, funding_spread=0.005, collateral=np.full(sim_small.n_times, np.nan))
    with pytest.raises(ValueError):
        compute_fva(sim_small, funding_spread=0.005, collateral=np.zeros(5))


# ------------------------------------------------------------------- MVA


def test_mva_zero_alpha_vanishes(sim_small: ExposureSimulation) -> None:
    out = compute_mva(sim_small, alpha=0.0)
    assert _f(out, "mva") == 0.0
    assert np.all(_a(out, "capital_curve") == 0.0)
    assert "andersen" in str(out["convention"])


def test_mva_linear_in_alpha_and_charge_and_monotone_in_quantile(
    sim_small: ExposureSimulation,
) -> None:
    m1 = _f(compute_mva(sim_small, alpha=0.03), "mva")
    m2 = _f(compute_mva(sim_small, alpha=0.06), "mva")
    assert m1 > 0.0
    assert m2 == pytest.approx(2.0 * m1, rel=1e-12)
    q95 = _f(compute_mva(sim_small, quantile=0.95), "mva")
    q99 = _f(compute_mva(sim_small, quantile=0.99), "mva")
    assert q99 > q95 > 0.0
    r1 = _f(compute_mva(sim_small, capital_cost_rate=0.05), "mva")
    r2 = _f(compute_mva(sim_small, capital_cost_rate=0.10), "mva")
    assert r2 == pytest.approx(2.0 * r1, rel=1e-12)


def test_mva_capital_curve(sim_small: ExposureSimulation) -> None:
    out = compute_mva(sim_small, alpha=0.06, quantile=0.99)
    cap = _a(out, "capital_curve")
    vol = _a(out, "exposure_vol_curve")
    assert cap.shape == (sim_small.n_times,)
    assert (cap >= 0.0).all() and np.isfinite(cap).all()
    assert (vol >= 0.0).all() and np.isfinite(vol).all()
    assert float(cap.max()) > 0.0


def test_mva_fail_closed(sim_small: ExposureSimulation) -> None:
    with pytest.raises(ValueError, match="alpha"):
        compute_mva(sim_small, alpha=-0.01)
    with pytest.raises(ValueError, match="quantile"):
        compute_mva(sim_small, quantile=0.4)
    with pytest.raises(ValueError, match="quantile"):
        compute_mva(sim_small, quantile=1.0)
    with pytest.raises(ValueError, match="mpor_years"):
        compute_mva(sim_small, mpor_years=0.0)
    with pytest.raises(ValueError, match="capital_cost_rate"):
        compute_mva(sim_small, capital_cost_rate=-0.1)


# ------------------------------------------------------------------ suite


def test_xva_suite_labels_and_positivity(sim_small: ExposureSimulation) -> None:
    out = compute_xva_suite(sim_small, hazard=0.02, funding_spread=0.005, alpha=0.06)
    assert out["source"] == "SYNTHETIC"
    assert out["claim"] == "diagnostics_only"
    assert out["cva_mode"] == "independent"
    assert _f(out, "cva") > 0.0
    assert _f(out, "fva") > 0.0
    assert _f(out, "mva") > 0.0
    assert _f(out, "epe_final") > 0.0


# ------------------------------------------------------------------ bench


def test_bench_xva_deterministic_labeled_and_clean() -> None:
    kw = {"n_paths": 1200, "n_steps": 60, "seed": 5}
    b1 = bench_xva(**kw)
    b2 = bench_xva(**kw)
    assert b1.keys() == b2.keys()
    for k in b1:
        assert b1[k] == b2[k], k
    expected_keys = {
        "synthetic_xva_cva",
        "synthetic_xva_fva",
        "synthetic_xva_mva",
        "synthetic_xva_cva_mc_rho0",
        "synthetic_xva_cva_wwr",
        "synthetic_xva_cva_rwr",
        "synthetic_xva_wwr_uplift",
        "synthetic_xva_rwr_relief",
        "synthetic_xva_epe_final",
        "synthetic_xva_ee_q99_max",
        "synthetic_xva_pd_horizon",
        "synthetic_xva_closed_form_abs_err",
        "synthetic_source",
        "synthetic_claim",
        "synthetic_dgp",
    }
    assert expected_keys <= set(b1)
    assert b1["synthetic_source"] == "SYNTHETIC"
    assert b1["synthetic_claim"] == "diagnostics_only"
    # closed-form anchor: deterministic-exposure CVA reproduced exactly
    assert _f(b1, "synthetic_xva_closed_form_abs_err") < 1e-8
    # planted WWR uplift positive even at bench size (CRN baseline)
    assert _f(b1, "synthetic_xva_wwr_uplift") > 0.0
    assert (
        _f(b1, "synthetic_xva_cva") > 0.0
        and _f(b1, "synthetic_xva_fva") > 0.0
        and _f(b1, "synthetic_xva_mva") > 0.0
    )
    # honesty: no forbidden headline-metric tokens in bench keys
    for key in b1:
        tokens = set(str(key).lower().split("_"))
        assert not tokens & FORBIDDEN_RESEARCH_METRIC_KEYS, key


def test_bench_xva_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        bench_xva(n_paths=1)
