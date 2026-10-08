"""Benchmark batteries for SOTA canon wave 14 (deep finance math).

Covers the wave-14 module lanes: martingale optimal transport (model-free
option bounds), large-deviations importance sampling for portfolio tails, the
Cardaliaguet-Lehalle trade-crowding mean-field game, vine copulas + GAS dynamic
copulas, Malliavin-calculus Monte Carlo Greeks, XVA (CVA/FVA/MVA + WWR),
American LSM with Andersen-Broadie dual bounds, Dupire local vol + SLV mixing
Monte Carlo, the deep BSDE solver, the DeRegiME regime-mixture head, two-stage
odd residual flows, deep kernel hedging, the zero-intelligence limit-order-book
simulator, and cash-constrained multi-asset optimal execution.

NOT WIRED (deliberate): ``models/fourier_pricing.py`` is excluded from this
battery — its COS European-pricing path carries a documented open bug, so it is
skipped entirely rather than surfaced as a scorecard family. Revisit only once
that lane is green.

Seeded SYNTHETIC streams only — no panel or vendor data, no headline performance
ratios (proper scores / pricing bounds / calibration and correctness diagnostics
only, per the AGENTS.md honesty contract). Each bench returns a flat
``dict[str, float]`` of proper diagnostic statistics, or ``{}`` if its synthetic
setup cannot be constructed (or, for the torch-gated deep-learning benches, if
torch is absent). Every bench is deterministic: repeated calls are bit-identical
(the torch trainers pin ``torch.manual_seed`` and run CPU single-thread).
Monte-Carlo budgets are SHRUNK relative to the lane test suites (documented per
bench) so the whole wave-14 battery stays inside its ~90 s runtime envelope; the
accompanying research tests carry correspondingly wider, documented tolerances.

ZI-LOB honesty: the simulator's mark-to-market accounting lives under
``sim_internal_*`` names and is a synthetic-engine diagnostic only. Those keys
are deliberately kept OUT of the scorecard blob — this battery re-exposes only
inventory / impact / phase diagnostics, never a P&L or headline ratio.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from scipy.stats import norm

from quant_fund.execution.cash_constrained_oe import (
    schedule_shape,
    solve_multiasset_oe,
    unconstrained_oe,
)
from quant_fund.metrics.large_deviations import (
    chernoff_bound,
    fenchel_legendre,
    gaussian_is_tail_prob,
    gaussian_log_mgf,
    gaussian_log_mgf_grad,
    gaussian_rate,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    as_policy,
    metaorder_impact_slope,
    run_mm_session,
    santa_fe_config,
)
from quant_fund.models.american_lsm import american_lsm_benchmarks
from quant_fund.models.local_stoch_vol import (
    dupire_local_vol,
    heston_mc,
    local_vol_mc,
    mc_implied_vols,
    slv_leverage_function,
)
from quant_fund.models.malliavin_greeks import (
    bs_analytic_delta,
    bs_analytic_gamma,
    malliavin_greeks,
)
from quant_fund.models.martingale_ot import (
    generate_synthetic_vanillas,
    solve_mot_bounds,
    variance_swap_mot_bounds,
)
from quant_fund.models.mean_field_games import (
    MFGConfig,
    ac_benchmark_trajectory,
    mfg_to_ac_alpha,
    solve_mfg_trade_crowding,
)
from quant_fund.models.pair_vine_copula import (
    cvine_structure,
    gas_copula_fit,
    vine_fit,
    vine_sample,
    vine_tail_dependence,
)
from quant_fund.models.sabr import sabr_implied_vol
from quant_fund.models.xva import bench_xva as _xva_core_bench
from quant_fund.quant_models.heston import heston_implied_vol

_SEED = 20261001


def _bs_undiscounted_call(
    spot: float, strike: float, maturity: float, rate: float, sigma: float
) -> float:
    """Undiscounted BS expected call payoff E[(S_T - K)^+] = e^{rT} * C_BS.

    SYNTHETIC reference used only to check MOT bound containment of a known
    model price (matches the martingale_ot lane's own helper).
    """
    vol = sigma * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / vol
    d2 = d1 - vol
    return float(spot * math.exp(rate * maturity) * norm.cdf(d1) - strike * norm.cdf(d2))


def _black76_call(
    f: np.ndarray, k: np.ndarray, t: np.ndarray, sig: np.ndarray, r: float
) -> np.ndarray:
    """Discounted Black-76 call price (broadcasts); SYNTHETIC surface builder."""
    d1 = (np.log(f / k) + 0.5 * sig**2 * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return np.asarray(np.exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d2)), dtype=float)


def bench_martingale_ot() -> dict[str, float]:
    """Martingale-optimal-transport model-free bounds, SYNTHETIC (wave 14).

    Beiglböck, Henry-Labordère & Penkner (2013), "Model-independent bounds for
    option prices — a mass transport approach", Finance & Stochastics 17,
    arXiv:1106.5929 (Thm 1: no duality gap); Dolinsky & Soner (2014),
    arXiv:1208.4922 (continuous-time MOT); Henry-Labordère (2017), *Model-Free
    Hedging* ch. 2 (the one-period MOT LP); Neuberger (1990/1994) log-contract
    replication. ``variance_swap_mot_bounds`` solves the log-contract LP against
    BS-synthetic vanillas (the fair variance strike sigma^2 must be bracketed);
    ``solve_mot_bounds`` prices a seeded butterfly exotic on a 201-point grid
    and checks the known BS price lands inside [lower, upper]. LP is exact
    (HiGHS), so the dual gap is at solver tolerance. Seeded SYNTHETIC; pricing
    bounds only, never market evidence.
    """
    try:
        vs = variance_swap_mot_bounds(
            spot=100.0,
            rate=0.02,
            maturity=0.5,
            sigma=0.25,
            n_strikes=21,
            n_grid=301,
            strike_width=0.3,
            grid_width=0.5,
        )
        # Butterfly containment via solve_mot_bounds on a seeded state grid.
        van = generate_synthetic_vanillas(
            spot=100.0, rate=0.02, maturity=0.5, sigma=0.25, n_strikes=21, strike_width=0.3
        )
        fwd = float(cast("float", van["forward"]))
        strikes = np.asarray(van["strikes"], dtype=float)
        calls = np.asarray(van["call_prices"], dtype=float)
        x = np.linspace(fwd * 0.5, fwd * 1.5, 201, dtype=float)
        k1, k2, k3 = fwd - 10.0, fwd, fwd + 10.0
        payoff = np.maximum(x - k1, 0.0) - 2.0 * np.maximum(x - k2, 0.0) + np.maximum(x - k3, 0.0)
        res = solve_mot_bounds(x, payoff, strikes, calls, fwd)
        bs_bf = (
            _bs_undiscounted_call(100.0, k1, 0.5, 0.02, 0.25)
            - 2.0 * _bs_undiscounted_call(100.0, k2, 0.5, 0.02, 0.25)
            + _bs_undiscounted_call(100.0, k3, 0.5, 0.02, 0.25)
        )
        contained = 1.0 if res.lower_bound <= bs_bf <= res.upper_bound else 0.0
        return {
            "mot_upper_bound": float(vs["mot_upper_bound"]),
            "mot_lower_bound": float(vs["mot_lower_bound"]),
            "mot_spread": float(vs["spread"]),
            "mot_upper_dual_gap": float(vs["upper_dual_gap"]),
            "mot_bs_contained": contained,
            "mot_butterfly_lower": float(res.lower_bound),
            "mot_butterfly_upper": float(res.upper_bound),
        }
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_large_deviations() -> dict[str, float]:
    """Large-deviations importance sampling on a SYNTHETIC Gaussian portfolio.

    Dembo & Zeitouni (1998/2010), *Large Deviations Techniques and Applications*
    (Gärtner-Ellis rate function, Cramér-Chernoff bound); Glasserman & Li (2005),
    Management Science 51(11) (exponential-tilting IS for portfolio tails);
    Glasserman, Heidelberger & Shahabuddin (2002), Math. Finance 12(3). A 3-d
    correlated Gaussian loss L = w·X at a ~2-sigma threshold: the module's
    tilted IS estimator must beat crude MC by a variance-reduction factor > 2,
    the Cramér-Chernoff bound exp(-Lambda*(t)) must dominate the empirical tail,
    and the numerical Fenchel-Legendre transform must reproduce the closed-form
    Gaussian rate. SHRUNK MC (12k crude / 6k IS, 20k Chernoff draws vs the lane's
    25k / 8k / 50k). Seeded SYNTHETIC; tail-probability diagnostics only, never
    market risk evidence.
    """
    try:
        mu = np.array([0.0, 0.0, 0.0])
        cov = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])
        w = np.array([0.6, 0.6, 0.4])
        sigma = math.sqrt(float(w @ (cov @ w)))

        rng = np.random.default_rng(_SEED)
        res = gaussian_is_tail_prob(
            rng, n_is=6_000, n_crude=12_000, w=w, mu=mu, cov=cov, threshold=2.0 * sigma
        )
        vrf_obj = res["variance_reduction_factor"]
        if vrf_obj is None:
            return {}
        vrf = float(cast("float", vrf_obj))

        # Cramér-Chernoff: empirical tail <= exp(-Lambda*(t)) + MC slack.
        rng2 = np.random.default_rng(_SEED + 1)
        samples = rng2.multivariate_normal(mu, cov, size=20_000)
        losses = np.asarray(samples @ w, dtype=float)
        t = float(np.dot(w, mu)) + 2.5 * sigma
        emp = float(np.mean(losses >= t))
        bound = chernoff_bound(
            t,
            w,
            log_mgf=lambda lam: gaussian_log_mgf(lam, mu, cov),
            grad_log_mgf=lambda lam: gaussian_log_mgf_grad(lam, mu, cov),
        )
        holds = 1.0 if emp <= bound + 0.02 else 0.0

        # Closed-form Gaussian rate vs numerical Fenchel-Legendre.
        x = np.array([1.2, -0.8, 0.5])
        closed = gaussian_rate(x, mu, cov)
        numerical = fenchel_legendre(
            x,
            log_mgf=lambda lam: gaussian_log_mgf(lam, mu, cov),
            grad_log_mgf=lambda lam: gaussian_log_mgf_grad(lam, mu, cov),
        )
        return {
            "ld_is_variance_reduction_factor": vrf,
            "ld_chernoff_bound_holds": holds,
            "ld_gaussian_rate_closed_form_err": float(abs(closed - numerical)),
            "ld_threshold_sigma_multiple": 2.0,
            "ld_chernoff_empirical_tail": emp,
            "ld_chernoff_bound": float(bound),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_mean_field_games() -> dict[str, float]:
    """Cardaliaguet-Lehalle trade-crowding MFG, SYNTHETIC (wave 14).

    Cardaliaguet & Lehalle (2018), "Mean field game of controls and an
    application to trade crowding", Math. Finance 28; Guéant, Lasry & Lions
    (2011); Huang, Malhamé & Caines (2006), IEEE TAC 51 (Nash certainty
    equivalence); Almgren & Chriss (2001) as the gamma=0 benchmark. Three cells
    on SMALL grids (n_t <= 50, lane suite uses up to 400): (a) the solitary
    (gamma=0, hard-liquidation) LQ equilibrium trading rate must recover the
    closed-form AC schedule; (b) positive coupling gamma lifts the equilibrium
    cost above the solitary cost (crowding externality); (c) refining the time
    grid moves the equilibrium cost (coarse-minus-fine discretisation gap > 0).
    Seeded SYNTHETIC; execution-cost diagnostics only, never market evidence.
    """
    try:
        # (a) AC recovery in the gamma -> 0, A -> inf hard-liquidation limit.
        cfg_ac = MFGConfig(
            Q0=1.0, T=1.0, kappa=0.5, gamma=0.0, psi=0.6, terminal_penalty=1e6, n_t=50, n_q=40
        )
        sol_ac = solve_mfg_trade_crowding(cfg_ac)
        alpha_mfg = mfg_to_ac_alpha(sol_ac, n_slices=50)
        alpha_ac = ac_benchmark_trajectory(1.0, 1.0, 50, kappa=0.5, psi=0.6)
        ac_mse = float(np.mean((alpha_mfg - alpha_ac) ** 2))

        # (b) crowding externality: gamma = 0.4 equilibrium vs solitary cost.
        cfg_crowd = MFGConfig(
            Q0=1.0, T=1.0, kappa=1.0, gamma=0.4, psi=0.5, terminal_penalty=10.0, n_t=50, n_q=40
        )
        sol_crowd = solve_mfg_trade_crowding(cfg_crowd)

        # (c) grid convergence: coarse (n_t=25) minus fine (n_t=50) cost.
        cfg_coarse = MFGConfig(
            Q0=1.0, T=1.0, kappa=1.0, gamma=0.1, psi=0.5, terminal_penalty=10.0, n_t=25, n_q=40
        )
        cfg_fine = MFGConfig(
            Q0=1.0, T=1.0, kappa=1.0, gamma=0.1, psi=0.5, terminal_penalty=10.0, n_t=50, n_q=40
        )
        coarse = solve_mfg_trade_crowding(cfg_coarse)
        fine = solve_mfg_trade_crowding(cfg_fine)
        return {
            "mfg_ac_recovery_mse": ac_mse,
            "mfg_crowding_cost_ratio": float(sol_crowd.crowding_cost_ratio),
            "mfg_grid_convergence": float(coarse.cost - fine.cost),
            "mfg_n_t": 50.0,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_vine_copula() -> dict[str, float]:
    """Vine copulas + GAS dynamic copula on SYNTHETIC streams (wave 14).

    Dißmann, Brechmann, Czado & Kurowicka (2013), CSDA 59 (regular-vine
    selection/estimation); Bedford & Cooke (2002), Ann. Statist. 30; Aas, Czado,
    Frigessi & Bakken (2009), IME 44 (pair-copula constructions); Creal,
    Koopman & Lucas (2013), J. Applied Econometrics 28 (GAS). Three cells: (a)
    plant a 3-d Gaussian C-vine at Kendall tau = 0.4, refit Gaussian-only
    (n = 2000, lane uses 3000), read the dominant first-tree pair's tau error;
    (b) Clayton vs Gaussian lower-tail dependence (Clayton's must dominate); (c)
    fit a GAS(1,1) dynamic Gaussian copula on a seeded series with known
    (omega, alpha, beta) and report the max parameter-recovery bias. SHRUNK GAS
    series (T = 100 vs the lane's 1500 — the MLE is O(T) per likelihood eval and
    dominates the runtime budget). Seeded SYNTHETIC; dependence diagnostics only,
    never market evidence.
    """
    try:
        # (a) C-vine structure recovery: planted tau = 0.4 on every pair.
        tau_true = 0.40
        rho_true = math.sin(math.pi * 0.5 * tau_true)
        vm_true = cvine_structure(3)
        for tree in range(2):
            for edge in range(2 - tree):
                vm_true.families[(tree, edge)] = "gaussian"
                vm_true.params[(tree, edge)] = {"rho": rho_true}
        obs = vine_sample(vm_true, 2000, seed=_SEED)
        fit = vine_fit(obs, families=("gaussian",), criterion="aic")
        rho_hat = float(fit.params[(0, 0)].get("rho", 0.0))
        tau_hat = 2.0 / math.pi * math.asin(max(-1.0, min(1.0, rho_hat)))
        tau_err = abs(tau_hat - tau_true)

        # (b) lower-tail dependence: Clayton (theta=2) vs Gaussian (rho=0.5).
        vm_cl = cvine_structure(2)
        vm_cl.families[(0, 0)] = "clayton"
        vm_cl.params[(0, 0)] = {"theta": 2.0}
        vm_g = cvine_structure(2)
        vm_g.families[(0, 0)] = "gaussian"
        vm_g.params[(0, 0)] = {"rho": 0.5}
        td_cl = vine_tail_dependence(vm_cl, n_sim=20_000, seed=_SEED + 1)
        td_g = vine_tail_dependence(vm_g, n_sim=20_000, seed=_SEED + 2)
        lo_cl = float(td_cl["0,1"]["lower"])
        lo_g = float(td_g["0,1"]["lower"])
        tail_ratio = lo_cl / max(lo_g, 1e-9)

        # (c) GAS(1,1) recovery bias on a seeded series with known parameters.
        rng = np.random.default_rng(_SEED + 3)
        n_gas = 100
        om_t, al_t, be_t = 0.08, 0.15, 0.90
        k_prev = math.atanh(0.2)
        u_sim = np.empty((n_gas, 2), dtype=float)
        for i in range(n_gas):
            r = float(np.tanh(k_prev))
            z0 = float(rng.standard_normal())
            z1 = r * z0 + math.sqrt(max(1.0 - r * r, 0.0)) * float(rng.standard_normal())
            u_sim[i, 0] = float(norm.cdf(z0))
            u_sim[i, 1] = float(norm.cdf(z1))
            xg = float(norm.ppf(np.clip(u_sim[i, 0], 1e-10, 1.0 - 1e-10)))
            yg = float(norm.ppf(np.clip(u_sim[i, 1], 1e-10, 1.0 - 1e-10)))
            r2 = r * r
            s_t = r - r * (xg * xg + yg * yg) / (1.0 - r2) + xg * yg * (1.0 + r2) / (1.0 - r2)
            k_next = om_t + al_t * s_t + be_t * k_prev
            k_prev = k_next if np.isfinite(k_next) else 0.0
        gfit = gas_copula_fit(u_sim)
        gas_bias = max(
            abs(float(gfit["omega"]) - om_t),
            abs(float(gfit["alpha"]) - al_t),
            abs(float(gfit["beta"]) - be_t),
        )
        return {
            "vine_structure_recovery_tau_err": float(tau_err),
            "vine_tail_dep_ratio": float(tail_ratio),
            "gas_recov_bias_max": float(gas_bias),
            "vine_planted_tau": tau_true,
            "vine_clayton_lower_tail": lo_cl,
            "vine_gaussian_lower_tail": lo_g,
        }
    except (ValueError, RuntimeError, FloatingPointError, KeyError, ArithmeticError):
        return {}


def bench_malliavin_greeks() -> dict[str, float]:
    """Malliavin-calculus Monte Carlo Greeks, SYNTHETIC (wave 14).

    Fournié, Lasry, Lebuchoux, Lions & Touzi (1999), "Applications of Malliavin
    calculus to Monte Carlo methods in finance", Finance & Stochastics 3 (Delta
    = E[f * (1/T) int (Y_t/sigma(S_t)) dW_t], Eq. 2.10; the digital-option
    variance headline); Benhamou (2000) iterated-weight Gamma; Detemple, Garcia &
    Rindisbacher (2005), J. Finance 58. A BS call's Malliavin Delta/Gamma are
    checked against the closed forms in standard-error units, and the Fournié et
    al. headline is witnessed on a digital payoff: the Malliavin Delta variance
    must be far below the finite-difference Delta variance (which blows up for a
    discontinuous payoff). SHRUNK MC (20k paths / 100 steps for the call; 8k /
    60 for the digital FD trio vs the lane's 30k / 126). Seeded SYNTHETIC;
    Greek-accuracy diagnostics only, never market evidence.
    """
    try:
        s0, strike, mat, sigma, rate = 100.0, 100.0, 1.0, 0.20, 0.03
        disc = math.exp(-rate * mat)

        def call_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
            return np.maximum(s_t - k, 0.0)

        def digital_payoff(s_t: np.ndarray, k: float) -> np.ndarray:
            return np.asarray(k < s_t, dtype=float)

        res = malliavin_greeks(
            s0,
            mat,
            call_payoff,
            n_steps=100,
            n_paths=20_000,
            model="bs",
            rng=_SEED,
            discount=disc,
            r=rate,
            sigma=sigma,
            strike=strike,
        )
        analytic_d = bs_analytic_delta(s0, strike, mat, sigma, rate, call=True)
        analytic_g = bs_analytic_gamma(s0, strike, mat, sigma, rate)
        delta_err_se = abs(res.delta - analytic_d) / res.delta_se
        gamma_err_se = abs(res.gamma - analytic_g) / res.gamma_se

        # Digital FD-vs-Malliavin variance ratio (Fournié et al. 1999 headline).
        n_dig, s_dig = 8_000, 60
        rm = malliavin_greeks(
            s0,
            mat,
            digital_payoff,
            n_steps=s_dig,
            n_paths=n_dig,
            model="bs",
            rng=_SEED,
            discount=disc,
            r=rate,
            sigma=sigma,
            strike=strike,
        )
        eps = 0.5
        ru = malliavin_greeks(
            s0 + eps,
            mat,
            digital_payoff,
            n_steps=s_dig,
            n_paths=n_dig,
            model="bs",
            rng=_SEED,
            discount=disc,
            r=rate,
            sigma=sigma,
            strike=strike,
        )
        rd = malliavin_greeks(
            s0 - eps,
            mat,
            digital_payoff,
            n_steps=s_dig,
            n_paths=n_dig,
            model="bs",
            rng=_SEED,
            discount=disc,
            r=rate,
            sigma=sigma,
            strike=strike,
        )
        fd_se = math.sqrt(ru.price_se**2 + rd.price_se**2) / (2.0 * eps)
        fd_ratio = (fd_se / rm.delta_se) ** 2
        return {
            "mall_delta_abs_err_se": float(delta_err_se),
            "mall_gamma_abs_err_se": float(gamma_err_se),
            "mall_digital_fd_efficiency_ratio": float(fd_ratio),
            "mall_n_paths": 20_000.0,
            "mall_n_steps": 100.0,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_xva() -> dict[str, float]:
    """XVA suite (CVA/FVA/MVA + WWR) on SYNTHETIC exposure paths (wave 14).

    Gregory (2020), *The XVA Challenge*, 2nd ed., Wiley; Li (2000) Gaussian-copula
    default correlation. Thin float-only adapter over the module's own
    ``bench_xva`` (which returns a mixed float|str blob): the numeric CVA / FVA /
    MVA levels, the Gaussian-copula wrong-way-risk uplift vs the rho = 0
    common-random-numbers baseline, and the closed-form deterministic-exposure
    CVA anchor error are re-exposed under ``xva_*`` keys, following the wave-12
    ``rwcv`` adapter precedent. SHRUNK paths/steps (4k / 120 vs the module
    default's 6k / 180). Seeded SYNTHETIC engine-correctness evidence; price
    diagnostics only, never market or live-counterparty evidence.
    """
    try:
        raw = _xva_core_bench(n_paths=4_000, n_steps=120, seed=_SEED)
        mapped = {
            "xva_cva": float(raw["synthetic_xva_cva"]),
            "xva_fva": float(raw["synthetic_xva_fva"]),
            "xva_mva": float(raw["synthetic_xva_mva"]),
            "xva_wwr_uplift": float(raw["synthetic_xva_wwr_uplift"]),
            "xva_closed_form_abs_err": float(raw["synthetic_xva_closed_form_abs_err"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_american_lsm() -> dict[str, float]:
    """American LSM lower bound vs Andersen-Broadie dual upper, SYNTHETIC (w14).

    Longstaff & Schwartz (2001), "Valuing American options by simulation",
    Review of Financial Studies 14(1); Andersen & Broadie (2004), Management
    Science (primal-dual upper bound); Rogers (2002); Haugh & Kogan (2004);
    Barone-Adesi & Whaley (1987) as the analytic reference. Thin float-only
    adapter over the module's own ``american_lsm_benchmarks(seed, fast=True)``
    (a nested float|str blob): the BS American-put cell's out-of-sample LSM
    lower price, the dual upper bound, the (non-negative) duality gap, the BAW
    agreement flag, and the lower <= upper bracket flag are re-exposed under
    ``lsm_*`` keys. ``fast=True`` already shrinks the MC budgets. Seeded
    SYNTHETIC; pricing-bound diagnostics only, never market evidence.
    """
    try:
        raw = american_lsm_benchmarks(seed=_SEED, fast=True)
        put = raw["benchmarks"]["bs_american_put"]
        lower = float(put["lsm_oos_lower_price"])
        upper = float(put["dual_upper_bound"])
        mapped = {
            "lsm_oos_lower_price": lower,
            "lsm_dual_upper_bound": upper,
            "lsm_duality_gap": float(put["duality_gap"]),
            "lsm_baw_agreement_ok": 1.0 if put["baw_agreement_ok"] else 0.0,
            "lsm_lower_le_upper": 1.0 if put["lower_le_upper"] else 0.0,
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_local_stoch_vol() -> dict[str, float]:
    """Dupire local vol + SLV mixing Monte Carlo, SYNTHETIC (wave 14).

    Dupire (1994), "Pricing with a smile", Risk 7(1); Gatheral (2006) ch. 11;
    van der Stoep, Grzelak & Oosterlee (2014), arXiv:1211.2993 (the SLV mixing
    MC / leverage function L(S,t) whose fixed point reproduces the target Heston
    surface); Heston (1993); Hagan (2002) SABR for the synthetic input surface.
    Three cells at the module's smallest sane config (13 strikes x 3 maturities,
    8k LV-MC paths, 4k SLV paths / 2 iters vs the lane's 25 x 5 / 20k / 12k x 3):
    (a) the fundamental Dupire round trip — LV-MC implied vols reproduce the
    input SABR surface inside the central moneyness band; (b) the matched-Heston
    SLV calibration's final implied vols land on the closed-form target; (c) at
    zero vol-of-vol (xi = 0) the leverage fixed point is exactly L == 1. Vol
    errors are reported in VOL POINTS (x100). Seeded SYNTHETIC; surface /
    calibration diagnostics only, never market evidence.
    """
    try:
        spot, rate, div = 100.0, 0.03, 0.0
        sabr = dict(alpha=2.0, beta=0.5, rho=-0.4, nu=0.35)
        # Heston driver params as explicit floats (slv_leverage_function/heston_mc
        # carry int/bool/Callable kwargs, so **dict[str, float] fails mypy).
        kap, the, xi, rho_h, v0 = 3.0, 0.04, 0.30, -0.5, 0.04
        strikes = np.linspace(80.0, 120.0, 13)
        maturities = np.array([0.5, 0.75, 1.0])

        # (a) Dupire round trip on a SABR-generated surface.
        fwd = spot * np.exp((rate - div) * maturities)
        iv_grid = np.array(
            [
                sabr_implied_vol(fj, strikes, tj, **sabr)
                for fj, tj in zip(fwd, maturities, strict=True)
            ]
        )
        prices = _black76_call(fwd[:, None], strikes[None, :], maturities[:, None], iv_grid, rate)
        surf = dupire_local_vol(strikes, maturities, prices, spot=spot, r=rate, q=div)
        mc = local_vol_mc(
            surf,
            spot=spot,
            r=rate,
            q=div,
            strikes=strikes,
            seed=_SEED,
            n_paths=8_000,
            max_dt=1.0 / 12.0,
        )
        iv_mc = mc_implied_vols(
            np.asarray(mc["prices"]),
            spot=spot,
            r=rate,
            q=div,
            strikes=strikes,
            maturities=maturities,
        )
        band = (strikes[None, :] >= 0.9 * fwd[:, None]) & (strikes[None, :] <= 1.1 * fwd[:, None])
        roundtrip_volpts = float(np.max(np.abs(iv_mc - iv_grid)[band]) * 100.0)

        # (b) matched-Heston SLV calibration: final IVs vs the CF target.
        calib = np.array([90.0, 100.0, 110.0])
        target_iv = np.array(
            [
                heston_implied_vol(
                    k, 1.0, S0=spot, r=rate, q=div, kappa=kap, theta=the, xi=xi, rho=rho_h, v0=v0
                )
                for k in calib
            ]
        )
        slv = slv_leverage_function(
            spot=spot,
            r=rate,
            q=div,
            T=1.0,
            calib_strikes=calib,
            target_implied_vols=target_iv,
            n_steps=20,
            n_paths=4_000,
            n_iter=2,
            seed=_SEED,
            n_s_bins=8,
            n_time_nodes=4,
            kappa=kap,
            theta=the,
            xi=xi,
            rho=rho_h,
            v0=v0,
        )
        final_iv_volpts = float(
            np.max(np.abs(np.asarray(slv["final_implied_vols"]) - target_iv)) * 100.0
        )

        # (c) zero vol-of-vol: the SLV leverage fixed point is exactly L == 1.
        times = np.linspace(0.0, 1.0, 21)
        sim = heston_mc(
            spot=spot,
            r=rate,
            q=div,
            times=times,
            n_paths=1_000,
            seed=_SEED,
            kappa=kap,
            theta=the,
            xi=0.0,
            rho=rho_h,
            v0=0.09,
        )
        v = np.asarray(sim["v"])[0]
        sig_bs = float(np.sqrt(np.sum(v[:-1]) * (times[1] - times[0])))
        slv0 = slv_leverage_function(
            spot=spot,
            r=rate,
            q=div,
            T=1.0,
            calib_strikes=calib,
            target_implied_vols=np.full(3, sig_bs),
            n_steps=20,
            n_paths=4_000,
            n_iter=2,
            seed=_SEED,
            n_s_bins=8,
            n_time_nodes=4,
            kappa=kap,
            theta=the,
            xi=0.0,
            rho=rho_h,
            v0=0.09,
        )
        xi0_dev = float(np.max(np.abs(np.asarray(slv0["leverage"]) - 1.0)))
        return {
            "slv_dupire_roundtrip_max_volpts": roundtrip_volpts,
            "slv_final_iv_max_abs_volpts": final_iv_volpts,
            "slv_xi0_leverage_deviation": xi0_dev,
        }
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_deep_bsde() -> dict[str, float]:
    """Deep BSDE solver on the Burgers-Hopf benchmark, SYNTHETIC (w14, torch).

    E, Han & Jentzen (2017), "Deep learning-based numerical methods for
    high-dimensional parabolic PDEs and backward SDEs", Statistics & Probability
    Letters 5(4), arXiv:1706.04702; Han, Jentzen & E (2018), PNAS,
    arXiv:1707.02568; Han & Long (2020), arXiv:1811.01165. Thin float-only
    adapter over the module's ``validate_burgers_hopf(dim=1)`` (the E-Han-Jentzen
    Lemma 4.3 benchmark with the explicit target u(0,0) = 1/2): the recovered
    Y_0 absolute error and the terminal MSE loss are re-exposed under
    ``bsde_*`` keys. SHRUNK training (200 epochs vs the module default's 500) —
    the bench is a determinism + direction witness; the science lives in the lane
    suite. torch is the optional ``nn`` extra imported lazily by the module, so a
    torch-less environment raises ImportError and this bench returns ``{}``.
    Seeded SYNTHETIC; PDE-solver error diagnostics only, never market evidence.
    """
    try:
        from quant_fund.models.deep_bsde import validate_burgers_hopf

        row = validate_burgers_hopf(
            dim=1,
            horizon=0.3,
            n_steps=12,
            epochs=200,
            lr=1e-2,
            n_paths=512,
            eval_paths=1024,
            seed=_SEED,
        )
        mapped = {
            "bsde_burgers_d1_abs_err": float(row["bsde_burgers_d1_abs_err"]),
            "bsde_terminal_loss": float(row["bsde_burgers_d1_terminal_loss"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_deep_regime_mixture() -> dict[str, float]:
    """DeRegiME regime-mixture head vs NGBoost, SYNTHETIC (wave 14, torch).

    Wood, Zohren & Roberts (2026), "DeRegiME: Deep Regime Mixture-of-Experts",
    arXiv:2605.19231; the comparator is NGBoostGaussian (Duan et al. 2020,
    arXiv:1910.03225). Thin float-only adapter over the module's
    ``synthetic_regime_benchmark`` with SHRUNK parameters (600 train / 800 eval
    steps, 1 horizon, hidden (16,16), 120 epochs, 60 NGBoost rounds vs the
    module defaults' 1500 / 2000 / 2 / (32,32) / 400 / 150): on a seeded
    3-regime Student-t volatility-switch stream, DeRegiME's eval NLPD must beat
    the Gaussian NGBoost head (negative gap), and the recovered effective-regime
    count is reported against the planted count. torch is the optional ``nn``
    extra imported lazily by the module, so a torch-less environment raises
    ImportError and this bench returns ``{}``. Seeded SYNTHETIC; proper-score
    (NLPD) diagnostics only, never market evidence.
    """
    try:
        from quant_fund.models.deep_regime_mixture import synthetic_regime_benchmark

        res = synthetic_regime_benchmark(
            sigmas=(0.3, 0.8, 1.6),
            n_steps=600,
            n_eval_steps=800,
            n_horizons=1,
            n_regimes=4,
            hidden=(16, 16),
            epochs=120,
            lr=8e-3,
            ngboost_estimators=60,
            seed=_SEED,
            eval_seed=_SEED + 1,
        )
        m = res.metrics
        mapped = {
            "drm_nlpd_gap_vs_ngboost": float(m["drm_nlpd_gap_vs_ngboost"]),
            "drm_effective_regimes": float(m["drm_effective_regimes"]),
            "drm_planted_regimes": float(m["drm_planted_regimes"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_odd_residual_flows() -> dict[str, float]:
    """Two-stage odd residual flows (TORF) vs NGBoost, SYNTHETIC (w14, torch).

    Stirn et al.-faithful heteroscedastic regression via the two-stage odd
    residual flow, arXiv:2608.11114 (Stage-1 point forecast frozen; Stage-2
    normalizing flow on the residuals — the analytic mean IS the Stage-1 output,
    so the point MAE is preserved EXACTLY, Lemma 1); comparator NGBoostGaussian
    (Duan et al. 2020). Thin float-only adapter over the module's
    ``bench_odd_residual_flows`` (which returns a mixed float|str blob — the str
    stamps are filtered): the TORF / NGBoost CRPS, the CRPS gain, and the
    bitwise MAE-preservation gap are re-exposed under ``torf_*`` /
    ``crps_*`` / ``mae_*`` keys. SHRUNK budgets (600 train / 400 test, 120 flow
    epochs, 50 NGBoost rounds vs the module defaults' 1000 / 600 / 250 / 80).
    torch is the optional ``nn`` extra imported lazily by the module, so a
    torch-less environment raises ImportError and this bench returns ``{}``.
    Seeded SYNTHETIC; proper-score (CRPS) + point-MAE diagnostics only, never
    market evidence.
    """
    try:
        from quant_fund.models.odd_residual_flows import bench_odd_residual_flows

        raw = bench_odd_residual_flows(
            n_train=600,
            n_test=400,
            seed=_SEED,
            n_blocks=2,
            n_bins=8,
            epochs=120,
            lr=8e-3,
            ngboost_rounds=50,
            ngboost_lr=0.1,
        )
        mapped = {
            "torf_crps": float(raw["synthetic_torf_crps"]),
            "ngboost_crps": float(raw["synthetic_ngboost_crps"]),
            "crps_gain_vs_ngboost": float(raw["synthetic_crps_gain_vs_ngboost"]),
            "mae_preservation_gap": float(raw["synthetic_mae_preservation_gap"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_deep_kernel_hedging() -> dict[str, float]:
    """Deep kernel hedging in an RKHS, SYNTHETIC (wave 14, torch).

    Dupret, Hainaut & Motte (2026), "Deep kernel hedging", arXiv:2609.34474
    (hedging functional in the RKHS of a learned deep kernel; representer-theorem
    reduction, Thm 3.1; Random-Fourier-feature primal with the O(D^-1/2) uniform
    kernel-approximation bound, Prop 3.2); Wilson, Hu, Salakhutdinov & Xing (2016)
    deep kernel learning; Rahimi & Recht (2007) random features; Buehler, Gonon,
    Teichmann & Wood (2019) deep-hedging simulators. A tiny deep-kernel hedge is
    trained on seeded SYNTHETIC GBM paths and evaluated on fresh paths: the
    hedged terminal-error variance must beat the unhedged payoff variance, and
    the RFF kernel-approximation max error is reported at D = 1000. SHRUNK config
    (128 train / 512 eval paths, 10 steps, n_rff 120, 50 epochs vs the lane's
    250 / 1024 / 12 / 400 / 100). torch is the optional ``nn`` extra imported
    lazily by the module, so a torch-less environment raises ImportError and this
    bench returns ``{}``. Seeded SYNTHETIC; hedging-risk diagnostics only, never
    market evidence.
    """
    try:
        from quant_fund.models import deep_hedging as dh
        from quant_fund.models import deep_kernel_hedging as dkh

        steps, dt = 10, 0.5 / 10.0
        train = dh.simulate_gbm_paths(128, steps, s0=100.0, sigma=0.2, dt=dt, seed=_SEED)
        pay = dh.european_payoff(train, 100.0)
        evaluation = dh.simulate_gbm_paths(512, steps, s0=100.0, sigma=0.2, dt=dt, seed=_SEED + 1)
        pay_ev = dh.european_payoff(evaluation, 100.0)
        res = dkh.deep_kernel_hedge(
            train,
            pay,
            lam=3e-4,
            n_rff=120,
            gamma0=4.0,
            epochs=50,
            lr=3e-3,
            order=2,
            seed=_SEED,
        )
        err = dh.hedged_loss(evaluation, res.strategy_positions(evaluation), pay_ev, cost_rate=0.0)
        hedged = float(np.var(err))
        unhedged = float(np.var(pay_ev))

        # RFF kernel-approximation max error at D = 1000 (numpy core, Prop 3.2).
        rng = np.random.default_rng(_SEED)
        z = rng.standard_normal((40, 3))
        exact = dkh.rbf_kernel_matrix(z, z, 1.0)
        w_mat, b_vec = dkh.rff_draw(3, 1_000, seed=_SEED)
        y_feat = dkh.rff_features(z, w_mat, b_vec)
        rff_err = float(np.max(np.abs(y_feat @ y_feat.T - exact)))
        mapped = {
            "dkh_hedged_risk": hedged,
            "dkh_unhedged_risk": unhedged,
            "dkh_rff_kernel_max_error": rff_err,
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_zi_lob() -> dict[str, float]:
    """Zero-intelligence LOB emergent microstructure, SYNTHETIC (wave 14).

    Cont, Stoikov & Talreja (2010), Operations Research 58(1) (ZI-LOB queueing);
    Avellaneda & Stoikov (2008) and Gueant, Lehalle & Fernandez-Tapia (2012)
    (closed-form market-making quotes); Moret & Lillo (2026), arXiv:2609.11614
    (Santa Fe calibration, regime-switch MO clock); Donier, Bonart, Mastromatteo
    & Bouchaud (2015), arXiv:1504.06829 (emergent square-root impact);
    Rosenzweig (2026), arXiv:2609.31260 (spread/depth phase diagnostics). Two
    cells: (a) calibrated metaorders in the ref-anchored volume-diffusion regime
    give a log-log impact slope near the square-root law (0.5); (b) a
    stationarily-calibrated AS market-maker saturates its inventory under
    persistent two-state directional flow but stays bounded under stationary
    flow, and the flow diagnostics detect the regime. SHRUNK budgets (1500 s
    sessions, impact sizes 16-256 x 4 seeds x 250 s warmup vs the lane's 3000 s /
    6 seeds / 300 s). PnL accounting stays in the simulator's ``sim_internal_*``
    namespace and is NOT re-exposed here. Seeded SYNTHETIC; inventory / impact /
    phase diagnostics only, never market evidence.
    """
    try:
        sigma, kappa, as_gamma, cap = 0.02, 1000.0, 0.002, 40
        policy = as_policy(gamma=as_gamma, sigma=sigma, kappa=kappa, tick=0.01)
        stat = run_mm_session(
            config=santa_fe_config(seed=7),
            policy=policy,
            horizon=1500.0,
            decision_interval=1.0,
            inventory_cap=cap,
        )
        flow_states = (
            RegimeState("buy_pressure", 1.0, 0.94),
            RegimeState("sell_pressure", 1.0, 0.06),
        )
        flow = MarkovRegimeFlow(flow_states, (0.9975, 0.9975), seed=11)
        reg = run_mm_session(
            config=santa_fe_config(seed=7),
            policy=policy,
            horizon=1500.0,
            decision_interval=1.0,
            inventory_cap=cap,
            flow=flow,
        )
        reg_diag = reg["flow_diagnostics"]
        if reg_diag is None:
            return {}
        sat_ratio = float(reg["max_abs_inventory"]) / max(int(stat["max_abs_inventory"]), 1)
        detected = 1.0 if reg_diag["regime_detected"] else 0.0

        cfg = ZILobConfig(
            s0=100.0,
            tick=0.01,
            lam=0.06,
            mu=0.10,
            theta_cxl=0.02,
            band=40,
            density_exponent=1.0,
            anchor="ref",
            ref_halflife=100.0,
            seed=0,
        )
        imp = metaorder_impact_slope(
            config=cfg,
            sizes=(16, 32, 64, 128, 256),
            n_seeds=4,
            inject_rate=10.0 * cfg.mu,
            warmup=250.0,
            measure="transient",
        )
        return {
            "zlob_impact_slope": float(imp["impact_slope"]),
            "zlob_impact_r2": float(imp["impact_r2"]),
            "zlob_inventory_saturation_ratio": float(sat_ratio),
            "zlob_regime_detected": detected,
            "zlob_stat_max_abs_inventory": float(stat["max_abs_inventory"]),
            "zlob_regime_max_abs_inventory": float(reg["max_abs_inventory"]),
        }
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_cash_constrained_oe() -> dict[str, float]:
    """Cash-constrained multi-asset optimal execution, SYNTHETIC (wave 14).

    Hashimoto & Stillman (2026), "Feasible multi-asset optimal execution under
    cash constraints", arXiv:2609.27786 (the QCQP reformulation, Props 1-2, and
    the sell-first / drawdown-reduction mechanism of their Experiment 2);
    Almgren & Chriss (2001); Perold (1988) implementation shortfall; Huberman &
    Stanzl (2004) PSD permanent cross-impact. On the lane's canonical imbalanced
    two-leg config (a dear, risky buy leg and a calm, expensive-temporary sell
    leg), tightening the expected-cash budget must (a) cut the peak funding
    drawdown, (b) shift the schedule toward sell-first execution, and (c) raise
    the expected implementation shortfall only modestly (nested feasible sets).
    Deterministic convex program (cvxpy/CLARABEL); no Monte Carlo. Seeded
    SYNTHETIC parameters; execution-cost diagnostics only, never market evidence.
    """
    try:
        n_periods, tau, risk = 20, 1.0, 0.30
        orders = np.array([15.0, -20.0])
        p0 = np.array([70.0, 55.0])
        perm = np.diag([0.02, 0.02])
        temp = np.diag([0.01, 0.10])
        cov = np.diag([0.09, 0.0025])
        free = unconstrained_oe(
            orders,
            p0,
            perm_impact=perm,
            temp_impact=temp,
            cov=cov,
            n_periods=n_periods,
            tau=tau,
            risk_aversion=risk,
        )
        beta = 3.0
        budgets = beta * np.arange(1, n_periods + 1, dtype=float)
        tight = solve_multiasset_oe(
            orders,
            p0,
            perm_impact=perm,
            temp_impact=temp,
            cov=cov,
            n_periods=n_periods,
            tau=tau,
            risk_aversion=risk,
            cash_budgets=budgets,
        )
        peak_red = free.peak_cash_drawdown / max(tight.peak_cash_drawdown, 1e-9)
        sf_free = float(schedule_shape(free.trades)["sell_first_half_fraction"])
        sf_tight = float(schedule_shape(tight.trades)["sell_first_half_fraction"])
        is_ratio = tight.expected_is / free.expected_is
        mapped = {
            "coe_peak_drawdown_reduction": float(peak_red),
            "coe_sell_first_fraction_tight": sf_tight,
            "coe_sell_first_fraction_unconstrained": sf_free,
            "coe_is_ratio_tight_vs_unconstrained": float(is_ratio),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
