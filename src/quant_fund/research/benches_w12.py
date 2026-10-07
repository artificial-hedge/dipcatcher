"""Benchmark batteries for SOTA canon wave 12.

Covers the wave-12 module lanes: NexCP nonexchangeable conformal bounds,
multi-horizon EnbPI, Bayesian stacking / pseudo-BMA(+), regime-weighted
conformal VaR, sliced-Wasserstein two-sample testing, proper-score
decompositions, anytime-valid confidence sequences, and (torch-gated) deep
hedging.

Seeded SYNTHETIC streams only — no panel or vendor data, no headline
performance ratios (proper scores / coverage / interval diagnostics only, per
the AGENTS.md honesty contract). Each bench returns a flat ``dict[str, float]``
of proper diagnostic statistics, or ``{}`` if its synthetic setup cannot be
constructed (or, for the torch-gated deep-hedging bench, if torch is absent).
Every bench is deterministic: repeated calls are bit-identical.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.stats import norm

from quant_fund.metrics.confidence_sequences import (
    check_time_uniform_coverage,
    empirical_bernstein_cs,
    subgaussian_cs,
    width_decay_slope,
    width_ratio_vs_fixed,
    wsr_cs,
)
from quant_fund.metrics.score_decomposition import (
    brier_decomposition,
    broecker_ensemble_crps_decomposition,
    kolassa_crps_decomposition,
    mean_ranked_probability_score,
    sharpness_decomposition,
)
from quant_fund.metrics.sliced_wasserstein import (
    sliced_wasserstein_barycenter,
    sliced_wasserstein_distance,
    sliced_wasserstein_test,
)
from quant_fund.metrics.wasserstein import wasserstein_1d
from quant_fund.models.enbpi_multihorizon import MultiHorizonEnbPI
from quant_fund.models.nexcp import NexCPSplit
from quant_fund.models.regime_conformal_var import bench_regime_weighted_conformal_var
from quant_fund.models.stacking import (
    gaussian_log_dens_matrix,
    pseudo_bma_weights,
    stacked_crps,
    stacked_log_score,
    stacking_weights,
)

_SEED = 20260929

# Declared total-variation budget per calibration point for the NexCP Theorem
# 2/3 coverage bracket. The population TV distances are not estimable from the
# data without assumptions (see the nexcp module docstring), so the bench
# declares a conservative constant budget and asserts the empirical coverage
# lands inside the resulting [lower, upper] bracket.
_NEXCP_TV_BUDGET = 0.2


class _RidgeRegressor:
    """Minimal pure-numpy ridge regressor for the multi-horizon EnbPI ensemble.

    Satisfies the ``quant_fund.models.enbpi.Regressor`` protocol (``fit`` /
    ``predict``) so benches_w12 stays numpy/scipy-only (no sklearn). A tiny L2
    term keeps the normal equations well conditioned on bootstrap resamples.
    """

    def __init__(self, ridge: float = 1e-6) -> None:
        self._ridge = float(ridge)
        self._coef: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _RidgeRegressor:
        xm = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float).ravel()
        gram = xm.T @ xm + self._ridge * np.eye(xm.shape[1])
        self._coef = np.asarray(np.linalg.solve(gram, xm.T @ yv), dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("_RidgeRegressor.predict called before fit")
        return np.asarray(np.asarray(X, dtype=float) @ self._coef, dtype=float)


def bench_nexcp() -> dict[str, float]:
    """NexCP split conformal under SYNTHETIC covariate shift (wave 12).

    Barber, Candès, Ramdas & Tibshirani (2023), "Conformal prediction beyond
    exchangeability", Ann. Statist. 51(2), arXiv:2202.13415. Calibration
    residuals carry a covariate-dependent scale; the test covariate law is
    shifted upward, so likelihood-ratio trust weights (Eq. (10)/(11)) widen the
    interval to restore coverage while classical uniform-weight split conformal
    under-covers. Theorem 2/3 coverage bounds are evaluated at a declared
    total-variation budget. Seeded SYNTHETIC stream; correctness diagnostic
    only, never market evidence.
    """
    try:
        rng = np.random.default_rng(_SEED)
        n_cal, n_test = 600, 600
        delta = 1.0
        alpha = 0.10
        nominal = 1.0 - alpha
        # Calibration covariates x ~ N(0, 1); residual scale grows with x.
        x_cal = rng.standard_normal(n_cal)
        r_cal = np.abs(rng.standard_normal(n_cal)) * np.exp(0.6 * x_cal)
        # Test covariates shifted to N(delta, 1) -> systematically larger scale.
        x_test = delta + rng.standard_normal(n_test)
        r_test = np.abs(rng.standard_normal(n_test)) * np.exp(0.6 * x_test)
        # Likelihood-ratio trust weights for N(delta, 1) vs N(0, 1).
        w_lr = np.exp(delta * x_cal - 0.5 * delta * delta)

        weighted = NexCPSplit(alpha=alpha).calibrate(r_cal, w_lr)
        q_w = weighted.quantile_
        cov_w = float(np.mean(r_test <= q_w))
        uniform = NexCPSplit(alpha=alpha).calibrate(r_cal, np.ones(n_cal))
        q_u = uniform.quantile_
        cov_u = float(np.mean(r_test <= q_u))
        bounds = weighted.coverage_bounds(np.full(n_cal, _NEXCP_TV_BUDGET))
        return {
            "nexcp_alpha": alpha,
            "nexcp_nominal_coverage": nominal,
            "nexcp_coverage_shift": cov_w,
            "nexcp_coverage_gap_vs_nominal": abs(cov_w - nominal),
            "nexcp_unweighted_coverage": cov_u,
            "nexcp_interval_width_ratio": q_w / q_u,
            "nexcp_bound_lower": bounds.lower,
            "nexcp_bound_upper": bounds.upper,
            "nexcp_bound_gap": bounds.gap,
            "nexcp_effective_sample_size": float(weighted.effective_sample_size_),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_mh_enbpi() -> dict[str, float]:
    """Multi-horizon EnbPI on a SYNTHETIC AR(1) stream (wave 12).

    Xu & Xie (2021, ICML; 2023, IEEE TPAMI), "Conformal prediction for time
    series", arXiv:2010.09107, extended to H = 3 horizons with per-horizon
    residual ensembles, h-step-delayed online updates, and a Bonferroni joint
    coverage bound (``bonferroni_joint`` calibrates each horizon at alpha / H).
    Base learner is a pure-numpy ridge regressor. Seeded SYNTHETIC stream;
    correctness diagnostic only, never market evidence.
    """
    try:
        rng = np.random.default_rng(_SEED)
        n_total, split = 520, 340
        horizons = 3
        alpha = 0.10
        # AR(1) with a mild autoregressive coefficient and unit innovations.
        phi = 0.7
        y = np.empty(n_total, dtype=float)
        y[0] = float(rng.standard_normal())
        for t in range(1, n_total):
            y[t] = phi * y[t - 1] + float(rng.standard_normal())
        # Design at index i uses [y_i, y_{i-1}, 1]; horizon h targets y_{i+h}.
        idx = np.arange(1, n_total)
        features = np.column_stack([y[idx], y[idx - 1], np.ones(idx.size)])
        target = y[idx]
        model = MultiHorizonEnbPI(
            _RidgeRegressor,
            horizons=horizons,
            n_estimators=40,
            alpha=alpha,
            block_size=1,
            seed=_SEED,
            bonferroni_joint=True,
        )
        model.fit(features[:split], target[:split])
        result = model.predict_online(features[split:], target[split:])
        marginal = np.asarray(result.marginal_coverage, dtype=float)
        widths = np.asarray(result.mean_width, dtype=float)
        per_horizon_target = 1.0 - alpha / horizons
        return {
            "mh_enbpi_horizons": float(horizons),
            "mh_enbpi_per_horizon_target": per_horizon_target,
            "mh_enbpi_coverage_h1": float(marginal[0]),
            "mh_enbpi_coverage_h2": float(marginal[1]),
            "mh_enbpi_coverage_h3": float(marginal[2]),
            "mh_enbpi_coverage_h1_abs_err": abs(float(marginal[0]) - per_horizon_target),
            "mh_enbpi_joint_coverage": float(result.joint_coverage),
            "mh_enbpi_joint_coverage_bound": float(result.joint_coverage_bound),
            "mh_enbpi_width_h1": float(widths[0]),
            "mh_enbpi_width_h3": float(widths[2]),
            "mh_enbpi_width_growth_h3_h1": float(widths[2] / widths[0]),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_stacking() -> dict[str, float]:
    """Bayesian stacking / pseudo-BMA(+) on a SYNTHETIC Gaussian forecast set.

    Yao, Vehtari, Simpson & Gelman (2018), "Using stacking to average Bayesian
    predictive distributions", Bayesian Analysis 13(1), arXiv:1704.02030;
    Vehtari, Gelman & Gabry (2017); Clyde, Ghosh & Littman (2011, Bayesian
    bootstrap for pseudo-BMA+); Gneiting & Raftery (2007, proper scores). Three
    Gaussian candidates forecast a N(0, 1) truth: the true generator, a
    mean-misspecified, and a variance-misspecified component. Stacking must
    recover the true generator (weight > 0.5) and never lose log score / CRPS
    versus the best single candidate. Seeded SYNTHETIC; proper scores only.
    """
    try:
        rng = np.random.default_rng(_SEED)
        n, k = 2000, 3
        n_crps, n_q = 300, 15
        y = rng.standard_normal(n)
        mu = np.zeros((n, k))
        mu[:, 1] = 2.5  # mean-misspecified
        sigma = np.ones((n, k))
        sigma[:, 2] = 3.0  # variance-misspecified (too wide)
        log_dens = gaussian_log_dens_matrix(y, mu, sigma)

        w_stack = stacking_weights(log_dens)
        equal = np.full(k, 1.0 / k)
        single_scores = [stacked_log_score(log_dens, np.eye(k)[j]) for j in range(k)]
        best_single_logscore = max(single_scores)
        stack_logscore = stacked_log_score(log_dens, w_stack)
        equal_logscore = stacked_log_score(log_dens, equal)

        # CRPS of the stacked predictive vs the best single component.
        levels = np.linspace(0.05, 0.95, n_q)
        zq = norm.ppf(levels)
        quants = mu[:n_crps][:, :, None] + sigma[:n_crps][:, :, None] * zq[None, None, :]
        y_c = y[:n_crps]
        stack_crps = stacked_crps(y_c, quants, w_stack, levels)
        single_crps = [stacked_crps(y_c, quants, np.eye(k)[j], levels) for j in range(k)]
        best_single_crps = min(single_crps)

        w_plain = pseudo_bma_weights(log_dens, bb=False)
        w_bb = pseudo_bma_weights(log_dens, bb=True, n_boot=256, seed=_SEED)
        positive = w_stack[w_stack > 0.0]
        entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
        return {
            "stacking_n_candidates": float(k),
            "stacking_true_weight": float(w_stack[0]),
            "stacking_true_weight_recovered": 1.0 if int(np.argmax(w_stack)) == 0 else 0.0,
            "stacking_logscore_gain": stack_logscore - best_single_logscore,
            "stacking_vs_equal_logscore_gain": stack_logscore - equal_logscore,
            "stacking_crps_gain": best_single_crps - stack_crps,
            "pseudo_bma_weight_top1": float(np.max(w_plain)),
            "pseudo_bma_plus_weight_top1": float(np.max(w_bb)),
            "stacking_weight_entropy": entropy,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_rwcv() -> dict[str, float]:
    """Regime-weighted conformal VaR on a SYNTHETIC two-regime HMM (wave 12).

    Tibshirani, Barber, Candès & Ramdas (2019), arXiv:1802.09184 (weighted split
    conformal); Vovk, Gammerman & Shafer (2005); Lei et al. (2018); Hamilton
    (1989). Thin float-only adapter over the module's own
    ``bench_regime_weighted_conformal_var`` (which returns a mixed float/str
    blob): it re-exposes the numeric coverage/width diagnostics under ``rwcv_*``
    keys so the scorecard blob stays float-only like the other wave batteries.
    ORACLE regime posteriors — a correctness check, never market evidence.
    """
    try:
        raw = bench_regime_weighted_conformal_var()
        mapped = {
            "rwcv_unconditional_coverage": float(raw["synthetic_coverage"]),
            "rwcv_highvol_coverage": float(raw["synthetic_highvol_coverage"]),
            "rwcv_highvol_coverage_gain": float(raw["synthetic_highvol_coverage_gain"]),
            "rwcv_unweighted_highvol_coverage": float(raw["synthetic_unweighted_highvol_coverage"]),
            "rwcv_mean_width": float(raw["synthetic_mean_width"]),
            "rwcv_alpha": float(raw["synthetic_alpha"]),
            "rwcv_n": float(raw["synthetic_n"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sliced_wasserstein() -> dict[str, float]:
    """Sliced-Wasserstein distance / two-sample test on SYNTHETIC clouds.

    Rabin, Flamary, Cuturi & Villani (2010); Bonneel et al. (2015, barycenter);
    Nadjahi et al. (2021), arXiv:2003.07817; Phipson & Smyth (2010, permutation
    p-values); Ramdas, Garcia Trillos & Cuturi (2017, consistency). Checks the
    d = 1 exactness SW_1 == W_1, self-distance 0, calibrated null size, and
    shift-detection power over seeded permutation trials, plus identical-input
    barycenter convergence. Seeded SYNTHETIC; distributional diagnostics only.
    """
    try:
        rng = np.random.default_rng(_SEED)
        n = 2000
        a = rng.standard_normal(n)
        b = a + 1.0
        # d = 1 exactness: SW_1 with the single deterministic direction == W_1.
        sw_d1 = sliced_wasserstein_distance(
            a.reshape(-1, 1), b.reshape(-1, 1), p=1.0, n_projections=1, projection="deterministic"
        )
        w1 = wasserstein_1d(a, b, p=1.0)
        # self-distance vanishes.
        cloud = rng.standard_normal((300, 2))
        sw_self = sliced_wasserstein_distance(cloud, cloud.copy(), n_projections=32, seed=_SEED)

        # Null size and shift-detection power over seeded permutation trials.
        n_trials, n_perm, dim, m = 40, 199, 2, 60
        null_rejections = 0
        power_rejections = 0
        for t in range(n_trials):
            base = rng.standard_normal((m, dim))
            null_b = rng.standard_normal((m, dim))
            res_null = sliced_wasserstein_test(
                base, null_b, n_projections=16, seed=_SEED + t, n_perm=n_perm
            )
            if float(res_null["pvalue"]) < 0.05:
                null_rejections += 1
            shift_b = base + np.array([1.0, 0.0])
            res_alt = sliced_wasserstein_test(
                base, shift_b, n_projections=16, seed=_SEED + t, n_perm=n_perm
            )
            if float(res_alt["pvalue"]) < 0.05:
                power_rejections += 1

        # Barycenter of identical clouds converges exactly to that cloud.
        bary = sliced_wasserstein_barycenter([cloud, cloud.copy()], n_projections=16, seed=_SEED)
        return {
            "sw_d1_exactness_err": abs(sw_d1 - w1),
            "sw_self_distance": float(sw_self),
            "sw_null_size": null_rejections / float(n_trials),
            "sw_shift_detection_power": power_rejections / float(n_trials),
            "sw_n_trials": float(n_trials),
            "sw_n_perm": float(n_perm),
            "sw_barycenter_identical_converged": 1.0 if bary.converged else 0.0,
            "sw_barycenter_final_delta": float(bary.final_delta),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_score_decomposition() -> dict[str, float]:
    """Proper-score decomposition identities on SYNTHETIC forecasts (wave 12).

    Bröcker (2012, ensemble CRPS = potential + quality - reliability); Kolassa
    (2016, discrete/count CRPS = reliability - resolution + potential); Epstein
    (1969, ranked probability score); Brier (1950) / Murphy (1973) exact Brier
    identity; Gneiting-type Gaussian sharpness split (reusing
    ``metrics.scoring``). Every decomposition error must vanish to floating
    point, and the true pmf must beat a perturbed pmf under the strictly proper
    RPS. Seeded SYNTHETIC; proper scores only.
    """
    try:
        rng = np.random.default_rng(_SEED)
        # Bröcker ensemble CRPS decomposition (algebraic identity -> error 0).
        m, n_mem = 500, 8
        ensembles = rng.standard_normal((m, n_mem))
        y_ens = rng.standard_normal(m)
        broecker = broecker_ensemble_crps_decomposition(ensembles, y_ens)

        # Kolassa discrete CRPS decomposition on random valid pmfs.
        n_d, k_d = 400, 5
        pmfs = rng.dirichlet(np.ones(k_d), size=n_d)
        y_d = rng.integers(0, k_d, size=n_d).astype(float)
        kolassa = kolassa_crps_decomposition(pmfs, y_d)

        # Exact multi-category Brier identity (bins = distinct forecast vectors).
        brier = brier_decomposition(pmfs, y_d)

        # RPS strict propriety: the true pmf beats a perturbed (uniform) pmf.
        k_r = 4
        true_pmf = np.array([0.1, 0.2, 0.4, 0.3])
        pert_pmf = np.full(k_r, 1.0 / k_r)
        n_r = 4000
        y_r = rng.choice(k_r, size=n_r, p=true_pmf).astype(float)
        rps_true = mean_ranked_probability_score(true_pmf, y_r)
        rps_pert = mean_ranked_probability_score(pert_pmf, y_r)

        # Gaussian sharpness decomposition (crps = sharpness + error term).
        n_s = 600
        y_s = rng.standard_normal(n_s)
        mu_s = rng.standard_normal(n_s) * 0.5
        sigma_s = np.abs(rng.standard_normal(n_s)) + 0.5
        sharp = sharpness_decomposition(y_s, mu_s, sigma_s)
        return {
            "broecker_crps": float(broecker["crps"]),
            "broecker_decomp_error": float(broecker["decomp_error"]),
            "kolassa_crps": float(kolassa["crps"]),
            "kolassa_decomp_error": float(kolassa["decomp_error"]),
            "rps_at_truth": float(rps_true),
            "rps_at_perturbed": float(rps_pert),
            "rps_at_truth_optimality_gap": float(rps_pert - rps_true),
            "brier_identity_error": float(brier["decomp_error"]),
            "sharpness_plus_error_vs_crps_gap": float(sharp["decomp_error"]),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_confidence_sequences() -> dict[str, float]:
    """Anytime-valid confidence sequences on SYNTHETIC streams (wave 12).

    Howard, Ramdas, McAuliffe & Sekhon (2021), "Time-uniform, nonparametric,
    nonasymptotic confidence sequences", Ann. Statist. 49(2), arXiv:1810.08240
    (sub-Gaussian mixture / poly-stitching, empirical-Bernstein); Waudby-Smith &
    Ramdas (2024), arXiv:2010.09686 (WSR betting). Time-uniform coverage is
    measured over seeded Monte-Carlo replicates; the width ratio vs the
    fixed-sample CI and the log-log width-decay slope quantify the price of
    anytime validity. Seeded SYNTHETIC; inferential diagnostics only.
    """
    try:
        rng = np.random.default_rng(_SEED)
        alpha = 0.05
        n_reps, t_cov = 80, 1000
        # Sub-Gaussian streams (Gaussian, sd 1 = declared proxy): true mean 0.
        gauss = rng.standard_normal((n_reps, t_cov))
        cs_mix = subgaussian_cs(gauss, sigma=1.0, alpha=alpha, boundary="two_sided_mixture")
        cov_mix = check_time_uniform_coverage(cs_mix.lower, cs_mix.upper, 0.0)
        cs_ps = subgaussian_cs(gauss, sigma=1.0, alpha=alpha, boundary="poly_stitching")
        cov_ps = check_time_uniform_coverage(cs_ps.lower, cs_ps.upper, 0.0)

        # Bounded streams for the support-constrained builders: Uniform[-1, 1].
        n_b, t_b = 40, 200
        bounded = rng.uniform(-1.0, 1.0, size=(n_b, t_b))
        eb = empirical_bernstein_cs(bounded, lower=-1.0, upper=1.0, alpha=alpha)
        cov_eb = check_time_uniform_coverage(eb.lower, eb.upper, 0.0)
        wsr = wsr_cs(bounded[:8, :120], lower=-1.0, upper=1.0, alpha=alpha, n_bisect=24)
        cov_wsr = check_time_uniform_coverage(wsr.lower, wsr.upper, 0.0)

        # Width diagnostics on one long sub-Gaussian stream (T = 4000).
        long_stream = rng.standard_normal(4000)
        cs_long = subgaussian_cs(long_stream, sigma=1.0, alpha=alpha)
        ratio = width_ratio_vs_fixed(cs_long.lower, cs_long.upper, long_stream, alpha, sigma=1.0)
        slope = width_decay_slope(cs_long.lower, cs_long.upper)
        return {
            "cs_alpha": alpha,
            "cs_n_reps": float(n_reps),
            "cs_time_uniform_coverage": cast("float", cov_mix["time_uniform_coverage"]),
            "cs_violation_rate": cast("float", cov_mix["violation_rate"]),
            "cs_poly_stitching_coverage": cast("float", cov_ps["time_uniform_coverage"]),
            "cs_empirical_bernstein_coverage": cast("float", cov_eb["time_uniform_coverage"]),
            "cs_wsr_coverage": cast("float", cov_wsr["time_uniform_coverage"]),
            "cs_width_ratio_vs_fixed": float(ratio),
            "cs_width_decay_slope": float(slope),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_deep_hedging() -> dict[str, float]:
    """Deep hedging vs baselines on SYNTHETIC GBM paths (wave 12, torch-gated).

    Buehler, Gonon, Teichmann & Wood (2019), "Deep Hedging", Quantitative
    Finance 19(10), arXiv:1802.03042; Föllmer & Schied (2016, entropic risk);
    Rockafellar & Uryasev (2000/2002, ES/CVaR); Black & Scholes (1973) / Merton
    (1973, delta baseline); Davis & Norman (1990, proportional costs). A tiny
    MLP learns a friction-aware hedge of a short European call on small seeded
    GBM paths; the learned risk must beat the unhedged short-payoff risk. torch
    is the optional ``nn`` extra and is imported lazily by the module, so a
    torch-less environment raises ImportError and this bench returns ``{}``.
    SYNTHETIC correctness comparison, never market evidence.
    """
    try:
        # Local import: the deep-hedging module needs torch only at call time,
        # and the guard below lets torch-less environments skip cleanly.
        from quant_fund.models.deep_hedging import compare_hedges, simulate_gbm_paths

        s0, strike, sigma, maturity = 100.0, 100.0, 0.2, 0.5
        n_steps, n_paths, epochs = 16, 512, 30
        dt = maturity / n_steps
        train = simulate_gbm_paths(n_paths, n_steps, s0=s0, sigma=sigma, dt=dt, seed=_SEED)
        evaluation = simulate_gbm_paths(n_paths, n_steps, s0=s0, sigma=sigma, dt=dt, seed=_SEED + 1)
        comparison = compare_hedges(
            train,
            strike=strike,
            maturity=maturity,
            sigma=sigma,
            cost_rate=1e-3,
            risk="expected_shortfall",
            alpha=0.9,
            hidden=(8, 8),
            epochs=epochs,
            lr=8e-3,
            seed=_SEED,
            eval_paths=evaluation,
        )
        metrics = comparison.metrics
        hedged = float(metrics["dh_hedged_risk"])
        unhedged = float(metrics["dh_unhedged_risk"])
        return {
            "dh_hedged_risk": hedged,
            "dh_unhedged_risk": unhedged,
            "dh_friction_gap_vs_bs_delta": float(metrics["dh_friction_gap_vs_bs_delta"]),
            "dh_risk_reduction": unhedged - hedged,
        }
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError):
        return {}
