"""Benchmark family registry and forbidden research-metric keys.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import math

BENCHMARK_CATALOG_VERSION = 2
# Schema 2 stamps ``backtest_overfitting`` (PBO, DSR, PSR, MinTRL, trial counts).
# Schema 1 receipts remain valid: they predate the block and are not rewritten.
RESEARCH_RECEIPT_SCHEMA_VERSION = 2
RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED = frozenset({1, 2})
# Kept for the pre-split catalog API. These are diagnostic key groups.
REQUIRED_CHRISTOFFERSEN_CC_KEYS = frozenset({"christoffersen_cc_p", "christoffersen_cc_lr"})
PREFERRED_CHRISTOFFERSEN_IND_KEYS = frozenset({"christoffersen_ind_p", "christoffersen_ind_lr"})
REQUIRED_BENCHMARK_FAMILIES = frozenset(
    {
        "ranking",
        "alpha",
        "volatility",
        "distribution",
        "regime",
        "tail",
        "drawdown",
        "liquidity",
        "reinforcement",
        "conformal",
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
        "cv_plus",
        "cpcv",
        "localized_conformal",
        "conformal_rank",
        "online_crc",
        "portfolio_conformal",
        "northset",
    }
)
# Optional families may appear in a receipt (and are then fully soft-verified)
# but are never required by the benchmark catalog. ``candle_order_book`` is the
# optional candlestick+L2 family emitted by the ``candle-book`` CLI; without
# this allow-list the verify.py candle honesty dispatch could never run.
OPTIONAL_BENCHMARK_FAMILIES = frozenset(
    {
        "candle_order_book",
        "robinhood_plus",
        # Descriptive proper-score diagnostics of the SYNTHETIC return panel
        # (see research/benches_extra.py); never live-P&L or headline ratios.
        "complexity",
        "roughness",
        "serial_randomness",
        # SOTA canon waves 8-10 batteries (see research/benches_w810.py):
        # anytime-valid inference, multivariate proper scores, TS conformal,
        # regime-conditional evaluation, the SYNTHETIC leaky-oracle red
        # team, and distributional-ML baselines. All seeded SYNTHETIC
        # streams; correctness diagnostics only, never promotion gates.
        "anytime_valid",
        "energy_score",
        "ts_conformal",
        "regime_eval",
        "leakage_redteam",
        "distributional_ml",
        # SOTA canon wave 11 batteries (see research/benches_w11.py):
        # rough-path signatures, optimal transport, Wasserstein DRO, and
        # conformal PID control. Seeded SYNTHETIC streams; correctness
        # diagnostics only, never promotion gates.
        "rough_paths",
        "optimal_transport",
        "wasserstein_dro",
        "conformal_pid",
        # SOTA canon wave 12 batteries (see research/benches_w12.py):
        # NexCP nonexchangeable conformal bounds, multi-horizon EnbPI,
        # Bayesian stacking / pseudo-BMA(+), regime-weighted conformal VaR,
        # sliced-Wasserstein two-sample testing, proper-score decompositions,
        # anytime-valid confidence sequences, and (torch-gated) deep hedging.
        # Seeded SYNTHETIC streams; correctness diagnostics only, never
        # promotion gates.
        "nexcp",
        "mh_enbpi",
        "stacking",
        "rwcv",
        "sliced_wasserstein",
        "score_decomposition",
        "confidence_sequences",
        "deep_hedging",
        # SOTA canon wave 13 batteries (see research/benches_w13.py):
        # conformal coverage inference under temporal dependence, minimax-
        # optimal conformal e-detectors, rolling conformal prediction for
        # sequential training, anytime-valid rank confidence sequences /
        # BB-EDGE leaderboards, PICPI self-consistency intervals, replicable
        # conformal calibration, reference-null calibrated e-thresholds, and
        # delayed-feedback ACI. Seeded SYNTHETIC streams; correctness
        # diagnostics only, never promotion gates.
        "coverage_inference",
        "conformal_e_detectors",
        "rolling_conformal",
        "rank_cs",
        "picpi",
        "replicable_conformal",
        "reference_null",
        "delayed_aci",
        # SOTA canon wave 14 batteries (see research/benches_w14.py): deep
        # finance math — martingale optimal transport model-free bounds,
        # large-deviations importance sampling, the trade-crowding mean-field
        # game, vine + GAS copulas, Malliavin-calculus Monte Carlo Greeks,
        # XVA (CVA/FVA/MVA + WWR), American LSM with dual bounds, Dupire local
        # vol + SLV mixing MC, the deep BSDE solver, the DeRegiME regime-mixture
        # head, two-stage odd residual flows, deep kernel hedging, the
        # zero-intelligence LOB simulator, and cash-constrained multi-asset
        # optimal execution. (models/fourier_pricing.py is deliberately NOT
        # wired — its COS European path has a documented open bug.) Seeded
        # SYNTHETIC streams; correctness diagnostics only, never promotion
        # gates.
        "martingale_ot",
        "large_deviations",
        "mean_field_games",
        "vine_copula",
        "malliavin_greeks",
        "xva",
        "american_lsm",
        "local_stoch_vol",
        "deep_bsde",
        "deep_regime_mixture",
        "odd_residual_flows",
        "deep_kernel_hedging",
        "zi_lob",
        "cash_constrained_oe",
        # SOTA canon wave 15 batteries (see research/benches_w15.py): agentic
        # research integrity + repaired Fourier pricing — dynamic-subspace
        # denoising under oblique structured noise, Transported Conformal
        # Calibration (TCC), HPD split conformal (C-USIM) for multimodal
        # predictive laws, EverMine capability-value Cap-swap accounting, the
        # anytime-valid frozen referee for LLM factor mining, revision-aware
        # vintage evaluation, the entropy-Shapley attribution hierarchy, and
        # the Fourier pricing suite (COS / COS-Bermudan / Hilbert barrier) now
        # that its European leg is REPAIRED and matches BS to ~1e-13 (wave 14
        # had deliberately skipped it; that comment block stays as history).
        # Seeded SYNTHETIC streams; correctness diagnostics only, never
        # promotion gates.
        "subspace_denoising",
        "conformal_transfer",
        "hpd_conformal",
        "capability_value",
        "agent_referee",
        "vintage_eval",
        "entropy_shapley",
        "fourier_pricing",
        # SOTA canon wave 16 batteries (see research/benches_w16.py):
        # loss-choice-vs-model-choice decomposition for volatility forecasts
        # (validation level alignment and the loss-dominated -> model-dominated
        # QLIKE-ratio flip), generalized hierarchical conformal prediction
        # (GHCP), multi-source randomly localized conformal prediction
        # (MS-RLCP), conformal prediction under an exponential-tilt joint shift
        # (ExTRA-WCP / -WCP-T), target-alignment dilution accounting with
        # cautious forecast selection, and (torch-gated) the C51 distributional
        # RL market maker on the zero-intelligence LOB. Seeded SYNTHETIC
        # streams; correctness diagnostics only, never promotion gates.
        "vol_loss_decomposition",
        "hierarchical_conformal",
        "multisource_conformal",
        "extra_tilt",
        "forecast_selection",
        "rl_market_maker",
        # SOTA canon wave 17 batteries (see research/benches_w17.py):
        # conformal risk-averse decision making with optimized-certainty-
        # equivalent (OCE) risk control (high-probability CVaR certificates,
        # the Hoeffding-margin ablation, the sqrt(n) radius law), e-PS
        # sample-efficient multiple testing with adaptive data collection
        # (simple-vs-simple specialization), (torch-gated) greek-neutral
        # option portfolios — hedging as a training inductive bias (delta-
        # exposure monotonicity + interior optimum), and (torch-gated) the
        # DiffPTS full-ELBO diffusion forecaster vs the NGBoost Gaussian
        # baseline (CRPS gain, coverage, PIT). Seeded SYNTHETIC streams;
        # correctness diagnostics only, never promotion gates.
        "conformal_oce",
        "adaptive_eps",
        "greek_neutral",
        "diffusion_forecaster",
        # SOTA canon wave 17 (execution theory / distributional RL / DiffPTS
        # ELBO / multi-step UQ) + wave 18 (path-space flows, neural SDEs,
        # StocBench sampler budgets, agentic LOB, FASE eval, K-line paths).
        # Seeded SYNTHETIC streams; correctness diagnostics only, never
        # promotion gates.
        "diffpts",
        "extra_conformal",
        "multilevel_mm",
        "rlmm_c51",
        "sga_uq",
        "passive_impact",
        "stochastic_tracking",
        "gslice",
        "neural_sde",
        "stocbench",
        "agentic_lob",
        "fase_eval",
        "kit_paths",
        # SOTA canon waves 19+ (numpy/scipy lanes): Langevin impact, event-time
        # flow, Fukasawa IV, IVS diffusion, RCCP, DCP, and the wave 20-29
        # families below. Seeded SYNTHETIC streams; correctness diagnostics
        # only, never promotion gates.
        "langevin_impact",
        "event_time_flow",
        "fukasawa_iv",
        "ivs_diffusion",
        "rccp",
        "dcp",
        "varswap_stopping",
        "hidden_markov_equilibrium",
        "gaussian_normalized_coords",
        "liquidity_tail_lob",
        "arl_mm",
        "bocpd_changepoint",
        "rough_heston_rbergomi",
        "signature_features",
        "signature_martingale_test",
        "svi_surface",
        "propagator_impact",
        "queue_reactive",
        "koopman_edmd",
        "sig_gan",
        "neural_tpp",
        "pmcmc_sv",
        "multifractal_vol",
        "spci_conformal",
        "hawkes_em",
        "fernholz_spt",
        "breeden_litzenberger",
        "tda_persistence",
        "fractional_ou",
        "fourier_hermite",
        "kernel_changepoint",
        "kinetic_ising",
        "heterogeneous_abm",
        "marchenko_pastur",
        "factor_nowcast",
        "stationary_bootstrap",
        "skill_ratings",
        "modularity_communities",
        "stein_thinning",
        "durbin_koopman",
        "dp_mixture",
        "expert_aggregation",
        "instrumental_quantile",
        "implied_tree",
        "ensemble_kalman_inversion",
        "enkf",
        "causal_discovery",
        "knockoffs",
        "callaway_did",
        "surrogate_nonlinear",
        "sindy",
        "hmc",
        "proxy_svar",
        "sbi",
        "rqa",
        "hj_distance",
        "tensor_decomp",
        # SOTA canon wave 30 batteries (see research/benches_w30.py):
        # local-projection IV impulse responses with weak-IV
        # first-stage F and AR grid-inversion bands, bispectral
        # Hinich gaussianity test + quadratic phase coupling,
        # functional linear regression via B-spline/FPCA,
        # score-driven (GAS/GAS-t/GAS-Poisson) filters, NB-2/
        # ZIP/hurdle count regression + Vuong non-nested test,
        # largest-Lyapunov/FNN/Cao nonlinear-dynamics estimators.
        # Same SYNTHETIC diagnostic contract.
        "lp_iv",
        "bispectrum",
        "functional_linear",
        "gas_score",
        "count_data",
        "lyapunov",
        # SOTA canon wave 31 batteries (see research/benches_w31.py):
        # kernel nonparametric IV (regularized Fredholm inverse +
        # Landweber-Fridman + polynomial conditional-moment test),
        # continuous-time Markov multistate models (intensity MLE,
        # transition matrix expm, illness-death Aalen-Johansen),
        # partially-linear semiparametrics (Robinson/Speckman/series
        # + CV bandwidth), Heckman two-step + ML selection correction,
        # sharp/fuzzy regression discontinuity (IK/CCT bandwidth,
        # McCrary density test, donut), Manski/Lee partial-
        # identification bounds. Same SYNTHETIC diagnostic contract.
        "kernel_iv",
        "multistate",
        "partial_linear",
        "heckman",
        "rd",
        "bounds",
        # SOTA canon wave 32 batteries (see research/benches_w32.py):
        # peaks-over-threshold extreme value (Hill tail index, GPD
        # MLE, POT VaR/ES, return levels), cross-fitted double/debiased
        # ML (PLR + AIPW IRM on ML nuisance residuals), bunching
        # estimation at kinks (excess mass vs polynomial counterfactual,
        # Poisson-WLS + bootstrap se), honest causal forests
        # (propensity-transformed leaves, honest split/est halves),
        # Gaussian-process regression (ARD kernel, Type-II ML,
        # closed-form LOO), Hamilton/Kim Markov-switching EM
        # (filter/smoother/regime forecasts). Same SYNTHETIC contract.
        "extreme_value",
        "double_ml",
        "bunching",
        "causal_forest",
        "gaussian_process",
        "markov_switching",
        # SOTA canon wave 33 batteries (see research/benches_w33.py):
        # BSTS causal impact (local-level + spike-slab regression,
        # simulation counterfactual, posterior tail prob), weak-IV
        # robust inference (Anderson-Rubin, Moreira CLR, first-stage
        # F + partial R2, AR-inverted confidence sets), synthetic
        # DiD (unit + time simplex weights, placebo inference),
        # Fisher/Pitman randomization inference (sharp-null
        # permutation, Westfall-Young max-T, Pitman CI inversion),
        # propensity-score pipeline (IRLS PS, caliper matching, IPW
        # + overlap weights, SMD balance), cluster-robust inference
        # (CR1/CR2 sandwich, CGM/Webb wild cluster bootstrap).
        # Same SYNTHETIC diagnostic contract.
        "causal_impact",
        "weak_iv",
        "synth_did",
        "permutation_inference",
        "propensity_score",
        "cluster_robust",
        # SOTA canon wave 34 batteries (see research/benches_w34.py):
        # matrix-completion causal panels (Nuclear-norm soft-impute,
        # MCPanel ATT), generalized synthetic control (Xu IFE factor
        # EM + treated-loading counterfactual), RIF regressions
        # (Firpo-Fortin-Lemieux unconditional quantile/variance/Gini
        # effects), shift-share IV (Bartik first-stage F, Rotemberg
        # weights, AKM-style SEs), entropy balancing (Hainmueller
        # calibration weights, exact moment match, ESS), DiD
        # diagnostics (Goodman-Bacon decomposition of TWFE into
        # clean vs forbidden 2x2s, Sun-Abraham cohort CATTs).
        # Same SYNTHETIC diagnostic contract.
        "matrix_completion",
        "gsynth",
        "rif_regression",
        "shift_share",
        "entropy_balancing",
        "did_diagnostics",
        # SOTA canon wave 35 batteries (see research/benches_w35.py):
        # honest DiD (Rambachan-Roth Δ^SD sensitivity + FLCI breakdown
        # M̄*), many-weak-IV estimators (LIML via AR-ratio minimization,
        # JIVE, HFUL leverage-robust), Fama-MacBeth two-pass risk
        # prices (FM SEs + Shanken EIV inflation), specification-curve
        # multiverse (full spec grid + shuffle p-value), sign-restricted
        # VAR (Uhlig QR draws, median IRF under sign restrictions),
        # panel quantile FE (Machado-Santos Silva moments: location +
        # absolute-residual scale + Φ^{-1}(τ) shift). Same SYNTHETIC
        # diagnostic contract.
        "honest_did",
        "many_iv",
        "fama_macbeth",
        "specification_curve",
        "sign_restricted_var",
        "panel_quantile_fe",
        # SOTA canon wave 36 batteries (see research/benches_w36.py):
        # Arellano-Bond FD-GMM dynamic panels (collapsed lags, AR(2)
        # and Sargan diagnostics), Minnesota-prior BVAR (Theil dummy
        # observations, per-equation shrinkage profile), quasi-Bayesian
        # mediation analysis (ACME/ADE via coefficient draws),
        # competing-risks Aalen-Johansen CIFs with Klein-Andersen
        # pseudo-value regressions, stochastic-frontier composed-error
        # MLE (normal/half-normal + Jondrow efficiency), regression-
        # kink design (slope-discontinuity ratio, delta-method SE).
        # Same SYNTHETIC diagnostic contract.
        "arellano_bond",
        "bvar_minnesota",
        "mediation_analysis",
        "competing_risks",
        "stochastic_frontier",
        "regression_kink",
        # SOTA canon wave 37 batteries (see research/benches_w37.py):
        # spatial autoregression (concentrated-likelihood SAR with
        # log-determinant Jacobian + Moran's I residual test),
        # ordered probit/logit (latent-threshold MLE with ordered
        # cutpoints), triple difference (saturated DDD + eight-cell
        # contrast under confounded post shocks), distribution
        # regression (per-threshold logit CDF path, Chernozhukov-
        # Fernández-Val-Melly), SIMEX measurement-error correction
        # (noise-dose quadratic extrapolation + jackknife SE),
        # LP-DiD clean event studies (per-horizon clean weighting
        # of Dube-Girardi-Jordà-Taylor). Same SYNTHETIC diagnostic
        # contract.
        "spatial_econometrics",
        "ordered_choice",
        "triple_difference",
        "distribution_regression",
        "simex",
        "lp_did",
    }
)
BENCHMARK_FAMILY_ORDER = (
    "ranking",
    "alpha",
    "volatility",
    "distribution",
    "regime",
    "tail",
    "drawdown",
    "liquidity",
    "reinforcement",
    "conformal",
    "evalues",
    "jackknife_plus",
    "crc",
    "weighted_conformal",
    "interval_risk",
    "quantile_bandit",
    "cv_plus",
    "cpcv",
    "localized_conformal",
    "conformal_rank",
    "online_crc",
    "portfolio_conformal",
    "northset",
)
# Headline P&L / ratio key *tokens* that must never appear in research family /
# scorecard blobs (not the paper analytics_export schema — that catalog allows
# equity/stress pnl/nav diagnostics under live_pnl_claim=false).
# Matching is by underscore-token on mapping keys (case-insensitive), so
# ``flag_high_sharpe`` / ``shock_down_pnl`` / ``calmar_diagnostic`` all fail closed.
FORBIDDEN_RESEARCH_METRIC_KEYS = frozenset(
    {
        "sharpe",
        "sortino",
        "calmar",
        "pnl",
        "nav",
    }
)


def _iter_mapping_keys(obj: object) -> list[str]:
    """Collect nested mapping keys (dicts only; list elements walked)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_iter_mapping_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_iter_mapping_keys(item))
    return keys


def family_blob_forbidden_metrics_absent(payload: object) -> bool:
    """Return True iff *payload* has no forbidden research-headline metric keys.

    Fail closed: any mapping key whose underscore tokens include sharpe / sortino /
    calmar / pnl / nav marks the blob unclean. Values are not scanned (keys only).

    Scope: research family / scorecard blobs only. Paper ``analytics_export`` may
    contain equity ``nav_*`` / stress ``*_pnl`` diagnostics; validate those with
    ``validate_analytics_export`` (live_pnl_claim fail-closed), not this helper.

    ``live_pnl_claim`` itself is exempt at any depth: it is the honesty flag,
    not a metric — receipts that embed other receipts carry it nested (e.g. a
    tournament manifest quoting its benchmark manifest).
    """
    for key in _iter_mapping_keys(payload):
        if key == "live_pnl_claim":
            continue
        parts = str(key).lower().replace("-", "_").split("_")
        if any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in parts if tok):
            return False
    return True


def family_blob_executed(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``executed`` twin)."""
    return bool(payload)


def family_blob_nonempty(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``nonempty`` twin)."""
    return bool(payload)


def family_blob_has_finite_observation(payload: object) -> bool:
    """True iff *payload* recursively contains at least one finite observation.

    Lifted from research.agent ``_benchmark_scorecard`` nested ``has_observation``
    (Day Wave 44). Semantics:
    - dict / list / tuple → any child is an observation
    - bool → True
    - int (incl. numpy integer scalars via ``.item()``) → True
    - float (incl. numpy floating) → ``math.isfinite``
    - str → nonempty strip and not ``nan`` / ``none`` (case-insensitive)
    - else → False

    Empty ``{}`` / all-NaN blobs → False. Used by soft scorecard forge verify.
    """
    if isinstance(payload, dict):
        return any(family_blob_has_finite_observation(item) for item in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(family_blob_has_finite_observation(item) for item in payload)
    if isinstance(payload, bool):
        return True
    if isinstance(payload, int):
        return True
    if isinstance(payload, float):
        return math.isfinite(payload)
    # Numpy scalars are not always subclasses of int/float — recurse via .item().
    if (
        hasattr(payload, "dtype")
        and hasattr(payload, "item")
        and not isinstance(payload, (bytes, bytearray, memoryview))
    ):
        try:
            return family_blob_has_finite_observation(payload.item())
        except (ValueError, TypeError, AttributeError):
            return False
    return (
        isinstance(payload, str)
        and bool(payload.strip())
        and payload.lower() not in {"nan", "none"}
    )


__all__ = [
    "BENCHMARK_CATALOG_VERSION",
    "BENCHMARK_FAMILY_ORDER",
    "FORBIDDEN_RESEARCH_METRIC_KEYS",
    "OPTIONAL_BENCHMARK_FAMILIES",
    "REQUIRED_BENCHMARK_FAMILIES",
    "PREFERRED_CHRISTOFFERSEN_IND_KEYS",
    "REQUIRED_CHRISTOFFERSEN_CC_KEYS",
    "RESEARCH_RECEIPT_SCHEMA_VERSION",
    "RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED",
    "family_blob_executed",
    "family_blob_forbidden_metrics_absent",
    "family_blob_has_finite_observation",
    "family_blob_nonempty",
]
