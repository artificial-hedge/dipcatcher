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
        # SOTA canon wave 38 batteries (see research/benches_w38.py):
        # control-function endogeneity correction (two-stage CF +
        # Durbin-Wu-Hausman), kernel regression (Nadaraya-Watson +
        # local-linear derivatives, LOO-CV bandwidth), censored
        # quantile regression (Powell CLAD via Chernozhukov-Hong
        # three-step), SETAR threshold autoregression (CLS threshold
        # + sup-F linearity), Papke-Wooldridge fractional response
        # (quasi-MLE logit for [0,1] outcomes), Turnbull interval-
        # censored NPMLE (self-consistency EM). Same SYNTHETIC
        # diagnostic contract.
        "control_function",
        "kernel_regression",
        "censored_quantile",
        "threshold_ar",
        "fractional_response",
        "interval_censoring",
        # SOTA canon wave 39 batteries (see research/benches_w39.py):
        # Manski maximum score (smoothed-score distribution-free
        # binary response), Chen sieve partial-linear estimation
        # (B-spline basis), McFadden nested logit (two-level FIML,
        # inclusive-value λ), Weibull AFT (SEV MLE under right
        # censoring), distance covariance dependence test
        # (Székely-Rizzo-Bakirov + permutation), panel unit-root
        # tests (IPS + LLC, simulated moments). Same SYNTHETIC
        # diagnostic contract.
        "maximum_score",
        "sieve_estimation",
        "nested_logit",
        "aft_model",
        "distance_covariance",
        "panel_unitroot",
        # SOTA canon wave 40 batteries (see research/benches_w40.py):
        # McFadden-Train mixed logit (random-coefficients
        # simulated MLE, quasi-random draws), Cragg two-part
        # hurdle (participation logit + truncated-normal amount),
        # Zellner SUR (feasible-GLS Kronecker system), Diebold-
        # Yilmaz connectedness (VAR generalized FEVD), Newey-
        # Powell nonparametric series IV (basis projection +
        # DWH endogeneity check), Politis-Romano-Wolf
        # subsampling (block recentered CI, minimal-assumption
        # coverage). Same SYNTHETIC diagnostic contract.
        "mixed_logit",
        "hurdle",
        "sur_model",
        "connectedness",
        "nonparametric_iv",
        "subsampling",
        # SOTA canon wave 41 batteries (see research/benches_w41.py):
        # shared gamma frailty (Vaupel/Clayton clustered survival,
        # marginal likelihood), interrupted/comparative time
        # series (segmented regression, Newey-West SEs), Hayashi-
        # Yoshida lead-lag covariance (non-synchronous ticks,
        # shift-scan direction), PPML gravity (Santos Silva-
        # Tenreyro multiplicative mean under heteroskedasticity),
        # MacKinlay event study (market-model CAR, Patell z +
        # BMP t), Mallows model averaging (Hansen Cp-simplex
        # weights). Same SYNTHETIC diagnostic contract.
        "frailty",
        "interrupted_ts",
        "lead_lag",
        "ppml",
        "event_study",
        "model_averaging",
        # SOTA canon wave 42 batteries (see research/benches_w42.py):
        # Lo-MacKinlay variance-ratio test (VR(q) with
        # heteroskedastic-robust z*), Corsi HAR realized-vol
        # cascade (daily/weekly/monthly aggregates), Clark-West
        # MSPE-adjusted nested-forecast test, Stambaugh
        # predictive-regression bias + Campbell-Yogo
        # Bonferroni-Q CI, Roy two-sector self-selection
        # (probit + Mills-corrected wage equations), Cameron-
        # Gelbach-Miller two-way clustered SEs (V1+V2−V12).
        # Same SYNTHETIC diagnostic contract.
        "variance_ratio",
        "har_rv",
        "clark_west",
        "stambaugh",
        "roy_model",
        "two_way_cluster",
        # SOTA canon wave 43 batteries (see research/benches_w43.py):
        # Bai-Perron multiple structural breaks (sequential F-tests
        # + BIC), Bernanke-Boivin-Eliasz FAVAR (PCA factors +
        # VAR), Kao/Pedroni panel cointegration tests, Vuong
        # non-nested model selection (omega^2 distinguishability
        # + LR), Merton structural credit distance-to-default
        # (KMV fixed-point inversion), White Reality Check +
        # Hansen SPA data-snooping control (stationary-bootstrap
        # max-statistics). Same SYNTHETIC diagnostic contract.
        "bai_perron",
        "favar",
        "panel_coint",
        "vuong_test",
        "merton_model",
        "white_reality",
        # SOTA canon wave 44 batteries (see research/benches_w44.py):
        # Johansen ML cointegration rank (trace/lmax vs
        # Osterwald-Lenum) + reduced-rank VECM, Easley-O'Hara
        # PIN (stabilized EHO mixture MLE), Kyle (1985) lambda
        # + single-auction equilibrium, Oster (2019) selection-
        # on-observables delta*/beta* bounds, Storey-Tibshirani
        # q-values + pi0 smoother, Kiefer-Vogelsang fixed-b HAR
        # inference (simulated Brownian-bridge limit).
        # Same SYNTHETIC diagnostic contract.
        "johansen_vecm",
        "pin_model",
        "kyle_lambda",
        "oster_bounds",
        "storey_fdr",
        "kiefer_vogelsang",
        # wave 45 — Conley spatial HAC, Driscoll-Kraay panel SEs,
        # Pesaran CCE common-factors, Wald SPRT sequential test,
        # Lee bounds on selection, Barrett-Donald dominance KS.
        # Same SYNTHETIC diagnostic contract.
        "conley_se",
        "driscoll_kraay",
        "pesaran_cce",
        "wald_sprt",
        "lee_bounds",
        "barrett_donald",
        # wave 46 — BLP random-coefficients demand, Olley-Pakes
        # production proxy, Rust dynamic discrete choice,
        # Oaxaca-Blinder wage decomposition, binscatter CEF +
        # spec test, DFL reweighting decomposition. Same
        # SYNTHETIC diagnostic contract.
        "blp_demand",
        "olley_pakes",
        "rust_ddc",
        "oaxaca_blinder",
        "binscatter",
        "dfl_decomp",
        # wave 47 — Rosenbaum sensitivity bounds for matched pairs,
        # AIPW doubly-robust ATE, CAVI mean-field Gaussian mixture,
        # Pesaran CD cross-section dependence, Hausman FE-RE + DWH
        # endogeneity batteries, CUSUM structural-break monitoring
        # (Chu-Stinchcombe-White boundary). Same SYNTHETIC
        # diagnostic contract.
        "rosenbaum_sensitivity",
        "aipw_ate",
        "cavi_gmm",
        "pesaran_cd",
        "hausman_tests",
        "cusum_monitor",
        # wave 48 — targeted maximum likelihood (TMLE) ATE, Lewbel
        # heteroskedasticity-generated instruments, proximal/
        # negative-control confounding bridge, Cover universal
        # portfolio (best-CRP tracking), VPIN volume-clock flow
        # toxicity, marginal treatment effects (local-IV MTE
        # curve). Same SYNTHETIC diagnostic contract.
        "tmle",
        "lewbel_iv",
        "proximal_causal",
        "cover_up",
        "vpin",
        "marginal_treatment",
        # wave 49 — Eisenberg-Noe clearing-vector default contagion,
        # Cont-Wagalath fire-sale deleveraging cascades, Adrian-
        # Brunnermeier delta-CoVaR systemic contribution, Blanchard-Quah
        # long-run-restriction SVAR identification, Kalman/RTS TVP
        # regression, DerSimonian-Laird random-effects meta-analysis
        # with Egger funnel asymmetry. Same SYNTHETIC diagnostic
        # contract.
        "eisenberg_noe",
        "fire_sales",
        "delta_covar",
        "blanchard_quah",
        "tvp_var",
        "meta_analysis",
        # wave 50 — Engle-Russell ACD(1,1) duration clustering
        # (QMLE), Heath-Jarrow-Morton Gaussian forward-curve
        # simulation, Vasicek affine term structure + Campbell-
        # Shiller expectations-hypothesis regression, Gil-Pelaez
        # characteristic-function inversion (CDF/quantiles),
        # hedonic time-dummy + Bailey-Muth-Nourse repeat-sales
        # indices, DEA CCR/BCC efficiency frontiers. Same
        # SYNTHETIC diagnostic contract.
        "acd_duration",
        "hjm",
        "affine_term",
        "gil_pelaez",
        "hedonic",
        "dea",
        # wave 51 — Bakshi-Kapadia-Madan model-free option-implied
        # variance/skew/kurtosis spanning integrals, Phillips-Shi-Yu
        # GSADF recursive bubble detection + date stamping, Pesaran-
        # Shin-Smith pooled-mean-group panel ARDL, Ross recovery
        # theorem state-price to physical transitions, Ait-Sahalia
        # closed-form CKLS likelihood expansion, Toda-Yamamoto
        # augmented-lag Granger MWALD. Same SYNTHETIC diagnostic
        # contract.
        "bkm_moments",
        "gsadf_bubble",
        "pmg_ardl",
        "ross_recovery",
        "ait_sahalia",
        "toda_yamamoto",
        # wave 55 — Diebold-Mariano + Harvey-Leybourne-Newbold
        # predictive-accuracy test, Engle-Granger/Phillips-Ouliaris
        # residual cointegration + ECM adjustment speed, Glosten-
        # Milgrom sequential-trade learning with martingale-price
        # diagnostics, Hasbrouck information share bounds +
        # Gonzalo-Granger permanent weights, BDS correlation-
        # integral independence test, Cochrane-Piazzesi tent-shaped
        # return-forecasting bond factor, Engle-Ng sign/size-bias
        # asymmetry diagnostics. Same SYNTHETIC diagnostic contract.
        "diebold_mariano",
        "engle_granger",
        "glosten_milgrom",
        "hasbrouck_is",
        "bds",
        "cochrane_piazzesi",
        "engle_ng",
        # wave 52 — Shin-Yu-Greenwood-Nimmo nonlinear ARDL
        # (asymmetric long/short-run multipliers, bounds-F),
        # Adrian-Boyarchenko-Giannone growth-at-risk (Koenker-
        # Bassett LP quantiles + isotonic crossing fix), Melick-
        # Thomas mixture implied-PDF recovery (martingale-pinned
        # least squares), Bandi-Russell noise/volatility
        # separation (optimal sparse sampling), Hong-Li PIT
        # density-forecast M-statistic, Beveridge-Nelson
        # permanent/transitory decomposition. Same SYNTHETIC
        # diagnostic contract.
        "nardl",
        "growth_at_risk",
        "melick_thomas",
        "bandi_russell",
        "hong_li",
        "beveridge_nelson",
        # wave 53 — Corradi-Swanson out-of-sample predictive-
        # accuracy test (moving-block bootstrap), Engle-Kroner
        # variance-targeted diagonal BEKK(1,1) multivariate
        # GARCH, Andersen quadratic-exponential Heston
        # discretization (positive under Feller violation),
        # Hansen-Lunde-Nason model confidence set (block-boot
        # T_max step-down), Christoffersen-Pelletier Weibull-
        # duration VaR clustering backtest, Shephard-Sheppard
        # HEAVY(P) two-equation realized-measure volatility.
        # Same SYNTHETIC diagnostic contract.
        "corradi_swanson",
        "engle_kroner_bekk",
        "heston_qe",
        "model_confidence_set",
        "christoffersen_pelletier",
        "sheppard_heavy",
        # wave 54 — Pesaran-Timmermann directional-accuracy sign test,
        # Giacomini-Rossi fluctuation predictive-ability break
        # detection, Muller-Watson low-frequency correlation and
        # predictive tests over cosine transforms, Romano-Wolf
        # stepdown familywise multiple-testing, Christensen-
        # Diebold-Rudebusch arbitrage-free Nelson-Siegel yield curve
        # with the verified adjustment-to-yield formula, Danielsson-
        # de Vries tail-simulation extreme VaR. Same SYNTHETIC
        # diagnostic contract.
        "pesaran_timmermann",
        "giacomini_rossi",
        "muller_watson",
        "romano_wolf",
        "christensen_diebold_rudebusch",
        "danielsson_devries",
        # wave 56 — Kwiatkowski-Phillips-Schmidt-Shin level/trend
        # stationarity LM test, Elliott-Rothenberg-Stock DF-GLS
        # quasi-differenced unit-root test with Ng-Perron MAIC lag
        # selection, Ng-Perron modified-ADF battery (MZ_a/MZ_t/MSB/
        # MPT with AR spectral-density LRV), Phillips-Perron Z
        # nonparametric unit-root corrections, Zivot-Andrews
        # endogenous level+trend break minimum-t test, Lee-
        # Strazicich LM unit-root with endogenous crash break.
        # Same SYNTHETIC diagnostic contract.
        "kpss",
        "ers_dfgls",
        "ng_perron",
        "phillips_perron",
        "zivot_andrews",
        "lee_strazicich",
        # wave 57 — MODWT maximal-overlap discrete wavelet
        # multiresolution with adjoint synthesis and boundary-
        # trimmed scale variance/correlation, Geweke spectral
        # frequency-domain Granger-causality measure with VAR
        # transfer-function decomposition, Kostakis-Magdalinos-
        # Stamatogiannis IVX-Wald persistence-robust predictive
        # inference, Bai-Ng panel information criteria + Ahn-
        # Horenstein eigenvalue-ratio factor-rank selection,
        # Wooldridge cluster-robust serial-correlation test on
        # within-transformed residuals, Belloni-Chernozhukov-
        # Hansen post-double-selection lasso inference. Same
        # SYNTHETIC diagnostic contract.
        "wavelet_modwt",
        "geweke_spectral",
        "ivx",
        "bai_ng_ic",
        "wooldridge_serial",
        "lasso_pds",
        # wave 58 — Davis-Mikosch extremogram + Ferro-Segers
        # extremal index tail-dependence battery, Jaeger echo-
        # state reservoir (spectral-radius normalized, ridge
        # readout, NARMA driver), Bates stochastic-volatility-
        # plus-jump pricing (little-trap Riccati + Merton
        # compensator, Gauss-Legendre probabilities), Cleveland
        # STL robust LOESS seasonal decomposition, Killick-
        # Fearnhead-Eckley PELT pruned optimal partitioning +
        # Fryzlewicz wild binary segmentation, Brillinger
        # dynamic principal components (Daniell-smoothed
        # cross-spectral eigendecomposition). Same SYNTHETIC
        # diagnostic contract.
        "extremogram",
        "echo_state",
        "bates_svj",
        "stl_loess",
        "pelt_wbs",
        "spectral_pca",
        # wave 59 — Huang empirical-mode decomposition +
        # Hilbert marginal spectrum (cubic-envelope sifting,
        # instantaneous-frequency extraction), Gallant-Nychka
        # semi-nonparametric density (squared Hermite-polynomial
        # expansion over a Gaussian kernel, penalized ML),
        # Acharya-Engle-Richardson SRISK systemic capital
        # shortfall (worst-alpha MES + long-run compounding),
        # Torrence-Compo Morlet cross-wavelet coherence (scale-
        # and time-smoothed squared coherency), Balke-Fomby /
        # Enders-Granger threshold cointegration (TAR/MTAR ECM,
        # SSR-grid threshold search, asymmetry F-test),
        # Chernozhukov extremal quantile regression (inter-
        # mediate-order QR + Hill tail index + Weissman
        # extrapolation). Same SYNTHETIC diagnostic contract.
        "emd_hht",
        "gallant_snp",
        "srisk",
        "wavelet_coherence",
        "tar_coint",
        "extreme_qr",
        # wave 60 — Shumway-Stoffer EM state-space estimation
        # (RTS smoother + closed-form M-step, lag-1 covariance
        # recursion), Geweke-Porter-Hudak residual memory test
        # + Marinucci-Robinson fractional cointegration, Teras-
        # vira LSTAR/ESTAR grid-NLS with Luukkonen-Saikkonen-
        # Terasvirta LM3 linearity test, Engle-Lilien-Robins
        # GARCH-in-mean joint QMLE (risk premium in the mean),
        # Bauwens-Giot logarithmic ACD (Weibull/lognormal
        # innovations, unconstrained positivity), Wigner-Ville
        # and pseudo-WVD analytic-signal time-frequency
        # distribution. Same SYNTHETIC diagnostic contract.
        "kalman_em",
        "fractional_coint",
        "star_model",
        "garch_in_mean",
        "log_acd",
        "wigner_ville",
        # wave 61 — Dümbgen-Rufibach log-concave density MLE
        # (shape-constrained, bandwidth-free nonparametric
        # estimation), Sakoe-Chiba banded DTW with warp
        # registration (phase/amplitude decomposition),
        # Chow-Lin GLS + Denton proportional temporal
        # disaggregation with exact additivity, Chen-Liu-Tiao
        # joint-iterative AO/IO/LS/TC outlier battery
        # (Bonferroni threshold, MAD-robust sigma), Rocha-
        # Cribari-Neto beta autoregression (link-scale
        # recursion, joint precision MLE), Ferland-Latour-
        # Oraichi Poisson INGARCH(1,1). Same SYNTHETIC
        # diagnostic contract.
        "log_concave",
        "dtw_warp",
        "chow_lin",
        "chen_tiao_outliers",
        "beta_ar",
        "ingarch",
        # wave 62 — Meucci entropy pooling (min relative
        # entropy posterior under scenario views, dual-
        # exponential tilting), Antolin-Diaz & Rubio-Ramirez
        # narrative SVAR (event-level sign + dominance
        # restrictions over rotation draws), Verbesselt BFAST
        # seasonal+trend break detection (Chow-F scan + BIC
        # second break), Nolan alpha-stable fit (McCulloch
        # quantile init + Kogon-Williams CF regression),
        # Kemna-Vorst geometric Asian closed form +
        # Turnbull-Wakeman moment match + geometric-CV
        # arithmetic MC, Black-Litterman reverse-optimized
        # equilibrium + Idzorek view posterior. Same
        # SYNTHETIC diagnostic contract.
        "entropy_pooling",
        "narrative_svar",
        "bfast",
        "stable_dist",
        "asian_option",
        "black_litterman",
        # wave 63 — Higham/Qi-Sun nearest correlation
        # matrix (Dykstra alternating corrections and
        # semismooth Newton on the dual), Lee-Carter
        # stochastic mortality (age-centred SVD +
        # random-walk kappa), Clauset-Shalizi-Newman
        # power-law tails (KS-optimal x_min + Hill MLE +
        # parametric-bootstrap p-value), Vasicek/Hull-
        # White futures convexity adjustment (Gaussian
        # integral closed form vs Hull heuristic),
        # Jurado-Ludvigson-Ng macro uncertainty factor
        # (common factor of forecast-error variances),
        # inverse-Gaussian first-passage law with the
        # Siegmund corrected-continuity barrier lift.
        # Same SYNTHETIC diagnostic contract.
        "higham_corr",
        "lee_carter",
        "power_law",
        "convexity_adj",
        "jln_uncertainty",
        "first_passage",
        # wave 64 — Lugannani-Rice saddlepoint tail
        # probabilities and the renormalized saddlepoint
        # density (gamma/normal CGF), Kraskov-Stogbauer-
        # Grassberger kNN mutual information plus its
        # conditional form, Schreiber transfer entropy
        # built on the conditional KSG estimator,
        # Brownian-bridge midpoint moments with the
        # closed-form barrier hit probability for MC
        # refinement, Jarrow-Turnbull reduced-form
        # credit (piecewise-flat hazard bootstrap,
        # survival curve, risky bond, par CDS), and
        # Campbell-Shiller log-linear VAR return
        # variance decomposition (dividend-news vs
        # discount-rate news). Same SYNTHETIC
        # diagnostic contract.
        "saddlepoint",
        "mutual_info",
        "transfer_entropy",
        "brownian_bridge",
        "jarrow_turnbull",
        "campbell_shiller",
        # Wave 65 — LIBOR market model (terminal-
        # measure forward simulation, Black caplets,
        # MC swaptions), Fang-Oosterlee COS Fourier-
        # cosine pricing, Obizhaeva-Wang transient-
        # impact optimal execution (block + rate
        # schedule, KKT cost), Margrabe/Kirk spread
        # and quanto options, Gerber-Shiu Esscher-
        # measure pricing on exponential-Levy CFs,
        # and Wu-Xia shadow-rate EKF term structure
        # at the zero lower bound. Same SYNTHETIC
        # diagnostic contract.
        "libor_market",
        "cos_method",
        "obizhaeva_wang",
        "spread_options",
        "esscher",
        "shadow_rate",
        # Wave 66 — Giles multilevel Monte Carlo
        # (coupled coarse/fine Euler levels, variance
        # decay beta), Black-Karasinski calibrated
        # trinomial lattice (Arrow-Debreu bond
        # repricing, caplets), Battiston DebtRank
        # systemic-risk propagation vs in-strength
        # centrality, Scheffer critical-slowing-down
        # early-warning signals (rolling AC1/variance
        # Kendall-tau + IAAFT surrogates), Bandt-
        # Pompe permutation entropy on the Rosso
        # complexity-entropy plane, and Liu-Wang
        # SVGD particle posterior transport. Same
        # SYNTHETIC diagnostic contract.
        "mlmc",
        "black_karasinski",
        "debtrank",
        "ews_signals",
        "permutation_entropy",
        "svgd",
        # Wave 67 — Skilling nested-sampling
        # evidence estimation (prior-shrinkage
        # trajectory, posterior-weighted dead
        # points), Del-Moral tempering SMC with
        # ESS-triggered resample + RW mutation,
        # Wood/Drovandi Bayesian synthetic
        # likelihood (Gaussian surrogate on
        # summaries + RW-MH), Auerbach-
        # Gorodnichenko state-dependent local
        # projections (logistic transition,
        # Newey-West), Andrews-Soares GMS
        # moment-inequality testing (kappa
        # selection + bootstrap max-stat), and
        # Euler risk contributions (ES tail
        # conditional means, kernel-smoothed
        # VaR). Same SYNTHETIC diagnostic
        # contract.
        "nested_sampling",
        "smc_samplers",
        "synthetic_likelihood",
        "state_dependent_lp",
        "moment_inequalities",
        "euler_risk",
        # Wave 68 — Demeterfi variance-swap
        # replication (OTM-strip quadrature,
        # Andersen-Bondarenko corridor IV),
        # Roberts-Tweedie MALA/ULA Langevin
        # MCMC with optimal-scaling acceptance
        # advantage, Ramsay-Silverman
        # trapezoid-weighted FPCA (Karhunen-
        # Loeve eigenpairs, FVE), Andrieu-
        # Doucet-Holenstein particle Gibbs
        # (conditional SMC + PMMH marginal
        # parameter update), Ripley K/L
        # second-order clustering with
        # Monte-Carlo CSR envelopes, and
        # Daubechies-Lu-Wu synchrosqueezed
        # wavelet ridges. Same SYNTHETIC
        # diagnostic contract.
        "vix_replication",
        "mala",
        "functional_pca",
        "particle_gibbs",
        "ripley_k",
        "synchrosqueezing",
        # Wave 69 — Dragomiretskiy-Zosso
        # variational mode decomposition
        # (one-sided-spectrum ADMM, spectral-
        # peak initialization), Stockwell-
        # Mansinha-Lowe S-transform (1/f-
        # width Gaussian TF), Hotelling CCA
        # with Wold PLS-SVD and Anderson
        # reduced-rank regression, Tenen-
        # baum-de Silva-Langford Isomap
        # (kNN-graph geodesics + classical
        # MDS), Matheron ordinary kriging
        # with fitted exponential vario-
        # gram, and Hyndman innovations
        # ETS damped-trend + Theta + SBA-
        # corrected Croston. Same
        # SYNTHETIC diagnostic contract.
        "vmd",
        "stockwell",
        "cca",
        "isomap",
        "kriging",
        "innovations_ets",
        # Wave 70 — Dandawate-Giannakis
        # cyclic-moment cyclostationarity
        # (per-segment demeaned second-
        # order moment), Gilles empirical
        # wavelet transform (spectrum-
        # minima band boundaries + Meyer-
        # raised-cosine filters), rank-
        # based inference (Mann-Whitney
        # common-language effect size,
        # Wilcoxon, Kruskal-Wallis,
        # Jonckheere-Terpstra trend),
        # Drasgow polychoric/tetrachoric
        # latent correlations (threshold-
        # ML + bivariate-normal integrals),
        # Aitchison compositional analy-
        # sis (clr/ilr + variation matrix
        # + Dirichlet moment fit), and
        # Fan-Lv SIS/ISIS sure-indepen-
        # dence screening. Same
        # SYNTHETIC diagnostic contract.
        "cyclostationary",
        "empirical_wavelets",
        "nonparametric_tests",
        "polychoric",
        "compositional",
        "sure_screening",
        # Wave 71 — Birnbaum/Lord item
        # response theory (Rasch JML +
        # two-stage 2PL calibration),
        # Lazarsfeld-Goodman latent class
        # EM (restarted Bernoulli mixture
        # + BIC), Wilks/Pillai/Hotelling-
        # Roy MANOVA omnibus (Bartlett
        # chi^2), Schonemann-Gower
        # orthogonal + generalized
        # Procrustes shape alignment,
        # Barlow PAVA isotonic regres-
        # sion + Zadrozny-Elkan calibra-
        # tion, and Lopez de Prado
        # hierarchical risk parity (Ward
        # dendrogram + recursive IVP
        # bisection). Same SYNTHETIC
        # diagnostic contract.
        "item_response",
        "latent_class",
        "manova",
        "procrustes",
        "isotonic",
        "hrp",
        # Wave 72 — Thurstone-Harman
        # principal-axis factor analysis
        # (SMC-iterated communalities +
        # Kaiser varimax), Neal (2003)
        # slice sampling (stepping-out +
        # shrinkage MCMC), Kuiper (1960)
        # rotation-invariant V-statistic
        # (one- and two-sample, Stephens
        # tail), Eilers-Marx P-splines
        # (de Boor basis + difference
        # penalty + GCV), Duchon-Wahba
        # thin-plate splines (r^2 log r
        # radial kernel + ridge), and
        # James-Stein minimax shrinkage
        # (positive-part + Efron-Morris
        # empirical Bayes). Same
        # SYNTHETIC diagnostic contract.
        "factor_analysis",
        "slice_sampling",
        "kuiper",
        "p_spline",
        "thin_plate",
        "james_stein",
        # Wave-73 families — Mardia
        # (1970) multivariate skew/kurtosis
        # omnibus (n*b1p chi^2 + b2p z),
        # Mantel (1967) distance-matrix
        # correlation + partial form
        # (row/col permutation), Moran
        # (1950) I + Geary (1954) c +
        # Getis-Ord (1992) G spatial
        # autocorrelation, Friedman (1937)
        # blocked ranks + Kendall W +
        # Page (1963) ordered L, Fisher
        # (1922) exact + McNemar (1947)
        # + Cochran-Mantel-Haenszel (1959)
        # contingency tables, and
        # Hoeffding (1948) D non-monotone
        # independence. Same SYNTHETIC
        # diagnostic contract.
        "mardia",
        "mantel",
        "moran",
        "friedman",
        "contingency",
        "hoeffding",
        # Wave-74 families — Siegel &
        # Tukey (1960) rank-spread +
        # Ansari-Bradley (1960) dispersion,
        # Mood (1950) chi^2 median test,
        # Cochran (1950) Q for related
        # binary columns, Quade (1979)
        # range-weighted block ranks, van
        # der Waerden (1952) normal-scores
        # k-sample, and Dunn (1964) post-hoc
        # with Holm (1979) step-down. Same
        # SYNTHETIC diagnostic contract.
        "dispersion_tests",
        "median_tests",
        "cochran_q",
        "quade",
        "van_der_waerden",
        "dunn_test",
        # Wave-75 families — Cronbach
        # (1951) alpha / KR-20 / Spearman-
        # Brown split-half reliability,
        # Shrout-Fleiss (1979) / McGraw-
        # Wong (1996) ICC forms + SEM,
        # Holland-Thayer (1988) MH DIF
        # (ETS A/B/C) + Swaminathan-Rogers
        # (1990) logistic DIF, Cronbach et
        # al. (1972) G-theory variance
        # components + D-study, McDonald
        # (1999) omega composite
        # reliability, and Wright-Masters
        # (1982) Rasch infit/outfit. Same
        # SYNTHETIC diagnostic contract.
        "cronbach",
        "icc",
        "dif",
        "g_theory",
        "omega",
        "rasch_fit",
        # Wave-76 families — Horvitz-Thompson
        # (1952) / Hajek (1971) design-based
        # survey estimation, post-stratification
        # + Deming-Stephan (1940) iterative
        # raking, Deville-Sarndal (1992) GREG
        # calibration, Fay-Herriot (1979) EBLUP
        # small-area estimation, one-stage
        # cluster sampling, and Kish (1965)
        # design effects. Same SYNTHETIC
        # diagnostic contract.
        "horvitz_thompson",
        "poststrat",
        "calibration_survey",
        "fay_herriot",
        "cluster_sampling",
        "design_effects",
        # Wave-77 families — Liang-Zeger
        # (1986) GEE marginal models,
        # Laird-Ware (1982) LMM via EM
        # ML + BLUPs, Cohen (1960) /
        # Fleiss (1971) / Krippendorff
        # (1970) inter-rater agreement +
        # Lin (1989) CCC + Bland-Altman
        # (1986) LoA, Liu-Ting-Zhou
        # (2008) isolation forest +
        # Hariri (2019) extended variant,
        # HEGY (1990) seasonal unit
        # roots + Canova-Hansen (1995)
        # seasonal stability, and van
        # Buuren (2011) MICE PMM +
        # Rubin (1987) pooling. Same
        # SYNTHETIC diagnostic contract.
        "gee",
        "lmm",
        "interrater",
        "isolation_forest",
        "hegy",
        "mice",
        # Wave-78 families — Tukey (1949)
        # HSD / Dunnett (1955) many-to-
        # one / Games-Howell (1976) /
        # Scheffe (1953) S-method post-
        # hoc comparisons, Plackett
        # (1975)-Luce (1959) MM + Borda
        # + Condorcet-Copeland + MC3
        # rank aggregation, Montgomery
        # / Roberts (1959) / Page (1954)
        # SPC (xbar-R, EWMA, CUSUM,
        # Kane capability), Roncalli
        # (2013)/Maillard (2010) ERC
        # risk parity, Mantegna (1999)
        # MST + Tumminello (2005) PMFG
        # topology, and Matteson-James
        # (2014) E-divisive energy
        # changepoints. Same SYNTHETIC
        # diagnostic contract.
        "multiple_comparisons",
        "rank_aggregation",
        "spc",
        "risk_parity",
        "mst_topology",
        "e_divisive",
        # wave 79 — Hyvarinen (1999)
        # FastICA deflationary ICA,
        # Reiner-Rubinstein (1991)
        # closed-form barrier options,
        # Schuirmann (1987) TOST
        # equivalence testing, Robins
        # (2000) marginal structural
        # models via stabilized IPTW,
        # Lee-Seung (1999/2001)
        # multiplicative-update NMF
        # with cophenetic consensus,
        # and Pitt-Shephard (1999)
        # auxiliary particle filter.
        # Same SYNTHETIC diagnostic
        # contract.
        "fastica",
        "barrier_options",
        "tost",
        "msm_causal",
        "nmf",
        "auxiliary_pf",
        # wave 80 — Royston-Parmar
        # (2011/2013) RMST + Uno (2004)
        # variance, Hull (2018) OIS
        # zero-curve bootstrap with
        # fixed-point tenor stripping,
        # Henze-Zirkler (1990) BHEP
        # MVN test, Epps-Singleton
        # (1985) ECF normality,
        # Watson (1961) U^2 circular
        # uniformity + Stephens (1970)
        # table, and Szekely-Rizzo
        # (2005) energy-distance MVN.
        # Same SYNTHETIC diagnostic
        # contract.
        "rmst",
        "ois_curve",
        "henze_zirkler",
        "epps_singleton",
        "watson",
        "energy_test",
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
