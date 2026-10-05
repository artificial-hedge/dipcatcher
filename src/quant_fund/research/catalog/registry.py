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
        # wave 81 — Mosimann (1962) /
        # Minka (2000) Dirichlet-
        # multinomial, Banerjee (2005)
        # vMF mixture EM, Freimer-
        # Mudholkar-Kollia-Lin (1988)
        # GLD + King-MacGillivray
        # (1999) starship, Tukey (1977)
        # g-and-h letter values,
        # Robbins-Monro (1951) /
        # Kiefer-Wolfowitz (1952) /
        # Spall (1992) stochastic
        # approximation, and von
        # Neumann (1949) / Dykstra
        # (1983) / Douglas-Rachford
        # (1956) convex projections.
        # Same SYNTHETIC diagnostic
        # contract.
        "dirichlet_multinomial",
        "vonmises_fisher",
        "fkml",
        "gandh",
        "robbins_monro",
        "pocs",
        # wave 82 — DeLong (1988)
        # correlated-AUC variance +
        # pairwise comparison, Passing-
        # Bablok (1983) robust method
        # comparison + Deming (1943)
        # orthogonal fit, McNemar
        # (1947) / Bowker (1948) /
        # Stuart (1955) / Bhapkar
        # (1979) marginal homogeneity,
        # Mardia-Watson-Wheeler (1972)
        # + Rao (1976) spacing +
        # Watson-Beran runs circular
        # tests, Samejima (1969) GRM +
        # Masters (1982) PCM IRT, and
        # Welch (1951) heteroscedastic
        # ANOVA + Games-Howell (1976).
        # Same SYNTHETIC diagnostic
        # contract.
        "delong_auc",
        "passing_bablok",
        "marginal_homogeneity",
        "circular_tests",
        "graded_irt",
        "welch_anova",
        # wave 83 — Lin (1989)
        # concordance correlation +
        # Bland-Altman (1986) limits
        # of agreement, Belsley-Kuh-
        # Welsch (1980) influence
        # diagnostics, Lan-DeMets
        # (1983) alpha-spending group
        # sequential (O'Brien-Fleming
        # 1979 / Pocock 1977) +
        # conditional power, Torgerson
        # (1958) classical MDS +
        # SMACOF (de Leeuw 1977),
        # Benzécri (1973) correspon-
        # dence analysis, and Mardia
        # (1976) / Fisher-Lee (1983) /
        # Jammalamadaka-Sarma (1988)
        # circular correlation.
        # Same SYNTHETIC diagnostic
        # contract.
        "lin_ccc",
        "influence",
        "group_sequential",
        "mds",
        "correspondence_analysis",
        "circular_correlation",
        # wave 84 — Hosking (1990)
        # L-moments + regional
        # frequency (GEV/GLO/GPA,
        # discordancy/heterogeneity),
        # Saltelli (2010) Sobol
        # indices + Morris (1991)
        # elementary effects,
        # Andersen-Gill (1982)
        # recurrent events (MCF,
        # PWP, WLW + cluster
        # sandwich), Dawid-Skene
        # (1979) EM + GLAD (2009)
        # annotation models,
        # Keogh (2007) matrix
        # profile + SAX (Lin 2007),
        # and Hyndman (2011) MinT
        # reconciliation (OLS/WLS/
        # shrunk). Same SYNTHETIC
        # diagnostic contract.
        "lmoments",
        "sobol_sensitivity",
        "recurrent_events",
        "dawid_skene",
        "matrix_profile",
        "hierarchical_reconciliation",
        # wave 85 — Hansen (2006)
        # CMA-ES + CSA, Dunning-Ertl
        # (2019) t-digest + Flajolet
        # (2007) HLL + Cormode-
        # Muthukrishnan (2005) CMS +
        # GK (2001) sketches, Schmidt
        # (1986) MUSIC + Roy-Kailath
        # (1989) ESPRIT, Mack (1993)
        # chain-ladder + BF (1972) +
        # ODP bootstrap (1999), Erlang
        # (1917) B/C/A + PK + Whitt
        # G/G/c + Jackson (1957),
        # and Atkinson (1970) /
        # Shorrocks (1980) / FGT
        # (1984) inequality indices.
        # Same SYNTHETIC diagnostic
        # contract.
        "cma_es",
        "sketches",
        "music_esprit",
        "chain_ladder",
        "erlang_queueing",
        "inequality_indices",
        # wave 86 — ASTM E1049-85
        # rainflow + Miner damage +
        # Goodman/Gerber/SWT
        # corrections, Blom &
        # Bar-Shalom (1988) IMM +
        # Bar-Shalom & Tse (1975)
        # PDA, Karrer-Newman (2011)
        # DC-SBM + Amini (2013)
        # spectral + Danon (2005)
        # NMI, Elkan-Noto (2008) +
        # du Plessis (2014) + Kiryo
        # (2017) nnPU, Perrin (2003)
        # GR4J + McCarthy (1938)
        # Muskingum + NSE/KGE, and
        # Brinson-Hood-Beebower
        # (1986) + Carino (1999)
        # attribution. Same SYNTHETIC
        # diagnostic contract.
        "rainflow_fatigue",
        "bayesian_tracking",
        "sbm_inference",
        "pu_learning",
        "gr4j_hydrology",
        "brinson_attribution",
        # wave 87 — Avellaneda-Stoikov
        # (2008) optimal market making
        # (reservation price + GLFT
        # intensity), Gillespie (1977)
        # direct SSA + tau-leaping with
        # SIR/Schlogl networks, Hamilton
        # (2018) + HP + Baxter-King/
        # Christiano-Fitzgerald cycle
        # filters, Corwin-Schultz (2012)
        # high-low spread + Roll (1984)
        # + Amihud (2002), Fotheringham-
        # Brunsdon-Charlton (2002) GWR
        # adaptive bandwidth, and
        # Schmittlein (1987)/Fader-
        # Hardie-Lee (2005) Pareto/BG-NBD
        # CLV. Same SYNTHETIC diagnostic
        # contract.
        "avellaneda_stoikov",
        "gillespie_ssa",
        "hamilton_filter",
        "corwin_schultz",
        "gwr_spatial",
        "pareto_nbd",
        # wave 88 — Tauchen (1986),
        # Tauchen-Hussey (1991) and
        # Rouwenhorst (1995) AR(1)
        # discretization, Lomb (1976)/
        # Scargle (1982)/Press-Rybicki
        # (1989) irregular periodogram,
        # Broomhead-King (1986)/
        # Golyandina (2001) SSA,
        # Beck-Katz (1995) PCSE +
        # Parks (1967) FGLS, Quandt
        # (1960)/Andrews (1993)/
        # Andrews-Ploberger (1994)/
        # Nyblom (1989) stability, and
        # Friedman (1984) supersmoother
        # with Cleveland (1979) LOWESS.
        # Same SYNTHETIC diagnostic
        # contract.
        "markov_discretization",
        "lomb_scargle",
        "singular_spectrum",
        "beck_katz",
        "quandt_andrews",
        "friedman_supersmoother",
        # Wave-89: classical hypothesis-
        # testing canon — KS/Cramer-von
        # Mises/Anderson-Darling (1952/
        # 1954) EDF tests, Shapiro-Wilk
        # (1965)/Jarque-Bera (1980)/
        # D'Agostino-Pearson (1973)
        # normality, Levene (1960)/
        # Brown-Forsythe (1974)/
        # Fligner-Killeen (1976)/
        # O'Brien (1979) scale
        # homogeneity, Ansari-Bradley
        # (1960)/Mood (1954)/Klotz
        # (1962)/Conover (1980)/
        # Gastwirth (1965) rank scale,
        # Goldfeld-Quandt (1965)/Park
        # (1966)/Glejser (1969)/
        # Breusch-Pagan (1979)/White
        # (1980) heteroskedasticity, and
        # Durbin-Watson (1950)/Durbin-h
        # (1970)/Breusch-Godfrey (1978)/
        # Ljung-Box (1978) serial
        # diagnostics. Same SYNTHETIC
        # diagnostic contract.
        "edf_tests",
        "normality_tests",
        "scale_homogeneity",
        "score_scale",
        "het_regressions",
        "serial_diagnostics",
        # Wave-90: Friedman (1991)
        # MARS hinge splines,
        # Breiman-Friedman (1985) ACE /
        # Tibshirani (1988) AVAS,
        # Friedman-Stuetzle (1981)
        # projection pursuit,
        # Newton-Raftery (1994)/
        # Gelfand-Dey (1994)/Chib
        # (1995)/Savage-Dickey/Ogata
        # (1989) marginal likelihoods,
        # Brent (1973)/Ridders (1979)/
        # Illinois root finders, and
        # SOBI (1997)/JADE (1993)/FOBI
        # (1989) blind source
        # separation. Same SYNTHETIC
        # diagnostic contract.
        "mars_regression",
        "ace_avas",
        "projection_pursuit",
        "marginal_likelihood",
        "root_finders",
        "blind_sources",
        # Wave-91:
        # Nelder-Mead (1965)/Powell
        # (1964)/nonlinear-CG/BFGS +
        # Levenberg-Marquardt,
        # k-means++/PAM/DBSCAN/OPTICS
        # clustering, LLE/Laplacian
        # eigenmaps/diffusion map/
        # t-SNE manifold learning,
        # Huber/S/LTS/MM robust
        # regression, Robbins/
        # Tweedie/Kiefer-Wolfowitz
        # empirical Bayes, and
        # 2^k/PB/CCD/Box-Behnken/
        # LHS/Fedorov-Dopt design.
        # Same SYNTHETIC diagnostic
        # contract.
        "unconstrained_optimizers",
        "clustering_methods",
        "manifold_learning",
        "robust_regression",
        "empirical_bayes",
        "design_experiments",
        # Wave-92 canon: simulated
        # annealing/DE/PSO/GA +
        # NSGA-II metaheuristics,
        # GP-EI/UCB/PI Bayesian
        # optimization, fuzzy
        # c-means + PC/PE/Xie-Beni,
        # Kohonen SOM + LVQ1,
        # PageRank/HITS/conductance
        # link topology, and
        # successive-halving/
        # Hyperband search. Same
        # SYNTHETIC diagnostic
        # contract.
        "metaheuristic_optimizers",
        "bayesian_optimization",
        "fuzzy_clustering",
        "self_organizing_maps",
        "pagerank_topology",
        "hyperband_search",
        # Wave-93 canon: CART/random
        # forest/GBM tree ensembles,
        # NCA/LMNN metric learning,
        # SVDD/Mahalanobis/LOF
        # one-class classification,
        # Laplace GP classification,
        # Gerchberg-Saxton HIO +
        # Wirtinger-flow phase
        # retrieval, and Rendle
        # factorization machines.
        # Same SYNTHETIC diagnostic
        # contract.
        "tree_ensembles",
        "metric_learning",
        "one_class_classification",
        "gp_classification",
        "phase_retrieval",
        "factorization_machine",
        # Wave-94 canon: Pegasos +
        # kernel SVMs, Fisher LDA/QDA
        # + regularized DA, coordinate
        # -descent elastic-net paths,
        # KRR/RFF/Nyström kernel
        # methods, collapsed-Gibbs
        # LDA topics, and
        # perceptron/PA/OGD/FTRL
        # online convex learners.
        # Same SYNTHETIC diagnostic
        # contract.
        "svm_classifiers",
        "discriminant_analysis",
        "coordinate_descent_enet",
        "kernel_methods",
        "lda_topics",
        "online_convex",
        # Wave-95 canon: Bayesian
        # linear + ARD evidence,
        # Bayes-net structure +
        # exact marginals, linear
        # + Polak-Ribière CG,
        # FW / pairwise FW over
        # Δ and ℓ1, OMP + K-SVD,
        # and (μ/μ,λ)/(1+1) ES.
        # Same SYNTHETIC diagnostic
        # contract.
        "bayesian_linear",
        "graphical_models",
        "conjugate_gradient",
        "frank_wolfe",
        "sparse_coding",
        "evolution_strategies",
        # Wave-96 canon: Gaussian /
        # multinomial / Bernoulli NB,
        # AdaBoost.M1 + LogitBoost
        # stumps, EP Bayesian probit,
        # item-kNN + ALS-WR + bias-MF
        # collaborative filtering,
        # Apriori association rules,
        # and fictitious play /
        # support enumeration /
        # regret matching.
        # Same SYNTHETIC diagnostic
        # contract.
        "naive_bayes",
        "adaboost",
        "expectation_propagation",
        "collaborative_filtering",
        "association_rules",
        "nash_equilibrium",
        # Wave-97 canon: ISTA /
        # FISTA prox-gradient,
        # Gauss/CC/Simpson
        # quadrature, RK45 +
        # EM/Milstein solvers,
        # label-prop + self-train,
        # OvR / softmax / ECOC,
        # and sparse PCA. Same
        # SYNTHETIC diagnostic
        # contract.
        "proximal_gradient",
        "quadrature",
        "ode_solvers",
        "semisupervised",
        "multiclass",
        "sparse_pca",
        # Wave-98 canon: value /
        # policy iteration, TD(0) /
        # SARSA / Q-learning,
        # sum-product + loopy BP,
        # ELM, nearest shrunken
        # centroids, CEM. Same
        # SYNTHETIC diagnostic
        # contract.
        "mdp_solvers",
        "td_learning",
        "belief_propagation",
        "extreme_learning",
        "nearest_centroid",
        "cross_entropy_method",
        # Wave-99 canon: conjugate
        # Gibbs, Laplace approx,
        # Gaussian KDE + LOO-CV,
        # whitened tensor power,
        # Dirichlet evidential,
        # and mean-shrinkage MTL.
        # Same SYNTHETIC diagnostic
        # contract.
        "gibbs_sampler",
        "laplace_approx",
        "kde",
        "tensor_power",
        "evidential",
        "multi_task",
        # Wave-100 canon: homotopy
        # continuation, Anderson
        # acceleration, sequence
        # accel, LSQR/CGLS, low-
        # discrepancy QMC, and
        # symplectic integrators.
        # Same SYNTHETIC diagnostic
        # contract.
        "homotopy_continuation",
        "anderson_accel",
        "sequence_accel",
        "iterative_ls",
        "qmc_sequences",
        "symplectic_ode",
        # Wave-101 canon: graph
        # traversal/topo/bipartite,
        # shortest paths, Dinic
        # max-flow + min-cut,
        # Hungarian + Hopcroft-Karp,
        # Tarjan SCC + bridges/
        # articulation, Algorithm X.
        # Same SYNTHETIC diagnostic
        # contract.
        "graph_traversal",
        "shortest_paths",
        "network_flow",
        "assignment",
        "graph_components",
        "exact_cover",
        # Wave-102 canon: multi-
        # armed bandits — UCB1/
        # eps-greedy/ETC, KL-UCB,
        # LinUCB + linear TS,
        # EXP3 + Hedge, SE + LUCB
        # best arm, SW-/D-UCB.
        # Same SYNTHETIC diagnostic
        # contract.
        "stochastic_bandits",
        "kl_bandits",
        "contextual_bandits",
        "adversarial_bandits",
        "best_arm",
        "nonstationary_bandits",
        # Wave-103 canon: Krylov +
        # randomized NLA — Lanczos,
        # Arnoldi/GMRES, HMT rSVD,
        # Nystrom, CUR leverage,
        # interpolative decomp.
        # Same SYNTHETIC diagnostic
        # contract.
        "lanczos",
        "arnoldi_gmres",
        "randomized_svd",
        "nystrom",
        "cur_decomp",
        "interpolative_decomp",
        # Wave-104 canon: OT II —
        # Sinkhorn + transport LP,
        # EMD 1-D/LP + Bures,
        # Gromov-Wasserstein,
        # unbalanced KL-UOT,
        # fixed-support barycenter,
        # fused GW. Same SYNTHETIC
        # diagnostic contract.
        "sinkhorn",
        "emd_lp",
        "gromov_wasserstein",
        "unbalanced_ot",
        "wasserstein_barycenter",
        "fused_gromov",
        # Wave-105 canon: game-tree
        # search on a subtraction-race
        # DAG — alpha-beta + TT, UCT,
        # PUCT, NegaScout, PN, df-pn.
        "alpha_beta",
        "mcts",
        "puct",
        "negascout",
        "proof_number",
        "dfpn",
        # Wave-106 canon: stiff
        # time integration — BDF,
        # Adams PECE, Radau IIA,
        # Strang split, ETDRK4,
        # Crank–Nicolson. Orders
        # measured vs exact.
        "bdf",
        "adams",
        "radau",
        "strang",
        "etdrk4",
        "crank_nicolson",
        # Wave-107 canon: transport
        # & HJB PDE — ADI,
        # Lax–Wendroff, WENO5,
        # level set, fast
        # marching, Godunov.
        "adi",
        "lax_wendroff",
        "weno",
        "level_set",
        "fast_marching",
        "godunov",
        # Wave-108 canon: matrix
        # functions & equations —
        # expm Padé, sqrtm,
        # Sylvester, CARE,
        # matrix sign, Toeplitz.
        "expm_pade",
        "matrix_sqrt",
        "sylvester",
        "riccati_care",
        "matrix_sign",
        "toeplitz_solve",
        # Wave-109 canon: motion
        # planning — Dubins,
        # RRT*, PRM, DWA,
        # min-snap, Frenet.
        "dubins",
        "rrt",
        "prm",
        "dwa",
        "min_snap",
        "frenet",
        # Wave-110 canon: digital
        # comms — Viterbi,
        # GF(256), RS, Costas,
        # Gardner, RRC.
        "viterbi_decode",
        "gf256",
        "reed_solomon",
        "costas",
        "gardner",
        "rrc_filter",
        # Wave-111 canon: geometry —
        # Kabsch, ICP, Fréchet,
        # Hausdorff, hull,
        # Delaunay.
        "kabsch",
        "icp",
        "frechet",
        "hausdorff",
        "convex_hull",
        "delaunay",
        # Wave-112 canon: DSP filters —
        # remez, IIR design, biquad,
        # filtfilt, resample, Farrow.
        "remez",
        "iir_design",
        "biquad",
        "filtfilt",
        "resample_poly",
        "farrow",
        # Wave-113 canon: multi-target
        # tracking — JV, JPDA, PHD,
        # MHT, CI, TDOA.
        "jonker_volgenant",
        "jpda",
        "phd",
        "mht",
        "cov_int",
        "tdoa",
        # Wave-114 canon: GNSS — Gold
        # codes, Klobuchar, Allan
        # variance, strapdown, LAMBDA,
        # RTK.
        "gold_code",
        "klobuchar",
        "allan_variance",
        "strapdown",
        "lambda_method",
        "rtk",
        # Wave-115 canon: UQ — Smolyak,
        # PCE, BQ, KL, active
        # subspace, MIMC.
        "smolyak",
        "pce",
        "bayesian_quadrature",
        "kl_expand",
        "active_subspace",
        "mimc",
        # Wave-116 canon: game theory —
        # CFR, Lemke–Howson,
        # replicator, Wardrop, VCG,
        # Nash bargaining.
        "cfr",
        "lemke_howson",
        "replicator",
        "wardrop",
        "vcg",
        "nash_bargain",
        # Wave-117 canon: reinforcement
        # learning — GAE, V-trace, TRPO,
        # PPO, DDPG, TD3.
        "gae",
        "vtrace",
        "trpo",
        "ppo",
        "ddpg",
        "td3",
        # Wave-118 canon: POMDP solvers —
        # QMDP, grid VI, PBVI, Perseus,
        # HSVI, POMCP.
        "qmdp",
        "grid_pomdp",
        "pbvi",
        "perseus",
        "hsvi",
        "pomcp",
        # Wave-119 canon: multi-agent RL —
        # VDN, QMIX, COMA, MADDPG,
        # MAPPO, mean-field Q.
        "vdn",
        "qmix",
        "coma",
        "maddpg",
        "mappo",
        "mf_q",
        # Wave-120 canon: cooperative games —
        # nucleolus, Banzhaf, Owen, Myerson
        # auction, Groves, envy-free.
        "nucleolus",
        "banzhaf",
        "owen",
        "myerson_auction",
        "groves",
        "envy_free",
        # Wave-121 canon: best-arm
        # identification — LUCB,
        # sequential halving, median
        # elim, UGapE, TTTS, TnS.
        "lil_ucb",
        "sequential_halving",
        "median_elim",
        "ugape",
        "ttts",
        "track_stop",
        # Wave-122 exec-summary SOTA:
        # RL execution, ML SOR,
        # OFI, PG market maker,
        # options flow, dark pool.
        "exec_rl",
        "smart_router",
        "order_flow_imbalance",
        "pg_mm",
        "options_flow",
        "dark_pool",
        # Wave-123 exec-summary NLP/gen:
        # say-echo-do, multimodal,
        # ts diffusion, ts GAN,
        # econ calendar, quantcode.
        "say_echo_do",
        "multimodal_fusion",
        "ts_diffusion",
        "synthetic_gan",
        "econ_calendar",
        "quantcode_bench",
        # Wave-124 exec-summary graph/meta:
        # asset + counterparty GNNs,
        # MAML, EWC, FedAvg, insider.
        "asset_gnn",
        "counterparty_gnn",
        "maml_portfolio",
        "continual_learning",
        "fed_avg",
        "insider_anomaly",
        # Wave-125 exec-summary pricing/XAI:
        # PINN, QUBO, SHAP, adv-robust,
        # risk flow, PCMCI miner.
        "pinn_pricing",
        "qubo_portfolio",
        "xai_shap",
        "adversarial_robust",
        "risk_flow",
        "causal_miner",
        # Wave-126 exec-summary deep-DL:
        # TFT, PatchTST, DeepLOB, set-
        # transformer, NODE, world model.
        "tft_forecaster",
        "patchtst",
        "lob_transformer",
        "set_transformer",
        "neural_ode",
        "world_model",
        # Wave-127 exec-summary DL-2:
        # contrastive repr, hypernet
        # alloc, neural Thompson, BNN
        # ensemble, option VAE, diff
        # policy.
        "contrastive_repr",
        "hypernetwork_alloc",
        "neural_thompson",
        "bnn_ensemble",
        "option_vae",
        "diff_policy",
        # Wave-128 exec-summary DL-3:
        # KAN, TSMixer, Informer,
        # chart-CNN, MAE, graph-
        # temporal.
        "kan_forecaster",
        "ts_mixer",
        "informer_attn",
        "cnn_alpha",
        "mask_autoencoder",
        "graph_temporal",
        # Wave-129 exec-summary DL-4:
        # iTransformer, TCN, FT-
        # Transformer, N-BEATS,
        # Mamba, CrossFormer.
        "itransformer",
        "tcn_forecaster",
        "ft_transformer",
        "nbeats_deep",
        "mambats",
        "crossformer",
        # Wave-130 offline-RL:
        # decision trf, CQL, IQL,
        # trajectory trf, SAC,
        # GAIL.
        "decision_transformer",
        "cql_agent",
        "iql_agent",
        "trajectory_transformer",
        "sac_agent",
        "gail_imitation",
        # Wave-131 generative-seq:
        # VQ-VAE, flow matching,
        # score SDE, consistency,
        # EBM, Perceiver.
        "vq_vae_ts",
        "flow_matching_ts",
        "score_sde_ts",
        "consistency_ts",
        "energy_ts",
        "perceiver_ts",
        # Wave-132 OOD canon:
        # Mahalanobis, MSP+ODIN,
        # GradNorm, energy, kNN,
        # ViM.
        "mahalanobis_ood",
        "max_softmax_ood",
        "gradient_norm_ood",
        "energy_ood",
        "knn_ood",
        "vim_ood",
        # Wave-133 amortized-UQ
        # canon: CNP, ANP, deep
        # kernel GP, ConvNP,
        # meta-UQ, LL-GP.
        "neural_process",
        "attentive_np",
        "deep_kernel_gp",
        "convnp",
        "meta_uq",
        "llaplace_gp",
        # Wave-134 differentiable
        # optimization: OptNet QP,
        # cvx layer, ICNN,
        # declarative, SPD, MPC.
        "optnet_qp",
        "cvxpy_layer",
        "input_convex",
        "deep_declarative",
        "spd_net",
        "diff_mpc",
        # Wave-135 GNN canon:
        # ChebNet, SAGE, GIN,
        # U-Net, APPNP, JK.
        "chebnet",
        "graphsage",
        "gin_gnn",
        "graph_unet",
        "apnp_prop",
        "jk_net",
        # Wave-136 attention canon:
        # Linformer, Performer,
        # linear, sliding, Sinkhorn,
        # Nyström.
        "linformer_attn",
        "performer_attn",
        "linear_attn",
        "sliding_attn",
        "sinkhorn_attn",
        "nystrom_attn",
        # Wave-137 distributional-RL
        # canon: C51, QR-DQN, IQN,
        # NoisyNet, PER, bootstrap.
        "c51_dqn",
        "qr_dqn",
        "iqn_dqn",
        "noisy_net",
        "prioritized_replay",
        "bootstrapped_dqn",
        # Wave-138 memory + world
        # model: LSH, mem-kNN, NTM,
        # DNC, RSSM, MPC.
        "reformer_lsh",
        "memorizing_transformer",
        "ntm_memory",
        "dnc_memory",
        "rssm_world",
        "mpc_planning",
        # Wave-139 SSL + TTA canon:
        # BYOL, Barlow, VICReg, Tent,
        # SHOT, TTT.
        "byol",
        "barlow_twins",
        "vicreg",
        "tent_tta",
        "shot_tta",
        "ttt_layer",
        # Wave-140 geometric canon:
        # hyperbolic, capsule, SIREN,
        # E(n)-GNN, monotone, soft-sort.
        "hyperbolic_nn",
        "capsule_dynamic",
        "siren_inr",
        "equivar_gnn",
        "monotonic_net",
        "sort_net",
        # Wave-141 certified robustness:
        # smoothing, IBP, CROWN,
        # Lipschitz, vector neurons, Gumbel-top-k.
        "randomized_smoothing",
        "ibp_bounds",
        "crown_bound",
        "lipschitz_net",
        "vector_neurons",
        "gumbel_topk",
        # Wave-142 sequence exotics:
        # S4, RWKV, Hyena, RetNet,
        # DeltaNet, MoD routing.
        "s4_ssm",
        "rwkv_wkv",
        "hyena_conv",
        "retnet_decay",
        "delta_net",
        "mixture_of_depths",
        # Wave-143 PEFT canon:
        # LoRA, QLoRA-NF4, DoRA,
        # prompt, prefix, task-vector merge.
        "lora_ft",
        "qlora_nf4",
        "dora_weight",
        "prompt_tuning",
        "prefix_tuning",
        "task_vector_merge",
        # Wave-144 inference canon:
        # spec decode, paged KV, flash,
        # GQA, sliding window, ring.
        "speculative_decoding",
        "paged_kv_cache",
        "flash_attn",
        "gqa_attn",
        "sliding_window_cache",
        "ring_attn",
        # Wave-145 retrieval canon:
        # BM25, DPR, ColBERT,
        # HyDE, cross-encoder, RRF.
        "bm25_retriever",
        "dpr_retriever",
        "colbert_late",
        "hyde_retrieval",
        "reranker_crossenc",
        "rrf_fusion",
        # Wave-146 test-time compute:
        # SC, PRM, MCTS, debate,
        # unlearning, KG embeddings.
        "consistency_vote",
        "verifier_prm",
        "mcts_reason",
        "debate_multiagent",
        "unlearn_ga",
        "knowledge_graph_embed",
        # Wave-147 alignment canon:
        # BT reward model, DPO, IPO,
        # KTO, GRPO, KL-PPO RLHF.
        "reward_model",
        "dpo_train",
        "ipo_train",
        "kto_train",
        "grpo_train",
        "rlhf_ppo",
        # Wave-148 interpretability canon:
        # SAE, steering, probes,
        # lens, patching, ablation.
        "sae_feature",
        "activation_steering",
        "probe_linear",
        "logit_lens",
        "patch_activation",
        "circuit_ablation",
        # Wave-149 data-dynamics canon:
        # distill, herding, curriculum,
        # smoothing, mixup, SAM.
        "dataset_distillation",
        "coreset_herding",
        "curriculum_magnitude",
        "label_smoothing",
        "mixup_cutmix",
        "sharpness_sam",
        # Wave-150 compression canon:
        # mag-prune, LTH, int8,
        # KD, low-rank, Fisher.
        "magnitude_pruning",
        "lottery_ticket",
        "quant_int8",
        "kd_distill",
        "lowrank_factor",
        "fisher_prune",
        # Wave-151 agentic canon:
        # ReAct, Toolformer, ToT,
        # Reflexion, multi-agent, judge.
        "react_loop",
        "toolformer_call",
        "plan_search",
        "reflexion_retry",
        "multi_agent_pipeline",
        "judge_pairwise",
        # Wave-152 privacy canon:
        # DP-SGD, sec-agg, FedAvg,
        # PATE, DLG, canary.
        "dp_sgd",
        "secure_agg",
        "fedavg_hetero",
        "pate_teacher",
        "gradient_leakage",
        "canary_exposure",
        # Wave-153 NAS canon:
        # random, evolution, DARTS,
        # ENAS-RL, one-shot, surrogate.
        "random_search_nas",
        "evolution_nas",
        "darts_nas",
        "enas_controller",
        "one_shot_nas",
        "arch_predictor",
        # Wave-154 vision canon:
        # CNN, ViT, CLIP, SimCLR,
        # DDIM, rollout.
        "convnet_baseline",
        "vit_classifier",
        "clip_align",
        "simclr_views",
        "diffusion_ddim",
        "attention_rollout",
        # Wave-155 causal-DL canon:
        # TARNet, Dragonnet, DeepIV,
        # CEVAE, CFRNet, DR-value.
        "tarnet_ite",
        "dragonnet_dr",
        "deep_iv",
        "cevae_latent",
        "causal_rep",
        "policy_value",
        # Wave-156 tabular canon:
        # BPE, tab-ResNet, NODE,
        # GrowNet, soft tree, TabM.
        "tokenizer_bpe",
        "tabular_resnet",
        "node_net",
        "grownet_boost",
        "soft_tree",
        "tabm_mini",
        # Wave-157 anomaly canon:
        # SVDD, DAGMM, USAD,
        # anom-Transformer, RRCF, TranAD.
        "deep_svdd",
        "dagmm",
        "usad",
        "anom_transformer",
        "rrcf",
        "tranad",
        # Wave-158 LTR canon:
        # RankNet, ListNet, ListMLE,
        # LambdaRank, ApproxNDCG, NeuralSort.
        "ranknet_ltr",
        "listnet_ltr",
        "listmle_ltr",
        "lambdarank_ltr",
        "approx_ndcg_ltr",
        "neural_sort_ltr",
        # Wave-159 optimizer canon:
        # Muon, Lion, Sophia,
        # Lookahead, LAMB, Adafactor.
        "muon_opt",
        "lion_opt",
        "sophia_opt",
        "lookahead_opt",
        "lamb_opt",
        "adafactor_opt",
        # Wave-160 FL canon:
        # SCAFFOLD, FedNova, Ditto,
        # MOON, FedOpt-Adam, MimeLite.
        "scaffold_fl",
        "fednova_fl",
        "ditto_fl",
        "moon_fl",
        "fedopt_adam",
        "mime_lite",
        # Wave-161 neural-operator canon:
        # FNO, DeepONet, low-rank,
        # PINO, GNO, CNO.
        "fno_1d",
        "deeponet",
        "lowrank_op",
        "pino_residual",
        "gno_lite",
        "cno_lite",
        # Wave-162 TS-foundation canon:
        # Chronos, TimesFM, Moirai,
        # Lag-Llama, Timer, MOMENT.
        "chronos_lite",
        "timesfm_lite",
        "moirai_lite",
        "lagllama_lite",
        "timer_lite",
        "moment_lite",
        # Wave-163 lifelong-CL canon:
        # PackNet, LwF, DER,
        # A-GEM, Piggyback, HAT.
        "packnet_cl",
        "lwf_cl",
        "der_cl",
        "agem_cl",
        "piggyback_cl",
        "hat_cl",
        # Wave-164 Bayesian-DL canon:
        # SWA-Gaussian, MC-dropout,
        # BBB, snapshot ens, concrete
        # dropout, VCL.
        "swag_diag",
        "mc_dropout",
        "bbb_vi",
        "snapshot_ens",
        "concrete_dropout",
        "vcl_online",
        # Wave-165 conditional-density canon:
        # MDN, cond-flow, diffusion
        # regressor, het-GP, CRPS net,
        # kernel mixture.
        "mdn_cond",
        "flow_regression",
        "diffusion_regressor",
        "het_gp",
        "crps_net",
        "kernel_mixture",
        # Wave-166 meta-learning canon:
        # Reptile, ProtoNet, Matching,
        # ANIL, Meta-SGD, R2D2.
        "reptile",
        "protonet",
        "matching_net",
        "anil_meta",
        "meta_sgd",
        "r2d2_meta",
        # Wave-167 graph-exotics canon:
        # algo reasoning, PNA, virtual
        # node, GPS, oversmooth, DGN.
        "algo_reasoning",
        "pna_agg",
        "virtual_node",
        "gps_transformer",
        "oversmooth_metric",
        "dgn_directional",
        # Wave-168 RL-exotics canon:
        # AWAC, REDQ, TD7-SALE,
        # CrossQ, DR3, OB2I.
        "awac",
        "redq",
        "td7_lite",
        "crossq",
        "dr3_reg",
        "ob2i",
        # Wave-169 amortized-inference canon:
        # ADVI, IWAE, NF-VI, SVGP,
        # structured VI, VRNN.
        "advi_bbvi",
        "iwae_bound",
        "nf_vi",
        "sparse_gp_sv",
        "structured_vi",
        "vrnn_seq",
        # Wave-170 causal-DL-2 canon:
        # X/R/S-T learners, CFRNET-balance,
        # CATE distill, DR-learner.
        "xlearner",
        "rlearner",
        "slearner_tlearner",
        "causal_rep_bal",
        "cate_distill",
        "net_drlearner",
        # Wave-171 conformal-2 canon:
        # CQR, survival CP, APS, LTT,
        # full CP, risk control.
        "cqr_pred",
        "survival_cp",
        "aps_cp",
        "ltt_cp",
        "full_cp",
        "risk_cp",
        # Wave-172 bandit-exotics canon:
        # PSRL, Gittins, Whittle,
        # CUCB, corruption-robust, NeuralUCB.
        "psrl",
        "gittins_index",
        "whittle_restless",
        "cucb",
        "corrupt_bandit",
        "neural_ucb",
        # Wave-173 diffusion-exotics canon:
        # EDM, rectified flow, stoch interp,
        # DDIM-ODE, cold diffusion, distill.
        "edm_karras",
        "rectified_flow",
        "stoch_interp",
        "ddim_ode",
        "cold_diffusion",
        "diff_distill",
        # Wave-174 graph-temporal canon:
        # DCRNN, STGCN, Graph WaveNet,
        # ASTGCN, MTGNN, AGCRN.
        "dcrnn_lite",
        "stgcn_lite",
        "gwnet_lite",
        "astgcn",
        "mtgnn_lite",
        "agcrn",
        # Wave-175 LM-components canon:
        # RoPE, ALiBi, SwiGLU, RMSNorm,
        # MoE router, muP init.
        "rope_attn",
        "alibi_attn",
        "swiglu_ffn",
        "rmsnorm_block",
        "moe_router",
        "mup_init",
        # Wave-176 data-centric canon:
        # BALD, cartography, EL2N,
        # forgetting, influence, prototypicality.
        "active_bald",
        "data_cartography",
        "el2n_scoring",
        "forgetting_events",
        "influence_func",
        "proto_prune",
        # Wave-177 neuromorphic canon:
        # LIF, STDP, surrogate SNN,
        # Izhikevich, LSM, temporal coding.
        "lif_neuron",
        "stdp_learn",
        "surrogate_snn",
        "izhikevich",
        "lsm_reservoir",
        "temporal_code",
        # Wave-178 multi-task-gradient canon:
        # PCGrad, MGDA, CAGrad, GradNorm,
        # Nash-MTL, IMTL-G.
        "pcgrad",
        "mgda_mtl",
        "cagrad_mtl",
        "gradnorm_bal",
        "nash_mtl",
        "imtl_g",
        # Wave-179 survival-DL canon:
        # DeepSurv, DeepHit, Cox-Time,
        # Nnet-survival, DRSA, PC-Hazard.
        "deepsurv",
        "deephit",
        "cox_time",
        "nnet_surv",
        "drsa_surv",
        "pchazard",
        # Wave-180 PDMP / exotic-sampling canon:
        # BPS, Zig-Zag, Boomerang, kinetic
        # Langevin, elliptical slice, RMALA.
        "bouncy_particle",
        "zigzag_sampler",
        "boomerang_sampler",
        "kinetic_langevin",
        "elliptical_slice",
        "riemannian_mala",
        # Wave-181 LM-arch-2 canon:
        # Mamba-2 SSD, xLSTM mLSTM, RWKV-7,
        # Titans memory, Gated DeltaNet, Longhorn.
        "mamba2_ssd",
        "xlstm_mlstm",
        "rwkv7",
        "titans_memory",
        "gated_deltanet",
        "longhorn_ssm",
        # Wave-182 normalizing-flow canon:
        # RealNVP, Glow, NSF, MAF, planar, IAF.
        "real_nvp",
        "glow_flow",
        "neural_spline_flow",
        "maf_flow",
        "planar_flow",
        "iaf_flow",
        # Wave-183 causal-structure-DL canon:
        # NOTEARS, DAGMA, GOLEM, NOTEARS-MLP,
        # DAG-GNN, CAM-prune.
        "notears",
        "dagma_lin",
        "golem_ev",
        "notears_mlp",
        "dag_gnn",
        "cam_prune",
        # Wave-184 training-dynamics canon:
        # Hessian eig, NTK, edge-of-stability,
        # LMC, catapult, grokking.
        "hessian_eig",
        "ntk_kernel",
        "edge_stability",
        "mode_connectivity",
        "catapult_phase",
        "neural_grok",
        # Wave-185 differentiable-algorithm canon:
        # STE, Gumbel relax, P&M grad,
        # IFT, ODE adjoint, smooth argmax.
        "st_estimator",
        "gumbel_relax",
        "perturb_map",
        "implicit_diff",
        "ode_adjoint",
        "smooth_argmax",
        # Wave-186 energy-based-model canon:
        # ISM, DSM, NCE, CD, PCD, adversarial.
        "score_matching",
        "denoising_sm",
        "noise_contrastive",
        "contrastive_divergence",
        "persistent_cd",
        "adversarial_ebm",
        # Wave-187 scientific-ML/PDE canon:
        # DeepRitz, weak form, BSDE, spectral,
        # MOL, Feynman-Kac MC.
        "deepritz_pinn",
        "weak_form_pinn",
        "fbsde_solver",
        "spectral_pde",
        "moc_lines",
        "feynman_kac_mc",
        # Wave-188 active-learning canon:
        # entropy, margin, QBC, k-center,
        # BADGE, EGL.
        "entropy_query",
        "margin_sampling",
        "qbc_committee",
        "coreset_kcenter",
        "badge_embed",
        "egl_change",
        # Wave-189 self-play/game-AI canon:
        # AZ-lite, ExIt, NFSP, PSRO,
        # deep-CFR, MCCFR.
        "alphazero_lite",
        "expert_iteration",
        "nfsp",
        "psro",
        "deep_cfr",
        "mccfr_outcome",
        # Wave-190 classical causal-discovery canon:
        # GES, FCI, ICA/Direct/VAR-LiNGAM, MMPC.
        "ges_search",
        "fci_alg",
        "ica_lingam",
        "direct_lingam",
        "var_lingam",
        "mmmb_select",
        # Wave-191 exploration canon: count bonus, RND, ICM, NGU,
        # RIDE, Go-Explore.
        "count_bonus",
        "rnd_explore",
        "icm_explore",
        "ngu_explore",
        "ride_explore",
        "go_explore",
        # Wave-192 info-theory canon: MMD, HSIC, MINE, NWJ,
        # copula MI, LSD.
        "mmd_two_sample",
        "hsic_independence",
        "mine_mi",
        "nwj_mi",
        "copula_mi",
        "lsd_deptest",
        # Wave-193 optimal-control canon: LQR, DDP, MPPI, PMP,
        # MPC-QP, LQG.
        "lqr_control",
        "ddp_solve",
        "mppi_control",
        "pmp_bangbang",
        "mpc_qp",
        "lqg_control",
        # Wave-194 stochastic-process canon: CIR, OU bridge,
        # Poisson/Hawkes thinning, Merton jumps, GP bridge.
        "cir_sim",
        "ou_bridge",
        "poisson_thinning",
        "hawkes_thinning",
        "levy_jump",
        "gp_bridge",
        # Wave-195 ensemble/adaptive-MCMC canon: emcee stretch,
        # DE-MCMC, DRAM, RJMCMC, pCN, independence MH.
        "emcee_stretch",
        "de_mcmc",
        "dram",
        "rjmcmc",
        "pcn_sampler",
        "indep_mh",
        # Wave-196 eigen canon: power+deflation, inverse iteration,
        # Jacobi, shifted QR, Hessenberg, bidiagonalization.
        "power_iter",
        "inverse_iter",
        "jacobi_eig",
        "qr_eig",
        "hessenberg_red",
        "bidiag_svd",
        # Wave-197 inventory canon: EOQ, newsvendor, (s,S), Wagner-Whitin,
        # base stock, Clark-Scarf echelon.
        "eoq_model",
        "newsvendor",
        "ss_policy",
        "wagner_whitin",
        "base_stock",
        "clark_scarf",
        # Wave-198 scheduling canon: Johnson flow-shop, NEH, LPT,
        # knapsack DP, TSP branch-bound, WSPT.
        "johnson_flowshop",
        "neh_heuristic",
        "lpt_schedule",
        "knapsack_dp",
        "tsp_branchbound",
        "spt_weighted",
        # Wave-199 quantum canon: QAOA MaxCut, VQE Ising, Grover, QPE,
        # quantum kernel, continuous-time quantum walk.
        "qaoa_maxcut",
        "vqe_ising",
        "grover_search",
        "qpe_phase",
        "qkernel_svm",
        "quantum_walk",
        # Wave-200 tensor-network canon: TT-SVD, DMRG, cross interp,
        # TEBD, MPS fidelity, TT rounding.
        "tt_svd",
        "dmrg_tfim",
        "tensor_cross",
        "tebd_quench",
        "mps_fidelity",
        "tt_round",
        # Wave-201 stochastic-control canon: PSOR, CRR, Kushner MCA,
        # HJB penalty, AB dual, exercise boundary.
        "psor_american",
        "crr_tree",
        "kushner_mca",
        "hjb_penalty",
        "dual_american",
        "exercise_boundary",
        # Wave-202 game-theory canon: LQ-MFG, flocking, Cournot,
        # Stackelberg, stochastic-game VI, potential game.
        "mfg_lq",
        "mfg_flocking",
        "nash_cournot",
        "stackelberg_game",
        "stochastic_game_vi",
        "potential_game",
        # Wave-203 information-geometry canon: Fisher-Rao, natural
        # gradient, mirror descent, Bregman NMF, alpha geodesic, JKO.
        "fisher_rao",
        "natural_gradient",
        "mirror_descent",
        "bregman_nmf",
        "alpha_geodesic",
        "jko_scheme",
        # Wave-204 queueing + reliability canon: Jackson, BCMP MVA,
        # Gordon-Newell, CTMC availability, renewal reward, vacations.
        "jackson_network",
        "bcmp_mva",
        "gordon_newell",
        "ctmc_availability",
        "renewal_reward",
        "vacation_queue",
        # Wave-205 auction canon: Vickrey, first-price BNE, all-pay,
        # ascending clock, double auction, GSP positions.
        "vickrey_auction",
        "first_price_auction",
        "all_pay_auction",
        "ascending_clock",
        "double_auction",
        "gsp_auction",
        # Wave-206 RL-theory canon: UCB bound, Hedge, eps-decay, PI
        # contraction, TD rate, Q-learning rate.
        "ucb_bound",
        "mw_hedge",
        "egreedy_decay",
        "pi_contraction",
        "td_rate",
        "qlearn_rate",
        # Wave-207 crypto canon: SHA-256, AES S-box, Shamir, Pedersen,
        # Diffie-Hellman, secp256k1.
        "sha256_impl",
        "aes_sbox",
        "shamir_secret",
        "pedersen_commit",
        "diffie_hellman",
        "ecc_secp256k1",
        # Wave-208 signal canon: CWT ridge, cepstrum, MVDR, Hilbert,
        # LPC formants, Goertzel.
        "cwt_ridge",
        "cepstrum_pitch",
        "mvdr_beamformer",
        "hilbert_instant",
        "lpc_formant",
        "goertzel_detect",
        # Wave-209 reliability canon: Weibull life, fault tree, RAM
        # Markov, FMEA, Arrhenius life-stress, RBD redundancy.
        "weibull_life",
        "fault_tree",
        "ram_markov",
        "fmea_rpn",
        "life_stress",
        "redundancy_block",
        # Wave-210 coding canon: LDPC BP, turbo BCJR, polar SC, BCH,
        # CRC, block interleaver.
        "ldpc_decoder",
        "turbo_decoder",
        "polar_code",
        "bch_code",
        "crc_check",
        "conv_interleaver",
        # Wave-211 integer-programming canon: Gomory cuts, column
        # generation, Benders, Lagrangian, branch-and-cut, Held-Karp.
        "gomory_cut",
        "column_generation",
        "benders_decomp",
        "lagrangian_relax",
        "branch_and_cut",
        "held_karp",
        # Wave-212 approximation-algorithm canon: greedy, primal-dual,
        # LP rounding, FPTAS, local search, Christofides.
        "greedy_set_cover",
        "primal_dual_vc",
        "lp_rounding_sc",
        "fptas_knapsack",
        "local_search_maxcut",
        "christofides_tsp",
        # Wave-213 SDP/relaxation canon: Goemans-Williamson SDP,
        # eigenvalue opt, SoS, Shor QCQP, spectral bisection, Hoffman.
        "sdp_maxcut",
        "eigenvalue_opt",
        "sos_certificate",
        "qcqp_relax",
        "spectral_bisection",
        "hoffman_bound",
        # Wave-214 online-algorithms canon: ski rental, marking paging,
        # work-function k-server, RANKING, secretary/prophet, OGD.
        "ski_rental",
        "marking_paging",
        "work_function_kserver",
        "ranking_matching",
        "secretary_prophet",
        "online_gradient",
        # Wave-215 stochastic-programming canon: L-shaped, scenario
        # tree, SAA, chance-scenario, DRO Wasserstein, robust budget.
        "two_stage_lshaped",
        "scenario_tree",
        "saa_consistency",
        "chance_scenario",
        "dro_wasserstein",
        "robust_budget",
        # Wave-216 advanced-MC-sampling canon: parallel tempering,
        # Wang-Landau, umbrella, metadynamics, WHAM, thermo-integration.
        "parallel_tempering",
        "wang_landau",
        "umbrella_sampling",
        "metadynamics",
        "wham",
        "thermo_integration",
        # Wave-217 estimation/filtering canon: H-inf, cubature KF,
        # MHE, variational Bayes, Huber filter, particle smoother.
        "hinf_filter",
        "cubature_kalman",
        "mhe",
        "variational_bayes",
        "huber_filter",
        "particle_smoother",
        # Wave-218 advanced-derivatives canon: Dupire local vol, SABR,
        # deep hedging, Heston calib, barrier adjoint, Andreasen-Huge.
        "dupire_localvol",
        "sabr_calib",
        "deep_hedge",
        "heston_calib",
        "barrier_adjoint",
        "andreasen_huge",
        # Wave-219 SAT/symbolic canon: CDCL, WalkSAT, unit prop,
        # 2-SAT SCC, BDD, LTL model check.
        "cdcl_solver",
        "walksat",
        "unit_propagation",
        "twosat_scc",
        "bdd_ops",
        "ltl_mc",
        # Wave-220 verification canon: k-induction, IC3/PDR,
        # BMC, invariant synth, Hoare, ranking fns, CEGAR.
        "k_induction",
        "ic3_pdr",
        "bmc_unroll",
        "invariant_synth",
        "hoare_logic",
        "ranking_function",
        "cegar_loop",
        # Wave-221 algebra canon: Groebner, resultant,
        # poly GCD, GF(2) factor, LLL, Newton interp.
        "buchberger",
        "resultant",
        "poly_gcd",
        "gf2_factor",
        "lll_reduce",
        "newton_interp",
        # Wave-222 number theory: Miller-Rabin, Pollard
        # rho, Tonelli-Shanks, CF/Pell, CRT, EC scalar.
        "miller_rabin",
        "pollard_rho",
        "tonelli_shanks",
        "continued_fraction",
        "crt_garner",
        "ec_scalar",
        # Wave-223 distributed systems: Paxos, Raft,
        # vector clocks, consistent hashing, gossip, PBFT.
        "paxos",
        "raft_election",
        "vector_clock",
        "consistent_hash",
        "gossip_epidemic",
        "pbft_lite",
        # Wave-224 string algorithms: Aho-Corasick, SAM,
        # KMP, edit distance, LZ77, BWT.
        "aho_corasick",
        "suffix_automaton",
        "kmp_search",
        "edit_distance",
        "lz77",
        "bwt_transform",
        # Wave-225 physics simulation: leapfrog N-body,
        # Barnes-Hut, SPH, rigid impulses, Verlet cloth, FEM truss.
        "nbody_leapfrog",
        "barnes_hut",
        "sph_fluid",
        "rigid_collision",
        "verlet_cloth",
        "fem_truss",
        # Wave-226 compiler/formal-language canon: regex NFA, Hopcroft
        # DFA minimization, CYK, dominators, liveness/DCE, linear-scan.
        "regex_engine",
        "dfa_minimize",
        "cyk_parser",
        "dominance_tree",
        "liveness_dce",
        "linscan_regalloc",
        # Wave-227 compression canon: Huffman, arithmetic, LZW/LZ78,
        # Golomb-Rice, rANS.
        "huffman_codes",
        "arithmetic_coding",
        "lzw_compress",
        "golomb_rice",
        "rans_coder",
        "lz78_dict",
        # Wave-228 CRDT canon: counters, sets, registers, sequences.
        "gcounter",
        "pncounter",
        "orset",
        "lww_map",
        "twopset",
        "rga_sequence",
        # Wave-229 probabilistic-membership canon: Bloom, cuckoo,
        # XOR/quotient filters, MinHash-LSH, SimHash.
        "bloom_filter",
        "cuckoo_filter",
        "xor_filter",
        "quotient_filter",
        "minhash_lsh",
        "simhash",
        # Wave-230 computational-geometry canon: triangulation,
        # clipping, intersection, point location, closest pair,
        # calipers.
        "ear_clipping",
        "sutherland_hodgman",
        "segment_intersection",
        "point_in_polygon",
        "closest_pair",
        "rotating_calipers",
        # Wave-231 architecture canon: pipeline, cache, branch
        # prediction, Tomasulo, paging/TLB, roofline.
        "cpu_pipeline",
        "cache_sim",
        "branch_predictor",
        "tomasulo_sim",
        "paging_sim",
        "roofline_model",
        # Wave-232 blockchain canon: Merkle, PoW, UTXO, retarget,
        # fork choice, block validation.
        "merkle_tree",
        "proof_of_work",
        "utxo_set",
        "difficulty_retarget",
        "fork_resolution",
        "block_validator",
        # Wave-233 compiler-2 canon: SSA, SCCP, GVN, coalescing,
        # scheduling, LICM.
        "ssa_construct",
        "sccp_const",
        "gvn_elim",
        "reg_coalesce",
        "instr_sched",
        "licm_hoist",
        # Wave-234 database canon: B+tree, WAL, joins, planner, MVCC,
        # LSM.
        "btree_index",
        "wal_recovery",
        "join_algos",
        "query_planner",
        "mvcc_isolation",
        "lsm_tree",
        # Wave-235 OS canon: schedulers, paging, deadlock, disk, journal.
        "round_robin_sched",
        "cfs_scheduler",
        "demand_paging",
        "deadlock_detect",
        "disk_sched",
        "fs_journal",
        # Wave-236 graphics canon.
        "raycaster",
        "bresenham_line",
        "scanline_fill",
        "zbuffer_render",
        "quaternion_slerp",
        "bsp_tree",
        "mvp_transform",
        # Wave-237 parser canon.
        "recursive_descent",
        "pratt_parser",
        "earley_parser",
        "slr_parser",
        "peg_packrat",
        "ll1_table",
        # Wave-238 networking canon.
        "tcp_aimd",
        "sliding_window",
        "token_bucket",
        "rtt_estimator",
        "nat_table",
        "http2_flow",
        # Wave-239 PL canon: HM inference, interpreter, CPS, macros, GC.
        "hm_inference",
        "tree_walk_interp",
        "cps_transform",
        "macro_expand",
        "gc_marksweep",
        "simple_types",
        # Wave-240 applied-crypto canon.
        "rsa_toy",
        "winternitz_ots",
        "merkle_ots",
        "blind_sig",
        "zkp_schnorr",
        "commit_reveal",
        # Wave-241 databases-2 canon.
        "aries_recovery",
        "two_phase_lock",
        "selinger_join",
        "mvcc_gc",
        "buffer_pool",
        "blink_tree",
        # Wave-242 consensus/distributed-2 canon.
        "multi_paxos",
        "epaxos",
        "viewstamped",
        "zab_protocol",
        "swim_gossip",
        "two_three_pc",
        # Wave-243 numeric-2 canon.
        "bignum",
        "fft_radix2",
        "int_sqrt",
        "karatsuba",
        "ntt",
        "strassen",
        # Wave-244 IR canon.
        "inverted_index",
        "lsh_dedup",
        "ngram_spell",
        "positional_index",
        # Wave-245 OS-2 canon.
        "elf_loader",
        "malloc_freelist",
        "mlfq_sched",
        "mmap_pager",
        "semaphore_monitor",
        "syscall_layer",
        # Wave-246 language-runtime canon.
        "bytecode_vm",
        "closure_conv",
        "inline_cache",
        "nan_tagging",
        # Wave-247 concurrency canon.
        "atomics_tas",
        "bakery_lock",
        "channel_select",
        # Wave-248 formal-language canon.
        "brzozowski_deriv",
        "cellular_automata",
        # Wave-249 bioinformatics canon.
        "debruijn_assemble",
        "fm_index",
        "motif_scan",
        # Wave-250 graph-3 canon.
        "astar_search",
        "bidirectional_dijkstra",
        "bron_kerbosch",
        # Wave-251 numerical-linalg-2 canon.
        "givens_qr",
        "jacobi_svd",
        "ldlt_solve",
        "lu_pivots",
        "orth_iter",
        "sturm_eig",
        # Wave-252 interpreters-3 canon.
        "gen_gc",
        "compacting_gc",
        "dispatch_table",
        "anf_cps",
        "trampoline_tc",
        # Wave-253 automata-3 canon.
        "tree_automata",
        "buchi_automata",
        "weighted_fst",
        "cfg_pda_equiv",
        "two_way_dfa",
        "register_automata",
        # Wave-254 applied-crypto-2 canon.
        "tls_handshake",
        "hmac_construct",
        "aead_etm",
        "merkle_damgard",
        "cbc_padding",
        "pbkdf2_kdf",
        # Wave-255 optimization-2 canon.
        "simplex_lp",
        "ellipsoid_method",
        "barrier_ip",
        "admm_lasso",
        "coord_descent",
        "proj_gradient",
        # Wave-256 memory-models canon.
        "hazard_pointer",
        "seqlock",
        "ms_queue",
        "epoch_reclaim",
        "flat_combining",
        "rcu_lock",
        # Wave-257 lattice-crypto canon.
        "lwe_kex",
        "ntru_toy",
        "bfv_fhe",
        "sis_hash",
        "sigma_or_proof",
        "chaum_pedersen",
        # Wave-258 databases-3 canon.
        "cascades_opt",
        "vectorized_exec",
        "zone_map",
        "func_dep",
        "bitmap_index",
        "adaptive_qp",
        # Wave-259 networking-2 canon.
        "bgp_pathvec",
        "dns_resolver",
        "nat_traversal",
        "arp_table",
        "dhcp_lease",
        "eth_switch",
        # Wave-260 matching canon.
        # Wave-261 robotics-2 canon.
        # Wave-262 HPC canon.
        # Wave-263 real-time canon.
        # Wave-264 numerical-linalg-3 canon.
        # Wave-265 program-analysis canon.
        "fuzzer_mutate",
        "taint_track",
        "asan_shadow",
        "symbolic_exec",
        "contract_check",
        "grammar_fuzz",
        # Wave-266 graphics-2 canon.
        "triangle_raster",
        "phong_shade",
        "mipmap_sample",
        "shadow_map",
        "bump_map",
        "ssao_lite",
        # Wave-267 GPU-architecture canon.
        "warp_scheduler",
        "simt_divergence",
        "bank_conflict",
        "mem_coalesce",
        "occupancy_calc",
        "shared_mem_tile",
        # Wave-268 crypto-4 canon.
        "elgamal_enc",
        "paillier_he",
        "fiat_shamir",
        "ot_12",
        "chacha_stream",
        "poly1305_mac",
        # Wave-269 computer-vision canon.
        "lk_flow",
        "orb_feature",
        "homography_4pt",
        "ransac_plane",
        "epipolar_8pt",
        "stereo_disparity",
        # Wave-270 computational-physics-2 canon.
        "lj_md",
        "fdtd_wave",
        "lattice_boltzmann",
        "ising_metro",
        "pic_plasma",
        "dmc_solver",
        # Wave-271 networking-3 canon.
        "ospf_lsa",
        "stp_spanning",
        "vlan_tag",
        "csma_ca",
        "icmp_path",
        "diffserv_qos",
        # Wave-272 control-theory-2 canon.
        "pid_antiwindup",
        "sliding_mode",
        "gain_schedule",
        "smith_predictor",
        "backstepping",
        "repetitive_ctrl",
        # Wave-273 compiler-3 canon.
        "partial_eval",
        "peephole_opt",
        "strength_red",
        "const_fold",
        "loop_unroll",
        "inline_expand",
        # Wave-274 bioinformatics-2 canon.
        "hmm_profile",
        "star_msa",
        "gc_skew",
        "orf_find",
        "kmer_count",
        "seq_logo",
        # Wave-275 databases-4 canon.
        "columnar_scan",
        "simd_filter",
        "late_materialize",
        "radix_join",
        "graceful_hash",
        "index_intersect",
        # Wave-276 distributed-systems-3 canon.
        "ra_mutex",
        "token_ring",
        "bully_elect",
        "chord_look",
        "quorum_rw",
        "causal_bcast",
        # Wave-277 signal-processing-4 canon.
        "stft_istft",
        "chirp_z",
        "fir_window",
        "prony_model",
        "wola_synth",
        "decimate_int",
        # Wave-278 econ-models-2 canon.
        "rbc_sim",
        "nk_phillips",
        "taylor_rule",
        "solow_model",
        "olg_model",
        "cobweb_model",
        # Wave-279 control-theory-3 canon.
        "luen_obsv",
        "dist_obsv",
        "mrac_adapt",
        "flat_track",
        "lyap_synth",
        "l2_gain",
        # Wave-280 algebraic-topology canon.
        "simp_betti",
        "boundary_sq",
        "euler_char",
        "rips_h1",
        "graph_h1",
        "winding_deg",
        # Wave-281 abstract-algebra canon.
        "group_table",
        "perm_group",
        "galois_field",
        "poly_ring",
        "ideal_member",
        "matrix_grp",
        # Wave-282 combinatorics canon.
        "subset_sum_dp",
        "stirling_count",
        "gray_code",
        "inversion_count",
        "ramsey_bound",
        "latin_square",
        # Wave-1400 KG-QA canon.
        "cronqa_lite_studies",
        "cwq_lite_studies",
        "grailqa_studies",
        "kgqa_lite_studies",
        "pweb_qa_studies",
        "qald_lite_studies",
        # Wave-1401 math-word-2 canon.
        "alg514_lite_studies",
        "dolphin_lite_studies",
        "draw_lite_studies",
        "lila_lite_studies",
        "math_doc_studies",
        "math_eval_studies",
        # Wave-1402 science-QA-2 canon.
        "ai2_arc_lite_studies",
        "arc_da_lite_studies",
        "drug_qa_lite_studies",
        "emrqa_lite_studies",
        "head_qa_lite_studies",
        "medmcqa_lite_studies",
        # Wave-1403 vision-doc-QA canon.
        "ai2d_lite_studies",
        "chart_qa_lite_studies",
        "docvqa_lite_studies",
        "infovqa_lite_studies",
        "mmqa_lite_studies",
        "ocrvqa_lite_studies",
        # Wave-1404 video-QA canon.
        "activitynet_qa_studies",
        "how2qa_lite_studies",
        "movie_qa_lite_studies",
        "msrvtt_qa_studies",
        "nextqa_lite_studies",
        "star_qa_lite_studies",
        # Wave-1405 audio-QA canon.
        "ambi_qa_studies",
        "audio_qa_lite_studies",
        "avsd_lite_studies",
        "clotho_qa_studies",
        "esc_qa_studies",
        "music_avqa_studies",
        # Wave-1406 temporal-QA canon.
        "menat_qa_studies",
        "syndq_lite_studies",
        "teas_qa_studies",
        "time_qa_studies",
        "timedial_qa_studies",
        "timetravel_lite_studies",
        # Wave-1407 conversational-QA canon.
        "canard_lite_studies",
        "clarq_lite_studies",
        "doqa_lite_studies",
        "duread_qa_studies",
        "orchid_qa_studies",
        "qrecc_lite_studies",
        # Wave-1408 abductive-reasoning canon.
        "abduct_qa_studies",
        "analogy_qa_studies",
        "arct_lite_studies",
        "entailment_qa_studies",
        "fusion_qa_studies",
        "proof_qa_studies",
        # Wave-1409 multi-hop-QA-2 canon.
        "bamboogle_lite_studies",
        "beerqa_lite_studies",
        "cider_qa_studies",
        "ensem_qa_studies",
        "fanqa_lite_studies",
        "hops_qa_studies",
        # Wave-1410 event-causality canon.
        "causal_qa_studies",
        "ecare_lite_studies",
        "event2mind_lite_studies",
        "event_qa_studies",
        "hippo_qa_studies",
        "intent_qa_studies",
        # Wave-1411 stance-toxicity canon.
        "fakeqa_lite_studies",
        "flame_qa_studies",
        "hate_qa_studies",
        "ironic_qa_studies",
        "offensive_qa_studies",
        "politeness_qa_studies",
        # Wave-1412 emotion-affect canon.
        "affect_qa_studies",
        "anger_qa_studies",
        "comfort_qa_studies",
        "distress_qa_studies",
        "emotion_qa_studies",
        "empathy_qa_studies",
        # Wave-1413 discourse-pragmatics canon.
        "anaphora_qa_studies",
        "coherence_qa_studies",
        "dialogue_act_studies",
        "discourse_qa_studies",
        "hedge_qa_studies",
        "implicit_qa_studies",
        # Wave-1414 folk-commonsense canon.
        "afford_qa_studies",
        "counter_qa_studies",
        "custom_qa_studies",
        "everyday_qa_studies",
        "folk_qa_studies",
        "moral_qa_studies",
        # Wave-1415 legal-regulatory canon.
        "case_qa_studies",
        "clause_qa_studies",
        "contract_qa_studies",
        "lawqa_lite_studies",
        "legal_qa_studies",
        "statute_qa_studies",
        # Wave-1416 financial-NLP canon.
        "analyst_qa_studies",
        "audit_qa_studies",
        "bank_qa_studies",
        "broker_qa_studies",
        "credit_qa_studies",
        "earnings_qa_studies",
        # Wave-1417 instruction-task canon.
        "checklist_qa_studies",
        "flow_qa_studies",
        "guide_qa_studies",
        "howto_qa_studies",
        "instruct_qa_studies",
        "lesson_qa_studies",
        # Wave-1418 lore-reference canon.
        "almanac_qa_studies",
        "atlas_qa_studies",
        "idiom_qa_studies",
        "jeopardy_qa_studies",
        "misc_qa_studies",
        "myth_qa_studies",
        # Wave-1419 narrative-genre canon.
        "anecdote_qa_studies",
        "ballad_qa_studies",
        "biography_qa_studies",
        "chronicle_qa_studies",
        "epic_qa_studies",
        "fable_qa_studies",
        # Wave-1420 reasoning-exotics canon.
        "cause_qa_studies",
        "claim_qa_studies",
        "conclusion_qa_studies",
        "deduction_qa_studies",
        "effect_qa_studies",
        "fallacy_qa_studies",
        # Wave-1421 spatial-navigation canon.
        "geospatial_qa_studies",
        "itinerary_qa_studies",
        "journey_qa_studies",
        "route_qa_studies",
        "spatial_qa_studies",
        "terrain_qa_studies",
        # Wave-1422 temporal-era canon.
        "calendar_qa_studies",
        "century_qa_studies",
        "date_qa_studies",
        "decade_qa_studies",
        "epoch_qa_studies",
        "era_qa_studies",
        # Wave-1423 education canon.
        "class_qa_studies",
        "course_qa_studies",
        "exam_qa_studies",
        "homework_qa_studies",
        "lecture_qa_studies",
        "seminar_qa_studies",
        # Wave-1424 design-spec canon.
        "blueprint_qa_studies",
        "design_qa_studies",
        "format_qa_studies",
        "layout_qa_studies",
        "pattern_qa_studies",
        "schema_qa_studies",
        # Wave-1425 media canon.
        "article_qa_studies",
        "broadcast_qa_studies",
        "column_qa_studies",
        "debate_qa_studies",
        "editorial_qa_studies",
        "headline_qa_studies",
        # Wave-1426 leisure canon.
        "challenge_qa_studies",
        "contest_qa_studies",
        "game_qa_studies",
        "hobby_qa_studies",
        "leisure_qa_studies",
        "match_qa_studies",
        # Wave-1427 terrain-2 canon.
        "canyon_qa_studies",
        "coast_qa_studies",
        "desert_qa_studies",
        "field_qa_studies",
        "forest_qa_studies",
        "glacier_qa_studies",
        # Wave-1428 governance canon.
        "agency_qa_studies",
        "bureau_qa_studies",
        "cabinet_qa_studies",
        "election_qa_studies",
        "government_qa_studies",
        "ministry_qa_studies",
        # Wave-1429 particle canon.
        "atom_qa_studies",
        "electron_qa_studies",
        "ion_qa_studies",
        "molecule_qa_studies",
        "neutron_qa_studies",
        "photon_qa_studies",
        # Wave-1430 wildlife canon.
        "animal_qa_studies",
        "bird_qa_studies",
        "ecosystem_qa_studies",
        "fish_qa_studies",
        "habitat_qa_studies",
        "insect_qa_studies",
        # Wave-1431 vehicle canon.
        "aircraft_qa_studies",
        "bike_qa_studies",
        "bus_qa_studies",
        "car_qa_studies",
        "engine_qa_studies",
        "plane_qa_studies",
        # Wave-1432 cuisine canon.
        "beverage_qa_studies",
        "cuisine_qa_studies",
        "dessert_qa_studies",
        "dish_qa_studies",
        "fruit_qa_studies",
        "ingredient_qa_studies",
        # Wave-1433 weather canon.
        "cloud_qa_studies",
        "frost_qa_studies",
        "hurricane_qa_studies",
        "rain_qa_studies",
        "storm_qa_studies",
        "wind_qa_studies",
        # Wave-1434 material canon.
        "alloy_qa_studies",
        "ceramic_qa_studies",
        "glass_qa_studies",
        "iron_qa_studies",
        "steel_qa_studies",
        "wood_qa_studies",
        # Wave-1435 anatomy canon.
        "blood_qa_studies",
        "bone_qa_studies",
        "brain_qa_studies",
        "heart_qa_studies",
        "muscle_qa_studies",
        "nerve_qa_studies",
        # Wave-1436 celestial canon.
        "comet_qa_studies",
        "galaxy_qa_studies",
        "moon_qa_studies",
        "nebula_qa_studies",
        "planet_qa_studies",
        "star_qa_studies",
        # Wave-1437 mythic canon.
        "deity_qa_studies",
        "dragon_qa_studies",
        "hero_qa_studies",
        "olympus_qa_studies",
        "phoenix_qa_studies",
        "titan_qa_studies",
        # Wave-1438 marine canon.
        "coral_qa_studies",
        "dolphin_qa_studies",
        "reef_qa_studies",
        "shark_qa_studies",
        "turtle_qa_studies",
        "whale_qa_studies",
        # Wave-1439 landform canon.
        "cliff_qa_studies",
        "crater_qa_studies",
        "dune_qa_studies",
        "fjord_qa_studies",
        "gorge_qa_studies",
        "mesa_qa_studies",
        # Wave-1440 flora canon.
        "bamboo_qa_studies",
        "cactus_qa_studies",
        "fern_qa_studies",
        "moss_qa_studies",
        "pine_qa_studies",
        "vine_qa_studies",
        # Wave-1441 avian canon.
        "crane_qa_studies",
        "eagle_qa_studies",
        "falcon_qa_studies",
        "owl_qa_studies",
        "raven_qa_studies",
        "swan_qa_studies",
        # Wave-1442 insect canon.
        "ant_qa_studies",
        "bee_qa_studies",
        "beetle_qa_studies",
        "butterfly_qa_studies",
        "cricket_qa_studies",
        "moth_qa_studies",
        # Wave-1443 gem canon.
        "amber_qa_studies",
        "amethyst_qa_studies",
        "crystal_qa_studies",
        "diamond_qa_studies",
        "emerald_qa_studies",
        "jade_qa_studies",
        # Wave-1444 wetland canon.
        "brook_qa_studies",
        "creek_qa_studies",
        "delta_qa_studies",
        "estuary_qa_studies",
        "marsh_qa_studies",
        "pond_qa_studies",
        # Wave-1445 arboreal canon.
        "birch_qa_studies",
        "cedar_qa_studies",
        "elm_qa_studies",
        "maple_qa_studies",
        "oak_qa_studies",
        "willow_qa_studies",
        # Wave-1446 instrument canon.
        "cello_qa_studies",
        "drum_qa_studies",
        "flute_qa_studies",
        "guitar_qa_studies",
        "piano_qa_studies",
        "violin_qa_studies",
        # Wave-1447 predator canon.
        "bear_qa_studies",
        "cheetah_qa_studies",
        "fox_qa_studies",
        "leopard_qa_studies",
        "lion_qa_studies",
        "wolf_qa_studies",
        # Wave-1448 reptile canon.
        "cobra_qa_studies",
        "frog_qa_studies",
        "gecko_qa_studies",
        "iguana_qa_studies",
        "python_qa_studies",
        "viper_qa_studies",
        # Wave-1449 marine mammal canon.
        "beluga_qa_studies",
        "manatee_qa_studies",
        "narwhal_qa_studies",
        "orca_qa_studies",
        "otter_qa_studies",
        "walrus_qa_studies",
        # Wave-1450 farm canon.
        "barn_qa_studies",
        "cow_qa_studies",
        "goat_qa_studies",
        "horse_qa_studies",
        "pig_qa_studies",
        "sheep_qa_studies",
        # Wave-1451 fruit canon.
        "apple_qa_studies",
        "cherry_qa_studies",
        "grape_qa_studies",
        "lemon_qa_studies",
        "mango_qa_studies",
        "peach_qa_studies",
        # Wave-1452 vegetable canon.
        "carrot_qa_studies",
        "cucumber_qa_studies",
        "garlic_qa_studies",
        "onion_qa_studies",
        "potato_qa_studies",
        "tomato_qa_studies",
        # Wave-1453 raptor canon.
        "condor_qa_studies",
        "harrier_qa_studies",
        "kestrel_qa_studies",
        "kite_qa_studies",
        "osprey_qa_studies",
        "vulture_qa_studies",
        # Wave-1454 insect-2 canon.
        "aphid_qa_studies",
        "hornet_qa_studies",
        "locust_qa_studies",
        "mosquito_qa_studies",
        "scarab_qa_studies",
        "termite_qa_studies",
        # Wave-1455 amphibian canon.
        "axolotl_qa_studies",
        "bullfrog_qa_studies",
        "newt_qa_studies",
        "salamander_qa_studies",
        "toad_qa_studies",
        "tree_frog_qa_studies",
        # Wave-1456 fish canon.
        "barracuda_qa_studies",
        "catfish_qa_studies",
        "cod_qa_studies",
        "piranha_qa_studies",
        "salmon_qa_studies",
        "tuna_qa_studies",
        # Wave-1457 forest-mammal canon.
        "badger_qa_studies",
        "beaver_qa_studies",
        "bison_qa_studies",
        "cougar_qa_studies",
        "elk_qa_studies",
        "lynx_qa_studies",
        # Wave-1458 desert-2 canon.
        "arroyo_qa_studies",
        "butte_qa_studies",
        "camel_qa_studies",
        "caravan_qa_studies",
        "mirage_qa_studies",
        "oasis_qa_studies",
        # Wave-1459 arctic canon.
        "arctic_fox_qa_studies",
        "caribou_qa_studies",
        "musk_ox_qa_studies",
        "penguin_qa_studies",
        "polar_bear_qa_studies",
        "reindeer_qa_studies",
        # Wave-1460 savanna canon.
        "baboon_qa_studies",
        "elephant_qa_studies",
        "gazelle_qa_studies",
        "giraffe_qa_studies",
        "wildebeest_qa_studies",
        "zebra_qa_studies",
        # Wave-1461 jungle canon.
        "gorilla_qa_studies",
        "jaguar_qa_studies",
        "macaw_qa_studies",
        "orangutan_qa_studies",
        "sloth_qa_studies",
        "toucan_qa_studies",
        # Wave-1462 ocean-life canon.
        "crab_qa_studies",
        "jellyfish_qa_studies",
        "octopus_qa_studies",
        "seahorse_qa_studies",
        "squid_qa_studies",
        "stingray_qa_studies",
        # Wave-1463 meadow canon.
        "acorn_qa_studies",
        "blossom_qa_studies",
        "canopy_qa_studies",
        "firefly_qa_studies",
        "sprout_qa_studies",
        "truffle_qa_studies",
        # Wave-1464 monolith canon.
        "abyss_qa_studies",
        "beacon_qa_studies",
        "blizzard_qa_studies",
        "monolith_qa_studies",
        "spire_qa_studies",
        "tempest_qa_studies",
        # Wave-1465 forge canon.
        "citadel_qa_studies",
        "forge_qa_studies",
        "grotto_qa_studies",
        "lighthouse_qa_studies",
        "quarry_qa_studies",
        "vault_qa_studies",
        # Wave-1466 bedrock canon.
        "basalt_qa_studies",
        "cathedral_qa_studies",
        "chasm_qa_studies",
        "crag_qa_studies",
        "plateau_qa_studies",
        "ravine_qa_studies",
        # Wave-1467 highland canon.
        "arch_qa_studies",
        "steppe_qa_studies",
        "summit_qa_studies",
        "tundra_qa_studies",
        "valley_qa_studies",
        "volcano_qa_studies",
        # Wave-1468 moorland canon.
        "dale_qa_studies",
        "fen_qa_studies",
        "glen_qa_studies",
        "heath_qa_studies",
        "knoll_qa_studies",
        "moor_qa_studies",
        # Wave-1469 coastal canon.
        "atoll_qa_studies",
        "bluff_qa_studies",
        "cove_qa_studies",
        "headland_qa_studies",
        "inlet_qa_studies",
        "islet_qa_studies",
        # Wave-1470 evergreen canon.
        "aspen_qa_studies",
        "fir_qa_studies",
        "holly_qa_studies",
        "juniper_qa_studies",
        "redwood_qa_studies",
        "sequoia_qa_studies",
        # Wave-1471 wildflower canon.
        "crocus_qa_studies",
        "daffodil_qa_studies",
        "daisy_qa_studies",
        "foxglove_qa_studies",
        "iris_qa_studies",
        "poppy_qa_studies",
        # Wave-1472 herb canon.
        "basil_qa_studies",
        "cardamom_qa_studies",
        "chervil_qa_studies",
        "cinnamon_qa_studies",
        "coriander_qa_studies",
        "cumin_qa_studies",
        # Wave-1473 spice canon.
        "clove_qa_studies",
        "dill_qa_studies",
        "fennel_qa_studies",
        "lemongrass_qa_studies",
        "mint_qa_studies",
        "nutmeg_qa_studies",
        # Wave-1474 waterbird canon.
        "bittern_qa_studies",
        "cormorant_qa_studies",
        "curlew_qa_studies",
        "ibis_qa_studies",
        "kingfisher_qa_studies",
        "loon_qa_studies",
        # Wave-1475 invertebrate canon.
        "cicada_qa_studies",
        "dragonfly_qa_studies",
        "grasshopper_qa_studies",
        "ladybug_qa_studies",
        "mantis_qa_studies",
        "scorpion_qa_studies",
        # Wave-1476 wildcat canon.
        "caracal_qa_studies",
        "jaguarundi_qa_studies",
        "margay_qa_studies",
        "ocelot_qa_studies",
        "puma_qa_studies",
        "serval_qa_studies",
        # Wave-1477 seabird canon.
        "albatross_qa_studies",
        "gannet_qa_studies",
        "petrel_qa_studies",
        "puffin_qa_studies",
        "shearwater_qa_studies",
        "skua_qa_studies",
        # Wave-1478 antelope canon.
        "antelope_qa_studies",
        "eland_qa_studies",
        "impala_qa_studies",
        "kudu_qa_studies",
        "oryx_qa_studies",
        "springbok_qa_studies",
        # Wave-1479 neotropical canon.
        "agouti_qa_studies",
        "armadillo_qa_studies",
        "capybara_qa_studies",
        "coati_qa_studies",
        "peccary_qa_studies",
        "tapir_qa_studies",
        # Wave-1480 reptile canon.
        "adder_qa_studies",
        "boa_qa_studies",
        "krait_qa_studies",
        "mamba_qa_studies",
        "monitor_qa_studies",
        "taipan_qa_studies",
        # Wave-1481 arthropod canon.
        "earwig_qa_studies",
        "katydid_qa_studies",
        "mayfly_qa_studies",
        "stonefly_qa_studies",
        "wasp_qa_studies",
        "weevil_qa_studies",
        # Wave-1482 mustelid canon.
        "ermine_qa_studies",
        "fisher_qa_studies",
        "marten_qa_studies",
        "mink_qa_studies",
        "polecat_qa_studies",
        "wolverine_qa_studies",
        # Wave-1483 mammal canon.
        "coyote_qa_studies",
        "ferret_qa_studies",
        "jackal_qa_studies",
        "marmot_qa_studies",
        "moose_qa_studies",
        "raccoon_qa_studies",
        # Wave-1484 marsupial canon.
        "bandicoot_qa_studies",
        "koala_qa_studies",
        "numbat_qa_studies",
        "quokka_qa_studies",
        "wallaby_qa_studies",
        "wombat_qa_studies",
        # Wave-1485 shorebird canon.
        "avocet_qa_studies",
        "egret_qa_studies",
        "heron_qa_studies",
        "plover_qa_studies",
        "sandpiper_qa_studies",
        "tern_qa_studies",
        # Wave-1486 wader canon.
        "flamingo_qa_studies",
        "godwit_qa_studies",
        "grebe_qa_studies",
        "pelican_qa_studies",
        "spoonbill_qa_studies",
        "stork_qa_studies",
        # Wave-1487 seabird-2 canon.
        "auk_qa_studies",
        "fulmar_qa_studies",
        "gull_qa_studies",
        "jaeger_qa_studies",
        "kittiwake_qa_studies",
        "tropicbird_qa_studies",
        # Wave-1488 wildcat-2 canon.
        "bobcat_qa_studies",
        "dingo_qa_studies",
        "kodkod_qa_studies",
        "oncilla_qa_studies",
        "panther_qa_studies",
        "tiger_qa_studies",
        # Wave-1489 bloom canon.
        "clover_qa_studies",
        "heather_qa_studies",
        "lavender_qa_studies",
        "lilac_qa_studies",
        "marigold_qa_studies",
        "primrose_qa_studies",
        # Wave-1490 blossom canon.
        "camellia_qa_studies",
        "dahlia_qa_studies",
        "sage_qa_studies",
        "thyme_qa_studies",
        "violet_qa_studies",
        "zinnia_qa_studies",
        # Wave-1491 tree canon.
        "cypress_qa_studies",
        "eucalyptus_qa_studies",
        "hemlock_qa_studies",
        "laurel_qa_studies",
        "magnolia_qa_studies",
        "spruce_qa_studies",
        # Wave-1492 marsupial-2 canon.
        "bilby_qa_studies",
        "echidna_qa_studies",
        "platypus_qa_studies",
        "possum_qa_studies",
        "quoll_qa_studies",
        "thylacine_qa_studies",
        # Wave-1493 antelope-2 canon.
        "bongo_qa_studies",
        "duiker_qa_studies",
        "hartebeest_qa_studies",
        "nyala_qa_studies",
        "topi_qa_studies",
        "waterbuck_qa_studies",
        # Wave-1494 reptile-2 canon.
        "anole_qa_studies",
        "chameleon_qa_studies",
        "hognose_qa_studies",
        "skink_qa_studies",
        "terrapin_qa_studies",
        "tuatara_qa_studies",
        # Wave-1495 mustelid-2 canon.
        "grison_qa_studies",
        "sable_qa_studies",
        "stoat_qa_studies",
        "tayra_qa_studies",
        "weasel_qa_studies",
        "zorilla_qa_studies",
        # Wave-1496 wader-2 canon.
        "jacana_qa_studies",
        "lapwing_qa_studies",
        "moorhen_qa_studies",
        "railbird_qa_studies",
        "snipe_qa_studies",
        "turnstone_qa_studies",
        # Wave-1497 marsupial-3 canon.
        "bettong_qa_studies",
        "cuscus_qa_studies",
        "numbat2_qa_studies",
        "pademelon_qa_studies",
        "potoroo_qa_studies",
        "woylie_qa_studies",
        # Wave-1498 seabird-3 canon.
        "auklet_qa_studies",
        "booby_qa_studies",
        "frigatebird_qa_studies",
        "guillemot_qa_studies",
        "murrelet_qa_studies",
        "razorbill_qa_studies",
        # Wave-1499 spice-2 canon.
        "oregano_qa_studies",
        "parsley_qa_studies",
        "rosemary_qa_studies",
        "saffron_qa_studies",
        "tarragon_qa_studies",
        "turmeric_qa_studies",
        # Wave-1500 invertebrate-2 canon.
        "caddisfly_qa_studies",
        "centipede_qa_studies",
        "horntail_qa_studies",
        "lacewing_qa_studies",
        "millipede_qa_studies",
        "spider_qa_studies",
        # Wave-1501 tree-2 canon.
        "acacia_qa_studies",
        "alder_qa_studies",
        "baobab_qa_studies",
        "olive_qa_studies",
        "palm_qa_studies",
        "sycamore_qa_studies",
        # Wave-1502 neotropical-2 canon.
        "anteater_qa_studies",
        "coatimundi_qa_studies",
        "kinkajou_qa_studies",
        "opossum_qa_studies",
        "paca_qa_studies",
        "tamandua_qa_studies",
        # Wave-1503 serpent-2 canon.
        "garter_qa_studies",
        "keelback_qa_studies",
        "kingsnake_qa_studies",
        "mockviper_qa_studies",
        "racer_qa_studies",
        "sidewinder_qa_studies",
        # Wave-1504 seabird-4 canon.
        "murre_qa_studies",
        "noddie_qa_studies",
        "prion_qa_studies",
        "shag_qa_studies",
        "skimmer_qa_studies",
        "storm_petrel_qa_studies",
        # Wave-1505 waterfowl canon.
        "gadwall_qa_studies",
        "pintail_qa_studies",
        "pochard_qa_studies",
        "shoveler_qa_studies",
        "teal_qa_studies",
        "wigeon_qa_studies",
        # Wave-1506 duck canon.
        "bufflehead_qa_studies",
        "canvasback_qa_studies",
        "eider_qa_studies",
        "mallard_qa_studies",
        "merganser_qa_studies",
        "scoter_qa_studies",
        # Wave-1507 raptor-2 canon.
        "buzzard_qa_studies",
        "caracara_qa_studies",
        "goshawk_qa_studies",
        "merlin_qa_studies",
        "peregrine_qa_studies",
        "sparrowhawk_qa_studies",
        # Wave-1508 songbird canon.
        "chickadee_qa_studies",
        "finch_qa_studies",
        "sparrow_qa_studies",
        "thrush_qa_studies",
        "warbler_qa_studies",
        "wren_qa_studies",
        # Wave-1509 songbird-2 canon.
        "bunting_qa_studies",
        "grosbeak_qa_studies",
        "nuthatch_qa_studies",
        "tanager_qa_studies",
        "titmouse_qa_studies",
        "vireo_qa_studies",
        # Wave-1510 corvid canon.
        "chough_qa_studies",
        "crow_qa_studies",
        "jackdaw_qa_studies",
        "jay_qa_studies",
        "magpie_qa_studies",
        "rook_qa_studies",
        # Wave-1511 hummingbird canon.
        "brilliant_qa_studies",
        "hermit_qa_studies",
        "hummingbird_qa_studies",
        "sapphire_qa_studies",
        "topaz_qa_studies",
        "woodstar_qa_studies",
        # Wave-1512 shorebird-2 canon.
        "dunlin_qa_studies",
        "knot_qa_studies",
        "oystercatcher_qa_studies",
        "phalarope_qa_studies",
        "stilt_qa_studies",
        "whimbrel_qa_studies",
        # Wave-1513 owl canon.
        "barnowl_qa_studies",
        "barred_owl_qa_studies",
        "eagle_owl_qa_studies",
        "screech_owl_qa_studies",
        "snowy_owl_qa_studies",
        "tawny_owl_qa_studies",
        # Wave-1514 butterfly canon.
        "blue_morpho_qa_studies",
        "cabbage_white_qa_studies",
        "fritillary_qa_studies",
        "monarch_qa_studies",
        "painted_lady_qa_studies",
        "swallowtail_qa_studies",
        # Wave-1515 beetle canon.
        "click_beetle_qa_studies",
        "dung_beetle_qa_studies",
        "ground_beetle_qa_studies",
        "rhino_beetle_qa_studies",
        "stag_beetle_qa_studies",
        "tiger_beetle_qa_studies",
        # Wave-1516 moth canon.
        "atlas_moth_qa_studies",
        "gypsy_moth_qa_studies",
        "hawk_moth_qa_studies",
        "luna_moth_qa_studies",
        "tussock_moth_qa_studies",
        "underwing_qa_studies",
        # Wave-1517 dragonfly canon.
        "clubtail_qa_studies",
        "damselfly_qa_studies",
        "darner_qa_studies",
        "forktail_qa_studies",
        "hawker_qa_studies",
        "spreadwing_qa_studies",
        # Wave-1518 hummingbird-2 canon.
        "coquette_qa_studies",
        "fairy_qa_studies",
        "jacobin_qa_studies",
        "lancebill_qa_studies",
        "sabrewing_qa_studies",
        "sheartail_qa_studies",
        # Wave-1519 mantis canon.
        "empusa_qa_studies",
        "ghost_mantis_qa_studies",
        "mantidfly_qa_studies",
        "orchid_mantis_qa_studies",
        "praying_mantis_qa_studies",
        "shield_mantis_qa_studies",
        # Wave-1520 fungi canon.
        "agaric_qa_studies",
        "bolete_qa_studies",
        "chanterelle_qa_studies",
        "inkcap_qa_studies",
        "morel_qa_studies",
        "puffball_qa_studies",
        # Wave-1521 orchid canon.
        "cattleya_qa_studies",
        "cymbidium_qa_studies",
        "dendrobium_qa_studies",
        "oncidium_qa_studies",
        "paphiopedilum_qa_studies",
        "phalaenopsis_qa_studies",
        # Wave-1522 succulent canon.
        "agave_qa_studies",
        "aloe_qa_studies",
        "echeveria_qa_studies",
        "haworthia_qa_studies",
        "lithops_qa_studies",
        "sedum_qa_studies",
        # Wave-1523 grass canon.
        "bluegrass_qa_studies",
        "fescue_qa_studies",
        "miscanthus_qa_studies",
        "pampas_qa_studies",
        "ryegrass_qa_studies",
        "switchgrass_qa_studies",
        # Wave-1524 sedge canon.
        "bulrush_qa_studies",
        "carex_qa_studies",
        "cattail_qa_studies",
        "cottongrass_qa_studies",
        "reed_qa_studies",
        "rush_qa_studies",
        # Wave-1525 moss canon.
        "clubmoss_qa_studies",
        "haircap_qa_studies",
        "hornwort_qa_studies",
        "liverwort_qa_studies",
        "quillwort_qa_studies",
        "sphagnum_qa_studies",
        # Wave-1526 fern canon.
        "bracken_qa_studies",
        "horsetail_qa_studies",
        "maidenhair_qa_studies",
        "staghorn_qa_studies",
        "swordfern_qa_studies",
        "treefern_qa_studies",
        # Wave-1527 lichen canon.
        "crustose_qa_studies",
        "foliose_qa_studies",
        "fruticose_qa_studies",
        "oakmoss_qa_studies",
        "usnea_qa_studies",
        "xanthoria_qa_studies",
        # Wave-1528 mineral canon.
        "calcite_qa_studies",
        "feldspar_qa_studies",
        "fluorite_qa_studies",
        "gypsum_qa_studies",
        "olivine_qa_studies",
        "quartz_qa_studies",
        # Wave-1529 gemstone canon.
        "aquamarine_qa_studies",
        "garnet_qa_studies",
        "opal_qa_studies",
        "ruby_qa_studies",
        "tanzanite_qa_studies",
        "tourmaline_qa_studies",
        # Wave-1530 alloy canon.
        "amalgam_qa_studies",
        "brass_qa_studies",
        "bronze_qa_studies",
        "nichrome_qa_studies",
        "pewter_qa_studies",
        "solder_qa_studies",
        # Wave-1531 woodpecker canon.
        "downy_qa_studies",
        "flicker_qa_studies",
        "pileated_qa_studies",
        "sapsucker_qa_studies",
        "woodpecker_qa_studies",
        "wryneck_qa_studies",
        # Wave-1532 riverbird canon.
        "bee_eater_qa_studies",
        "jacamar_qa_studies",
        "kookaburra_qa_studies",
        "motmot_qa_studies",
        "roller_qa_studies",
        "tody_qa_studies",
        # Wave-1533 canopybird canon.
        "aracari_qa_studies",
        "barbet_qa_studies",
        "honeyguide_qa_studies",
        "hornbill_qa_studies",
        "quetzal_qa_studies",
        "trogon_qa_studies",
        # Wave-1534 nightbird canon.
        "cuckoo_qa_studies",
        "frogmouth_qa_studies",
        "koel_qa_studies",
        "nighthawk_qa_studies",
        "nightjar_qa_studies",
        "roadrunner_qa_studies",
        # Wave-1535 aerialist canon.
        "martin_qa_studies",
        "needletail_qa_studies",
        "swallow_qa_studies",
        "swift_qa_studies",
        "swiftlet_qa_studies",
        "treeswift_qa_studies",
        # Wave-1536 columbid canon.
        "collared_dove_qa_studies",
        "dove_qa_studies",
        "mourning_dove_qa_studies",
        "pigeon_qa_studies",
        "turtle_dove_qa_studies",
        "woodpigeon_qa_studies",
        # Wave-1537 marshbird canon.
        "coot_qa_studies",
        "crake_qa_studies",
        "dabchick_qa_studies",
        "gallinule_qa_studies",
        "rail_qa_studies",
        "waterhen_qa_studies",
        # Wave-1538 wetland canon.
        "crowned_crane_qa_studies",
        "demoiselle_qa_studies",
        "finfoot_qa_studies",
        "limpkin_qa_studies",
        "trumpeter_qa_studies",
        "whooping_qa_studies",
        # Wave-1539 heron canon.
        "goliath_heron_qa_studies",
        "green_heron_qa_studies",
        "grey_heron_qa_studies",
        "night_heron_qa_studies",
        "purple_heron_qa_studies",
        "tiger_heron_qa_studies",
        # Wave-1540 egret canon.
        "cattle_egret_qa_studies",
        "glossy_ibis_qa_studies",
        "great_egret_qa_studies",
        "sacred_ibis_qa_studies",
        "snowy_egret_qa_studies",
        "squacco_qa_studies",
        # Wave-1541 pelagic canon.
        "anhinga_qa_studies",
        "darter_qa_studies",
        "diving_petrel_qa_studies",
        "gadfly_qa_studies",
        "manx_qa_studies",
        "mollymawk_qa_studies",
        # Wave-1542 raptor-3 canon.
        "accipiter_qa_studies",
        "bateleur_qa_studies",
        "falconet_qa_studies",
        "harpy_qa_studies",
        "lammergeier_qa_studies",
        "seriema_qa_studies",
        # Wave-1543 nightjar-2 canon.
        "oilbird_qa_studies",
        "owlet_nightjar_qa_studies",
        "pauraque_qa_studies",
        "poorwill_qa_studies",
        "potoo_qa_studies",
        "whip_poor_will_qa_studies",
        # Wave-1544 coraciiform canon.
        "hoopoe_qa_studies",
        "nunbird_qa_studies",
        "nunlet_qa_studies",
        "puffbird_qa_studies",
        "toco_qa_studies",
        "woodhoopoe_qa_studies",
        # Wave-1545 parrot canon.
        "amazon_qa_studies",
        "cockatoo_qa_studies",
        "conure_qa_studies",
        "kakapo_qa_studies",
        "kea_qa_studies",
        "lorikeet_qa_studies",
        # Wave-1546 rail-2 canon.
        "corncrake_qa_studies",
        "flufftail_qa_studies",
        "sora_qa_studies",
        "sungrebe_qa_studies",
        "swamphen_qa_studies",
        "takhe_qa_studies",
        # Wave-1547 columbid-2 canon.
        "crowned_pigeon_qa_studies",
        "cuckoo_dove_qa_studies",
        "emerald_dove_qa_studies",
        "fruit_dove_qa_studies",
        "ground_dove_qa_studies",
        "quail_dove_qa_studies",
        # Wave-1548 cuckoo-turaco canon.
        "ani_qa_studies",
        "coua_qa_studies",
        "guira_qa_studies",
        "hoatzin_qa_studies",
        "malkoha_qa_studies",
        "turaco_qa_studies",
        # Wave-1549 ratite canon.
        "cassowary_qa_studies",
        "emu_qa_studies",
        "kiwi_qa_studies",
        "ostrich_qa_studies",
        "rhea_qa_studies",
        "tinamou_qa_studies",
        # Wave-1550 lizard canon.
        "agama_qa_studies",
        "chuckwalla_qa_studies",
        "frilled_lizard_qa_studies",
        "monitor_lizard_qa_studies",
        "tegu_qa_studies",
        "uromastyx_qa_studies",
        # Wave-1551 viper canon.
        "bushmaster_qa_studies",
        "copperhead_qa_studies",
        "coral_snake_qa_studies",
        "cottonmouth_qa_studies",
        "fer_de_lance_qa_studies",
        "rattlesnake_qa_studies",
        # Wave-1552 turtle canon.
        "box_turtle_qa_studies",
        "map_turtle_qa_studies",
        "painted_turtle_qa_studies",
        "slider_qa_studies",
        "snapping_turtle_qa_studies",
        "tortoise_qa_studies",
        # Wave-1553 frog canon.
        "dart_frog_qa_studies",
        "horned_frog_qa_studies",
        "leopard_frog_qa_studies",
        "spring_peeper_qa_studies",
        "treefrog_qa_studies",
        "wood_frog_qa_studies",
        # Wave-1554 spider canon.
        "black_widow_qa_studies",
        "huntsman_qa_studies",
        "jumping_spider_qa_studies",
        "orb_weaver_qa_studies",
        "tarantula_qa_studies",
        "wolf_spider_qa_studies",
        # Wave-1555 arachnid-2 canon.
        "harvestman_qa_studies",
        "pseudoscorpion_qa_studies",
        "solifuge_qa_studies",
        "tick_qa_studies",
        "vinegaroon_qa_studies",
        "whip_scorpion_qa_studies",
        # Wave-1556 detritivore canon.
        "bristletail_qa_studies",
        "pillbug_qa_studies",
        "silverfish_qa_studies",
        "springtail_qa_studies",
        "velvet_worm_qa_studies",
        "woodlouse_qa_studies",
        # Wave-1557 amazon-fish canon.
        "arapaima_qa_studies",
        "electric_eel_qa_studies",
        "knifefish_qa_studies",
        "oscar_qa_studies",
        "pacu_qa_studies",
        "tetra_qa_studies",
        # Wave-1558 salmonid canon.
        "char_qa_studies",
        "dolly_varden_qa_studies",
        "grayling_qa_studies",
        "sockeye_qa_studies",
        "steelhead_qa_studies",
        "whitefish_qa_studies",
        # Wave-1559 pelagic-fish canon.
        "anchovy_qa_studies",
        "bonito_qa_studies",
        "herring_qa_studies",
        "kingfish_qa_studies",
        "mackerel_qa_studies",
        "sardine_qa_studies",
        # Wave-1560 reef-fish canon.
        "butterflyfish_qa_studies",
        "damselfish_qa_studies",
        "grouper_qa_studies",
        "parrotfish_qa_studies",
        "snapper_qa_studies",
        "wrasse_qa_studies",
        # Wave-1561 reef-fish-2 canon.
        "angelfish_qa_studies",
        "blenny_qa_studies",
        "goby_qa_studies",
        "lionfish_qa_studies",
        "surgeonfish_qa_studies",
        "triggerfish_qa_studies",
        # Wave-1562 reef-fish-3 canon.
        "boxfish_qa_studies",
        "clownfish_qa_studies",
        "dragonet_qa_studies",
        "mandarinfish_qa_studies",
        "pipefish_qa_studies",
        "pufferfish_qa_studies",
        # Wave-1563 crustacean canon.
        "cleaner_shrimp_qa_studies",
        "decorator_crab_qa_studies",
        "hermit_crab_qa_studies",
        "mantis_shrimp_qa_studies",
        "pistol_shrimp_qa_studies",
        "porcelain_crab_qa_studies",
        # Wave-1564 cephalopod canon.
        "bobtail_squid_qa_studies",
        "cuttlefish_qa_studies",
        "nautilus_qa_studies",
        "nudibranch_qa_studies",
        "sea_slug_qa_studies",
        "vampire_squid_qa_studies",
        # Wave-1565 annelid canon.
        "earthworm_qa_studies",
        "feather_duster_qa_studies",
        "leech_qa_studies",
        "lugworm_qa_studies",
        "polychaete_qa_studies",
        "ragworm_qa_studies",
        # Wave-1566 freshwater-fish canon.
        "bluegill_qa_studies",
        "crappie_qa_studies",
        "perch_qa_studies",
        "pike_qa_studies",
        "sturgeon_qa_studies",
        "walleye_qa_studies",
        # Wave-1567 cyprinid canon.
        "barbel_qa_studies",
        "bream_qa_studies",
        "carp_qa_studies",
        "minnow_qa_studies",
        "roach_qa_studies",
        "tench_qa_studies",
        # Wave-1568 eel canon.
        "conger_qa_studies",
        "garden_eel_qa_studies",
        "hagfish_qa_studies",
        "lamprey_qa_studies",
        "moray_qa_studies",
        "ribbon_eel_qa_studies",
        # Wave-1569 ray canon.
        "eagle_ray_qa_studies",
        "guitarfish_qa_studies",
        "manta_qa_studies",
        "sawfish_qa_studies",
        "thornback_qa_studies",
        "torpedo_ray_qa_studies",
        # Wave-1570 crab canon.
        "fiddler_crab_qa_studies",
        "ghost_crab_qa_studies",
        "horseshoe_qa_studies",
        "mud_crab_qa_studies",
        "porcelain_qa_studies",
        "spider_crab_qa_studies",
        # Wave-1571 bivalve canon.
        "clam_qa_studies",
        "conch_qa_studies",
        "mussel_qa_studies",
        "oyster_qa_studies",
        "scallop_qa_studies",
        "whelk_qa_studies",
        # Wave-1572 wildflower canon.
        "aster_qa_studies",
        "bluebell_qa_studies",
        "buttercup_qa_studies",
        "columbine_qa_studies",
        "cornflower_qa_studies",
        "lupine_qa_studies",
        # Wave-1573 desert canon.
        "addax_qa_studies",
        "fennec_qa_studies",
        "jerboa_qa_studies",
        "meerkat_qa_studies",
        "onager_qa_studies",
        "pangolin_qa_studies",
        # Wave-1574 primate canon.
        "gibbon_qa_studies",
        "langur_qa_studies",
        "lemur_qa_studies",
        "macaque_qa_studies",
        "marmoset_qa_studies",
        "tamarin_qa_studies",
        # Wave-1575 rodent canon.
        "chinchilla_qa_studies",
        "degu_qa_studies",
        "gerbil_qa_studies",
        "hamster_qa_studies",
        "lemming_qa_studies",
        "vole_qa_studies",
        # Wave-1576 mammal canon.
        "binturong_qa_studies",
        "fossa_qa_studies",
        "honey_badger_qa_studies",
        "kusimanse_qa_studies",
        "maned_wolf_qa_studies",
        "sun_bear_qa_studies",
        # Wave-1577 carnivore canon.
        "civet_qa_studies",
        "genet_qa_studies",
        "manul_qa_studies",
        "mongoose_qa_studies",
        "sloth_bear_qa_studies",
        "suricate_qa_studies",
        # Wave-1578 ungulate canon.
        "gerenuk_qa_studies",
        "markhor_qa_studies",
        "nilgai_qa_studies",
        "okapi_qa_studies",
        "saiga_qa_studies",
        "takin_qa_studies",
        # Wave-1579 dwarf-antelope canon.
        "dikdik_qa_studies",
        "grysbok_qa_studies",
        "klipspringer_qa_studies",
        "rhebok_qa_studies",
        "steenbok_qa_studies",
        "suni_qa_studies",
        # Wave-1580 deer canon.
        "chital_qa_studies",
        "fallow_qa_studies",
        "muntjac_qa_studies",
        "pudu_qa_studies",
        "roe_qa_studies",
        "sika_qa_studies",
        # Wave-1581 pinniped canon.
        "elephant_seal_qa_studies",
        "fur_seal_qa_studies",
        "harp_seal_qa_studies",
        "leopard_seal_qa_studies",
        "monk_seal_qa_studies",
        "weddell_qa_studies",
        # Wave-1582 bat canon.
        "flying_fox_qa_studies",
        "horseshoe_bat_qa_studies",
        "leaf_nosed_qa_studies",
        "noctule_qa_studies",
        "pipistrelle_qa_studies",
        "vampire_qa_studies",
        # Wave-1583 cetacean canon.
        "bowhead_qa_studies",
        "fin_whale_qa_studies",
        "humpback_qa_studies",
        "minke_qa_studies",
        "pilot_whale_qa_studies",
        "sperm_whale_qa_studies",
        # Wave-1584 small-mammal canon.
        "cottontail_qa_studies",
        "hare_qa_studies",
        "hedgehog_qa_studies",
        "hyrax_qa_studies",
        "jackrabbit_qa_studies",
        "pika_qa_studies",
        # Wave-1585 insectivore canon.
        "aardvark_qa_studies",
        "elephant_shrew_qa_studies",
        "golden_mole_qa_studies",
        "gymnure_qa_studies",
        "solenodon_qa_studies",
        "tenrec_qa_studies",
        # Wave-1586 plains-game canon.
        "beira_qa_studies",
        "gemsbok_qa_studies",
        "madoqua_qa_studies",
        "oribi_qa_studies",
        "reedbuck_qa_studies",
        "tsessebe_qa_studies",
        # Wave-1587 forest-deer canon.
        "barasingha_qa_studies",
        "brocket_qa_studies",
        "huemul_qa_studies",
        "mule_deer_qa_studies",
        "sambar_qa_studies",
        "taruca_qa_studies",
        # Wave-1588 pinniped-2 canon.
        "bearded_seal_qa_studies",
        "crabeater_qa_studies",
        "hooded_seal_qa_studies",
        "ribbon_seal_qa_studies",
        "ringed_seal_qa_studies",
        "ross_seal_qa_studies",
        # Wave-1589 cetacean-2 canon.
        "porpoise_qa_studies",
        "right_whale_qa_studies",
        "rissos_qa_studies",
        "river_dolphin_qa_studies",
        "spinner_qa_studies",
        "vaquita_qa_studies",
        # Wave-1590 small-cat canon.
        "black_footed_qa_studies",
        "fishing_cat_qa_studies",
        "jungle_cat_qa_studies",
        "pallas_qa_studies",
        "rusty_spotted_qa_studies",
        "sand_cat_qa_studies",
        # Wave-1591 new-world-monkey canon.
        "capuchin_qa_studies",
        "saki_qa_studies",
        "squirrel_monkey_qa_studies",
        "titi_qa_studies",
        "uakari_qa_studies",
        "woolly_qa_studies",
        # Wave-1592 prosimian canon.
        "bushbaby_qa_studies",
        "galago_qa_studies",
        "indri_qa_studies",
        "loris_qa_studies",
        "potto_qa_studies",
        "tarsier_qa_studies",
        # Wave-1593 old-world-monkey canon.
        "colobus_qa_studies",
        "drill_qa_studies",
        "gelada_qa_studies",
        "guenon_qa_studies",
        "mandrill_qa_studies",
        "mangabey_qa_studies",
        # Wave-1594 primate-2 canon.
        "bonobo_qa_studies",
        "chimpanzee_qa_studies",
        "douc_qa_studies",
        "proboscis_qa_studies",
        "siamang_qa_studies",
        "snub_nosed_qa_studies",
        # Wave-1595 felid-2 canon.
        "andean_cat_qa_studies",
        "bay_cat_qa_studies",
        "flat_headed_qa_studies",
        "geoffroys_qa_studies",
        "marbled_cat_qa_studies",
        "pampas_cat_qa_studies",
        # Wave-1596 ocean-mammal canon.
        "bottlenose_qa_studies",
        "dusky_dolphin_qa_studies",
        "false_killer_qa_studies",
        "melon_head_qa_studies",
        "pygmy_whale_qa_studies",
        "sea_lion_qa_studies",
        # Wave-1597 caprine canon.
        "bharal_qa_studies",
        "chamois_qa_studies",
        "goral_qa_studies",
        "ibex_qa_studies",
        "serow_qa_studies",
        "tahr_qa_studies",
        # Wave-1598 antelope-3 canon.
        "bontebok_qa_studies",
        "bushbuck_qa_studies",
        "greater_kudu_qa_studies",
        "lesser_kudu_qa_studies",
        "mountain_nyala_qa_studies",
        "sitatunga_qa_studies",
        # Wave-1599 primate-3 canon.
        "aye_aye_qa_studies",
        "howler_qa_studies",
        "mouse_lemur_qa_studies",
        "night_monkey_qa_studies",
        "ring_tailed_qa_studies",
        "spider_monkey_qa_studies",
        # Wave-1600 mollusk canon.
        "abalone_qa_studies",
        "chiton_qa_studies",
        "cockle_qa_studies",
        "cowrie_qa_studies",
        "limpet_qa_studies",
        "periwinkle_qa_studies",
        # Wave-1601 bat-2 canon.
        "blossom_bat_qa_studies",
        "bulldog_bat_qa_studies",
        "free_tailed_qa_studies",
        "fruit_bat_qa_studies",
        "mouse_eared_qa_studies",
        "tent_bat_qa_studies",
        # Wave-1602 fossorial canon.
        "desman_qa_studies",
        "marsupial_mole_qa_studies",
        "moles_lite_qa_studies",
        "monotreme_qa_studies",
        "moonrat_qa_studies",
        "sengi_qa_studies",
        # Wave-1603 deer-2 canon.
        "axis_qa_studies",
        "marsh_deer_qa_studies",
        "musk_deer_qa_studies",
        "pampas_deer_qa_studies",
        "tufted_qa_studies",
        "water_deer_qa_studies",
        # Wave-1604 burrow-mammal canon.
        "dassie_qa_studies",
        "gopher_qa_studies",
        "mole_qa_studies",
        "rabbit_qa_studies",
        "shrew_qa_studies",
        "springhare_qa_studies",
        # Wave-1605 intertidal-2 canon.
        "decorator_qa_studies",
        "fiddler_qa_studies",
        "rock_crab_qa_studies",
        "sea_snake_qa_studies",
        "skate_qa_studies",
        "wobbegong_qa_studies",
        # Wave-1606 camelid-steppe canon.
        "alpaca_qa_studies",
        "aoudad_qa_studies",
        "dromedary_qa_studies",
        "guanaco_qa_studies",
        "salt_qa_studies",
        "vicuna_qa_studies",
        # Wave-1607 primate-4 canon.
        "bamboo_lemur_qa_studies",
        "bearded_saki_qa_studies",
        "owl_monkey_qa_studies",
        "pale_titi_qa_studies",
        "uakari_2_qa_studies",
        "woolly_lemur_qa_studies",
        # Wave-1608 savanna-herd canon.
        "buffalo_qa_studies",
        "kob_qa_studies",
        "lechwe_qa_studies",
        "rhino_qa_studies",
        "roan_qa_studies",
        "warthog_qa_studies",
        # Wave-1609 lemur-2 canon.
        "black_lemur_qa_studies",
        "brown_lemur_qa_studies",
        "dwarf_lemur_qa_studies",
        "mongoose_lemur_qa_studies",
        "ruffed_qa_studies",
        "sportive_lemur_qa_studies",
        # Wave-1610 small-mammal-2 canon.
        "cavy_qa_studies",
        "coypu_qa_studies",
        "dhole_qa_studies",
        "mara_qa_studies",
        "porcupine_qa_studies",
        "ratel_qa_studies",
        # Wave-1611 lemur-3 canon.
        "crowned_lemur_qa_studies",
        "fat_tailed_qa_studies",
        "fork_marked_qa_studies",
        "needle_clawed_qa_studies",
        "ringtail_qa_studies",
        "sifaka_qa_studies",
        # Wave-1612 highland-grazer canon.
        "argali_qa_studies",
        "bighorn_qa_studies",
        "dall_qa_studies",
        "llama_qa_studies",
        "mouflon_qa_studies",
        "urial_qa_studies",
        # Wave-1613 bovine canon.
        "aurochs_qa_studies",
        "banteng_qa_studies",
        "gaur_qa_studies",
        "saola_qa_studies",
        "tamaraw_qa_studies",
        "yak_qa_studies",
        # Wave-1614 deer-3 canon.
        "hog_deer_qa_studies",
        "kouprey_qa_studies",
        "mule_qa_studies",
        "pere_david_qa_studies",
        "red_deer_qa_studies",
        "wapiti_qa_studies",
        # Wave-1615 lemur-4 canon.
        "golden_brown_qa_studies",
        "gray_mouse_qa_studies",
        "pygmy_qa_studies",
        "slender_qa_studies",
        "slow_qa_studies",
        "thin_spined_qa_studies",
        # Wave-1616 plankton-shore canon.
        "amphipod_qa_studies",
        "barnacle_qa_studies",
        "copepod_qa_studies",
        "isopod_qa_studies",
        "krill_qa_studies",
        "sandhopper_qa_studies",
        # Wave-1617 mouse-lemur-2 canon.
        "amber_mountain_qa_studies",
        "anosy_qa_studies",
        "daraina_qa_studies",
        "red_bellied_qa_studies",
        "russet_qa_studies",
        "white_footed_qa_studies",
        # Wave-1618 abyssal canon.
        "anglerfish_qa_studies",
        "bristlemouth_qa_studies",
        "grenadier_qa_studies",
        "hatchetfish_qa_studies",
        "lanternfish_qa_studies",
        "viperfish_qa_studies",
        # Wave-1619 abyssal-2 canon.
        "blobfish_qa_studies",
        "dragonfish_qa_studies",
        "dumbo_qa_studies",
        "fangtooth_qa_studies",
        "gulper_qa_studies",
        "tripodfish_qa_studies",
        # Wave-1620 venom-2 canon.
        "boomslang_qa_studies",
        "death_adder_qa_studies",
        "gaboon_qa_studies",
        "inland_taipan_qa_studies",
        "saw_scaled_qa_studies",
        "sea_krait_qa_studies",
        # Wave-1621 cave-dwelling canon.
        "cave_beetle_qa_studies",
        "cave_cricket_qa_studies",
        "cave_fish_qa_studies",
        "mudpuppy_qa_studies",
        "olm_qa_studies",
        "troglobite_qa_studies",
        # Wave-1622 alpine-ridgeline canon.
        "barbary_qa_studies",
        "blue_sheep_qa_studies",
        "himalayan_qa_studies",
        "nilgiri_qa_studies",
        "snow_leopard_qa_studies",
        "snowcock_qa_studies",
        # Wave-1623 alpine-bird canon.
        "altai_qa_studies",
        "blood_pheasant_qa_studies",
        "chukar_qa_studies",
        "monal_qa_studies",
        "snow_partridge_qa_studies",
        "wallcreeper_qa_studies",
        # Wave-1624 tundra canon.
        "arctic_hare_qa_studies",
        "gyrfalcon_qa_studies",
        "pallas_manul_qa_studies",
        "ptarmigan_qa_studies",
        "snowshoe_qa_studies",
        "tundra_swan_qa_studies",
        # Wave-1625 cave-2 canon.
        "blind_salamander_qa_studies",
        "cave_shrimp_qa_studies",
        "cave_spider_qa_studies",
        "cave_swiftlet_qa_studies",
        "grotto_salamander_qa_studies",
        "proteus_qa_studies",
        # Wave-1626 lemur-region canon.
        "bondolo_qa_studies",
        "madame_berthe_qa_studies",
        "mittermeier_qa_studies",
        "northern_qa_studies",
        "southern_qa_studies",
        "western_qa_studies",
        # Wave-1627 cave-3 canon.
        "cave_crayfish_qa_studies",
        "cave_scorpion_qa_studies",
        "cave_springtail_qa_studies",
        "cave_worm_qa_studies",
        "stygobite_qa_studies",
        "troglofish_qa_studies",
        # Wave-1628 exotic-fauna canon.
        "colugo_qa_studies",
        "geoffroy_qa_studies",
        "mandarin_qa_studies",
        "moray_eel_qa_studies",
        "pangolin_2_qa_studies",
        "satyr_qa_studies",
        # Wave-1629 cryptid canon.
        "chupacabra_qa_studies",
        "jersey_devil_qa_studies",
        "kraken_2_qa_studies",
        "mothman_qa_studies",
        "thunderbird_qa_studies",
        "yeti_2_qa_studies",
        # Wave-1630 cryptid-2 canon.
        "bigfoot_qa_studies",
        "bunyip_qa_studies",
        "loch_ness_qa_studies",
        "rougarou_qa_studies",
        "skinwalker_qa_studies",
        "wendigo_qa_studies",
        # Wave-1631 legendary-beast canon.
        "basilisk_qa_studies",
        "chimera_qa_studies",
        "gorgon_qa_studies",
        "griffin_2_qa_studies",
        "hydra_2_qa_studies",
        "manticore_qa_studies",
        # Wave-1632 legendary-2 canon.
        "cerberus_qa_studies",
        "dragon_2_qa_studies",
        "minotaur_qa_studies",
        "pegasus_qa_studies",
        "phoenix_2_qa_studies",
        "unicorn_2_qa_studies",
        # Wave-1633 elemental canon.
        "air_sylph_qa_studies",
        "earth_golem_qa_studies",
        "fire_spirit_qa_studies",
        "frost_wight_qa_studies",
        "storm_jinn_qa_studies",
        "water_sprite_qa_studies",
        # Wave-1634 elemental-2 canon.
        "gnome_2_qa_studies",
        "ifrit_qa_studies",
        "marid_qa_studies",
        "salamander_2_qa_studies",
        "sylph_2_qa_studies",
        "undine_qa_studies",
        # Wave-1635 yokai canon.
        "kappa_qa_studies",
        "kitsune_2_qa_studies",
        "oni_qa_studies",
        "tanuki_2_qa_studies",
        "tengu_qa_studies",
        "tsukumogami_qa_studies",
        # Wave-1636 yokai-2 canon.
        "gashadokuro_qa_studies",
        "jorogumo_qa_studies",
        "kodama_qa_studies",
        "namahage_qa_studies",
        "nue_2_qa_studies",
        "tsuchinoko_qa_studies",
        # Wave-1637 yokai-3 canon.
        "abura_sumashi_qa_studies",
        "azukiarai_qa_studies",
        "betobeto_2_qa_studies",
        "futakuchi_qa_studies",
        "rokurokubi_qa_studies",
        "shirime_qa_studies",
        # Wave-1638 yokai-4 canon.
        "akaname_qa_studies",
        "hitodama_qa_studies",
        "ittanmomen_qa_studies",
        "nurikabe_qa_studies",
        "shikigami_qa_studies",
        "ubume_qa_studies",
        # Wave-1639 yokai-5 canon.
        "dorotabo_qa_studies",
        "kitsune_3_qa_studies",
        "tanuki_3_qa_studies",
        "tengu_2_qa_studies",
        "yukionna_qa_studies",
        "zashiki_warashi_qa_studies",
        # Wave-1640 guardian-beast canon.
        "byakko_qa_studies",
        "genbu_qa_studies",
        "kirin_2_qa_studies",
        "kohryu_qa_studies",
        "seiryu_qa_studies",
        "suzaku_qa_studies",
        # Wave-1641 mythic-beast canon.
        "bixie_qa_studies",
        "fenghuang_qa_studies",
        "hundun_qa_studies",
        "qiongqi_qa_studies",
        "taotie_qa_studies",
        "taowu_qa_studies",
        # Wave-1642 chimera canon.
        "basilisk_2_qa_studies",
        "chimera_2_qa_studies",
        "cockatrice_qa_studies",
        "manticore_2_qa_studies",
        "sphinx_2_qa_studies",
        "wyvern_2_qa_studies",
        # Wave-1643 greek-beast canon.
        "centaur_2_qa_studies",
        "gryphon_qa_studies",
        "harpy_2_qa_studies",
        "hippogryph_qa_studies",
        "minotaur_2_qa_studies",
        "satyr_2_qa_studies",
        # Wave-1644 gorgon canon.
        "charybdis_qa_studies",
        "cyclops_2_qa_studies",
        "hydra_3_qa_studies",
        "medusa_2_qa_studies",
        "scylla_qa_studies",
        "siren_2_qa_studies",
        # Wave-1645 monster canon.
        "argus_qa_studies",
        "cerberus_2_qa_studies",
        "nemean_qa_studies",
        "orthrus_qa_studies",
        "pegasus_2_qa_studies",
        "typhon_qa_studies",
        # Wave-1646 norse-beast canon.
        "fenrir_2_qa_studies",
        "garm_qa_studies",
        "jormungandr_qa_studies",
        "nidhogg_qa_studies",
        "ratatoskr_qa_studies",
        "sleipnir_qa_studies",
        # Wave-1647 egyptian-beast canon.
        "akhekh_qa_studies",
        "ammit_qa_studies",
        "apophis_qa_studies",
        "bes_qa_studies",
        "sekhmet_qa_studies",
        "sphairo_qa_studies",
        # Wave-1648 slavic-beast canon.
        "aitvaras_qa_studies",
        "bilwis_qa_studies",
        "indus_qa_studies",
        "kudlak_qa_studies",
        "viy_qa_studies",
        "zilant_qa_studies",
        # Wave-1649 filipino-beast canon.
        "aswang_qa_studies",
        "bakunawa_qa_studies",
        "berbalang_qa_studies",
        "kapre_qa_studies",
        "sigbin_qa_studies",
        "tikbalang_qa_studies",
        # Wave-1650 african-beast canon.
        "adjule_qa_studies",
        "agogwe_qa_studies",
        "biloko_qa_studies",
        "kongamato_qa_studies",
        "popobawa_qa_studies",
        "rompo_qa_studies",
        # Wave-1651 australian-beast canon.
        "awgy_qa_studies",
        "kuritja_qa_studies",
        "minka_qa_studies",
        "papin_qa_studies",
        "yara_qa_studies",
        "yowie_qa_studies",
        # Wave-1652 european-beast canon.
        "cuco_qa_studies",
        "dahu_qa_studies",
        "gargouille_qa_studies",
        "lavellan_qa_studies",
        "muscaliet_qa_studies",
        "tarasque_qa_studies",
        # Wave-1653 global-beast canon.
        "alion_qa_studies",
        "catoblepas_qa_studies",
        "jasconius_qa_studies",
        "pard_qa_studies",
        "peluda_qa_studies",
        "zaratan_qa_studies",
        # Wave-1654 bestiary-beast canon.
        "amphisbaena_qa_studies",
        "bonnacon_qa_studies",
        "cerastes_qa_studies",
        "leucrotta_qa_studies",
        "parandrus_qa_studies",
        "questing_qa_studies",
        # Wave-1655 heraldic-beast canon.
        "basiliskcock_qa_studies",
        "calygreyhound_qa_studies",
        "cocatrix_qa_studies",
        "gryps_qa_studies",
        "mantygre_qa_studies",
        "opinicus_qa_studies",
        # Wave-1656 celtic-beast canon.
        "aatxe_qa_studies",
        "achiyalabopa_qa_studies",
        "afanc_qa_studies",
        "akhlut_qa_studies",
        "amarok_qa_studies",
        "eachuisge_qa_studies",
        # Wave-1657 french-beast canon.
        "gargoyle_qa_studies",
        "guivre_qa_studies",
        "melusine_qa_studies",
        "quinotaur_qa_studies",
        "tarascon_qa_studies",
        "tarrasque_qa_studies",
        # Wave-1658 mesoamerican-beast canon.
        "ahuizotl_qa_studies",
        "alicanto_qa_studies",
        "cadejo_qa_studies",
        "cipactli_qa_studies",
        "jinn_qa_studies",
        "quetzalcoat_qa_studies",
        # Wave-1659 philippine-beast canon.
        "manananggal_qa_studies",
        "minokawa_qa_studies",
        "nuno_qa_studies",
        "siyokoy_qa_studies",
        "tiyanak_qa_studies",
        "wakwak_qa_studies",
        # Wave-1660 greek-myth canon.
        "centaur_qa_studies",
        "cyclops_qa_studies",
        "griffin_qa_studies",
        "hydra_qa_studies",
        "medusa_qa_studies",
        "sphinx_qa_studies",
        # Wave-1661 norse-beast canon.
        "draugr_qa_studies",
        "fenrir_qa_studies",
        "gullinbursti_qa_studies",
        "hraesvelgr_qa_studies",
        "huginn_qa_studies",
        "muninn_qa_studies",
        # Wave-1662 celtic-beast canon.
        "banshee_qa_studies",
        "dullahan_qa_studies",
        "kelpie_qa_studies",
        "leprechaun_qa_studies",
        "puca_qa_studies",
        "selkie_qa_studies",
        # Wave-1663 british-folk canon.
        "barghest_qa_studies",
        "black_dog_qa_studies",
        "cat_sith_qa_studies",
        "church_grim_qa_studies",
        "cwn_annwn_qa_studies",
        "grimalkin_qa_studies",
        # Wave-1664 mythic-menagerie canon.
        "kraken_qa_studies",
        "krampus_qa_studies",
        "roc_qa_studies",
        "simurgh_qa_studies",
        "siren_qa_studies",
        "wyvern_qa_studies",
        # Wave-1665 norse-realm canon.
        "einherjar_qa_studies",
        "hati_qa_studies",
        "lindworm_qa_studies",
        "skoll_qa_studies",
        "vargbroder_qa_studies",
        "vedrfolnir_qa_studies",
        # Wave-1666 filipino-myth canon.
        "agta_qa_studies",
        "berberoka_qa_studies",
        "bungisngis_qa_studies",
        "dalaketnon_qa_studies",
        "ekek_qa_studies",
        "engkanto_qa_studies",
        # Wave-1667 greek-nature canon.
        "centauride_qa_studies",
        "dryad_qa_studies",
        "faun_qa_studies",
        "hamadryad_qa_studies",
        "nereid_qa_studies",
        "nymph_qa_studies",
        # Wave-1668 norse-warrior canon.
        "berserkr_qa_studies",
        "fafnir_qa_studies",
        "jotun_qa_studies",
        "regin_qa_studies",
        "ulfhednar_qa_studies",
        "vargr_qa_studies",
        # Wave-1669 filipino-creature-2 canon.
        "ghouling_qa_studies",
        "ikugan_qa_studies",
        "kataw_qa_studies",
        "lambana_qa_studies",
        "sarimanok_qa_studies",
        "tamahaling_qa_studies",
        # Wave-1670 greek-spirit canon.
        "alseid_qa_studies",
        "gnome_volk_qa_studies",
        "meliae_qa_studies",
        "napaea_qa_studies",
        "oread_qa_studies",
        "sylph_qa_studies",
        # Wave-1671 scandinavian-folk canon.
        "drakk_qa_studies",
        "grimr_qa_studies",
        "hildr_qa_studies",
        "mare_qa_studies",
        "nisse_qa_studies",
        "sigrun_qa_studies",
        # Wave-1672 slavic-domestic canon.
        "domovoi_qa_studies",
        "kikimora_qa_studies",
        "leshy_qa_studies",
        "polevik_qa_studies",
        "rusalka_qa_studies",
        "vodianoi_qa_studies",
        # Wave-1673 hindu-myth canon.
        "apsara_qa_studies",
        "asura_qa_studies",
        "gandharva_qa_studies",
        "naga_qa_studies",
        "rakshasa_qa_studies",
        "yaksha_qa_studies",
        # Wave-1674 roman-myth canon.
        "genii_qa_studies",
        "lares_qa_studies",
        "larvae_qa_studies",
        "lemures_qa_studies",
        "manes_qa_studies",
        "penates_qa_studies",
        # Wave-1675 aztec-myth canon.
        "chaneque_qa_studies",
        "cihuateteo_qa_studies",
        "nagual_qa_studies",
        "tlalocan_qa_studies",
        "tzitzimitl_qa_studies",
        "xiuhcoatl_qa_studies",
        # Wave-1676 hindu-myth-2 canon.
        "kinnara_qa_studies",
        "pisacha_qa_studies",
        "uraga_qa_studies",
        "vetala_qa_studies",
        "vidyadhara_qa_studies",
        "yakshini_qa_studies",
        # Wave-1677 greco-roman canon.
        "antheia_qa_studies",
        "aurae_qa_studies",
        "camenae_qa_studies",
        "fauns_qa_studies",
        "limoniad_qa_studies",
        "numina_qa_studies",
        # Wave-1678 slavic-folk-2 canon.
        "bannik_qa_studies",
        "dvorovoi_qa_studies",
        "mora_qa_studies",
        "ovinnik_qa_studies",
        "poludnica_qa_studies",
        "vila_qa_studies",
        # Wave-1679 norse-realm-2 canon.
        "alfheim_qa_studies",
        "bergrisi_qa_studies",
        "geirahod_qa_studies",
        "huldra_qa_studies",
        "troll_qa_studies",
        "vaetter_qa_studies",
        # Wave-1680 aztec-deity canon.
        "coatlicue_qa_studies",
        "mictlan_qa_studies",
        "mixcoatl_qa_studies",
        "tlaloc_qa_studies",
        "tonatiuh_qa_studies",
        "xipe_qa_studies",
        # Wave-1681 african-myth canon.
        "anansi_qa_studies",
        "impundulu_qa_studies",
        "kalulu_qa_studies",
        "mamiwata_qa_studies",
        "sasabonsam_qa_studies",
        "tokoloshe_qa_studies",
        # Wave-1682 slavic-wild canon.
        "alkonost_qa_studies",
        "gamayun_qa_studies",
        "sirin_qa_studies",
        "veles_qa_studies",
        "zhaba_qa_studies",
        "zmei_qa_studies",
        # Wave-1683 hindu-myth-3 canon.
        "danava_qa_studies",
        "gana_qa_studies",
        "gandharva_qa_studies",
        "kalakeya_qa_studies",
        "kimpurusha_qa_studies",
        "rakshasa_qa_studies",
        # Wave-1684 aztec-deity-2 canon.
        "cihuacoatl_qa_studies",
        "mayahuel_qa_studies",
        "oyohualli_qa_studies",
        "quetzalli_qa_studies",
        "teteoinnan_qa_studies",
        "yaotl_qa_studies",
        # Wave-1685 norse-spirit canon.
        "ettin_qa_studies",
        "fylgja_qa_studies",
        "landvaettir_qa_studies",
        "nokken_qa_studies",
        "seidr_qa_studies",
        "vette_qa_studies",
        # Wave-1686 african-myth-2 canon.
        "abada_qa_studies",
        "adze_qa_studies",
        "ilomba_qa_studies",
        "nbanda_qa_studies",
        "ninki_qa_studies",
        "okubi_qa_studies",
        # Wave-1687 egyptian-myth canon.
        "abti_qa_studies",
        "akh_qa_studies",
        "apep_qa_studies",
        "bastet_qa_studies",
        "khonsu_qa_studies",
        "sobek_qa_studies",
        # Wave-1688 roman-myth-2 canon.
        "indiges_qa_studies",
        "lar_qa_studies",
        "numen_qa_studies",
        "penates_qa_studies",
        "terminus_qa_studies",
        "vertumnus_qa_studies",
        # Wave-1689 hindu-myth-4 canon.
        "apsara_qa_studies",
        "bhairava_qa_studies",
        "bhuta_qa_studies",
        "pretas_qa_studies",
        "vetal_qa_studies",
        "yaksha_qa_studies",
        # Wave-1690 slavic-myth-3 canon.
        "kladenets_qa_studies",
        "kostroma_qa_studies",
        "leshii_qa_studies",
        "morozko_qa_studies",
        "vedmak_qa_studies",
        "yarilo_qa_studies",
        # Wave-1691 mesopotamian-myth canon.
        "abzu_qa_studies",
        "enki_qa_studies",
        "enlil_qa_studies",
        "nanna_qa_studies",
        "tiamat_qa_studies",
        "utu_qa_studies",
        # Wave-1692 norse-myth-4 canon.
        "alfar_qa_studies",
        "draugar_qa_studies",
        "hulder_qa_studies",
        "muspell_qa_studies",
        "svartalf_qa_studies",
        "ymir_qa_studies",
        # Wave-1693 mesopotamian-myth-2 canon.
        "asag_qa_studies",
        "edimmu_qa_studies",
        "galla_qa_studies",
        "lamassu_qa_studies",
        "shedu_qa_studies",
        "utukku_qa_studies",
        # Wave-1694 filipino-myth-3 canon.
        "duwende_qa_studies",
        "karibusa_qa_studies",
        "mambabarang_qa_studies",
        "mangkukulam_qa_studies",
        "sokoy_qa_studies",
        "tiktik_qa_studies",
        # Wave-1695 chinese-myth canon.
        "dijiang_qa_studies",
        "huli_qa_studies",
        "jiangshi_qa_studies",
        "mogwai_qa_studies",
        "yaoguai_qa_studies",
        "zhuyin_qa_studies",
        # Wave-1696 chinese-myth-2 canon.
        "dongwanggong_qa_studies",
        "fuxi_qa_studies",
        "kuafu_qa_studies",
        "nuwa_qa_studies",
        "shennong_qa_studies",
        "xiwangmu_qa_studies",
        # Wave-1697 chinese-myth-3 canon.
        "aoqin_qa_studies",
        "guandi_qa_studies",
        "houyi_qa_studies",
        "wenchang_qa_studies",
        "yutu_qa_studies",
        "zao_qa_studies",
        # Wave-1698 polynesian-myth canon.
        "maui_qa_studies",
        "menahune_qa_studies",
        "pele_qa_studies",
        "rangi_qa_studies",
        "tane_qa_studies",
        "tangaroa_qa_studies",
        # Wave-1699 japanese-myth canon.
        "amaterasu_qa_studies",
        "hachiman_qa_studies",
        "inari_qa_studies",
        "raijin_qa_studies",
        "susanoo_qa_studies",
        "tsukuyomi_qa_studies",
        # Wave-1700 celtic-myth canon.
        "brigid_qa_studies",
        "dagda_qa_studies",
        "danu_qa_studies",
        "lugh_qa_studies",
        "manannan_qa_studies",
        "morrigan_qa_studies",
        # Wave-1701 incan-myth canon.
        "illapa_qa_studies",
        "inti_qa_studies",
        "mamaquilla_qa_studies",
        "pachamama_qa_studies",
        "supay_qa_studies",
        "viracocha_qa_studies",
        # Wave-1702 mayan-myth canon.
        "chac_qa_studies",
        "hunab_qa_studies",
        "itzamna_qa_studies",
        "ixchel_qa_studies",
        "kukulcan_qa_studies",
        "yumkaax_qa_studies",
        # Wave-1703 sumerian-myth canon.
        "dumuzi_qa_studies",
        "inanna_qa_studies",
        "marduk_qa_studies",
        "namtar_qa_studies",
        "nergal_qa_studies",
        "ninhursag_qa_studies",
        # Wave-1704 babylonian-myth canon.
        "adad_qa_studies",
        "ashur_qa_studies",
        "ishtar_qa_studies",
        "nabu_qa_studies",
        "shamash_qa_studies",
        "sin_qa_studies",
        # Wave-1705 persian-myth canon.
        "ahriman_qa_studies",
        "ahuramazda_qa_studies",
        "anahita_qa_studies",
        "mithra_qa_studies",
        "verethragna_qa_studies",
        "yazata_qa_studies",
        # Wave-1706 siberian-myth canon.
        "akana_qa_studies",
        "erlik_qa_studies",
        "kayra_qa_studies",
        "perysh_qa_studies",
        "tengri_qa_studies",
        "ulgen_qa_studies",
        # Wave-1707 finno-ugric-myth canon.
        "ahti_qa_studies",
        "ilmatar_qa_studies",
        "kiputytto_qa_studies",
        "louhi_qa_studies",
        "tapio_qa_studies",
        "ukko_qa_studies",
        # Wave-1708 nenets-myth canon.
        "metsik_qa_studies",
        "naveluz_qa_studies",
        "numishi_qa_studies",
        "numit_qa_studies",
        "piryani_qa_studies",
        "yejmun_qa_studies",
        # Wave-1709 hittite-myth canon.
        "arinniti_qa_studies",
        "hannahanna_qa_studies",
        "inara_qa_studies",
        "kamrusepa_qa_studies",
        "tarhunna_qa_studies",
        "telepinu_qa_studies",
        # Wave-1710 canaanite-myth canon.
        "anat_qa_studies",
        "asherah_qa_studies",
        "baal_qa_studies",
        "lotan_qa_studies",
        "mot_qa_studies",
        "yam_qa_studies",
        # Wave-1711 phoenician-myth canon.
        "baalat_qa_studies",
        "dagon_qa_studies",
        "eshmun_qa_studies",
        "melqart_qa_studies",
        "resheph_qa_studies",
        "tanit_qa_studies",
        # Wave-1712 armenian-myth canon.
        "astghik_qa_studies",
        "hayk_qa_studies",
        "nahapet_qa_studies",
        "nane_qa_studies",
        "tir_qa_studies",
        "vahagn_qa_studies",
        # Wave-1713 georgian-myth canon.
        "amirani_qa_studies",
        "apsat_qa_studies",
        "barbale_qa_studies",
        "dalis_qa_studies",
        "ghmerti_qa_studies",
        "kamar_qa_studies",
        # Wave-1714 scythian-myth canon.
        "argimpasa_qa_studies",
        "arimasp_qa_studies",
        "papaios_qa_studies",
        "tabiti_qa_studies",
        "tavrita_qa_studies",
        "thagimasadas_qa_studies",
        # Wave-1715 dacian-myth canon.
        "bendis_qa_studies",
        "darzalas_qa_studies",
        "derzelas_qa_studies",
        "kezion_qa_studies",
        "sabazios_qa_studies",
        "zamolxis_qa_studies",
        # Wave-1716 thracian-myth canon.
        "heroas_qa_studies",
        "kottiso_qa_studies",
        "kotys_qa_studies",
        "semele_qa_studies",
        "theandrites_qa_studies",
        "zibelthiurdos_qa_studies",
        # Wave-1717 illyrian-myth canon.
        "bindus_qa_studies",
        "illyris_qa_studies",
        "medaurus_qa_studies",
        "redon_qa_studies",
        "thana_qa_studies",
        "vidasus_qa_studies",
        # Wave-1718 etruscan-myth canon.
        "fufluns_qa_studies",
        "menrva_qa_studies",
        "tinia_qa_studies",
        "turan_qa_studies",
        "veltha_qa_studies",
        "voltumna_qa_studies",
        # Wave-1719 basque-myth canon.
        "basajaun_qa_studies",
        "eguzki_qa_studies",
        "lamiak_qa_studies",
        "mairu_qa_studies",
        "mari_qa_studies",
        "sugaar_qa_studies",
        # Wave-1720 sami-myth canon.
        "akka_qa_studies",
        "juksakka_qa_studies",
        "lieaibolmmai_qa_studies",
        "radien_qa_studies",
        "sarakka_qa_studies",
        "ukso_qa_studies",
        # Wave-1721 tatar-myth canon.
        "albasti_qa_studies",
        "erlik_qa_studies",
        "shurale_qa_studies",
        "suana_qa_studies",
        "tengri_qa_studies",
        "umai_qa_studies",
        # Wave-1722 ossetian-myth canon.
        "barastir_qa_studies",
        "donbettyr_qa_studies",
        "nart_qa_studies",
        "safa_qa_studies",
        "styr_qa_studies",
        "tulur_qa_studies",
        # Wave-1723 norse-myth-5 canon.
        "honir_qa_studies",
        "kvasir_qa_studies",
        "lodurr_qa_studies",
        "mimir_qa_studies",
        "ve_qa_studies",
        "vili_qa_studies",
        # Wave-1724 polynesian-myth-2 canon.
        "kamapuaa_qa_studies",
        "kanaloa_qa_studies",
        "pele_qa_studies",
        "rongo_qa_studies",
        "tane_qa_studies",
        "tangaroa_qa_studies",
        # Wave-1725 japanese-myth-2 canon.
        "izanagi_qa_studies",
        "izanami_qa_studies",
        "kukunochi_qa_studies",
        "omoikane_qa_studies",
        "sarutahiko_qa_studies",
        "uzume_qa_studies",
        # Wave-1726 korean-myth canon.
        "dalnim_qa_studies",
        "dangun_qa_studies",
        "haenim_qa_studies",
        "hwanin_qa_studies",
        "hwanung_qa_studies",
        "samshin_qa_studies",
        # Wave-1727 mongolian-myth canon.
        "erlug_qa_studies",
        "etseg_qa_studies",
        "manzan_qa_studies",
        "otgon_qa_studies",
        "tenger_qa_studies",
        "ulgan_qa_studies",
        # Wave-1728 hungarian-myth canon.
        "boszorka_qa_studies",
        "csaba_qa_studies",
        "garabonci_qa_studies",
        "isten_qa_studies",
        "liderc_qa_studies",
        "taltos_qa_studies",
        # Wave-1729 baltic-myth canon.
        "austra_qa_studies",
        "jumis_qa_studies",
        "laima_qa_studies",
        "laume_qa_studies",
        "perkunas_qa_studies",
        "zemyna_qa_studies",
        # Wave-1730 turkic-myth canon.
        "aisit_qa_studies",
        "bayna_qa_studies",
        "kunkush_qa_studies",
        "payna_qa_studies",
        "taigan_qa_studies",
        "yalyk_qa_studies",
        # Wave-1731 taino-myth canon.
        "atabei_qa_studies",
        "boinayel_qa_studies",
        "deminan_qa_studies",
        "juracan_qa_studies",
        "karacarol_qa_studies",
        "yucahu_qa_studies",
        # Wave-1732 polish-myth canon.
        "dziewanna_qa_studies",
        "marzanna_qa_studies",
        "mokosz_qa_studies",
        "nija_qa_studies",
        "swarozyc_qa_studies",
        "zywie_qa_studies",
        # Wave-1733 irish-myth canon.
        "aengus_qa_studies",
        "brigid_qa_studies",
        "dagda_qa_studies",
        "lugh_qa_studies",
        "morrigan_qa_studies",
        "nuada_qa_studies",
        # Wave-1734 welsh-myth canon.
        "arawn_qa_studies",
        "ceridwen_qa_studies",
        "gwydion_qa_studies",
        "llew_qa_studies",
        "rhiannon_qa_studies",
        "taliesin_qa_studies",
        # Wave-1735 lithuanian-myth canon.
        "dievas_qa_studies",
        "gabija_qa_studies",
        "medeina_qa_studies",
        "ragana_qa_studies",
        "saulute_qa_studies",
        "velnias_qa_studies",
        # Wave-1736 finnish-myth-2 canon.
        "ilmarinen_qa_studies",
        "joukahainen_qa_studies",
        "lemminkainen_qa_studies",
        "marjatta_qa_studies",
        "tuoni_qa_studies",
        "vainamoinen_qa_studies",
        # Wave-1737 aztec-deity-3 canon.
        "coatlicue_qa_studies",
        "coyolxauhqui_qa_studies",
        "huitzilopochtli_qa_studies",
        "mictlantecuhtli_qa_studies",
        "tlaloc_qa_studies",
        "tonatiuh_qa_studies",
        # Wave-1738 sumerian-2 canon.
        "enlil_qa_studies",
        "ereshkigal_qa_studies",
        "nanna_qa_studies",
        "nergal_qa_studies",
        "ninhursag_qa_studies",
        "utu_qa_studies",
        # Wave-1739 persian-2 canon.
        "ahura_mazda_qa_studies",
        "anahita_qa_studies",
        "angra_mainyu_qa_studies",
        "mithra_qa_studies",
        "verethragna_qa_studies",
        "zahhak_qa_studies",
        # Wave-1740 canaanite-2 canon.
        "anat_qa_studies",
        "astarte_qa_studies",
        "kothar_qa_studies",
        "mot_qa_studies",
        "resheph_qa_studies",
        "yam_qa_studies",
        # Wave-1741 hawaiian-myth canon.
        "hina_qa_studies",
        "kanaloa_qa_studies",
        "kane_qa_studies",
        "ku_qa_studies",
        "lono_qa_studies",
        "pele_qa_studies",
        # Wave-1742 maori-myth canon.
        "haumia_qa_studies",
        "rongo_qa_studies",
        "tane_qa_studies",
        "tangaroa_qa_studies",
        "tawhirimatea_qa_studies",
        "tumatauenga_qa_studies",
        # Wave-1743 zulu-myth canon.
        "impundulu_qa_studies",
        "inkanyamba_qa_studies",
        "mamlambo_qa_studies",
        "tikoloshe_qa_studies",
        "unkulunkulu_qa_studies",
        "usilosimapundu_qa_studies",
        # Wave-1744 inuit-myth canon.
        "agloolik_qa_studies",
        "aumanil_qa_studies",
        "nuktessien_qa_studies",
        "sedna_qa_studies",
        "tekkeitsertok_qa_studies",
        "torngarsuk_qa_studies",
        # Wave-1745 aboriginal-myth canon.
        "altjira_qa_studies",
        "bunyip_qa_studies",
        "mimis_qa_studies",
        "rainbow_serpent_qa_studies",
        "wandjina_qa_studies",
        "yowie_qa_studies",
        # Wave-1746 sumerian-3 canon.
        "ishtar_qa_studies",
        "nanna_qa_studies",
        "nergal_qa_studies",
        "ninurta_qa_studies",
        "nisaba_qa_studies",
        "utu_qa_studies",
        # Wave-1747 norse-myth-6 canon.
        "bragi_qa_studies",
        "forseti_qa_studies",
        "idun_qa_studies",
        "sif_qa_studies",
        "ullr_qa_studies",
        "vidar_qa_studies",
        # Wave-1748 tibetan-myth canon.
        "beg_tse_qa_studies",
        "dorje_legpa_qa_studies",
        "palden_lhamo_qa_studies",
        "pehar_qa_studies",
        "tsen_god_qa_studies",
        "tsiu_marpo_qa_studies",
        # Wave-1749 slavic-myth-4 canon.
        "belobog_qa_studies",
        "chernobog_qa_studies",
        "dazhbog_qa_studies",
        "hors_qa_studies",
        "semargl_qa_studies",
        "stribog_qa_studies",
        # Wave-1750 babylonian-2 canon.
        "adad_qa_studies",
        "nabu_qa_studies",
        "ninlil_qa_studies",
        "shamash_qa_studies",
        "sin_qa_studies",
        "tiamat_qa_studies",
        # Wave-1751 egyptian-2 canon.
        "anubis_qa_studies",
        "bastet_qa_studies",
        "khonsu_qa_studies",
        "min_qa_studies",
        "neith_qa_studies",
        "sobek_qa_studies",
        # Wave-1752 greek-sea canon.
        "nereus_qa_studies",
        "phorcys_qa_studies",
        "pontus_qa_studies",
        "proteus_qa_studies",
        "thaumas_qa_studies",
        "triton_qa_studies",
        # Wave-1753 chinese-myth-4 canon.
        "changxi_qa_studies",
        "chiyou_qa_studies",
        "gonggong_qa_studies",
        "xihe_qa_studies",
        "yinglong_qa_studies",
        "zhurong_qa_studies",
        # Wave-1754 japanese-myth-3 canon.
        "fujin_qa_studies",
        "hachiman_qa_studies",
        "inari_qa_studies",
        "raijin_qa_studies",
        "sarutahiko_qa_studies",
        "uzume_qa_studies",
        # Wave-1755 norse-myth-7 canon.
        "dellingr_qa_studies",
        "gna_qa_studies",
        "jord_qa_studies",
        "mani_qa_studies",
        "sigyn_qa_studies",
        "sol_qa_studies",
        # Wave-1756 hindu-myth-5 canon.
        "apsara_qa_studies",
        "gandharva_qa_studies",
        "kinnara_qa_studies",
        "ratri_qa_studies",
        "rudra_qa_studies",
        "ushas_qa_studies",
        # Wave-1757 celtic-myth-2 canon.
        "aisling_qa_studies",
        "brigid_qa_studies",
        "dagda_qa_studies",
        "danu_qa_studies",
        "manannan_qa_studies",
        "morgen_qa_studies",
        # Wave-1758 african-myth-3 canon.
        "abiku_qa_studies",
        "anansi_qa_studies",
        "ifa_qa_studies",
        "obatala_qa_studies",
        "oya_qa_studies",
        "shango_qa_studies",
        # Wave-1759 aztec-deity-4 canon.
        "citlali_qa_studies",
        "malinal_qa_studies",
        "metzli_qa_studies",
        "tepoz_qa_studies",
        "tonaca_qa_studies",
        "xochipilli_qa_studies",
        # Wave-1760 japanese-myth-4 canon.
        "ebisu_qa_studies",
        "hiruko_qa_studies",
        "kikuzuki_qa_studies",
        "kisshoten_qa_studies",
        "morinaga_qa_studies",
        "senju_qa_studies",
        # Wave-1761 sumerian-4 canon.
        "enki_qa_studies",
        "enlil_qa_studies",
        "inanna_qa_studies",
        "nanna_qa_studies",
        "ninhursag_qa_studies",
        "utu_qa_studies",
        # Wave-1762 norse-myth-8 canon.
        "baldr_qa_studies",
        "bragi_qa_studies",
        "freya_qa_studies",
        "heimdall_qa_studies",
        "idunn_qa_studies",
        "tyr_qa_studies",
        # Wave-1763 egyptian-3 canon.
        "hapi_qa_studies",
        "khnum_qa_studies",
        "menhit_qa_studies",
        "nephthys_qa_studies",
        "serqet_qa_studies",
        "tefnut_qa_studies",
        # Wave-1764 greek-myth-6 canon.
        "eileithyia_qa_studies",
        "iris_qa_studies",
        "leto_qa_studies",
        "nemesis_qa_studies",
        "nike_qa_studies",
        "tyche_qa_studies",
        # Wave-1765 celtic-myth-3 canon.
        "arianrhod_qa_studies",
        "cerridwen_qa_studies",
        "lugh_qa_studies",
        "morrigan_qa_studies",
        "nuada_qa_studies",
        "rhiannon_qa_studies",
        # Wave-1766 mayan-myth-2 canon.
        "cabrakan_qa_studies",
        "camazotz_qa_studies",
        "hunab_qa_studies",
        "itzamna_qa_studies",
        "ixmucane_qa_studies",
        "zipacna_qa_studies",
        # Wave-1767 incan-myth-2 canon.
        "coniraya_qa_studies",
        "guanare_qa_studies",
        "inti_qa_studies",
        "pachacamac_qa_studies",
        "supay_qa_studies",
        "viracocha_qa_studies",
        # Wave-1768 african-myth-4 canon.
        "buluku_qa_studies",
        "chukwu_qa_studies",
        "eshu_qa_studies",
        "mawu_qa_studies",
        "nyambi_qa_studies",
        "oshumare_qa_studies",
        # Wave-1769 persian-3 canon.
        "angra_qa_studies",
        "arash_qa_studies",
        "haoma_qa_studies",
        "simurgh_qa_studies",
        "spenta_qa_studies",
        "zal_qa_studies",
        # Wave-1770 assyrian-myth canon.
        "enkidu_qa_studies",
        "etana_qa_studies",
        "gilgamesh_qa_studies",
        "kingu_qa_studies",
        "nabu_qa_studies",
        "tiamat_qa_studies",
        # Wave-1771 hittite-2 canon.
        "hannahanna_qa_studies",
        "illuyanka_qa_studies",
        "inara_qa_studies",
        "kamrusepa_qa_studies",
        "tarhunna_qa_studies",
        "telepinus_qa_studies",
        # Wave-1772 greek-myth-7 canon.
        "hecate_qa_studies",
        "helios_qa_studies",
        "hypnos_qa_studies",
        "selene_qa_studies",
        "thanatos_qa_studies",
        "zephyrus_qa_studies",
        # Wave-1773 norse-myth-9 canon.
        "balder_qa_studies",
        "frigg_qa_studies",
        "loki_qa_studies",
        "sif_qa_studies",
        "vali_qa_studies",
        "vitharr_qa_studies",
        # Wave-1774 egyptian-4 canon.
        "bastet_qa_studies",
        "hathor_qa_studies",
        "nut_qa_studies",
        "sekhmet_qa_studies",
        "sobek_qa_studies",
        "thoth_qa_studies",
        # Wave-1775 japanese-myth-5 canon.
        "inari_qa_studies",
        "kaguya_qa_studies",
        "momotaro_qa_studies",
        "shichifukujin_qa_studies",
        "takemikazuchi_qa_studies",
        "urashima_qa_studies",
        # Wave-1776 korean-myth-2 canon.
        "bari_qa_studies",
        "kongjwi_qa_studies",
        "ondal_qa_studies",
        "pyonggang_qa_studies",
        "samshin_qa_studies",
        "shimchong_qa_studies",
        # Wave-1777 indonesian-myth canon.
        "barong_qa_studies",
        "garuda_qa_studies",
        "nyai_qa_studies",
        "raksasa_qa_studies",
        "rangda_qa_studies",
        "semar_qa_studies",
        # Wave-1778 hindu-myth-6 canon.
        "hanuman_qa_studies",
        "lakshmi_qa_studies",
        "parvati_qa_studies",
        "rama_qa_studies",
        "sita_qa_studies",
        "vishnu_qa_studies",
        # Wave-1779 chinese-myth-5 canon.
        "fuxi_qa_studies",
        "huangdi_qa_studies",
        "nuwa_qa_studies",
        "shennong_qa_studies",
        "xihe_qa_studies",
        "yandi_qa_studies",
        # Wave-1780 welsh-myth-2 canon.
        "arawn_qa_studies",
        "branwen_qa_studies",
        "gwydion_qa_studies",
        "lleu_qa_studies",
        "lludd_qa_studies",
        "taliesin_qa_studies",
        # Wave-1781 greek-myth-8 canon.
        "apollo_qa_studies",
        "artemis_qa_studies",
        "athena_qa_studies",
        "demeter_qa_studies",
        "hera_qa_studies",
        "persephone_qa_studies",
        # Wave-1782 egyptian-5 canon.
        "geb_qa_studies",
        "horus_qa_studies",
        "isis_qa_studies",
        "osiris_qa_studies",
        "set_qa_studies",
        "shu_qa_studies",
        # Wave-1783 norse-myth-10 canon.
        "baldur_qa_studies",
        "freyr_qa_studies",
        "hermodr_qa_studies",
        "hodr_qa_studies",
        "njord_qa_studies",
        "skadi_qa_studies",
        # Wave-1784 greek-myth-9 canon.
        "ares_qa_studies",
        "hades_qa_studies",
        "hephaestus_qa_studies",
        "hestia_qa_studies",
        "poseidon_qa_studies",
        "zeus_qa_studies",
        # Wave-1785 japanese-myth-6 canon.
        "benzaiten_qa_studies",
        "hoori_qa_studies",
        "jurojin_qa_studies",
        "kushinadahime_qa_studies",
        "toyotamahime_qa_studies",
        "yamatotakeru_qa_studies",
        # Wave-1786 mesopotamian-2 canon.
        "ea_qa_studies",
        "humbaba_qa_studies",
        "pazuzu_qa_studies",
        "sargon_qa_studies",
        "semiramis_qa_studies",
        "utnapishtim_qa_studies",
        # Wave-1787 norse-myth-11 canon.
        "eir_qa_studies",
        "heimdal_qa_studies",
        "norns_qa_studies",
        "odin_qa_studies",
        "thor_qa_studies",
        "valkyrie_qa_studies",
        # Wave-1788 roman-rural canon.
        "ceres_qa_studies",
        "flora_qa_studies",
        "janus_qa_studies",
        "pomona_qa_studies",
        "silvanus_qa_studies",
        "solinvictus_qa_studies",
        # Wave-1789 greek-minor canon.
        "eris_qa_studies",
        "ganymede_qa_studies",
        "hebe_qa_studies",
        "hermes_qa_studies",
        "momus_qa_studies",
        "oneiros_qa_studies",
        # Wave-1790 egyptian-6 canon.
        "amun_qa_studies",
        "atum_qa_studies",
        "khepri_qa_studies",
        "mut_qa_studies",
        "ptah_qa_studies",
        "seth_qa_studies",
        # Wave-1791 norse-myth-12 canon.
        "frey_qa_studies",
        "freya2_qa_studies",
        "magni_qa_studies",
        "modi_qa_studies",
        "njord_qa_studies",
        "tyr2_qa_studies",
        # Wave-1792 sumerian-5 canon.
        "agga_qa_studies",
        "babbar_qa_studies",
        "enmerkar2_qa_studies",
        "lugulbanda_qa_studies",
        "ninsun_qa_studies",
        "urukagina_qa_studies",
        # Wave-1793 roman-minor-2 canon.
        "cluentia_qa_studies",
        "faunus_qa_studies",
        "larunda_qa_studies",
        "mutina_qa_studies",
        "quirinus_qa_studies",
        "tellus_qa_studies",
        # Wave-1794 maori-2 canon.
        "maru2_qa_studies",
        "pere_qa_studies",
        "rongomai_qa_studies",
        "tuhi2_qa_studies",
        "uenuku2_qa_studies",
        "wairere_qa_studies",
        # Wave-1795 egyptian-7 canon.
        "anubis2_qa_studies",
        "isis2_qa_studies",
        "khonsu2_qa_studies",
        "osiris2_qa_studies",
        "ra2_qa_studies",
        "sobek2_qa_studies",
        # Wave-1399 retrieval-eval canon.
        "asqa_lite_studies",
        "eli5_lite_studies",
        "fresh_qa_studies",
        "nq_lite_studies",
        "trivia_lite_studies",
        "xor_tydi_studies",
        # Wave-1398 table-QA canon.
        "doc2dial_studies",
        "finqa_lite_studies",
        "hybridqa_lite_studies",
        "infotabs_studies",
        "ottqa_lite_studies",
        "tab_cwq_studies",
        # Wave-1397 multilingual-QA canon.
        "coma_qa_studies",
        "gaia_lite_studies",
        "simple_qa_studies",
        "sqa_lite_studies",
        "tqa_lite_studies",
        "tydiqa_lite_studies",
        # Wave-1396 summarization-2 canon.
        "agnews_lite_studies",
        "dialsum_lite_studies",
        "facet_lite_studies",
        "medsum_lite_studies",
        "oposum_lite_studies",
        "qsum_lite_studies",
        # Wave-1395 QA-exotics-3 canon.
        "bamboogle_studies",
        "fine_qa_studies",
        "hotpot2_studies",
        "kwik_qa_studies",
        "quest_qa_studies",
        "tatqa2_studies",
        # Wave-1394 code-agent canon.
        "api_eval_studies",
        "apps_lite_studies",
        "livecode_studies",
        "mbpp_lite_studies",
        "restbench_studies",
        "swe_gym_studies",
        # Wave-1393 MCP-web canon.
        "hamming_mcp_studies",
        "mcp_bench_studies",
        "net_hack_studies",
        "tool_sandbox_studies",
        "videoweb_studies",
        "webshop_lite_studies",
        # Wave-1392 toolbench canon.
        "meta_tool_studies",
        "nest_tools_studies",
        "toolbench2_studies",
        "toolqa_lite_studies",
        "ultra_tool_studies",
        "work_plus_studies",
        # Wave-1391 QA-exotics-2 canon.
        "archer_qa_studies",
        "argue_eval_studies",
        "expert_qa_studies",
        "mintaka_lite_studies",
        "musique_lite_studies",
        "wiki2_qa_studies",
        # Wave-1390 multi-doc-sum canon.
        "episum_lite_studies",
        "fsum_lite_studies",
        "mds_news_studies",
        "sqcs_lite_studies",
        "summon_fce_studies",
        "wcep_lite_studies",
        # Wave-1389 long-doc-sum canon.
        "book_sum_studies",
        "fanout_qa_studies",
        "infinitesum_studies",
        "marlense_studies",
        "narra_sum_studies",
        "quote_sum_studies",
        # Wave-1388 RAG-eval canon.
        "corpus_qa_studies",
        "crag_bench_studies",
        "domain_rag_studies",
        "freshqa_studies",
        "ragas_lite_studies",
        "rgb_eval_studies",
        # Wave-1387 tool-use canon.
        "api_blend_studies",
        "bfcl_v3_studies",
        "gorilla_eval_studies",
        "gta_bench_studies",
        "seal_tools_studies",
        "stabletoolbench_studies",
        # Wave-1386 embodied-game canon.
        "alfworld_lite_studies",
        "babyai_lite_studies",
        "crafter_lite_studies",
        "jericho_lite_studies",
        "scienceworld_studies",
        "textworld_lite_studies",
        # Wave-1385 web-agent canon.
        "airtasks_studies",
        "browsergym_studies",
        "maze_eval_studies",
        "mmind2web_studies",
        "screenqa_studies",
        "weblinx_studies",
        # Wave-1384 live-eval canon.
        "aider_polyglot_studies",
        "hum_eval_studies",
        "livebench_arena_studies",
        "mbti_eval_studies",
        "olmes_lite_studies",
        "plus_eval_studies",
        # Wave-1383 frontier-eval canon.
        "aime24_studies",
        "gpqa_diamond_studies",
        "hle_lite_studies",
        "mmmlu_lite_studies",
        "olympiadbench_studies",
        "super_gpqa_studies",
        # Wave-1382 LLM-eval-2 canon.
        "fact_score_studies",
        "gpt_score_studies",
        "helm_lite_studies",
        "lmsys_eval_studies",
        "nugget_eval_studies",
        "vicuna_bench_studies",
        # Wave-1381 dialogue-2 canon.
        "begins_lite_studies",
        "diamonds_lite_studies",
        "faithful_dial_studies",
        "multi_woz_studies",
        "top_dialog_studies",
        "wow_lite_studies",
        # Wave-1380 intent-paraphrase canon.
        "art_nli_studies",
        "para_paws_studies",
        "recast_lite_studies",
        "snips_lite_studies",
        "social_lite_studies",
        "subj_lite_studies",
        # Wave-1379 metric-exotics canon.
        "bary_score_studies",
        "cider_lite_studies",
        "gleu_lite_studies",
        "kl_div_eval_studies",
        "rouge_we_studies",
        "wmt_metric_studies",
        # Wave-1378 numerical-reasoning canon.
        "aqua_lite_studies",
        "fin_qa_studies",
        "math_qa_studies",
        "num_glue_studies",
        "tab_fact_studies",
        "tat_qa_studies",
        # Wave-1377 NLU-exotics canon.
        "abduction_lite_studies",
        "board_game_qa_studies",
        "conv_finqa_studies",
        "dream_lite_studies",
        "equiv_lite_studies",
        "wsc_lite_studies",
        # Wave-1376 dialogue-system canon.
        "blender_bot_studies",
        "conv_ai2_studies",
        "daily_dialog_studies",
        "dstc_lite_studies",
        "empathy_dialog_studies",
        "persona_chat_studies",
        # Wave-1375 scientific-summarization canon.
        "facet_sum_studies",
        "ms2_lite_studies",
        "patent_sum_studies",
        "sci_lay_studies",
        "scitldr_lite_studies",
        "spectrum_sum_studies",
        # Wave-1374 proof-entailment canon.
        "deduc_lite_studies",
        "entail_bank_studies",
        "folio_lite_studies",
        "logic_nli_studies",
        "proof_writer_studies",
        "rule_taker_studies",
        # Wave-1373 commonsense-reasoning canon.
        "commonsense_lite_studies",
        "logi_qa_studies",
        "mr_lite_studies",
        "muin_lite_studies",
        "qasc_sci2_studies",
        "winogrande_lite_studies",
        # Wave-1372 long-doc-summarization canon.
        "billsum_lite_studies",
        "booksum_lite_studies",
        "elm_lite_studies",
        "govreport_lite_studies",
        "qmsum_lite_studies",
        "wikisum_lite_studies",
        # Wave-1371 social-reasoning canon.
        "ethos_lite_studies",
        "moral_stories_studies",
        "mutual_lite_studies",
        "prosocial_lite_studies",
        "scruples_studies",
        "siqa_lite_studies",
        # Wave-1370 challenge-benchmark canon.
        "bbh_lite_studies",
        "gpqa_lite_studies",
        "if_eval_studies",
        "live_bench_studies",
        "olympic_bench_studies",
        "trivia_qa_lite_studies",
        # Wave-1369 math-word-problem canon.
        "aime_eval_studies",
        "asdiv_lite_studies",
        "math500_lite_studies",
        "mgsm_lite_studies",
        "minerva_math_studies",
        "svamp_lite_studies",
        # Wave-1368 translation-metric canon.
        "chr_f_studies",
        "mover_score_studies",
        "nist_metric_studies",
        "prism_mt_studies",
        "sacrebleu_lite_studies",
        "ter_lite_studies",
        # Wave-1367 generation-metric canon.
        "bert_score_studies",
        "bleu_rouge_studies",
        "bleurt_lite_studies",
        "comet_mt_studies",
        "meteor_lite_studies",
        "rouge_lite_studies",
        # Wave-1366 faithfulness-eval canon.
        "align_score_studies",
        "dice_eval_studies",
        "factcc_lite_studies",
        "faith_eval_studies",
        "quest_eval_studies",
        "summa_eval_studies",
        # Wave-1365 summarization canon.
        "arxiv_sum_studies",
        "cnn_dailymail_studies",
        "dialogsum_lite_studies",
        "multi_news_studies",
        "pubmed_sum_studies",
        "samsum_lite_studies",
        # Wave-1364 sentence-pair canon.
        "anli_lite_studies",
        "mnli_lite_studies",
        "mrpc_lite_studies",
        "paws_lite_studies",
        "quora_dup_studies",
        "rte_lite_studies",
        # Wave-1363 rumor-bias canon.
        "age_bias_studies",
        "curry_qa_studies",
        "cw_qa2_studies",
        "dialect_bias_studies",
        "politi_fact_studies",
        "rumor_twitter_studies",
        # Wave-1362 fake-news canon.
        "check_that_studies",
        "claim_buster_studies",
        "emergent_lite_studies",
        "fake_news_studies",
        "snopes_lite_studies",
        "stance_detect_studies",
        # Wave-1361 misinformation canon.
        "covid_lies_studies",
        "evidence_inf_studies",
        "hoax_detect_studies",
        "liar_lite_studies",
        "rumor_eval_studies",
        "scidtb_lite_studies",
        # Wave-1360 fact-check canon.
        "bioasq_lite_studies",
        "cite_worth_studies",
        "climate_fever_studies",
        "fever_lite_studies",
        "touch_e_studies",
        "verdict_qa_studies",
        # Wave-1359 QA-exotics canon.
        "argu_ana_studies",
        "babi_lite_studies",
        "curious_qa_studies",
        "qasper_lite_studies",
        "scifact_lite_studies",
        "web_questions_studies",
        # Wave-1358 reading-comp-4 canon.
        "adver_qa_studies",
        "coqa_lite_studies",
        "drop_lite_studies",
        "duo_rc_studies",
        "quac_lite_studies",
        "trivia_web_studies",
        # Wave-1357 reading-comp-3 canon.
        "hendrycks_test_studies",
        "hotpot_lite_studies",
        "multirc_lite_studies",
        "quoref_lite_studies",
        "record_lite_studies",
        "squad_lite2_studies",
        # Wave-1356 knowledge-QA canon.
        "bigbench_lite_studies",
        "entity_qa_studies",
        "mmlu_lite_studies",
        "natural_qa_studies",
        "pop_qa_studies",
        "triviaqa_lite_studies",
        # Wave-1355 commonsense-eval canon.
        "arc_hard2_studies",
        "csqa_lite_studies",
        "hellaswag_lite_studies",
        "piqa_lite_studies",
        "prost_lite_studies",
        "swag_lite_studies",
        # Wave-1354 MC-eval canon.
        "arc_easy2_studies",
        "boolq_lite_studies",
        "cosmos_qa_studies",
        "race_lite_studies",
        "sciq_lite_studies",
        "social_qa_studies",
        # Wave-1353 NLI-eval-2 canon.
        "creak_lite_studies",
        "entailment_bn_studies",
        "hans_lite_studies",
        "prove_it_studies",
        "strategy_qa_studies",
        "sup_nli_studies",
        # Wave-1352 social-bias-eval canon.
        "bias_bench_studies",
        "crowsp_lite_studies",
        "honesty_lie_studies",
        "social_iqa2_studies",
        "stereo_lite_studies",
        "wino_bias_studies",
        # Wave-1351 ethics-eval canon.
        "ethic_jiminy_studies",
        "moral_exc_studies",
        "moral_found_studies",
        "principlism_toy_studies",
        "scruples_lite_studies",
        "virtue_ethics_studies",
        # Wave-1350 compositional-generalization canon.
        "dyck_lang_studies",
        "hops_add_studies",
        "lcmc_lite_studies",
        "mco_lite_studies",
        "scan_cfsp_studies",
        "shuffle_expr_studies",
        # Wave-1349 GLUE-eval-2 canon.
        "cola_lite_studies",
        "qnli_lite_studies",
        "qqp_lite_studies",
        "sst2_lite_studies",
        "stsb_lite_studies",
        "wnli_lite_studies",
        # Wave-1348 NLI-eval canon.
        "anli_r1_studies",
        "anli_r2_studies",
        "anli_r3_studies",
        "mnli_match_studies",
        "scitail_lite_studies",
        "snli_lite_studies",
        # Wave-1347 logical-reasoning-eval canon.
        "abductive_nli_studies",
        "conseq_log_studies",
        "logiqa_log_studies",
        "lsat_log_studies",
        "reason_mc_studies",
        "recli_log_studies",
        # Wave-1346 KB-QA canon.
        "grail_qa_studies",
        "graph_questions_studies",
        "kqa_pro_studies",
        "lc_quad_studies",
        "mintaka_qa_studies",
        "spinach_qa_studies",
        # Wave-1345 open-domain-QA canon.
        "complex_qa_studies",
        "entity_quests_studies",
        "freebase_qa_studies",
        "nq_open_studies",
        "trivia_qa_studies",
        "web_qa_studies",
        # Wave-1344 reading-comprehension-3 canon.
        "boolq_qa_studies",
        "dream_qa_studies",
        "duorc_qa_studies",
        "mctest_qa_studies",
        "qasper_qa_studies",
        "race_qa_studies",
        # Wave-1343 reading-comprehension-2 canon.
        "coqa_qa_studies",
        "drop_qa_studies",
        "news_qa_studies",
        "quac_qa_studies",
        "quail_qa_studies",
        "quoref_qa_studies",
        # Wave-1342 bias-eval-2 canon.
        "fairness_eval_studies",
        "gender_bias_studies",
        "jigsaw_tox_studies",
        "nlp_bias_studies",
        "pronoun_bias_studies",
        "regard_metric_studies",
        # Wave-1341 bias-eval canon.
        "bold_eval_studies",
        "crow_s_pairs_studies",
        "hate_speech_eval_studies",
        "holo_bias_studies",
        "real_toxicity_studies",
        "stereo_set_studies",
        # Wave-1340 science-eval canon.
        "arc_challenge_studies",
        "bio_qa_studies",
        "med_qa_studies",
        "openbook_qa_studies",
        "pubmed_qa_studies",
        "sci_q_studies",
        # Wave-1339 math-eval-2 canon.
        "aqua_rat_studies",
        "geo_qa_studies",
        "hol_step_studies",
        "math_odyssey_studies",
        "tab_math_studies",
        "uni_math_studies",
        # Wave-1338 math-reasoning-eval canon.
        "arith_qa_studies",
        "gsm_hard_studies",
        "math_reason_studies",
        "mini_f2f_studies",
        "proof_pile_studies",
        "theorem_qa_studies",
        # Wave-1337 agentic-eval-3 canon.
        "android_env_studies",
        "api_bank_studies",
        "gaia_level_studies",
        "video_game_studies",
        "voyager_minecraft_studies",
        "web_shopping_studies",
        # Wave-1336 agentic-eval-2 canon.
        "assistantbench_studies",
        "mind2web_studies",
        "miniwob_studies",
        "visual_web_studies",
        "web_nav_studies",
        "webarena_studies",
        # Wave-1335 code-eval-4 canon.
        "code_rag_studies",
        "codegen_universal_studies",
        "long_code_bench_studies",
        "odex_eval_studies",
        "swe_dev_studies",
        "swe_multimodal_studies",
        # Wave-1334 long-context-3 canon.
        "books_qa_studies",
        "lcc_codebase_studies",
        "multi_news_eval_studies",
        "narrative_qa_studies",
        "needle_multi_studies",
        "qmsum_eval_studies",
        # Wave-1333 multilingual-eval canon.
        "fava_studies",
        "polyglo_tox_studies",
        "regard_eval_studies",
        "unqover_studies",
        "vlur_studies",
        "xlsum_studies",
        # Wave-1332 eval-tooling canon.
        "decontaminate_studies",
        "eval_bias_studies",
        "fair_eval_studies",
        "g_eval_studies",
        "ngram_overlap_studies",
        "pandalm_studies",
        # Wave-1331 code-eval-3 canon.
        "codescope_studies",
        "concode_eval_studies",
        "crosscodeeval_studies",
        "mer_bench_studies",
        "project_eval_studies",
        "swe_bench_verified_studies",
        # Wave-1330 safety-alignment-2 canon.
        "beaver_safe_studies",
        "do_not_answer_studies",
        "hh_rlhf_studies",
        "honest_eval_studies",
        "safe_rlhf_studies",
        "sos_bench_studies",
        # Wave-1329 long-context-2 canon.
        "gov_report_studies",
        "looogle_studies",
        "lost_middle_studies",
        "marathon_eval_studies",
        "niah_v2_studies",
        "passkey_retrieval_studies",
        # Wave-1328 code-eval-2 canon.
        "apps_bench_studies",
        "class_eval_studies",
        "code_contests_studies",
        "multipl_e_studies",
        "polyglot_bench_studies",
        "repobench_studies",
        # Wave-1327 safety-bias-eval canon.
        "bbq_bias_studies",
        "bold_bias_studies",
        "crowspairs_studies",
        "holist_bias_studies",
        "realtoxicity_studies",
        "toxigen_eval_studies",
        # Wave-1326 privacy-inference-2 canon.
        "attribute_inference_studies",
        "canary_memorization_studies",
        "extraction_attack_studies",
        "membership_inference_studies",
        "model_inversion_studies",
        "privacy_meter_studies",
        # Wave-1325 long-context-eval canon.
        "babilong_studies",
        "infinitebench_studies",
        "longbench_studies",
        "lv_eval_studies",
        "ruler_bench_studies",
        "zero_scrolls_studies",
        # Wave-1324 judge-eval canon.
        "alpacaeval_studies",
        "arena_hard_studies",
        "judge_bench_studies",
        "mt_bench_judge_studies",
        "prometheus_eval_studies",
        "reward_bench_studies",
        # Wave-1323 code-eval canon.
        "bigcodebench_studies",
        "ds1000_studies",
        "humaneval_plus_studies",
        "livecodebench_studies",
        "mbpp_plus_studies",
        "swe_perf_studies",
        # Wave-1322 reasoning-eval canon.
        "logic_bench_studies",
        "minif2f_studies",
        "olympiad_bench_studies",
        "putnam_studies",
        "truthfulqa_studies",
        "zebra_logic_studies",
        # Wave-1321 multimodal-eval canon.
        "chart_gqa_studies",
        "mathvista_studies",
        "mkqa_studies",
        "mmmlu_studies",
        "mmmu_studies",
        "videomme_studies",
        # Wave-1320 agent-eval canon.
        "gaia_bench_studies",
        "mmbench_agent_studies",
        "osworld_studies",
        "screen_eval_studies",
        "vsi_bench_studies",
        "webvoyager_studies",
        # Wave-1319 benchmark-eval canon.
        "math_bench_studies",
        "multirc_studies",
        "ninco_studies",
        "objectnet_studies",
        "ood_bench_studies",
        "wild_bench_studies",
        # Wave-1318 NLP-eval-2 canon.
        "cb_studies",
        "cola_studies",
        "qqp_studies",
        "squad_v2_studies",
        "sst2_studies",
        "wic_studies",
        # Wave-1317 eval-science-2 canon.
        "arc_eval_studies",
        "do_anything_studies",
        "step_eval_studies",
        "strong_reject_studies",
        "verifier_reward_studies",
        "winogrande_studies",
        # Wave-1316 cue-conflict canon.
        "backgrounds_studies",
        "cue_conflict_studies",
        "geirhos_studies",
        "imagenet_bg_studies",
        "shape_bias_studies",
        "texture_bias_studies",
        # Wave-1315 OOD-robustness canon.
        "imagenet_a_studies",
        "imagenet_e_studies",
        "imagenet_o_studies",
        "imagenet_sketch_studies",
        "imagenet_v2_studies",
        "stylized_studies",
        # Wave-1314 privacy-inference canon.
        "attribute_infer_studies",
        "extraction_studies",
        "inversion_studies",
        "model_stealing_studies",
        "property_infer_studies",
        "reconstruction_studies",
        # Wave-1313 privacy-attack canon.
        "abs_scan_studies",
        "activation_cluster_studies",
        "fine_pruning_studies",
        "sleepless_studies",
        "strip_defense_studies",
        "watermark_studies",
        # Wave-1312 risk-domain canon.
        "bio_risk_eval_studies",
        "chem_risk_eval_studies",
        "cyber_sec_eval_studies",
        "lab_bench_studies",
        "malicious_instruct_studies",
        "wmdp_studies",
        # Wave-1311 safety-benchmark canon.
        "aegis_studies",
        "air_bench_studies",
        "overkill_studies",
        "salad_bench_studies",
        "sorry_bench_studies",
        "wildguard_studies",
        # Wave-1310 generation-quality canon.
        "alpaca_eval_studies",
        "attribution_eval_studies",
        "citation_eval_studies",
        "diversity_eval_studies",
        "factscore_studies",
        "self_bleu_studies",
        # Wave-1309 hard-benchmark canon.
        "frontier_math_studies",
        "gpqa_studies",
        "hle_studies",
        "mmlu_pro_studies",
        "tau_bench_studies",
        "workarena_studies",
        # Wave-1308 long-context-factuality canon.
        "halu_eval_studies",
        "infinite_bench_studies",
        "longmem_studies",
        "needle_haystack_studies",
        "ruler_studies",
        "truthful_qa_studies",
        # Wave-1307 winograd-eval canon.
        "lambada_studies",
        "record_studies",
        "story_cloze_studies",
        "winogender_studies",
        "winograd_studies",
        "wsc_studies",
        # Wave-1306 GLUE-eval canon.
        "glue_studies",
        "mnli_studies",
        "qnli_studies",
        "rte_studies",
        "super_glue_studies",
        "wnli_studies",
        # Wave-1305 reading-comprehension canon.
        "coqa_studies",
        "drop_studies",
        "hotpotqa_studies",
        "nq_studies",
        "squad_studies",
        "triviaqa_studies",
        # Wave-1304 commonsense-eval canon.
        "boolq_studies",
        "copa_studies",
        "hellaswag_studies",
        "openbookqa_studies",
        "piqa_studies",
        "siqa_studies",
        # Wave-1303 LLM-academic-eval canon.
        "bbh_studies",
        "gsm8k_studies",
        "humaneval_studies",
        "ifeval_studies",
        "mmlu_studies",
        "mt_bench_studies",
        # Wave-1302 privacy-inference canon.
        "canary_infer_studies",
        "deep_leak_studies",
        "gradient_leak_studies",
        "lira_studies",
        "membership_infer_studies",
        "shadow_model_studies",
        # Wave-1301 backdoor-eval canon.
        "backdoor_studies",
        "clean_label_studies",
        "data_poison_studies",
        "neural_cleanse_studies",
        "spectral_signature_studies",
        "trojan_studies",
        # Wave-1300 robustness-eval canon.
        "adversarial_eval_studies",
        "autoattack_studies",
        "corruption_studies",
        "imagenet_c_studies",
        "imagenet_r_studies",
        "robust_bench_studies",
        # Wave-1299 safety-eval canon.
        "agent_harm_studies",
        "harm_bench_studies",
        "jailbreak_bench_studies",
        "prompt_inject_studies",
        "safety_bench_studies",
        "xstest_studies",
        # Wave-1298 agentic-eval canon.
        "browse_eval_studies",
        "os_world_studies",
        "swe_bench_studies",
        "terminal_bench_studies",
        "tool_use_eval_studies",
        "web_arena_studies",
        # Wave-1297 eval-science canon.
        "benchmark_gaming_studies",
        "benchmark_saturate_studies",
        "contamination_studies",
        "eval_coverage_studies",
        "eval_reliability_studies",
        "lm_eval_harness_studies",
        # Wave-1296 reward-modeling-2 canon.
        "ensemble_rm_studies",
        "judge_reward_studies",
        "margin_reward_studies",
        "reward_hacking_studies",
        "reward_uncertainty_studies",
        "rm_btd_studies",
        # Wave-1295 mech-anomaly/jailbreak canon.
        "activation_patch_studies",
        "circuit_tracer_studies",
        "feature_dashboard_studies",
        "jailbreak_detect_studies",
        "mech_anomaly_studies",
        "sae_linter_studies",
        # Wave-1294 representation-engineering canon.
        "activation_oracle_studies",
        "concept_vector_studies",
        "feature_ablation_studies",
        "honesty_vector_studies",
        "reading_vector_studies",
        "refusal_vector_studies",
        # Wave-1293 constitutional-AI canon.
        "cai_critique_studies",
        "constitutional_studies",
        "harmlessness_rl_studies",
        "principle_eval_studies",
        "rlaif_studies",
        "sleeper_eval_studies",
        # Wave-1292 RLVR/verifiable-rewards canon.
        "grpo_studies",
        "math_reward_studies",
        "outcome_reward_studies",
        "process_reward_studies",
        "rlvr_studies",
        "verifiable_reward_studies",
        # Wave-1291 data-filtering/dedup canon.
        "data_mix_studies",
        "dedup_minhash_studies",
        "dedup_studies",
        "domain_classifier_studies",
        "perplexity_filter_studies",
        "quality_filter_studies",
        # Wave-1290 quantization/compression canon.
        "awq_studies",
        "entropy_code_quant_studies",
        "gptq_studies",
        "kv_cache_quant_studies",
        "smoothquant_studies",
        "weight_share_studies",
        # Wave-1289 reasoning/CoT canon.
        "analogical_prompt_studies",
        "cot_studies",
        "reflexion_studies",
        "scratchpad_studies",
        "self_consistency_studies",
        "stepwise_verify_studies",
        # Wave-1288 RL-imitation canon.
        "adversarial_irl_studies",
        "behavior_cloning_studies",
        "dagger_studies",
        "offline_distill_studies",
        "preference_irl_studies",
        "skill_extraction_studies",
        # Wave-1287 agent-safety canon.
        "capability_eval_studies",
        "control_eval_studies",
        "deception_eval_studies",
        "prompt_injection_studies",
        "sandbox_escape_studies",
        "tool_call_verify_studies",
        # Wave-1286 long-context canon.
        "beacon_context_studies",
        "hierarchical_context_studies",
        "infini_attention_studies",
        "landmark_attention_studies",
        "ntk_scaling_studies",
        "yarn_scaling_studies",
        # Wave-1285 multimodal-2 canon.
        "audio_lm_studies",
        "chart_reasoning_studies",
        "doc_vqa_studies",
        "gui_agent_studies",
        "video_understanding_studies",
        "vision_pretraining_studies",
        # Wave-1284 grounding/hallucination canon.
        "citation_check_studies",
        "claim_verifier_studies",
        "entailment_studies",
        "factuality_score_studies",
        "grounding_verify_studies",
        "self_reflect_studies",
        # Wave-1283 agent-memory canon.
        "context_compression_studies",
        "episodic_memory_studies",
        "memory_bank_studies",
        "retrieval_memory_studies",
        "semantic_memory_studies",
        "working_memory_studies",
        # Wave-1282 embodied-VLA canon.
        "affordance_map_studies",
        "embodied_agent_studies",
        "spatial_reasoning_studies",
        "video_diffusion_studies",
        "vla_model_studies",
        "world_sim_studies",
        # Wave-1281 reasoning-prompt canon.
        "analogical_prompting_studies",
        "graph_of_thought_studies",
        "least_to_most_studies",
        "plan_and_solve_studies",
        "step_back_studies",
        "tree_of_thought_studies",
        # Wave-1280 pretraining-data canon.
        "data_mixture_studies",
        "data_quality_studies",
        "dedup_pipeline_studies",
        "domain_filtering_studies",
        "synthetic_data_studies",
        "token_budget_studies",
        # Wave-1279 distributed-training canon.
        "activation_checkpoint_studies",
        "fsdp_sharding_studies",
        "hybrid_parallel_studies",
        "pipeline_schedule_studies",
        "sequence_parallel_studies",
        "zero_optimizer_studies",
        # Wave-1278 LLM-serving canon.
        "chunked_prefill_studies",
        "continuous_batching_studies",
        "disaggregated_serving_studies",
        "early_exit_studies",
        "prefix_caching_studies",
        "tensor_parallel_studies",
        # Wave-1277 scalable-oversight canon.
        "debate_alignment_studies",
        "deliberative_alignment_studies",
        "iterated_amplification_studies",
        "recursive_reward_studies",
        "scalable_oversight_studies",
        "weak_to_strong_studies",
        # Wave-1276 interpretability-3 canon.
        "attribution_patching_studies",
        "causal_scrubbing_studies",
        "function_vector_studies",
        "induction_head_studies",
        "monosemantic_studies",
        "superposition_studies",
        # Wave-1275 inference-scaling canon.
        "deliberate_search_studies",
        "latent_reasoning_studies",
        "self_improvement_studies",
        "test_time_scaling_studies",
        "tree_thought_studies",
        "verifier_gated_studies",
        # Wave-1274 agent-infrastructure canon.
        "agent_memory_studies",
        "code_agent_studies",
        "computer_use_studies",
        "mcp_protocol_studies",
        "skill_library_studies",
        "web_agent_studies",
        # Wave-1273 AI-safety canon.
        "alignment_eval_studies",
        "guardrail_studies",
        "hallucination_detect_studies",
        "jailbreak_defense_studies",
        "red_team_studies",
        "sleeper_agent_studies",
        # Wave-1272 omni-modal canon.
        "audio_encoder_studies",
        "document_ai_studies",
        "omni_modal_studies",
        "unified_tokenizer_studies",
        "video_llm_studies",
        "visual_grounding_studies",
        # Wave-1271 LLM-evaluation canon.
        "arena_battle_studies",
        "bigbench_studies",
        "capability_elicitation_studies",
        "contamination_detect_studies",
        "helm_eval_studies",
        "llm_judge_studies",
        # Wave-1270 mech-interp-2 canon.
        "attribution_graph_studies",
        "causal_tracing_studies",
        "circuit_discovery_studies",
        "feature_geometry_studies",
        "gated_sae_studies",
        "transcoder_studies",
        # Wave-1269 post-training-2 canon.
        "best_of_n_studies",
        "cdpo_studies",
        "constitutional_ai_studies",
        "orpo_studies",
        "simpo_studies",
        "sppo_studies",
        # Wave-1268 RL-skills/goal canon.
        "curiosity_diversity_studies",
        "hindsight_relabel_studies",
        "occupancy_measure_studies",
        "option_discovery_studies",
        "skill_chain_studies",
        "successor_feature_studies",
        # Wave-1267 neuro-symbolic canon.
        "alpha_tensor_studies",
        "differentiable_sat_studies",
        "neural_theorem_studies",
        "program_synthesis_studies",
        "sketch_programming_studies",
        "symbolic_regression_dl_studies",
        # Wave-1266 LLM-inference-2 canon.
        "diffusion_lm_studies",
        "kv_compression_studies",
        "medusa_speculation_studies",
        "moe_shared_expert_studies",
        "rope_scaling_studies",
        "sparse_attention_studies",
        # Wave-1265 genetic-epidemiology/MR canon.
        "colocalization_studies",
        "genetic_correlation_studies",
        "heritability_ldscore_studies",
        "mendelian_randomization_studies",
        "pleiotropy_robust_studies",
        "polygenic_score_studies",
        # Wave-1264 causal-RWE-2 canon.
        "external_control_studies",
        "negative_control_studies",
        "probabilistic_bias_studies",
        "self_controlled_studies",
        "structural_nested_studies",
        "transportability_studies",
        # Wave-1263 evidence-synthesis canon.
        "diagnostic_meta_studies",
        "fragility_index_studies",
        "individual_patient_meta_studies",
        "network_meta_studies",
        "trial_sequential_studies",
        "umbrella_review_studies",
        # Wave-1262 target-trial/RWE canon.
        "dynamic_borrowing_studies",
        "e_value_studies",
        "master_protocol_studies",
        "stepped_wedge_studies",
        "target_trial_emulation_studies",
        "win_ratio_studies",
        # Wave-1261 trial-statistics/HEOR canon.
        "biostatistics_methods_studies",
        "epidemiology_methods_studies",
        "heor_studies",
        "regulatory_science_studies",
        "survival_trial_studies",
        "translational_studies",
        # Wave-1260 clinical-research-methods canon.
        "adaptive_trial_studies",
        "clinical_trial_studies",
        "comparative_effectiveness_studies",
        "meta_analysis_studies",
        "outcomes_research_studies",
        "rwe_studies",
        # Wave-1259 drug-discovery canon.
        "qsar_studies",
        "docking_studies",
        "admet_studies",
        "lead_optimization_studies",
        "virtual_screening_studies",
        "de_novo_design_studies",
        # Wave-1258 omics canon.
        "transcriptome_studies",
        "proteome_studies",
        "metabolome_studies",
        "microbiome_studies",
        "methylome_studies",
        "interactome_studies",
        # Wave-1257 clinical-lab canon.
        "immunoassay_studies",
        "pcr_studies",
        "serology_studies",
        "culture_studies",
        "microscopy_studies",
        "flow_cytometry_studies",
        # Wave-1256 public-health-2 canon.
        "screening_studies",
        "vaccination_studies",
        "outbreak_studies",
        "surveillance_studies",
        "health_disparities_studies",
        "community_health_studies",
        # Wave-1255 neurodegeneration canon.
        "parkinson_studies",
        "alzheimer_studies",
        "ms_studies",
        "als_studies",
        "huntington_studies",
        "dementia_studies",
        # Wave-1254 molecular-genetics-2 canon.
        "allele_studies",
        "snp_studies",
        "cnv_studies",
        "haplotype_studies",
        "penetrance_studies",
        "pedigree_studies",
        # Wave-1253 immune-mediators canon.
        "cytokine_studies",
        "chemokine_studies",
        "interferon_studies",
        "complement_studies",
        "antibody_studies",
        "lymphocyte_studies",
        # Wave-1252 metabolic-endocrine canon.
        "pituitary_studies",
        "parathyroid_studies",
        "lipid_studies",
        "obesity_studies",
        "metabolic_syndrome_studies",
        "hypothalamic_studies",
        # Wave-1251 urology-andrology canon.
        "prostate_studies",
        "bladder_studies",
        "andrology_studies",
        "erectile_studies",
        "incontinence_studies",
        "bph_studies",
        # Wave-1250 ent-head-neck canon.
        "sinus_studies",
        "laryngology_studies",
        "otology_studies",
        "rhinology_studies",
        "head_neck_surgery_studies",
        "cochlear_studies",
        # Wave-1249 dermatology-clinical canon.
        "skin_cancer_studies",
        "psoriasis_studies",
        "eczema_studies",
        "acne_studies",
        "vitiligo_studies",
        "alopecia_studies",
        # Wave-1248 ophthalmology-vision canon.
        "retinal_studies",
        "corneal_studies",
        "glaucoma_studies",
        "cataract_studies",
        "macular_studies",
        "refractive_studies",
        # Wave-1247 nephro-renal canon.
        "glomerular_studies",
        "tubulointerstitial_studies",
        "ckd_studies",
        "aki_studies",
        "electrolyte_studies",
        "stones_studies",
        # Wave-1246 surgical-subspecialty canon.
        "bariatric_surgery_studies",
        "pediatric_surgery_studies",
        "plastic_surgery_studies",
        "burn_surgery_studies",
        "endocrine_surgery_studies",
        "transplant_surgery_studies",
        # Wave-1245 oncology-subspecialty canon.
        "medical_oncology_studies",
        "immuno_oncology_studies",
        "targeted_therapy_studies",
        "breast_oncology_studies",
        "thoracic_oncology_studies",
        "gi_oncology_studies",
        # Wave-1244 imaging-modality canon.
        "neuroradiology_studies",
        "mammography_studies",
        "ultrasound_studies",
        "ct_imaging_studies",
        "mri_studies",
        "pet_imaging_studies",
        # Wave-1243 infectious-medicine canon.
        "sepsis_studies",
        "tuberculosis_studies",
        "mycosis_studies",
        "sexually_transmitted_studies",
        "healthcare_infection_studies",
        "opportunistic_studies",
        # Wave-1242 rehab-medicine canon.
        "rehabilitation_studies",
        "physical_therapy_studies",
        "sports_injury_studies",
        "fracture_studies",
        "osteoporosis_studies",
        "physiatry_studies",
        # Wave-1241 hematology canon.
        "anemia_studies",
        "coagulation_studies",
        "hemoglobin_studies",
        "thrombosis_medicine",
        "bleeding_disorders",
        "marrow_studies",
        # Wave-1240 pulmonology canon.
        "respiratory_studies",
        "asthma_studies",
        "copd_studies",
        "interstitial_lung_studies",
        "sleep_breathing_studies",
        "bronchiectasis_studies",
        # Wave-1239 gi-medicine canon.
        "gi_endoscopy_studies",
        "hepatology_medicine",
        "pancreatic_medicine",
        "ibd_studies",
        "celiac_studies",
        "motility_studies",
        # Wave-1238 vascular-medicine canon.
        "vascular_medicine_studies",
        "phlebology_studies",
        "lymphatic_medicine",
        "vascular_lab_studies",
        "peripheral_artery_studies",
        "aortic_medicine_studies",
        # Wave-1237 clinical-pharmacy canon.
        "clinical_pharmacy_studies",
        "pharmacy_practice_studies",
        "medication_therapy_mgmt",
        "compounding_pharmacy",
        "pharmacovigilance_studies",
        "hospital_pharmacy_studies",
        # Wave-1236 pain canon.
        "chronic_pain_studies",
        "fibromyalgia_studies",
        "headache_studies",
        "neuropathic_pain_studies",
        "opioid_stewardship_studies",
        "interventional_pain_studies",
        # Wave-1235 rheumatology canon.
        "rheumatology_medicine",
        "spondyloarthritis_studies",
        "inflammatory_arthritis_studies",
        "connective_tissue_studies",
        "osteoarthritis_studies",
        "myositis_studies",
        # Wave-1234 womens-health canon.
        "menopause_medicine",
        "urogynecology_studies",
        "breast_medicine",
        "infertility_studies",
        "contraception_studies",
        "pelvic_health_studies",
        # Wave-1233 clinical-genetics canon.
        "medical_genetics_studies",
        "genetic_diagnostics",
        "lysosomal_medicine",
        "mitochondrial_medicine",
        "dysmorphology_studies",
        "pharmacogenomics_studies",
        # Wave-1232 geriatrics canon.
        "geriatrics_studies",
        "frailty_medicine",
        "memory_clinic_studies",
        "falls_prevention_studies",
        "polypharmacy_studies",
        "caregiver_medicine",
        # Wave-1231 transplant-immunology canon.
        "transplant_medicine_studies",
        "immunology_medicine",
        "allergy_studies",
        "autoimmunity_studies",
        "hematopoietic_transplant",
        "immunodeficiency_studies",
        # Wave-1230 emergency-medicine canon.
        "emergency_medicine_studies",
        "trauma_medicine",
        "toxicology_medicine",
        "disaster_medicine",
        "acute_care_studies",
        "resuscitation_medicine",
        # Wave-1229 pediatrics canon.
        "pediatrics_studies",
        "neonatal_medicine_studies",
        "pediatric_cardiology",
        "pediatric_oncology",
        "adolescent_medicine_studies",
        "developmental_pediatrics",
        # Wave-1228 dentistry canon.
        "dental_studies",
        "oral_surgery_studies",
        "endodontic_studies",
        "periodontal_studies",
        "orthodontic_studies",
        "pediatric_dentistry",
        # Wave-1227 radiology canon.
        "radiology_studies",
        "diagnostic_imaging",
        "interventional_neuroradiology",
        "pediatric_imaging",
        "musculoskeletal_imaging",
        "body_imaging",
        # Wave-1226 pathology canon.
        "pathology_studies",
        "anatomical_pathology",
        "clinical_pathology",
        "histopathology_studies",
        "cytopathology",
        "molecular_pathology",
        # Wave-1225 surgery canon.
        "general_surgery_studies",
        "trauma_surgery",
        "colorectal_surgery",
        "hepatobiliary_surgery",
        "surgical_oncology_studies",
        "minimally_invasive_surgery",
        # Wave-1224 obgyn canon.
        "obstetrics_studies",
        "gynecology_studies",
        "maternal_fetal_medicine",
        "reproductive_endocrinology",
        "gynecologic_oncology",
        "fetal_medicine",
        # Wave-1223 anesthesia canon.
        "anesthesiology_studies",
        "perioperative_medicine",
        "pain_medicine_studies",
        "regional_anesthesia",
        "sedation_medicine",
        "airway_management",
        # Wave-1222 ortho canon.
        "orthopedics_studies",
        "sports_medicine_orthopedics",
        "musculoskeletal_medicine",
        "spine_surgery",
        "joint_replacement",
        "hand_surgery",
        # Wave-1221 derm-eye-ent canon.
        "dermatology_studies",
        "ophthalmology_studies",
        "otolaryngology_studies",
        "audiology_medicine",
        "optometry_studies",
        "dermatopathology",
        # Wave-1220 infectious-immune canon.
        "infectious_disease_medicine",
        "hiv_medicine",
        "antimicrobial_stewardship",
        "rheumatology_studies",
        "immunology_studies",
        "allergy_immunology",
        # Wave-1219 hem-onc canon.
        "hematology_studies",
        "oncology_studies",
        "hematologic_malignancies",
        "solid_tumor_oncology",
        "transfusion_medicine",
        "radiation_oncology",
        # Wave-1218 nephrology canon.
        "nephrology_studies",
        "dialysis_medicine",
        "renal_transplant",
        "acid_base_medicine",
        "hypertension_medicine",
        "urology_studies",
        # Wave-1217 endocrinology canon.
        "endocrinology_studies",
        "diabetes_medicine",
        "thyroid_medicine",
        "metabolic_medicine",
        "bone_metabolism",
        "adrenal_medicine",
        # Wave-1216 internal-medicine canon.
        "internal_medicine",
        "hospital_medicine",
        "critical_care_medicine",
        "pulmonary_medicine",
        "gastroenterology_studies",
        "hepatology_studies",
        # Wave-1215 cardio-surgery canon.
        "vascular_surgery",
        "cardiac_surgery",
        "thoracic_surgery",
        "transplant_cardiology",
        "structural_heart",
        "adult_congenital",
        # Wave-1214 cardiology canon.
        "cardiology_studies",
        "interventional_cardiology",
        "electrophysiology_studies",
        "heart_failure_medicine",
        "preventive_cardiology",
        "cardiovascular_imaging",
        # Wave-1213 neurology-3 canon.
        "neurosurgery_studies",
        "neurotrauma",
        "neurotoxicology",
        "neurorehabilitation",
        "neurovascular_surgery",
        "spinal_cord_medicine",
        # Wave-1212 neurology-2 canon.
        "neurocritical_care",
        "neurovascular_studies",
        "neuromuscular_medicine",
        "neuro_ophthalmology",
        "neuroimmunology",
        "neurogenetics",
        # Wave-1211 neurology canon.
        "pediatric_neurology",
        "neurodevelopmental_disorders",
        "neuropsychiatry_studies",
        "headache_medicine",
        "epilepsy_studies",
        "movement_disorders",
        # Wave-1210 psychiatry canon.
        "forensic_psychiatry",
        "geriatric_psychiatry",
        "mood_disorders",
        "psychotic_disorders",
        "personality_disorders",
        "anxiety_disorders",
        # Wave-1209 behavioral-health canon.
        "addiction_medicine",
        "eating_disorders",
        "sleep_disorders",
        "psychosomatic_medicine",
        "consultation_liaison",
        "community_psychiatry",
        # Wave-1208 therapy-modalities canon.
        "marriage_family_therapy",
        "group_therapy",
        "couples_therapy",
        "family_therapy",
        "child_adolescent_therapy",
        "trauma_therapy",
        # Wave-1207 psychotherapy canon.
        "psychoanalysis_studies",
        "psychotherapy_studies",
        "behavioral_therapy_cognitive",
        "art_therapy",
        "music_therapy",
        "play_therapy",
        # Wave-1206 transplantation canon.
        "transplantation_medicine",
        "organ_donation",
        "immunosuppression",
        "xenotransplantation",
        "stem_cell_therapy",
        "regenerative_medicine",
        # Wave-1205 extreme-environment canon.
        "aerospace_medicine",
        "diving_medicine",
        "wilderness_medicine",
        "space_physiology",
        "hyperbaric_oxygen",
        "high_altitude_medicine",
        # Wave-1204 geriatric-care canon.
        "geriatric_medicine",
        "palliative_care",
        "hospice_care",
        "gerontology_studies",
        "aging_research",
        "longevity_medicine",
        # Wave-1203 molecular-medicine canon.
        "tropical_medicine",
        "travel_medicine",
        "genomic_medicine",
        "precision_medicine",
        "molecular_diagnostics",
        "laboratory_medicine",
        # Wave-1202 infectious-disease canon.
        "public_health_microbiology",
        "medical_microbiology",
        "parasitology_studies",
        "mycology_studies",
        "entomology_medical",
        "vector_borne_diseases",
        # Wave-1201 health-informatics canon.
        "health_informatics",
        "medical_records",
        "health_information",
        "biomedical_informatics",
        "clinical_informatics",
        "health_data_science",
        # Wave-1200 medical-physics canon.
        "cardiovascular_technology",
        "nuclear_medicine_technology",
        "radiation_dosimetry",
        "medical_physics_studies",
        "dosimetry_studies",
        "radiopharmacy",
        # Wave-1199 interventional-medicine canon.
        "dialysis_technology",
        "transplant_studies",
        "hepatobiliary_studies",
        "cardiac_electrophysiology",
        "interventional_radiology",
        "nuclear_cardiology",
        # Wave-1198 procedural-medicine canon.
        "sleep_medicine",
        "pain_management",
        "wound_care",
        "infusion_therapy",
        "hyperbaric_medicine",
        "electrodiagnostic_studies",
        # Wave-1197 counseling-neonatal canon.
        "addiction_counseling",
        "rehabilitation_counseling",
        "genetic_screening",
        "prenatal_studies",
        "neonatology_studies",
        "pediatric_therapeutics",
        # Wave-1196 emergency-safety canon.
        "emergency_medical_technician",
        "fire_science_studies",
        "paramedic_studies",
        "disaster_management",
        "occupational_safety",
        "industrial_hygiene",
        # Wave-1195 clinical-support canon.
        "medical_imaging_studies",
        "clinical_laboratory",
        "mortuary_science",
        "phlebotomy_studies",
        "surgical_technology",
        "sterile_processing",
        # Wave-1194 clinical-specialties canon.
        "genetic_counseling",
        "lactation_consulting",
        "podiatric_medicine",
        "respiratory_therapy",
        "perfusion_technology",
        "radiation_therapy",
        # Wave-1193 allied-health-2 canon.
        "midwifery_studies",
        "orthoptics",
        "audiology_studies",
        "opticianry",
        "prosthetics_orthotics",
        "clinical_psychology_2",
        # Wave-1192 integrative-medicine canon.
        "acupuncture_studies",
        "chiropractic_studies",
        "naturopathy",
        "homeopathy",
        "herbal_medicine",
        "osteopathy_studies",
        # Wave-1191 media canon.
        "journalism_studies",
        "advertising_studies",
        "broadcasting_studies",
        "news_media",
        "public_relations_studies",
        "publishing_studies",
        # Wave-1190 visual-design canon.
        "graphic_design",
        "typography_studies",
        "photography_studies",
        "print_media",
        "web_design",
        "motion_graphics",
        # Wave-1189 hospitality canon.
        "hospitality_studies",
        "event_management",
        "hotel_management",
        "tourism_studies",
        "recreation_management",
        "leisure_science",
        # Wave-1188 culinary canon.
        "culinary_science",
        "pastry_arts",
        "brewing_science",
        "enology",
        "fermentation_studies",
        "gastronomy_2",
        # Wave-1187 fashion canon.
        "fashion_studies",
        "textile_studies",
        "costume_design",
        "jewelry_design",
        "footwear_design",
        "apparel_studies",
        # Wave-1186 ux canon.
        "ux_design",
        "hci_studies",
        "information_architecture",
        "interaction_design",
        "accessibility_studies",
        "service_design",
        # Wave-1185 game canon.
        "game_design",
        "esports_studies",
        "interactive_media",
        "game_studies",
        "ludology",
        "game_development",
        # Wave-1184 film-production canon.
        "film_production",
        "cinematography_studies",
        "film_editing",
        "sound_design",
        "documentary_production",
        "animation_studies",
        # Wave-1183 music canon.
        "music_theory_2",
        "musicology_2",
        "ethnomusicology_2",
        "music_cognition_2",
        "organology_2",
        "composition_studies",
        # Wave-1182 dance canon.
        "ballet_studies",
        "choreography_2",
        "dance_pedagogy",
        "somatic_practices",
        "dance_science",
        "movement_studies",
        # Wave-1181 performing-arts canon.
        "performing_arts_2",
        "theater_arts",
        "acting_studies",
        "directing_studies",
        "playwriting",
        "scenography",
        # Wave-1180 allied-health canon.
        "nursing_studies",
        "allied_health",
        "midwifery",
        "paramedicine",
        "occupational_science",
        "speech_pathology",
        # Wave-1179 security canon.
        "war_studies",
        "strategic_analysis",
        "intelligence_analysis",
        "peace_research",
        "conflict_studies",
        "military_history_2",
        # Wave-1178 interdisciplinary canon.
        "interdisciplinary_studies",
        "cognitive_science_2",
        "futures_studies",
        "complexity_science",
        "systems_science",
        "human_computer_interaction",
        # Wave-1177 formal-sciences canon.
        "formal_sciences",
        "mathematical_logic",
        "axiomatic_systems",
        "proof_calculus",
        "model_checking_2",
        "formal_ontology",
        # Wave-1176 theology canon.
        "theology_3",
        "religious_studies_3",
        "comparative_religion_2",
        "biblical_studies_2",
        "islamic_studies_2",
        "buddhist_studies_2",
        # Wave-1175 trades canon.
        "electrical_trades",
        "plumbing_hvac",
        "welding_technology",
        "carpentry_trades",
        "automotive_technology",
        "refrigeration_technology",
        # Wave-1174 recreation canon.
        "recreation",
        "leisure_studies",
        "tourism",
        "hospitality",
        "sports_management",
        "recreation_therapy",
        # Wave-1173 media canon.
        "communication_3",
        "journalism_3",
        "media_studies_3",
        "rhetoric_2",
        "information_science_3",
        "digital_media_2",
        # Wave-1172 business canon.
        "management_3",
        "marketing_3",
        "accounting_3",
        "finance_5",
        "entrepreneurship_3",
        "organizational_behavior",
        # Wave-1171 geography canon.
        "geography_2",
        "regional_science",
        "demography_2",
        "urbanization",
        "land_use",
        "gis_science_2",
        # Wave-1170 justice canon.
        "criminology_3",
        "forensic_science_2",
        "penology_2",
        "victimology_2",
        "security_studies_2",
        "intelligence_studies_2",
        # Wave-1169 social-policy canon.
        "social_work_2",
        "public_policy_2",
        "urban_studies_2",
        "gender_studies_2",
        "ethnic_studies_2",
        "disability_studies_2",
        # Wave-1168 logistics canon.
        "transportation_2",
        "logistics_2",
        "supply_chain_2",
        "warehousing_2",
        "maritime_studies_2",
        "aviation_2",
        # Wave-1167 performing-arts canon.
        "music_2",
        "theater_2",
        "dance_2",
        "film_studies_3",
        "art_history_2",
        "performance_studies_2",
        # Wave-1166 design canon.
        "architecture_2",
        "urban_planning_2",
        "interior_design_2",
        "landscape_architecture_2",
        "industrial_design_2",
        "graphic_design_2",
        # Wave-1165 agriculture canon.
        "agriculture_2",
        "food_science_2",
        "forestry_2",
        "fisheries_2",
        "horticulture_2",
        "veterinary_science_2",
        # Wave-1164 communication canon.
        "education_5",
        "communication_studies_2",
        "media_studies_2",
        "journalism_2",
        "library_science_2",
        "information_science_2",
        # Wave-1163 business canon.
        "accounting_2",
        "finance_4",
        "marketing_2",
        "management_2",
        "entrepreneurship_2",
        "business_administration",
        # Wave-1162 governance canon.
        "law_5",
        "political_science_4",
        "public_administration_2",
        "international_relations_2",
        "criminology_2",
        "military_science_2",
        # Wave-1161 health-sciences canon.
        "medicine_7",
        "dentistry_3",
        "nursing_2",
        "public_health_2",
        "veterinary_medicine_2",
        "pharmacy_2",
        # Wave-1160 humanities canon.
        "philosophy_6",
        "history_5",
        "religious_studies_2",
        "classics_2",
        "area_studies_2",
        "humanities_2",
        # Wave-1159 engineering canon.
        "biomedical_engineering_2",
        "chemical_engineering_2",
        "mechanical_engineering_2",
        "civil_engineering_2",
        "electrical_engineering_2",
        "aerospace_engineering_2",
        # Wave-1158 social-sciences canon.
        "sociology_6",
        "economics_6",
        "political_science_3",
        "psychology_5",
        "anthropology_6",
        "linguistics_7",
        # Wave-1157 computing-sciences canon.
        "computer_science_2",
        "software_engineering",
        "machine_learning_2",
        "artificial_intelligence",
        "data_engineering",
        "information_theory_2",
        # Wave-1156 mathematical-sciences canon.
        "applied_mathematics",
        "statistics_2",
        "probability_4",
        "computational_science",
        "data_science",
        "bioinformatics_5",
        # Wave-1155 earth-systems canon.
        "earth_system_science",
        "oceanography_2",
        "atmospheric_science",
        "environmental_science_2",
        "soil_science_2",
        "hydrology_3",
        # Wave-1154 fundamental-physics canon.
        "electromagnetism",
        "optics_4",
        "nuclear_physics_2",
        "particle_physics",
        "quantum_physics",
        "relativity_3",
        # Wave-1153 physical-sciences canon.
        "physics_6",
        "astrophysics_3",
        "cosmology_3",
        "geophysics_3",
        "mechanics",
        "thermodynamics_3",
        # Wave-1152 molecular-life-sciences canon.
        "biochemistry_2",
        "molecular_biology_2",
        "cell_biology_2",
        "genetics_2",
        "pharmacology_2",
        "toxicology_3",
        # Wave-1151 chemical-sciences canon.
        "chemistry_3",
        "organic_chemistry_2",
        "inorganic_chemistry_2",
        "physical_chemistry_2",
        "analytical_chemistry_2",
        "electrochemistry_2",
        # Wave-1150 microbial-genetics canon.
        "microbiology_2",
        "bacteriology",
        "virology_2",
        "immunogenetics",
        "molecular_genetics",
        "epigenetics",
        # Wave-1149 clinical-medicine canon.
        "toxicology_2",
        "dermatology_2",
        "hematology_2",
        "pulmonology_2",
        "nephrology_2",
        "hepatology_2",
        # Wave-1148 geological-sciences canon.
        "geology_3",
        "petrology_2",
        "mineralogy_2",
        "stratigraphy_2",
        "geomorphology_2",
        "geochronology_2",
        # Wave-1147 biomedical-science canon.
        "anatomy",
        "physiology_2",
        "endocrinology_2",
        "neuroscience_2",
        "cardiology_2",
        "immunology_2",
        # Wave-1146 organismal-biology canon.
        "virology",
        "parasitology",
        "mycology",
        "entomology_2",
        "limnology",
        "wildlife_biology",
        # Wave-1145 life-science canon.
        "genomicsciences",
        "proteomics",
        "bioinformatics_4",
        "systems_biology_2",
        "synthetic_biology",
        "epidemiology_3",
        # Wave-1144 space-science canon.
        "space_weather",
        "planetology",
        "asteroid_science",
        "comet_science",
        "astrophotonics",
        "grav_waves_2",
        # Wave-1143 physics-5 canon.
        "nanotechnology",
        "biophysics_2",
        "condensed_matter_3",
        "optics_3",
        "acoustics_2",
        "thermodynamics_2",
        # Wave-1142 quantum-technology canon.
        "quantum_computing",
        "quantum_information_2",
        "quantum_chemistry_2",
        "quantum_optics",
        "quantum_sensing",
        "quantum_error_2",
        # Wave-1141 astronomy-4 canon.
        "cosmology_2",
        "astrobiology",
        "astrochemistry",
        "helio_seismology",
        "exoplanet_science",
        "galactic_dynamics",
        # Wave-1140 earth-science canon.
        "oceanography",
        "hydrology_2",
        "seismology",
        "glaciology",
        "paleoclimatology",
        "volcanology_2",
        # Wave-1139 medicine-6 canon.
        "optometry",
        "dentistry_2",
        "podiatry",
        "dietetics",
        "physiotherapy",
        "occupational_therapy",
        # Wave-1138 economics-5 canon.
        "behavioral_economics",
        "econ_neuroscience",
        "experimental_economics_2",
        "institutional_economics",
        "evolutionary_economics",
        "political_economy_2",
        # Wave-1137 education-4 canon.
        "early_childhood_education",
        "bilingual_education",
        "gifted_education",
        "adult_education",
        "instructional_design",
        "educational_leadership",
        # Wave-1136 law-4 canon.
        "environmental_law",
        "family_law",
        "labor_law",
        "tax_law",
        "evidence_law",
        "immigration_law",
        # Wave-1135 medicine-5 canon.
        "urology",
        "ophthalmology",
        "otolaryngology",
        "palliative_medicine",
        "sports_medicine",
        "rehabilitation_medicine",
        # Wave-1134 computational-math canon.
        "finite_element_theory",
        "spectral_theory_numerics",
        "adaptive_method_theory",
        "reduced_order_modeling",
        "uncertainty_quantification_2",
        "high_performance_numerics",
        # Wave-1133 physics-4 canon.
        "statistical_field_theory",
        "conformal_field_theory",
        "lattice_field_theory",
        "string_theory_math",
        "loop_quantum_gravity",
        "holography_ads",
        # Wave-1132 history-4 canon.
        "history_of_emotions",
        "history_of_sexuality",
        "history_of_the_book",
        "history_of_capitalism",
        "history_of_religions",
        "microhistory",
        # Wave-1131 sociology-5 canon.
        "sociology_of_migration",
        "sociology_of_housing",
        "sociology_of_disaster",
        "sociology_of_the_body",
        "sociology_of_risk",
        "digital_sociology",
        # Wave-1130 anthropology-5 canon.
        "social_anthropology",
        "cognitive_anthropology",
        "anthropology_of_religion",
        "kinship_studies",
        "material_culture",
        "museum_anthropology",
        # Wave-1129 philosophy-5 canon.
        "bioethics",
        "philosophy_of_education",
        "feminist_philosophy",
        "african_philosophy",
        "environmental_philosophy",
        "philosophy_of_medicine",
        # Wave-1128 linguistics-6 canon.
        "lexical_semantics",
        "computational_stylistics",
        "stylistics",
        "corpus_phonology",
        "language_documentation",
        "translation_technology",
        # Wave-1127 history-3 canon.
        "oral_history",
        "public_history",
        "digital_history",
        "environmental_history",
        "global_history",
        "maritime_history",
        # Wave-1126 psychology-4 canon.
        "social_cognition",
        "positive_psychology",
        "cross_cultural_psychology",
        "consumer_psychology",
        "political_psychology",
        "community_psychology",
        # Wave-1125 economics-4 canon.
        "development_economics",
        "environmental_economics",
        "health_economics",
        "urban_economics",
        "agricultural_economics",
        "energy_economics",
        # Wave-1124 sociology-4 canon.
        "sociology_of_work",
        "sociology_of_emotions",
        "sociology_of_food",
        "sociology_of_media",
        "sociology_of_sport",
        "sociology_of_aging",
        # Wave-1123 linguistics-5 canon.
        "contact_linguistics",
        "descriptive_linguistics",
        "philological_studies",
        "etymology",
        "dialectometry",
        "lexicography",
        # Wave-1122 archaeology-2 canon.
        "geoarchaeology",
        "zooarchaeology",
        "paleoethnobotany",
        "ceramic_analysis",
        "lithic_analysis",
        "archaeogenetics",
        # Wave-1121 geography-3 canon.
        "regional_geography",
        "health_geography",
        "population_geography",
        "economic_geography",
        "political_geography",
        "gis_science",
        # Wave-1120 anthropology-4 canon.
        "visual_anthropology",
        "applied_anthropology",
        "forensic_anthropology",
        "digital_anthropology",
        "environmental_anthropology",
        "psychological_anthropology",
        # Wave-1119 history-2 canon.
        "social_history",
        "cultural_history",
        "military_history",
        "diplomatic_history",
        "history_of_technology",
        "history_of_medicine",
        # Wave-1118 linguistics-4 canon.
        "theoretical_linguistics",
        "field_linguistics",
        "translation_theory",
        "sign_linguistics",
        "linguistic_typology",
        "language_acquisition",
        # Wave-1117 sociology-3 canon.
        "mathematical_sociology",
        "historical_sociology",
        "science_studies",
        "sociology_of_knowledge",
        "military_sociology",
        "legal_sociology",
        # Wave-1116 psychology-3 canon.
        "experimental_psychology",
        "comparative_psychology",
        "evolutionary_psychology",
        "psychopathology",
        "environmental_psychology",
        "sport_psychology",
        # Wave-1115 economics-3 canon.
        "labor_economics",
        "public_economics",
        "industrial_organization",
        "international_economics",
        "financial_economics",
        "monetary_economics",
        # Wave-1114 physics-3 canon.
        "classical_mechanics",
        "quantum_mechanics_2",
        "statistical_mechanics_2",
        "nuclear_physics",
        "plasma_physics",
        "condensed_matter_2",
        # Wave-1113 medicine-4 canon.
        "surgery",
        "anesthesiology",
        "obstetrics_gynecology",
        "pediatrics",
        "emergency_medicine",
        "family_medicine",
        # Wave-1112 biology-2 canon.
        "biophysics",
        "evolutionary_biology",
        "developmental_biology",
        "neurobiology",
        "ethology",
        "comparative_anatomy",
        # Wave-1111 chemistry-2 canon.
        "quantum_chemistry",
        "spectroscopy",
        "photochemistry",
        "stereochemistry",
        "supramolecular_chemistry",
        "medicinal_chemistry",
        # Wave-1110 philosophy-4 canon.
        "philosophy_of_biology",
        "philosophy_of_mathematics",
        "philosophy_of_religion",
        "phenomenology_2",
        "philosophy_of_history",
        "process_philosophy",
        # Wave-1109 linguistics-3 canon.
        "applied_linguistics",
        "anthropological_linguistics",
        "neurolinguistics",
        "evolutionary_linguistics",
        "forensic_linguistics",
        "discourse_analysis",
        # Wave-1108 sociology-3 canon.
        "industrial_sociology",
        "political_sociology",
        "sociology_of_education",
        "sociology_of_religion",
        "environmental_sociology",
        "cultural_sociology",
        # Wave-1107 medicine-3 canon.
        "gastroenterology",
        "endocrinology",
        "hematology",
        "pulmonology",
        "nephrology",
        "infectious_diseases",
        # Wave-1106 materials-2 canon.
        "semiconductors_materials",
        "composite_materials",
        "thin_films",
        "biomaterials",
        "phase_diagrams",
        "characterization_methods",
        # Wave-1105 behavioral-econ-2 canon.
        "prospect_theory",
        "bounded_rationality",
        "nudge_theory",
        "neuroeconomics",
        "experimental_economics",
        "financial_behavior",
        # Wave-1104 political-science-2 canon.
        "american_politics",
        "political_behavior",
        "public_law",
        "political_methodology",
        "security_studies",
        "policy_analysis",
        # Wave-1103 meteorology-2 canon.
        "severe_weather",
        "boundary_layer_meteorology",
        "radar_meteorology",
        "tropical_meteorology",
        "polar_meteorology",
        "micrometeorology",
        # Wave-1102 geology-2 canon.
        "mineralogy",
        "volcanology",
        "sedimentology",
        "tectonics",
        "hydrogeology",
        "geophysics_applied",
        # Wave-1101 biology canon.
        "molecular_biology",
        "cell_biology",
        "genetics",
        "microbiology",
        "zoology",
        "botany",
        # Wave-1100 chemistry canon.
        "organic_chemistry",
        "inorganic_chemistry",
        "physical_chemistry",
        "analytical_chemistry",
        "biochemistry",
        "electrochemistry",
        # Wave-1099 psychology-2 canon.
        "personality_psychology",
        "abnormal_psychology",
        "health_psychology",
        "neuropsychology",
        "forensic_psychology",
        "organizational_psychology",
        # Wave-1098 anthropology-2 canon.
        "biological_anthropology",
        "paleoanthropology",
        "medical_anthropology",
        "economic_anthropology",
        "political_anthropology",
        "urban_anthropology",
        # Wave-1097 philosophy-3 canon.
        "moral_philosophy",
        "political_philosophy",
        "philosophy_of_mind",
        "philosophy_of_language",
        "philosophy_of_law",
        "eastern_philosophy",
        # Wave-1096 environmental-2 canon.
        "pollution_science",
        "conservation_biology",
        "environmental_toxicology",
        "urban_ecology",
        "landscape_ecology",
        "marine_conservation",
        # Wave-1095 sociology-2 canon.
        "medical_sociology",
        "deviance_studies",
        "family_sociology",
        "organization_theory",
        "social_movements",
        "rural_sociology",
        # Wave-1094 medicine-2 canon.
        "oncology",
        "neurology",
        "dermatology",
        "orthopedics",
        "psychiatry",
        "radiology",
        # Wave-1093 law-2 canon.
        "civil_law",
        "common_law",
        "canon_law",
        "maritime_law",
        "property_law",
        "procedural_law",
        # Wave-1092 art-historiography canon.
        "iconography",
        "iconology",
        "connoisseurship",
        "provenance_studies",
        "curation_practice",
        "formal_analysis",
        # Wave-1091 education-2 canon.
        "higher_education",
        "vocational_education",
        "special_education",
        "comparative_education",
        "literacy_studies",
        "distance_learning",
        # Wave-1090 history-of-science canon.
        "history_of_science",
        "sts_studies",
        "philosophy_of_technology",
        "media_archaeology",
        "information_history",
        "technology_studies",
        # Wave-1089 linguistics-2 canon.
        "sociolinguistics",
        "psycholinguistics",
        "computational_linguistics",
        "corpus_linguistics",
        "dialectology",
        "historical_linguistics",
        # Wave-1088 philosophy-2 canon.
        "ancient_philosophy",
        "medieval_philosophy",
        "continental_philosophy",
        "analytic_philosophy",
        "pragmatism",
        "existentialism",
        # Wave-1087 documentary-sciences canon.
        "epigraphy",
        "diplomatics",
        "sigillography",
        "heraldry",
        "genealogy_studies",
        "onomastics",
        # Wave-1086 near-eastern canon.
        "assyriology",
        "egyptology",
        "sinology",
        "indology",
        "iranian_studies",
        "ottoman_studies",
        # Wave-1085 literary-periods canon.
        "medieval_literature",
        "renaissance_literature",
        "romanticism",
        "modernism",
        "postmodernism",
        "victorian_studies",
        # Wave-1084 comparative-literature canon.
        "comparative_literature",
        "literary_theory",
        "postcolonial_studies",
        "world_literature",
        "translation_studies",
        "critical_theory",
        # Wave-1083 humanities-theory canon.
        "semiotics",
        "narratology",
        "hermeneutics",
        "phenomenology",
        "structuralism",
        "poststructuralism",
        # Wave-1082 jewish studies canon.
        "jewish_studies",
        "talmudic_studies",
        "hebrew_language",
        "rabbinics",
        "kabbalah",
        "jewish_philosophy",
        # Wave-1081 renaissance/early modern canon.
        "renaissance_studies",
        "early_modern",
        "humanism",
        "reformation_studies",
        "baroque_studies",
        "enlightenment_studies",
        # Wave-1080 medieval studies canon.
        "medieval_studies",
        "paleography",
        "codicology",
        "hagiography",
        "byzantine_studies",
        "numismatics",
        # Wave-1079 classics canon.
        "classical_studies",
        "latin_language",
        "ancient_greek",
        "classical_archaeology",
        "philology",
        "papyrology",
        # Wave-1078 performing arts canon.
        "theater_studies",
        "dance_studies",
        "performance_theory",
        "dramaturgy",
        "choreography",
        "stage_design",
        # Wave-1077 visual arts canon.
        "painting_techniques",
        "sculpture_methods",
        "printmaking",
        "art_conservation",
        "art_history",
        "visual_culture",
        # Wave-1076 archaeology canon.
        "field_archaeology",
        "archaeometry",
        "bioarchaeology",
        "underwater_archaeology",
        "landscape_archaeology",
        "experimental_archaeology",
        # Wave-1075 architecture/design canon.
        "architecture_theory",
        "urban_design",
        "landscape_architecture",
        "interior_design",
        "industrial_design",
        "building_science",
        # Wave-1074 culinary arts canon.
        "culinary_arts",
        "gastronomy",
        "food_studies",
        "baking_science",
        "flavor_science",
        "fermentation_science",
        # Wave-1073 theology-2 canon.
        "systematic_theology",
        "biblical_exegesis",
        "church_history",
        "pastoral_theology",
        "liturgical_studies",
        "missiology",
        # Wave-1072 film studies canon.
        "film_studies",
        "cinema_studies",
        "film_theory",
        "film_history",
        "documentary_studies",
        "screenwriting",
        # Wave-1071 musicology canon.
        "musicology",
        "ethnomusicology",
        "music_theory",
        "music_cognition",
        "organology",
        "music_history",
        # Wave-1070 sports science canon.
        "sports_science",
        "exercise_physiology",
        "sports_biomechanics",
        "sports_psychology",
        "athletic_training",
        "sports_analytics",
        # Wave-1069 criminal justice canon.
        "criminal_justice",
        "forensic_science",
        "penology",
        "policing_studies",
        "victimology",
        "criminal_procedure",
        # Wave-1068 military/defense studies canon.
        "military_science",
        "defense_studies",
        "strategic_studies",
        "intelligence_studies",
        "peace_studies",
        "conflict_resolution",
        # Wave-1067 library/information science canon.
        "library_science",
        "information_science",
        "archival_studies",
        "museum_studies",
        "digital_humanities",
        "knowledge_organization",
        # Wave-1066 area studies canon.
        "latin_american_studies",
        "asian_studies",
        "european_studies",
        "middle_eastern_studies",
        "african_studies",
        "slavic_studies",
        # Wave-1065 geography canon.
        "physical_geography",
        "human_geography",
        "cartography",
        "remote_sensing",
        "geomorphology",
        "climatology",
        # Wave-1064 social-work/policy canon.
        "social_work",
        "public_policy",
        "urban_studies",
        "gender_studies",
        "ethnic_studies",
        "disability_studies",
        # Wave-1063 communications/media canon.
        "media_studies",
        "journalism",
        "public_relations",
        "rhetoric",
        "communication_theory",
        "digital_media",
        # Wave-1062 religious-studies canon.
        "theology",
        "comparative_religion",
        "biblical_studies",
        "islamic_studies",
        "buddhist_studies",
        "religious_ethics",
        # Wave-1061 law canon.
        "constitutional_law",
        "criminal_law",
        "contract_law",
        "tort_law",
        "administrative_law",
        "international_law",
        # Wave-1060 education canon.
        "curriculum_design",
        "pedagogy",
        "educational_psychology",
        "assessment_theory",
        "learning_sciences",
        "educational_technology",
        # Wave-1059 history canon.
        "historiography",
        "ancient_history",
        "medieval_history",
        "modern_history",
        "economic_history",
        "intellectual_history",
        # Wave-1058 philosophy canon.
        "metaphysics",
        "epistemology",
        "ethics_philosophy",
        "logic_philosophy",
        "philosophy_of_science",
        "aesthetics",
        # Wave-1057 linguistics canon.
        "phonetics",
        "phonology",
        "morphology",
        "syntax_theory",
        "semantics",
        "pragmatics",
        # Wave-1056 political-science canon.
        "comparative_politics",
        "international_relations",
        "political_theory",
        "public_administration",
        "political_economy",
        "electoral_systems",
        # Wave-1055 anthropology canon.
        "physical_anthropology",
        "cultural_anthropology",
        "archaeology",
        "linguistic_anthropology",
        "primatology",
        "ethnography",
        # Wave-1054 sociology canon.
        "social_networks",
        "demography",
        "criminology",
        "urban_sociology",
        "economic_sociology",
        "social_stratification",
        # Wave-1053 psychology canon.
        "cognitive_psychology",
        "psychometrics",
        "behavioral_neuroscience",
        "social_psychology",
        "developmental_psychology",
        "clinical_psychology",
        # Wave-1052 nutrition canon.
        "nutritional_biochemistry",
        "dietary_assessment",
        "clinical_nutrition",
        "sports_nutrition",
        "nutritional_epidemiology",
        "metabolic_health",
        # Wave-1051 public-health canon.
        "epidemiology_2",
        "biostatistics_2",
        "health_policy",
        "global_health",
        "occupational_health",
        "preventive_medicine",
        # Wave-1050 pharmacology canon.
        "pharmacodynamics",
        "pharmacokinetics_2",
        "toxicology",
        "clinical_pharmacology",
        "neuropharmacology",
        "drug_metabolism",
        # Wave-1049 dentistry canon.
        "dental_anatomy",
        "oral_pathology",
        "periodontology",
        "endodontics",
        "orthodontics",
        "prosthodontics",
        # Wave-1048 veterinary-medicine canon.
        "veterinary_anatomy",
        "veterinary_pathology",
        "veterinary_pharmacology",
        "animal_surgery",
        "veterinary_epidemiology",
        "equine_medicine",
        # Wave-1047 marine-biology canon.
        "plankton_dynamics",
        "marine_ecology",
        "fisheries_science",
        "aquaculture",
        "benthic_biology",
        "coral_reef_ecology",
        # Wave-1046 meteorology canon.
        "atmospheric_dynamics",
        "synoptic_meteorology",
        "cloud_physics",
        "numerical_weather",
        "mesoscale_meteorology",
        "climate_dynamics",
        # Wave-1045 geology canon.
        "stratigraphy",
        "structural_geology",
        "petrology",
        "geochemistry",
        "geochronology",
        "paleontology",
        # Wave-1044 mining-engineering canon.
        "mine_design",
        "rock_mechanics",
        "mineral_processing",
        "blasting_engineering",
        "mine_ventilation",
        "ore_reserve_estimation",
        # Wave-1043 forestry canon.
        "silviculture",
        "forest_ecology",
        "timber_harvesting",
        "forest_economics",
        "dendrology",
        "wildfire_management",
        # Wave-1042 food-science canon.
        "food_chemistry",
        "food_microbiology",
        "food_processing",
        "nutrition_science",
        "sensory_evaluation",
        "food_safety",
        # Wave-1041 ocean-engineering canon.
        "naval_architecture",
        "offshore_engineering",
        "marine_propulsion",
        "ocean_waves",
        "coastal_engineering",
        "submarine_systems",
        # Wave-1040 robotics-engineering canon.
        "robot_kinematics",
        "robot_dynamics",
        "motion_control",
        "sensor_fusion",
        "path_planning",
        "actuator_design",
        # Wave-1039 environmental-engineering canon.
        "water_treatment",
        "air_pollution_control",
        "waste_management",
        "environmental_remediation",
        "wastewater_engineering",
        "noise_control",
        # Wave-1038 medicine canon.
        "human_physiology",
        "pharmacokinetics",
        "immunology",
        "pathology",
        "neuroscience_med",
        "cardiology",
        # Wave-1037 agriculture canon.
        "crop_science",
        "soil_science",
        "agronomy",
        "animal_science",
        "horticulture",
        "pest_management",
        # Wave-1036 petroleum-engineering canon.
        "reservoir_engineering",
        "drilling_engineering",
        "production_engineering",
        "formation_evaluation",
        "well_testing",
        "enhanced_recovery",
        # Wave-1035 nuclear-engineering canon.
        "reactor_physics",
        "radiation_protection",
        "nuclear_fuel_cycle",
        "thermal_hydraulics",
        "nuclear_safety",
        "isotope_production",
        # Wave-1034 industrial-engineering canon.
        "operations_research",
        "supply_chain",
        "manufacturing_sys",
        "quality_control",
        "ergonomics",
        "facility_layout",
        # Wave-1033 biomedical-engineering canon.
        "biomechanics",
        "medical_devices",
        "tissue_engineering",
        "bioinstrumentation",
        "physiological_modeling",
        "biomedical_imaging2",
        # Wave-1032 aerospace-engineering canon.
        "aerodynamics",
        "propulsion",
        "orbital_mechanics2",
        "flight_dynamics",
        "spacecraft_design",
        "airfoil_theory",
        # Wave-1031 civil-engineering canon.
        "structural_analysis",
        "geotechnics",
        "transportation_eng",
        "water_resources",
        "construction_mgmt",
        "surveying",
        # Wave-1030 electrical-engineering canon.
        "circuit_analysis",
        "power_systems",
        "control_systems",
        "signal_processing2",
        "electromagnetics",
        "semiconductor",
        # Wave-1029 mechanical-engineering canon.
        "solid_mechanics",
        "vibration_analysis",
        "fatigue_life",
        "tribology",
        "machine_design",
        "kinematics",
        # Wave-1028 chemical-engineering canon.
        "reaction_kinetics",
        "thermo_props",
        "separation_proc",
        "heat_exchanger",
        "fluid_dynamics2",
        "process_control",
        # Wave-1027 materials-science canon.
        "crystal_structure",
        "polymer_physics",
        "metallurgy",
        "ceramics",
        "nanomaterials",
        "superconductivity",
        # Wave-1026 environmental-science canon.
        "climate_model",
        "ocean_circulation",
        "atmospheric_chem",
        "hydrology",
        "carbon_cycle",
        "ecosystem_model",
        # Wave-1025 social-science canon.
        "game_theory2",
        "behavioral_econ",
        "political_science",
        "sociology_net",
        "cognitive_science",
        "linguistics",
        # Wave-1024 computational-biology canon.
        "protein_folding",
        "dna_sequencing",
        "phylogenetics",
        "gene_expression",
        "metabolomics",
        "systems_biology",
        # Wave-1023 finance-theory canon.
        "capm_model",
        "arbitrage_pricing",
        "black_scholes",
        "yield_curve",
        "default_risk",
        "corporate_finance",
        # Wave-1022 economics canon.
        "growth_theory",
        "overlapping_gens",
        "real_business",
        "search_matching",
        "mechanism_design",
        "auction_theory2",
        # Wave-1021 epidemiology canon.
        "sir_epidemic",
        "sis_epidemic",
        "seir_epidemic",
        "r0_estimation",
        "herd_immunity",
        "branching_epidemic",
        # Wave-1020 ecology/evolution canon.
        "predator_prey",
        "lotka_volterra",
        "logistic_growth",
        "island_biogeography",
        "neutral_theory",
        "food_web",
        # Wave-1019 geophysics-3 canon.
        "seismic_waves",
        "earthquake_magnitude",
        "plate_tectonics",
        "gravity_anomaly",
        "geomagnetism",
        "heat_flow_geo",
        # Wave-1018 relativity-2 canon.
        "lorentz_transformation",
        "spacetime_interval",
        "four_vectors",
        "geodesic_motion",
        "gravitational_lensing",
        "gravitational_waves",
        # Wave-1017 continuum-mechanics canon.
        "navier_cauchy",
        "stress_tensor",
        "rheology",
        "viscoelasticity",
        "plasticity",
        "poroelasticity",
        # Wave-1016 optics-2 canon.
        "diffraction_grating",
        "fourier_optics",
        "interference_fringes",
        "polarization_states",
        "coherence_theory",
        "holography",
        # Wave-1015 acoustics canon.
        "acoustic_wave_eq",
        "helmholtz_eq",
        "sound_absorption",
        "room_acoustics",
        "rayleigh_scattering",
        "doppler_effect",
        # Wave-1014 astrophysics/cosmology canon.
        "jeans_instability",
        "stellar_structure",
        "stellar_evolution",
        "hubble_law",
        "cmb_anisotropy",
        "dark_matter",
        # Wave-1013 atomic/molecular-physics canon.
        "hartree_fock",
        "born_oppenheimer",
        "molecular_orbitals",
        "rotational_spectra",
        "vibrational_spectra",
        "zeeman_effect",
        # Wave-1012 nuclear/particle-physics canon.
        "bcs_theory",
        "nuclear_shell_model",
        "nuclear_liquid_drop",
        "quark_model",
        "parton_model",
        "cabibbo_km",
        # Wave-1011 condensed-matter canon.
        "bloch_theorem",
        "tight_binding",
        "phonon_spectrum",
        "band_structure",
        "hubbard_model",
        "kondo_effect",
        # Wave-1010 quantum-field-theory canon.
        "klein_gordon",
        "dirac_equation",
        "feynman_rules",
        "renormalization_group",
        "path_integral_qm",
        "canonical_quantization",
        # Wave-1009 electrodynamics/optics canon.
        "maxwell_equations",
        "poynting_vector",
        "fresnel_eq",
        "wave_guides",
        "dipole_radiation",
        "lorentz_lorenz",
        # Wave-1008 thermodynamics canon.
        "carnot_cycle",
        "maxwell_relations",
        "phase_transitions",
        "critical_phenomena",
        "fluctuation_dissipation",
        "entropy_production",
        # Wave-1007 statistical-mechanics canon.
        "ising_model",
        "partition_function",
        "bose_einstein",
        "fermi_dirac",
        "gibbs_measure",
        "free_energy",
        # Wave-1006 quantum-mechanics canon.
        "schrodinger_eq",
        "hydrogen_atom",
        "harmonic_oscillator",
        "spin_half",
        "wigner_wick",
        "fock_space",
        # Wave-1005 general-relativity canon.
        "einstein_equations",
        "schwarzschild_metric",
        "friedmann_eq",
        "kerr_metric",
        "gr_birkhoff",
        "penrose_diagrams",
        # Wave-1004 kinetic-theory canon.
        "boltzmann_eq",
        "vlasov_eq",
        "bgk_model",
        "chapman_enskog",
        "h_theorem",
        "landau_damping",
        # Wave-1003 MHD/plasma canon.
        "mhd_equations",
        "alfven_waves",
        "parker_solar_wind",
        "magnetic_reconnection",
        "frozen_flux",
        "elsaesser_vars",
        # Wave-1002 turbulence canon.
        "kolmogorov_theory",
        "reynolds_decomp",
        "energy_spectrum",
        "intermittency_models",
        "wall_turbulence",
        "taylor_series_hyp",
        # Wave-1001 fluid-dynamics canon.
        "euler_equations",
        "navier_stokes",
        "vorticity_form",
        "beale_kato_majda",
        "ladyzhenskaya_weak",
        "leray_theory",
        # Wave-1000 elasticity canon.
        "navier_elasticity",
        "kirchhoff_plate",
        "mindlin_reissner",
        "contact_mechanics",
        "fracture_mechanics",
        "homogenized_elasticity",
        # Wave-999 GMT-2 canon.
        "currents_theory",
        "varifold_theory",
        "flat_chains",
        "integral_currents",
        "rectifiable_measures",
        "brakke_varifolds",
        # Wave-998 singularity/blow-up canon.
        "semilinear_heat",
        "fujita_exponent",
        "singularity_formation",
        "matched_asymptotic_pde",
        "self_similar_blowup",
        "regularity_critical",
        # Wave-997 dispersive-PDE canon.
        "nls_dispersion",
        "kdv_dispersion",
        "strichartz_estimates",
        "local_smoothing",
        "bilinear_estimates",
        "i_method",
        # Wave-996 Riemann-Hilbert canon.
        "dbar_method",
        "orthogonal_poly_rh",
        "isomonodromy",
        "fokas_unified",
        "deift_zhou",
        "small_norm_rh",
        # Wave-995 integrable-systems canon.
        "sine_gordon",
        "nls_soliton",
        "toda_lattice",
        "calogero_moser",
        "kp_hierarchy",
        "painleve_eq",
        # Wave-994 inverse-spectral canon.
        "inverse_scattering",
        "marchenko_eq",
        "gelfand_levitan",
        "kdv_isospectral",
        "trace_formulas",
        "borg_levinson",
        # Wave-993 nonlinear-functional-analysis canon.
        "monotone_op",
        "degree_theory",
        "schauder_fixed",
        "krein_rutman",
        "minty_browder",
        "maximal_monotone",
        # Wave-992 scattering-theory canon.
        "wave_operators",
        "scattering_matrix",
        "limiting_absorption",
        "trace_class_scatt",
        "resonances_thy",
        "radiation_cond",
        # Wave-991 homogenization canon.
        "homogenization",
        "two_scale_conv",
        "gamma_convergence",
        "mosco_conv",
        "bloch_decomp",
        "h_convergence",
        # Wave-990 calculus-of-variations canon.
        "euler_lagrange",
        "legendre_cond",
        "jacobi_eq",
        "geodesic_var",
        "isoperimetric_var",
        "soap_film",
        # Wave-989 microlocal-2 canon.
        "parametrix",
        "wave_eq_group",
        "propagation_thm",
        "melrose_bdy",
        "fbi_transform",
        "sg_calculus",
        # Wave-988 spectral-geometry canon.
        "spectral_geometry",
        "heat_invariants",
        "weyl_law",
        "isospectral",
        "cheeger_ineq",
        "nodal_domain",
        # Wave-987 parabolic/Li-Yau canon.
        "parabolic_harnack",
        "gaussian_upper",
        "li_yau",
        "nash_ineq",
        "davies_gaffney",
        "grad_est",
        # Wave-986 Calderon-Zygmund canon.
        "calderon_zygmund",
        "cz_decomp",
        "cotlar_ineq",
        "good_lambda",
        "ap_weight",
        "reverse_holder",
        # Wave-985 Hardy-space/BMO canon.
        "hardy_h1",
        "bmo_space",
        "atomic_h1",
        "carleson_measure",
        "john_nirenberg",
        "fefferman_stein",
        # Wave-984 modulation-spaces canon.
        "modulation_space",
        "short_time_ft",
        "gabor_frame",
        "wigner_dist",
        "ambiguity_fn",
        "feichtinger_alg",
        # Wave-983 Besov/Triebel-Lizorkin canon.
        "besov_space",
        "triebel_lizorkin",
        "atoms_decomp",
        "wavelet_char",
        "besov_embed",
        "hardy_littlewood_max",
        # Wave-982 potential-theory-2 canon.
        "dirichlet_problem",
        "energy_principle",
        "equilibrium_measure",
        "thin_set",
        "boundary_regular",
        "capacitary_pot",
        # Wave-981 convex-geometry-2 canon.
        "john_ellipsoid",
        "loewner_ellipsoid",
        "milman_rev_thm",
        "grothendieck_const",
        "kadison_singer",
        "milman_isotropic",
        # Wave-980 approximation-theory-2 canon.
        "walsh_series",
        "haar_system",
        "faber_schauder",
        "de_boor_stable",
        "whitney_ext",
        "korovkin_thm",
        # Wave-979 ergodic-2 canon.
        "weyl_equidist",
        "van_der_corput",
        "horocycle_flow",
        "unipotent_ergodic",
        "ratner_thm",
        "disjointness_dyn",
        # Wave-978 harmonic-analysis-2 canon.
        "hausdorff_young",
        "restriction_est",
        "bochner_riesz",
        "lp_multiplier",
        "oscillatory_int",
        "strichartz_est",
        # Wave-977 Bochner/vector-valued canon.
        "bochner_integral",
        "lusin_rep",
        "radon_nikodym_prop",
        "bochner_meas",
        "norm_integrable",
        "pettis_weak",
        # Wave-976 nuclear-spaces canon.
        "nuclear_map",
        "frechet_nuclear",
        "gelfand_triple",
        "hilbert_schmidt_emb",
        "trace_duality",
        "diam_dim",
        # Wave-975 distribution-theory canon.
        "schwartz_dist",
        "temper_dist",
        "dist_convolution",
        "sing_support",
        "paley_wiener",
        "sobolev_trace",
        # Wave-974 interpolation-theory canon.
        "real_interp_k",
        "complex_interp",
        "lorentz_space",
        "marcinkiewicz_interp",
        "peetre_kfunctor",
        "reiteration_thm",
        # Wave-973 Banach-space-geometry canon.
        "banach_mazur",
        "type_cotype",
        "gl_property",
        "djt_space",
        "schauder_basis",
        "kalton_loc",
        # Wave-972 free-probability-2 canon.
        "free_entropy",
        "free_fisher_info",
        "free_cumulant",
        "freeness_check",
        "matrix_model_free",
        "free_berg",
        # Wave-971 noncommutative-geometry canon.
        "spectral_triple",
        "connes_metric",
        "index_pairing",
        "hochschild_cycle",
        "differential_form_nc",
        "geodesic_nc",
        # Wave-970 subfactor-theory canon.
        "subfactor",
        "standard_invariant",
        "planar_algebra",
        "paragroup",
        "principal_graph",
        "fusion_algebra",
        # Wave-969 operator K-theory canon.
        "k0_algebra",
        "k1_algebra",
        "bott_periodicity_k",
        "six_term_exact",
        "pimsner_voicul",
        "elliott_invariant",
        # Wave-968 KK-theory canon.
        "kk_theory",
        "kasparov_prod",
        "ext_functor",
        "baaj_julg",
        "cuntz_picture",
        "kk_duality",
        # Wave-967 C*-dynamics canon.
        "cstar_dynamics",
        "crossed_product",
        "rokhlin_action",
        "kirchberg_absorb",
        "taf_dim",
        "z_stability",
        # Wave-966 operator-space canon.
        "operator_space",
        "cb_map",
        "complete_contraction",
        "injective_space",
        "noncommutative_lp",
        "oh_emb",
        # Wave-965 Schatten/compact canon.
        "hilbert_schmidt_op",
        "trace_class_op",
        "singular_value_op",
        "schmidt_decomp",
        "compact_normal",
        "polar_operator",
        # Wave-964 unbounded-operator canon.
        "unbounded_operator",
        "closed_operator",
        "domain_dense",
        "adjoint_unbounded",
        "resolvent_op",
        "spectral_measure",
        # Wave-963 spectral-theory-2 canon.
        "fredholm_index",
        "weyl_theorem",
        "essential_spectrum",
        "browder_operator",
        "riesz_schauder",
        "atkinson_thm",
        # Wave-962 von-Neumann-algebra canon.
        "von_neumann_alg",
        "double_commutant",
        "predual_space",
        "normal_state",
        "tomita_takesaki",
        "jones_index",
        # Wave-961 semigroup-theory canon.
        "c0_semigroup",
        "hille_yosida",
        "lumer_phillips",
        "analytic_semigroup",
        "cosine_family",
        "trotter_kato",
        # Wave-960 banach-algebra canon.
        "banach_algebra",
        "gelfand_transform",
        "c_star_algebra",
        "spectrum_algebra",
        "holomorphic_calculus",
        "positive_functional",
        # Wave-959 operator-theory-3 canon.
        "toeplitz_op",
        "integral_op",
        "differential_op",
        "contraction_op",
        "accretive_op",
        "sectorial_op",
        # Wave-958 matrix-inequalities canon.
        "ky_fan",
        "lidskii_thm",
        "von_neumann_trace",
        "pinching_ineq",
        "araki_lieb_thirring",
        "hadamard_fischer",
        # Wave-957 operator-theory-2 canon.
        "selfadjoint_op",
        "unitary_operator",
        "shift_operator",
        "fredholm_op",
        "normal_operator",
        "multiplication_op",
        # Wave-956 tensor-theory canon.
        "tucker_rank",
        "cp_rank",
        "tensor_norm",
        "tensor_trace",
        "mode_n_product",
        "tensor_symmetry",
        # Wave-955 linear-systems canon.
        "gram_matrix",
        "gram_determinant",
        "householder_reflect",
        "givens_rotation",
        "back_substitution",
        "forward_substitution",
        # Wave-954 matrix-approximation canon.
        "low_rank_approx",
        "nuclear_norm",
        "spectral_threshold",
        "matrix_truncate",
        "rank_estimate",
        "condition_number",
        # Wave-953 operator-theory canon.
        "bounded_operator",
        "operator_norm",
        "operator_adjoint",
        "projection_operator",
        "positive_operator",
        "isometry_operator",
        # Wave-952 spectral-decomposition canon.
        "qr_iteration",
        "power_deflation",
        "schur_decomp",
        "eigval_bounds",
        "spectral_radius",
        "spectral_gap",
        # Wave-951 matrix-function canon.
        "determinant_cofactor",
        "permanent_matrix",
        "matrix_exponential",
        "frechet_derivative",
        "vec_operator",
        "kronecker_sum",
        # Wave-950 tensor-algebra canon.
        "tensor_contraction",
        "khatri_rao",
        "kron_product",
        "hadamard_product",
        "tensor_unfold",
        "outer_product",
        # Wave-949 matrix-pencil canon.
        "matrix_pencil",
        "kronecker_canonical",
        "invariant_subspace",
        "deflating_subspace",
        "jordan_form",
        "rational_canonical",
        # Wave-948 structured-matrix canon.
        "circulant_matrix",
        "companion_matrix",
        "vandermonde_matrix",
        "krylov_matrix",
        "hessenberg_form",
        "hankel_matrix",
        # Wave-947 spectral-interlacing canon.
        "cauchy_interlace",
        "sylvester_law",
        "haynsworth_inertia",
        "min_max_eig",
        "sturm_sequence",
        "bezout_matrix",
        # Wave-946 positive-matrix canon.
        "perron_frobenius",
        "douglas_factor",
        "cholesky_piv",
        "matrix_square_root",
        "polar_decomp",
        "sylvester_matrix",
        # Wave-945 matrix-norm canon.
        "kyfan_norm",
        "schatten_norm",
        "numerical_radius",
        "matrix_det",
        "pfaffian_poly",
        "hankel_op",
        # Wave-944 matrix-analysis-2 canon.
        "fan_inequality",
        "horn_inequality",
        "weyl_ineq",
        "cauchy_binet",
        "schur_complement",
        "majorization_vec",
        # Wave-943 matrix-analysis canon.
        "loewner_matrix",
        "operator_convex",
        "kadison_ineq",
        "wielandt_ineq",
        "ostrowski_bound",
        "gershgorin_disc",
        # Wave-942 primal-dual canon.
        "primal_dual_hybrid",
        "vu_condat",
        "backward_forward",
        "malitsky_golden",
        "mann_iter",
        "ishikawa_iter",
        # Wave-941 nonsmooth-Newton canon.
        "limiting_subdiff",
        "proximal_subdiff",
        "ekeland_var",
        "monteiro_semismooth",
        "semismooth_newton",
        "augmented_lagr",
        # Wave-940 variational-inequality canon.
        "subgradient_extragradient",
        "korpelevich_eg",
        "popov_alg",
        "tseng_fb",
        "forward_reflected",
        "reflected_golden",
        # Wave-939 set-feasibility canon.
        "split_feasibility",
        "cq_algorithm",
        "dykstra_proj",
        "haugazeau_proj",
        "parallel_prox",
        "halpern_iter",
        # Wave-938 fixed-point canon.
        "fejer_monotone",
        "firmly_nonexpansive",
        "averaged_operator",
        "cocoercive",
        "quasinonexpansive",
        "monotone_inclusion",
        # Wave-937 operator-splitting canon.
        "douglas_rachford",
        "peaceman_rachford",
        "tseng_split",
        "forward_backward",
        "chambolle_pock",
        "davis_yin",
        # Wave-936 nonsmooth-analysis canon.
        "subdiff_compute",
        "epigraph_proj",
        "gauge_fn",
        "gauge_duality",
        "bundle_level",
        "clarke_subdiff",
        # Wave-935 convex-optimization canon.
        "kkt_solve",
        "cvx_reform",
        "self_concordant",
        "logbarrier_fn",
        "analytic_center",
        "dik_ellipsoid",
        # Wave-934 convex-analysis-2 canon.
        "inf_convolution",
        "legendre_transform",
        "support_fn",
        "perspective_fn",
        "polar_cone",
        "normal_cone",
        # Wave-933 convex-analysis canon.
        "subgradient_proj",
        "proximal_map",
        "fenchel_dual",
        "moreau_env",
        "bregman_proj",
        "conjugate_fn",
        # Wave-932 local-search-3 canon.
        "vns_search",
        "large_neighborhood",
        "ruin_recreate",
        "path_relinking",
        "guided_local",
        "simulated_annealing",
        # Wave-931 population-metaheuristics canon.
        "ant_colony",
        "pso_swarm",
        "diff_evolution",
        "genetic_tsp",
        "firefly_algo",
        "harmony_search",
        # Wave-930 metaheuristics canon.
        "lin_kernighan",
        "two_opt_move",
        "three_opt_move",
        "tabu_search",
        "iterated_local",
        "grasp_meta",
        # Wave-929 information-geometry-4 canon.
        "mahalanobis_div",
        "bhat_distance",
        "hellinger_dist",
        "jeffreys_div",
        "d_total_var",
        "chi_square_div",
        # Wave-928 computational-geometry-6 canon.
        "convex_hull_3d",
        "polygon_boolean",
        "medial_axis",
        "polygon_centroid",
        "shape_context",
        "beta_skeleton",
        # Wave-927 information-geometry-3 canon.
        "fisher_metric2",
        "expectation_param",
        "potential_fn",
        "ebanch_diverge",
        "shannon_gibbs",
        "renyi_div",
        # Wave-926 information-geometry-2 canon.
        "f_divergence",
        "alpha_divergence",
        "csiszar_div",
        "amari_connection",
        "dual_connection",
        "tsallis_entropy",
        # Wave-925 Bayesian-nonparametrics-3 canon.
        "exchangeable_pf",
        "normalized_rm",
        "sigma_stable",
        "nggp_process",
        "bondesson_shot",
        "kingman_paintbox",
        # Wave-924 Bayesian-nonparametrics-2 canon.
        "gem_distribution",
        "dp_mm",
        "crp_table",
        "beta_bernoulli",
        "neutral_process",
        "gibbs_type",
        # Wave-923 Bayesian-nonparametrics canon.
        "dirichlet_process",
        "stick_breaking",
        "pitman_yor",
        "indian_buffet",
        "chinese_restaurant",
        "hierarchical_dp",
        # Wave-922 distributed-systems-6 canon.
        "abcast_lite",
        "cbc_bcast",
        "slush_consensus",
        "snowflake_consensus",
        "cap_theorem",
        "lake_wisc",
        # Wave-921 distributed-systems-5 canon.
        "virtual_synchrony",
        "isis_bcast",
        "atomic_bcast",
        "honey_badger",
        "avalanche_consensus",
        "snowball_consensus",
        # Wave-920 computational-geometry-5 canon.
        "monotone_partition",
        "polygon_triangulate",
        "min_area_rect",
        "diameter_pair",
        "alpha_shape",
        "minkowski_sum_poly",
        # Wave-919 computational-geometry-4 canon.
        "voronoi_lite",
        "delaunay_flip",
        "convex_layers",
        "polygon_offset",
        "rotating_sweep",
        "visibility_graph",
        # Wave-918 data-structures-4/geometry-3 canon.
        "fractional_cascade",
        "range_min_query",
        "free_list",
        "object_pool",
        "welzl_circle",
        "halfplane_isect",
        # Wave-917 data-structures-3 canon.
        "chained_hash",
        "linear_probe",
        "bucket_sort",
        "shell_sort",
        "rand_access_list",
        "skew_list",
        # Wave-916 data-structures-2 canon.
        "soft_heap",
        "hollow_heap",
        "rank_pairing",
        "hollow_dsu",
        "da_trie",
        "fst_index",
        # Wave-915 BVP/tree-exotics canon.
        "superposition_bvp",
        "continuation_bvp",
        "robbins_bvp",
        "bvp_eigen",
        "loser_tree",
        "fusion_tree",
        # Wave-914 RK/BVP-methods-2 canon.
        "ralston_rk",
        "verner_rk",
        "ralston_second",
        "runge_kutta4",
        "invariant_imbedding",
        "green_function_bvp",
        # Wave-913 interpolation-3 canon.
        "scattered_interp",
        "spline_interp",
        "monotone_interp",
        "akima_interp",
        "pchip_interp",
        "makima_interp",
        # Wave-912 spatial-index-2 canon.
        "octree_index",
        "range_tree",
        "hilbert_curve",
        "z_curve",
        "morton_order",
        "rstar_tree",
        # Wave-911 computational-geometry-2 canon.
        "monotone_chain",
        "gift_wrap",
        "chan_hull",
        "liang_barsky",
        "cohen_sutherland",
        "bezier_eval",
        # Wave-910 b-tree family canon.
        "b_tree",
        "b_plus_tree",
        "b_star_tree",
        "weight_balanced_tree",
        "wavl_tree",
        "tango_tree",
        # Wave-909 deque/linked-structure canon.
        "doubly_linked_list",
        "unrolled_list",
        "gap_buffer",
        "piece_table",
        "deque_array",
        "xor_linked_list",
        # Wave-908 range-query canon.
        "segment_tree",
        "fenwick_tree",
        "sparse_table",
        "sqrt_decomp",
        "wavelet_tree",
        "merge_sort_tree",
        # Wave-907 union-find + priority-queue canon.
        "union_find",
        "dsu_rollback",
        "potential_dsu",
        "van_emde_boas",
        "interval_heap",
        "weak_heap",
        # Wave-906 persistent-structure canon.
        "skip_list",
        "persistent_array",
        "finger_tree",
        "rope_string",
        "vlist",
        "pure_queue",
        # Wave-905 sorting canon.
        "quicksort",
        "mergesort",
        "heapsort",
        "introsort",
        "timsort",
        "radix_sort",
        # Wave-904 balanced-tree canon.
        "avl_tree",
        "red_black_tree",
        "splay_tree",
        "treap",
        "scapegoat_tree",
        "aa_tree",
        # Wave-903 hash-table canon.
        "cuckoo_hash",
        "hopscotch_hash",
        "robin_hood_hash",
        "swiss_table",
        "open_addr_hash",
        "perfect_hash",
        # Wave-902 trie/string-index canon.
        "trie",
        "patricia_trie",
        "suffix_trie",
        "ternary_trie",
        "radix_trie",
        "crit_bit_tree",
        # Wave-901 heap canon.
        "binary_heap",
        "fibonacci_heap",
        "pairing_heap",
        "binomial_heap",
        "leftist_heap",
        "skew_heap",
        # Wave-900 spatial-index canon.
        "kd_tree",
        "ball_tree",
        "cover_tree",
        "r_tree",
        "quad_tree",
        "vp_tree",
        # Wave-899 interpolation canon.
        "cardinal_interp",
        "bernstein_form",
        "shanks_trans",
        "chebyshev_interp",
        "osculating_interp",
        "rational_interp",
        # Wave-898 BVP canon.
        "shooting_bvp",
        "multiple_shooting",
        "collocation_bvp",
        "finite_diff_bvp",
        "relaxation_bvp",
        "riccati_bvp",
        # Wave-897 ODE-theory canon.
        "linear_multistep",
        "dahlquist_test",
        "explicit_midpoint",
        "heun_method",
        "trapezoid_rule",
        "order_barrier",
        # Wave-896 RK/IVP canon.
        "fehlberg_rk",
        "dormand_prince",
        "cash_karp",
        "bogacki_shampine",
        "backward_euler",
        "predictor_corrector",
        # Wave-895 root-finding canon.
        "secant_root",
        "regula_falsi",
        "muller_root",
        "aitken_steffensen",
        "richardson_limit",
        "bulirsch_stoer",
        # Wave-894 interpolation canon.
        "lagrange_interp",
        "neville_interp",
        "hermite_interp",
        "divid_diff_table",
        "barycentric_wts",
        "floater_hormann",
        # Wave-893 collocation canon.
        "covello_est",
        "dual_goal_est",
        "pseudospectral_coll",
        "tau_method",
        "galerkin_least_sq",
        "coarsening_mark",
        # Wave-892 adaptive-mesh canon.
        "space_time_adapt",
        "greedy_marking",
        "form_analysis",
        "hp_adaptive",
        "wavelet_adapt",
        "residual_marking",
        # Wave-891 optimization/IGA canon.
        "trust_region_dogleg",
        "bfgs_update",
        "lebesgue_const",
        "iga_colloc",
        "trimmed_cad",
        "newton_armijo",
        # Wave-890 special-function canon.
        "zeta_fn",
        "elliptic_fn",
        "hartley_transform",
        "radon_transform",
        "clenshaw_quad",
        "fejer_nested",
        # Wave-889 FE-basis/sequence canon.
        "quadrilateral_basis",
        "hexahedral_basis",
        "chebyshev_u",
        "walsh_table",
        "epsilon_algo",
        "spline_theory",
        # Wave-888 RBF/basis canon.
        "thin_plate_spline",
        "polyharmonic_rbf",
        "trefethen_diff",
        "galerkin_projection",
        "periodic_spline",
        "zernike_poly",
        # Wave-887 solver/transport canon.
        "epi_rk",
        "gauss_rk",
        "adjoint_sparse",
        "element_free",
        "diffusion_approx_sp",
        "importance_rel",
        # Wave-886 QMC/tensor canon.
        "faure_seq",
        "importance_mc",
        "gauss_hermite",
        "gauss_laguerre",
        "tensor_train",
        "hiot_decomp",
        # Wave-885 DG-flux/asymptotics canon.
        "ldg_flux",
        "entropy_stable_dg",
        "wkb_turning",
        "averaging_method",
        "laplace_method",
        "hyperasymptotic",
        # Wave-884 wavelet-2/spectral-elem canon.
        "wavelet_matrix",
        "second_gen_wavelet",
        "nodal_dg",
        "multidomain_sem",
        "boundary_element",
        "marquina_flux",
        # Wave-883 fixed-point-acceleration canon.
        "tangent_predictor",
        "is_drift",
        "cv_optimal",
        "nest_accel",
        "subgradient_descent",
        "min_var_closure",
        # Wave-882 preconditioner/domain-decomposition canon.
        "spai_precond",
        "diagonal_scale",
        "nonoverlap_dd",
        "overlap_dd",
        "restrictive_dd",
        "balanced_dd",
        # Wave-881 Krylov-solver canon.
        "minres_solver",
        "cgs_solver",
        "tfqmr",
        "qmr_solver",
        "bicgstab2",
        "block_cg",
        # Wave-880 quadrature/cubature canon.
        "gq_adaptive",
        "adaptive_quad2",
        "pod_deim",
        "empirical_interp",
        "cubature_rule",
        "tensor_interp",
        # Wave-879 exponential-time-integrator canon.
        "expm_int",
        "expokit",
        "krylov_subspace_time",
        "leja_point",
        "phi_function",
        "etd_rk4_classic",
        # Wave-878 low-rank/hierarchical-matrix canon.
        "low_rank_svd",
        "h_matrix",
        "hss_matrix",
        "randomized_nystrom",
        "block_low_rank",
        "kronecker_approx",
        # Wave-877 transport/SPn canon.
        "transport_sn",
        "discrete_ordinates",
        "spherical_harmonics",
        "spn_equations",
        "moc_transport",
        "pn_closure",
        # Wave-876 nonlinear-solver canon.
        "moore_penrose",
        "landweber_iter",
        "conjugate_grad_ls",
        "gauss_newton",
        "levenberg_marq",
        "anderson_mixing",
        # Wave-875 adaptive-marking canon.
        "adaptive_marking",
        "hierarchical_est",
        "dorfler_marking",
        "convergence_theory",
        "adaptive_finite",
        "goal_adaptive",
        # Wave-874 reliability-analysis canon.
        "uq_reliability",
        "first_order_rel",
        "sorm_method",
        "subset_sim",
        "line_sampling",
        "metamodel_rel",
        # Wave-873 domain-decomposition canon.
        "dd_partition",
        "baldding_dd",
        "neumann_dd",
        "feti_dp",
        "subspace_dd",
        "asm_precond",
        # Wave-872 preconditioner canon.
        "jacobi_precond",
        "ilut_precond",
        "ssor_precond",
        "amg_precond",
        "ic_precond",
        "polynomial_precond",
        # Wave-871 Krylov-solver canon.
        "cg_solver",
        "gmres_solver",
        "bicg_solver",
        "arnoldi_eig",
        "lanczos_eig",
        "lsqr_solver",
        # Wave-870 optimization canon.
        "newton_method",
        "quasi_newton_lbfgs",
        "augmented_lagrangian",
        "interior_point2",
        "grad_descent_nest",
        "conjugate_opt",
        # Wave-869 MC-variance-reduction canon.
        "antithetic_var",
        "control_variate",
        "importance_sampling",
        "stratified_var",
        "common_random",
        "conditional_mc",
        # Wave-868 continuation/homotopy canon.
        "arc_continuation",
        "pseudo_arclength",
        "deflation_method",
        "bifurcation_track",
        "homotopy_solver",
        "davidenko_ode",
        # Wave-867 inverse-problem canon.
        "tikhonov_reg",
        "morozov_dp",
        "l_curve_opt",
        "iter_regularize",
        "tv_denoise",
        "bayes_inverse",
        # Wave-866 model-order-reduction canon.
        "pod_galerkin",
        "reduced_basis",
        "deim_point",
        "greedy_rb",
        "eim_interp",
        "proper_gen",
        # Wave-865 a-posteriori error-estimation canon.
        "residual_estimator",
        "zienkiewicz_zhu",
        "recovery_error",
        "dual_weighted_res",
        "goal_oriented",
        "equilibrated_flux",
        # Wave-864 stochastic-Galerkin/UQ canon.
        "stochastic_galerkin",
        "poly_chaos_uq",
        "intrusive_pce",
        "nonintrusive_pce",
        "stochastic_colloc",
        "stochastic_fem",
        # Wave-863 domain-decomposition canon.
        "schwarz_add",
        "schwarz_mult",
        "coarse_correction",
        "mortar_dd",
        "feti_lite",
        "bddc_lite",
        # Wave-862 sparse-grid/dimension-adaptive canon.
        "smolyak_grid",
        "sparse_tensor",
        "anisotropic_quad",
        "gerstner_griebel",
        "combination_technique",
        "dimension_adaptive",
        # Wave-861 time-marching/ODE canon.
        "imex_rk",
        "ssp_rk",
        "exponential_euler",
        "rosenbrock_w",
        "ars_imex",
        "dirk_scheme",
        # Wave-860 isogeometric/immersed-methods canon.
        "iso_geom",
        "nurbs_elem",
        "xfem",
        "immersed_boundary",
        "cut_cell",
        "fictitious_domain",
        # Wave-859 meshfree/moving-least-squares canon.
        "moving_least_sq",
        "mls_shape",
        "hp_clouds",
        "meshless_local",
        "point_cloud_interp",
        "diffuse_element",
        # Wave-858 adaptive/oscillatory-quadrature canon.
        "adaptive_simpsons",
        "tanh_sinh",
        "double_exp_quad",
        "osc_singular",
        "filon_quad",
        "levin_quad",
        # Wave-857 classical-quadrature canon.
        "gauss_legendre",
        "gauss_chebyshev",
        "clenshaw_curtis",
        "newton_cotes",
        "gauss_kronrod",
        "fejer_quad",
        # Wave-856 quadrature/quasi-MC canon.
        "monte_carlo_quad",
        "quasi_mc",
        "halton_seq",
        "sobol_seq",
        "latin_hypercube",
        "stratified_mc",
        # Wave-855 wavelet-Galerkin canon.
        "wavelet_galerkin",
        "daubechies_basis",
        "coiflet_basis",
        "spline_wavelet",
        "wavelet_collocation",
        "adapt_wavelet",
        # Wave-854 radial-basis-function canon.
        "rbf_interp",
        "gaussian_rbf",
        "multiquadric_rbf",
        "kansa_collocation",
        "rbf_finite_diff",
        "wendland_rbf",
        # Wave-853 spectral-element canon.
        "sem_grid",
        "gll_nodes",
        "spectral_element",
        "mortar_method",
        "tensor_product_sem",
        "hp_refinement",
        # Wave-852 boundary-element canon.
        "bem_kernel",
        "fredholm_solve",
        "nystrom_method",
        "singular_integrals",
        "fast_multipole",
        "galerkin_bem",
        # Wave-851 discontinuous-Galerkin canon.
        "dg_discretization",
        "numerical_flux_dg",
        "penalty_dg",
        "modal_basis",
        "limiter_tvb",
        "rkdg_step",
        # Wave-850 Riemann-solver canon.
        "roe_solver",
        "hllc_solver",
        "ausm_flux",
        "lax_friedrichs",
        "godunov_exact",
        "osher_solver",
        # Wave-849 finite-volume canon.
        "fdm_grid",
        "compact_scheme",
        "crank_nicholson2",
        "upwind_scheme",
        "muscl_reconstruct",
        "flux_splitting",
        # Wave-848 finite-element canon.
        "fem_assembly",
        "isoparametric_map",
        "quadrature_rules",
        "triangular_basis",
        "edge_elements",
        "dof_management",
        # Wave-847 perturbation-theory canon.
        "regular_perturbation",
        "singular_perturbation",
        "matched_asymptotic",
        "multiple_scales",
        "lindstedt_poincare",
        "boundary_layer",
        # Wave-846 special-functions canon.
        "gamma_fn",
        "beta_fn",
        "bessel_fn",
        "airy_fn",
        "error_fn",
        "hypergeometric_fn",
        # Wave-845 integral-transforms canon.
        "laplace_transform",
        "mellin_transform",
        "hankel_transform",
        "z_transform",
        "hilbert_transform",
        "abel_transform",
        # Wave-844 asymptotic-analysis canon.
        "asymptotic_series",
        "poincare_expansion",
        "steepest_descent",
        "stationary_phase",
        "borel_resum",
        "wkb_approx",
        # Wave-843 spline-theory canon.
        "b_spline",
        "de_boor",
        "cardinal_spline",
        "knot_insertion",
        "blossoming",
        "box_spline",
        # Wave-842 spectral-methods canon.
        "chebyshev_grid",
        "fourier_galerkin",
        "legendre_tau",
        "chebyshev_collocation",
        "spectral_deriv",
        "dealiasing",
        # Wave-841 orthogonal-polynomial canon.
        "legendre_poly",
        "chebyshev_t",
        "hermite_poly",
        "laguerre_poly",
        "jacobi_poly",
        "gegenbauer_poly",
        # Wave-840 rational-approximation canon.
        "pade_approx",
        "rational_chebyshev",
        "stieltjes_fraction",
        "loewner_interp",
        "nevanlinna_pick",
        "schur_continued",
        # Wave-839 approximation-theory canon.
        "jackson_direct",
        "chebyshev_alternation",
        "kolmogorov_nwidth",
        "bernstein_poly",
        "markov_brothers",
        "fourier_decay",
        # Wave-838 discrete-geometry canon.
        "radon_theorem",
        "caratheodory_thm",
        "farkas_lemma",
        "separation_thm",
        "lattice_point",
        "tverberg_thm",
        # Wave-837 convex-geometry canon.
        "brunn_minkowski",
        "alexandrov_fenchel",
        "isoperimetric_ineq",
        "minkowski_sum",
        "mixed_volume",
        "helly_theorem",
        # Wave-836 integral-geometry canon.
        "crofton_formula",
        "kinematic_measure",
        "buffon_needle",
        "santalo_measure",
        "kubota_mean_width",
        "hadwiger_chars",
        # Wave-835 stochastic-geometry canon.
        "poisson_voronoi",
        "boolean_model",
        "germ_grain",
        "steiner_formula",
        "miles_matheron",
        "intrinsic_volumes",
        # Wave-834 Markov-semigroup canon.
        "dirichlet_form",
        "markov_semigroup",
        "poincare_semigroup",
        "log_sobolev_sem",
        "hypercontractive",
        "spectral_gap_sem",
        # Wave-833 functional-limit-theory canon.
        "fclt_invariance",
        "donsker_invariance",
        "martingale_fclt",
        "stable_limit",
        "brownian_approx",
        "strassen_flln",
        # Wave-832 weak-convergence-2 canon.
        "porte_manteau",
        "continuous_map",
        "delta_method",
        "skorohod_embed",
        "kmt_approx",
        "empirical_bridge",
        # Wave-831 empirical-process-3 canon.
        "entropy_integral",
        "uniform_clt",
        "symmetrization",
        "rademacher_cplx",
        "covering_number",
        "metric_entropy",
        # Wave-830 concentration-of-measure canon.
        "azuma_ineq",
        "mcdiarmid_ineq",
        "talagrand_ineq",
        "efron_stein",
        "bounded_diff",
        "hoeffding_ineq",
        # Wave-829 continuous-martingale canon.
        "local_time_process",
        "bounded_mart",
        "fv_mart",
        "cadlag_mart",
        "locator_proc",
        "decomp_mart",
        # Wave-828 projection/section canon.
        "projection_theorem",
        "uniform_section",
        "dellacherie_section",
        "cross_section",
        "maharam_lift",
        "von_neumann_sel",
        # Wave-827 random-series canon.
        "three_series",
        "kolmogorov_two",
        "ito_nisio",
        "chung_series",
        "ortega_series",
        "salem_zygmund",
        # Wave-826 maximal-inequality canon.
        "doob_ineq",
        "max_ineq",
        "bj_ineq",
        "kolmogorov_ineq",
        "etemadi_ineq",
        "levy_ineq",
        # Wave-825 measurable-selection canon.
        "measur_select",
        "kura_ryll",
        "castaing_rep",
        "measurable_graph",
        "integrand_map",
        "stoch_open",
        # Wave-824 stochastic-order canon.
        "usual_stoch_order",
        "first_order_dom",
        "second_order_dom",
        "convex_order",
        "hazard_rate_order",
        "supermodular_order",
        # Wave-823 law-of-process canon.
        "support_law",
        "polish_law",
        "tight_law",
        "law_convergence",
        "finite_dim",
        "cylindrical_law",
        # Wave-822 filtration canon.
        "natural_filtration",
        "right_continuous_f",
        "usual_aug",
        "enlargement_f",
        "initial_enlarg",
        "progressive_enlarg",
        # Wave-821 random-measure canon.
        "random_measure",
        "integer_measure",
        "poisson_rm",
        "compensator_rm",
        "jump_measure",
        "sato_measure",
        # Wave-820 semimartingale-decomp canon.
        "doom_decomp",
        "pcdt",
        "special_sem",
        "canonical_decomp",
        "sem_loc_char",
        "triplet_char",
        # Wave-819 stopping-time canon.
        "first_hitting",
        "last_exit",
        "stopping_sigma",
        "progressive_set",
        "debuts_theorem",
        "accessible_time",
        # Wave-818 stochastic-integration canon.
        "ito_integral",
        "mart_meas",
        "vector_mart",
        "bounded_var",
        "stochastic_int2",
        "covariation",
        # Wave-817 Levy-fluctuation canon.
        "wiener_hopf_f",
        "ladder_height",
        "renewal_measure",
        "overshoot_levy",
        "levy_fluct",
        "spitzer_levy",
        # Wave-816 Markov-process canon.
        "hunt_process",
        "cadlag_markov",
        "transition_semigroup",
        "resolvent_markov",
        "generator_markov",
        "characteristic_markov",
        # Wave-815 excursion-theory canon.
        "excursion_proc",
        "inverse_local",
        "ray_knight",
        "knight_theorem",
        "mazza_yor",
        "pitman_thm",
        # Wave-814 stochastic-flow canon.
        "stochastic_flow",
        "kunita_flow",
        "liouville_flow",
        "stochastic_damping",
        "meyers_process",
        "karal_flow",
        # Wave-813 diffusion-theory canon.
        "feller_boundary",
        "scale_measure",
        "speed_measure",
        "diffusion_semigroup",
        "yosida_op",
        "kreyn_resolvent",
        # Wave-812 regenerative canon.
        "regenerative",
        "epsilon_coupling",
        "small_set",
        "petite_set",
        "split_chain",
        "nummelin",
        # Wave-811 point-process canon.
        "cambrian_pp",
        "papangelou",
        "gneding_metric",
        "void_prob",
        "j_function",
        "ergodic_pp",
        # Wave-809 Markov-chain canon.
        "doeblin_coupling",
        "harris_recurrent",
        "ergodic_markov",
        "mixing_time",
        "drift_lyapunov",
        "cutoff_phenomenon",
        # Wave-808 filtering canon.
        "zakai_eq",
        "kushner_strat",
        "kalman_bucy",
        "bene_filter",
        "hidden_markov_filter",
        "particle_filter2",
        # Wave-807 optimal-stopping canon.
        "snell_envelope",
        "secretary_dp",
        "cayley_moser",
        "chow_robbins",
        "markov_stopping",
        "free_boundary",
        # Wave-806 jump-process canon.
        "jump_diffusion",
        "merton_jump",
        "kou_model",
        "compound_poisson",
        "excursion_theory",
        "marked_hawkes",
        # Wave-805 stochastic-calculus canon.
        "ito_isometry",
        "stratonovich_conv",
        "tanaka_meyer",
        "follmer_strat",
        "skorohod_lemma",
        "doss_sussmann",
        # Wave-804 Malliavin canon.
        "nualart_zakai",
        "watanabe_map",
        "malliavin_cov",
        "density_bound",
        "absolute_cont",
        "smoothness_h",
        # Wave-803 signature canon.
        "signature_kernel",
        "pde_signature",
        "truncated_sig",
        "signature_gan2",
        "expected_sig",
        "sig_inversion",
        # Wave-802 neural-SDE canon.
        "latent_sde",
        "neural_cde",
        "neural_rde",
        "sde_gan",
        "sde_matching",
        "logsig_rde",
        # Wave-801 rough-volatility canon.
        "fractional_heston",
        "rough_bergomi",
        "rough_sabr",
        "rough_variance",
        "volterra_sde",
        "multifactor_rough",
        # Wave-800 stochastic-vol canon.
        "heston_model",
        "bates_model",
        "rough_heston",
        "sabr_model",
        "three_two_vol",
        "scott_vol",
        # Wave-799 path-PDE canon.
        "path_dependent_pde",
        "functional_ito",
        "dupire_functional",
        "viscosity_path",
        "path_sobolev",
        "kolmogorov_path",
        # Wave-798 forward-SDE canon.
        "forward_sde",
        "random_sde",
        "anticipating_sde",
        "functional_sde",
        "delayed_sde",
        "neutral_sde",
        # Wave-797 2BSDE canon.
        "second_bsde",
        "doubly_bsde",
        "reflected_bsde2",
        "obstacle_bsde",
        "quadratic_bsde",
        "super_linear",
        # Wave-796 FBSDE-2 canon.
        "four_step_scheme",
        "decoupling_field2",
        "quasi_bsde",
        "coupled_fbsde",
        "random_bsde",
        "time_bsde",
        # Wave-795 stochastic-games canon.
        "dynkin_game",
        "stochastic_game2",
        "differential_game",
        "zero_sum_game",
        "nonzero_sum_game",
        "isaacs_equation",
        # Wave-794 stochastic-control canon.
        "dynamic_programming",
        "verification_thm",
        "hamilton_jacobi",
        "viscosity_solution",
        "quasi_variational",
        "impulsive_control",
        # Wave-793 stochastic-expansion canon.
        "wong_zakai",
        "stochastic_taylor",
        "milstein_scheme",
        "wagner_platen",
        "cubature_wiener",
        "rough_vol2",
        # Wave-792 BSDE canon.
        "bsde_solver",
        "fbsde_markov",
        "backward_sde",
        "pardoux_peng",
        "reflected_bsde",
        "second_order_bsde",
        # Wave-791 SPDE canon.
        "spde_heat",
        "stochastic_burgers",
        "kpz_equation",
        "doering_mueller",
        "quasilinear_spde",
        "paracontrolled_spde",
        # Wave-790 McKean-Vlasov canon.
        "mckean_vlasov",
        "mean_field_game2",
        "propagation_chaos",
        "kac_theorem",
        "nonlinear_markov",
        "self_stabilizing",
        # Wave-789 optimal-transport canon.
        "wasserstein_grad",
        "jko_step",
        "benamou_brenier",
        "entropy_regular",
        "fokker_planck2",
        "gradient_flow",
        # Wave-788 Malliavin-calculus canon.
        "clark_ocone",
        "nualart_pardoux",
        "divergence_op",
        "wiener_chaos",
        "skorohod_int",
        "nourdin_peccati",
        # Wave-787 regularity-structure canon.
        "ito_signature",
        "lyons_extension",
        "tame_map",
        "step_signature",
        "gubinelli_sewing",
        "young_integral",
        # Wave-786 rough-path canon.
        "rough_path",
        "signature_transform2",
        "controlled_path",
        "lyons_lift",
        "hairspring_map",
        "area_mart",
        # Wave-785 semimartingale-2 canon.
        "usual_cond",
        "dolean_mart",
        "strong_sol",
        "local_mart2",
        "follmer_mart",
        "protter_ito",
        # Wave-784 Levy canon.
        "levy_khinchine",
        "subordinator",
        "stable_levy",
        "self_decomp",
        "levy_measure",
        "girsanov_thm2",
        # Wave-783 stochastic-order canon.
        "semi_mart",
        "predictable_bracket",
        "cramer_wold",
        "stricker_thm",
        "likelihood_order",
        "hazard_order",
        # Wave-782 filtration/Jacod-Shiryaev canon.
        "pinsky_proc",
        "ffusion_lims",
        "kunita_watanabe",
        "filt_proc",
        "slivnyak",
        "jacod_shiryaev",
        # Wave-781 regeneration/Khinchin canon.
        "karlin_mcg",
        "keilson_stieltjes",
        "palm_khinchin",
        "regen_proc",
        "wold_proc",
        "korolyuk",
        # Wave-780 queueing-network canon.
        "bcmp_net",
        "mean_value",
        "convoy_net",
        "insensitive_thm",
        "kaufman_roberts",
        "orku_loss",
        # Wave-779 matrix-analytic canon.
        "neuts_map",
        "phase_type",
        "matrix_geom",
        "quasi_birth",
        "ramaswami",
        "logarithmic_red",
        # Wave-778 loss/vacation-queue canon.
        "engset",
        "erlang_b",
        "erlang_c",
        "pollaczek_khinchine",
        "borel_tanner",
        "takacs_vacation",
        # Wave-777 heavy-traffic canon.
        "fluid_limit",
        "heavy_traffic",
        "diffusion_approx",
        "kingman_bound",
        "halfin_whitt",
        "qed_regime",
        # Wave-776 point-process canon.
        "cox_process",
        "hawkes_point",
        "self_excite",
        "marked_point",
        "campbell_thm",
        "palm_dist",
        # Wave-775 martingale-theory canon.
        "doleans_meas",
        "predictable_proc",
        "local_mart",
        "square_bracket",
        "burkholder_davis",
        "gundy_mart",
        # Wave-774 random-walk canon.
        "sparc_rw",
        "spitzer_rw",
        "fluctuation_rw",
        "ladder_epoch",
        "wiener_hopf_rw",
        "maxwell_rw",
        # Wave-773 queueing canon.
        "mm1_queue",
        "mg1_queue",
        "gm_queue",
        "bulk_queue",
        "retrial_queue",
        "priority_queue",
        # Wave-772 branching-process canon.
        "galton_watson",
        "branching_imm",
        "multi_type_branch",
        "crump_mode",
        "kimmel_branch",
        "sevastyanov",
        # Wave-771 copula canon.
        "copula_gauss",
        "copula_t",
        "clayton_copula",
        "gumbel_copula",
        "frank_copula",
        "joe_copula",
        # Wave-770 extreme-value canon.
        "gumbel_domain",
        "weibull_domain",
        "frechet_domain",
        "peak_over",
        "hill_est",
        "pickands_est",
        # Wave-769 Stein-method canon.
        "stein_method",
        "stein_equation",
        "barbour_stein",
        "chen_stein",
        "ross_stein",
        "chatt_stein",
        # Wave-768 large-deviation canon.
        "varadhan_ldp",
        "freidlin_wentzell",
        "dw_ldp",
        "sanov_thm",
        "mogulskii_thm",
        "schider_thm",
        # Wave-767 Gaussian-process canon.
        "slepian_lemma",
        "fernique_thm",
        "borell_tis",
        "sudakov_min",
        "talagrand_conc",
        "gordon_thm",
        # Wave-766 empirical-process-2 canon.
        "dudley_theorem",
        "varadarajan_thm",
        "dvoretzky_thm",
        "vc_class",
        "bracketing_ent",
        "bounded_lip",
        # Wave-765 LIL/LLN canon.
        "strassen_lil",
        "chung_lil",
        "kolmogorov_3series",
        "khintchine_lln",
        "levy_convergence",
        "glivenko_cantelli",
        # Wave-764 empirical-process canon.
        "wiener_measure",
        "dz_invariance",
        "donsker_thm",
        "empirical_process",
        "donsker_class",
        "osj_metric",
        # Wave-763 mixing/urn canon.
        "polya_urn",
        "hopf_chain",
        "boneschi_boal",
        "bradley_mixing",
        "rosenthal_mom",
        "ibagimov_mixing",
        # Wave-762 weak-convergence canon.
        "martin_boundary",
        "doob_meyer",
        "cadlag_space",
        "skohorod_metric",
        "prohorov_thm2",
        "tightness_check",
        # Wave-761 renewal-theory canon.
        "blackwell_renewal",
        "key_renewal",
        "excess_renewal",
        "alternating_renewal",
        "renewal_reward2",
        "delayed_renewal",
        # Wave-760 Brownian-motion canon.
        "levy_bm",
        "wiener_bm",
        "doob_bm",
        "ito_bm",
        "cameron_martin",
        "gikhman_skorokhod",
        # Wave-759 CLE-2 canon.
        "gwynne_cle",
        "hospitsky_cle",
        "apu_cle",
        "nolin_cle",
        "sun_cle",
        "zhan_cle",
        # Wave-758 GFF-2 canon.
        "powell_gff",
        "aru_gff",
        "ding_zeitouni",
        "chatterjee_gff",
        "bolthausen_gff",
        "najafi_gff",
        # Wave-757 CLE canon.
        "sheffield_werner_cle",
        "miller_watson_cle",
        "camia_newman",
        "dubedat_cle",
        "kemppainen_werner",
        "rivera_cle",
        # Wave-756 GFF canon.
        "berestycki_gff",
        "duplantier_sheffield",
        "houchmandzadeh_gff",
        "nick_gff",
        "sheffield_miller",
        "wiegmann_zabrodin",
        # Wave-755 loop-soup canon.
        "lupu_loop",
        "lejan_loop",
        "kassel_wu",
        "kenyon_wilson",
        "barlow_ust",
        "lyons_peres",
        # Wave-754 random-cluster canon.
        "sokal_bcc",
        "caracciolo_pelissetto",
        "grimmett_rc",
        "hara_hara",
        "brydges_spencer",
        "glasner_aizenman",
        # Wave-753 O(N)-model canon.
        "fernandez_frohlich",
        "aizenman_irf",
        "fradkin_sokal",
        "nienhuis_on",
        "cardy_on",
        "pelissetto_vicari",
        # Wave-752 UST/LERW canon.
        "wilson_ust",
        "lawler_lerw",
        "benjamini_ust",
        "kirchhoff_matrix",
        "pemantle_ust",
        "schramm_lerw",
        # Wave-751 dimer-2 canon.
        "kassel_kenyon",
        "ciucu_dimers",
        "karl_dimers",
        "petrov_dimer",
        "durfee_arctic",
        "cohn_elkies",
        # Wave-750 dimer/Ising canon.
        "smirnov_ising",
        "chelkak_ising",
        "kenyon_dimers",
        "thurston_tiling",
        "duminil_copin2",
        "hongler_ising",
        # Wave-749 vertex-model-2 canon.
        "bufetov_sixv",
        "borodin_bufetov",
        "kuan_sixv",
        "dimitrov_sixv",
        "borodin_wheeler",
        "wheeler_zinn",
        # Wave-748 vertex-model canon.
        "borodin_sixv",
        "gowers_knot",
        "baxter_vertex",
        "reshetikhin_vertex",
        "corwin_petrov",
        "aggarwal_sixv",
        # Wave-747 ASEP-2 canon.
        "bertini_giacomin",
        "gardina_asym",
        "schutz_tasep",
        "balazs_seppalainen",
        "quastel_valko",
        "timar_tasep",
        # Wave-746 ASEP canon.
        "liggett_exclusion",
        "spitzer_exclusion",
        "sasamoto_tasep",
        "tracy_widom_tasep",
        "derrida_tasep",
        "ferrari_tasep",
        # Wave-745 KPZ-2 canon.
        "dotsenko_kpz",
        "hairer_kpz",
        "bernard_nicola",
        "imamura_sasamoto",
        "tracy_widom_kpz",
        "spohn_kpz",
        # Wave-744 KPZ canon.
        "kardar_parisi",
        "corwin_kpz",
        "quastel_spohn",
        "borodin_corwin",
        "amir_corwin",
        "calabrese_kpz",
        # Wave-743 random-matrix-2 canon.
        "baik_rmt",
        "tao_vu",
        "borodin_olshanski",
        "cipolloni_erdos",
        "bourgade_rmt",
        "chafai_rmt",
        # Wave-742 random-matrix canon.
        "soshnikov_rmt",
        "erdos_yau",
        "forrester_rmt",
        "mehta_rmt",
        "deift_rmt",
        "johansson_rmt",
        # Wave-741 percolation-2 canon.
        "beffara_nolin",
        "hara_slade",
        "gandre_liggett",
        "heyman_redner",
        "aiten_chayes",
        "newman_percolation",
        # Wave-740 percolation canon.
        "smirnov_percolation",
        "duminil_copin",
        "kesten_percolation",
        "cardy_formula",
        "russo_seymour",
        "grimmett_percolation",
        # Wave-739 Brownian-map-2 canon.
        "caraceni_curien",
        "bonzom_combe",
        "mullin_bijection",
        "bernardi_bijection",
        "schaeffer_bijection",
        "bouttier_guiter",
        # Wave-738 Brownian-map canon.
        "marckert_mokkadem",
        "le_gall_miermont",
        "curien_legall",
        "abraham_bipartite",
        "bettinelli_jacob",
        "chapuy_dolega",
        # Wave-737 LQG-2 canon.
        "sheffield_quantum",
        "gaines_sle",
        "miller_wu",
        "rhoade_vargas",
        "ding_dupias",
        "gwynne_miller",
        # Wave-736 LQG canon.
        "sheffield_gff",
        "berestycki_sheffield",
        "aru_powell",
        "huang_rhodes",
        "bisbisot_sheffield",
        "dhms_lqg",
        # Wave-735 SLE-2 canon.
        "beffara_sle",
        "kemppainen_sle",
        "zykin_sle",
        "viklund_sle",
        "benoist_sle",
        "holden_sle",
        # Wave-734 SLE canon.
        "osgood_schramm",
        "lawler_werner",
        "werner_wilson",
        "smirnov_parafermion",
        "garmadon_sle",
        "miller_sheffield",
        # Wave-733 Hall-algebra-2 canon.
        "green_hall",
        "bridgeland_hall",
        "kontsevich_soibelman",
        "mozgovoy_hall",
        "morita_hall",
        "calaque_hall",
        # Wave-732 Hall-algebra canon.
        "hall_algebra",
        "ringel_hall",
        "toen_hall",
        "lusztig_hall",
        "schiffmann_hall",
        "joyce_hall",
        # Wave-731 ramification-2 canon.
        "higher_ramif",
        "brylinski_kato",
        "log_ramification",
        "semi_stable_model",
        "neron_raynaud",
        "temkin_alter",
        # Wave-730 ramification canon.
        "grothendieck_muw",
        "raynaud_pencil",
        "saito_epsilon",
        "swan_conductor",
        "groth_tame",
        "kato_swan",
        # Wave-729 motivic-A1-2 canon.
        "roald_suslin",
        "jogiad_motive",
        "hauwas_nori",
        "motivic_pipe",
        "thom_mgl2",
        "voev_suslin",
        # Wave-728 motivic-A1 canon.
        "emerton_glass",
        "luan_yao",
        "morel_voev",
        "voev_homotopy",
        "totaro_cycle",
        "a1_degrees",
        # Wave-727 Galois-deformation-2 canon.
        "galdef_ring",
        "patching_arg",
        "taylor_wiles",
        "breuil_meizard",
        "gee_kisin",
        "caruso_lebaron",
        # Wave-726 Galois-deformation canon.
        "jetchev_skinner",
        "wan_sss",
        "wiles_taylor",
        "diamond_taylor_wiles",
        "kisin_crystalline",
        "mazur_deform",
        # Wave-725 automorphic-points canon.
        "p_group_iwasawa",
        "shimura_period",
        "arithmetic_arnold",
        "darmon_point",
        "bertolini_darmon",
        "howard_main",
        # Wave-724 arithmetic-cycles canon.
        "coates_wiles",
        "iwasawa_lfunc",
        "greenberg_selmer",
        "kurihara_iwasawa",
        "heegner_cycle",
        "gan_gross_prasad",
        # Wave-723 Iwasawa/Euler-system canon.
        "gross_zagier",
        "kolyvagin_sys",
        "euler_system",
        "iwasawa_motive",
        "rubin_main_conj",
        "perrin_riou",
        # Wave-722 special-values canon.
        "period_poly",
        "specialization_motive",
        "borel_motivic",
        "zagier_polylog",
        "deligne_period",
        "motivic_multiple_zeta",
        # Wave-721 mixed-motives canon.
        "brown_motives",
        "mzc_motive",
        "zeta_element",
        "mixed_elliptic",
        "motivic_pi",
        "beilinson_height",
        # Wave-720 derived-dimension canon.
        "cluster_tilting",
        "derived_morita",
        "preprojective_alg",
        "categorical_entropy",
        "serre_dim",
        "rouquier_dim",
        # Wave-719 Calabi-Yau/Gorenstein canon.
        "calabi_yau_tri",
        "d_calabi_yau",
        "gorenstein_proj",
        "frobenius_cat",
        "stable_category",
        "orbit_category",
        # Wave-718 representation-theory canon.
        "helix_theory",
        "mutation_class",
        "rep_finite",
        "der_bimodule",
        "icy_paper",
        "higher_auslander",
        # Wave-717 nc-motives canon.
        "nc_motive",
        "dg_enhancement",
        "bondal_kapranov",
        "enhanced_triangulated",
        "tabuada_motive",
        "nc_k_theory",
        # Wave-716 triangulated-geometry canon.
        "exceptional_coll",
        "spherical_functor",
        "serre_functor",
        "sod_decomp",
        "fourier_mukai",
        "semi_orthogonal",
        # Wave-715 cluster-algebra canon.
        "cluster_algebra",
        "quiver_mutation",
        "tilting_object",
        "auslander_reiten",
        "cluster_category",
        "silting_object",
        # Wave-714 motivic-25 canon.
        "motivic_total",
        "motivic_partial",
        "motivic_functor",
        "motivic_nerve",
        "derived_proper2",
        "derived_separated2",
        # Wave-713 spectral-AG-10 canon.
        "spectral_semi",
        "spectral_artin",
        "spectral_gal",
        "spectral_dirac",
        "derived_affine",
        "derived_projective",
        # Wave-712 motivic-24 canon.
        "motivic_additive",
        "motivic_additive_cat",
        "motivic_cover",
        "motivic_gysin2",
        "motivic_chern2",
        "motivic_filtration2",
        # Wave-711 homotopy-31 canon.
        "homotopy_sheaf2",
        "stable_inf_cat",
        "homotopy_stable4",
        "homotopy_local",
        "stable_sheaf2",
        "stable_coalgebra",
        # Wave-710 higher-algebra-11 canon.
        "higher_algebra8",
        "operad_infty4",
        "floyd_farey",
        "operad_swiss3",
        "little_discs3",
        "operad_twisted",
        # Wave-709 category-22 canon.
        "cat_univariant2",
        "cat_ab2",
        "cat_exact3",
        "cat_freyd",
        "cat_ab_loc",
        "cat_pro_object2",
        # Wave-708 category-21 canon.
        "cat_pseudo_limit",
        "cat_weak_eq",
        "cat_reedy_cat",
        "cat_dold_kan",
        "cat_hoc",
        "cat_enriched_lim",
        # Wave-707 spectral-AG-9 canon.
        "spectral_prime",
        "spectral_residue",
        "spectral_level",
        "spectral_polynomial2",
        "spectral_coord",
        "spectral_ext_field",
        # Wave-706 chromatic-9 canon.
        "chromatic_layer",
        "morava_maven",
        "chromatic_square2",
        "lubin_tate3",
        "elliptic_morava",
        "chromatic_base",
        # Wave-705 derived-geometry-10 canon.
        "derived_geometry7",
        "derived_abelian2",
        "derived_stack3",
        "derived_morph",
        "derived_cover",
        "derived_topos",
        # Wave-704 higher-algebra-10 canon.
        "higher_algebra9",
        "operad_infty5",
        "operad_swiss4",
        "koszul_duality3",
        "braces_e5",
        "delooping3",
        # Wave-703 homotopy-30 canon.
        "homotopy_suspension2",
        "homotopy_fiber3",
        "stable_derivator",
        "homotopy_spectrum2",
        "stable_excisive",
        "homotopy_vn",
        # Wave-702 motivic-23 canon.
        "motivic_spark",
        "motivic_fundamental",
        "motivic_hochschild",
        "motivic_field",
        "motivic_degree",
        "motivic_diagonal",
        # Wave-701 derived-geometry-9 canon.
        "derived_conn",
        "derived_local",
        "derived_reduced",
        "derived_integral",
        "derived_normal",
        "derived_noether",
        # Wave-700 category-20 canon.
        "cat_pushout",
        "cat_span",
        "cat_lax",
        "cat_street",
        "cat_size",
        "cat_total",
        # Wave-699 homotopy-29 canon.
        "homotopy_general",
        "homotopy_rational",
        "stable_dual",
        "stable_lie",
        "stable_motivic",
        "stable_perf",
        # Wave-698 motivic-22 canon.
        "motivic_frobenius",
        "motivic_cartier",
        "motivic_hodge",
        "motivic_span",
        "motivic_lax",
        "motivic_street",
        # Wave-697 spectral-AG-8 canon.
        "spectral_dvr",
        "spectral_noether",
        "spectral_regular",
        "spectral_dedekind",
        "spectral_jacobson",
        "spectral_excellent",
        # Wave-696 category-19 canon.
        "cat_rank",
        "cat_index",
        "cat_monotone",
        "cat_kernel",
        "cat_image",
        "cat_pullback",
        # Wave-695 homotopy-28 canon.
        "homotopy_abelian",
        "homotopy_finite",
        "homotopy_infinite",
        "homotopy_extended",
        "stable_synthetic",
        "stable_compact",
        # Wave-694 motivic-21 canon.
        "motivic_trace",
        "motivic_transfer2",
        "motivic_coniveau",
        "motivic_atiyah",
        "motivic_deligne",
        "motivic_residue",
        # Wave-693 derived-geometry-8 canon.
        "derived_etale",
        "derived_flat",
        "derived_smooth2",
        "derived_quasi_coherent",
        "derived_represent",
        "derived_cartesian",
        # Wave-692 higher-algebra-9 canon.
        "e5_algebra",
        "little_cubes2",
        "swiss_cheese3",
        "framed_discs",
        "factorization_hom3",
        "centralizer_alg2",
        # Wave-691 category-18 canon.
        "cat_pretopos",
        "cat_semisimple",
        "cat_fusion",
        "cat_tannakian2",
        "cat_ribbon",
        "cat_semiadd",
        # Wave-690 homotopy-27 canon.
        "homotopy_sheaf",
        "homotopy_model",
        "stable_monoid",
        "stable_group",
        "stable_module",
        "stable_algebra",
        # Wave-689 motivic-20 canon.
        "motivic_jouanolou",
        "motivic_infinite2",
        "motivic_ext",
        "motivic_norm",
        "motivic_ramified",
        "motivic_euler",
        # Wave-688 spectral-AG-7 canon.
        "spectral_field",
        "spectral_lattice",
        "spectral_filtration",
        "spectral_cellular",
        "spectral_cohomological",
        "spectral_finite",
        # Wave-687 category-17 canon.
        "cat_fibrant_obj",
        "cat_cofibrant",
        "cat_bicomplete",
        "cat_univariant",
        "cat_descent",
        "cat_glueable",
        # Wave-686 higher-algebra-8 canon.
        "e4_algebra",
        "centralizer_alg",
        "delooping2",
        "factorization_hom2",
        "koszul_duality2",
        "braces_e4",
        # Wave-685 homotopy-26 canon.
        "homotopy_limit",
        "homotopy_tower",
        "spectral_sequence5",
        "homotopy_class2",
        "stable_mapping",
        "stable_bousfield",
        # Wave-684 motivic-19 canon.
        "motivic_weight2",
        "motivic_infinite",
        "motivic_suslin",
        "motivic_frequency",
        "motivic_wit",
        "motivic_base2",
        # Wave-683 chromatic-8 canon.
        "blue_shift2",
        "chromatic_fracture2",
        "morava_stabilizer2",
        "fgsl_group2",
        "tate_spec2",
        "k_n_local2",
        # Wave-682 motivic-18 canon.
        "motivic_tower",
        "motivic_sphere3",
        "motivic_etale",
        "motivic_crystal",
        "motivic_prism",
        "motivic_cycle",
        # Wave-681 homotopy-25 canon.
        "homotopy_lift",
        "homotopy_orbit",
        "homotopy_fixed",
        "stable_operad",
        "homotopy_factor",
        "stable_sheaf",
        # Wave-680 spectral-AG-6 canon.
        "spectral_perfect",
        "spectral_smooth2",
        "spectral_etale2",
        "spectral_abelian",
        "spectral_crystal",
        "spectral_proper",
        # Wave-679 motivic-17 canon.
        "motivic_thh",
        "motivic_realization",
        "etale_motive",
        "relative_motive",
        "absolute_motive",
        "motivic_heart",
        # Wave-678 category-16 canon.
        "simplicial_cat",
        "homotopical_cat",
        "relative_cat",
        "equipment_cat",
        "fibrant_cat",
        "pointed_cat",
        # Wave-677 category-15 canon.
        "derivator_cat",
        "quillen_cat",
        "combinatorial_mc",
        "cat_dg",
        "univalent_cat",
        "cat_structure",
        # Wave-676 higher-algebra-7 canon.
        "en_algebra2",
        "thom_transpose",
        "higher_brace2",
        "koszul_operad2",
        "operad_lie",
        "center_hochschild",
        # Wave-675 higher-algebra-6 canon.
        "e3_algebra",
        "getzler_jones",
        "tadv_hochschild",
        "cyclotomic_e_n",
        "surfaces_operad",
        "boards_operad",
        # Wave-674 spectral-AG-5 canon.
        "derived_k3",
        "spectral_gm",
        "analytic_spec",
        "graded_spec",
        "equivariant_spec",
        "spectral_curve",
        # Wave-673 derived-geometry-6 canon.
        "derived_cohom",
        "spectral_deformation2",
        "virtual_class2",
        "derived_intersection",
        "derived_fiber2",
        "relative_trace",
        # Wave-672 category-14 canon.
        "tannakian_cat",
        "super_cat",
        "perverse_cat",
        "smashing_cat",
        "compact_cat",
        "monoidal_derived",
        # Wave-671 homotopy-24 canon.
        "gray_periodic",
        "stunted_proj",
        "adams_edge",
        "periodic_family",
        "unstable_adams2",
        "homotopy_exponent",
        # Wave-670 chromatic-7 canon.
        "morava_k3",
        "morava_e2",
        "chromatic_l3",
        "telescope_tower3",
        "picard_spec2",
        "red_shift2",
        # Wave-669 motivic-16 canon.
        "slice_filtration2",
        "milnor_operations2",
        "motivic_bordism",
        "motivic_eilenberg2",
        "f_motive2",
        "motivic_ss2",
        # Wave-668 motivic-15 canon.
        "norimotive3",
        "motivic_galois2",
        "tannakian_motive2",
        "period_realization2",
        "beilinson_regulator2",
        "hodge_motive2",
        # Wave-667 category-13 canon.
        "stable_cat2",
        "exact_cat2",
        "ab_cat",
        "grothendieck_cat",
        "coniveau_fil",
        "special_cat",
        # Wave-666 derived-geometry-5 canon.
        "derived_abelian",
        "simplicial_comm",
        "derived_bezout",
        "derived_hecke",
        "cotangent_stack",
        "derived_bun",
        # Wave-665 homotopy-23 canon.
        "cohen_moore2",
        "whitehead_product",
        "homotopy_decomp",
        "kervaire_inv2",
        "unstable_vn",
        "moore_space2",
        # Wave-664 spectral-AG-4 canon.
        "spectral_moduli",
        "e_ring_moduli",
        "tmf_stack",
        "spectral_artstack",
        "structured_spec",
        "elliptic_spec2",
        # Wave-663 higher-algebra-5 canon.
        "e2_algebra",
        "dunn_additivity",
        "tensor_factorization",
        "swiss_cheese2",
        "mckay_correspond",
        "khovanov_2",
        # Wave-662 higher-algebra-4 canon.
        "bar_resolution2",
        "hochschild_hom2",
        "factor_homology2",
        "deligne_conj2",
        "braces_higher",
        "little_cubes",
        # Wave-661 chromatic-6 canon.
        "ambidexterity",
        "higher_semiadditivity",
        "tate_height",
        "dieudonne_module",
        "honda_formal",
        "raynaud_height",
        # Wave-660 chromatic-5 canon.
        "morava_k2",
        "telescope_tower2",
        "chromatic_l2",
        "picard_spec",
        "periodicity_height",
        "chromatic_completion",
        # Wave-659 motivic-14 canon.
        "norimotive2",
        "motivic_tate2",
        "absolute_cohom",
        "motivic_weight",
        "tate_triple",
        "motivic_pairing",
        # Wave-658 motivic-13 canon.
        "motivic_galois",
        "tannakian_motive",
        "period_realization",
        "beilinson_regulator",
        "hodge_motive",
        "f_motive",
        # Wave-657 homotopy-22 canon.
        "ravenel_htpy",
        "bousfield_period",
        "snake_constr",
        "homotopy_cartesian",
        "p_local_htpy",
        "completion_htpy",
        # Wave-656 category-12 canon.
        "essentially_small",
        "finitely_accessible",
        "admissible_cat",
        "definable_cat",
        "cocomplete_cat",
        "cartesian_cat2",
        # Wave-655 homotopy-21 canon.
        "devinatz_htpy",
        "hopkins_smith",
        "morava_stab",
        "chromatic_square",
        "telescope_tower",
        "bo_htpy",
        # Wave-654 category-11 canon.
        "compactly_generated",
        "presentable_cat2",
        "accessible_cat2",
        "flat_monad",
        "locally_presentable",
        "regular_cat2",
        # Wave-653 arithmetic-geometry-2 canon.
        "fargues_scholze3",
        "integral_padic2",
        "ainf_cohom",
        "period_ring",
        "galois_padic",
        "hodge_tate_padic",
        # Wave-652 motivic-12 canon.
        "strict_motive",
        "sheaf_motive",
        "numerical_motive",
        "asymptotic_motive",
        "exponential_motive",
        "log_motive",
        # Wave-651 category-10 canon.
        "flat_functor",
        "filtered_cat",
        "sifted_cat2",
        "regular_cat",
        "abelian_cat",
        "malcev_cat",
        # Wave-650 homotopy-20 canon.
        "bousfield_htpy",
        "dror_htpy",
        "kane_htpy",
        "moore_htpy",
        "neisendorfer_htpy",
        "anick_htpy",
        # Wave-649 prismatic-3 canon.
        "prism_site2",
        "cartier_prism",
        "breuil_prism",
        "filtered_prism",
        "frobenius_prism",
        "stacky_prism",
        # Wave-648 algebraic-K-10 canon.
        "kodaira_k",
        "lindenstrauss_k",
        "tsukada_k",
        "guin_k",
        "dupont_k",
        "suslin_k2",
        # Wave-647 homotopy-19 canon.
        "selick_htpy",
        "arkowitz_htpy",
        "lin_htpy",
        "kahn_priddy",
        "bochner_htpy",
        "tits_building",
        # Wave-646 p-adic-7 canon.
        "fargues_cat",
        "v_stack",
        "untilt2",
        "spatial_diamond",
        "diamond_sheaf",
        "bdr_plus",
        # Wave-645 algebraic-K-9 canon.
        "witt_k",
        "schlichting_k",
        "balmer_k",
        "hermitian_k3",
        "thomason_les",
        "vishik_k",
        # Wave-644 algebraic-K-8 canon.
        "s_multicat",
        "allday_k",
        "residue_k",
        "suslin_wagoner",
        "weibel_nil",
        "hall_alg",
        # Wave-643 homotopy-18 canon.
        "toda_smith",
        "mahowald_inv",
        "calc_tower",
        "goodwillie_deriv",
        "snaith_split",
        "kervaire_inv",
        # Wave-642 prismatic-2 canon.
        "prismatic_f",
        "bhatt_scholze",
        "q_crystal",
        "prismatic_dieudonne",
        "q_prism",
        "derived_prism",
        # Wave-641 homotopy-17 canon.
        "smash_prod",
        "stable_stem2",
        "homotopy_colim",
        "unstable_tower",
        "periodic_fam",
        "chromatic_htpy",
        # Wave-640 spectral-AG-3 canon.
        "spectral_group",
        "azure_space",
        "spectral_scheme3",
        "spectral_smooth",
        "spectral_etale",
        "elliptic_cohom2",
        # Wave-639 homotopy-16 canon.
        "homotopy_fiber2",
        "stable_htpy2",
        "finite_htpy",
        "rational_spec",
        "finite_chromatic",
        "periodic_htpy",
        # Wave-638 algebraic-K-7 canon.
        "grayson_s",
        "karoubi_v2",
        "vorst_descent",
        "quillen_ldev",
        "fundamental_cat",
        "seg_street",
        # Wave-637 p-adic-6 canon.
        "fargues_scholze2",
        "curve_padic",
        "diamond_mod",
        "etale_phiphi",
        "cocartesian_diamond",
        "scholze_bc",
        # Wave-636 tensor-category-3 canon.
        "sylleptic",
        "haagerup_sub",
        "ek_subfactor",
        "gyro_cat",
        "yang_lee_cat",
        "sovereign_cat",
        # Wave-635 commutative-algebra-5 canon.
        "excellent_ring",
        "zariski_main",
        "going_up",
        "lying_over",
        "integral_closure2",
        "weil_divisor2",
        # Wave-634 category-9 canon.
        "icon_cat",
        "bicat2",
        "vert_cat",
        "double_lim",
        "two_transform",
        "cat_3cell",
        # Wave-633 motivic-11 canon.
        "motivic_coho2",
        "cone_theorem",
        "motivic_landweber",
        "motivic_abelian",
        "motivic_compact",
        "contr_rational",
        # Wave-632 witt-vectors-2 canon.
        "witt_len2",
        "big_witt",
        "good_reduction",
        "potential_reduction",
        "tate_curve",
        "odeur_zarba",
        # Wave-631 etale-2 canon.
        "etale_cover3",
        "etale_site3",
        "constructible_sh",
        "weil_sheaf",
        "torsion_sheaf",
        "ql_sheaf",
        # Wave-630 infinity-categories-4 canon.
        "quasi_cat2",
        "inner_horn",
        "joyal_horn",
        "fib_infty",
        "cartesian_morphism",
        "infty_functor",
        # Wave-629 deformations-3 canon.
        "deform_functor2",
        "tangent_def",
        "rim_deform",
        "small_ext",
        "hull_deform",
        "artinian_alg",
        # Wave-628 operads-2 canon.
        "moerdijk_weiss",
        "higher_operad",
        "operad_infty3",
        "operad_cat2",
        "dendroidal_seg",
        "operad_module",
        # Wave-627 condensed-4 canon.
        "discrete_liquid",
        "smith_project",
        "condensed_ring",
        "liquid_ring",
        "scholze_trace",
        "condensed_coh",
        # Wave-626 stacks-3 canon.
        "algebraic_stack2",
        "artin_stack",
        "quotient_stack2",
        "stacky_point",
        "orbifold_stack",
        "gerbe_cohomology",
        # Wave-625 homotopy-15 canon.
        "stable_cohomology2",
        "woodward_op",
        "spectrum_type",
        "complexity_spectrum",
        "small_spec",
        "simplicial_htpy",
        # Wave-624 formal-geometry canon.
        "raynaud_formal",
        "formal_completion",
        "adic_formal",
        "formal_neighborhood",
        "groth_existence",
        "algebraization",
        # Wave-623 higher-operads canon.
        "dendroidal2",
        "operadic_nerve",
        "infty_operad2",
        "a_infinity2",
        "e_infinity3",
        "cyclic_operad",
        # Wave-622 prismatic canon.
        "prism2",
        "prismatic_site",
        "delta_ring",
        "prismatic_crystal",
        "hodge_tate",
        "nygaard2",
        # Wave-621 motivic-10 canon.
        "motivic_k",
        "motivic_borel",
        "motivic_height",
        "motivic_chow",
        "motivic_homology",
        "motivic_class",
        # Wave-620 stacks-2 canon.
        "gerbe2",
        "band_gerbe",
        "rigid_stack",
        "dm_stack2",
        "inertia_stack",
        "root_stack",
        # Wave-619 homotopy-14 canon.
        "unstable_htpy",
        "tame_htpy",
        "devissage_ss",
        "andersen_lannes",
        "chromatic_hopkins",
        "thick_spectrum",
        # Wave-618 p-adic-5 canon.
        "fontaine_curve",
        "untilt",
        "perfectoid_c",
        "b_drb",
        "phi_mod",
        "ad_period",
        # Wave-617 algebraic-K-6 canon.
        "gillet_thomason",
        "khomo_k",
        "k_theory4",
        "gersen_suslin",
        "berrick_k",
        "hermitian_quillen",
        # Wave-616 tensor-category-2 canon.
        "multifusion",
        "premodular2",
        "braided_functor",
        "center_cat",
        "fusion_ring",
        "ds_category",
        # Wave-615 motivic-9 canon.
        "motivic_adams",
        "motivic_classifying",
        "tate_object",
        "motivic_dg",
        "mori_bir",
        "extremal_ray",
        # Wave-614 homotopy-13 canon.
        "primary_op",
        "secondary_op",
        "steenrod_sq",
        "peterson_stein",
        "moore_spec",
        "finite_spectra",
        # Wave-613 arithmetic-geometry-2 canon.
        "witt_vector",
        "witt_teich",
        "verschiebung_witt",
        "perfect_witt",
        "neron_smooth",
        "semistable_reduction",
        # Wave-612 commutative-algebra-4 canon.
        "regular_ring",
        "gorenstein_ring",
        "normal_ring",
        "factorial_ring",
        "jacobson_ring",
        "discrete_valuation",
        # Wave-611 category-8 canon.
        "pasting_diag",
        "mate_dual",
        "whisker_comp",
        "pseudo_naturality",
        "two_adjoint",
        "modification",
        # Wave-610 deformations-2 canon.
        "schlessinger2",
        "prorepresent",
        "versal_def",
        "semiuniversal",
        "first_order",
        "obstruction_def",
        # Wave-609 spectral-AG-2 canon.
        "e_infty_space",
        "brave_new_ring",
        "thom_constr",
        "log_ring",
        "orient_cohom",
        "formal_moduli",
        # Wave-608 sheaf-4 canon.
        "etale_descent",
        "etale_morphism",
        "fppf_site",
        "fpqc_site",
        "ladic_sheaf",
        "lisse_sheaf",
        # Wave-607 infinity-categories-3 canon.
        "kan_complex",
        "horn_filler",
        "nerve_cat",
        "mapping_space",
        "homotopy_cat",
        "cocartesian",
        # Wave-606 algebraic-K-5 canon.
        "connective_k",
        "higher_k",
        "k_spectrum",
        "nil_k",
        "karoubi_k",
        "pedersen_weibel",
        # Wave-605 motivic-8 canon.
        "motivic_steenrod",
        "motivic_adem",
        "power_operations",
        "simplicial_motive",
        "dk_motive",
        "motivic_transfer",
        # Wave-604 homotopy-12 canon.
        "unstable_cohomology",
        "may_ss",
        "bokstedt_periodicity",
        "topo_k_theory",
        "elliptic_k",
        "equivariant_cohomology2",
        # Wave-603 topos-4 canon.
        "slice_topos",
        "logical_morph",
        "classifying_topos",
        "atomic_topos",
        "essential_morph",
        "giraud_axiom",
        # Wave-602 cyclic-homology canon.
        "cyclotomic_spec",
        "tr_structure",
        "tc_spec",
        "negative_cyclic",
        "periodic_cyclic",
        "tate_construction",
        # Wave-601 derived-geometry-4 canon.
        "dg_algebra",
        "derived_loop",
        "derived_tangent",
        "virtual_fund",
        "structured_space",
        "e_infinity_ring",
        # Wave-600 operad-theory canon.
        "a_infty_alg",
        "l_infty_alg",
        "koszul_duality",
        "minimal_model_op",
        "operadic_bar",
        "operad_cobar",
        # Wave-599 monad-theory canon.
        "monad_theorem",
        "klesli_cat",
        "codensity_monad",
        "monadicity",
        "distributive_law",
        "algebra_cat",
        # Wave-598 algebraic-K-4 canon.
        "borel_regulator",
        "soul_elem",
        "lichtenbaum_k",
        "bloch_beilinson",
        "etale_ktheory",
        "thh_trace",
        # Wave-597 chromatic-homotopy canon.
        "curtis_lower",
        "bousfield_kan",
        "lannes_t",
        "dror_smith",
        "telescope_conj",
        "periodicity_thm",
        # Wave-596 p-adic-Hodge canon.
        "breuil_mod",
        "kisin_mod",
        "galois_lattice",
        "padic_hodge",
        "finite_height",
        "etale_phi",
        # Wave-595 crystalline-cohomology canon.
        "crys_cohom",
        "syntomic",
        "divided_power",
        "pd_envelope",
        "nygaard_filt",
        "conjugate_fil",
        # Wave-594 etale-cohomology canon.
        "etale_homotopy",
        "pro_etale",
        "etale_fund",
        "galois_cat",
        "artin_neighborhood",
        "shapiro_lemma",
        # Wave-593 nonabelian-Hodge canon.
        "higgs_bundle2",
        "hitchin_section",
        "simpson_corr",
        "nonabelian_hodge",
        "harmonic_bdl",
        "hodge_moduli",
        # Wave-592 tensor-category canon.
        "tensor_cat",
        "braided_cat",
        "rigid_cat",
        "fusion_cat",
        "spherical_cat",
        "premodular",
        # Wave-591 motivic-7 canon.
        "friedlander_voev",
        "motivic_eilenberg",
        "motivic_zeta",
        "motivic_purity",
        "motivic_descent",
        "motivic_invert",
        # Wave-590 algebraic-K-3 canon.
        "quillen_plus",
        "gersten_ss",
        "loday_k",
        "volodin_k",
        "suslin_k",
        "bloch_k",
        # Wave-589 topos-3 canon.
        "cartesian_closed",
        "internal_logic",
        "subobject_lattice",
        "power_object",
        "pretopos",
        "coherent_topos",
        # Wave-588 homotopy-11 canon.
        "ehp_sequence",
        "james_period",
        "whitehead_prod",
        "freudenthal_susp",
        "moore_space",
        "unstable_adams",
        # Wave-587 birational-2 canon.
        "terminal_sing",
        "canonical_sing2",
        "klt_mmp",
        "mmp_flip",
        "abundance_conj",
        "bdd_fano",
        # Wave-586 duality-theory canon.
        "groth_duality",
        "dualizing_cmplx",
        "residue_thm",
        "verdier_duality",
        "dualizing_sheaf",
        "relative_duality",
        # Wave-585 intersection-theory canon.
        "intersection_theory",
        "macpherson_chern",
        "weil_divisor",
        "picard_group",
        "line_bundle",
        "canonical_bundle",
        # Wave-584 condensed-3 canon.
        "clausen_scholze2",
        "solid_cohom",
        "nuclear_space",
        "analytic_sheaf",
        "solid_tensor2",
        "proetale_site2",
        # Wave-583 motivic-6 canon.
        "motivic_base_change",
        "six_op_motivic",
        "motivic_smooth",
        "motivic_proper",
        "fulton_mclarty",
        "motivic_homotopy2",
        # Wave-582 geometric-Langlands-2 canon.
        "arinkin_gaitsgory",
        "derived_satake",
        "spectral_bung",
        "nilp_cone",
        "geometric_satake2",
        "fusion_product",
        # Wave-581 spectral-sequences canon.
        "serre_ss4",
        "bockstein_ss",
        "eilenberg_moore",
        "bousfield_ss",
        "lyndon_ss",
        "cartan_ss",
        # Wave-580 enumerative-combinatorics canon.
        "species_theory",
        "cycle_index",
        "lagrange_inversion",
        "transfer_matrix",
        "matrix_tree",
        "exponential_gf",
        # Wave-579 geometric-invariant-theory canon.
        "git_quotient",
        "hilbert_mumford",
        "moment_polytope",
        "kirwan_strat",
        "symplectic_quot",
        "luna_slice",
        # Wave-578 arithmetic-geometry-2 canon.
        "global_height",
        "bogomolov_conj",
        "equidistribution_thm",
        "canonical_height",
        "nevanlinna_th",
        "vojta_conj",
        # Wave-577 perverse-sheaves canon.
        "perverse_sheaf",
        "intersection_homology",
        "nearby_cycles",
        "d_module2",
        "char_cycle",
        "middle_perversity",
        # Wave-576 free-probability canon.
        "free_prob",
        "r_transform",
        "s_transform",
        "free_convolution",
        "voiculescu_thm",
        "operator_valued",
        # Wave-575 random-matrix-2 canon.
        "circular_law",
        "dyson_brownian",
        "sine_kernel",
        "airy_process",
        "tracy_widom",
        "beta_ensemble",
        # Wave-574 differential-topology-2 canon.
        "exotic_sphere",
        "kervaire_milnor",
        "surgery_theory",
        "smale_hcob",
        "whitney_trick",
        "immersion_thm",
        # Wave-573 Hodge-2/periods canon.
        "griffiths_transv",
        "period_domain",
        "mumford_tate",
        "hodge_class",
        "absolute_hodge",
        "hodge_conj",
        # Wave-572 abelian-varieties canon.
        "abelian_variety",
        "isogeny_av",
        "tate_module",
        "shafarevich_conj",
        "faltings_thm",
        "mordell_weil_av",
        # Wave-571 singularity-theory canon.
        "du_val_sing",
        "rational_sing",
        "log_canonical",
        "multiplier_ideal",
        "bernstein_sato",
        "milnor_fiber",
        # Wave-570 positivity/moduli canon.
        "hodge_index",
        "kodaira_vanishing",
        "kollar_mori",
        "boundedness_moduli",
        "stability_sheaf",
        "bogomolov_ineq",
        # Wave-569 geometric-PDE canon.
        "yamabe_problem",
        "prescribed_curvature",
        "nirenberg_problem",
        "kazdan_warner",
        "aubin_thm",
        "trudinger_thm",
        # Wave-568 symplectic-field-theory canon.
        "symplectic_field",
        "contact_homology3",
        "floer_homol",
        "reeb_orbit",
        "sft_algebra",
        "eliashberg_givental",
        # Wave-567 algebraic-combinatorics canon.
        "littlewood_richardson",
        "knuth_rsk",
        "macdonald_poly",
        "schubert_calc",
        "honeycomb_tiling",
        "berenstein_zelevinsky",
        # Wave-566 L-functions/random-matrix canon.
        "selberg_trace2",
        "zero_spacing",
        "montgomery_pair",
        "gue_statistics",
        "keating_snaith",
        "rudnick_sarnak",
        # Wave-565 arithmetic-statistics canon.
        "bhargava_lic",
        "cohen_lenstra",
        "elliptic_rank",
        "malle_conj",
        "prime_gaps",
        "zhang_maynard",
        # Wave-564 geometric-group-theory canon.
        "gromov_hyperbolic",
        "quasi_isometry",
        "thin_triangle",
        "word_problem",
        "baumslag_solitar",
        "asymptotic_cone",
        # Wave-563 harmonic-maps canon.
        "harmonic_map",
        "eells_sampson",
        "schoen_uhlenbeck",
        "bubbling_hm",
        "heat_flow_hm",
        "sacks_uhlenbeck",
        # Wave-562 symplectic-geometry-2 canon.
        "gromov_width",
        "hofer_metric",
        "symplectic_capacity",
        "symplectic_packing",
        "mcduff_polterovich",
        "ekeland_hofer",
        # Wave-561 minimal-surfaces canon.
        "minimal_surface",
        "plateau_problem",
        "brakke_flow",
        "almgren_pitts",
        "simon_regularity",
        "stable_minimal",
        # Wave-560 geometric-flows canon.
        "hamilton_ricci",
        "perelman_entropy",
        "ricci_soliton",
        "kahler_ricci_flow",
        "mean_curvature_flow",
        "ancient_solution",
        # Wave-559 complex-geometry canon.
        "calabi_yau_mfd",
        "calabi_conjecture",
        "kahler_einstein",
        "k_stability",
        "csck_metric",
        "futaki_invariant",
        # Wave-558 gauge-theory canon.
        "yang_mills",
        "instanton_moduli",
        "anti_self_dual",
        "higgs_bundle",
        "kapustin_witten",
        "nahm_transform",
        # Wave-557 3-manifold-theory canon.
        "heegaard_splitting",
        "dehn_surgery",
        "sutured_mfd",
        "taut_foliation",
        "thin_position",
        "normal_surface",
        # Wave-556 Teichmueller-theory canon.
        "weil_petersson",
        "mapping_class",
        "quadratic_diff",
        "earthquake_map",
        "extremal_length",
        "pseudo_anosov",
        # Wave-555 homological-mirror-symmetry canon.
        "hms_conjecture",
        "landau_ginzburg",
        "syz_mirror",
        "torus_fibration",
        "wrapped_fukaya",
        "mirror_functor",
        # Wave-554 DT/GW-theory canon.
        "kontsevich_mgn",
        "gw_descendant",
        "donaldson_thomas",
        "pandharipande_thomas",
        "gopakumar_vafa",
        "mnop_conj",
        # Wave-553 mirror-symmetry canon.
        "mirror_symmetry",
        "givental_j",
        "quantum_cohomology",
        "quintic_invariants",
        "toric_mirror",
        "frobenius_mfd",
        # Wave-552 contact-topology canon.
        "contact_form",
        "legendrian_knot",
        "overtwisted",
        "tight_contact",
        "giroux_corr",
        "convex_surface",
        # Wave-551 foliation-theory canon.
        "foliation",
        "holonomy_grp",
        "godbillon_vey",
        "haefliger_struct",
        "novikov_thm",
        "thurston_fol",
        # Wave-550 characteristic-classes canon.
        "chern_class",
        "pontryagin_class",
        "euler_class",
        "todd_genus",
        "chern_character",
        "hirzebruch_sig",
        # Wave-549 index-theory canon.
        "atiyah_singer",
        "dirac_op",
        "eta_invariant",
        "heat_kernel2",
        "signature_op",
        "analytic_torsion",
        # Wave-548 Floer-theory canon.
        "floer_homology",
        "knot_floer",
        "instanton_floer",
        "monopole_floer",
        "lagrangian_floer",
        "fukaya_cat",
        # Wave-547 4-manifold canon.
        "four_mfd",
        "donaldson_thm",
        "seiberg_witten",
        "exotic_r4",
        "intersection_form",
        "freedman_thm",
        # Wave-546 knot-theory canon.
        "knot_invariant",
        "jones_poly",
        "alexander_poly",
        "knot_group",
        "knot_signature",
        "vassiliev_inv",
        # Wave-545 Thurston-geometrization canon.
        "thurston_geometrization",
        "eight_geometries",
        "seifert_fibered",
        "haken_mfd",
        "jsj_decomp",
        "ricci_flow",
        # Wave-544 Kleinian-groups canon.
        "kleinian_group",
        "limit_set",
        "hyperbolic_3mfd",
        "mostow_rigidity",
        "jorgensen_thurston",
        "tameness_thm",
        # Wave-543 Riemann-surfaces canon.
        "riemann_surface",
        "branched_cover",
        "abel_jacobi",
        "riemann_hurwitz",
        "fuchsian_group",
        "teichmuller_space",
        # Wave-542 several-complex-variables canon.
        "hartogs_thm",
        "domain_holo",
        "pseudoconvex",
        "levi_problem",
        "oka_coherence",
        "d_bar_neumann",
        # Wave-541 complex-analysis-2 canon.
        "riemann_mapping",
        "schwarz_lemma",
        "picard_thm",
        "montel_normal",
        "runge_approx",
        "jensen_formula",
        # Wave-540 transcendence-theory canon.
        "hermite_lindemann",
        "gelfond_schneider",
        "baker_thm",
        "lindemann_weier",
        "schanuel_conj",
        "siegel_shidlovskii",
        # Wave-539 diophantine-approximation canon.
        "dirichlet_approx",
        "roth_thm2",
        "continued_frac2",
        "kronecker_thm",
        "liouville_number",
        "subspace_thm",
        # Wave-538 elliptic-PDE canon.
        "sobolev_space",
        "poincare_ineq",
        "trace_thm",
        "harnack_thm",
        "schauder_est",
        "degiorgi_nash",
        # Wave-537 Riemannian-geometry canon.
        "riemann_metric",
        "levi_civita",
        "riemann_curvature",
        "ricci_scalar",
        "jacobi_field",
        "comparison_thm",
        # Wave-536 symplectic-geometry canon.
        "symplectic_form",
        "lagrangian_mfd",
        "hamiltonian_flow",
        "poisson_bracket",
        "contact_geom",
        "gromov_nonsq",
        # Wave-535 microlocal-analysis canon.
        "wavefront_set",
        "pseudodiff_op",
        "fourier_io",
        "symbol_calc",
        "propagation_sing",
        "elliptic_est",
        # Wave-534 potential-theory canon.
        "harmonic_fn",
        "potential_thy",
        "capacity_theory",
        "balayage",
        "green_fn",
        "fine_topology",
        # Wave-533 geometric-measure-theory canon.
        "rectifiability",
        "tangent_measure",
        "density_thm",
        "marstrand",
        "besicovitch",
        "preiss_rect",
        # Wave-532 fractal-geometry canon.
        "hausdorff_dim",
        "box_counting",
        "self_similar",
        "iterated_function",
        "frostman",
        "multifractal_formal",
        # Wave-531 bifurcation-theory canon.
        "saddle_node",
        "hopf_bif",
        "period_doubling",
        "neimark_sacker",
        "bogdanov_takens",
        "homoclinic_bif",
        # Wave-530 nonuniform-hyperbolicity canon.
        "pesin_theory",
        "nonuniform_hyp",
        "dominated_split",
        "osceledets_reg",
        "lyapunov_chart",
        "katok_horseshoe",
        # Wave-529 KAM/Aubry-Mather canon.
        "kam_theorem",
        "aubry_mather",
        "twist_map",
        "cantorus",
        "greene_crit",
        "arnold_diff",
        # Wave-528 thermodynamic-formalism canon.
        "transfer_op",
        "thermo_formal",
        "pressure_thm",
        "equilibrium_state",
        "ruelle_zeta",
        "lasota_yorke",
        # Wave-527 hyperbolic-dynamics canon.
        "anosov",
        "srb_measure",
        "horseshoe",
        "stable_mfld",
        "bowen_spec",
        "markov_partition",
        # Wave-526 complex-dynamics canon.
        "julia_set",
        "mandelbrot_set",
        "fatou_set",
        "sullivan_no_wander",
        "douady_hubbard",
        "parabolic_impl",
        # Wave-525 ergodic-theory canon.
        "birkhoff",
        "mean_ergodic",
        "mixing_weak",
        "entropy_ks",
        "bernoulli_shift",
        "osceledets",
        # Wave-524 Ramsey-theory canon.
        "hales_jewett",
        "rado_thm",
        "gallai_thm",
        "schur_thm",
        "hindman",
        "furstenberg",
        # Wave-523 incidence-geometry canon.
        "erdos_distinct",
        "sz_trotter",
        "kakeya",
        "ff_kakeya",
        "joints_thm",
        "guth_katz",
        # Wave-522 additive-combinatorics canon.
        "freiman_thm",
        "szemeredi",
        "green_tao",
        "roth_thm",
        "gowers_norm",
        "plunnecke",
        # Wave-521 analytic-NT canon.
        "explicit_formula",
        "zero_density",
        "riemann_zeta",
        "dirichlet_l",
        "linnik_thm",
        "chebyshev_bias",
        # Wave-520 automorphic-GL(n) canon.
        "gln_automorphic",
        "whittaker_model",
        "godement_jacq",
        "rankin_selberg",
        "langlands_lfunc",
        "converse_thm",
        # Wave-519 modular-forms canon.
        "modular_form",
        "hecke_op2",
        "eisenstein_srs2",
        "cusp_form",
        "theta_func",
        "dedekind_eta",
        # Wave-518 Yang-Baxter/braid canon.
        "yang_baxter",
        "braid_rep",
        "yangian",
        "rtt_formalism",
        "quantum_double",
        "ribbon_cat",
        # Wave-517 quantum-groups canon.
        "quantum_group",
        "crystal_base",
        "quantum_rmatrix",
        "jimbo_drin",
        "lusztig_can",
        "quantum_schur",
        # Wave-516 Kac-Moody/VOA canon.
        "kac_moody",
        "weyl_kac",
        "vertex_alg",
        "moonshine_module",
        "affine_lie",
        "zhu_algebra",
        # Wave-515 syzygy-theory canon.
        "betti_series",
        "minimal_free",
        "auslander_buchs",
        "serre_conj",
        "quillen_suslin",
        "green_koszul",
        # Wave-514 differential-cohomology canon.
        "diff_cohom",
        "cheeger_simons",
        "deligne_cohom",
        "flat_bundle",
        "beilinson_reg",
        "secondary_inv",
        # Wave-513 categorification canon.
        "categorify",
        "khovanov_hom",
        "hecke_cat",
        "soergel_bim",
        "rasmussen_inv",
        "uq_sl2",
        # Wave-512 super-geometry canon.
        "super_space",
        "super_manifold",
        "super_lie",
        "odd_variables",
        "berezin_int",
        "super_scheme",
        # Wave-511 Fargues-Scholze canon.
        "fs_diamond",
        "geometric_satake",
        "v_sheaf",
        "bun_g",
        "hecke_stack",
        "y_diamond",
        # Wave-510 spectral-AG canon.
        "spectral_scheme2",
        "connective_e_ring",
        "spectral_alg",
        "spectral_stack",
        "elliptic_cohor",
        "taf_lurie",
        # Wave-509 Weil-II/l-adic canon.
        "etale_site2",
        "l_adic_sheaf",
        "frobenius_action",
        "groth_lefschetz",
        "deligne_weil2",
        "purity_thm",
        # Wave-508 anabelian-geometry canon.
        "anabelian_geo",
        "section_conj",
        "fundamental_grp",
        "etale_pi1",
        "groth_tei",
        "tamagawa_mochi",
        # Wave-507 quasi-category/Joyal canon.
        "quasi_cat",
        "joyal_model",
        "homotopy_coherent",
        "nerve_quasi",
        "htc_colimit",
        "marking_qcat",
        # Wave-506 NIP/distal model-theory canon.
        "dp_rank",
        "forking_seq",
        "honest_def",
        "uniform_def",
        "distality",
        "nip_formula",
        # Wave-505 cobordism-theory canon.
        "cobordism_grp",
        "oriented_cob",
        "unoriented_cob",
        "complex_cob",
        "framed_cob",
        "thom_cob",
        # Wave-504 Steenrod/cohomology-operations canon.
        "steenrod_algebra",
        "adem_relations",
        "serre_cartan",
        "unstable_modules",
        "lambda_algebra",
        "bar_resolution",
        # Wave-503 elliptic-surface canon.
        "elliptic_surface",
        "weierstrass_eq",
        "kodaira_fiber",
        "tate_algorithm",
        "mordell_weil2",
        "neron_model",
        # Wave-502 dg-category canon.
        "dg_cat2",
        "dg_morita",
        "dg_quotient",
        "drinfeld_quotient",
        "dg_nerve",
        "keller_dg",
        # Wave-501 F-singularity canon.
        "f_regular",
        "f_rational",
        "f_pure",
        "f_threshold",
        "test_ideal",
        "tight_closure",
        # Wave-500 arithmetic-Langlands canon.
        "local_langlands",
        "harris_taylor",
        "weil_group",
        "langlands_functoriality",
        "epsilon_factor",
        "l_packet",
        # Wave-499 Hodge-theory canon.
        "hodge_decomp",
        "l2_hodge",
        "mixed_hodge",
        "period_map",
        "vhs_polarized",
        "limit_mhs",
        # Wave-498 moduli/Gromov-Witten canon.
        "kuranishi",
        "hilbert_scheme2",
        "quot_scheme",
        "m_bar_gn",
        "stable_map",
        "gromov_witten",
        # Wave-497 birational-geometry canon.
        "minimal_model",
        "klt_pair",
        "flip_cone",
        "fano_mori",
        "mmp_algorithm",
        "toric_flip",
        # Wave-496 DAG-deformation canon.
        "derived_deformation",
        "formal_deformation",
        "dag_representation",
        "derived_moduli",
        "tangent_coh",
        "obstruction_2",
        # Wave-495 infinity-2-category canon.
        "globular_model",
        "opetopic",
        "theta_space",
        "complicial",
        "verity_gray",
        "weak_infty",
        # Wave-494 noncommutative-geometry canon.
        "hochschild_coh",
        "cyclic_coh",
        "nc_scheme",
        "calabi_yau_alg",
        "ginzburg_dga",
        "connes_nc",
        # Wave-493 tropical-geometry canon.
        "tropical_poly",
        "berkovich_an",
        "skeleton_trop",
        "tropical_curve",
        "mikhalkin",
        "tropical_cycle",
        # Wave-492 log-geometry canon.
        "log_structure",
        "kato_fontaine",
        "log_smooth",
        "log_etale",
        "log_derham",
        "log_crystalline",
        # Wave-491 automorphic-2 canon.
        "shimura_var",
        "l_function",
        "hecke_alg2",
        "theta_lift",
        "arthur_param",
        "satake_param",
        # Wave-490 arithmetic-geometry canon.
        "arakelov_deg",
        "adelic_curve",
        "height_arakelov",
        "faltings_metric",
        "arithmetic_chow",
        "arith_rr",
        # Wave-489 group-theory-4 canon.
        "building_toy",
        "coxeter_grp",
        "bn_pair",
        "braid_grp",
        "hecke_bm",
        "parabolic_grp",
        # Wave-488 derived-geometry-3 canon.
        "shifted_tangent",
        "derived_quot",
        "virtual_pull",
        "intrinsic_be",
        "d_critical",
        "perfect_obstruction",
        # Wave-487 category-7 canon.
        "compact_obj",
        "dualizable_cat",
        "comma_cat",
        "prestack",
        "endo_prof",
        "exact_cat",
        # Wave-486 algebraic-K-theory-2 canon.
        "waldhausen_k",
        "plus_k",
        "kv_theory",
        "karoubi_v",
        "vorst_stab",
        "nk_theory",
        # Wave-485 p-adic-4 canon.
        "lubin_tate2",
        "bc_space",
        "local_shimura",
        "scholze_weinstein",
        "fargues_curve2",
        "banach_colmez2",
        # Wave-484 synthetic-math-2 canon.
        "internal_univ",
        "virtual_hodge",
        "stein_space",
        "formal_model",
        "univalent_found",
        "synth_stable",
        # Wave-483 homotopy-10 canon.
        "steenrod_ops",
        "dyer_lashof",
        "bar_spec",
        "free_loop",
        "sullivan_min",
        "loop_functor",
        # Wave-482 chromatic-4 canon.
        "chromatic_fracture",
        "morava_stabilizer",
        "fgsl_group",
        "tate_spec",
        "blue_shift",
        "red_shift",
        # Wave-481 motivic-5 canon.
        "mtc_motive",
        "fqmotive",
        "triang_motive",
        "motivic_chern",
        "motivic_landin",
        "higher_chow2",
        # Wave-480 infinity-topos-3 canon.
        "etale_geom",
        "gros_topos",
        "local_homeo",
        "classify_obj",
        "pi_infty",
        "exponentiable",
        # Wave-479 p-adic-3 canon.
        "fargues_diam",
        "tilting_equiv",
        "scholze_diamond",
        "ahb_ring",
        "prism_2",
        "drinfeld_sym",
        # Wave-478 motivic-4 canon.
        "levine_morel",
        "quadratic_k",
        "mgl_spec",
        "cellular_motive",
        "motivic_pi0",
        "beilinson_con",
        # Wave-477 arithmetic-D-modules-2 canon.
        "dagger_dm",
        "spencer_dm",
        "caro_dm",
        "berthelot_rigid",
        "berthelo_crys",
        "arithmetic_ht",
        # Wave-476 chromatic-3 canon.
        "bp_spectrum",
        "adams_novikov",
        "landweber_exact",
        "greek_letter",
        "smith_toda",
        "picard_grp",
        # Wave-475 homotopical-algebra canon.
        "dendroidal",
        "infty_operad",
        "cyclic_hk",
        "chiral_alg",
        "sifted_cat",
        "seq_spectra",
        # Wave-474 TQFT-2 canon.
        "reshet_turaev",
        "khovanov",
        "heegaard_floer",
        "cobordism_hyp",
        "modular_cat",
        "topological_order",
        # Wave-473 higher-algebra-3 canon.
        "thh_2",
        "cyclotomic2",
        "cartier_mod",
        "witt_vec2",
        "crystalline_stack",
        "trt_functor",
        # Wave-472 motivic-3 canon.
        "alg_cobordism",
        "hermitian_k",
        "oriented_coh",
        "slice_spec",
        "motivic_stem2",
        "rostmotive",
        # Wave-471 p-adic-geometry-2/perfectoid canon.
        "perfectoid2",
        "diamond_geo",
        "integral_padic",
        "breuil_kisin",
        "banach_colmez",
        "drinfeld_tower",
        # Wave-470 homotopy-9 canon.
        "e_infty2",
        "power_op",
        "obstruction_th",
        "rational_htpy",
        "h_space",
        "james_constr",
        # Wave-469 category-6 canon.
        "enriched_cat",
        "weight_lim",
        "fibered_cat",
        "derivator2",
        "accessible_cat",
        "day_conv",
        # Wave-468 model-theory-7 canon.
        "o_minimal",
        "nip_theory",
        "nonforking",
        "simple_theory",
        "abstract_erc",
        "tame_metric",
        # Wave-467 set-theory-5 canon.
        "constructible_l",
        "large_card",
        "pcf_theory",
        "proper_forcing",
        "core_model",
        "square_princ",
        # Wave-466 sheaf-3/microlocal canon.
        "micro_supp",
        "kashiwara_schapira",
        "loc_system",
        "perverse_2",
        "stacky_sheaf",
        "sheaf_homotopy",
        # Wave-465 analytic-geometry-3 canon.
        "kedlaya_renorm",
        "dagger_groth",
        "raynaud_gen",
        "weierstrass_prep",
        "gauss_point",
        "affinoid_alg",
        # Wave-464 chromatic-2 canon.
        "morava_e",
        "tmf_spectrum",
        "k_n_local",
        "chromatic_conv",
        "nilpotence_dev",
        "telescopic",
        # Wave-463 arithmetic-D-modules canon.
        "overconv_dm",
        "arithmetic_dm",
        "frobenius_dm",
        "holonomic_dm",
        "rigid_dm",
        "isocrystal",
        # Wave-462 infinity-topos-2 canon.
        "n_localic",
        "shape_theory",
        "descent_cond",
        "lex_reflect",
        "cartesian_fib2",
        "cohesive_struct",
        # Wave-461 DAG-2/shifted-symplectic canon.
        "shifted_sympl",
        "lagrangian_int",
        "derived_critical",
        "lie_algebroid",
        "moment_map",
        "quant_dag",
        # Wave-460 condensed-2/analytic-rings canon.
        "analytic_ring2",
        "solid_tensor",
        "trace_class",
        "clausen_scholze",
        "solid_derived",
        "pyknotic",
        # Wave-459 order-theory-2/domain-theory canon.
        "fixed_points_ord",
        "chain_cond",
        "scott_cpo",
        "way_below",
        "galois_insertion",
        "denotational",
        # Wave-458 matroid-3 canon.
        "transversal_mat",
        "matroid_rep",
        "tutte_poly",
        "matroid_minor",
        "regular_mat",
        "delta_matroid",
        # Wave-457 double-category/proarrow canon.
        "proarrow",
        "virtual_equip",
        "fibrant_double",
        "tabulation",
        "companion_conj",
        "framed_bicat",
        # Wave-456 pure-motives canon.
        "chow_motive",
        "nori_motive",
        "num_equiv",
        "standard_conj",
        "voev_motive",
        "tate_motive",
        # Wave-455 synthetic-math canon.
        "cubical_path",
        "hcomp_fill",
        "glue_types",
        "interval_obj",
        "kan_op",
        "transport_coe",
        # Wave-454 intersection-cohomology-2 canon.
        "ic_stalk",
        "decomp_thm",
        "riemann_hilbert",
        "fourier_sato",
        "vanishing_cycles",
        "middle_ext",
        # Wave-453 higher-algebra-2 canon.
        "operad_koszul",
        "bar_cobar",
        "factor_homology",
        "hochschild_hom",
        "deligne_conj",
        "primitive_elts",
        # Wave-452 geometric-Langlands canon.
        "d_module",
        "geometric_langlands",
        "hecke_eig",
        "opers_g",
        "ramified_l",
        "kernel_fun",
        # Wave-451 equivariant-homotopy canon.
        "g_spectrum",
        "mackey_functor",
        "norm_map",
        "ro_grading",
        "wirthmuller",
        "tom_dieck",
        # Wave-450 stable-infinity canon.
        "stable_infty",
        "spectra_cat",
        "exact_seq",
        "stable_tstruct",
        "smash_monoidal",
        "thh_tc",
        # Wave-449 analytic-geometry-2 canon.
        "dagger_space",
        "huber_ring",
        "adic_generic",
        "witt_perfect",
        "fargues_curve",
        "prism_site",
        # Wave-448 higher-topos canon.
        "infty_topos",
        "univ_colimit",
        "object_classif",
        "trunc_modal",
        "cohesive_top",
        "hypercomplete",
        # Wave-447 TQFT canon.
        "tqft_axiom",
        "bord_cat",
        "frobenius_2d",
        "extended_tqft",
        "dw_theory",
        "chern_simons",
        # Wave-446 Goodwillie-calculus canon.
        "goodwillie_tower",
        "excisive_fn",
        "linearization",
        "deriv_layer",
        "calc_converge",
        "orth_calc",
        # Wave-445 six-functor canon.
        "six_functors",
        "base_change",
        "projection_frm",
        "verdier_dual",
        "constructible",
        "perverse_sh",
        # Wave-444 DAG-stacks canon.
        "derived_stack",
        "cotangent_cx",
        "geometric_stk",
        "tannaka_rec",
        "quasi_smooth",
        "perf_stack",
        # Wave-443 motivic-2 canon.
        "motivic_coh",
        "chow_group",
        "milnor_conj",
        "voevodsky_dm",
        "motivic_stem",
        "brauer_grp",
        # Wave-442 model-categories-2 canon.
        "cofibrant_rep",
        "quillen_equiv",
        "monoidal_model",
        "enriched_model",
        "reedy_model",
        "localization_mc",
        # Wave-441 Galois-representations canon.
        "gal_rep",
        "fontaine_ring",
        "filtered_module",
        "weil_deligne",
        "hecke_eigensys",
        "ribet_toy",
        # Wave-440 homotopy-8 canon.
        "thom_iso",
        "postnikov_twr",
        "whitehead_twr",
        "bott_period",
        "stable_stem",
        "hopf_map",
        # Wave-439 formal-groups/chromatic canon.
        "formal_group",
        "lazard_ring",
        "formal_module",
        "height_strata",
        "lubin_tate",
        "morava_k",
        # Wave-438 condensed-mathematics canon.
        "condensed_set",
        "solid_group",
        "liquid_group",
        "proetale_site",
        "light_condensed",
        "analytic_ring",
        # Wave-437 p-adic cohomology canon.
        "crystalline_coh",
        "prismatic_coh",
        "etale_coh",
        "derham_coh",
        "frobenius_coh",
        "comparison_iso",
        # Wave-436 infinity-categories-2 canon.
        "complete_seg",
        "cartesian_fib",
        "straightening",
        "presentable_cat",
        "adjoint_functor",
        "bousfield_loc",
        # Wave-435 derived-schemes canon.
        "derived_scheme",
        "quasi_coherent",
        "derived_fiber",
        "spectral_scheme",
        "virtual_class",
        "shifted_symplectic",
        # Wave-434 Langlands-toy canon.
        "satake_iso",
        "hecke_operator",
        "langlands_dual",
        "eisenstein_srs",
        "automorphic_rep",
        "fourier_coeff",
        # Wave-433 p-adic-geometry canon.
        "rigid_analytic",
        "berkovich_space",
        "perfectoid_space",
        "adic_space",
        "etale_ph2",
        "diamond_toy",
        # Wave-432 spectral-sequences-3 canon.
        "atiyah_hirzebruch",
        "serre_ss3",
        "leary_ss",
        "descent_ss",
        "motivic_ss",
        "vanishing_ss",
        # Wave-431 motivic-homotopy canon.
        "a1_homotopy",
        "motivic_sphere",
        "morel_degree",
        "voevodsky_motive",
        "slice_filtration",
        "milnor_operations",
        # Wave-430 higher-algebra canon.
        "e_n_algebra",
        "operad_infty",
        "monoidal_infty",
        "module_cat",
        "brane_tensor",
        "delooping",
        # Wave-429 algebraic-K-theory canon.
        "k0_group",
        "k1_group",
        "milnor_k2",
        "quillen_q",
        "k_theory_spec",
        "bass_heller_swan",
        # Wave-428 deformation-theory canon.
        "deformation_functor",
        "schlessinger",
        "tangent_space_def",
        "obstruction_theory",
        "versal_deformation",
        "maurer_cartan",
        # Wave-427 number-theory-4 canon.
        "elliptic_height",
        "mordell_weil",
        "lseries_toy",
        "bsd_toy",
        "modularity_toy",
        "padic_integral",
        # Wave-426 homotopy-7 canon.
        "model_category",
        "quillen_adj",
        "simplicial_set",
        "infinity_cat",
        "derived_alg",
        "stable_cat",
        # Wave-425 stacks/moduli canon.
        "moduli_stack",
        "stacky_curve",
        "coarse_space",
        "quotient_stack",
        "gerbe_toy",
        "stack_morph",
        # Wave-424 set-theory-4 canon.
        "club_set",
        "stationary_set",
        "ultrafilter_toy",
        "partition_calc",
        "closed_unbounded",
        "mahlo_cardinal",
        # Wave-423 homological-algebra-4 canon.
        "groth_spectral",
        "serre_ss2",
        "hypercohom",
        "deriv_hom",
        "cartan_eilenberg",
        "adams_diff",
        # Wave-422 Lie-theory-2 canon.
        "weyl_chamber",
        "root_height",
        "borel_subalgebra",
        "levi_factor",
        "nilpotent_orbit",
        "verma_module",
        # Wave-421 category-theory-4 canon.
        "traced_monoidal",
        "star_autonomous",
        "frobenius_alg",
        "span_compose",
        "profunctor_toy",
        "endo_coend",
        # Wave-420 algebraic-NT-3 canon.
        "dirichlet_unit",
        "regulator",
        "ideal_class",
        "minkowski_bound",
        "dedekind_zeta",
        "splitting_prime",
        # Wave-419 algebraic-topology-5 canon.
        "serre_fibration",
        "path_fibration",
        "bundle_section",
        "classify_space",
        "vector_bundle",
        "thom_space",
        # Wave-418 probability-5 canon.
        "weak_law",
        "strong_lln",
        "clt_classic",
        "borel_cantelli",
        "dominated_conv",
        "uniform_lln",
        # Wave-417 commutative-algebra-3 canon.
        "groebner_syz",
        "free_resolution",
        "hilbert_syzygy",
        "regular_seq",
        "depth_ring",
        "cohen_mac",
        # Wave-416 model-theory-6 canon.
        "decidable_theory",
        "indiscernible_seq",
        "saturated_model",
        "omitting_prime",
        "interpol_thm",
        "definable_set",
        # Wave-415 topology-4 canon.
        "quotient_map",
        "open_cover",
        "locally_compact",
        "homeo_top",
        "paracompact",
        "partition_unity",
        # Wave-414 homotopy-6 canon.
        "exact_couple",
        "adams_ss",
        "stable_homotopy",
        "whitehead_thm",
        "obstruction",
        "cofiber",
        # Wave-413 Galois-3 canon.
        "artin_lemma",
        "normal_basis",
        "kummer_ext",
        "abelian_ext",
        "frobenius_el",
        "inseparable",
        # Wave-412 2-category canon.
        "two_cat",
        "bicat_comp",
        "mate_calc",
        "double_cat",
        "lax_functor",
        "cat_enriched",
        # Wave-411 algebraic-geometry-9 canon.
        "blow_up",
        "intersection_mult",
        "tangent_cone",
        "normalization",
        "divisor_class",
        "dualizing",
        # Wave-410 set-theory-3 canon.
        "forcing2",
        "inner_model",
        "descriptive3",
        "recursion3",
        "proof_mining",
        "ordinal_notation",
        # Wave-409 homotopy-5 canon.
        "spectral_seq2",
        "eilenberg_zilber",
        "dold_kan",
        "postnikov",
        "stable_range",
        "cohend",
        # Wave-408 representation-theory-4 canon.
        "schur_functor",
        "brauer_alg",
        "hecke_alg",
        "casimir_op",
        "weight_space",
        "bz_category",
        # Wave-407 algebraic-geometry-8 canon.
        "cech_cohom",
        "serre_duality",
        "adjunction2",
        "scheme_fiber",
        "hilbert_scheme",
        "flattening",
        # Wave-406 homological-algebra-3 canon.
        "poincare_duality2",
        "universal_coeff",
        "kunneth",
        "leray_hirsch",
        "hopf_algebra2",
        "functor_derived",
        # Wave-405 derived-categories canon.
        "derived_functor2",
        "triangulated",
        "bounded_complex",
        "mapping_cone_tri",
        "koszul_dual",
        "t_structure",
        # Wave-404 operad-2 canon.
        "operad_algt",
        "brace_operad",
        "swiss_cheese",
        "little_intervals",
        "operad_homology",
        "props_toy",
        # Wave-403 homotopy-4 canon.
        "j_hom_toy",
        "toda_bracket",
        "spectral_atiyah",
        "pi_stems",
        "hopf_invariant",
        "thom_spectrum",
        # Wave-402 topos-2 canon.
        "topos_subobj",
        "groth_topo",
        "sheaf_cond",
        "logic_topos",
        "geometric_morph",
        "etale_space",
        # Wave-401 proof-theory-3 canon.
        "herbrand_thm",
        "interp_equality",
        "cut_elim_seq",
        "finitary_induct",
        "hilbert_system",
        "reverse_math",
        # Wave-400 algebraic-geometry-7 canon.
        "etale_cover",
        "jacobian_toy",
        "hom_stack_toy",
        "seesaw_theorem",
        "picard_variety",
        "dual_ab_var",
        # Wave-399 model-theory-5 canon.
        "ef_game_toy",
        "vaught_test",
        "real_closed",
        "boolean_prime",
        "fraisse_limit",
        "qe_dense_order",
        # Wave-398 representation-theory-3 canon.
        "induced_char",
        "artins_theorem",
        "tensor_char",
        "clifford_toy",
        "schur_index",
        "frobenius_group",
        # Wave-397 stochastic-analysis-2 canon.
        "ost_calcul",
        "tanaka",
        "bessel3",
        "reflect_bm",
        "occupation_bm",
        "h_transform",
        # Wave-396 algebraic-number-theory-2 canon.
        "cyclotomic_field",
        "kronecker_weber",
        "local_field",
        "hensel_field",
        "cm_points",
        "idele_class",
        # Wave-395 combinatorial-enumeration canon.
        "catalan_dp",
        "stirling_cycle",
        "partition_count",
        "bell_triangle",
        "eulerian_num",
        "inclusion_excl",
        # Wave-394 order-theory canon.
        "downset_lattice",
        "zeta_mobius",
        "linear_extension",
        "sperner_bound",
        "dilworth_partition",
        "birkhoff_rep",
        # Wave-393 algebraic-topology-4 canon.
        "eilenberg_steenrod",
        "cap_product",
        "thom_isom",
        "serre_class",
        "obstruction_toy",
        "k_theory",
        # Wave-392 computability canon.
        "mu_recursion",
        "primitive_recursion",
        "diagonal_lemma",
        "arithmetization",
        "fixed_point_combinator",
        "kleene_normal",
        # Wave-391 category-theory-3 canon.
        "monoidal_cat",
        "closed_cat",
        "presheaf",
        "kan_extension",
        "distributor",
        "equivalence_cat",
        # Wave-390 design-theory canon.
        "latin_trade",
        "steiner_system",
        "inc_structure",
        "orthogonal_array",
        "hadamard_matrix",
        "finite_difference",
        # Wave-389 number-fields canon.
        "norm_subring",
        "discriminant_field",
        "decomposition_group",
        "ramification",
        "artin_symbol",
        "class_group_toy",
        # Wave-388 homological-algebra-2 canon.
        "derived_functor",
        "ext_compute",
        "tor_compute",
        "spectral_seq",
        "koszul_homology",
        "mapping_degree",
        # Wave-387 probability-4 canon.
        "uniform_integrability",
        "vitali_conv",
        "ldp_theory",
        "concentration_ineq",
        "kolmogorov_01",
        "prokhorov_metric",
        # Wave-386 proof-theory-2 canon.
        "sequent_calculus",
        "natural_ded",
        "godel_incomp",
        "interp_proof",
        "proof_complexity",
        "modal_completeness",
        # Wave-385 commutative-algebra-2 canon.
        "hilbert_samuel",
        "krull_dim",
        "noether_normal",
        "primary_decomp",
        "completion_ring",
        "dimension_fiber",
        # Wave-384 group-theory-3 canon.
        "hall_subgroup",
        "transfer_hom",
        "schur_multiplier",
        "aut_group",
        "composition_series",
        "permutation_poly",
        # Wave-383 homotopy-theory-3 canon.
        "mapping_cone",
        "loop_space",
        "em_space",
        "co_homology",
        "stiefel_whitney",
        "transfer",
        # Wave-382 model-theory-4 canon.
        "stone_duality",
        "saturation_test",
        "omitting_types",
        "indiscernibles",
        "stability_spec",
        "back_forth",
        # Wave-381 algebraic-geometry-6 canon.
        "grothendieck_grp",
        "chow_ring",
        "gysin",
        "toric_variety",
        "proj_morph",
        "ample_test",
        # Wave-380 descriptive-set-theory-2 canon.
        "baire_space",
        "polish_topology",
        "borel_functions",
        "souslin_op",
        "determinacy_toy",
        "perfect_set_prop",
        # Wave-379 matroid-2 canon.
        "matroid_axioms",
        "greedy_matroid",
        "matroid_intersect",
        "dual_matroid",
        "matroid_union",
        "represented_matroid",
        # Wave-378 set-theory-2/forcing canon.
        "forcing_poset",
        "dense_filter",
        "names_eval",
        "cohen_adds",
        "ma_toy",
        "large_cardinal",
        # Wave-377 operad canon.
        "operad_assoc",
        "operad_comm",
        "little_discs",
        "operad_tree",
        "endomorphism_op",
        "may_recognition",
        # Wave-376 homotopy-theory-2 canon.
        "fibration",
        "cofibration",
        "serre_ss",
        "whitehead",
        "suspension",
        "spectra",
        # Wave-375 algebraic-geometry-5 canon.
        "riemann_roch",
        "sheaf_cohomology",
        "scheme_local",
        "blowup",
        "elliptic_group",
        "moduli_stable",
        # Wave-374 model-theory-3 canon.
        "quantifier_elim",
        "realize_types",
        "omega_categoricity",
        "acl_closure",
        "morley_rank",
        "vocab_interp",
        # Wave-373 optimization-3 canon.
        "bundle_method",
        "sqp",
        "ip_qp",
        "trust_region",
        "frank_wolfe2",
        "bfgs_wolfe",
        # Wave-372 stochastic-analysis canon.
        "ito_lemma",
        "girsanov",
        "sde_strong",
        "local_time",
        "quadratic_var",
        "malliavin",
        # Wave-371 differential-topology canon.
        "morse_theory",
        "transversality",
        "regular_value",
        "degree_mod2",
        "handle_decomp",
        "poincare_hopf",
        # Wave-370 graph-theory-2 canon.
        "tutte_berge",
        "dirac_ore",
        "turan_theorem",
        "planar_five",
        "graph_minor",
        "ramsey_num",
        # Wave-369 numerical-6 canon.
        "broyden",
        "cheb_approx",
        "brent_root",
        "romberg",
        "aitken_delta",
        "collocation_ode",
        # Wave-368 model-theory-2/logic canon.
        "unification_fol",
        "skolem_normal",
        "herbrand_model",
        "presburger",
        "los_theorem",
        # Wave-367 Galois-2/field-theory canon.
        "finite_field",
        "galois_corresp",
        "normality_check",
        "separable_check",
        "cyclotomic_poly",
        "primitive_elem",
        # Wave-366 algebraic-topology-3 canon.
        "singular_homology",
        "cw_complex",
        "spectral_seq_toy",
        "homotopy_group",
        "excision",
        "poincare_dual",
        # Wave-365 probability-3 canon.
        "optional_stopping",
        "doob_decomp",
        "martingale_clt",
        "azuma",
        "coupling_arg",
        "ergodic_thm",
        # Wave-364 functional-analysis-3 canon.
        "hahn_banach",
        "riesz_repr",
        "adjoint_op",
        "selfadjoint_spectrum",
        "compact_resolvent",
        "projection_thm",
        # Wave-363 real-analysis canon.
        "cantor_set",
        "baire_category",
        "vitali_set",
        "egorov_thm",
        "fatou_lemma",
        "monotone_conv",
        # Wave-362 complex-analysis canon.
        "cauchy_integral",
        "residue_calc",
        "laurent_series",
        "argument_principle",
        "conformal_map",
        "liouville",
        # Wave-361 algebraic-topology-2 canon.
        "homotopy_pi1",
        "simplicial_homology",
        "chain_homotopy",
        "euler_homology",
        "degree_map",
        "covering_lift",
        # Wave-360 PDE-theory canon.
        "energy_method",
        "maximum_principle",
        "heat_kernel",
        "wave_dalembert",
        "weak_solution",
        "fundamental_laplace",
        # Wave-359 harmonic-analysis canon.
        "plancherel",
        "poisson_summation",
        "fejer_kernel",
        "uncertainty",
        "fourier_multiplier",
        "sobolev_embed",
        # Wave-358 functional-analysis-2 canon.
        "open_mapping",
        "uniform_bounded",
        "weak_convergence",
        "banach_alaoglu",
        "reflexive_space",
        "closed_graph",
        # Wave-357 differential-geometry-2 canon.
        "connection_form",
        "parallel_transport",
        "holonomy",
        "gauss_bonnet",
        "geodesic_eq",
        "sectional_curv",
        # Wave-356 stochastic-processes-2 canon.
        "markov_chain",
        "martingale_check",
        "poisson_process",
        "gambler_ruin",
        "stopping_time",
        "markov_hitting",
        # Wave-355 probability-2 canon.
        "kolmogorov_axioms",
        "conditional_expect",
        "markov_ineq",
        "conv_sum",
        "moment_generating",
        "stochastic_order",
        # Wave-354 algebraic-geometry-4 canon.
        "sheaf_gluing",
        "local_ring_zn",
        "dedekind_check",
        "divisor_group",
        "genus_riemann",
        "moduli_naive",
        # Wave-353 ODE-theory canon.
        "picard_lindelof",
        "gronwall_lemma",
        "sturm_liouville",
        "phase_plane",
        "lyapunov_stability",
        "variation_params",
        # Wave-352 Lie-theory canon.
        "cartan_matrix",
        "weyl_group_a2",
        "killing_form",
        "root_lattice_a2",
        "sl2_structure",
        "su2_algebra",
        # Wave-351 functional-analysis canon.
        "banach_fixed",
        "spectral_theorem",
        "lp_duality",
        "fourier_finite",
        "compact_operator",
        "gram_schmidt",
        # Wave-350 representation-theory-2 canon.
        "character_table_s3",
        "perm_rep",
        "schur_ortho",
        "induced_rep",
        "fourier_sn",
        "regular_rep",
        # Wave-349 algebraic-geometry-3 canon.
        "zariski_topo",
        "projective_plane",
        "bezout_bezout",
        "variety_dim",
        "monomial_ideal",
        "hilbert_poly",
        # Wave-348 graph-theory/combinatorics-2 canon.
        "graph_coloring",
        "euler_trail",
        "matroid_greedy",
        "planar_check",
        "poset_dimension",
        "ramsey_r33",
        # Wave-347 topology-3/point-set canon.
        "compact_space",
        "connected_space",
        "quotient_topology",
        "product_topology",
        "convergence_space",
        "tietze_urysohn",
        # Wave-346 number-theory-2/homological-2 canon.
        "quadratic_recip",
        "elliptic_curve",
        "p_adic_val",
        "cohomology_cup",
        "koszul_complex",
        "mayer_vietoris",
        # Wave-345 commutative-algebra/ring-theory canon.
        "ring_ideals",
        "quotient_ring",
        "pid_check",
        "minimal_poly",
        "norm_trace",
        "spec_ring",
        # Wave-344 group-theory-2 canon.
        "sylow_theorems",
        "group_presentation",
        "burnside_lemma",
        "free_group",
        "conjugacy_classes",
        "cayley_graph",
        # Wave-343 lattice/universal-algebra canon.
        "lattice_check",
        "galois_connection",
        "tarski_fixed",
        "boolean_algebra",
        "congruence_lattice",
        "term_algebra",
        # Wave-342 descriptive-set-theory/recursion-2 canon.
        "borel_hierarchy",
        "analytic_sets",
        "forcing_lite",
        "arith_hierarchy",
        "jump_operator",
        "rice_theorem",
        # Wave-341 modal-logic/topology-2 canon.
        "kripke_semantics",
        "bisimulation",
        "ef_game",
        "fundamental_group",
        "covering_space",
        "topo_separation",
        # Wave-340 homological-algebra/algebraic-geometry canon.
        "chain_complex",
        "tor_ext",
        "sheaf_check",
        "hilbert_series",
        "snake_lemma",
        "variety_morph",
        # Wave-339 algebra canon.
        "field_ext",
        "galois_group",
        "splitting_field",
        "lie_bracket",
        "rep_theory",
        "root_system",
        # Wave-338 set-theory canon.
        "ordinal_arith",
        "cardinal_arith",
        "transfinite_induct",
        "well_founded",
        "v_omega",
        "ac_choice",
        # Wave-337 lambda-calculus/rewriting canon.
        "ski_combinator",
        "de_bruijn",
        "church_encoding",
        "lambda_typing",
        "unification",
        "knuth_bendix",
        # Wave-336 computability/model-theory canon.
        "pr_functions",
        "turing_degrees",
        "busy_beaver",
        "ultraproduct",
        "ramsey_theory",
        "compactness_lite",
        # Wave-335 category-2/topos canon.
        "fin_limit",
        "subobject_classifier",
        "exponential_obj",
        "yoneda_embed",
        "adjoint_check",
        "cat_colimit",
        # Wave-334 secure-computation canon.
        "garbled_circuit",
        "bgw_mpc",
        "beaver_triple",
        "ot_extension",
        "spdz_mac",
        "psi_intersect",
        # Wave-333 computer-algebra-2 canon.
        "poly_factor_fp",
        "hensel_lift",
        "poly_crt",
        "subresultant",
        "sparse_interp",
        "poly_eval_interp",
        # Wave-332 quantum-information canon.
        "density_matrix",
        "povm_measure",
        "qchannel",
        "entanglement",
        "bell_ineq",
        "state_tomo",
        # Wave-331 complexity-theory canon.
        "np_reduce",
        "fpras_dnf",
        "sumcheck",
        "param_fpt",
        "pcp_verify",
        "circuit_lb",
        # Wave-330 SMT-theory canon.
        "diff_logic",
        "array_theory",
        "bv_ops",
        "dpllt",
        "lia_branch",
        "mcsat_lite",
        # Wave-329 proof-theory canon.
        "nd_check",
        "sequent_prove",
        "cut_elim",
        "resolution_fol",
        "linear_logic",
        "intuit_class",
        # Wave-328 homotopy-type-theory canon.
        "path_types",
        "hlevel_check",
        "univalence_toy",
        "kan_hcomp",
        "funext_toy",
        "hit_quotient",
        # Wave-327 zero-knowledge canon.
        "r1cs_check",
        "qap_encode",
        "kzg_commit",
        "bulletproof_ip",
        "plonkish_gate",
        "snark_circuit",
        # Wave-326 verification-3 canon.
        "timed_automata",
        "parity_game",
        "nba_emptiness",
        "ctl_mc",
        "bisim_refine",
        "wsts_cover",
        # Wave-325 PL-8 ownership/substructural canon.
        "borrow_check",
        "lifetime_outlives",
        "linear_use",
        "escape_region",
        "capability_perm",
        "refinement_liquid",
        # Wave-324 polyhedral-compiler canon.
        "fourier_motzkin",
        "banerjee_dep",
        "pluto_schedule",
        "tiling_legality",
        "omega_test",
        "vec_legality",
        # Wave-323 PL-7 effect/session-types canon.
        "free_monad",
        "alg_effects",
        "shift_reset",
        "row_types",
        "session_types",
        "gradual_types",
        # Wave-322 verification-2 canon.
        "weakest_precond",
        "sygus_synth",
        "horn_clauses",
        "interpolant_mc",
        "predicate_abs",
        "cegis_loop",
        # Wave-321 shape-analysis canon.
        "three_valued_logic",
        "shape_graph",
        "separation_logic",
        "context_pta",
        "recency_abstraction",
        "interproc_summary",
        # Wave-320 abstract-interpretation canon.
        "interval_analysis",
        "sign_domain",
        "zone_dbm",
        "affine_karr",
        "chaotic_widen",
        "andersen_pta",
        # Wave-319 proof-automation canon.
        "congruence_closure",
        "ring_normalize",
        "omega_lia",
        "nelson_oppen",
        "term_rewrite",
        "tseitin_cnf",
        # Wave-318 type-theory canon.
        "bidirectional_tc",
        "nbe_eval",
        "dep_types",
        "unify_meta",
        "proof_kernel",
        "tactic_engine",
        # Wave-317 image-processing canon.
        "canny_edge",
        "otsu_threshold",
        "watershed_seg",
        "slic_superpixels",
        "nlm_denoise",
        "distance_transform",
        # Wave-316 quantum-3 canon.
        "trotter_suzuki",
        "qdrift",
        "shadow_tomography",
        "vqd_states",
        "adapt_vqe",
        "hhl_lite",
        # Wave-315 robotics-5 canon.
        "rmpflow",
        "ds_motion",
        "wbc_qp",
        "grasp_epsilon",
        "rrt_connect",
        "dmp_control",
        # Wave-314 geometry-processing canon.
        "nurbs_eval",
        "catmull_clark",
        "loop_subdiv",
        "marching_cubes",
        "half_edge",
        "laplacian_smooth",
        # Wave-313 numerical-4/multigrid canon.
        "v_cycle",
        "amg_lite",
        "bicgstab",
        "minres",
        "chebyshev_iter",
        "ilu_precond",
        # Wave-312 distributed-4 canon.
        "hlc_clock",
        "delta_crdt",
        "raft_log",
        "bracha_bcast",
        "tot_order",
        "quorum_weighted",
        "abd_register",
        # Wave-311 VLSI-2 canon.
        "fm_partition",
        "lee_router",
        "clock_tree",
        "aig_rewrite",
        "power_est",
        "floorplan_sa",
        # Wave-310 quantum-error-correction canon.
        "gottesman_knill",
        "steane_code",
        "surface_code",
        "shor_code",
        "syndrome_circuit",
        "repetition_qec",
        # Wave-309 post-quantum-3 canon.
        "mceliece_lite",
        "bike_lite",
        "hqc_lite",
        "uov_sig",
        "rainbow_sig",
        "sidh_lite",
        # Wave-308 geophysics-2 canon.
        "reflectivity_synth",
        "gassmann_sub",
        "spectral_decomp",
        "semblance_scan",
        "gardner_relation",
        "vz_raytrace",
        # Wave-307 regex-2 canon.
        "pike_vm",
        "lazy_dfa",
        "bitap_fuzzy",
        "literal_prefilter",
        "glushkov_nfa",
        "regex_simplify",
        # Wave-306 astronomy-3/IOD canon.
        "laplace_iod",
        "cowell_j2",
        "batch_od",
        "cr3bp_dynamics",
        "porkchop_grid",
        "davenport_q",
        # Wave-305 text-index-2/stringology canon.
        "suffix_array_lcp",
        "z_function",
        "suffix_tree_lex",
        "booth_rotation",
        "lyndon_factor",
        "palindromic_tree",
        # Wave-304 computer-vision-2 canon.
        "harris_corner",
        "hough_lines",
        "integral_image",
        "seam_carving",
        "grabcut_lite",
        "meanshift_track",
        # Wave-303 speech/audio codec canon.
        "mulaw_compand",
        "adpcm_ima",
        "lpc_analysis",
        "celp_encode",
        "mel_cepstrum",
        "viterbi_vad",
        # Wave-302 compiler-5/JIT canon.
        "card_table_gc",
        "escape_analysis",
        "osr_deopt",
        "trace_tree",
        "ssa_repair",
        "gvn_pre",
        # Wave-301 geophysics/seismic canon.
        "nmo_dix",
        "taup_transform",
        "kirchhoff_mig",
        "avo_shuey",
        "vibroseis_sweep",
        "eikonal_fmm",
        # Wave-300 astronomy-2 canon.
        "equinox_prec",
        "nutation_lite",
        "rise_set",
        "eclipse_circ",
        "delta_t",
        "planet_vsop",
        # Wave-299 game-playing-2 canon.
        "tablebase_dtm",
        "retrograde_wdl",
        "rave_mc",
        "mast_playout",
        "expectimax",
        "isomcts",
        # Wave-298 medical-imaging canon.
        "radon_fbp",
        "art_sirt",
        "cs_mri",
        "hu_moments",
        "chan_vese",
        "mi_register",
        # Wave-297 post-quantum crypto canon.
        "ntt_ring",
        "kyber_kem",
        "dilithium_sig",
        "frodokem",
        "xmss_sig",
        "sphincs_sig",
        # Wave-296 robotics-4 canon.
        "lqr_funnel",
        "chomp",
        "gjk_epa",
        "ilqr",
        "rts_smoother",
        "se3_spline",
        # Wave-295 astronomy/orbital-mechanics canon.
        "orbital_elements",
        "kepler_solve",
        "lambert_problem",
        "tle_propagate",
        "orbit_maneuver",
        "gauss_iod",
        # Wave-294 compiler-4 canon.
        "tree_cover",
        "modulo_sched",
        "jump_thread",
        "tail_dup",
        "cfg_simplify",
        "bb_reorder",
        # Wave-293 graphics-3 canon.
        "deferred_shade",
        "sdf_raymarch",
        "frustum_cull",
        "lod_select",
        "env_map",
        "shadow_pcf",
        # Wave-292 networking-4 canon.
        "quic_streams",
        "tls13_trans",
        "qpack_pack",
        "wg_ik",
        "doh_wire",
        "sctp_tsn",
        # Wave-291 VLSI/EDA canon.
        "netlist_parse",
        "sta_timing",
        "a_star_route",
        "drc_check",
        "place_quadratic",
        "levelize",
        # Wave-290 chem-informatics canon.
        "smiles_parse",
        "morgan_fp",
        "tanimoto",
        "mol_descriptors",
        "substruct",
        "ring_detect",
        # Wave-289 category-theory canon.
        "fin_cat",
        "functor_check",
        "nat_trans",
        "adjunction",
        "limit_prod",
        "monad_laws",
        # Wave-288 measure-theory canon.
        "leb_measure",
        "leb_integral",
        "conv_prob",
        "weak_conv",
        "fubini_swap",
        "radon_nikodym",
        # Wave-287 differential-geometry canon.
        "first_ff",
        "gauss_curve",
        "frenet_frame",
        "christoffel",
        "geodesic_sphere",
        "surf_area",
        # Wave-286 security-defensive canon.
        "beacon_detect",
        "entropy_dns",
        "cred_stuffing",
        "impossible_travel",
        "exfil_zscore",
        "sig_score",
        # Wave-285 information-theory canon.
        "markov_entropy",
        "blahut_arimoto",
        "kl_knn",
        "type_class",
        "elias_gamma",
        "miller_madow",
        # Wave-284 bioinformatics-3 canon.
        "nj_tree",
        "fitch_pars",
        "seed_extend",
        "band_align",
        "jc69_lik",
        "codon_usage",
        # Wave-283 robotics-3 canon.
        "fk_dh",
        "ik_jac",
        "ray_lidar",
        "pot_field",
        "bezier_curve",
        "odom_comp",
        "divide_conquer_eig",
        "dqds",
        "block_lanczos",
        "randomized_qb",
        "sparse_cholesky",
        "fgmres",
        "edf_scheduler",
        "rms_scheduler",
        "wcet_est",
        "debounce_fsm",
        "watchdog_task",
        "ring_buffer",
        "stencil_halo",
        "mesi_cache",
        "ring_allreduce",
        "simd_lanes",
        "task_dag",
        "numa_alloc",
        "ekf_slam",
        "occupancy_grid",
        "pure_pursuit",
        "stanley",
        "particle_slam",
        "frontier_explore",
        "gale_shapley",
        "hopcroft_karp",
        "kuhn_munkres",
        "konig_cover",
        "gale_chu",
        "topo_layers",
        "critical_path",
        "dinic_flow",
        "mincost_flow",
        "needleman_wunsch",
        "smith_waterman",
        "upgma_tree",
        "dfa_equiv",
        "mealy_moore",
        "pda_sim",
        "turing_machine",
        "peterson_lock",
        "rw_lock",
        "work_stealing",
        "tail_call_tramp",
        "threaded_interp",
        "posting_merge",
        "wand_bmw",
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
