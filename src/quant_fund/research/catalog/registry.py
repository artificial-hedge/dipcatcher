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
