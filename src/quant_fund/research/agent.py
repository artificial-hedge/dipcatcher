"""Dipcatcher — Artificial Hedge's proprietary research lab runner.

Families: ranking, alpha, volatility, distribution, regime, tail,
drawdown, liquidity, contextual bandits, conformal, e-values,
Jackknife+, CRC, weighted CQR, interval caps, quantile Thompson,
Northset (order book + candlesticks). robinhood+ is a core forecast
engine (ADR-023) and an optional catalog family via
``dipcatcher robinhood-plus``.
Proper scores only — no Sharpe.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund import __firm__, __version__
from quant_fund.config.models import AppConfig
from quant_fund.metrics.inference import (
    benjamini_hochberg,
    diebold_mariano,
    mean_difference_t,
    mean_tstat,
    onesided_from_twosided,
    two_proportion_test,
)
from quant_fund.metrics.overfitting import overfitting_diagnostics
from quant_fund.northset.benches import bench_northset
from quant_fund.pipeline.dataset import build_gold, ensure_silver, panel
from quant_fund.reporting.report import latest_report_dir, write_report
from quant_fund.research.benches import (
    bench_alpha,
    bench_conformal,
    bench_conformal_topk_from_panel,
    bench_cpcv_audit,
    bench_crc,
    bench_cv_plus,
    bench_distribution,
    bench_drawdown,
    bench_evalues,
    bench_interval_risk,
    bench_jackknife_plus,
    bench_liquidity,
    bench_localized_from_panel,
    bench_online_crc_from_panel,
    bench_portfolio_from_panel,
    bench_quantile_bandit,
    bench_ranking,
    bench_regime,
    bench_rl,
    bench_tail,
    bench_volatility,
    bench_weighted_conformal,
)
from quant_fund.research.benches_extra import (
    bench_complexity,
    bench_roughness,
    bench_serial_randomness,
)
from quant_fund.research.benches_w11 import (
    bench_conformal_pid,
    bench_dro,
    bench_optimal_transport,
    bench_rough_paths,
)
from quant_fund.research.benches_w12 import (
    bench_confidence_sequences,
    bench_deep_hedging,
    bench_mh_enbpi,
    bench_nexcp,
    bench_rwcv,
    bench_score_decomposition,
    bench_sliced_wasserstein,
    bench_stacking,
)
from quant_fund.research.benches_w13 import (
    bench_conformal_e_detectors,
    bench_coverage_inference,
    bench_delayed_aci,
    bench_picpi,
    bench_rank_cs,
    bench_reference_null,
    bench_replicable_conformal,
    bench_rolling_conformal,
)
from quant_fund.research.benches_w14 import (
    bench_american_lsm,
    bench_cash_constrained_oe,
    bench_deep_bsde,
    bench_deep_kernel_hedging,
    bench_deep_regime_mixture,
    bench_large_deviations,
    bench_local_stoch_vol,
    bench_malliavin_greeks,
    bench_martingale_ot,
    bench_mean_field_games,
    bench_odd_residual_flows,
    bench_vine_copula,
    bench_xva,
    bench_zi_lob,
)
from quant_fund.research.benches_w15 import (
    bench_agent_referee,
    bench_capability_value,
    bench_conformal_transfer,
    bench_entropy_shapley,
    bench_fourier_pricing,
    bench_hpd_conformal,
    bench_subspace_denoising,
    bench_vintage_eval,
)
from quant_fund.research.benches_w16 import (
    bench_extra_tilt,
    bench_forecast_selection,
    bench_hierarchical_conformal,
    bench_multisource_conformal,
    bench_rl_market_maker,
    bench_vol_loss_decomposition,
)
from quant_fund.research.benches_w17 import (
    bench_adaptive_eps,
    bench_conformal_oce,
    bench_diffpts,
    bench_diffusion_forecaster,
    bench_extra_conformal,
    bench_greek_neutral,
    bench_multilevel_mm,
    bench_passive_impact,
    bench_rlmm_c51,
    bench_sga_uq,
    bench_stochastic_tracking,
)
from quant_fund.research.benches_w18 import (
    bench_agentic_lob,
    bench_fase_eval,
    bench_gslice,
    bench_kit_paths,
    bench_neural_sde,
    bench_stocbench,
)
from quant_fund.research.benches_w19 import (
    bench_dcp,
    bench_event_time_flow,
    bench_fukasawa_iv,
    bench_ivs_diffusion,
    bench_langevin_impact,
    bench_rccp,
)
from quant_fund.research.benches_w20 import (
    bench_arl_mm,
    bench_gaussian_normalized_coords,
    bench_hidden_markov_equilibrium,
    bench_liquidity_tail_lob,
    bench_varswap_stopping,
)
from quant_fund.research.benches_w21 import (
    bench_bocpd_changepoint,
    bench_rough_heston_rbergomi,
    bench_signature_features,
)
from quant_fund.research.benches_w22 import (
    bench_signature_martingale_test,
)
from quant_fund.research.benches_w23 import (
    bench_koopman_edmd,
    bench_neural_tpp,
    bench_propagator_impact,
    bench_queue_reactive,
    bench_sig_gan,
    bench_svi_surface,
)
from quant_fund.research.benches_w24 import (
    bench_breeden_litzenberger,
    bench_fernholz_spt,
    bench_hawkes_em,
    bench_multifractal_vol,
    bench_pmcmc_sv,
    bench_spci_conformal,
)
from quant_fund.research.benches_w25 import (
    bench_fourier_hermite,
    bench_fractional_ou,
    bench_heterogeneous_abm,
    bench_kernel_changepoint,
    bench_kinetic_ising,
    bench_tda_persistence,
)
from quant_fund.research.benches_w26 import (
    bench_factor_nowcast,
    bench_marchenko_pastur,
    bench_modularity_communities,
    bench_skill_ratings,
    bench_stationary_bootstrap,
    bench_stein_thinning,
)
from quant_fund.research.benches_w27 import (
    bench_dp_mixture,
    bench_durbin_koopman,
    bench_ensemble_kalman_inversion,
    bench_expert_aggregation,
    bench_implied_tree,
    bench_instrumental_quantile,
)
from quant_fund.research.benches_w28 import (
    bench_callaway_did,
    bench_causal_discovery,
    bench_enkf,
    bench_knockoffs,
    bench_sindy,
    bench_surrogate_nonlinear,
)
from quant_fund.research.benches_w29 import (
    bench_hj_distance,
    bench_hmc,
    bench_proxy_svar,
    bench_rqa,
    bench_sbi,
    bench_tensor_decomp,
)
from quant_fund.research.benches_w30 import (
    bench_bispectrum,
    bench_count_data,
    bench_functional_linear,
    bench_gas_score,
    bench_lp_iv,
    bench_lyapunov,
)
from quant_fund.research.benches_w31 import (
    bench_bounds,
    bench_heckman,
    bench_kernel_iv,
    bench_multistate,
    bench_partial_linear,
    bench_rd,
)
from quant_fund.research.benches_w32 import (
    bench_bunching,
    bench_causal_forest,
    bench_double_ml,
    bench_extreme_value,
    bench_gaussian_process,
    bench_markov_switching,
)
from quant_fund.research.benches_w33 import (
    bench_causal_impact,
    bench_cluster_robust,
    bench_permutation_inference,
    bench_propensity_score,
    bench_synth_did,
    bench_weak_iv,
)
from quant_fund.research.benches_w34 import (
    bench_did_diagnostics,
    bench_entropy_balancing,
    bench_gsynth,
    bench_matrix_completion,
    bench_rif_regression,
    bench_shift_share,
)
from quant_fund.research.benches_w35 import (
    bench_fama_macbeth,
    bench_honest_did,
    bench_many_iv,
    bench_panel_quantile_fe,
    bench_sign_restricted_var,
    bench_specification_curve,
)
from quant_fund.research.benches_w36 import (
    bench_arellano_bond,
    bench_bvar_minnesota,
    bench_competing_risks,
    bench_mediation_analysis,
    bench_regression_kink,
    bench_stochastic_frontier,
)
from quant_fund.research.benches_w37 import (
    bench_distribution_regression,
    bench_lp_did,
    bench_ordered_choice,
    bench_simex,
    bench_spatial_econometrics,
    bench_triple_difference,
)
from quant_fund.research.benches_w38 import (
    bench_censored_quantile,
    bench_control_function,
    bench_fractional_response,
    bench_interval_censoring,
    bench_kernel_regression,
    bench_threshold_ar,
)
from quant_fund.research.benches_w39 import (
    bench_aft_model,
    bench_distance_covariance,
    bench_maximum_score,
    bench_nested_logit,
    bench_panel_unitroot,
    bench_sieve_estimation,
)
from quant_fund.research.benches_w40 import (
    bench_connectedness,
    bench_hurdle,
    bench_mixed_logit,
    bench_nonparametric_iv,
    bench_subsampling,
    bench_sur_model,
)
from quant_fund.research.benches_w41 import (
    bench_event_study,
    bench_frailty,
    bench_interrupted_ts,
    bench_lead_lag,
    bench_model_averaging,
    bench_ppml,
)
from quant_fund.research.benches_w42 import (
    bench_clark_west,
    bench_har_rv,
    bench_roy_model,
    bench_stambaugh,
    bench_two_way_cluster,
    bench_variance_ratio,
)
from quant_fund.research.benches_w43 import (
    bench_bai_perron,
    bench_favar,
    bench_merton_model,
    bench_panel_coint,
    bench_vuong_test,
    bench_white_reality,
)
from quant_fund.research.benches_w44 import (
    bench_johansen_vecm,
    bench_kiefer_vogelsang,
    bench_kyle_lambda,
    bench_oster_bounds,
    bench_pin_model,
    bench_storey_fdr,
)
from quant_fund.research.benches_w45 import (
    bench_barrett_donald,
    bench_conley_se,
    bench_driscoll_kraay,
    bench_lee_bounds,
    bench_pesaran_cce,
    bench_wald_sprt,
)
from quant_fund.research.benches_w46 import (
    bench_binscatter,
    bench_blp_demand,
    bench_dfl_decomp,
    bench_oaxaca_blinder,
    bench_olley_pakes,
    bench_rust_ddc,
)
from quant_fund.research.benches_w47 import (
    bench_aipw_ate,
    bench_cavi_gmm,
    bench_cusum_monitor,
    bench_hausman_tests,
    bench_pesaran_cd,
    bench_rosenbaum_sensitivity,
)
from quant_fund.research.benches_w48 import (
    bench_cover_up,
    bench_lewbel_iv,
    bench_marginal_treatment,
    bench_proximal_causal,
    bench_tmle,
    bench_vpin,
)
from quant_fund.research.benches_w49 import (
    bench_blanchard_quah,
    bench_delta_covar,
    bench_eisenberg_noe,
    bench_fire_sales,
    bench_meta_analysis,
    bench_tvp_var,
)
from quant_fund.research.benches_w50 import (
    bench_acd_duration,
    bench_affine_term,
    bench_dea,
    bench_gil_pelaez,
    bench_hedonic,
    bench_hjm,
)
from quant_fund.research.benches_w51 import (
    bench_ait_sahalia,
    bench_bkm_moments,
    bench_gsadf_bubble,
    bench_pmg_ardl,
    bench_ross_recovery,
    bench_toda_yamamoto,
)
from quant_fund.research.benches_w52 import (
    bench_bandi_russell,
    bench_beveridge_nelson,
    bench_growth_at_risk,
    bench_hong_li,
    bench_melick_thomas,
    bench_nardl,
)
from quant_fund.research.benches_w53 import (
    bench_christoffersen_pelletier,
    bench_corradi_swanson,
    bench_engle_kroner_bekk,
    bench_heston_qe,
    bench_model_confidence_set,
    bench_sheppard_heavy,
)
from quant_fund.research.benches_w54 import (
    bench_christensen_diebold_rudebusch,
    bench_danielsson_devries,
    bench_giacomini_rossi,
    bench_muller_watson,
    bench_pesaran_timmermann,
    bench_romano_wolf,
)
from quant_fund.research.benches_w55 import (
    bench_bds,
    bench_cochrane_piazzesi,
    bench_diebold_mariano,
    bench_engle_granger,
    bench_engle_ng,
    bench_glosten_milgrom,
    bench_hasbrouck_is,
)
from quant_fund.research.benches_w56 import (
    bench_ers_dfgls,
    bench_kpss,
    bench_lee_strazicich,
    bench_ng_perron,
    bench_phillips_perron,
    bench_zivot_andrews,
)
from quant_fund.research.benches_w57 import (
    bench_bai_ng_ic,
    bench_geweke_spectral,
    bench_ivx,
    bench_lasso_pds,
    bench_wavelet_modwt,
    bench_wooldridge_serial,
)
from quant_fund.research.benches_w58 import (
    bench_bates_svj,
    bench_echo_state,
    bench_extremogram,
    bench_pelt_wbs,
    bench_spectral_pca,
    bench_stl_loess,
)
from quant_fund.research.benches_w59 import (
    bench_emd_hht,
    bench_extreme_qr,
    bench_gallant_snp,
    bench_srisk,
    bench_tar_coint,
    bench_wavelet_coherence,
)
from quant_fund.research.benches_w60 import (
    bench_fractional_coint,
    bench_garch_in_mean,
    bench_kalman_em,
    bench_log_acd,
    bench_star_model,
    bench_wigner_ville,
)
from quant_fund.research.benches_w61 import (
    bench_beta_ar,
    bench_chen_tiao_outliers,
    bench_chow_lin,
    bench_dtw_warp,
    bench_ingarch,
    bench_log_concave,
)
from quant_fund.research.benches_w62 import (
    bench_asian_option,
    bench_bfast,
    bench_black_litterman,
    bench_entropy_pooling,
    bench_narrative_svar,
    bench_stable_dist,
)
from quant_fund.research.benches_w63 import (
    bench_convexity_adj,
    bench_first_passage,
    bench_higham_corr,
    bench_jln_uncertainty,
    bench_lee_carter,
    bench_power_law,
)
from quant_fund.research.benches_w64 import (
    bench_brownian_bridge,
    bench_campbell_shiller,
    bench_jarrow_turnbull,
    bench_mutual_info,
    bench_saddlepoint,
    bench_transfer_entropy,
)
from quant_fund.research.benches_w65 import (
    bench_cos_method,
    bench_esscher,
    bench_libor_market,
    bench_obizhaeva_wang,
    bench_shadow_rate,
    bench_spread_options,
)
from quant_fund.research.benches_w66 import (
    bench_black_karasinski,
    bench_debtrank,
    bench_ews_signals,
    bench_mlmc,
    bench_permutation_entropy,
    bench_svgd,
)
from quant_fund.research.benches_w67 import (
    bench_euler_risk,
    bench_moment_inequalities,
    bench_nested_sampling,
    bench_smc_samplers,
    bench_state_dependent_lp,
    bench_synthetic_likelihood,
)
from quant_fund.research.benches_w68 import (
    bench_functional_pca,
    bench_mala,
    bench_particle_gibbs,
    bench_ripley_k,
    bench_synchrosqueezing,
    bench_vix_replication,
)
from quant_fund.research.benches_w69 import (
    bench_cca,
    bench_innovations_ets,
    bench_isomap,
    bench_kriging,
    bench_stockwell,
    bench_vmd,
)
from quant_fund.research.benches_w70 import (
    bench_compositional,
    bench_cyclostationary,
    bench_empirical_wavelets,
    bench_nonparametric_tests,
    bench_polychoric,
    bench_sure_screening,
)
from quant_fund.research.benches_w71 import (
    bench_hrp,
    bench_isotonic,
    bench_item_response,
    bench_latent_class,
    bench_manova,
    bench_procrustes,
)
from quant_fund.research.benches_w72 import (
    bench_factor_analysis,
    bench_james_stein,
    bench_kuiper,
    bench_p_spline,
    bench_slice_sampling,
    bench_thin_plate,
)
from quant_fund.research.benches_w73 import (
    bench_contingency,
    bench_friedman,
    bench_hoeffding,
    bench_mantel,
    bench_mardia,
    bench_moran,
)
from quant_fund.research.benches_w74 import (
    bench_cochran_q,
    bench_dispersion_tests,
    bench_dunn_test,
    bench_median_tests,
    bench_quade,
    bench_van_der_waerden,
)
from quant_fund.research.benches_w75 import (
    bench_cronbach,
    bench_dif,
    bench_g_theory,
    bench_icc,
    bench_omega,
    bench_rasch_fit,
)
from quant_fund.research.benches_w76 import (
    bench_calibration_survey,
    bench_cluster_sampling,
    bench_design_effects,
    bench_fay_herriot,
    bench_horvitz_thompson,
    bench_poststrat,
)
from quant_fund.research.benches_w77 import (
    bench_gee,
    bench_hegy,
    bench_interrater,
    bench_isolation_forest,
    bench_lmm,
    bench_mice,
)
from quant_fund.research.benches_w78 import (
    bench_e_divisive,
    bench_mst_topology,
    bench_multiple_comparisons,
    bench_rank_aggregation,
    bench_risk_parity,
    bench_spc,
)
from quant_fund.research.benches_w79 import (
    bench_auxiliary_pf,
    bench_barrier_options,
    bench_fastica,
    bench_msm_causal,
    bench_nmf,
    bench_tost,
)
from quant_fund.research.benches_w80 import (
    bench_energy_test,
    bench_epps_singleton,
    bench_henze_zirkler,
    bench_ois_curve,
    bench_rmst,
    bench_watson,
)
from quant_fund.research.benches_w81 import (
    bench_dirichlet_multinomial,
    bench_fkml,
    bench_gandh,
    bench_pocs,
    bench_robbins_monro,
    bench_vonmises_fisher,
)
from quant_fund.research.benches_w82 import (
    bench_circular_tests,
    bench_delong_auc,
    bench_graded_irt,
    bench_marginal_homogeneity,
    bench_passing_bablok,
    bench_welch_anova,
)
from quant_fund.research.benches_w83 import (
    bench_circular_correlation,
    bench_correspondence_analysis,
    bench_group_sequential,
    bench_influence,
    bench_lin_ccc,
    bench_mds,
)
from quant_fund.research.benches_w84 import (
    bench_dawid_skene,
    bench_hierarchical_reconciliation,
    bench_lmoments,
    bench_matrix_profile,
    bench_recurrent_events,
    bench_sobol_sensitivity,
)
from quant_fund.research.benches_w85 import (
    bench_chain_ladder,
    bench_cma_es,
    bench_erlang_queueing,
    bench_inequality_indices,
    bench_music_esprit,
    bench_sketches,
)
from quant_fund.research.benches_w86 import (
    bench_bayesian_tracking,
    bench_brinson_attribution,
    bench_gr4j_hydrology,
    bench_pu_learning,
    bench_rainflow_fatigue,
    bench_sbm_inference,
)
from quant_fund.research.benches_w87 import (
    bench_avellaneda_stoikov,
    bench_corwin_schultz,
    bench_gillespie_ssa,
    bench_gwr_spatial,
    bench_hamilton_filter,
    bench_pareto_nbd,
)
from quant_fund.research.benches_w88 import (
    bench_beck_katz,
    bench_friedman_supersmoother,
    bench_lomb_scargle,
    bench_markov_discretization,
    bench_quandt_andrews,
    bench_singular_spectrum,
)
from quant_fund.research.benches_w89 import (
    bench_edf_tests,
    bench_het_regressions,
    bench_normality_tests,
    bench_scale_homogeneity,
    bench_score_scale,
    bench_serial_diagnostics,
)
from quant_fund.research.benches_w90 import (
    bench_ace_avas,
    bench_blind_sources,
    bench_marginal_likelihood,
    bench_mars_regression,
    bench_projection_pursuit,
    bench_root_finders,
)
from quant_fund.research.benches_w91 import (
    bench_clustering_methods,
    bench_design_experiments,
    bench_empirical_bayes,
    bench_manifold_learning,
    bench_robust_regression,
    bench_unconstrained_optimizers,
)
from quant_fund.research.benches_w92 import (
    bench_bayesian_optimization,
    bench_fuzzy_clustering,
    bench_hyperband_search,
    bench_metaheuristic_optimizers,
    bench_pagerank_topology,
    bench_self_organizing_maps,
)
from quant_fund.research.benches_w93 import (
    bench_factorization_machine,
    bench_gp_classification,
    bench_metric_learning,
    bench_one_class_classification,
    bench_phase_retrieval,
    bench_tree_ensembles,
)
from quant_fund.research.benches_w94 import (
    bench_coordinate_descent_enet_family,
    bench_discriminant_analysis,
    bench_kernel_methods_family,
    bench_lda_topics_family,
    bench_online_convex_family,
    bench_svm_classifiers,
)
from quant_fund.research.benches_w95 import (
    bench_bayesian_linear_family,
    bench_conjugate_gradient_family,
    bench_evolution_strategies_family,
    bench_frank_wolfe_family,
    bench_graphical_models_family,
    bench_sparse_coding_family,
)
from quant_fund.research.benches_w96 import (
    bench_adaboost_family,
    bench_association_rules_family,
    bench_collaborative_filtering_family,
    bench_expectation_propagation_family,
    bench_naive_bayes_family,
    bench_nash_equilibrium_family,
)
from quant_fund.research.benches_w97 import (
    bench_multiclass_family,
    bench_ode_solvers_family,
    bench_proximal_gradient_family,
    bench_quadrature_family,
    bench_semisupervised_family,
    bench_sparse_pca_family,
)
from quant_fund.research.benches_w98 import (
    bench_belief_propagation_family,
    bench_cross_entropy_method_family,
    bench_extreme_learning_family,
    bench_mdp_solvers_family,
    bench_nearest_centroid_family,
    bench_td_learning_family,
)
from quant_fund.research.benches_w99 import (
    bench_evidential_family,
    bench_gibbs_sampler_family,
    bench_kde_family,
    bench_laplace_approx_family,
    bench_multi_task_family,
    bench_tensor_power_family,
)
from quant_fund.research.benches_w100 import (
    bench_anderson_accel_family,
    bench_homotopy_continuation_family,
    bench_iterative_ls_family,
    bench_qmc_sequences_family,
    bench_sequence_accel_family,
    bench_symplectic_ode_family,
)
from quant_fund.research.benches_w101 import (
    bench_assignment_family,
    bench_exact_cover_family,
    bench_graph_components_family,
    bench_graph_traversal_family,
    bench_network_flow_family,
    bench_shortest_paths_family,
)
from quant_fund.research.benches_w102 import (
    bench_adversarial_bandits_family,
    bench_best_arm_family,
    bench_contextual_bandits_family,
    bench_kl_bandits_family,
    bench_nonstationary_bandits_family,
    bench_stochastic_bandits_family,
)
from quant_fund.research.benches_w103 import (
    bench_arnoldi_gmres_family,
    bench_cur_decomp_family,
    bench_interpolative_decomp_family,
    bench_lanczos_family,
    bench_nystrom_family,
    bench_randomized_svd_family,
)
from quant_fund.research.benches_w104 import (
    bench_emd_lp_family,
    bench_fused_gromov_family,
    bench_gromov_wasserstein_family,
    bench_sinkhorn_family,
    bench_unbalanced_ot_family,
    bench_wasserstein_barycenter_family,
)
from quant_fund.research.benches_w105 import (
    bench_alpha_beta_family,
    bench_dfpn_family,
    bench_mcts_family,
    bench_negascout_family,
    bench_proof_number_family,
    bench_puct_family,
)
from quant_fund.research.benches_w106 import (
    bench_adams_family,
    bench_bdf_family,
    bench_crank_nicolson_family,
    bench_etdrk4_family,
    bench_radau_family,
    bench_strang_family,
)
from quant_fund.research.benches_w107 import (
    bench_adi_family,
    bench_fast_marching_family,
    bench_godunov_family,
    bench_lax_wendroff_family,
    bench_level_set_family,
    bench_weno_family,
)
from quant_fund.research.benches_w108 import (
    bench_expm_pade_family,
    bench_matrix_sign_family,
    bench_matrix_sqrt_family,
    bench_riccati_care_family,
    bench_sylvester_family,
    bench_toeplitz_solve_family,
)
from quant_fund.research.benches_w109 import (
    bench_dubins_family,
    bench_dwa_family,
    bench_frenet_family,
    bench_min_snap_family,
    bench_prm_family,
    bench_rrt_family,
)
from quant_fund.research.benches_w110 import (
    bench_costas_family,
    bench_gardner_family,
    bench_gf256_family,
    bench_reed_solomon_family,
    bench_rrc_filter_family,
    bench_viterbi_decode_family,
)
from quant_fund.research.benches_w111 import (
    bench_convex_hull_family,
    bench_delaunay_family,
    bench_frechet_family,
    bench_hausdorff_family,
    bench_icp_family,
    bench_kabsch_family,
)
from quant_fund.research.benches_w112 import (
    bench_biquad_family,
    bench_farrow_family,
    bench_filtfilt_family,
    bench_iir_design_family,
    bench_remez_family,
    bench_resample_poly_family,
)
from quant_fund.research.benches_w113 import (
    bench_cov_int_family,
    bench_jonker_volgenant_family,
    bench_jpda_family,
    bench_mht_family,
    bench_phd_family,
    bench_tdoa_family,
)
from quant_fund.research.benches_w114 import (
    bench_allan_variance_family,
    bench_gold_code_family,
    bench_klobuchar_family,
    bench_lambda_method_family,
    bench_rtk_family,
    bench_strapdown_family,
)
from quant_fund.research.benches_w115 import (
    bench_active_subspace_family,
    bench_bayesian_quadrature_family,
    bench_kl_expand_family,
    bench_mimc_family,
    bench_pce_family,
    bench_smolyak_family,
)
from quant_fund.research.benches_w116 import (
    bench_cfr_family,
    bench_lemke_howson_family,
    bench_nash_bargain_family,
    bench_replicator_family,
    bench_vcg_family,
    bench_wardrop_family,
)
from quant_fund.research.benches_w117 import (
    bench_ddpg_family,
    bench_gae_family,
    bench_ppo_family,
    bench_td3_family,
    bench_trpo_family,
    bench_vtrace_family,
)
from quant_fund.research.benches_w118 import (
    bench_grid_pomdp_family,
    bench_hsvi_family,
    bench_pbvi_family,
    bench_perseus_family,
    bench_pomcp_family,
    bench_qmdp_family,
)
from quant_fund.research.benches_w119 import (
    bench_coma_family,
    bench_maddpg_family,
    bench_mappo_family,
    bench_mf_q_family,
    bench_qmix_family,
    bench_vdn_family,
)
from quant_fund.research.benches_w120 import (
    bench_banzhaf_family,
    bench_envy_free_family,
    bench_groves_family,
    bench_myerson_auction_family,
    bench_nucleolus_family,
    bench_owen_family,
)
from quant_fund.research.benches_w121 import (
    bench_lil_ucb_family,
    bench_median_elim_family,
    bench_sequential_halving_family,
    bench_track_stop_family,
    bench_ttts_family,
    bench_ugape_family,
)
from quant_fund.research.benches_w122 import (
    bench_dark_pool_family,
    bench_exec_rl_family,
    bench_options_flow_family,
    bench_order_flow_imbalance_family,
    bench_pg_mm_family,
    bench_smart_router_family,
)
from quant_fund.research.benches_w123 import (
    bench_econ_calendar_family,
    bench_multimodal_fusion_family,
    bench_quantcode_bench_family,
    bench_say_echo_do_family,
    bench_synthetic_gan_family,
    bench_ts_diffusion_family,
)
from quant_fund.research.benches_w124 import (
    bench_asset_gnn_family,
    bench_continual_learning_family,
    bench_counterparty_gnn_family,
    bench_fed_avg_family,
    bench_insider_anomaly_family,
    bench_maml_portfolio_family,
)
from quant_fund.research.benches_w125 import (
    bench_adversarial_robust_family,
    bench_causal_miner_family,
    bench_pinn_pricing_family,
    bench_qubo_portfolio_family,
    bench_risk_flow_family,
    bench_xai_shap_family,
)
from quant_fund.research.benches_w126 import (
    bench_lob_transformer_family,
    bench_neural_ode_family,
    bench_patchtst_family,
    bench_set_transformer_family,
    bench_tft_forecaster_family,
    bench_world_model_family,
)
from quant_fund.research.benches_w127 import (
    bench_bnn_ensemble_family,
    bench_contrastive_repr_family,
    bench_diff_policy_family,
    bench_hypernetwork_alloc_family,
    bench_neural_thompson_family,
    bench_option_vae_family,
)
from quant_fund.research.benches_w128 import (
    bench_cnn_alpha_family,
    bench_graph_temporal_family,
    bench_informer_attn_family,
    bench_kan_forecaster_family,
    bench_mask_autoencoder_family,
    bench_ts_mixer_family,
)
from quant_fund.research.benches_w129 import (
    bench_crossformer_family,
    bench_ft_transformer_family,
    bench_itransformer_family,
    bench_mambats_family,
    bench_nbeats_deep_family,
    bench_tcn_forecaster_family,
)
from quant_fund.research.benches_w130 import (
    bench_cql_agent_family,
    bench_decision_transformer_family,
    bench_gail_imitation_family,
    bench_iql_agent_family,
    bench_sac_agent_family,
    bench_trajectory_transformer_family,
)
from quant_fund.research.benches_w131 import (
    bench_consistency_ts_family,
    bench_energy_ts_family,
    bench_flow_matching_ts_family,
    bench_perceiver_ts_family,
    bench_score_sde_ts_family,
    bench_vq_vae_ts_family,
)
from quant_fund.research.benches_w132 import (
    bench_energy_ood_family,
    bench_gradient_norm_ood_family,
    bench_knn_ood_family,
    bench_mahalanobis_ood_family,
    bench_max_softmax_ood_family,
    bench_vim_ood_family,
)
from quant_fund.research.benches_w133 import (
    bench_attentive_np_family,
    bench_convnp_family,
    bench_deep_kernel_gp_family,
    bench_llaplace_gp_family,
    bench_meta_uq_family,
    bench_neural_process_family,
)
from quant_fund.research.benches_w134 import (
    bench_cvxpy_layer_family,
    bench_deep_declarative_family,
    bench_diff_mpc_family,
    bench_input_convex_family,
    bench_optnet_qp_family,
    bench_spd_net_family,
)
from quant_fund.research.benches_w135 import (
    bench_apnp_prop_family,
    bench_chebnet_family,
    bench_gin_gnn_family,
    bench_graph_unet_family,
    bench_graphsage_family,
    bench_jk_net_family,
)
from quant_fund.research.benches_w136 import (
    bench_linear_attn_family,
    bench_linformer_attn_family,
    bench_nystrom_attn_family,
    bench_performer_attn_family,
    bench_sinkhorn_attn_family,
    bench_sliding_attn_family,
)
from quant_fund.research.benches_w137 import (
    bench_bootstrapped_dqn_family,
    bench_c51_dqn_family,
    bench_iqn_dqn_family,
    bench_noisy_net_family,
    bench_prioritized_replay_family,
    bench_qr_dqn_family,
)
from quant_fund.research.benches_w138 import (
    bench_dnc_memory_family,
    bench_memorizing_transformer_family,
    bench_mpc_planning_family,
    bench_ntm_memory_family,
    bench_reformer_lsh_family,
    bench_rssm_world_family,
)
from quant_fund.research.benches_w139 import (
    bench_barlow_twins_family,
    bench_byol_family,
    bench_shot_tta_family,
    bench_tent_tta_family,
    bench_ttt_layer_family,
    bench_vicreg_family,
)
from quant_fund.research.benches_w140 import (
    bench_capsule_dynamic_family,
    bench_equivar_gnn_family,
    bench_hyperbolic_nn_family,
    bench_monotonic_net_family,
    bench_siren_inr_family,
    bench_sort_net_family,
)
from quant_fund.research.benches_w141 import (
    bench_crown_bound_family,
    bench_gumbel_topk_family,
    bench_ibp_bounds_family,
    bench_lipschitz_net_family,
    bench_randomized_smoothing_family,
    bench_vector_neurons_family,
)
from quant_fund.research.benches_w142 import (
    bench_delta_net_family,
    bench_hyena_conv_family,
    bench_mixture_of_depths_family,
    bench_retnet_decay_family,
    bench_rwkv_wkv_family,
    bench_s4_ssm_family,
)
from quant_fund.research.benches_w143 import (
    bench_dora_weight_family,
    bench_lora_ft_family,
    bench_prefix_tuning_family,
    bench_prompt_tuning_family,
    bench_qlora_nf4_family,
    bench_task_vector_merge_family,
)
from quant_fund.research.benches_w144 import (
    bench_flash_attn_family,
    bench_gqa_attn_family,
    bench_paged_kv_cache_family,
    bench_ring_attn_family,
    bench_sliding_window_cache_family,
    bench_speculative_decoding_family,
)
from quant_fund.research.benches_w145 import (
    bench_bm25_retriever_family,
    bench_colbert_late_family,
    bench_dpr_retriever_family,
    bench_hyde_retrieval_family,
    bench_reranker_crossenc_family,
    bench_rrf_fusion_family,
)
from quant_fund.research.benches_w146 import (
    bench_consistency_vote_family,
    bench_debate_multiagent_family,
    bench_knowledge_graph_embed_family,
    bench_mcts_reason_family,
    bench_unlearn_ga_family,
    bench_verifier_prm_family,
)
from quant_fund.research.benches_w147 import (
    bench_dpo_train_family,
    bench_grpo_train_family,
    bench_ipo_train_family,
    bench_kto_train_family,
    bench_reward_model_family,
    bench_rlhf_ppo_family,
)
from quant_fund.research.benches_w148 import (
    bench_activation_steering_family,
    bench_circuit_ablation_family,
    bench_logit_lens_family,
    bench_patch_activation_family,
    bench_probe_linear_family,
    bench_sae_feature_family,
)
from quant_fund.research.benches_w149 import (
    bench_coreset_herding_family,
    bench_curriculum_magnitude_family,
    bench_dataset_distillation_family,
    bench_label_smoothing_family,
    bench_mixup_cutmix_family,
    bench_sharpness_sam_family,
)
from quant_fund.research.benches_w150 import (
    bench_fisher_prune_family,
    bench_kd_distill_family,
    bench_lottery_ticket_family,
    bench_lowrank_factor_family,
    bench_magnitude_pruning_family,
    bench_quant_int8_family,
)
from quant_fund.research.benches_w151 import (
    bench_judge_pairwise_family,
    bench_multi_agent_pipeline_family,
    bench_plan_search_family,
    bench_react_loop_family,
    bench_reflexion_retry_family,
    bench_toolformer_call_family,
)
from quant_fund.research.benches_w152 import (
    bench_canary_exposure_family,
    bench_dp_sgd_family,
    bench_fedavg_hetero_family,
    bench_gradient_leakage_family,
    bench_pate_teacher_family,
    bench_secure_agg_family,
)
from quant_fund.research.benches_w153 import (
    bench_arch_predictor_family,
    bench_darts_nas_family,
    bench_enas_controller_family,
    bench_evolution_nas_family,
    bench_one_shot_nas_family,
    bench_random_search_nas_family,
)
from quant_fund.research.benches_w154 import (
    bench_attention_rollout_family,
    bench_clip_align_family,
    bench_convnet_baseline_family,
    bench_diffusion_ddim_family,
    bench_simclr_views_family,
    bench_vit_classifier_family,
)
from quant_fund.research.benches_w155 import (
    bench_causal_rep_family,
    bench_cevae_latent_family,
    bench_deep_iv_family,
    bench_dragonnet_dr_family,
    bench_policy_value_family,
    bench_tarnet_ite_family,
)
from quant_fund.research.benches_w156 import (
    bench_grownet_boost_family,
    bench_node_net_family,
    bench_soft_tree_family,
    bench_tabm_mini_family,
    bench_tabular_resnet_family,
    bench_tokenizer_bpe_family,
)
from quant_fund.research.benches_w157 import (
    bench_anom_transformer_family,
    bench_dagmm_family,
    bench_deep_svdd_family,
    bench_rrcf_family,
    bench_tranad_family,
    bench_usad_family,
)
from quant_fund.research.benches_w158 import (
    bench_approx_ndcg_ltr_family,
    bench_lambdarank_ltr_family,
    bench_listmle_ltr_family,
    bench_listnet_ltr_family,
    bench_neural_sort_ltr_family,
    bench_ranknet_ltr_family,
)
from quant_fund.research.benches_w159 import (
    bench_adafactor_opt_family,
    bench_lamb_opt_family,
    bench_lion_opt_family,
    bench_lookahead_opt_family,
    bench_muon_opt_family,
    bench_sophia_opt_family,
)
from quant_fund.research.benches_w160 import (
    bench_ditto_fl_family,
    bench_fednova_fl_family,
    bench_fedopt_adam_family,
    bench_mime_lite_family,
    bench_moon_fl_family,
    bench_scaffold_fl_family,
)
from quant_fund.research.benches_w161 import (
    bench_cno_lite_family,
    bench_deeponet_family,
    bench_fno_1d_family,
    bench_gno_lite_family,
    bench_lowrank_op_family,
    bench_pino_residual_family,
)
from quant_fund.research.benches_w162 import (
    bench_chronos_lite_family,
    bench_lagllama_lite_family,
    bench_moirai_lite_family,
    bench_moment_lite_family,
    bench_timer_lite_family,
    bench_timesfm_lite_family,
)
from quant_fund.research.benches_w163 import (
    bench_agem_cl_family,
    bench_der_cl_family,
    bench_hat_cl_family,
    bench_lwf_cl_family,
    bench_packnet_cl_family,
    bench_piggyback_cl_family,
)
from quant_fund.research.benches_w164 import (
    bench_bbb_vi_family,
    bench_concrete_dropout_family,
    bench_mc_dropout_family,
    bench_snapshot_ens_family,
    bench_swag_diag_family,
    bench_vcl_online_family,
)
from quant_fund.research.benches_w165 import (
    bench_crps_net_family,
    bench_diffusion_regressor_family,
    bench_flow_regression_family,
    bench_het_gp_family,
    bench_kernel_mixture_family,
    bench_mdn_cond_family,
)
from quant_fund.research.benches_w166 import (
    bench_anil_meta_family,
    bench_matching_net_family,
    bench_meta_sgd_family,
    bench_protonet_family,
    bench_r2d2_meta_family,
    bench_reptile_family,
)
from quant_fund.research.benches_w167 import (
    bench_algo_reasoning_family,
    bench_dgn_directional_family,
    bench_gps_transformer_family,
    bench_oversmooth_metric_family,
    bench_pna_agg_family,
    bench_virtual_node_family,
)
from quant_fund.research.benches_w168 import (
    bench_awac_family,
    bench_crossq_family,
    bench_dr3_reg_family,
    bench_ob2i_family,
    bench_redq_family,
    bench_td7_lite_family,
)
from quant_fund.research.benches_w169 import (
    bench_advi_bbvi_family,
    bench_iwae_bound_family,
    bench_nf_vi_family,
    bench_sparse_gp_sv_family,
    bench_structured_vi_family,
    bench_vrnn_seq_family,
)
from quant_fund.research.benches_w170 import (
    bench_cate_distill_family,
    bench_causal_rep_bal_family,
    bench_net_drlearner_family,
    bench_rlearner_family,
    bench_slearner_tlearner_family,
    bench_xlearner_family,
)
from quant_fund.research.benches_w171 import (
    bench_aps_cp_family,
    bench_cqr_pred_family,
    bench_full_cp_family,
    bench_ltt_cp_family,
    bench_risk_cp_family,
    bench_survival_cp_family,
)
from quant_fund.research.benches_w172 import (
    bench_corrupt_bandit_family,
    bench_cucb_family,
    bench_gittins_index_family,
    bench_neural_ucb_family,
    bench_psrl_family,
    bench_whittle_restless_family,
)
from quant_fund.research.benches_w173 import (
    bench_cold_diffusion_family,
    bench_ddim_ode_family,
    bench_diff_distill_family,
    bench_edm_karras_family,
    bench_rectified_flow_family,
    bench_stoch_interp_family,
)
from quant_fund.research.benches_w174 import (
    bench_agcrn_family,
    bench_astgcn_family,
    bench_dcrnn_lite_family,
    bench_gwnet_lite_family,
    bench_mtgnn_lite_family,
    bench_stgcn_lite_family,
)
from quant_fund.research.benches_w175 import (
    bench_alibi_attn_family,
    bench_moe_router_family,
    bench_mup_init_family,
    bench_rmsnorm_block_family,
    bench_rope_attn_family,
    bench_swiglu_ffn_family,
)
from quant_fund.research.benches_w176 import (
    bench_active_bald_family,
    bench_data_cartography_family,
    bench_el2n_scoring_family,
    bench_forgetting_events_family,
    bench_influence_func_family,
    bench_proto_prune_family,
)
from quant_fund.research.benches_w177 import (
    bench_izhikevich_family,
    bench_lif_neuron_family,
    bench_lsm_reservoir_family,
    bench_stdp_learn_family,
    bench_surrogate_snn_family,
    bench_temporal_code_family,
)
from quant_fund.research.benches_w178 import (
    bench_cagrad_mtl_family,
    bench_gradnorm_bal_family,
    bench_imtl_g_family,
    bench_mgda_mtl_family,
    bench_nash_mtl_family,
    bench_pcgrad_family,
)
from quant_fund.research.benches_w179 import (
    bench_cox_time_family,
    bench_deephit_family,
    bench_deepsurv_family,
    bench_drsa_surv_family,
    bench_nnet_surv_family,
    bench_pchazard_family,
)
from quant_fund.research.benches_w180 import (
    bench_boomerang_sampler_family,
    bench_bouncy_particle_family,
    bench_elliptical_slice_family,
    bench_kinetic_langevin_family,
    bench_riemannian_mala_family,
    bench_zigzag_sampler_family,
)
from quant_fund.research.benches_w181 import (
    bench_gated_deltanet_family,
    bench_longhorn_ssm_family,
    bench_mamba2_ssd_family,
    bench_rwkv7_family,
    bench_titans_memory_family,
    bench_xlstm_mlstm_family,
)
from quant_fund.research.benches_w182 import (
    bench_glow_flow_family,
    bench_iaf_flow_family,
    bench_maf_flow_family,
    bench_neural_spline_flow_family,
    bench_planar_flow_family,
    bench_real_nvp_family,
)
from quant_fund.research.benches_w183 import (
    bench_cam_prune_family,
    bench_dag_gnn_family,
    bench_dagma_lin_family,
    bench_golem_ev_family,
    bench_notears_family,
    bench_notears_mlp_family,
)
from quant_fund.research.benches_w184 import (
    bench_catapult_phase_family,
    bench_edge_stability_family,
    bench_hessian_eig_family,
    bench_mode_connectivity_family,
    bench_neural_grok_family,
    bench_ntk_kernel_family,
)
from quant_fund.research.benches_w185 import (
    bench_gumbel_relax_family,
    bench_implicit_diff_family,
    bench_ode_adjoint_family,
    bench_perturb_map_family,
    bench_smooth_argmax_family,
    bench_st_estimator_family,
)
from quant_fund.research.benches_w186 import (
    bench_adversarial_ebm_family,
    bench_contrastive_divergence_family,
    bench_denoising_sm_family,
    bench_noise_contrastive_family,
    bench_persistent_cd_family,
    bench_score_matching_family,
)
from quant_fund.research.benches_w187 import (
    bench_deepritz_pinn_family,
    bench_fbsde_solver_family,
    bench_feynman_kac_mc_family,
    bench_moc_lines_family,
    bench_spectral_pde_family,
    bench_weak_form_pinn_family,
)
from quant_fund.research.benches_w188 import (
    bench_badge_embed_family,
    bench_coreset_kcenter_family,
    bench_egl_change_family,
    bench_entropy_query_family,
    bench_margin_sampling_family,
    bench_qbc_committee_family,
)
from quant_fund.research.benches_w189 import (
    bench_alphazero_lite_family,
    bench_deep_cfr_family,
    bench_expert_iteration_family,
    bench_mccfr_outcome_family,
    bench_nfsp_family,
    bench_psro_family,
)
from quant_fund.research.benches_w190 import (
    bench_direct_lingam_family,
    bench_fci_alg_family,
    bench_ges_search_family,
    bench_ica_lingam_family,
    bench_mmmb_select_family,
    bench_var_lingam_family,
)
from quant_fund.research.benches_w191 import (
    bench_count_bonus_family,
    bench_go_explore_family,
    bench_icm_explore_family,
    bench_ngu_explore_family,
    bench_ride_explore_family,
    bench_rnd_explore_family,
)
from quant_fund.research.benches_w192 import (
    bench_copula_mi_family,
    bench_hsic_independence_family,
    bench_lsd_deptest_family,
    bench_mine_mi_family,
    bench_mmd_two_sample_family,
    bench_nwj_mi_family,
)
from quant_fund.research.benches_w193 import (
    bench_ddp_solve_family,
    bench_lqg_control_family,
    bench_lqr_control_family,
    bench_mpc_qp_family,
    bench_mppi_control_family,
    bench_pmp_bangbang_family,
)
from quant_fund.research.benches_w194 import (
    bench_cir_sim_family,
    bench_gp_bridge_family,
    bench_hawkes_thinning_family,
    bench_levy_jump_family,
    bench_ou_bridge_family,
    bench_poisson_thinning_family,
)
from quant_fund.research.benches_w195 import (
    bench_de_mcmc_family,
    bench_dram_family,
    bench_emcee_stretch_family,
    bench_indep_mh_family,
    bench_pcn_sampler_family,
    bench_rjmcmc_family,
)
from quant_fund.research.benches_w196 import (
    bench_bidiag_svd_family,
    bench_hessenberg_red_family,
    bench_inverse_iter_family,
    bench_jacobi_eig_family,
    bench_power_iter_family,
    bench_qr_eig_family,
)
from quant_fund.research.benches_w197 import (
    bench_base_stock_family,
    bench_clark_scarf_family,
    bench_eoq_model_family,
    bench_newsvendor_family,
    bench_ss_policy_family,
    bench_wagner_whitin_family,
)
from quant_fund.research.benches_w198 import (
    bench_johnson_flowshop_family,
    bench_knapsack_dp_family,
    bench_lpt_schedule_family,
    bench_neh_heuristic_family,
    bench_spt_weighted_family,
    bench_tsp_branchbound_family,
)
from quant_fund.research.benches_w199 import (
    bench_grover_search_family,
    bench_qaoa_maxcut_family,
    bench_qkernel_svm_family,
    bench_qpe_phase_family,
    bench_quantum_walk_family,
    bench_vqe_ising_family,
)
from quant_fund.research.benches_w200 import (
    bench_dmrg_tfim_family,
    bench_mps_fidelity_family,
    bench_tebd_quench_family,
    bench_tensor_cross_family,
    bench_tt_round_family,
    bench_tt_svd_family,
)
from quant_fund.research.benches_w201 import (
    bench_crr_tree_family,
    bench_dual_american_family,
    bench_exercise_boundary_family,
    bench_hjb_penalty_family,
    bench_kushner_mca_family,
    bench_psor_american_family,
)
from quant_fund.research.benches_w202 import (
    bench_mfg_flocking_family,
    bench_mfg_lq_family,
    bench_nash_cournot_family,
    bench_potential_game_family,
    bench_stackelberg_game_family,
    bench_stochastic_game_vi_family,
)
from quant_fund.research.benches_w203 import (
    bench_alpha_geodesic_family,
    bench_bregman_nmf_family,
    bench_fisher_rao_family,
    bench_jko_scheme_family,
    bench_mirror_descent_family,
    bench_natural_gradient_family,
)
from quant_fund.research.benches_w204 import (
    bench_bcmp_mva_family,
    bench_ctmc_availability_family,
    bench_gordon_newell_family,
    bench_jackson_network_family,
    bench_renewal_reward_family,
    bench_vacation_queue_family,
)
from quant_fund.research.benches_w205 import (
    bench_all_pay_auction_family,
    bench_ascending_clock_family,
    bench_double_auction_family,
    bench_first_price_auction_family,
    bench_gsp_auction_family,
    bench_vickrey_auction_family,
)
from quant_fund.research.benches_w206 import (
    bench_egreedy_decay_family,
    bench_mw_hedge_family,
    bench_pi_contraction_family,
    bench_qlearn_rate_family,
    bench_td_rate_family,
    bench_ucb_bound_family,
)
from quant_fund.research.benches_w207 import (
    bench_aes_sbox_family,
    bench_diffie_hellman_family,
    bench_ecc_secp256k1_family,
    bench_pedersen_commit_family,
    bench_sha256_impl_family,
    bench_shamir_secret_family,
)
from quant_fund.research.benches_w208 import (
    bench_cepstrum_pitch_family,
    bench_cwt_ridge_family,
    bench_goertzel_detect_family,
    bench_hilbert_instant_family,
    bench_lpc_formant_family,
    bench_mvdr_beamformer_family,
)
from quant_fund.research.benches_w209 import (
    bench_fault_tree_family,
    bench_fmea_rpn_family,
    bench_life_stress_family,
    bench_ram_markov_family,
    bench_redundancy_block_family,
    bench_weibull_life_family,
)
from quant_fund.research.benches_w210 import (
    bench_bch_code_family,
    bench_conv_interleaver_family,
    bench_crc_check_family,
    bench_ldpc_decoder_family,
    bench_polar_code_family,
    bench_turbo_decoder_family,
)
from quant_fund.research.benches_w211 import (
    bench_benders_decomp_family,
    bench_branch_and_cut_family,
    bench_column_generation_family,
    bench_gomory_cut_family,
    bench_held_karp_family,
    bench_lagrangian_relax_family,
)
from quant_fund.research.benches_w212 import (
    bench_christofides_tsp_family,
    bench_fptas_knapsack_family,
    bench_greedy_set_cover_family,
    bench_local_search_maxcut_family,
    bench_lp_rounding_sc_family,
    bench_primal_dual_vc_family,
)
from quant_fund.research.benches_w213 import (
    bench_eigenvalue_opt_family,
    bench_hoffman_bound_family,
    bench_qcqp_relax_family,
    bench_sdp_maxcut_family,
    bench_sos_certificate_family,
    bench_spectral_bisection_family,
)
from quant_fund.research.benches_w214 import (
    bench_marking_paging_family,
    bench_online_gradient_family,
    bench_ranking_matching_family,
    bench_secretary_prophet_family,
    bench_ski_rental_family,
    bench_work_function_kserver_family,
)
from quant_fund.research.benches_w215 import (
    bench_chance_scenario_family,
    bench_dro_wasserstein_family,
    bench_robust_budget_family,
    bench_saa_consistency_family,
    bench_scenario_tree_family,
    bench_two_stage_lshaped_family,
)
from quant_fund.research.benches_w216 import (
    bench_metadynamics_family,
    bench_parallel_tempering_family,
    bench_thermo_integration_family,
    bench_umbrella_sampling_family,
    bench_wang_landau_family,
    bench_wham_family,
)
from quant_fund.research.benches_w217 import (
    bench_cubature_kalman_family,
    bench_hinf_filter_family,
    bench_huber_filter_family,
    bench_mhe_family,
    bench_particle_smoother_family,
    bench_variational_bayes_family,
)
from quant_fund.research.benches_w218 import (
    bench_andreasen_huge_family,
    bench_barrier_adjoint_family,
    bench_deep_hedge_family,
    bench_dupire_localvol_family,
    bench_heston_calib_family,
    bench_sabr_calib_family,
)
from quant_fund.research.benches_w219 import (
    bench_bdd_ops_family,
    bench_cdcl_solver_family,
    bench_ltl_mc_family,
    bench_twosat_scc_family,
    bench_unit_propagation_family,
    bench_walksat_family,
)
from quant_fund.research.benches_w220 import (
    bench_bmc_unroll_family,
    bench_cegar_loop_family,
    bench_hoare_logic_family,
    bench_ic3_pdr_family,
    bench_invariant_synth_family,
    bench_k_induction_family,
    bench_ranking_function_family,
)
from quant_fund.research.benches_w221 import (
    bench_buchberger_family,
    bench_gf2_factor_family,
    bench_lll_reduce_family,
    bench_newton_interp_family,
    bench_poly_gcd_family,
    bench_resultant_family,
)
from quant_fund.research.benches_w222 import (
    bench_continued_fraction_family,
    bench_crt_garner_family,
    bench_ec_scalar_family,
    bench_miller_rabin_family,
    bench_pollard_rho_family,
    bench_tonelli_shanks_family,
)
from quant_fund.research.benches_w223 import (
    bench_consistent_hash_family,
    bench_gossip_epidemic_family,
    bench_paxos_family,
    bench_pbft_lite_family,
    bench_raft_election_family,
    bench_vector_clock_family,
)
from quant_fund.research.benches_w224 import (
    bench_aho_corasick_family,
    bench_bwt_transform_family,
    bench_edit_distance_family,
    bench_kmp_search_family,
    bench_lz77_family,
    bench_suffix_automaton_family,
)
from quant_fund.research.benches_w225 import (
    bench_barnes_hut_family,
    bench_fem_truss_family,
    bench_nbody_leapfrog_family,
    bench_rigid_collision_family,
    bench_sph_fluid_family,
    bench_verlet_cloth_family,
)
from quant_fund.research.benches_w226 import (
    bench_cyk_parser_family,
    bench_dfa_minimize_family,
    bench_dominance_tree_family,
    bench_linscan_regalloc_family,
    bench_liveness_dce_family,
    bench_regex_engine_family,
)
from quant_fund.research.benches_w227 import (
    bench_arithmetic_coding_family,
    bench_golomb_rice_family,
    bench_huffman_codes_family,
    bench_lz78_dict_family,
    bench_lzw_compress_family,
    bench_rans_coder_family,
)
from quant_fund.research.benches_w228 import (
    bench_gcounter_family,
    bench_lww_map_family,
    bench_orset_family,
    bench_pncounter_family,
    bench_rga_sequence_family,
    bench_twopset_family,
)
from quant_fund.research.benches_w229 import (
    bench_bloom_filter_family,
    bench_cuckoo_filter_family,
    bench_minhash_lsh_family,
    bench_quotient_filter_family,
    bench_simhash_family,
    bench_xor_filter_family,
)
from quant_fund.research.benches_w230 import (
    bench_closest_pair_family,
    bench_ear_clipping_family,
    bench_point_in_polygon_family,
    bench_rotating_calipers_family,
    bench_segment_intersection_family,
    bench_sutherland_hodgman_family,
)
from quant_fund.research.benches_w231 import (
    bench_branch_predictor_family,
    bench_cache_sim_family,
    bench_cpu_pipeline_family,
    bench_paging_sim_family,
    bench_roofline_model_family,
    bench_tomasulo_sim_family,
)
from quant_fund.research.benches_w232 import (
    bench_block_validator_family,
    bench_difficulty_retarget_family,
    bench_fork_resolution_family,
    bench_merkle_tree_family,
    bench_proof_of_work_family,
    bench_utxo_set_family,
)
from quant_fund.research.benches_w233 import (
    bench_gvn_elim_family,
    bench_instr_sched_family,
    bench_licm_hoist_family,
    bench_reg_coalesce_family,
    bench_sccp_const_family,
    bench_ssa_construct_family,
)
from quant_fund.research.benches_w234 import (
    bench_btree_index_family,
    bench_join_algos_family,
    bench_lsm_tree_family,
    bench_mvcc_isolation_family,
    bench_query_planner_family,
    bench_wal_recovery_family,
)
from quant_fund.research.benches_w235 import (
    bench_cfs_scheduler_family,
    bench_deadlock_detect_family,
    bench_demand_paging_family,
    bench_disk_sched_family,
    bench_fs_journal_family,
    bench_round_robin_sched_family,
)
from quant_fund.research.benches_w236 import (
    bench_bresenham_line_family,
    bench_bsp_tree_family,
    bench_mvp_transform_family,
    bench_quaternion_slerp_family,
    bench_raycaster_family,
    bench_scanline_fill_family,
    bench_zbuffer_render_family,
)
from quant_fund.research.benches_w237 import (
    bench_earley_parser_family,
    bench_ll1_table_family,
    bench_peg_packrat_family,
    bench_pratt_parser_family,
    bench_recursive_descent_family,
    bench_slr_parser_family,
)
from quant_fund.research.benches_w238 import (
    bench_http2_flow_family,
    bench_nat_table_family,
    bench_rtt_estimator_family,
    bench_sliding_window_family,
    bench_tcp_aimd_family,
    bench_token_bucket_family,
)
from quant_fund.research.benches_w239 import (
    bench_cps_transform_family,
    bench_gc_marksweep_family,
    bench_hm_inference_family,
    bench_macro_expand_family,
    bench_simple_types_family,
    bench_tree_walk_interp_family,
)
from quant_fund.research.benches_w240 import (
    bench_blind_sig_family,
    bench_commit_reveal_family,
    bench_merkle_ots_family,
    bench_rsa_toy_family,
    bench_winternitz_ots_family,
    bench_zkp_schnorr_family,
)
from quant_fund.research.benches_w241 import (
    bench_aries_recovery_family,
    bench_blink_tree_family,
    bench_buffer_pool_family,
    bench_mvcc_gc_family,
    bench_selinger_join_family,
    bench_two_phase_lock_family,
)
from quant_fund.research.benches_w242 import (
    bench_epaxos_family,
    bench_multi_paxos_family,
    bench_swim_gossip_family,
    bench_two_three_pc_family,
    bench_viewstamped_family,
    bench_zab_protocol_family,
)
from quant_fund.research.benches_w243 import (
    bench_bignum_family,
    bench_fft_radix2_family,
    bench_int_sqrt_family,
    bench_karatsuba_family,
    bench_ntt_family,
    bench_strassen_family,
)
from quant_fund.research.benches_w244 import (
    bench_inverted_index_family,
    bench_lsh_dedup_family,
    bench_ngram_spell_family,
    bench_positional_index_family,
    bench_posting_merge_family,
    bench_wand_bmw_family,
)
from quant_fund.research.benches_w245 import (
    bench_elf_loader_family,
    bench_malloc_freelist_family,
    bench_mlfq_sched_family,
    bench_mmap_pager_family,
    bench_semaphore_monitor_family,
    bench_syscall_layer_family,
)
from quant_fund.research.benches_w246 import (
    bench_bytecode_vm_family,
    bench_closure_conv_family,
    bench_inline_cache_family,
    bench_nan_tagging_family,
    bench_tail_call_tramp_family,
    bench_threaded_interp_family,
)
from quant_fund.research.benches_w247 import (
    bench_atomics_tas_family,
    bench_bakery_lock_family,
    bench_channel_select_family,
    bench_peterson_lock_family,
    bench_rw_lock_family,
    bench_work_stealing_family,
)
from quant_fund.research.benches_w248 import (
    bench_brzozowski_deriv_family,
    bench_cellular_automata_family,
    bench_dfa_equiv_family,
    bench_mealy_moore_family,
    bench_pda_sim_family,
    bench_turing_machine_family,
)
from quant_fund.research.benches_w249 import (
    bench_debruijn_assemble_family,
    bench_fm_index_family,
    bench_motif_scan_family,
    bench_needleman_wunsch_family,
    bench_smith_waterman_family,
    bench_upgma_tree_family,
)
from quant_fund.research.benches_w250 import (
    bench_astar_search_family,
    bench_bidirectional_dijkstra_family,
    bench_bron_kerbosch_family,
    bench_critical_path_family,
    bench_dinic_flow_family,
    bench_mincost_flow_family,
)
from quant_fund.research.benches_w251 import (
    bench_givens_qr_family,
    bench_jacobi_svd_family,
    bench_ldlt_solve_family,
    bench_lu_pivots_family,
    bench_orth_iter_family,
    bench_sturm_eig_family,
)
from quant_fund.research.benches_w252 import (
    bench_anf_cps_family,
    bench_compacting_gc_family,
    bench_dispatch_table_family,
    bench_gen_gc_family,
    bench_poly_inline_cache_family,
    bench_trampoline_tc_family,
)
from quant_fund.research.benches_w253 import (
    bench_buchi_automata_family,
    bench_cfg_pda_equiv_family,
    bench_register_automata_family,
    bench_tree_automata_family,
    bench_two_way_dfa_family,
    bench_weighted_fst_family,
)
from quant_fund.research.benches_w254 import (
    bench_aead_etm_family,
    bench_cbc_padding_family,
    bench_hmac_construct_family,
    bench_merkle_damgard_family,
    bench_pbkdf2_kdf_family,
    bench_tls_handshake_family,
)
from quant_fund.research.benches_w255 import (
    bench_admm_lasso_family,
    bench_barrier_ip_family,
    bench_coord_descent_family,
    bench_ellipsoid_method_family,
    bench_proj_gradient_family,
    bench_simplex_lp_family,
)
from quant_fund.research.benches_w256 import (
    bench_epoch_reclaim_family,
    bench_flat_combining_family,
    bench_hazard_pointer_family,
    bench_ms_queue_family,
    bench_rcu_lock_family,
    bench_seqlock_family,
)
from quant_fund.research.benches_w257 import (
    bench_bfv_fhe_family,
    bench_chaum_pedersen_family,
    bench_lwe_kex_family,
    bench_ntru_toy_family,
    bench_sigma_or_proof_family,
    bench_sis_hash_family,
)
from quant_fund.research.benches_w258 import (
    bench_adaptive_qp_family,
    bench_bitmap_index_family,
    bench_cascades_opt_family,
    bench_func_dep_family,
    bench_vectorized_exec_family,
    bench_zone_map_family,
)
from quant_fund.research.benches_w259 import (
    bench_arp_table_family,
    bench_bgp_pathvec_family,
    bench_dhcp_lease_family,
    bench_dns_resolver_family,
    bench_eth_switch_family,
    bench_nat_traversal_family,
)
from quant_fund.research.benches_w260 import (
    bench_gale_chu_family,
    bench_gale_shapley_family,
    bench_hopcroft_karp_family,
    bench_konig_cover_family,
    bench_kuhn_munkres_family,
    bench_topo_layers_family,
)
from quant_fund.research.benches_w261 import (
    bench_ekf_slam_family,
    bench_frontier_explore_family,
    bench_occupancy_grid_family,
    bench_particle_slam_family,
    bench_pure_pursuit_family,
    bench_stanley_family,
)
from quant_fund.research.benches_w262 import (
    bench_mesi_cache_family,
    bench_numa_alloc_family,
    bench_ring_allreduce_family,
    bench_simd_lanes_family,
    bench_stencil_halo_family,
    bench_task_dag_family,
)
from quant_fund.research.benches_w263 import (
    bench_debounce_fsm_family,
    bench_edf_scheduler_family,
    bench_ring_buffer_family,
    bench_rms_scheduler_family,
    bench_watchdog_task_family,
    bench_wcet_est_family,
)
from quant_fund.research.benches_w264 import (
    bench_block_lanczos_family,
    bench_divide_conquer_eig_family,
    bench_dqds_family,
    bench_fgmres_family,
    bench_randomized_qb_family,
    bench_sparse_cholesky_family,
)
from quant_fund.research.benches_w265 import (
    bench_asan_shadow_family,
    bench_contract_check_family,
    bench_fuzzer_mutate_family,
    bench_grammar_fuzz_family,
    bench_symbolic_exec_family,
    bench_taint_track_family,
)
from quant_fund.research.benches_w266 import (
    bench_bump_map_family,
    bench_mipmap_sample_family,
    bench_phong_shade_family,
    bench_shadow_map_family,
    bench_ssao_lite_family,
    bench_triangle_raster_family,
)
from quant_fund.research.benches_w267 import (
    bench_bank_conflict_family,
    bench_mem_coalesce_family,
    bench_occupancy_calc_family,
    bench_shared_mem_tile_family,
    bench_simt_divergence_family,
    bench_warp_scheduler_family,
)
from quant_fund.research.benches_w268 import (
    bench_chacha_stream_family,
    bench_elgamal_enc_family,
    bench_fiat_shamir_family,
    bench_ot_12_family,
    bench_paillier_he_family,
    bench_poly1305_mac_family,
)
from quant_fund.research.benches_w269 import (
    bench_epipolar_8pt_family,
    bench_homography_4pt_family,
    bench_lk_flow_family,
    bench_orb_feature_family,
    bench_ransac_plane_family,
    bench_stereo_disparity_family,
)
from quant_fund.research.benches_w270 import (
    bench_dmc_solver_family,
    bench_fdtd_wave_family,
    bench_ising_metro_family,
    bench_lattice_boltzmann_family,
    bench_lj_md_family,
    bench_pic_plasma_family,
)
from quant_fund.research.benches_w271 import (
    bench_csma_ca_family,
    bench_diffserv_qos_family,
    bench_icmp_path_family,
    bench_ospf_lsa_family,
    bench_stp_spanning_family,
    bench_vlan_tag_family,
)
from quant_fund.research.benches_w272 import (
    bench_backstepping_family,
    bench_gain_schedule_family,
    bench_pid_antiwindup_family,
    bench_repetitive_ctrl_family,
    bench_sliding_mode_family,
    bench_smith_predictor_family,
)
from quant_fund.research.benches_w273 import (
    bench_const_fold_family,
    bench_inline_expand_family,
    bench_loop_unroll_family,
    bench_partial_eval_family,
    bench_peephole_opt_family,
    bench_strength_red_family,
)
from quant_fund.research.benches_w274 import (
    bench_gc_skew_family,
    bench_hmm_profile_family,
    bench_kmer_count_family,
    bench_orf_find_family,
    bench_seq_logo_family,
    bench_star_msa_family,
)
from quant_fund.research.benches_w275 import (
    bench_columnar_scan_family,
    bench_graceful_hash_family,
    bench_index_intersect_family,
    bench_late_materialize_family,
    bench_radix_join_family,
    bench_simd_filter_family,
)
from quant_fund.research.benches_w276 import (
    bench_bully_elect_family,
    bench_causal_bcast_family,
    bench_chord_look_family,
    bench_quorum_rw_family,
    bench_ra_mutex_family,
    bench_token_ring_family,
)
from quant_fund.research.benches_w277 import (
    bench_chirp_z_family,
    bench_decimate_int_family,
    bench_fir_window_family,
    bench_prony_model_family,
    bench_stft_istft_family,
    bench_wola_synth_family,
)
from quant_fund.research.benches_w278 import (
    bench_cobweb_model_family,
    bench_nk_phillips_family,
    bench_olg_model_family,
    bench_rbc_sim_family,
    bench_solow_model_family,
    bench_taylor_rule_family,
)
from quant_fund.research.benches_w279 import (
    bench_dist_obsv_family,
    bench_flat_track_family,
    bench_l2_gain_family,
    bench_luen_obsv_family,
    bench_lyap_synth_family,
    bench_mrac_adapt_family,
)
from quant_fund.research.benches_w280 import (
    bench_boundary_sq_family,
    bench_euler_char_family,
    bench_graph_h1_family,
    bench_rips_h1_family,
    bench_simp_betti_family,
    bench_winding_deg_family,
)
from quant_fund.research.benches_w281 import (
    bench_galois_field_family,
    bench_group_table_family,
    bench_ideal_member_family,
    bench_matrix_grp_family,
    bench_perm_group_family,
    bench_poly_ring_family,
)
from quant_fund.research.benches_w282 import (
    bench_gray_code_family,
    bench_inversion_count_family,
    bench_latin_square_family,
    bench_ramsey_bound_family,
    bench_stirling_count_family,
    bench_subset_sum_dp_family,
)
from quant_fund.research.benches_w283 import (
    bench_bezier_curve_family,
    bench_fk_dh_family,
    bench_ik_jac_family,
    bench_odom_comp_family,
    bench_pot_field_family,
    bench_ray_lidar_family,
)
from quant_fund.research.benches_w284 import (
    bench_band_align_family,
    bench_codon_usage_family,
    bench_fitch_pars_family,
    bench_jc69_lik_family,
    bench_nj_tree_family,
    bench_seed_extend_family,
)
from quant_fund.research.benches_w285 import (
    bench_blahut_arimoto_family,
    bench_elias_gamma_family,
    bench_kl_knn_family,
    bench_markov_entropy_family,
    bench_miller_madow_family,
    bench_type_class_family,
)
from quant_fund.research.benches_w286 import (
    bench_beacon_detect_family,
    bench_cred_stuffing_family,
    bench_entropy_dns_family,
    bench_exfil_zscore_family,
    bench_impossible_travel_family,
    bench_sig_score_family,
)
from quant_fund.research.benches_w287 import (
    bench_christoffel_family,
    bench_first_ff_family,
    bench_frenet_frame_family,
    bench_gauss_curve_family,
    bench_geodesic_sphere_family,
    bench_surf_area_family,
)
from quant_fund.research.benches_w288 import (
    bench_conv_prob_family,
    bench_fubini_swap_family,
    bench_leb_integral_family,
    bench_leb_measure_family,
    bench_radon_nikodym_family,
    bench_weak_conv_family,
)
from quant_fund.research.benches_w289 import (
    bench_adjunction_family,
    bench_fin_cat_family,
    bench_functor_check_family,
    bench_limit_prod_family,
    bench_monad_laws_family,
    bench_nat_trans_family,
)
from quant_fund.research.benches_w290 import (
    bench_mol_descriptors_family,
    bench_morgan_fp_family,
    bench_ring_detect_family,
    bench_smiles_parse_family,
    bench_substruct_family,
    bench_tanimoto_family,
)
from quant_fund.research.benches_w291 import (
    bench_a_star_route_family,
    bench_drc_check_family,
    bench_levelize_family,
    bench_netlist_parse_family,
    bench_place_quadratic_family,
    bench_sta_timing_family,
)
from quant_fund.research.benches_w292 import (
    bench_doh_wire_family,
    bench_qpack_pack_family,
    bench_quic_streams_family,
    bench_sctp_tsn_family,
    bench_tls13_trans_family,
    bench_wg_ik_family,
)
from quant_fund.research.benches_w293 import (
    bench_deferred_shade_family,
    bench_env_map_family,
    bench_frustum_cull_family,
    bench_lod_select_family,
    bench_sdf_raymarch_family,
    bench_shadow_pcf_family,
)
from quant_fund.research.benches_w294 import (
    bench_bb_reorder_family,
    bench_cfg_simplify_family,
    bench_jump_thread_family,
    bench_modulo_sched_family,
    bench_tail_dup_family,
    bench_tree_cover_family,
)
from quant_fund.research.benches_w295 import (
    bench_gauss_iod_family,
    bench_kepler_solve_family,
    bench_lambert_problem_family,
    bench_orbit_maneuver_family,
    bench_orbital_elements_family,
    bench_tle_propagate_family,
)
from quant_fund.research.benches_w296 import (
    bench_chomp_family,
    bench_gjk_epa_family,
    bench_ilqr_family,
    bench_lqr_funnel_family,
    bench_rts_smoother_family,
    bench_se3_spline_family,
)
from quant_fund.research.benches_w297 import (
    bench_dilithium_sig_family,
    bench_frodokem_family,
    bench_kyber_kem_family,
    bench_ntt_ring_family,
    bench_sphincs_sig_family,
    bench_xmss_sig_family,
)
from quant_fund.research.benches_w298 import (
    bench_art_sirt_family,
    bench_chan_vese_family,
    bench_cs_mri_family,
    bench_hu_moments_family,
    bench_mi_register_family,
    bench_radon_fbp_family,
)
from quant_fund.research.benches_w299 import (
    bench_expectimax_family,
    bench_isomcts_family,
    bench_mast_playout_family,
    bench_rave_mc_family,
    bench_retrograde_wdl_family,
    bench_tablebase_dtm_family,
)
from quant_fund.research.benches_w300 import (
    bench_delta_t_family,
    bench_eclipse_circ_family,
    bench_equinox_prec_family,
    bench_nutation_lite_family,
    bench_planet_vsop_family,
    bench_rise_set_family,
)
from quant_fund.research.benches_w301 import (
    bench_avo_shuey_family,
    bench_eikonal_fmm_family,
    bench_kirchhoff_mig_family,
    bench_nmo_dix_family,
    bench_taup_transform_family,
    bench_vibroseis_sweep_family,
)
from quant_fund.research.benches_w302 import (
    bench_card_table_gc_family,
    bench_escape_analysis_family,
    bench_gvn_pre_family,
    bench_osr_deopt_family,
    bench_ssa_repair_family,
    bench_trace_tree_family,
)
from quant_fund.research.benches_w303 import (
    bench_adpcm_ima_family,
    bench_celp_encode_family,
    bench_lpc_analysis_family,
    bench_mel_cepstrum_family,
    bench_mulaw_compand_family,
    bench_viterbi_vad_family,
)
from quant_fund.research.benches_w304 import (
    bench_grabcut_lite_family,
    bench_harris_corner_family,
    bench_hough_lines_family,
    bench_integral_image_family,
    bench_meanshift_track_family,
    bench_seam_carving_family,
)
from quant_fund.research.benches_w305 import (
    bench_booth_rotation_family,
    bench_lyndon_factor_family,
    bench_palindromic_tree_family,
    bench_suffix_array_lcp_family,
    bench_suffix_tree_lex_family,
    bench_z_function_family,
)
from quant_fund.research.benches_w306 import (
    bench_batch_od_family,
    bench_cowell_j2_family,
    bench_cr3bp_dynamics_family,
    bench_davenport_q_family,
    bench_laplace_iod_family,
    bench_porkchop_grid_family,
)
from quant_fund.research.benches_w307 import (
    bench_bitap_fuzzy_family,
    bench_glushkov_nfa_family,
    bench_lazy_dfa_family,
    bench_literal_prefilter_family,
    bench_pike_vm_family,
    bench_regex_simplify_family,
)
from quant_fund.research.benches_w308 import (
    bench_gardner_relation_family,
    bench_gassmann_sub_family,
    bench_reflectivity_synth_family,
    bench_semblance_scan_family,
    bench_spectral_decomp_family,
    bench_vz_raytrace_family,
)
from quant_fund.research.benches_w309 import (
    bench_bike_lite_family,
    bench_hqc_lite_family,
    bench_mceliece_lite_family,
    bench_rainbow_sig_family,
    bench_sidh_lite_family,
    bench_uov_sig_family,
)
from quant_fund.research.benches_w310 import (
    bench_gottesman_knill_family,
    bench_repetition_qec_family,
    bench_shor_code_family,
    bench_steane_code_family,
    bench_surface_code_family,
    bench_syndrome_circuit_family,
)
from quant_fund.research.benches_w311 import (
    bench_aig_rewrite_family,
    bench_clock_tree_family,
    bench_floorplan_sa_family,
    bench_fm_partition_family,
    bench_lee_router_family,
    bench_power_est_family,
)
from quant_fund.research.benches_w312 import (
    bench_abd_register_family,
    bench_bracha_bcast_family,
    bench_delta_crdt_family,
    bench_hlc_clock_family,
    bench_quorum_weighted_family,
    bench_raft_log_family,
    bench_tot_order_family,
)
from quant_fund.research.benches_w313 import (
    bench_amg_lite_family,
    bench_bicgstab_family,
    bench_chebyshev_iter_family,
    bench_ilu_precond_family,
    bench_minres_family,
    bench_v_cycle_family,
)
from quant_fund.research.benches_w314 import (
    bench_catmull_clark_family,
    bench_half_edge_family,
    bench_laplacian_smooth_family,
    bench_loop_subdiv_family,
    bench_marching_cubes_family,
    bench_nurbs_eval_family,
)
from quant_fund.research.benches_w315 import (
    bench_dmp_control_family,
    bench_ds_motion_family,
    bench_grasp_epsilon_family,
    bench_rmpflow_family,
    bench_rrt_connect_family,
    bench_wbc_qp_family,
)
from quant_fund.research.benches_w316 import (
    bench_adapt_vqe_family,
    bench_hhl_lite_family,
    bench_qdrift_family,
    bench_shadow_tomography_family,
    bench_trotter_suzuki_family,
    bench_vqd_states_family,
)
from quant_fund.research.benches_w317 import (
    bench_canny_edge_family,
    bench_distance_transform_family,
    bench_nlm_denoise_family,
    bench_otsu_threshold_family,
    bench_slic_superpixels_family,
    bench_watershed_seg_family,
)
from quant_fund.research.benches_w318 import (
    bench_bidirectional_tc_family,
    bench_dep_types_family,
    bench_nbe_eval_family,
    bench_proof_kernel_family,
    bench_tactic_engine_family,
    bench_unify_meta_family,
)
from quant_fund.research.benches_w319 import (
    bench_congruence_closure_family,
    bench_nelson_oppen_family,
    bench_omega_lia_family,
    bench_ring_normalize_family,
    bench_term_rewrite_family,
    bench_tseitin_cnf_family,
)
from quant_fund.research.benches_w320 import (
    bench_affine_karr_family,
    bench_andersen_pta_family,
    bench_chaotic_widen_family,
    bench_interval_analysis_family,
    bench_sign_domain_family,
    bench_zone_dbm_family,
)
from quant_fund.research.benches_w321 import (
    bench_context_pta_family,
    bench_interproc_summary_family,
    bench_recency_abstraction_family,
    bench_separation_logic_family,
    bench_shape_graph_family,
    bench_three_valued_logic_family,
)
from quant_fund.research.benches_w322 import (
    bench_cegis_loop_family,
    bench_horn_clauses_family,
    bench_interpolant_mc_family,
    bench_predicate_abs_family,
    bench_sygus_synth_family,
    bench_weakest_precond_family,
)
from quant_fund.research.benches_w323 import (
    bench_alg_effects_family,
    bench_free_monad_family,
    bench_gradual_types_family,
    bench_row_types_family,
    bench_session_types_family,
    bench_shift_reset_family,
)
from quant_fund.research.benches_w324 import (
    bench_banerjee_dep_family,
    bench_fourier_motzkin_family,
    bench_omega_test_family,
    bench_pluto_schedule_family,
    bench_tiling_legality_family,
    bench_vec_legality_family,
)
from quant_fund.research.benches_w325 import (
    bench_borrow_check_family,
    bench_capability_perm_family,
    bench_escape_region_family,
    bench_lifetime_outlives_family,
    bench_linear_use_family,
    bench_refinement_liquid_family,
)
from quant_fund.research.benches_w326 import (
    bench_bisim_refine_family,
    bench_ctl_mc_family,
    bench_nba_emptiness_family,
    bench_parity_game_family,
    bench_timed_automata_family,
    bench_wsts_cover_family,
)
from quant_fund.research.benches_w327 import (
    bench_bulletproof_ip_family,
    bench_kzg_commit_family,
    bench_plonkish_gate_family,
    bench_qap_encode_family,
    bench_r1cs_check_family,
    bench_snark_circuit_family,
)
from quant_fund.research.benches_w328 import (
    bench_funext_toy_family,
    bench_hit_quotient_family,
    bench_hlevel_check_family,
    bench_kan_hcomp_family,
    bench_path_types_family,
    bench_univalence_toy_family,
)
from quant_fund.research.benches_w329 import (
    bench_cut_elim_family,
    bench_intuit_class_family,
    bench_linear_logic_family,
    bench_nd_check_family,
    bench_resolution_fol_family,
    bench_sequent_prove_family,
)
from quant_fund.research.benches_w330 import (
    bench_array_theory_family,
    bench_bv_ops_family,
    bench_diff_logic_family,
    bench_dpllt_family,
    bench_lia_branch_family,
    bench_mcsat_lite_family,
)
from quant_fund.research.benches_w331 import (
    bench_circuit_lb_family,
    bench_fpras_dnf_family,
    bench_np_reduce_family,
    bench_param_fpt_family,
    bench_pcp_verify_family,
    bench_sumcheck_family,
)
from quant_fund.research.benches_w332 import (
    bench_bell_ineq_family,
    bench_density_matrix_family,
    bench_entanglement_family,
    bench_povm_measure_family,
    bench_qchannel_family,
    bench_state_tomo_family,
)
from quant_fund.research.benches_w333 import (
    bench_hensel_lift_family,
    bench_poly_crt_family,
    bench_poly_eval_interp_family,
    bench_poly_factor_fp_family,
    bench_sparse_interp_family,
    bench_subresultant_family,
)
from quant_fund.research.benches_w334 import (
    bench_beaver_triple_family,
    bench_bgw_mpc_family,
    bench_garbled_circuit_family,
    bench_ot_extension_family,
    bench_psi_intersect_family,
    bench_spdz_mac_family,
)
from quant_fund.research.benches_w335 import (
    bench_adjoint_check_family,
    bench_cat_colimit_family,
    bench_exponential_obj_family,
    bench_fin_limit_family,
    bench_subobject_classifier_family,
    bench_yoneda_embed_family,
)
from quant_fund.research.benches_w336 import (
    bench_busy_beaver_family,
    bench_compactness_lite_family,
    bench_pr_functions_family,
    bench_ramsey_theory_family,
    bench_turing_degrees_family,
    bench_ultraproduct_family,
)
from quant_fund.research.benches_w337 import (
    bench_church_encoding_family,
    bench_de_bruijn_family,
    bench_knuth_bendix_family,
    bench_lambda_typing_family,
    bench_ski_combinator_family,
    bench_unification_family,
)
from quant_fund.research.benches_w338 import (
    bench_ac_choice_family,
    bench_cardinal_arith_family,
    bench_ordinal_arith_family,
    bench_transfinite_induct_family,
    bench_v_omega_family,
    bench_well_founded_family,
)
from quant_fund.research.benches_w339 import (
    bench_field_ext_family,
    bench_galois_group_family,
    bench_lie_bracket_family,
    bench_rep_theory_family,
    bench_root_system_family,
    bench_splitting_field_family,
)
from quant_fund.research.benches_w340 import (
    bench_chain_complex_family,
    bench_hilbert_series_family,
    bench_sheaf_check_family,
    bench_snake_lemma_family,
    bench_tor_ext_family,
    bench_variety_morph_family,
)
from quant_fund.research.benches_w341 import (
    bench_bisimulation_family,
    bench_covering_space_family,
    bench_ef_game_family,
    bench_fundamental_group_family,
    bench_kripke_semantics_family,
    bench_topo_separation_family,
)
from quant_fund.research.benches_w342 import (
    bench_analytic_sets_family,
    bench_arith_hierarchy_family,
    bench_borel_hierarchy_family,
    bench_forcing_lite_family,
    bench_jump_operator_family,
    bench_rice_theorem_family,
)
from quant_fund.research.benches_w343 import (
    bench_boolean_algebra_family,
    bench_congruence_lattice_family,
    bench_galois_connection_family,
    bench_lattice_check_family,
    bench_tarski_fixed_family,
    bench_term_algebra_family,
)
from quant_fund.research.benches_w344 import (
    bench_burnside_lemma_family,
    bench_cayley_graph_family,
    bench_conjugacy_classes_family,
    bench_free_group_family,
    bench_group_presentation_family,
    bench_sylow_theorems_family,
)
from quant_fund.research.benches_w345 import (
    bench_minimal_poly_family,
    bench_norm_trace_family,
    bench_pid_check_family,
    bench_quotient_ring_family,
    bench_ring_ideals_family,
    bench_spec_ring_family,
)
from quant_fund.research.benches_w346 import (
    bench_cohomology_cup_family,
    bench_elliptic_curve_family,
    bench_koszul_complex_family,
    bench_mayer_vietoris_family,
    bench_p_adic_val_family,
    bench_quadratic_recip_family,
)
from quant_fund.research.benches_w347 import (
    bench_compact_space_family,
    bench_connected_space_family,
    bench_convergence_space_family,
    bench_product_topology_family,
    bench_quotient_topology_family,
    bench_tietze_urysohn_family,
)
from quant_fund.research.benches_w348 import (
    bench_euler_trail_family,
    bench_graph_coloring_family,
    bench_matroid_greedy_family,
    bench_planar_check_family,
    bench_poset_dimension_family,
    bench_ramsey_r33_family,
)
from quant_fund.research.benches_w349 import (
    bench_bezout_bezout_family,
    bench_hilbert_poly_family,
    bench_monomial_ideal_family,
    bench_projective_plane_family,
    bench_variety_dim_family,
    bench_zariski_topo_family,
)
from quant_fund.research.benches_w350 import (
    bench_character_table_s3_family,
    bench_fourier_sn_family,
    bench_induced_rep_family,
    bench_perm_rep_family,
    bench_regular_rep_family,
    bench_schur_ortho_family,
)
from quant_fund.research.benches_w351 import (
    bench_banach_fixed_family,
    bench_compact_operator_family,
    bench_fourier_finite_family,
    bench_gram_schmidt_family,
    bench_lp_duality_family,
    bench_spectral_theorem_family,
)
from quant_fund.research.benches_w352 import (
    bench_cartan_matrix_family,
    bench_killing_form_family,
    bench_root_lattice_a2_family,
    bench_sl2_structure_family,
    bench_su2_algebra_family,
    bench_weyl_group_a2_family,
)
from quant_fund.research.benches_w353 import (
    bench_gronwall_lemma_family,
    bench_lyapunov_stability_family,
    bench_phase_plane_family,
    bench_picard_lindelof_family,
    bench_sturm_liouville_family,
    bench_variation_params_family,
)
from quant_fund.research.benches_w354 import (
    bench_dedekind_check_family,
    bench_divisor_group_family,
    bench_genus_riemann_family,
    bench_local_ring_zn_family,
    bench_moduli_naive_family,
    bench_sheaf_gluing_family,
)
from quant_fund.research.benches_w355 import (
    bench_conditional_expect_family,
    bench_conv_sum_family,
    bench_kolmogorov_axioms_family,
    bench_markov_ineq_family,
    bench_moment_generating_family,
    bench_stochastic_order_family,
)
from quant_fund.research.benches_w356 import (
    bench_gambler_ruin_family,
    bench_markov_chain_family,
    bench_markov_hitting_family,
    bench_martingale_check_family,
    bench_poisson_process_family,
    bench_stopping_time_family,
)
from quant_fund.research.benches_w357 import (
    bench_connection_form_family,
    bench_gauss_bonnet_family,
    bench_geodesic_eq_family,
    bench_holonomy_family,
    bench_parallel_transport_family,
    bench_sectional_curv_family,
)
from quant_fund.research.benches_w358 import (
    bench_banach_alaoglu_family,
    bench_closed_graph_family,
    bench_open_mapping_family,
    bench_reflexive_space_family,
    bench_uniform_bounded_family,
    bench_weak_convergence_family,
)
from quant_fund.research.benches_w359 import (
    bench_fejer_kernel_family,
    bench_fourier_multiplier_family,
    bench_plancherel_family,
    bench_poisson_summation_family,
    bench_sobolev_embed_family,
    bench_uncertainty_family,
)
from quant_fund.research.benches_w360 import (
    bench_energy_method_family,
    bench_fundamental_laplace_family,
    bench_heat_kernel_family,
    bench_maximum_principle_family,
    bench_wave_dalembert_family,
    bench_weak_solution_family,
)
from quant_fund.research.benches_w361 import (
    bench_chain_homotopy_family,
    bench_covering_lift_family,
    bench_degree_map_family,
    bench_euler_homology_family,
    bench_homotopy_pi1_family,
    bench_simplicial_homology_family,
)
from quant_fund.research.benches_w362 import (
    bench_argument_principle_family,
    bench_cauchy_integral_family,
    bench_conformal_map_family,
    bench_laurent_series_family,
    bench_liouville_family,
    bench_residue_calc_family,
)
from quant_fund.research.benches_w363 import (
    bench_baire_category_family,
    bench_cantor_set_family,
    bench_egorov_thm_family,
    bench_fatou_lemma_family,
    bench_monotone_conv_family,
    bench_vitali_set_family,
)
from quant_fund.research.benches_w364 import (
    bench_adjoint_op_family,
    bench_compact_resolvent_family,
    bench_hahn_banach_family,
    bench_projection_thm_family,
    bench_riesz_repr_family,
    bench_selfadjoint_spectrum_family,
)
from quant_fund.research.benches_w365 import (
    bench_azuma_family,
    bench_coupling_arg_family,
    bench_doob_decomp_family,
    bench_ergodic_thm_family,
    bench_martingale_clt_family,
    bench_optional_stopping_family,
)
from quant_fund.research.benches_w366 import (
    bench_cw_complex_family,
    bench_excision_family,
    bench_homotopy_group_family,
    bench_poincare_dual_family,
    bench_singular_homology_family,
    bench_spectral_seq_toy_family,
)
from quant_fund.research.benches_w367 import (
    bench_cyclotomic_poly_family,
    bench_finite_field_family,
    bench_galois_corresp_family,
    bench_normality_check_family,
    bench_primitive_elem_family,
    bench_separable_check_family,
)
from quant_fund.research.benches_w368 import (
    bench_herbrand_model_family,
    bench_los_theorem_family,
    bench_presburger_family,
    bench_skolem_normal_family,
    bench_unification_fol_family,
)
from quant_fund.research.benches_w369 import (
    bench_aitken_delta_family,
    bench_brent_root_family,
    bench_broyden_family,
    bench_cheb_approx_family,
    bench_collocation_ode_family,
    bench_romberg_family,
)
from quant_fund.research.benches_w370 import (
    bench_dirac_ore_family,
    bench_graph_minor_family,
    bench_planar_five_family,
    bench_ramsey_num_family,
    bench_turan_theorem_family,
    bench_tutte_berge_family,
)
from quant_fund.research.benches_w371 import (
    bench_degree_mod2_family,
    bench_handle_decomp_family,
    bench_morse_theory_family,
    bench_poincare_hopf_family,
    bench_regular_value_family,
    bench_transversality_family,
)
from quant_fund.research.benches_w372 import (
    bench_girsanov_family,
    bench_ito_lemma_family,
    bench_local_time_family,
    bench_malliavin_family,
    bench_quadratic_var_family,
    bench_sde_strong_family,
)
from quant_fund.research.benches_w373 import (
    bench_bfgs_wolfe_family,
    bench_bundle_method_family,
    bench_frank_wolfe2_family,
    bench_ip_qp_family,
    bench_sqp_family,
    bench_trust_region_family,
)
from quant_fund.research.benches_w374 import (
    bench_acl_closure_family,
    bench_morley_rank_family,
    bench_omega_categoricity_family,
    bench_quantifier_elim_family,
    bench_realize_types_family,
    bench_vocab_interp_family,
)
from quant_fund.research.benches_w375 import (
    bench_blowup_family,
    bench_elliptic_group_family,
    bench_moduli_stable_family,
    bench_riemann_roch_family,
    bench_scheme_local_family,
    bench_sheaf_cohomology_family,
)
from quant_fund.research.benches_w376 import (
    bench_cofibration_family,
    bench_fibration_family,
    bench_serre_ss_family,
    bench_spectra_family,
    bench_suspension_family,
    bench_whitehead_family,
)
from quant_fund.research.benches_w377 import (
    bench_endomorphism_op_family,
    bench_little_discs_family,
    bench_may_recognition_family,
    bench_operad_assoc_family,
    bench_operad_comm_family,
    bench_operad_tree_family,
)
from quant_fund.research.benches_w378 import (
    bench_cohen_adds_family,
    bench_dense_filter_family,
    bench_forcing_poset_family,
    bench_large_cardinal_family,
    bench_ma_toy_family,
    bench_names_eval_family,
)
from quant_fund.research.benches_w379 import (
    bench_dual_matroid_family,
    bench_greedy_matroid_family,
    bench_matroid_axioms_family,
    bench_matroid_intersect_family,
    bench_matroid_union_family,
    bench_represented_matroid_family,
)
from quant_fund.research.benches_w380 import (
    bench_baire_space_family,
    bench_borel_functions_family,
    bench_determinacy_toy_family,
    bench_perfect_set_prop_family,
    bench_polish_topology_family,
    bench_souslin_op_family,
)
from quant_fund.research.benches_w381 import (
    bench_ample_test_family,
    bench_chow_ring_family,
    bench_grothendieck_grp_family,
    bench_gysin_family,
    bench_proj_morph_family,
    bench_toric_variety_family,
)
from quant_fund.research.benches_w382 import (
    bench_back_forth_family,
    bench_indiscernibles_family,
    bench_omitting_types_family,
    bench_saturation_test_family,
    bench_stability_spec_family,
    bench_stone_duality_family,
)
from quant_fund.research.benches_w383 import (
    bench_co_homology_family,
    bench_em_space_family,
    bench_loop_space_family,
    bench_mapping_cone_family,
    bench_stiefel_whitney_family,
    bench_transfer_family,
)
from quant_fund.research.benches_w384 import (
    bench_aut_group_family,
    bench_composition_series_family,
    bench_hall_subgroup_family,
    bench_permutation_poly_family,
    bench_schur_multiplier_family,
    bench_transfer_hom_family,
)
from quant_fund.research.benches_w385 import (
    bench_completion_ring_family,
    bench_dimension_fiber_family,
    bench_hilbert_samuel_family,
    bench_krull_dim_family,
    bench_noether_normal_family,
    bench_primary_decomp_family,
)
from quant_fund.research.benches_w386 import (
    bench_godel_incomp_family,
    bench_interp_proof_family,
    bench_modal_completeness_family,
    bench_natural_ded_family,
    bench_proof_complexity_family,
    bench_sequent_calculus_family,
)
from quant_fund.research.benches_w387 import (
    bench_concentration_ineq_family,
    bench_kolmogorov_01_family,
    bench_ldp_theory_family,
    bench_prokhorov_metric_family,
    bench_uniform_integrability_family,
    bench_vitali_conv_family,
)
from quant_fund.research.benches_w388 import (
    bench_derived_functor_family,
    bench_ext_compute_family,
    bench_koszul_homology_family,
    bench_mapping_degree_family,
    bench_spectral_seq_family,
    bench_tor_compute_family,
)
from quant_fund.research.benches_w389 import (
    bench_artin_symbol_family,
    bench_class_group_toy_family,
    bench_decomposition_group_family,
    bench_discriminant_field_family,
    bench_norm_subring_family,
    bench_ramification_family,
)
from quant_fund.research.benches_w390 import (
    bench_finite_difference_family,
    bench_hadamard_matrix_family,
    bench_inc_structure_family,
    bench_latin_trade_family,
    bench_orthogonal_array_family,
    bench_steiner_system_family,
)
from quant_fund.research.benches_w391 import (
    bench_closed_cat_family,
    bench_distributor_family,
    bench_equivalence_cat_family,
    bench_kan_extension_family,
    bench_monoidal_cat_family,
    bench_presheaf_family,
)
from quant_fund.research.benches_w392 import (
    bench_arithmetization_family,
    bench_diagonal_lemma_family,
    bench_fixed_point_combinator_family,
    bench_kleene_normal_family,
    bench_mu_recursion_family,
    bench_primitive_recursion_family,
)
from quant_fund.research.benches_w393 import (
    bench_cap_product_family,
    bench_eilenberg_steenrod_family,
    bench_k_theory_family,
    bench_obstruction_toy_family,
    bench_serre_class_family,
    bench_thom_isom_family,
)
from quant_fund.research.benches_w394 import (
    bench_birkhoff_rep_family,
    bench_dilworth_partition_family,
    bench_downset_lattice_family,
    bench_linear_extension_family,
    bench_sperner_bound_family,
    bench_zeta_mobius_family,
)
from quant_fund.research.benches_w395 import (
    bench_bell_triangle_family,
    bench_catalan_dp_family,
    bench_eulerian_num_family,
    bench_inclusion_excl_family,
    bench_partition_count_family,
    bench_stirling_cycle_family,
)
from quant_fund.research.benches_w396 import (
    bench_cm_points_family,
    bench_cyclotomic_field_family,
    bench_hensel_field_family,
    bench_idele_class_family,
    bench_kronecker_weber_family,
    bench_local_field_family,
)
from quant_fund.research.benches_w397 import (
    bench_bessel3_family,
    bench_h_transform_family,
    bench_occupation_bm_family,
    bench_ost_calcul_family,
    bench_reflect_bm_family,
    bench_tanaka_family,
)
from quant_fund.research.benches_w398 import (
    bench_artins_theorem_family,
    bench_clifford_toy_family,
    bench_frobenius_group_family,
    bench_induced_char_family,
    bench_schur_index_family,
    bench_tensor_char_family,
)
from quant_fund.research.benches_w399 import (
    bench_boolean_prime_family,
    bench_ef_game_toy_family,
    bench_fraisse_limit_family,
    bench_qe_dense_order_family,
    bench_real_closed_family,
    bench_vaught_test_family,
)
from quant_fund.research.benches_w400 import (
    bench_dual_ab_var_family,
    bench_etale_cover_family,
    bench_hom_stack_toy_family,
    bench_jacobian_toy_family,
    bench_picard_variety_family,
    bench_seesaw_theorem_family,
)
from quant_fund.research.benches_w401 import (
    bench_cut_elim_seq_family,
    bench_finitary_induct_family,
    bench_herbrand_thm_family,
    bench_hilbert_system_family,
    bench_interp_equality_family,
    bench_reverse_math_family,
)
from quant_fund.research.benches_w402 import (
    bench_etale_space_family,
    bench_geometric_morph_family,
    bench_groth_topo_family,
    bench_logic_topos_family,
    bench_sheaf_cond_family,
    bench_topos_subobj_family,
)
from quant_fund.research.benches_w403 import (
    bench_hopf_invariant_family,
    bench_j_hom_toy_family,
    bench_pi_stems_family,
    bench_spectral_atiyah_family,
    bench_thom_spectrum_family,
    bench_toda_bracket_family,
)
from quant_fund.research.benches_w404 import (
    bench_brace_operad_family,
    bench_little_intervals_family,
    bench_operad_algt_family,
    bench_operad_homology_family,
    bench_props_toy_family,
    bench_swiss_cheese_family,
)
from quant_fund.research.benches_w405 import (
    bench_bounded_complex_family,
    bench_derived_functor2_family,
    bench_koszul_dual_family,
    bench_mapping_cone_tri_family,
    bench_t_structure_family,
    bench_triangulated_family,
)
from quant_fund.research.benches_w406 import (
    bench_functor_derived_family,
    bench_hopf_algebra2_family,
    bench_kunneth_family,
    bench_leray_hirsch_family,
    bench_poincare_duality2_family,
    bench_universal_coeff_family,
)
from quant_fund.research.benches_w407 import (
    bench_adjunction2_family,
    bench_cech_cohom_family,
    bench_flattening_family,
    bench_hilbert_scheme_family,
    bench_scheme_fiber_family,
    bench_serre_duality_family,
)
from quant_fund.research.benches_w408 import (
    bench_brauer_alg_family,
    bench_bz_category_family,
    bench_casimir_op_family,
    bench_hecke_alg_family,
    bench_schur_functor_family,
    bench_weight_space_family,
)
from quant_fund.research.benches_w409 import (
    bench_cohend_family,
    bench_dold_kan_family,
    bench_eilenberg_zilber_family,
    bench_postnikov_family,
    bench_spectral_seq2_family,
    bench_stable_range_family,
)
from quant_fund.research.benches_w410 import (
    bench_descriptive3_family,
    bench_forcing2_family,
    bench_inner_model_family,
    bench_ordinal_notation_family,
    bench_proof_mining_family,
    bench_recursion3_family,
)
from quant_fund.research.benches_w411 import (
    bench_blow_up_family,
    bench_divisor_class_family,
    bench_dualizing_family,
    bench_intersection_mult_family,
    bench_normalization_family,
    bench_tangent_cone_family,
)
from quant_fund.research.benches_w412 import (
    bench_bicat_comp_family,
    bench_cat_enriched_family,
    bench_double_cat_family,
    bench_lax_functor_family,
    bench_mate_calc_family,
    bench_two_cat_family,
)
from quant_fund.research.benches_w413 import (
    bench_abelian_ext_family,
    bench_artin_lemma_family,
    bench_frobenius_el_family,
    bench_inseparable_family,
    bench_kummer_ext_family,
    bench_normal_basis_family,
)
from quant_fund.research.benches_w414 import (
    bench_adams_ss_family,
    bench_cofiber_family,
    bench_exact_couple_family,
    bench_obstruction_family,
    bench_stable_homotopy_family,
    bench_whitehead_thm_family,
)
from quant_fund.research.benches_w415 import (
    bench_homeo_top_family,
    bench_locally_compact_family,
    bench_open_cover_family,
    bench_paracompact_family,
    bench_partition_unity_family,
    bench_quotient_map_family,
)
from quant_fund.research.benches_w416 import (
    bench_decidable_theory_family,
    bench_definable_set_family,
    bench_indiscernible_seq_family,
    bench_interpol_thm_family,
    bench_omitting_prime_family,
    bench_saturated_model_family,
)
from quant_fund.research.benches_w417 import (
    bench_cohen_mac_family,
    bench_depth_ring_family,
    bench_free_resolution_family,
    bench_groebner_syz_family,
    bench_hilbert_syzygy_family,
    bench_regular_seq_family,
)
from quant_fund.research.benches_w418 import (
    bench_borel_cantelli_family,
    bench_clt_classic_family,
    bench_dominated_conv_family,
    bench_strong_lln_family,
    bench_uniform_lln_family,
    bench_weak_law_family,
)
from quant_fund.research.benches_w419 import (
    bench_bundle_section_family,
    bench_classify_space_family,
    bench_path_fibration_family,
    bench_serre_fibration_family,
    bench_thom_space_family,
    bench_vector_bundle_family,
)
from quant_fund.research.benches_w420 import (
    bench_dedekind_zeta_family,
    bench_dirichlet_unit_family,
    bench_ideal_class_family,
    bench_minkowski_bound_family,
    bench_regulator_family,
    bench_splitting_prime_family,
)
from quant_fund.research.benches_w421 import (
    bench_endo_coend_family,
    bench_frobenius_alg_family,
    bench_profunctor_toy_family,
    bench_span_compose_family,
    bench_star_autonomous_family,
    bench_traced_monoidal_family,
)
from quant_fund.research.benches_w422 import (
    bench_borel_subalgebra_family,
    bench_levi_factor_family,
    bench_nilpotent_orbit_family,
    bench_root_height_family,
    bench_verma_module_family,
    bench_weyl_chamber_family,
)
from quant_fund.research.benches_w423 import (
    bench_adams_diff_family,
    bench_cartan_eilenberg_family,
    bench_deriv_hom_family,
    bench_groth_spectral_family,
    bench_hypercohom_family,
    bench_serre_ss2_family,
)
from quant_fund.research.benches_w424 import (
    bench_closed_unbounded_family,
    bench_club_set_family,
    bench_mahlo_cardinal_family,
    bench_partition_calc_family,
    bench_stationary_set_family,
    bench_ultrafilter_toy_family,
)
from quant_fund.research.benches_w425 import (
    bench_coarse_space_family,
    bench_gerbe_toy_family,
    bench_moduli_stack_family,
    bench_quotient_stack_family,
    bench_stack_morph_family,
    bench_stacky_curve_family,
)
from quant_fund.research.benches_w426 import (
    bench_derived_alg_family,
    bench_infinity_cat_family,
    bench_model_category_family,
    bench_quillen_adj_family,
    bench_simplicial_set_family,
    bench_stable_cat_family,
)
from quant_fund.research.benches_w427 import (
    bench_bsd_toy_family,
    bench_elliptic_height_family,
    bench_lseries_toy_family,
    bench_modularity_toy_family,
    bench_mordell_weil_family,
    bench_padic_integral_family,
)
from quant_fund.research.benches_w428 import (
    bench_deformation_functor_family,
    bench_maurer_cartan_family,
    bench_obstruction_theory_family,
    bench_schlessinger_family,
    bench_tangent_space_def_family,
    bench_versal_deformation_family,
)
from quant_fund.research.benches_w429 import (
    bench_bass_heller_swan_family,
    bench_k0_group_family,
    bench_k1_group_family,
    bench_k_theory_spec_family,
    bench_milnor_k2_family,
    bench_quillen_q_family,
)
from quant_fund.research.benches_w430 import (
    bench_brane_tensor_family,
    bench_delooping_family,
    bench_e_n_algebra_family,
    bench_module_cat_family,
    bench_monoidal_infty_family,
    bench_operad_infty_family,
)
from quant_fund.research.benches_w431 import (
    bench_a1_homotopy_family,
    bench_milnor_operations_family,
    bench_morel_degree_family,
    bench_motivic_sphere_family,
    bench_slice_filtration_family,
    bench_voevodsky_motive_family,
)
from quant_fund.research.benches_w432 import (
    bench_atiyah_hirzebruch_family,
    bench_descent_ss_family,
    bench_leary_ss_family,
    bench_motivic_ss_family,
    bench_serre_ss3_family,
    bench_vanishing_ss_family,
)
from quant_fund.research.benches_w433 import (
    bench_adic_space_family,
    bench_berkovich_space_family,
    bench_diamond_toy_family,
    bench_etale_ph2_family,
    bench_perfectoid_space_family,
    bench_rigid_analytic_family,
)
from quant_fund.research.benches_w434 import (
    bench_automorphic_rep_family,
    bench_eisenstein_srs_family,
    bench_fourier_coeff_family,
    bench_hecke_operator_family,
    bench_langlands_dual_family,
    bench_satake_iso_family,
)
from quant_fund.research.benches_w435 import (
    bench_derived_fiber_family,
    bench_derived_scheme_family,
    bench_quasi_coherent_family,
    bench_shifted_symplectic_family,
    bench_spectral_scheme_family,
    bench_virtual_class_family,
)
from quant_fund.research.benches_w436 import (
    bench_adjoint_functor_family,
    bench_bousfield_loc_family,
    bench_cartesian_fib_family,
    bench_complete_seg_family,
    bench_presentable_cat_family,
    bench_straightening_family,
)
from quant_fund.research.benches_w437 import (
    bench_comparison_iso_family,
    bench_crystalline_coh_family,
    bench_derham_coh_family,
    bench_etale_coh_family,
    bench_frobenius_coh_family,
    bench_prismatic_coh_family,
)
from quant_fund.research.benches_w438 import (
    bench_analytic_ring_family,
    bench_condensed_set_family,
    bench_light_condensed_family,
    bench_liquid_group_family,
    bench_proetale_site_family,
    bench_solid_group_family,
)
from quant_fund.research.benches_w439 import (
    bench_formal_group_family,
    bench_formal_module_family,
    bench_height_strata_family,
    bench_lazard_ring_family,
    bench_lubin_tate_family,
    bench_morava_k_family,
)
from quant_fund.research.benches_w440 import (
    bench_bott_period_family,
    bench_hopf_map_family,
    bench_postnikov_twr_family,
    bench_stable_stem_family,
    bench_thom_iso_family,
    bench_whitehead_twr_family,
)
from quant_fund.research.benches_w441 import (
    bench_filtered_module_family,
    bench_fontaine_ring_family,
    bench_gal_rep_family,
    bench_hecke_eigensys_family,
    bench_ribet_toy_family,
    bench_weil_deligne_family,
)
from quant_fund.research.benches_w442 import (
    bench_cofibrant_rep_family,
    bench_enriched_model_family,
    bench_localization_mc_family,
    bench_monoidal_model_family,
    bench_quillen_equiv_family,
    bench_reedy_model_family,
)
from quant_fund.research.benches_w443 import (
    bench_brauer_grp_family,
    bench_chow_group_family,
    bench_milnor_conj_family,
    bench_motivic_coh_family,
    bench_motivic_stem_family,
    bench_voevodsky_dm_family,
)
from quant_fund.research.benches_w444 import (
    bench_cotangent_cx_family,
    bench_derived_stack_family,
    bench_geometric_stk_family,
    bench_perf_stack_family,
    bench_quasi_smooth_family,
    bench_tannaka_rec_family,
)
from quant_fund.research.benches_w445 import (
    bench_base_change_family,
    bench_constructible_family,
    bench_perverse_sh_family,
    bench_projection_frm_family,
    bench_six_functors_family,
    bench_verdier_dual_family,
)
from quant_fund.research.benches_w446 import (
    bench_calc_converge_family,
    bench_deriv_layer_family,
    bench_excisive_fn_family,
    bench_goodwillie_tower_family,
    bench_linearization_family,
    bench_orth_calc_family,
)
from quant_fund.research.benches_w447 import (
    bench_bord_cat_family,
    bench_chern_simons_family,
    bench_dw_theory_family,
    bench_extended_tqft_family,
    bench_frobenius_2d_family,
    bench_tqft_axiom_family,
)
from quant_fund.research.benches_w448 import (
    bench_cohesive_top_family,
    bench_hypercomplete_family,
    bench_infty_topos_family,
    bench_object_classif_family,
    bench_trunc_modal_family,
    bench_univ_colimit_family,
)
from quant_fund.research.benches_w449 import (
    bench_adic_generic_family,
    bench_dagger_space_family,
    bench_fargues_curve_family,
    bench_huber_ring_family,
    bench_prism_site_family,
    bench_witt_perfect_family,
)
from quant_fund.research.benches_w450 import (
    bench_exact_seq_family,
    bench_smash_monoidal_family,
    bench_spectra_cat_family,
    bench_stable_infty_family,
    bench_stable_tstruct_family,
    bench_thh_tc_family,
)
from quant_fund.research.benches_w451 import (
    bench_g_spectrum_family,
    bench_mackey_functor_family,
    bench_norm_map_family,
    bench_ro_grading_family,
    bench_tom_dieck_family,
    bench_wirthmuller_family,
)
from quant_fund.research.benches_w452 import (
    bench_d_module_family,
    bench_geometric_langlands_family,
    bench_hecke_eig_family,
    bench_kernel_fun_family,
    bench_opers_g_family,
    bench_ramified_l_family,
)
from quant_fund.research.benches_w810 import (
    bench_anytime_valid,
    bench_distributional_ml,
    bench_energy_score,
    bench_leakage_redteam,
    bench_regime_eval,
    bench_ts_conformal,
)
from quant_fund.research.catalog import (
    BENCHMARK_CATALOG_VERSION,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    dist_crps_eprocess_keys_present,
    family_blob_executed,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
    family_blob_nonempty,
    tail_es_battery_keys_present,
    tail_var_battery_keys_present,
)
from quant_fund.utils.hashing import canonical_frame_fingerprint, hash_bytes, hash_file
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256
from quant_fund.utils.seeds import set_global_seed


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete text artifact without exposing partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass
class HypothesisResult:
    id: str
    statement: str
    test: str
    statistic: float
    p_value: float
    reject_raw: bool
    reject_fdr: bool
    decision: str
    family: str = "discovery"
    meets_floor: bool | None = None


@dataclass
class ResearchNotebook:
    schema_version: int
    firm: str
    product: str
    version: str
    generated_at: str
    data_source: str
    synthetic: bool
    disclaimer: str
    ranking_target: str
    claim: str
    families: dict[str, Any]
    rankers: list[dict[str, Any]]
    hypotheses: list[HypothesisResult]
    scorecard: dict[str, dict[str, Any]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    backtest_overfitting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))


def format_p_value(value: float) -> str:
    """Render p-values without turning floating-point underflow into ``0``."""
    p = float(value)
    if not np.isfinite(p):
        return "n/a"
    if p <= 0.0:
        return f"<{np.finfo(float).tiny:.3g}"
    return f"{p:.4g}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None:
        return None
    return str(obj)


def _git_revision() -> str:
    """Return the checked-out revision, or an explicit unknown marker."""
    return git_revision()


def _git_worktree_sha256() -> str:
    """Hash tracked and relevant untracked changes so dirty runs get new IDs."""
    return git_worktree_sha256()


def _provenance(
    config: AppConfig,
    frame: pl.DataFrame,
    label: str,
    *,
    northset_frame: pl.DataFrame | None = None,
) -> dict[str, Any]:
    revision = _git_revision()
    worktree_hash = _git_worktree_sha256()
    columns = sorted(frame.columns)
    scope = {
        "rows": frame.height,
        "columns": columns,
        "label": label,
        "source": str(config.data.source),
        "bignum": bench_bignum_family,
        "fft_radix2": bench_fft_radix2_family,
        "int_sqrt": bench_int_sqrt_family,
        "karatsuba": bench_karatsuba_family,
        "ntt": bench_ntt_family,
        "strassen": bench_strassen_family,
        "inverted_index": bench_inverted_index_family,
        "lsh_dedup": bench_lsh_dedup_family,
        "ngram_spell": bench_ngram_spell_family,
        "positional_index": bench_positional_index_family,
        "posting_merge": bench_posting_merge_family,
        "wand_bmw": bench_wand_bmw_family,
        "elf_loader": bench_elf_loader_family,
        "malloc_freelist": bench_malloc_freelist_family,
        "mlfq_sched": bench_mlfq_sched_family,
        "mmap_pager": bench_mmap_pager_family,
        "semaphore_monitor": bench_semaphore_monitor_family,
        "syscall_layer": bench_syscall_layer_family,
        "bytecode_vm": bench_bytecode_vm_family,
        "closure_conv": bench_closure_conv_family,
        "inline_cache": bench_inline_cache_family,
        "nan_tagging": bench_nan_tagging_family,
        "tail_call_tramp": bench_tail_call_tramp_family,
        "threaded_interp": bench_threaded_interp_family,
        "atomics_tas": bench_atomics_tas_family,
        "bakery_lock": bench_bakery_lock_family,
        "channel_select": bench_channel_select_family,
        "peterson_lock": bench_peterson_lock_family,
        "rw_lock": bench_rw_lock_family,
        "work_stealing": bench_work_stealing_family,
        "brzozowski_deriv": bench_brzozowski_deriv_family,
        "cellular_automata": bench_cellular_automata_family,
        "dfa_equiv": bench_dfa_equiv_family,
        "mealy_moore": bench_mealy_moore_family,
        "pda_sim": bench_pda_sim_family,
        "turing_machine": bench_turing_machine_family,
        "debruijn_assemble": bench_debruijn_assemble_family,
        "fm_index": bench_fm_index_family,
        "motif_scan": bench_motif_scan_family,
        "needleman_wunsch": bench_needleman_wunsch_family,
        "smith_waterman": bench_smith_waterman_family,
        "upgma_tree": bench_upgma_tree_family,
        "astar_search": bench_astar_search_family,
        "bidirectional_dijkstra": bench_bidirectional_dijkstra_family,
        "bron_kerbosch": bench_bron_kerbosch_family,
        "critical_path": bench_critical_path_family,
        "dinic_flow": bench_dinic_flow_family,
        "mincost_flow": bench_mincost_flow_family,
        "givens_qr": bench_givens_qr_family,
        "jacobi_svd": bench_jacobi_svd_family,
        "ldlt_solve": bench_ldlt_solve_family,
        "lu_pivots": bench_lu_pivots_family,
        "orth_iter": bench_orth_iter_family,
        "sturm_eig": bench_sturm_eig_family,
        "gen_gc": bench_gen_gc_family,
        "compacting_gc": bench_compacting_gc_family,
        "dispatch_table": bench_dispatch_table_family,
        "poly_inline_cache": bench_poly_inline_cache_family,
        "anf_cps": bench_anf_cps_family,
        "trampoline_tc": bench_trampoline_tc_family,
        "tree_automata": bench_tree_automata_family,
        "buchi_automata": bench_buchi_automata_family,
        "weighted_fst": bench_weighted_fst_family,
        "cfg_pda_equiv": bench_cfg_pda_equiv_family,
        "two_way_dfa": bench_two_way_dfa_family,
        "register_automata": bench_register_automata_family,
        "tls_handshake": bench_tls_handshake_family,
        "hmac_construct": bench_hmac_construct_family,
        "aead_etm": bench_aead_etm_family,
        "merkle_damgard": bench_merkle_damgard_family,
        "cbc_padding": bench_cbc_padding_family,
        "pbkdf2_kdf": bench_pbkdf2_kdf_family,
        "simplex_lp": bench_simplex_lp_family,
        "ellipsoid_method": bench_ellipsoid_method_family,
        "barrier_ip": bench_barrier_ip_family,
        "admm_lasso": bench_admm_lasso_family,
        "coord_descent": bench_coord_descent_family,
        "proj_gradient": bench_proj_gradient_family,
        "hazard_pointer": bench_hazard_pointer_family,
        "seqlock": bench_seqlock_family,
        "ms_queue": bench_ms_queue_family,
        "epoch_reclaim": bench_epoch_reclaim_family,
        "flat_combining": bench_flat_combining_family,
        "rcu_lock": bench_rcu_lock_family,
        "lwe_kex": bench_lwe_kex_family,
        "ntru_toy": bench_ntru_toy_family,
        "bfv_fhe": bench_bfv_fhe_family,
        "sis_hash": bench_sis_hash_family,
        "sigma_or_proof": bench_sigma_or_proof_family,
        "chaum_pedersen": bench_chaum_pedersen_family,
        "cascades_opt": bench_cascades_opt_family,
        "vectorized_exec": bench_vectorized_exec_family,
        "zone_map": bench_zone_map_family,
        "func_dep": bench_func_dep_family,
        "bitmap_index": bench_bitmap_index_family,
        "adaptive_qp": bench_adaptive_qp_family,
        "bgp_pathvec": bench_bgp_pathvec_family,
        "dns_resolver": bench_dns_resolver_family,
        "nat_traversal": bench_nat_traversal_family,
        "arp_table": bench_arp_table_family,
        "dhcp_lease": bench_dhcp_lease_family,
        "eth_switch": bench_eth_switch_family,
        "gale_shapley": bench_gale_shapley_family,
        "hopcroft_karp": bench_hopcroft_karp_family,
        "kuhn_munkres": bench_kuhn_munkres_family,
        "konig_cover": bench_konig_cover_family,
        "gale_chu": bench_gale_chu_family,
        "topo_layers": bench_topo_layers_family,
        "ekf_slam": bench_ekf_slam_family,
        "occupancy_grid": bench_occupancy_grid_family,
        "pure_pursuit": bench_pure_pursuit_family,
        "stanley": bench_stanley_family,
        "particle_slam": bench_particle_slam_family,
        "frontier_explore": bench_frontier_explore_family,
        "stencil_halo": bench_stencil_halo_family,
        "mesi_cache": bench_mesi_cache_family,
        "ring_allreduce": bench_ring_allreduce_family,
        "simd_lanes": bench_simd_lanes_family,
        "task_dag": bench_task_dag_family,
        "numa_alloc": bench_numa_alloc_family,
        "edf_scheduler": bench_edf_scheduler_family,
        "rms_scheduler": bench_rms_scheduler_family,
        "wcet_est": bench_wcet_est_family,
        "debounce_fsm": bench_debounce_fsm_family,
        "watchdog_task": bench_watchdog_task_family,
        "ring_buffer": bench_ring_buffer_family,
        "divide_conquer_eig": bench_divide_conquer_eig_family,
        "dqds": bench_dqds_family,
        "block_lanczos": bench_block_lanczos_family,
        "randomized_qb": bench_randomized_qb_family,
        "sparse_cholesky": bench_sparse_cholesky_family,
        "fgmres": bench_fgmres_family,
        "fuzzer_mutate": bench_fuzzer_mutate_family,
        "taint_track": bench_taint_track_family,
        "asan_shadow": bench_asan_shadow_family,
        "symbolic_exec": bench_symbolic_exec_family,
        "contract_check": bench_contract_check_family,
        "grammar_fuzz": bench_grammar_fuzz_family,
        "triangle_raster": bench_triangle_raster_family,
        "phong_shade": bench_phong_shade_family,
        "mipmap_sample": bench_mipmap_sample_family,
        "shadow_map": bench_shadow_map_family,
        "bump_map": bench_bump_map_family,
        "ssao_lite": bench_ssao_lite_family,
        "warp_scheduler": bench_warp_scheduler_family,
        "simt_divergence": bench_simt_divergence_family,
        "bank_conflict": bench_bank_conflict_family,
        "mem_coalesce": bench_mem_coalesce_family,
        "occupancy_calc": bench_occupancy_calc_family,
        "shared_mem_tile": bench_shared_mem_tile_family,
        "elgamal_enc": bench_elgamal_enc_family,
        "paillier_he": bench_paillier_he_family,
        "fiat_shamir": bench_fiat_shamir_family,
        "ot_12": bench_ot_12_family,
        "chacha_stream": bench_chacha_stream_family,
        "poly1305_mac": bench_poly1305_mac_family,
        "lk_flow": bench_lk_flow_family,
        "orb_feature": bench_orb_feature_family,
        "homography_4pt": bench_homography_4pt_family,
        "ransac_plane": bench_ransac_plane_family,
        "epipolar_8pt": bench_epipolar_8pt_family,
        "stereo_disparity": bench_stereo_disparity_family,
        "lj_md": bench_lj_md_family,
        "fdtd_wave": bench_fdtd_wave_family,
        "lattice_boltzmann": bench_lattice_boltzmann_family,
        "ising_metro": bench_ising_metro_family,
        "pic_plasma": bench_pic_plasma_family,
        "dmc_solver": bench_dmc_solver_family,
        "ospf_lsa": bench_ospf_lsa_family,
        "stp_spanning": bench_stp_spanning_family,
        "vlan_tag": bench_vlan_tag_family,
        "csma_ca": bench_csma_ca_family,
        "icmp_path": bench_icmp_path_family,
        "diffserv_qos": bench_diffserv_qos_family,
        "pid_antiwindup": bench_pid_antiwindup_family,
        "sliding_mode": bench_sliding_mode_family,
        "gain_schedule": bench_gain_schedule_family,
        "smith_predictor": bench_smith_predictor_family,
        "backstepping": bench_backstepping_family,
        "repetitive_ctrl": bench_repetitive_ctrl_family,
        "partial_eval": bench_partial_eval_family,
        "peephole_opt": bench_peephole_opt_family,
        "strength_red": bench_strength_red_family,
        "const_fold": bench_const_fold_family,
        "loop_unroll": bench_loop_unroll_family,
        "inline_expand": bench_inline_expand_family,
        "hmm_profile": bench_hmm_profile_family,
        "star_msa": bench_star_msa_family,
        "gc_skew": bench_gc_skew_family,
        "orf_find": bench_orf_find_family,
        "kmer_count": bench_kmer_count_family,
        "seq_logo": bench_seq_logo_family,
        "columnar_scan": bench_columnar_scan_family,
        "simd_filter": bench_simd_filter_family,
        "late_materialize": bench_late_materialize_family,
        "radix_join": bench_radix_join_family,
        "graceful_hash": bench_graceful_hash_family,
        "index_intersect": bench_index_intersect_family,
        "ra_mutex": bench_ra_mutex_family,
        "token_ring": bench_token_ring_family,
        "bully_elect": bench_bully_elect_family,
        "chord_look": bench_chord_look_family,
        "quorum_rw": bench_quorum_rw_family,
        "causal_bcast": bench_causal_bcast_family,
        "stft_istft": bench_stft_istft_family,
        "chirp_z": bench_chirp_z_family,
        "fir_window": bench_fir_window_family,
        "prony_model": bench_prony_model_family,
        "wola_synth": bench_wola_synth_family,
        "decimate_int": bench_decimate_int_family,
        "rbc_sim": bench_rbc_sim_family,
        "nk_phillips": bench_nk_phillips_family,
        "taylor_rule": bench_taylor_rule_family,
        "solow_model": bench_solow_model_family,
        "olg_model": bench_olg_model_family,
        "cobweb_model": bench_cobweb_model_family,
        "luen_obsv": bench_luen_obsv_family,
        "dist_obsv": bench_dist_obsv_family,
        "mrac_adapt": bench_mrac_adapt_family,
        "flat_track": bench_flat_track_family,
        "lyap_synth": bench_lyap_synth_family,
        "l2_gain": bench_l2_gain_family,
        "simp_betti": bench_simp_betti_family,
        "boundary_sq": bench_boundary_sq_family,
        "euler_char": bench_euler_char_family,
        "rips_h1": bench_rips_h1_family,
        "graph_h1": bench_graph_h1_family,
        "winding_deg": bench_winding_deg_family,
        "group_table": bench_group_table_family,
        "perm_group": bench_perm_group_family,
        "galois_field": bench_galois_field_family,
        "poly_ring": bench_poly_ring_family,
        "ideal_member": bench_ideal_member_family,
        "matrix_grp": bench_matrix_grp_family,
        "subset_sum_dp": bench_subset_sum_dp_family,
        "stirling_count": bench_stirling_count_family,
        "gray_code": bench_gray_code_family,
        "inversion_count": bench_inversion_count_family,
        "ramsey_bound": bench_ramsey_bound_family,
        "latin_square": bench_latin_square_family,
        "d_module": bench_d_module_family,
        "geometric_langlands": bench_geometric_langlands_family,
        "hecke_eig": bench_hecke_eig_family,
        "opers_g": bench_opers_g_family,
        "ramified_l": bench_ramified_l_family,
        "kernel_fun": bench_kernel_fun_family,
        "g_spectrum": bench_g_spectrum_family,
        "mackey_functor": bench_mackey_functor_family,
        "norm_map": bench_norm_map_family,
        "ro_grading": bench_ro_grading_family,
        "wirthmuller": bench_wirthmuller_family,
        "tom_dieck": bench_tom_dieck_family,
        "stable_infty": bench_stable_infty_family,
        "spectra_cat": bench_spectra_cat_family,
        "exact_seq": bench_exact_seq_family,
        "stable_tstruct": bench_stable_tstruct_family,
        "smash_monoidal": bench_smash_monoidal_family,
        "thh_tc": bench_thh_tc_family,
        "dagger_space": bench_dagger_space_family,
        "huber_ring": bench_huber_ring_family,
        "adic_generic": bench_adic_generic_family,
        "witt_perfect": bench_witt_perfect_family,
        "fargues_curve": bench_fargues_curve_family,
        "prism_site": bench_prism_site_family,
        "infty_topos": bench_infty_topos_family,
        "univ_colimit": bench_univ_colimit_family,
        "object_classif": bench_object_classif_family,
        "trunc_modal": bench_trunc_modal_family,
        "cohesive_top": bench_cohesive_top_family,
        "hypercomplete": bench_hypercomplete_family,
        "tqft_axiom": bench_tqft_axiom_family,
        "bord_cat": bench_bord_cat_family,
        "frobenius_2d": bench_frobenius_2d_family,
        "extended_tqft": bench_extended_tqft_family,
        "dw_theory": bench_dw_theory_family,
        "chern_simons": bench_chern_simons_family,
        "goodwillie_tower": bench_goodwillie_tower_family,
        "excisive_fn": bench_excisive_fn_family,
        "linearization": bench_linearization_family,
        "deriv_layer": bench_deriv_layer_family,
        "calc_converge": bench_calc_converge_family,
        "orth_calc": bench_orth_calc_family,
        "six_functors": bench_six_functors_family,
        "base_change": bench_base_change_family,
        "projection_frm": bench_projection_frm_family,
        "verdier_dual": bench_verdier_dual_family,
        "constructible": bench_constructible_family,
        "perverse_sh": bench_perverse_sh_family,
        "derived_stack": bench_derived_stack_family,
        "cotangent_cx": bench_cotangent_cx_family,
        "geometric_stk": bench_geometric_stk_family,
        "tannaka_rec": bench_tannaka_rec_family,
        "quasi_smooth": bench_quasi_smooth_family,
        "perf_stack": bench_perf_stack_family,
        "motivic_coh": bench_motivic_coh_family,
        "chow_group": bench_chow_group_family,
        "milnor_conj": bench_milnor_conj_family,
        "voevodsky_dm": bench_voevodsky_dm_family,
        "motivic_stem": bench_motivic_stem_family,
        "brauer_grp": bench_brauer_grp_family,
        "cofibrant_rep": bench_cofibrant_rep_family,
        "quillen_equiv": bench_quillen_equiv_family,
        "monoidal_model": bench_monoidal_model_family,
        "enriched_model": bench_enriched_model_family,
        "reedy_model": bench_reedy_model_family,
        "localization_mc": bench_localization_mc_family,
        "gal_rep": bench_gal_rep_family,
        "fontaine_ring": bench_fontaine_ring_family,
        "filtered_module": bench_filtered_module_family,
        "weil_deligne": bench_weil_deligne_family,
        "hecke_eigensys": bench_hecke_eigensys_family,
        "ribet_toy": bench_ribet_toy_family,
        "thom_iso": bench_thom_iso_family,
        "postnikov_twr": bench_postnikov_twr_family,
        "whitehead_twr": bench_whitehead_twr_family,
        "bott_period": bench_bott_period_family,
        "stable_stem": bench_stable_stem_family,
        "hopf_map": bench_hopf_map_family,
        "formal_group": bench_formal_group_family,
        "lazard_ring": bench_lazard_ring_family,
        "formal_module": bench_formal_module_family,
        "height_strata": bench_height_strata_family,
        "lubin_tate": bench_lubin_tate_family,
        "morava_k": bench_morava_k_family,
        "condensed_set": bench_condensed_set_family,
        "solid_group": bench_solid_group_family,
        "liquid_group": bench_liquid_group_family,
        "proetale_site": bench_proetale_site_family,
        "light_condensed": bench_light_condensed_family,
        "analytic_ring": bench_analytic_ring_family,
        "crystalline_coh": bench_crystalline_coh_family,
        "prismatic_coh": bench_prismatic_coh_family,
        "etale_coh": bench_etale_coh_family,
        "derham_coh": bench_derham_coh_family,
        "frobenius_coh": bench_frobenius_coh_family,
        "comparison_iso": bench_comparison_iso_family,
        "complete_seg": bench_complete_seg_family,
        "cartesian_fib": bench_cartesian_fib_family,
        "straightening": bench_straightening_family,
        "presentable_cat": bench_presentable_cat_family,
        "adjoint_functor": bench_adjoint_functor_family,
        "bousfield_loc": bench_bousfield_loc_family,
        "derived_scheme": bench_derived_scheme_family,
        "quasi_coherent": bench_quasi_coherent_family,
        "derived_fiber": bench_derived_fiber_family,
        "spectral_scheme": bench_spectral_scheme_family,
        "virtual_class": bench_virtual_class_family,
        "shifted_symplectic": bench_shifted_symplectic_family,
        "satake_iso": bench_satake_iso_family,
        "hecke_operator": bench_hecke_operator_family,
        "langlands_dual": bench_langlands_dual_family,
        "eisenstein_srs": bench_eisenstein_srs_family,
        "automorphic_rep": bench_automorphic_rep_family,
        "fourier_coeff": bench_fourier_coeff_family,
        "rigid_analytic": bench_rigid_analytic_family,
        "berkovich_space": bench_berkovich_space_family,
        "perfectoid_space": bench_perfectoid_space_family,
        "adic_space": bench_adic_space_family,
        "etale_ph2": bench_etale_ph2_family,
        "diamond_toy": bench_diamond_toy_family,
        "atiyah_hirzebruch": bench_atiyah_hirzebruch_family,
        "serre_ss3": bench_serre_ss3_family,
        "leary_ss": bench_leary_ss_family,
        "descent_ss": bench_descent_ss_family,
        "motivic_ss": bench_motivic_ss_family,
        "vanishing_ss": bench_vanishing_ss_family,
        "a1_homotopy": bench_a1_homotopy_family,
        "motivic_sphere": bench_motivic_sphere_family,
        "morel_degree": bench_morel_degree_family,
        "voevodsky_motive": bench_voevodsky_motive_family,
        "slice_filtration": bench_slice_filtration_family,
        "milnor_operations": bench_milnor_operations_family,
        "e_n_algebra": bench_e_n_algebra_family,
        "operad_infty": bench_operad_infty_family,
        "monoidal_infty": bench_monoidal_infty_family,
        "module_cat": bench_module_cat_family,
        "brane_tensor": bench_brane_tensor_family,
        "delooping": bench_delooping_family,
        "k0_group": bench_k0_group_family,
        "k1_group": bench_k1_group_family,
        "milnor_k2": bench_milnor_k2_family,
        "quillen_q": bench_quillen_q_family,
        "k_theory_spec": bench_k_theory_spec_family,
        "bass_heller_swan": bench_bass_heller_swan_family,
        "deformation_functor": bench_deformation_functor_family,
        "schlessinger": bench_schlessinger_family,
        "tangent_space_def": bench_tangent_space_def_family,
        "obstruction_theory": bench_obstruction_theory_family,
        "versal_deformation": bench_versal_deformation_family,
        "maurer_cartan": bench_maurer_cartan_family,
        "elliptic_height": bench_elliptic_height_family,
        "mordell_weil": bench_mordell_weil_family,
        "lseries_toy": bench_lseries_toy_family,
        "bsd_toy": bench_bsd_toy_family,
        "modularity_toy": bench_modularity_toy_family,
        "padic_integral": bench_padic_integral_family,
        "model_category": bench_model_category_family,
        "quillen_adj": bench_quillen_adj_family,
        "simplicial_set": bench_simplicial_set_family,
        "infinity_cat": bench_infinity_cat_family,
        "derived_alg": bench_derived_alg_family,
        "stable_cat": bench_stable_cat_family,
        "moduli_stack": bench_moduli_stack_family,
        "stacky_curve": bench_stacky_curve_family,
        "coarse_space": bench_coarse_space_family,
        "quotient_stack": bench_quotient_stack_family,
        "gerbe_toy": bench_gerbe_toy_family,
        "stack_morph": bench_stack_morph_family,
        "club_set": bench_club_set_family,
        "stationary_set": bench_stationary_set_family,
        "ultrafilter_toy": bench_ultrafilter_toy_family,
        "partition_calc": bench_partition_calc_family,
        "closed_unbounded": bench_closed_unbounded_family,
        "mahlo_cardinal": bench_mahlo_cardinal_family,
        "groth_spectral": bench_groth_spectral_family,
        "serre_ss2": bench_serre_ss2_family,
        "hypercohom": bench_hypercohom_family,
        "deriv_hom": bench_deriv_hom_family,
        "cartan_eilenberg": bench_cartan_eilenberg_family,
        "adams_diff": bench_adams_diff_family,
        "weyl_chamber": bench_weyl_chamber_family,
        "root_height": bench_root_height_family,
        "borel_subalgebra": bench_borel_subalgebra_family,
        "levi_factor": bench_levi_factor_family,
        "nilpotent_orbit": bench_nilpotent_orbit_family,
        "verma_module": bench_verma_module_family,
        "traced_monoidal": bench_traced_monoidal_family,
        "star_autonomous": bench_star_autonomous_family,
        "frobenius_alg": bench_frobenius_alg_family,
        "span_compose": bench_span_compose_family,
        "profunctor_toy": bench_profunctor_toy_family,
        "endo_coend": bench_endo_coend_family,
        "dirichlet_unit": bench_dirichlet_unit_family,
        "regulator": bench_regulator_family,
        "ideal_class": bench_ideal_class_family,
        "minkowski_bound": bench_minkowski_bound_family,
        "dedekind_zeta": bench_dedekind_zeta_family,
        "splitting_prime": bench_splitting_prime_family,
        "serre_fibration": bench_serre_fibration_family,
        "path_fibration": bench_path_fibration_family,
        "bundle_section": bench_bundle_section_family,
        "classify_space": bench_classify_space_family,
        "vector_bundle": bench_vector_bundle_family,
        "thom_space": bench_thom_space_family,
        "weak_law": bench_weak_law_family,
        "strong_lln": bench_strong_lln_family,
        "clt_classic": bench_clt_classic_family,
        "borel_cantelli": bench_borel_cantelli_family,
        "dominated_conv": bench_dominated_conv_family,
        "uniform_lln": bench_uniform_lln_family,
        "groebner_syz": bench_groebner_syz_family,
        "free_resolution": bench_free_resolution_family,
        "hilbert_syzygy": bench_hilbert_syzygy_family,
        "regular_seq": bench_regular_seq_family,
        "depth_ring": bench_depth_ring_family,
        "cohen_mac": bench_cohen_mac_family,
        "decidable_theory": bench_decidable_theory_family,
        "indiscernible_seq": bench_indiscernible_seq_family,
        "saturated_model": bench_saturated_model_family,
        "omitting_prime": bench_omitting_prime_family,
        "interpol_thm": bench_interpol_thm_family,
        "definable_set": bench_definable_set_family,
        "quotient_map": bench_quotient_map_family,
        "open_cover": bench_open_cover_family,
        "locally_compact": bench_locally_compact_family,
        "homeo_top": bench_homeo_top_family,
        "paracompact": bench_paracompact_family,
        "partition_unity": bench_partition_unity_family,
        "exact_couple": bench_exact_couple_family,
        "adams_ss": bench_adams_ss_family,
        "stable_homotopy": bench_stable_homotopy_family,
        "whitehead_thm": bench_whitehead_thm_family,
        "obstruction": bench_obstruction_family,
        "cofiber": bench_cofiber_family,
        "artin_lemma": bench_artin_lemma_family,
        "normal_basis": bench_normal_basis_family,
        "kummer_ext": bench_kummer_ext_family,
        "abelian_ext": bench_abelian_ext_family,
        "frobenius_el": bench_frobenius_el_family,
        "inseparable": bench_inseparable_family,
        "two_cat": bench_two_cat_family,
        "bicat_comp": bench_bicat_comp_family,
        "mate_calc": bench_mate_calc_family,
        "double_cat": bench_double_cat_family,
        "lax_functor": bench_lax_functor_family,
        "cat_enriched": bench_cat_enriched_family,
        "blow_up": bench_blow_up_family,
        "intersection_mult": bench_intersection_mult_family,
        "tangent_cone": bench_tangent_cone_family,
        "normalization": bench_normalization_family,
        "divisor_class": bench_divisor_class_family,
        "dualizing": bench_dualizing_family,
        "forcing2": bench_forcing2_family,
        "inner_model": bench_inner_model_family,
        "descriptive3": bench_descriptive3_family,
        "recursion3": bench_recursion3_family,
        "proof_mining": bench_proof_mining_family,
        "ordinal_notation": bench_ordinal_notation_family,
        "spectral_seq2": bench_spectral_seq2_family,
        "eilenberg_zilber": bench_eilenberg_zilber_family,
        "dold_kan": bench_dold_kan_family,
        "postnikov": bench_postnikov_family,
        "stable_range": bench_stable_range_family,
        "cohend": bench_cohend_family,
        "schur_functor": bench_schur_functor_family,
        "brauer_alg": bench_brauer_alg_family,
        "hecke_alg": bench_hecke_alg_family,
        "casimir_op": bench_casimir_op_family,
        "weight_space": bench_weight_space_family,
        "bz_category": bench_bz_category_family,
        "cech_cohom": bench_cech_cohom_family,
        "serre_duality": bench_serre_duality_family,
        "adjunction2": bench_adjunction2_family,
        "scheme_fiber": bench_scheme_fiber_family,
        "hilbert_scheme": bench_hilbert_scheme_family,
        "flattening": bench_flattening_family,
        "poincare_duality2": bench_poincare_duality2_family,
        "universal_coeff": bench_universal_coeff_family,
        "kunneth": bench_kunneth_family,
        "leray_hirsch": bench_leray_hirsch_family,
        "hopf_algebra2": bench_hopf_algebra2_family,
        "functor_derived": bench_functor_derived_family,
        "derived_functor2": bench_derived_functor2_family,
        "triangulated": bench_triangulated_family,
        "bounded_complex": bench_bounded_complex_family,
        "mapping_cone_tri": bench_mapping_cone_tri_family,
        "koszul_dual": bench_koszul_dual_family,
        "t_structure": bench_t_structure_family,
        "operad_algt": bench_operad_algt_family,
        "brace_operad": bench_brace_operad_family,
        "swiss_cheese": bench_swiss_cheese_family,
        "little_intervals": bench_little_intervals_family,
        "operad_homology": bench_operad_homology_family,
        "props_toy": bench_props_toy_family,
        "j_hom_toy": bench_j_hom_toy_family,
        "toda_bracket": bench_toda_bracket_family,
        "spectral_atiyah": bench_spectral_atiyah_family,
        "pi_stems": bench_pi_stems_family,
        "hopf_invariant": bench_hopf_invariant_family,
        "thom_spectrum": bench_thom_spectrum_family,
        "topos_subobj": bench_topos_subobj_family,
        "groth_topo": bench_groth_topo_family,
        "sheaf_cond": bench_sheaf_cond_family,
        "logic_topos": bench_logic_topos_family,
        "geometric_morph": bench_geometric_morph_family,
        "etale_space": bench_etale_space_family,
        "herbrand_thm": bench_herbrand_thm_family,
        "interp_equality": bench_interp_equality_family,
        "cut_elim_seq": bench_cut_elim_seq_family,
        "finitary_induct": bench_finitary_induct_family,
        "hilbert_system": bench_hilbert_system_family,
        "reverse_math": bench_reverse_math_family,
        "etale_cover": bench_etale_cover_family,
        "jacobian_toy": bench_jacobian_toy_family,
        "hom_stack_toy": bench_hom_stack_toy_family,
        "seesaw_theorem": bench_seesaw_theorem_family,
        "picard_variety": bench_picard_variety_family,
        "dual_ab_var": bench_dual_ab_var_family,
        "ef_game_toy": bench_ef_game_toy_family,
        "vaught_test": bench_vaught_test_family,
        "real_closed": bench_real_closed_family,
        "boolean_prime": bench_boolean_prime_family,
        "fraisse_limit": bench_fraisse_limit_family,
        "qe_dense_order": bench_qe_dense_order_family,
        "induced_char": bench_induced_char_family,
        "artins_theorem": bench_artins_theorem_family,
        "tensor_char": bench_tensor_char_family,
        "clifford_toy": bench_clifford_toy_family,
        "schur_index": bench_schur_index_family,
        "frobenius_group": bench_frobenius_group_family,
        "ost_calcul": bench_ost_calcul_family,
        "tanaka": bench_tanaka_family,
        "bessel3": bench_bessel3_family,
        "reflect_bm": bench_reflect_bm_family,
        "occupation_bm": bench_occupation_bm_family,
        "h_transform": bench_h_transform_family,
        "cyclotomic_field": bench_cyclotomic_field_family,
        "kronecker_weber": bench_kronecker_weber_family,
        "local_field": bench_local_field_family,
        "hensel_field": bench_hensel_field_family,
        "cm_points": bench_cm_points_family,
        "idele_class": bench_idele_class_family,
        "catalan_dp": bench_catalan_dp_family,
        "stirling_cycle": bench_stirling_cycle_family,
        "partition_count": bench_partition_count_family,
        "bell_triangle": bench_bell_triangle_family,
        "eulerian_num": bench_eulerian_num_family,
        "inclusion_excl": bench_inclusion_excl_family,
        "downset_lattice": bench_downset_lattice_family,
        "zeta_mobius": bench_zeta_mobius_family,
        "linear_extension": bench_linear_extension_family,
        "sperner_bound": bench_sperner_bound_family,
        "dilworth_partition": bench_dilworth_partition_family,
        "birkhoff_rep": bench_birkhoff_rep_family,
        "eilenberg_steenrod": bench_eilenberg_steenrod_family,
        "cap_product": bench_cap_product_family,
        "thom_isom": bench_thom_isom_family,
        "serre_class": bench_serre_class_family,
        "obstruction_toy": bench_obstruction_toy_family,
        "k_theory": bench_k_theory_family,
        "mu_recursion": bench_mu_recursion_family,
        "primitive_recursion": bench_primitive_recursion_family,
        "diagonal_lemma": bench_diagonal_lemma_family,
        "arithmetization": bench_arithmetization_family,
        "fixed_point_combinator": bench_fixed_point_combinator_family,
        "kleene_normal": bench_kleene_normal_family,
        "monoidal_cat": bench_monoidal_cat_family,
        "closed_cat": bench_closed_cat_family,
        "presheaf": bench_presheaf_family,
        "kan_extension": bench_kan_extension_family,
        "distributor": bench_distributor_family,
        "equivalence_cat": bench_equivalence_cat_family,
        "latin_trade": bench_latin_trade_family,
        "steiner_system": bench_steiner_system_family,
        "inc_structure": bench_inc_structure_family,
        "orthogonal_array": bench_orthogonal_array_family,
        "hadamard_matrix": bench_hadamard_matrix_family,
        "finite_difference": bench_finite_difference_family,
        "norm_subring": bench_norm_subring_family,
        "discriminant_field": bench_discriminant_field_family,
        "decomposition_group": bench_decomposition_group_family,
        "ramification": bench_ramification_family,
        "artin_symbol": bench_artin_symbol_family,
        "class_group_toy": bench_class_group_toy_family,
        "derived_functor": bench_derived_functor_family,
        "ext_compute": bench_ext_compute_family,
        "tor_compute": bench_tor_compute_family,
        "spectral_seq": bench_spectral_seq_family,
        "koszul_homology": bench_koszul_homology_family,
        "mapping_degree": bench_mapping_degree_family,
        "uniform_integrability": bench_uniform_integrability_family,
        "vitali_conv": bench_vitali_conv_family,
        "ldp_theory": bench_ldp_theory_family,
        "concentration_ineq": bench_concentration_ineq_family,
        "kolmogorov_01": bench_kolmogorov_01_family,
        "prokhorov_metric": bench_prokhorov_metric_family,
        "sequent_calculus": bench_sequent_calculus_family,
        "natural_ded": bench_natural_ded_family,
        "godel_incomp": bench_godel_incomp_family,
        "interp_proof": bench_interp_proof_family,
        "proof_complexity": bench_proof_complexity_family,
        "modal_completeness": bench_modal_completeness_family,
        "hilbert_samuel": bench_hilbert_samuel_family,
        "krull_dim": bench_krull_dim_family,
        "noether_normal": bench_noether_normal_family,
        "primary_decomp": bench_primary_decomp_family,
        "completion_ring": bench_completion_ring_family,
        "dimension_fiber": bench_dimension_fiber_family,
        "hall_subgroup": bench_hall_subgroup_family,
        "transfer_hom": bench_transfer_hom_family,
        "schur_multiplier": bench_schur_multiplier_family,
        "aut_group": bench_aut_group_family,
        "composition_series": bench_composition_series_family,
        "permutation_poly": bench_permutation_poly_family,
        "mapping_cone": bench_mapping_cone_family,
        "loop_space": bench_loop_space_family,
        "em_space": bench_em_space_family,
        "co_homology": bench_co_homology_family,
        "stiefel_whitney": bench_stiefel_whitney_family,
        "transfer": bench_transfer_family,
        "stone_duality": bench_stone_duality_family,
        "saturation_test": bench_saturation_test_family,
        "omitting_types": bench_omitting_types_family,
        "indiscernibles": bench_indiscernibles_family,
        "stability_spec": bench_stability_spec_family,
        "back_forth": bench_back_forth_family,
        "grothendieck_grp": bench_grothendieck_grp_family,
        "chow_ring": bench_chow_ring_family,
        "gysin": bench_gysin_family,
        "toric_variety": bench_toric_variety_family,
        "proj_morph": bench_proj_morph_family,
        "ample_test": bench_ample_test_family,
        "baire_space": bench_baire_space_family,
        "polish_topology": bench_polish_topology_family,
        "borel_functions": bench_borel_functions_family,
        "souslin_op": bench_souslin_op_family,
        "determinacy_toy": bench_determinacy_toy_family,
        "perfect_set_prop": bench_perfect_set_prop_family,
        "matroid_axioms": bench_matroid_axioms_family,
        "greedy_matroid": bench_greedy_matroid_family,
        "matroid_intersect": bench_matroid_intersect_family,
        "dual_matroid": bench_dual_matroid_family,
        "matroid_union": bench_matroid_union_family,
        "represented_matroid": bench_represented_matroid_family,
        "forcing_poset": bench_forcing_poset_family,
        "dense_filter": bench_dense_filter_family,
        "names_eval": bench_names_eval_family,
        "cohen_adds": bench_cohen_adds_family,
        "ma_toy": bench_ma_toy_family,
        "large_cardinal": bench_large_cardinal_family,
        "operad_assoc": bench_operad_assoc_family,
        "operad_comm": bench_operad_comm_family,
        "little_discs": bench_little_discs_family,
        "operad_tree": bench_operad_tree_family,
        "endomorphism_op": bench_endomorphism_op_family,
        "may_recognition": bench_may_recognition_family,
        "fibration": bench_fibration_family,
        "cofibration": bench_cofibration_family,
        "serre_ss": bench_serre_ss_family,
        "whitehead": bench_whitehead_family,
        "suspension": bench_suspension_family,
        "spectra": bench_spectra_family,
        "riemann_roch": bench_riemann_roch_family,
        "sheaf_cohomology": bench_sheaf_cohomology_family,
        "scheme_local": bench_scheme_local_family,
        "blowup": bench_blowup_family,
        "elliptic_group": bench_elliptic_group_family,
        "moduli_stable": bench_moduli_stable_family,
        "quantifier_elim": bench_quantifier_elim_family,
        "realize_types": bench_realize_types_family,
        "omega_categoricity": bench_omega_categoricity_family,
        "acl_closure": bench_acl_closure_family,
        "morley_rank": bench_morley_rank_family,
        "vocab_interp": bench_vocab_interp_family,
        "bundle_method": bench_bundle_method_family,
        "sqp": bench_sqp_family,
        "ip_qp": bench_ip_qp_family,
        "trust_region": bench_trust_region_family,
        "frank_wolfe2": bench_frank_wolfe2_family,
        "bfgs_wolfe": bench_bfgs_wolfe_family,
        "ito_lemma": bench_ito_lemma_family,
        "girsanov": bench_girsanov_family,
        "sde_strong": bench_sde_strong_family,
        "local_time": bench_local_time_family,
        "quadratic_var": bench_quadratic_var_family,
        "malliavin": bench_malliavin_family,
        "morse_theory": bench_morse_theory_family,
        "transversality": bench_transversality_family,
        "regular_value": bench_regular_value_family,
        "degree_mod2": bench_degree_mod2_family,
        "handle_decomp": bench_handle_decomp_family,
        "poincare_hopf": bench_poincare_hopf_family,
        "tutte_berge": bench_tutte_berge_family,
        "dirac_ore": bench_dirac_ore_family,
        "turan_theorem": bench_turan_theorem_family,
        "planar_five": bench_planar_five_family,
        "graph_minor": bench_graph_minor_family,
        "ramsey_num": bench_ramsey_num_family,
        "broyden": bench_broyden_family,
        "cheb_approx": bench_cheb_approx_family,
        "brent_root": bench_brent_root_family,
        "romberg": bench_romberg_family,
        "aitken_delta": bench_aitken_delta_family,
        "collocation_ode": bench_collocation_ode_family,
        "unification_fol": bench_unification_fol_family,
        "skolem_normal": bench_skolem_normal_family,
        "herbrand_model": bench_herbrand_model_family,
        "presburger": bench_presburger_family,
        "los_theorem": bench_los_theorem_family,
        "finite_field": bench_finite_field_family,
        "galois_corresp": bench_galois_corresp_family,
        "normality_check": bench_normality_check_family,
        "separable_check": bench_separable_check_family,
        "cyclotomic_poly": bench_cyclotomic_poly_family,
        "primitive_elem": bench_primitive_elem_family,
        "singular_homology": bench_singular_homology_family,
        "cw_complex": bench_cw_complex_family,
        "spectral_seq_toy": bench_spectral_seq_toy_family,
        "homotopy_group": bench_homotopy_group_family,
        "excision": bench_excision_family,
        "poincare_dual": bench_poincare_dual_family,
        "optional_stopping": bench_optional_stopping_family,
        "doob_decomp": bench_doob_decomp_family,
        "martingale_clt": bench_martingale_clt_family,
        "azuma": bench_azuma_family,
        "coupling_arg": bench_coupling_arg_family,
        "ergodic_thm": bench_ergodic_thm_family,
        "hahn_banach": bench_hahn_banach_family,
        "riesz_repr": bench_riesz_repr_family,
        "adjoint_op": bench_adjoint_op_family,
        "selfadjoint_spectrum": bench_selfadjoint_spectrum_family,
        "compact_resolvent": bench_compact_resolvent_family,
        "projection_thm": bench_projection_thm_family,
        "cantor_set": bench_cantor_set_family,
        "baire_category": bench_baire_category_family,
        "vitali_set": bench_vitali_set_family,
        "egorov_thm": bench_egorov_thm_family,
        "fatou_lemma": bench_fatou_lemma_family,
        "monotone_conv": bench_monotone_conv_family,
        "cauchy_integral": bench_cauchy_integral_family,
        "residue_calc": bench_residue_calc_family,
        "laurent_series": bench_laurent_series_family,
        "argument_principle": bench_argument_principle_family,
        "conformal_map": bench_conformal_map_family,
        "liouville": bench_liouville_family,
        "homotopy_pi1": bench_homotopy_pi1_family,
        "simplicial_homology": bench_simplicial_homology_family,
        "chain_homotopy": bench_chain_homotopy_family,
        "euler_homology": bench_euler_homology_family,
        "degree_map": bench_degree_map_family,
        "covering_lift": bench_covering_lift_family,
        "energy_method": bench_energy_method_family,
        "maximum_principle": bench_maximum_principle_family,
        "heat_kernel": bench_heat_kernel_family,
        "wave_dalembert": bench_wave_dalembert_family,
        "weak_solution": bench_weak_solution_family,
        "fundamental_laplace": bench_fundamental_laplace_family,
        "plancherel": bench_plancherel_family,
        "poisson_summation": bench_poisson_summation_family,
        "fejer_kernel": bench_fejer_kernel_family,
        "uncertainty": bench_uncertainty_family,
        "fourier_multiplier": bench_fourier_multiplier_family,
        "sobolev_embed": bench_sobolev_embed_family,
        "open_mapping": bench_open_mapping_family,
        "uniform_bounded": bench_uniform_bounded_family,
        "weak_convergence": bench_weak_convergence_family,
        "banach_alaoglu": bench_banach_alaoglu_family,
        "reflexive_space": bench_reflexive_space_family,
        "closed_graph": bench_closed_graph_family,
        "connection_form": bench_connection_form_family,
        "parallel_transport": bench_parallel_transport_family,
        "holonomy": bench_holonomy_family,
        "gauss_bonnet": bench_gauss_bonnet_family,
        "geodesic_eq": bench_geodesic_eq_family,
        "sectional_curv": bench_sectional_curv_family,
        "markov_chain": bench_markov_chain_family,
        "martingale_check": bench_martingale_check_family,
        "poisson_process": bench_poisson_process_family,
        "gambler_ruin": bench_gambler_ruin_family,
        "stopping_time": bench_stopping_time_family,
        "markov_hitting": bench_markov_hitting_family,
        "kolmogorov_axioms": bench_kolmogorov_axioms_family,
        "conditional_expect": bench_conditional_expect_family,
        "markov_ineq": bench_markov_ineq_family,
        "conv_sum": bench_conv_sum_family,
        "moment_generating": bench_moment_generating_family,
        "stochastic_order": bench_stochastic_order_family,
        "sheaf_gluing": bench_sheaf_gluing_family,
        "local_ring_zn": bench_local_ring_zn_family,
        "dedekind_check": bench_dedekind_check_family,
        "divisor_group": bench_divisor_group_family,
        "genus_riemann": bench_genus_riemann_family,
        "moduli_naive": bench_moduli_naive_family,
        "picard_lindelof": bench_picard_lindelof_family,
        "gronwall_lemma": bench_gronwall_lemma_family,
        "sturm_liouville": bench_sturm_liouville_family,
        "phase_plane": bench_phase_plane_family,
        "lyapunov_stability": bench_lyapunov_stability_family,
        "variation_params": bench_variation_params_family,
        "cartan_matrix": bench_cartan_matrix_family,
        "weyl_group_a2": bench_weyl_group_a2_family,
        "killing_form": bench_killing_form_family,
        "root_lattice_a2": bench_root_lattice_a2_family,
        "sl2_structure": bench_sl2_structure_family,
        "su2_algebra": bench_su2_algebra_family,
        "banach_fixed": bench_banach_fixed_family,
        "spectral_theorem": bench_spectral_theorem_family,
        "lp_duality": bench_lp_duality_family,
        "fourier_finite": bench_fourier_finite_family,
        "compact_operator": bench_compact_operator_family,
        "gram_schmidt": bench_gram_schmidt_family,
        "character_table_s3": bench_character_table_s3_family,
        "perm_rep": bench_perm_rep_family,
        "schur_ortho": bench_schur_ortho_family,
        "induced_rep": bench_induced_rep_family,
        "fourier_sn": bench_fourier_sn_family,
        "regular_rep": bench_regular_rep_family,
        "zariski_topo": bench_zariski_topo_family,
        "projective_plane": bench_projective_plane_family,
        "bezout_bezout": bench_bezout_bezout_family,
        "variety_dim": bench_variety_dim_family,
        "monomial_ideal": bench_monomial_ideal_family,
        "hilbert_poly": bench_hilbert_poly_family,
        "graph_coloring": bench_graph_coloring_family,
        "euler_trail": bench_euler_trail_family,
        "matroid_greedy": bench_matroid_greedy_family,
        "planar_check": bench_planar_check_family,
        "poset_dimension": bench_poset_dimension_family,
        "ramsey_r33": bench_ramsey_r33_family,
        "compact_space": bench_compact_space_family,
        "connected_space": bench_connected_space_family,
        "quotient_topology": bench_quotient_topology_family,
        "product_topology": bench_product_topology_family,
        "convergence_space": bench_convergence_space_family,
        "tietze_urysohn": bench_tietze_urysohn_family,
        "quadratic_recip": bench_quadratic_recip_family,
        "elliptic_curve": bench_elliptic_curve_family,
        "p_adic_val": bench_p_adic_val_family,
        "cohomology_cup": bench_cohomology_cup_family,
        "koszul_complex": bench_koszul_complex_family,
        "mayer_vietoris": bench_mayer_vietoris_family,
        "ring_ideals": bench_ring_ideals_family,
        "quotient_ring": bench_quotient_ring_family,
        "pid_check": bench_pid_check_family,
        "minimal_poly": bench_minimal_poly_family,
        "norm_trace": bench_norm_trace_family,
        "spec_ring": bench_spec_ring_family,
        "sylow_theorems": bench_sylow_theorems_family,
        "group_presentation": bench_group_presentation_family,
        "burnside_lemma": bench_burnside_lemma_family,
        "free_group": bench_free_group_family,
        "conjugacy_classes": bench_conjugacy_classes_family,
        "cayley_graph": bench_cayley_graph_family,
        "lattice_check": bench_lattice_check_family,
        "galois_connection": bench_galois_connection_family,
        "tarski_fixed": bench_tarski_fixed_family,
        "boolean_algebra": bench_boolean_algebra_family,
        "congruence_lattice": bench_congruence_lattice_family,
        "term_algebra": bench_term_algebra_family,
        "borel_hierarchy": bench_borel_hierarchy_family,
        "analytic_sets": bench_analytic_sets_family,
        "forcing_lite": bench_forcing_lite_family,
        "arith_hierarchy": bench_arith_hierarchy_family,
        "jump_operator": bench_jump_operator_family,
        "rice_theorem": bench_rice_theorem_family,
        "kripke_semantics": bench_kripke_semantics_family,
        "bisimulation": bench_bisimulation_family,
        "ef_game": bench_ef_game_family,
        "fundamental_group": bench_fundamental_group_family,
        "covering_space": bench_covering_space_family,
        "topo_separation": bench_topo_separation_family,
        "chain_complex": bench_chain_complex_family,
        "tor_ext": bench_tor_ext_family,
        "sheaf_check": bench_sheaf_check_family,
        "hilbert_series": bench_hilbert_series_family,
        "snake_lemma": bench_snake_lemma_family,
        "variety_morph": bench_variety_morph_family,
        "field_ext": bench_field_ext_family,
        "galois_group": bench_galois_group_family,
        "splitting_field": bench_splitting_field_family,
        "lie_bracket": bench_lie_bracket_family,
        "rep_theory": bench_rep_theory_family,
        "root_system": bench_root_system_family,
        "ordinal_arith": bench_ordinal_arith_family,
        "cardinal_arith": bench_cardinal_arith_family,
        "transfinite_induct": bench_transfinite_induct_family,
        "well_founded": bench_well_founded_family,
        "v_omega": bench_v_omega_family,
        "ac_choice": bench_ac_choice_family,
        "ski_combinator": bench_ski_combinator_family,
        "de_bruijn": bench_de_bruijn_family,
        "church_encoding": bench_church_encoding_family,
        "lambda_typing": bench_lambda_typing_family,
        "unification": bench_unification_family,
        "knuth_bendix": bench_knuth_bendix_family,
        "pr_functions": bench_pr_functions_family,
        "turing_degrees": bench_turing_degrees_family,
        "busy_beaver": bench_busy_beaver_family,
        "ultraproduct": bench_ultraproduct_family,
        "ramsey_theory": bench_ramsey_theory_family,
        "compactness_lite": bench_compactness_lite_family,
        "fin_limit": bench_fin_limit_family,
        "subobject_classifier": bench_subobject_classifier_family,
        "exponential_obj": bench_exponential_obj_family,
        "yoneda_embed": bench_yoneda_embed_family,
        "adjoint_check": bench_adjoint_check_family,
        "cat_colimit": bench_cat_colimit_family,
        "garbled_circuit": bench_garbled_circuit_family,
        "bgw_mpc": bench_bgw_mpc_family,
        "beaver_triple": bench_beaver_triple_family,
        "ot_extension": bench_ot_extension_family,
        "spdz_mac": bench_spdz_mac_family,
        "psi_intersect": bench_psi_intersect_family,
        "poly_factor_fp": bench_poly_factor_fp_family,
        "hensel_lift": bench_hensel_lift_family,
        "poly_crt": bench_poly_crt_family,
        "subresultant": bench_subresultant_family,
        "sparse_interp": bench_sparse_interp_family,
        "poly_eval_interp": bench_poly_eval_interp_family,
        "density_matrix": bench_density_matrix_family,
        "povm_measure": bench_povm_measure_family,
        "qchannel": bench_qchannel_family,
        "entanglement": bench_entanglement_family,
        "bell_ineq": bench_bell_ineq_family,
        "state_tomo": bench_state_tomo_family,
        "np_reduce": bench_np_reduce_family,
        "fpras_dnf": bench_fpras_dnf_family,
        "sumcheck": bench_sumcheck_family,
        "param_fpt": bench_param_fpt_family,
        "pcp_verify": bench_pcp_verify_family,
        "circuit_lb": bench_circuit_lb_family,
        "diff_logic": bench_diff_logic_family,
        "array_theory": bench_array_theory_family,
        "bv_ops": bench_bv_ops_family,
        "dpllt": bench_dpllt_family,
        "lia_branch": bench_lia_branch_family,
        "mcsat_lite": bench_mcsat_lite_family,
        "nd_check": bench_nd_check_family,
        "sequent_prove": bench_sequent_prove_family,
        "cut_elim": bench_cut_elim_family,
        "resolution_fol": bench_resolution_fol_family,
        "linear_logic": bench_linear_logic_family,
        "intuit_class": bench_intuit_class_family,
        "path_types": bench_path_types_family,
        "hlevel_check": bench_hlevel_check_family,
        "univalence_toy": bench_univalence_toy_family,
        "kan_hcomp": bench_kan_hcomp_family,
        "funext_toy": bench_funext_toy_family,
        "hit_quotient": bench_hit_quotient_family,
        "r1cs_check": bench_r1cs_check_family,
        "qap_encode": bench_qap_encode_family,
        "kzg_commit": bench_kzg_commit_family,
        "bulletproof_ip": bench_bulletproof_ip_family,
        "plonkish_gate": bench_plonkish_gate_family,
        "snark_circuit": bench_snark_circuit_family,
        "timed_automata": bench_timed_automata_family,
        "parity_game": bench_parity_game_family,
        "nba_emptiness": bench_nba_emptiness_family,
        "ctl_mc": bench_ctl_mc_family,
        "bisim_refine": bench_bisim_refine_family,
        "wsts_cover": bench_wsts_cover_family,
        "borrow_check": bench_borrow_check_family,
        "lifetime_outlives": bench_lifetime_outlives_family,
        "linear_use": bench_linear_use_family,
        "escape_region": bench_escape_region_family,
        "capability_perm": bench_capability_perm_family,
        "refinement_liquid": bench_refinement_liquid_family,
        "fourier_motzkin": bench_fourier_motzkin_family,
        "banerjee_dep": bench_banerjee_dep_family,
        "pluto_schedule": bench_pluto_schedule_family,
        "tiling_legality": bench_tiling_legality_family,
        "omega_test": bench_omega_test_family,
        "vec_legality": bench_vec_legality_family,
        "free_monad": bench_free_monad_family,
        "alg_effects": bench_alg_effects_family,
        "shift_reset": bench_shift_reset_family,
        "row_types": bench_row_types_family,
        "session_types": bench_session_types_family,
        "gradual_types": bench_gradual_types_family,
        "weakest_precond": bench_weakest_precond_family,
        "sygus_synth": bench_sygus_synth_family,
        "horn_clauses": bench_horn_clauses_family,
        "interpolant_mc": bench_interpolant_mc_family,
        "predicate_abs": bench_predicate_abs_family,
        "cegis_loop": bench_cegis_loop_family,
        "three_valued_logic": bench_three_valued_logic_family,
        "shape_graph": bench_shape_graph_family,
        "separation_logic": bench_separation_logic_family,
        "context_pta": bench_context_pta_family,
        "recency_abstraction": bench_recency_abstraction_family,
        "interproc_summary": bench_interproc_summary_family,
        "interval_analysis": bench_interval_analysis_family,
        "sign_domain": bench_sign_domain_family,
        "zone_dbm": bench_zone_dbm_family,
        "affine_karr": bench_affine_karr_family,
        "chaotic_widen": bench_chaotic_widen_family,
        "andersen_pta": bench_andersen_pta_family,
        "congruence_closure": bench_congruence_closure_family,
        "ring_normalize": bench_ring_normalize_family,
        "omega_lia": bench_omega_lia_family,
        "nelson_oppen": bench_nelson_oppen_family,
        "term_rewrite": bench_term_rewrite_family,
        "tseitin_cnf": bench_tseitin_cnf_family,
        "bidirectional_tc": bench_bidirectional_tc_family,
        "nbe_eval": bench_nbe_eval_family,
        "dep_types": bench_dep_types_family,
        "unify_meta": bench_unify_meta_family,
        "proof_kernel": bench_proof_kernel_family,
        "tactic_engine": bench_tactic_engine_family,
        "canny_edge": bench_canny_edge_family,
        "otsu_threshold": bench_otsu_threshold_family,
        "watershed_seg": bench_watershed_seg_family,
        "slic_superpixels": bench_slic_superpixels_family,
        "nlm_denoise": bench_nlm_denoise_family,
        "distance_transform": bench_distance_transform_family,
        "trotter_suzuki": bench_trotter_suzuki_family,
        "qdrift": bench_qdrift_family,
        "shadow_tomography": bench_shadow_tomography_family,
        "vqd_states": bench_vqd_states_family,
        "adapt_vqe": bench_adapt_vqe_family,
        "hhl_lite": bench_hhl_lite_family,
        "rmpflow": bench_rmpflow_family,
        "ds_motion": bench_ds_motion_family,
        "wbc_qp": bench_wbc_qp_family,
        "grasp_epsilon": bench_grasp_epsilon_family,
        "rrt_connect": bench_rrt_connect_family,
        "dmp_control": bench_dmp_control_family,
        "nurbs_eval": bench_nurbs_eval_family,
        "catmull_clark": bench_catmull_clark_family,
        "loop_subdiv": bench_loop_subdiv_family,
        "marching_cubes": bench_marching_cubes_family,
        "half_edge": bench_half_edge_family,
        "laplacian_smooth": bench_laplacian_smooth_family,
        "v_cycle": bench_v_cycle_family,
        "amg_lite": bench_amg_lite_family,
        "bicgstab": bench_bicgstab_family,
        "minres": bench_minres_family,
        "chebyshev_iter": bench_chebyshev_iter_family,
        "ilu_precond": bench_ilu_precond_family,
        "hlc_clock": bench_hlc_clock_family,
        "delta_crdt": bench_delta_crdt_family,
        "raft_log": bench_raft_log_family,
        "bracha_bcast": bench_bracha_bcast_family,
        "tot_order": bench_tot_order_family,
        "quorum_weighted": bench_quorum_weighted_family,
        "abd_register": bench_abd_register_family,
        "fm_partition": bench_fm_partition_family,
        "lee_router": bench_lee_router_family,
        "clock_tree": bench_clock_tree_family,
        "aig_rewrite": bench_aig_rewrite_family,
        "power_est": bench_power_est_family,
        "floorplan_sa": bench_floorplan_sa_family,
        "gottesman_knill": bench_gottesman_knill_family,
        "steane_code": bench_steane_code_family,
        "surface_code": bench_surface_code_family,
        "shor_code": bench_shor_code_family,
        "syndrome_circuit": bench_syndrome_circuit_family,
        "repetition_qec": bench_repetition_qec_family,
        "mceliece_lite": bench_mceliece_lite_family,
        "bike_lite": bench_bike_lite_family,
        "hqc_lite": bench_hqc_lite_family,
        "uov_sig": bench_uov_sig_family,
        "rainbow_sig": bench_rainbow_sig_family,
        "sidh_lite": bench_sidh_lite_family,
        "reflectivity_synth": bench_reflectivity_synth_family,
        "gassmann_sub": bench_gassmann_sub_family,
        "spectral_decomp": bench_spectral_decomp_family,
        "semblance_scan": bench_semblance_scan_family,
        "gardner_relation": bench_gardner_relation_family,
        "vz_raytrace": bench_vz_raytrace_family,
        "pike_vm": bench_pike_vm_family,
        "lazy_dfa": bench_lazy_dfa_family,
        "bitap_fuzzy": bench_bitap_fuzzy_family,
        "literal_prefilter": bench_literal_prefilter_family,
        "glushkov_nfa": bench_glushkov_nfa_family,
        "regex_simplify": bench_regex_simplify_family,
        "laplace_iod": bench_laplace_iod_family,
        "cowell_j2": bench_cowell_j2_family,
        "batch_od": bench_batch_od_family,
        "cr3bp_dynamics": bench_cr3bp_dynamics_family,
        "porkchop_grid": bench_porkchop_grid_family,
        "davenport_q": bench_davenport_q_family,
        "suffix_array_lcp": bench_suffix_array_lcp_family,
        "z_function": bench_z_function_family,
        "suffix_tree_lex": bench_suffix_tree_lex_family,
        "booth_rotation": bench_booth_rotation_family,
        "lyndon_factor": bench_lyndon_factor_family,
        "palindromic_tree": bench_palindromic_tree_family,
        "harris_corner": bench_harris_corner_family,
        "hough_lines": bench_hough_lines_family,
        "integral_image": bench_integral_image_family,
        "seam_carving": bench_seam_carving_family,
        "grabcut_lite": bench_grabcut_lite_family,
        "meanshift_track": bench_meanshift_track_family,
        "mulaw_compand": bench_mulaw_compand_family,
        "adpcm_ima": bench_adpcm_ima_family,
        "lpc_analysis": bench_lpc_analysis_family,
        "celp_encode": bench_celp_encode_family,
        "mel_cepstrum": bench_mel_cepstrum_family,
        "viterbi_vad": bench_viterbi_vad_family,
        "card_table_gc": bench_card_table_gc_family,
        "escape_analysis": bench_escape_analysis_family,
        "osr_deopt": bench_osr_deopt_family,
        "trace_tree": bench_trace_tree_family,
        "ssa_repair": bench_ssa_repair_family,
        "gvn_pre": bench_gvn_pre_family,
        "nmo_dix": bench_nmo_dix_family,
        "taup_transform": bench_taup_transform_family,
        "kirchhoff_mig": bench_kirchhoff_mig_family,
        "avo_shuey": bench_avo_shuey_family,
        "vibroseis_sweep": bench_vibroseis_sweep_family,
        "eikonal_fmm": bench_eikonal_fmm_family,
        "equinox_prec": bench_equinox_prec_family,
        "nutation_lite": bench_nutation_lite_family,
        "rise_set": bench_rise_set_family,
        "eclipse_circ": bench_eclipse_circ_family,
        "delta_t": bench_delta_t_family,
        "planet_vsop": bench_planet_vsop_family,
        "tablebase_dtm": bench_tablebase_dtm_family,
        "retrograde_wdl": bench_retrograde_wdl_family,
        "rave_mc": bench_rave_mc_family,
        "mast_playout": bench_mast_playout_family,
        "expectimax": bench_expectimax_family,
        "isomcts": bench_isomcts_family,
        "radon_fbp": bench_radon_fbp_family,
        "art_sirt": bench_art_sirt_family,
        "cs_mri": bench_cs_mri_family,
        "hu_moments": bench_hu_moments_family,
        "chan_vese": bench_chan_vese_family,
        "mi_register": bench_mi_register_family,
        "ntt_ring": bench_ntt_ring_family,
        "kyber_kem": bench_kyber_kem_family,
        "dilithium_sig": bench_dilithium_sig_family,
        "frodokem": bench_frodokem_family,
        "xmss_sig": bench_xmss_sig_family,
        "sphincs_sig": bench_sphincs_sig_family,
        "lqr_funnel": bench_lqr_funnel_family,
        "chomp": bench_chomp_family,
        "gjk_epa": bench_gjk_epa_family,
        "ilqr": bench_ilqr_family,
        "rts_smoother": bench_rts_smoother_family,
        "se3_spline": bench_se3_spline_family,
        "orbital_elements": bench_orbital_elements_family,
        "kepler_solve": bench_kepler_solve_family,
        "lambert_problem": bench_lambert_problem_family,
        "tle_propagate": bench_tle_propagate_family,
        "orbit_maneuver": bench_orbit_maneuver_family,
        "gauss_iod": bench_gauss_iod_family,
        "tree_cover": bench_tree_cover_family,
        "modulo_sched": bench_modulo_sched_family,
        "jump_thread": bench_jump_thread_family,
        "tail_dup": bench_tail_dup_family,
        "cfg_simplify": bench_cfg_simplify_family,
        "bb_reorder": bench_bb_reorder_family,
        "deferred_shade": bench_deferred_shade_family,
        "sdf_raymarch": bench_sdf_raymarch_family,
        "frustum_cull": bench_frustum_cull_family,
        "lod_select": bench_lod_select_family,
        "env_map": bench_env_map_family,
        "shadow_pcf": bench_shadow_pcf_family,
        "quic_streams": bench_quic_streams_family,
        "tls13_trans": bench_tls13_trans_family,
        "qpack_pack": bench_qpack_pack_family,
        "wg_ik": bench_wg_ik_family,
        "doh_wire": bench_doh_wire_family,
        "sctp_tsn": bench_sctp_tsn_family,
        "netlist_parse": bench_netlist_parse_family,
        "sta_timing": bench_sta_timing_family,
        "a_star_route": bench_a_star_route_family,
        "drc_check": bench_drc_check_family,
        "place_quadratic": bench_place_quadratic_family,
        "levelize": bench_levelize_family,
        "smiles_parse": bench_smiles_parse_family,
        "morgan_fp": bench_morgan_fp_family,
        "tanimoto": bench_tanimoto_family,
        "mol_descriptors": bench_mol_descriptors_family,
        "substruct": bench_substruct_family,
        "ring_detect": bench_ring_detect_family,
        "fin_cat": bench_fin_cat_family,
        "functor_check": bench_functor_check_family,
        "nat_trans": bench_nat_trans_family,
        "adjunction": bench_adjunction_family,
        "limit_prod": bench_limit_prod_family,
        "monad_laws": bench_monad_laws_family,
        "leb_measure": bench_leb_measure_family,
        "leb_integral": bench_leb_integral_family,
        "conv_prob": bench_conv_prob_family,
        "weak_conv": bench_weak_conv_family,
        "fubini_swap": bench_fubini_swap_family,
        "radon_nikodym": bench_radon_nikodym_family,
        "first_ff": bench_first_ff_family,
        "gauss_curve": bench_gauss_curve_family,
        "frenet_frame": bench_frenet_frame_family,
        "christoffel": bench_christoffel_family,
        "geodesic_sphere": bench_geodesic_sphere_family,
        "surf_area": bench_surf_area_family,
        "beacon_detect": bench_beacon_detect_family,
        "entropy_dns": bench_entropy_dns_family,
        "cred_stuffing": bench_cred_stuffing_family,
        "impossible_travel": bench_impossible_travel_family,
        "exfil_zscore": bench_exfil_zscore_family,
        "sig_score": bench_sig_score_family,
        "markov_entropy": bench_markov_entropy_family,
        "blahut_arimoto": bench_blahut_arimoto_family,
        "kl_knn": bench_kl_knn_family,
        "type_class": bench_type_class_family,
        "elias_gamma": bench_elias_gamma_family,
        "miller_madow": bench_miller_madow_family,
        "nj_tree": bench_nj_tree_family,
        "fitch_pars": bench_fitch_pars_family,
        "seed_extend": bench_seed_extend_family,
        "band_align": bench_band_align_family,
        "jc69_lik": bench_jc69_lik_family,
        "codon_usage": bench_codon_usage_family,
        "fk_dh": bench_fk_dh_family,
        "ik_jac": bench_ik_jac_family,
        "ray_lidar": bench_ray_lidar_family,
        "pot_field": bench_pot_field_family,
        "bezier_curve": bench_bezier_curve_family,
        "odom_comp": bench_odom_comp_family,
    }
    config_hash = hash_bytes(json.dumps(config.dump(), sort_keys=True, default=str).encode("utf-8"))
    data_hash = hash_bytes(json.dumps(scope, sort_keys=True).encode("utf-8"))
    # Shape-only provenance is insufficient: two datasets can share the same
    # dimensions while containing different observations.  Polars' row hash
    # is deterministic for the materialized frame and keeps the receipt
    # independent of filesystem paths and cache locations.
    content_hash = canonical_frame_fingerprint(frame)
    northset_inputs: dict[str, Any] = {}
    if northset_frame is not None:
        northset_inputs["silver_rows"] = int(northset_frame.height)
        northset_inputs["silver_columns"] = sorted(northset_frame.columns)
        northset_inputs["silver_content_sha256"] = canonical_frame_fingerprint(northset_frame)
    book_path = config.northset.book_panel_path
    if book_path:
        target = Path(book_path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Northset book panel not found: {target}")
        northset_inputs["book_panel_path"] = str(target)
        northset_inputs["book_panel_sha256"] = hash_file(target)
        northset_inputs["book_panel_bytes"] = int(target.stat().st_size)
    northset_hash = hash_bytes(
        json.dumps(northset_inputs, sort_keys=True, default=str).encode("utf-8")
    )
    runtime_packages: dict[str, str] = {}
    for package in ("dipcatcher", "numpy", "polars", "scikit-learn", "scipy", "cvxpy"):
        try:
            runtime_packages[package] = version(package)
        except PackageNotFoundError:
            runtime_packages[package] = "UNAVAILABLE"
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": runtime_packages,
    }
    return {
        "run_id": hash_bytes(
            (
                f"{revision}:{worktree_hash}:{config_hash}:{data_hash}:"
                f"{content_hash}:{northset_hash}"
            ).encode()
        ),
        "git_revision": revision,
        "git_worktree_sha256": worktree_hash,
        "config_sha256": config_hash,
        "dataset_sha256": data_hash,
        "dataset_content_sha256": content_hash,
        "northset_inputs_sha256": northset_hash,
        "northset_inputs": northset_inputs,
        "row_count": frame.height,
        "column_count": len(columns),
        "label": label,
        "seed": config.train.random_seed,
        "source": str(config.data.source),
        "point_in_time": True,
        "execution_claim": "research_only",
        "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
        "runtime": runtime,
    }


def _benchmark_scorecard(families: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Classify benchmark evidence without turning diagnostics into passes."""

    scorecard: dict[str, dict[str, Any]] = {}
    for name, payload in families.items():
        entry: dict[str, Any] = {
            "executed": family_blob_executed(payload),
            "nonempty": family_blob_nonempty(payload),
            "finite_observation": family_blob_has_finite_observation(payload),
            "forbidden_metrics_absent": family_blob_forbidden_metrics_absent(payload),
            "claim": "research_metric_only",
        }
        # Soft research diagnostic (Day Wave 18/20/21): never a live promotion gate.
        if name == "tail":
            entry["tail_var_battery_ok"] = tail_var_battery_keys_present(payload)
            entry["tail_es_battery_ok"] = tail_es_battery_keys_present(payload)
        if name == "distribution":
            entry["dist_crps_eprocess_ok"] = dist_crps_eprocess_keys_present(payload)
        scorecard[name] = entry
    return scorecard


def _finite_number(value: Any) -> float | None:
    """Return float(value) when finite; else None (missing / NaN / inf / non-numeric)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _hyp(
    hid: str,
    statement: str,
    test: str,
    stat: float,
    p: float,
    yes: str,
    no: str,
    alpha: float = 0.05,
    *,
    family: str = "discovery",
    meets_floor: bool | None = None,
) -> HypothesisResult:
    # Non-finite p must never claim the fail-to-reject success string (calibration)
    # or a substantive discovery outcome — that polluted FDR / H-table honesty.
    rej = bool(np.isfinite(p) and p < alpha)
    if not np.isfinite(p):
        decision = "Inference unavailable (non-finite p-value)."
    else:
        decision = yes if rej else no
    return HypothesisResult(
        id=hid,
        statement=statement,
        test=test,
        statistic=float(stat) if np.isfinite(stat) else float("nan"),
        p_value=float(p) if np.isfinite(p) else float("nan"),
        reject_raw=rej,
        reject_fdr=False,
        decision=decision,
        family=family,
        meets_floor=meets_floor,
    )


def _series_array(blob: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        raw = blob.get(key)
        if raw is None or isinstance(raw, (str, bytes, dict)):
            continue
        if isinstance(raw, (int, float, np.floating, np.integer, bool)):
            continue
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if int(np.isfinite(arr).sum()) >= 3:
            return arr
    return None


def _first_int(blob: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        v = blob.get(key)
        if v is None or isinstance(v, (list, tuple, dict, np.ndarray)):
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def _contrast_inference(
    blob: dict[str, Any],
    mean_gap: float,
    *,
    series_keys: tuple[str, ...] = (),
    baseline_keys: tuple[str, ...] = (),
    gap_keys: tuple[str, ...] = (),
    n_keys: tuple[str, ...] = ("n_dates", "n"),
    loss_dm: bool = False,
) -> tuple[float, float, str]:
    """Real p-value for a mean gap. Never the dummy map p=0 if better else 1."""
    gap_s = _series_array(blob, *gap_keys) if gap_keys else None
    a = _series_array(blob, *series_keys) if series_keys else None
    b = _series_array(blob, *baseline_keys) if baseline_keys else None
    if gap_s is not None:
        _mu, t, p = mean_tstat(gap_s)
        return float(t), float(p), "HAC t-stat of date-level gap"
    if a is not None and b is not None:
        n_pair = min(int(a.size), int(b.size))
        a, b = a[:n_pair], b[:n_pair]
        if loss_dm:
            dm = diebold_mariano(a, b, name_a="model", name_b="baseline")
            return float(dm.statistic), float(dm.p_value), "Diebold–Mariano on date-level loss"
        _mu, t, p = mean_tstat(a - b)
        return float(t), float(p), "HAC t-stat of paired date-level difference"
    n_rep = _first_int(blob, *n_keys)
    t, p = mean_difference_t(mean_gap, n_rep or 0)
    return t, p, "mean-difference t using reported n_dates, not dummy 0/1"


def _apply_family_fdr(hyps: list[HypothesisResult], family: str, alpha: float = 0.05) -> None:
    idx = [i for i, h in enumerate(hyps) if h.family == family]
    if not idx:
        return
    pvals = np.array([hyps[i].p_value for i in idx], dtype=float)
    finite = np.isfinite(pvals)
    reject = np.zeros(len(idx), dtype=bool)
    if finite.any():
        sub, _ = benjamini_hochberg(pvals[finite], alpha=alpha)
        reject[np.where(finite)[0]] = sub
    for j, r in zip(idx, reject, strict=True):
        hyps[j].reject_fdr = bool(r)


def _ranker_data_snooping(
    rankers: list[dict[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 7,
) -> dict[str, Any] | None:
    """Data-snooping battery over the ranker universe (research diagnostic).

    Aligns the rankers' date-level IC series on the common date intersection and
    runs White's Reality Check, Hansen's SPA, Romano–Wolf StepM and the
    Hansen–Lunde–Nason MCS (larger IC is better). Returns None when fewer than
    two rankers expose >= 10 aligned finite dates. Never a live Sharpe claim.
    """
    from quant_fund.metrics.snooping import (
        model_confidence_set,
        reality_check,
        spa_test,
        stepm,
    )

    usable: list[tuple[str, dict[str, float]]] = []
    for r in rankers:
        name = str(r.get("name", ""))
        if not name or name.startswith("_"):
            continue
        series = r.get("ic_series")
        dates = r.get("ic_dates")
        if not isinstance(series, list) or not isinstance(dates, list):
            continue
        if len(series) != len(dates):
            continue
        by_date: dict[str, float] = {}
        for d, v in zip(dates, series, strict=True):
            fv = float(v)
            if np.isfinite(fv):
                by_date[str(d)] = fv
        if len(by_date) >= 10:
            usable.append((name, by_date))
    if len(usable) < 2:
        return None
    common = set(usable[0][1])
    for _name, by_date in usable[1:]:
        common &= set(by_date)
    if len(common) < 10:
        return None
    order = sorted(common)
    names = [name for name, _ in usable]
    matrix = np.column_stack(
        [np.asarray([by_date[d] for d in order], dtype=float) for _n, by_date in usable]
    )
    rc = reality_check(matrix, n_boot=n_boot, seed=seed)
    spa = spa_test(matrix, n_boot=n_boot, seed=seed)
    st = stepm(matrix, n_boot=n_boot, alpha=0.05, seed=seed)
    mcs = model_confidence_set(matrix, n_boot=n_boot, alpha=0.10, seed=seed)
    best = names[rc.best_index] if rc.best_index >= 0 else None
    return {
        "claim": "research_diagnostic_only",
        "research_only": True,
        "n_trials": len(names),
        "n_obs": rc.n_obs,
        "n_boot": int(n_boot),
        "block": rc.block,
        "best_trial": best,
        "best_mean_ic": rc.best_mean,
        "reality_check_stat": rc.statistic,
        "reality_check_p": rc.p_value,
        "spa_stat": spa.statistic,
        "spa_p_lower": spa.p_lower,
        "spa_p_consistent": spa.p_consistent,
        "spa_p_upper": spa.p_upper,
        "stepm_alpha": st.alpha,
        "stepm_n_rejected": st.n_rejected,
        "stepm_rejected": [names[i] for i, flag in enumerate(st.rejected) if flag],
        "stepm_adjusted_p": {names[i]: st.adjusted_p[i] for i in range(len(names))},
        "mcs_alpha": mcs.alpha,
        "mcs_n_included": mcs.n_included,
        "mcs_included": [names[i] for i, flag in enumerate(mcs.included) if flag],
        "mcs_p_values": {names[i]: mcs.p_values[i] for i in range(len(names))},
    }


def _horizon_bars(label: str) -> int:
    """Trailing integer on a label name (``future_return_5`` → 5)."""
    suffix = str(label).rsplit("_", 1)[-1]
    if suffix.isdigit() and int(suffix) >= 1:
        return int(suffix)
    return 1


def _aligned_ranker_scores(
    rankers: list[dict[str, Any]],
) -> tuple[int, list[str], np.ndarray | None]:
    """Trial count plus the common-date score matrix of evaluated rankers.

    ``n_trials`` counts every non-internal ranker the runner evaluated.
    The matrix contains only rankers with a finite date-level score series
    on the intersection of those dates. Rankers that were evaluated but
    cannot be aligned stay in ``n_trials`` and are absent from the matrix.
    """
    evaluated = [
        ranker
        for ranker in rankers
        if str(ranker.get("name", "")).strip() and not str(ranker.get("name", "")).startswith("_")
    ]
    usable: list[tuple[str, dict[str, float]]] = []
    for ranker in evaluated:
        series = ranker.get("ic_series")
        dates = ranker.get("ic_dates")
        if not isinstance(series, list) or not isinstance(dates, list):
            continue
        if len(series) != len(dates):
            continue
        by_date: dict[str, float] = {}
        for date, value in zip(dates, series, strict=True):
            number = float(value)
            if np.isfinite(number):
                by_date[str(date)] = number
        if by_date:
            usable.append((str(ranker["name"]), by_date))
    if len(usable) < 1:
        return len(evaluated), [], None
    common = set(usable[0][1])
    for _name, by_date in usable[1:]:
        common &= set(by_date)
    if not common:
        return len(evaluated), [], None
    order = sorted(common)
    names = [name for name, _by_date in usable]
    matrix = np.column_stack(
        [np.asarray([by_date[date] for date in order], dtype=float) for _name, by_date in usable]
    )
    return len(evaluated), names, matrix


def _overfitting_section(block: object) -> str:
    """One-line notebook summary of the backtest-overfitting diagnostics."""
    if not isinstance(block, dict) or not block:
        return "unavailable"

    def _fmt(value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "n/a"
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:.4g}"

    return (
        f"PBO={_fmt(block.get('pbo'))} DSR={_fmt(block.get('dsr'))} "
        f"PSR={_fmt(block.get('psr'))} MinTRL={_fmt(block.get('min_trl'))} "
        f"n_trials={block.get('n_trials')} "
        f"n_trials_effective={block.get('n_trials_effective')} "
        f"research_diagnostic_only"
    )


def _data_snooping_section(blob: object) -> str:
    """One-line notebook summary of the ranker data-snooping battery."""
    if not isinstance(blob, dict) or not blob:
        return "unavailable (needs >=2 rankers with >=10 aligned dates)"
    rc_p = float(blob.get("reality_check_p", float("nan")))
    spa_p = float(blob.get("spa_p_consistent", float("nan")))
    return (
        f"n_trials={blob.get('n_trials')} best={blob.get('best_trial')} "
        f"RC p={format_p_value(rc_p)} SPA(cons) p={format_p_value(spa_p)} "
        f"StepM rejected={blob.get('stepm_n_rejected')} "
        f"MCS included={blob.get('mcs_n_included')}"
    )


def _hypotheses_rankers_and_rewards(
    families: dict[str, Any],
    rankers: list[dict[str, Any]],
) -> list[HypothesisResult]:
    """Oracle, snooping, pairwise DM, vol/tail, and reward-policy hypotheses.

    Order matches the historical ``_build_hypotheses`` prefix.
    """
    hyps: list[HypothesisResult] = []
    model_rankers = [r for r in rankers if not str(r.get("name", "")).startswith("_")]
    by = {r["name"]: r for r in model_rankers}
    oracle = by.get("oracle_raw")
    if oracle is not None:
        # Skip each hyp independently when its p is non-finite (same as Kupiec skip).
        p_ic_f = _finite_number(oracle.get("p_ic"))
        if p_ic_f is not None:
            hyps.append(
                _hyp(
                    "H1_ranking_oracle",
                    "Labeled SYNTHETIC oracle has positive date-level ranking IC.",
                    "HAC t-stat of date IC",
                    float(oracle.get("t_ic", float("nan"))),
                    p_ic_f,
                    "Oracle ranking signal recovered.",
                    "Oracle IC not significant.",
                    family="discovery",
                )
            )
        ls_p_f = _finite_number(oracle.get("ls_p"))
        if ls_p_f is not None:
            hyps.append(
                _hyp(
                    "H2_decile_mono",
                    "Oracle decile long-short mean is positive (ranking science).",
                    "HAC t-stat of decile LS",
                    float(oracle.get("ls_t", float("nan"))),
                    ls_p_f,
                    "Deciles are informative.",
                    "Decile spread not distinguishable from 0.",
                    family="discovery",
                )
            )
    # H46: data-snooping over the ranker/feature-set universe (Hansen SPA).
    # Only minted when the ranking family carries a finite consistent p-value.
    ranking_family = families.get("ranking") or {}
    snooping_blob = (
        ranking_family.get("data_snooping") if isinstance(ranking_family, dict) else None
    )
    if isinstance(snooping_blob, dict):
        p_snoop = _finite_number(snooping_blob.get("spa_p_consistent"))
        if p_snoop is not None:
            hyps.append(
                _hyp(
                    "H99_ranking_data_snooping",
                    "Best ranker date-IC survives data-snooping over the full "
                    "ranker/feature-set universe.",
                    "Stationary-bootstrap Hansen SPA (consistent recentering)",
                    float(snooping_blob.get("spa_stat", float("nan"))),
                    p_snoop,
                    "Best ranker IC is significant after snooping.",
                    "Best ranker IC is not significant after snooping.",
                    family="discovery",
                )
            )
    # Pairwise DM among rankers (discovery family)
    dm_summary = next((r for r in rankers if r.get("name") == "_pairwise_dm_summary"), None)
    if dm_summary and dm_summary.get("pairwise_dm_all"):
        for _i, d in enumerate(dm_summary["pairwise_dm_all"]):
            # NaN p_value must skip (is not None would still mint unavailable rows).
            p_dm = _finite_number(d.get("p_value"))
            if p_dm is None:
                continue
            hyps.append(
                _hyp(
                    f"H_rank_dm_{d.get('a')}_vs_{d.get('b')}",
                    f"Diebold–Mariano equal accuracy of -IC loss: {d.get('a')} vs {d.get('b')}.",
                    "Diebold–Mariano pairwise on date-level -IC",
                    float(d.get("statistic", float("nan"))),
                    p_dm,
                    f"DM prefers {d.get('preferred')}.",
                    "Equal ranking forecast accuracy not rejected.",
                    family="discovery",
                )
            )
    vol = families.get("volatility") or {}
    dm_p_f = _finite_number(vol.get("dm_p"))
    if dm_p_f is not None:
        hyps.append(
            _hyp(
                "H3_vol_dm",
                "EWMA and rolling 20d vol forecasts have unequal MSE (Diebold–Mariano).",
                "Diebold–Mariano",
                float(vol.get("dm_stat", float("nan"))),
                dm_p_f,
                f"DM prefers {vol.get('dm_preferred')}.",
                "Equal vol-forecast accuracy not rejected.",
                family="discovery",
            )
        )
    tail = families.get("tail") or {}
    if _finite_number(tail.get("kupiec_p")) is not None:
        hyps.append(
            _hyp(
                "H4_var_kupiec",
                "Vol-scaled historical 95% VaR hit rate matches the 5% nominal (Kupiec POF).",
                "Kupiec POF",
                float(tail.get("kupiec_lr", float("nan"))),
                float(tail.get("kupiec_p", float("nan"))),
                "Hit rate rejects the nominal 5% (miscalibrated or small sample).",
                "VaR hits consistent with 5% nominal.",
                family="calibration",
            )
        )
    # H4b: Christoffersen CC only (nests ind+Kupiec) — skip ind to avoid BH double-count.
    cc_p_f = _finite_number(tail.get("christoffersen_cc_p"))
    if cc_p_f is not None:
        hyps.append(
            _hyp(
                "H4b_var_christoffersen_cc",
                "Vol-scaled historical 95% VaR hits have correct conditional coverage "
                "(Christoffersen CC: independence + unconditional).",
                "Christoffersen CC",
                float(tail.get("christoffersen_cc_lr", float("nan"))),
                cc_p_f,
                "Conditional coverage rejected (hit clustering and/or wrong unconditional rate).",
                "VaR hits consistent with independent 5% conditional coverage.",
                family="calibration",
            )
        )
    dd = families.get("drawdown") or {}
    brier_f = _finite_number(dd.get("brier"))
    brier_base_f = _finite_number(dd.get("brier_base_rate"))
    if brier_f is not None and brier_base_f is not None:
        gap = float(brier_base_f - brier_f)
        t, p_two, test = _contrast_inference(
            dd,
            gap,
            series_keys=("brier_by_date", "brier_clf_by_date"),
            baseline_keys=("brier_base_by_date", "brier_base_rate_by_date"),
            gap_keys=("brier_gap_by_date",),
            n_keys=("n_dates", "n_test", "n"),
            loss_dm=True,
        )
        # DM: mean(clf − base); negative t means lower Brier. Gap HAC: positive t means win.
        greater = "Diebold" not in test
        p = onesided_from_twosided(t, p_two, greater=greater)
        # Skip non-finite contrast p (align discovery DM hygiene; prefer omit over unavailable).
        if _finite_number(p) is not None:
            hyps.append(
                _hyp(
                    "H5_drawdown_brier",
                    "Drawdown classifier Brier beats the unconditional base rate.",
                    test + " (one-sided, lower Brier)",
                    gap,
                    p,
                    "Classifier improves on the base rate.",
                    "Classifier does not beat the base rate.",
                    family="discovery",
                )
            )
    rl = families.get("reinforcement") or {}
    adv_rl = _finite_number(rl.get("mean_advantage_vs_ridge"))
    if adv_rl is not None:
        adv = float(adv_rl)
        t, p_two, test = _contrast_inference(
            rl,
            adv,
            series_keys=("policy_reward",),
            baseline_keys=("ridge_reward",),
            gap_keys=("reward_gap_series",),
        )
        p = onesided_from_twosided(t, p_two, greater=True)
        if _finite_number(p) is not None:
            hyps.append(
                _hyp(
                    "H6_linucb_vs_uniform",
                    "LinUCB top-k reward exceeds a static public ridge top-k policy "
                    "(same public features, same dates).",
                    test + " (one-sided vs ridge)",
                    adv,
                    p,
                    "LinUCB beats static public ridge on the scientific reward.",
                    "LinUCB does not beat static public ridge.",
                    family="discovery",
                )
            )
    return hyps


def _hypotheses_conformal_bounds(families: dict[str, Any]) -> list[HypothesisResult]:
    """Conformal, e-value, jackknife, CRC, and coverage-bound hypotheses."""
    hyps: list[HypothesisResult] = []
    conf = families.get("conformal") or {}
    aci = conf.get("aci") if isinstance(conf, dict) else None
    if isinstance(aci, dict) and _finite_number(aci.get("kupiec_p")) is not None:
        hyps.append(
            _hyp(
                "H7_aci_coverage",
                "ACI 90% prediction-set miss rate wrapping the operational scaled wrappee matches nominal α=0.10 (Kupiec on conformal misses).",
                "Kupiec POF on ACI misses",
                float(aci.get("kupiec_lr", float("nan"))),
                float(aci.get("kupiec_p", float("nan"))),
                "ACI miss rate differs from 10% nominal.",
                "ACI coverage consistent with 90% nominal.",
                family="calibration",
            )
        )
    mond = conf.get("mondrian_aci") if isinstance(conf, dict) else None
    if isinstance(mond, dict) and _finite_number(mond.get("high_x_kupiec_p")) is not None:
        hyps.append(
            _hyp(
                "H8_mondrian_high_vol",
                "Mondrian ACI high-vol (X) slice miss rate matches nominal α=0.10 (not a |Y|-slice guarantee).",
                "Kupiec POF on Mondrian high-X misses",
                float(mond.get("high_x_kupiec_lr", float("nan"))),
                float(mond.get("high_x_kupiec_p", float("nan"))),
                "High-vol Mondrian miss rate differs from 10% nominal.",
                "High-vol Mondrian coverage consistent with 90% nominal.",
                family="calibration",
            )
        )
    ev = families.get("evalues") or {}
    # Skip non-finite e_sup: Python max(nan, 1.0) → 1.0 would mint p=1 success ("consistent").
    e_sup_f = _finite_number(ev.get("e_sup"))
    if e_sup_f is not None:
        e_sup = e_sup_f
        p_ville = float(min(1.0, 1.0 / max(e_sup, 1.0)))
        hyps.append(
            _hyp(
                "H9_eprocess_aci",
                "ACI miss e-process is consistent with nominal α=0.10 (Ville, level 0.05).",
                "Anytime-valid Bernoulli e-process (Ville)",
                e_sup,
                p_ville,
                "E-process crossed 20: too many ACI misses for α=0.10.",
                "E-process stayed below 20; ACI misses consistent with α=0.10.",
                family="calibration",
            )
        )
    jp = families.get("jackknife_plus") or {}
    # Skip non-finite coverage: NaN must not claim "below the 1-2α floor".
    jp_cov_f = _finite_number(jp.get("coverage"))
    if jp_cov_f is not None:
        jp_cov = jp_cov_f
        floor_f = _finite_number(jp.get("coverage_floor"))
        if floor_f is None:
            # Never fabricate the 1−2α floor: the bound row is minted as
            # "not evaluated" (meets_floor=None) when the receipt omits it.
            hyps.append(
                HypothesisResult(
                    id="H10_jackknife_coverage",
                    statement="Jackknife+ coverage meets the 1-2α finite-sample floor.",
                    test="coverage − (1−2α) floor check; not a Kupiec null, not FDR",
                    statistic=float("nan"),
                    p_value=float("nan"),
                    reject_raw=False,
                    reject_fdr=False,
                    decision="Coverage floor unavailable; bound not evaluated.",
                    family="bound",
                    meets_floor=None,
                )
            )
        else:
            floor = floor_f
            meets = bool(jp_cov + 1e-12 >= floor)
            hyps.append(
                HypothesisResult(
                    id="H10_jackknife_coverage",
                    statement="Jackknife+ coverage meets the 1-2α finite-sample floor.",
                    test="coverage − (1−2α) floor check; not a Kupiec null, not FDR",
                    statistic=jp_cov - floor,
                    p_value=float("nan"),
                    reject_raw=False,
                    reject_fdr=False,
                    decision="Jackknife+ coverage is above the 1-2α floor."
                    if meets
                    else "Jackknife+ coverage is below the 1-2α floor.",
                    family="bound",
                    meets_floor=meets,
                )
            )
    crc = families.get("crc") or {}
    if _finite_number(crc.get("kupiec_p")) is not None:
        hyps.append(
            _hyp(
                "H11_crc_var",
                "CRC 95% VaR-style hit rate on the scaled wrappee matches the 5% nominal (Kupiec POF).",
                "Kupiec POF on CRC hits",
                float(crc.get("kupiec_lr", float("nan"))),
                float(crc.get("kupiec_p", float("nan"))),
                "CRC hit rate rejects the nominal 5%.",
                "CRC hits consistent with 5% nominal.",
                family="calibration",
            )
        )
    wcqr = families.get("weighted_conformal") or {}
    if _finite_number(wcqr.get("kupiec_p")) is not None:
        hyps.append(
            _hyp(
                "H12_weighted_cqr",
                "Weighted split CQR miss rate matches nominal α=0.10 (Kupiec POF).",
                "Kupiec POF on weighted CQR misses",
                float(wcqr.get("kupiec_lr", float("nan"))),
                float(wcqr.get("kupiec_p", float("nan"))),
                "Weighted CQR miss rate differs from 10% nominal.",
                "Weighted CQR coverage consistent with 90% nominal.",
                family="calibration",
            )
        )
    cvp = families.get("cv_plus") or {}
    # Skip when coverage or floor non-finite — NaN >= NaN is False → false "below floor".
    cvp_cov = _finite_number(cvp.get("coverage"))
    cvp_floor = _finite_number(cvp.get("coverage_floor"))
    if cvp_cov is not None and cvp_floor is not None:
        meets_cv = bool(cvp_cov >= cvp_floor)
        hyps.append(
            HypothesisResult(
                id="H15_cv_plus_floor",
                statement="CV+ coverage meets its aggregation-specific finite-sample floor.",
                test=f"coverage − {cvp.get('coverage_identity')} floor check; not FDR",
                statistic=cvp_cov - cvp_floor,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision="CV+ coverage is above its stated floor."
                if meets_cv
                else "CV+ coverage is below its stated floor.",
                family="bound",
                meets_floor=meets_cv,
            )
        )
    # H16–H18: panel-only Kupiec coverage tests. Fixture dgp never shares this H-table.
    for hid, key, statement, alpha_key in [
        (
            "H16_localized_cqr",
            "localized_conformal",
            "Localized CQR miss rate matches nominal alpha on the lab panel.",
            "alpha",
        ),
        (
            "H17_online_crc",
            "online_crc",
            "Online CRC hit risk matches nominal alpha on the lab panel.",
            "nominal",
        ),
        (
            "H18_portfolio_conformal",
            "portfolio_conformal",
            "Book-level conformal coverage matches nominal alpha on the lab panel.",
            "alpha",
        ),
    ]:
        b = families.get(key) or {}
        if str(b.get("dgp") or "") == "fixture":
            continue
        # Skip missing OR non-finite kupiec_p (NaN must not mint a "consistent" row).
        if _finite_number(b.get("kupiec_p")) is None:
            continue
        _ = alpha_key  # selects which blob key holds the nominal miss level
        hyps.append(
            _hyp(
                hid,
                statement,
                "Kupiec POF vs nominal alpha (panel)",
                float(b.get("kupiec_lr", np.nan)),
                float(b.get("kupiec_p", np.nan)),
                f"{key} miss rate differs from nominal alpha.",
                f"{key} coverage consistent with nominal alpha.",
                family="calibration",
            )
        )
    topk = families.get("conformal_rank") or {}
    # H19: bound check on reported FDR vs alpha — skip non-finite (never "unavailable" row).
    fdr_f = _finite_number(topk.get("fdr"))
    alpha_f = _finite_number(topk.get("alpha"))
    if str(topk.get("dgp") or "") != "fixture" and fdr_f is not None:
        fdr = fdr_f
        if alpha_f is None:
            # Never assume the nominal level: bound not evaluated without alpha.
            hyps.append(
                HypothesisResult(
                    id="H19_conformal_rank",
                    statement="Conformal top-k date-grouped FDR stays at or below alpha on the lab panel.",
                    test="FDR ≤ alpha bound check; not FDR-discovery",
                    statistic=float("nan"),
                    p_value=float("nan"),
                    reject_raw=False,
                    reject_fdr=False,
                    decision="Nominal alpha unavailable; bound not evaluated.",
                    family="bound",
                    meets_floor=None,
                )
            )
        else:
            a = alpha_f
            meets = bool(fdr <= a + 1e-12)
            hyps.append(
                HypothesisResult(
                    id="H19_conformal_rank",
                    statement="Conformal top-k date-grouped FDR stays at or below alpha on the lab panel.",
                    test="FDR ≤ alpha bound check; not FDR-discovery",
                    statistic=fdr - a,
                    p_value=float("nan"),
                    reject_raw=False,
                    reject_fdr=False,
                    decision=(
                        "Top-k FDR at or below alpha." if meets else "Top-k FDR exceeds alpha."
                    ),
                    family="bound",
                    meets_floor=meets,
                )
            )
    return hyps


def _hypotheses_northset(families: dict[str, Any]) -> list[HypothesisResult]:
    """Northset book hypotheses and the remaining policy contrasts."""
    hyps: list[HypothesisResult] = []
    ns = families.get("northset") or {}
    # mean_session_spread_bps_mean: session-L2 path ≠ daily mean_spread_bps.
    # mean_session_close_spread_bps: last-snap ≠ path mean_session_spread_bps_mean and ≠ daily mean_spread_bps.
    # mean_session_close_imbalance: last-snap ≠ path mean_session_imbalance_mean ≠ daily imbalance_top mean.
    # mean_session_imbalance_std: path dispersion ≠ path mean ≠ last-snap close imbalance; ≥0 when finite; receipt-only.
    # mean_session_ofi_abs_sum: path |OFI| sum ≥0; ≠ |session_ofi_sum_mean|; VPIN denom companion.
    # Never equate session_book_vpin_mean ≈ |ofi_sum_mean|/ofi_abs_mean (Jensen); soft-verify |ofi_sum_mean| ≤ ofi_abs_mean when both finite.
    # CLI: northset + research family echo SESSION_RECEIPT_KEYS_CLI_ECHO (== SESSION_RECEIPT_KEYS; no blob-only exceptions).
    # mean_session_close_micro_bps: last-snap micro ≠ daily microprice_minus_mid_bps mean (fuse companion; no fake Sharpe).
    # mean_session_close_mid: last-snap mid ≠ daily mid/close; companion of close_micro (not a Sharpe claim).
    # mean_session_close_bid/ask_depth: last-snap depths ≠ daily bid_depth/ask_depth means; ≥0 when finite.
    book_hypothesis_eligible = bool(ns.get("book_hypothesis_eligible", True))
    session_book_hypothesis_eligible = bool(ns.get("session_book_hypothesis_eligible", True))
    ohlc_r = _finite_number(ns.get("ohlc_identity_rate"))
    if ohlc_r is not None:
        meets_ohlc = bool(ohlc_r + 1e-12 >= 1.0)
        hyps.append(
            HypothesisResult(
                id="H20_northset_ohlc",
                statement="Daily candlestick rows satisfy OHLC identities (high/low envelope).",
                test="ohlc_identity_rate ≥ 1 bound check; not FDR",
                statistic=ohlc_r - 1.0,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    "OHLC identities hold."
                    if meets_ohlc
                    else "OHLC identities fail on a nonempty share of bars."
                ),
                family="bound",
                meets_floor=meets_ohlc,
            )
        )
    book_r = _finite_number(ns.get("book_uncrossed_rate")) if book_hypothesis_eligible else None
    if book_r is not None:
        meets_book = bool(book_r + 1e-12 >= 1.0)
        hyps.append(
            HypothesisResult(
                id="H21_northset_book",
                statement="L2 snapshots are uncrossed (best bid strictly below best ask).",
                test="book_uncrossed_rate ≥ 1 bound check; not FDR",
                statistic=book_r - 1.0,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    "Books are uncrossed."
                    if meets_book
                    else "Crossed or locked books appear in the Northset sample."
                ),
                family="bound",
                meets_floor=meets_book,
            )
        )
    p_imb = _finite_number(ns.get("imbalance_top_p_ic")) if book_hypothesis_eligible else None
    if p_imb is not None:
        hyps.append(
            _hyp(
                "H22_northset_imbalance",
                "Top-of-book depth imbalance has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level imbalance IC",
                float(ns.get("imbalance_top_t_ic", float("nan"))),
                p_imb,
                "Imbalance IC is distinguishable from 0.",
                "Imbalance IC not distinguishable from 0.",
                family="discovery",
            )
        )
    sess_r = _finite_number(ns.get("session_reconstructs_daily_rate"))
    if sess_r is not None:
        meets_sess = bool(sess_r + 1e-12 >= 1.0)
        hyps.append(
            HypothesisResult(
                id="H23_northset_session",
                statement="Session candles reconstruct the daily OHLC envelope.",
                test="session_reconstructs_daily_rate ≥ 1 bound check; not FDR",
                statistic=sess_r - 1.0,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    "Session envelope matches the daily bar."
                    if meets_sess
                    else "Session candles fail to reconstruct a nonempty share of daily bars."
                ),
                family="bound",
                meets_floor=meets_sess,
            )
        )
    vol_r = _finite_number(ns.get("session_volume_conservation_rate"))
    if vol_r is not None:
        meets_vol = bool(vol_r + 1e-12 >= 1.0)
        hyps.append(
            HypothesisResult(
                id="H24_northset_volume",
                statement="Session volumes sum to the daily volume.",
                test="session_volume_conservation_rate ≥ 1 bound check; not FDR",
                statistic=vol_r - 1.0,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    "Session volume is conserved."
                    if meets_vol
                    else "Session volume does not match the daily bar."
                ),
                family="bound",
                meets_floor=meets_vol,
            )
        )
    p_micro = _finite_number(ns.get("microprice_p_ic")) if book_hypothesis_eligible else None
    if p_micro is not None:
        hyps.append(
            _hyp(
                "H25_northset_microprice",
                "Microprice minus mid has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level microprice IC",
                float(ns.get("microprice_t_ic", float("nan"))),
                p_micro,
                "Microprice IC is distinguishable from 0.",
                "Microprice IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_wick = _finite_number(ns.get("wick_skew_p_ic"))
    if p_wick is not None:
        hyps.append(
            _hyp(
                "H26_northset_wick",
                "Lower-minus-upper wick skew has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level wick-skew IC",
                float(ns.get("wick_skew_t_ic", float("nan"))),
                p_wick,
                "Wick-skew IC is distinguishable from 0.",
                "Wick-skew IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_ofi = _finite_number(ns.get("ofi_p_ic")) if book_hypothesis_eligible else None
    if p_ofi is not None:
        hyps.append(
            _hyp(
                "H27_northset_ofi",
                "Top-of-book order-flow imbalance has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level OFI IC",
                float(ns.get("ofi_t_ic", float("nan"))),
                p_ofi,
                "OFI IC is distinguishable from 0.",
                "OFI IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_gk = _finite_number(ns.get("dm_gk_vs_park_p"))
    if p_gk is not None:
        hyps.append(
            _hyp(
                "H28_northset_gk",
                "Garman–Klass and Parkinson have unequal QLIKE vs close-to-close RV (Diebold–Mariano).",
                "Diebold–Mariano on date-level QLIKE",
                float(ns.get("dm_gk_vs_park_stat", float("nan"))),
                p_gk,
                f"DM prefers {ns.get('dm_gk_vs_park_preferred')}.",
                "Equal range-variance accuracy not rejected.",
                family="discovery",
            )
        )
    chain_r = _finite_number(ns.get("session_chain_rate"))
    if chain_r is not None:
        meets_chain = bool(chain_r + 1e-12 >= 1.0)
        hyps.append(
            HypothesisResult(
                id="H29_northset_chain",
                statement="Consecutive session candles chain: close[i] equals open[i+1].",
                test="session_chain_rate ≥ 1 bound check; not FDR",
                statistic=chain_r - 1.0,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    "Session candles chain."
                    if meets_chain
                    else "Session candle open/close chain is broken."
                ),
                family="bound",
                meets_floor=meets_chain,
            )
        )

    p_sb_vpin = (
        _finite_number(ns.get("session_book_vpin_p_ic"))
        if session_book_hypothesis_eligible
        else None
    )
    if p_sb_vpin is not None:
        hyps.append(
            _hyp(
                "H43_northset_session_book_vpin",
                "Session multi-snapshot book VPIN has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level session_book_vpin IC",
                float(ns.get("session_book_vpin_t_ic", float("nan"))),
                p_sb_vpin,
                "Session-book VPIN IC is distinguishable from 0.",
                "Session-book VPIN IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_clv = _finite_number(ns.get("clv_p_ic"))
    if p_clv is not None:
        hyps.append(
            _hyp(
                "H30_northset_clv",
                "Close location value has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level close-location IC",
                float(ns.get("clv_t_ic", float("nan"))),
                p_clv,
                "Close-location IC is distinguishable from 0.",
                "Close-location IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_split = _finite_number(ns.get("dm_split_vs_park_p"))
    if p_split is not None:
        hyps.append(
            _hyp(
                "H31_northset_overnight",
                "Overnight+open-to-close split and Parkinson have unequal QLIKE vs close-to-close RV.",
                "Diebold–Mariano on date-level QLIKE",
                float(ns.get("dm_split_vs_park_stat", float("nan"))),
                p_split,
                f"DM prefers {ns.get('dm_split_vs_park_preferred')}.",
                "Equal split vs Parkinson accuracy not rejected.",
                family="discovery",
            )
        )
    p_vpin = _finite_number(ns.get("vpin_p_ic")) if book_hypothesis_eligible else None
    if p_vpin is not None:
        hyps.append(
            _hyp(
                "H32_northset_vpin",
                "Bulk-volume VPIN proxy has nonzero date-level IC vs next-bar return.",
                "HAC t-stat of date-level VPIN IC",
                float(ns.get("vpin_t_ic", float("nan"))),
                p_vpin,
                "VPIN IC is distinguishable from 0.",
                "VPIN IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_sweep_rej = _finite_number(ns.get("sweep_reject_signed_p_ic"))
    if p_sweep_rej is not None:
        hyps.append(
            _hyp(
                "H33_northset_sweep_reject",
                "Signed sweep-reclaim depth has nonzero date-level IC vs next-bar return "
                "(descriptive IC, not an executable event study).",
                "HAC t-stat of date-level sweep-reject IC",
                float(ns.get("sweep_reject_signed_t_ic", float("nan"))),
                p_sweep_rej,
                "Sweep-reject IC is distinguishable from 0.",
                "Sweep-reject IC not distinguishable from 0.",
                family="discovery",
            )
        )
    p_sweep_fol = _finite_number(ns.get("sweep_follow_signed_p_ic"))
    if p_sweep_fol is not None:
        hyps.append(
            _hyp(
                "H34_northset_sweep_follow",
                "Signed sweep follow-through depth has nonzero date-level IC vs next-bar return "
                "(descriptive IC, not an executable event study).",
                "HAC t-stat of date-level sweep-follow IC",
                float(ns.get("sweep_follow_signed_t_ic", float("nan"))),
                p_sweep_fol,
                "Sweep-follow IC is distinguishable from 0.",
                "Sweep-follow IC not distinguishable from 0.",
                family="discovery",
            )
        )
    for hyp_id, prefix, label in (
        ("H35_northset_reject_event", "sweep_reject", "Sweep-reclaim"),
        ("H36_northset_follow_event", "sweep_follow", "Sweep follow-through"),
    ):
        p_event = _finite_number(ns.get(f"{prefix}_event_p"))
        if p_event is None:
            continue
        hyps.append(
            _hyp(
                hyp_id,
                f"{label} has nonzero next-open one-bar market-neutral event return.",
                "horizon-aware HAC on date-level event returns",
                float(ns.get(f"{prefix}_event_t", float("nan"))),
                p_event,
                f"{label} event return is distinguishable from 0.",
                f"{label} event return not distinguishable from 0.",
                family="discovery",
            )
        )
    for hyp_id, prefix, label in (
        ("H37_northset_reject_placebo", "sweep_reject", "Sweep-reclaim"),
        ("H38_northset_follow_placebo", "sweep_follow", "Sweep follow-through"),
    ):
        p_placebo = _finite_number(ns.get(f"{prefix}_placebo_p"))
        if p_placebo is None:
            continue
        hyps.append(
            _hyp(
                hyp_id,
                f"{label} alignment exceeds a within-date permutation placebo.",
                "within-date permutation test of mean cross-sectional IC",
                float(ns.get(f"{prefix}_placebo_observed_ic", float("nan"))),
                p_placebo,
                f"{label} alignment survives the permutation placebo.",
                f"{label} alignment does not survive the permutation placebo.",
                family="discovery",
            )
        )
    for hyp_id, prefix, label in (
        ("H39_northset_reject_cost", "sweep_reject", "Sweep-reclaim"),
        ("H40_northset_follow_cost", "sweep_follow", "Sweep follow-through"),
    ):
        # Honesty: *_cost_adjusted_mean_bps ≠ *_event_mean_bps (pre-cost);
        # soft-verify only requires finite-when-present for follow costed.
        costed = _finite_number(ns.get(f"{prefix}_cost_adjusted_mean_bps"))
        if costed is None:
            continue
        meets_cost = bool(costed > 0.0)
        hyps.append(
            HypothesisResult(
                id=hyp_id,
                statement=f"{label} mean excess return clears the modeled round-trip cost hurdle.",
                test="one-bar next-open cost-adjusted mean > 0 bps bound check; not FDR",
                statistic=costed,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    f"{label} clears modeled costs."
                    if meets_cost
                    else f"{label} does not clear modeled costs."
                ),
                family="bound",
                meets_floor=meets_cost,
            )
        )
    stability_floor_f = _finite_number(ns.get("sweep_min_fold_positive_fraction"))
    for hyp_id, prefix, label in (
        ("H41_northset_reject_stability", "sweep_reject", "Sweep-reclaim"),
        ("H42_northset_follow_stability", "sweep_follow", "Sweep follow-through"),
    ):
        stability = _finite_number(ns.get(f"{prefix}_fold_positive_fraction"))
        if stability is None:
            continue
        if stability_floor_f is None:
            # Never fabricate the configured floor: bound not evaluated.
            hyps.append(
                HypothesisResult(
                    id=hyp_id,
                    statement=f"{label} effect is positive across chronological folds.",
                    test="positive-fold fraction ≥ configured floor bound check; not FDR",
                    statistic=float("nan"),
                    p_value=float("nan"),
                    reject_raw=False,
                    reject_fdr=False,
                    decision="Fold-stability floor unavailable; bound not evaluated.",
                    family="bound",
                    meets_floor=None,
                )
            )
            continue
        stability_floor = stability_floor_f
        meets_stability = bool(stability + 1e-12 >= stability_floor)
        hyps.append(
            HypothesisResult(
                id=hyp_id,
                statement=f"{label} effect is positive across chronological folds.",
                test=f"positive-fold fraction ≥ {stability_floor:.2f} bound check; not FDR",
                statistic=stability - stability_floor,
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    f"{label} meets fold-stability floor."
                    if meets_stability
                    else f"{label} misses fold-stability floor."
                ),
                family="bound",
                meets_floor=meets_stability,
            )
        )
    for hyp_id, prefix, label in (
        ("H44_northset_reject_control", "sweep_reject", "Sweep-reclaim"),
        ("H45_northset_follow_control", "sweep_follow", "Sweep follow-through"),
    ):
        p_control = _finite_number(ns.get(f"{prefix}_control_diff_p"))
        if p_control is None:
            continue
        hyps.append(
            _hyp(
                hyp_id,
                f"{label} excess return beats direction-matched same-date eligible "
                "non-swept controls.",
                "HAC t on date-level event-minus-control excess difference",
                float(ns.get(f"{prefix}_control_diff_t", float("nan"))),
                p_control,
                f"{label} edge survives matched non-event controls.",
                f"{label} edge does not survive matched non-event controls.",
                family="discovery",
            )
        )
    for hyp_id, prefix, label in (
        ("H46_northset_reject_liq_control", "sweep_reject", "Sweep-reclaim"),
        ("H47_northset_follow_liq_control", "sweep_follow", "Sweep follow-through"),
    ):
        p_liq = _finite_number(ns.get(f"{prefix}_liq_control_diff_p"))
        if p_liq is None:
            continue
        hyps.append(
            _hyp(
                hyp_id,
                f"{label} excess return beats same-date eligible non-swept controls "
                "in the same lagged dollar-volume quartile.",
                "HAC t on date-level liquidity-quartile event-minus-control difference",
                float(ns.get(f"{prefix}_liq_control_diff_t", float("nan"))),
                p_liq,
                f"{label} edge survives liquidity-matched controls.",
                f"{label} edge does not survive liquidity-matched controls.",
                family="discovery",
            )
        )
    oot_holdout = _finite_number(ns.get("sweep_follow_oot_holdout_mean_bps"))
    if oot_holdout is not None:
        same_sign = _finite_number(ns.get("sweep_follow_oot_same_sign"))
        meets_oot = bool(same_sign is not None and float(same_sign) >= 1.0 - 1e-12)
        hyps.append(
            HypothesisResult(
                id="H48_northset_follow_oot",
                statement=(
                    "Sweep follow-through holdout mean has the same sign as the "
                    "in-sample mean on the last chronological fold."
                ),
                test="last-fold holdout same-sign bound check; not FDR",
                statistic=float(oot_holdout),
                p_value=float("nan"),
                reject_raw=False,
                reject_fdr=False,
                decision=(
                    "Follow-through holdout keeps the in-sample sign."
                    if meets_oot
                    else "Follow-through holdout does not keep the in-sample sign."
                ),
                family="bound",
                meets_floor=meets_oot,
            )
        )
    p_cluster = _finite_number(ns.get("sweep_follow_name_cluster_p"))
    if p_cluster is not None:
        hyps.append(
            _hyp(
                "H49_northset_follow_name_cluster",
                "Sweep follow-through mean excess is nonzero after collapsing to "
                "one observation per security.",
                "iid t on per-security mean signed excess (lags=0)",
                float(ns.get("sweep_follow_name_cluster_t", float("nan"))),
                p_cluster,
                "Follow-through survives name-clustered inference.",
                "Follow-through does not survive name-clustered inference.",
                family="discovery",
            )
        )
    p_two_way = _finite_number(ns.get("sweep_follow_two_way_cluster_p"))
    if p_two_way is not None:
        hyps.append(
            _hyp(
                "H50_northset_follow_two_way_cluster",
                "Sweep follow-through mean excess is nonzero after two-way "
                "clustering by date and security.",
                "Cameron–Gelbach–Miller two-way clustered t on event-level signed excess",
                float(ns.get("sweep_follow_two_way_cluster_t", float("nan"))),
                p_two_way,
                "Follow-through survives two-way clustered inference.",
                "Follow-through does not survive two-way clustered inference.",
                family="discovery",
            )
        )
    p_overnight_gap = _finite_number(ns.get("sweep_follow_overnight_gap_p"))
    if p_overnight_gap is not None:
        hyps.append(
            _hyp(
                "H51_northset_follow_overnight_gap",
                "Sweep follow-through loads a nonzero close-to-next-open gap "
                "that next-open entry cannot capture.",
                "calendar HAC t on signed overnight excess (idle dates at 0)",
                float(ns.get("sweep_follow_overnight_gap_t", float("nan"))),
                p_overnight_gap,
                "Follow-through overnight gap is detectable; close-to-close IC "
                "is not the executable study.",
                "Follow-through overnight gap is not detectable.",
                family="discovery",
            )
        )
    caps = families.get("interval_risk") or {}
    wide_f = _finite_number(caps.get("bind_wide"))
    tight_f = _finite_number(caps.get("bind_tight"))
    if wide_f is not None and tight_f is not None:
        wide_b = float(wide_f)
        tight_b = float(tight_f)
        contrast = wide_b - tight_b
        gap_s = _series_array(caps, "bind_gap_by_date")
        n_wide = caps.get("n_wide")
        n_tight = caps.get("n_tight")
        if gap_s is not None:
            _mu, t, p_two = mean_tstat(gap_s)
            p = onesided_from_twosided(t, p_two, greater=True)
            test = "one-sided HAC t on date-level bind_wide − bind_tight"
        elif n_wide is not None and n_tight is not None:
            _z, p = two_proportion_test(
                int(caps.get("n_bind_wide") or 0),
                int(n_wide),
                int(caps.get("n_bind_tight") or 0),
                int(n_tight),
                alternative="greater",
            )
            test = "one-sided two-proportion test of bind rates"
        else:
            _t, p = mean_difference_t(contrast, _first_int(caps, "n_dates", "n") or 0)
            test = "mean-difference t using reported n_dates, not dummy 0/1"
        weight_rule = str(caps.get("weight_rule") or "")
        if weight_rule == "equal_weight_per_date":
            statement = (
                "Equal-weight names (1/n on the date) in the wide-interval half "
                "bind more than the tight half."
            )
        else:
            statement = (
                "Wide-interval names bind more than tight-interval names after interval caps."
            )
        if _finite_number(p) is not None:
            hyps.append(
                _hyp(
                    "H13_interval_caps",
                    statement,
                    test,
                    contrast,
                    p,
                    "Wide sets bind more than tight sets.",
                    "Interval caps do not bind the wide half more.",
                    family="discovery",
                )
            )
    qb = families.get("quantile_bandit") or {}
    adv_qb = _finite_number(qb.get("mean_advantage_vs_ridge"))
    if adv_qb is not None:
        adv = float(adv_qb)
        t, p_two, test = _contrast_inference(
            qb,
            adv,
            series_keys=("policy_reward",),
            baseline_keys=("ridge_reward",),
            gap_keys=("reward_gap_series",),
        )
        p = onesided_from_twosided(t, p_two, greater=True)
        if _finite_number(p) is not None:
            hyps.append(
                _hyp(
                    "H14_quantile_thompson",
                    "Quantile Thompson top-k reward exceeds a static public ridge top-k policy "
                    "(same public features, same dates).",
                    test + " (one-sided vs ridge)",
                    adv,
                    p,
                    "Quantile Thompson beats static public ridge on the scientific reward.",
                    "Quantile Thompson does not beat static public ridge.",
                    family="discovery",
                )
            )
    return hyps


def _build_hypotheses(
    families: dict[str, Any],
    rankers: list[dict[str, Any]],
) -> list[HypothesisResult]:
    hyps = _hypotheses_rankers_and_rewards(families, rankers)
    hyps.extend(_hypotheses_conformal_bounds(families))
    hyps.extend(_hypotheses_northset(families))
    _apply_family_fdr(hyps, "calibration")
    _apply_family_fdr(hyps, "discovery")
    return hyps


def run_research(config: AppConfig) -> ResearchNotebook:
    set_global_seed(config.train.random_seed)
    build_gold(config, refresh=config.data.source == "synthetic")
    df = panel(config)
    if "security_id" in df.columns:
        df = df.filter(pl.col("security_id") != config.data.benchmark_id)
    label = config.train.ranking_target
    if label not in df.columns:
        cands = [c for c in df.columns if c.startswith("future_idio_return")]
        if not cands:
            cands = [c for c in df.columns if c.startswith("future_excess_return")]
        label = cands[0] if cands else "future_return_1"

    rankers = bench_ranking(df, config, label)
    # Multi-fold stability from date-level IC series (promotion evidence)
    from quant_fund.validation.walk_forward import fold_ic_stability

    fold_stability_summary = {}
    for r in rankers:
        if str(r.get("name", "")).startswith("_"):
            continue
        series = r.get("ic_series") or []
        if series:
            stab = fold_ic_stability(series, min_ic=float(config.promotion.min_mean_ic))
            r["fold_ic_stability"] = stab["stability"]
            r["n_folds"] = stab["n_folds"]
            fold_stability_summary[str(r["name"])] = stab
    northset_frame = ensure_silver(config)
    if "security_id" in northset_frame.columns:
        northset_frame = northset_frame.filter(pl.col("security_id") != config.data.benchmark_id)
    provenance = _provenance(
        config,
        df,
        label,
        northset_frame=northset_frame,
    )
    ranking_blob: dict[str, Any] = {
        "target": label,
        "models": rankers,
        "fold_stability": fold_stability_summary,
    }
    # Data-snooping over the ranker universe: is the best date-level IC real
    # after accounting for every ranker/feature-set tried? Research diagnostic.
    snooping = _ranker_data_snooping(rankers)
    if snooping is not None:
        ranking_blob["data_snooping"] = snooping
    families = {
        "ranking": ranking_blob,
        "alpha": bench_alpha(df, config, label),
        "volatility": bench_volatility(df, config),
        "distribution": bench_distribution(df, config),
        "regime": bench_regime(df, config),
        "tail": bench_tail(df, config),
        "drawdown": bench_drawdown(df, config),
        "liquidity": bench_liquidity(df),
        "reinforcement": bench_rl(df, label),
        "conformal": bench_conformal(df, config),
        "evalues": bench_evalues(df, config),
        "jackknife_plus": bench_jackknife_plus(df, config),
        "crc": bench_crc(df, config),
        "weighted_conformal": bench_weighted_conformal(df, config),
        "interval_risk": bench_interval_risk(df, config),
        "quantile_bandit": bench_quantile_bandit(df, label),
        "cv_plus": bench_cv_plus(df, config),
        "cpcv": bench_cpcv_audit(),
        "localized_conformal": bench_localized_from_panel(df, config),
        "conformal_rank": bench_conformal_topk_from_panel(df, config),
        "online_crc": bench_online_crc_from_panel(df, config),
        "portfolio_conformal": bench_portfolio_from_panel(df, config),
        "northset": bench_northset(northset_frame, config),
        "complexity": bench_complexity(df),
        "roughness": bench_roughness(df),
        "serial_randomness": bench_serial_randomness(df),
        "anytime_valid": bench_anytime_valid(),
        "energy_score": bench_energy_score(),
        "ts_conformal": bench_ts_conformal(),
        "regime_eval": bench_regime_eval(),
        "leakage_redteam": bench_leakage_redteam(),
        "distributional_ml": bench_distributional_ml(),
        "rough_paths": bench_rough_paths(),
        "optimal_transport": bench_optimal_transport(),
        "wasserstein_dro": bench_dro(),
        "conformal_pid": bench_conformal_pid(),
        "nexcp": bench_nexcp(),
        "mh_enbpi": bench_mh_enbpi(),
        "stacking": bench_stacking(),
        "rwcv": bench_rwcv(),
        "sliced_wasserstein": bench_sliced_wasserstein(),
        "score_decomposition": bench_score_decomposition(),
        "confidence_sequences": bench_confidence_sequences(),
        "deep_hedging": bench_deep_hedging(),
        "coverage_inference": bench_coverage_inference(),
        "conformal_e_detectors": bench_conformal_e_detectors(),
        "rolling_conformal": bench_rolling_conformal(),
        "rank_cs": bench_rank_cs(),
        "picpi": bench_picpi(),
        "replicable_conformal": bench_replicable_conformal(),
        "reference_null": bench_reference_null(),
        "delayed_aci": bench_delayed_aci(),
        "martingale_ot": bench_martingale_ot(),
        "large_deviations": bench_large_deviations(),
        "mean_field_games": bench_mean_field_games(),
        "vine_copula": bench_vine_copula(),
        "malliavin_greeks": bench_malliavin_greeks(),
        "xva": bench_xva(),
        "american_lsm": bench_american_lsm(),
        "local_stoch_vol": bench_local_stoch_vol(),
        "deep_bsde": bench_deep_bsde(),
        "deep_regime_mixture": bench_deep_regime_mixture(),
        "odd_residual_flows": bench_odd_residual_flows(),
        "deep_kernel_hedging": bench_deep_kernel_hedging(),
        "zi_lob": bench_zi_lob(),
        "cash_constrained_oe": bench_cash_constrained_oe(),
        "subspace_denoising": bench_subspace_denoising(),
        "conformal_transfer": bench_conformal_transfer(),
        "hpd_conformal": bench_hpd_conformal(),
        "capability_value": bench_capability_value(),
        "agent_referee": bench_agent_referee(),
        "vintage_eval": bench_vintage_eval(),
        "entropy_shapley": bench_entropy_shapley(),
        "fourier_pricing": bench_fourier_pricing(),
        "vol_loss_decomposition": bench_vol_loss_decomposition(),
        "hierarchical_conformal": bench_hierarchical_conformal(),
        "multisource_conformal": bench_multisource_conformal(),
        "extra_tilt": bench_extra_tilt(),
        "forecast_selection": bench_forecast_selection(),
        "rl_market_maker": bench_rl_market_maker(),
        "conformal_oce": bench_conformal_oce(),
        "adaptive_eps": bench_adaptive_eps(),
        "greek_neutral": bench_greek_neutral(),
        "diffusion_forecaster": bench_diffusion_forecaster(),
        "diffpts": bench_diffpts(),
        "extra_conformal": bench_extra_conformal(),
        "multilevel_mm": bench_multilevel_mm(),
        "rlmm_c51": bench_rlmm_c51(),
        "sga_uq": bench_sga_uq(),
        "passive_impact": bench_passive_impact(),
        "stochastic_tracking": bench_stochastic_tracking(),
        "gslice": bench_gslice(),
        "neural_sde": bench_neural_sde(),
        "stocbench": bench_stocbench(),
        "agentic_lob": bench_agentic_lob(),
        "fase_eval": bench_fase_eval(),
        "kit_paths": bench_kit_paths(),
        "langevin_impact": bench_langevin_impact(),
        "event_time_flow": bench_event_time_flow(),
        "fukasawa_iv": bench_fukasawa_iv(),
        "ivs_diffusion": bench_ivs_diffusion(),
        "rccp": bench_rccp(),
        "dcp": bench_dcp(),
        "varswap_stopping": bench_varswap_stopping(),
        "hidden_markov_equilibrium": bench_hidden_markov_equilibrium(),
        "gaussian_normalized_coords": bench_gaussian_normalized_coords(),
        "liquidity_tail_lob": bench_liquidity_tail_lob(),
        "arl_mm": bench_arl_mm(),
        "bocpd_changepoint": bench_bocpd_changepoint(),
        "rough_heston_rbergomi": bench_rough_heston_rbergomi(),
        "signature_features": bench_signature_features(),
        "signature_martingale_test": bench_signature_martingale_test(),
        "svi_surface": bench_svi_surface(),
        "propagator_impact": bench_propagator_impact(),
        "queue_reactive": bench_queue_reactive(),
        "koopman_edmd": bench_koopman_edmd(),
        "sig_gan": bench_sig_gan(),
        "neural_tpp": bench_neural_tpp(),
        "pmcmc_sv": bench_pmcmc_sv(),
        "multifractal_vol": bench_multifractal_vol(),
        "spci_conformal": bench_spci_conformal(),
        "hawkes_em": bench_hawkes_em(),
        "fernholz_spt": bench_fernholz_spt(),
        "breeden_litzenberger": bench_breeden_litzenberger(),
        "tda_persistence": bench_tda_persistence(),
        "fractional_ou": bench_fractional_ou(),
        "fourier_hermite": bench_fourier_hermite(),
        "kernel_changepoint": bench_kernel_changepoint(),
        "kinetic_ising": bench_kinetic_ising(),
        "heterogeneous_abm": bench_heterogeneous_abm(),
        "marchenko_pastur": bench_marchenko_pastur(),
        "factor_nowcast": bench_factor_nowcast(),
        "stationary_bootstrap": bench_stationary_bootstrap(),
        "skill_ratings": bench_skill_ratings(),
        "modularity_communities": bench_modularity_communities(),
        "stein_thinning": bench_stein_thinning(),
        "durbin_koopman": bench_durbin_koopman(),
        "dp_mixture": bench_dp_mixture(),
        "expert_aggregation": bench_expert_aggregation(),
        "instrumental_quantile": bench_instrumental_quantile(),
        "implied_tree": bench_implied_tree(),
        "ensemble_kalman_inversion": bench_ensemble_kalman_inversion(),
        "enkf": bench_enkf(),
        "causal_discovery": bench_causal_discovery(),
        "knockoffs": bench_knockoffs(),
        "callaway_did": bench_callaway_did(),
        "surrogate_nonlinear": bench_surrogate_nonlinear(),
        "sindy": bench_sindy(),
        "hmc": bench_hmc(),
        "proxy_svar": bench_proxy_svar(),
        "sbi": bench_sbi(),
        "rqa": bench_rqa(),
        "hj_distance": bench_hj_distance(),
        "tensor_decomp": bench_tensor_decomp(),
        "lp_iv": bench_lp_iv(),
        "bispectrum": bench_bispectrum(),
        "functional_linear": bench_functional_linear(),
        "gas_score": bench_gas_score(),
        "count_data": bench_count_data(),
        "lyapunov": bench_lyapunov(),
        "kernel_iv": bench_kernel_iv(),
        "multistate": bench_multistate(),
        "partial_linear": bench_partial_linear(),
        "heckman": bench_heckman(),
        "rd": bench_rd(),
        "bounds": bench_bounds(),
        "extreme_value": bench_extreme_value(),
        "double_ml": bench_double_ml(),
        "bunching": bench_bunching(),
        "causal_forest": bench_causal_forest(),
        "gaussian_process": bench_gaussian_process(),
        "markov_switching": bench_markov_switching(),
        "causal_impact": bench_causal_impact(),
        "weak_iv": bench_weak_iv(),
        "synth_did": bench_synth_did(),
        "permutation_inference": bench_permutation_inference(),
        "propensity_score": bench_propensity_score(),
        "cluster_robust": bench_cluster_robust(),
        "matrix_completion": bench_matrix_completion(),
        "gsynth": bench_gsynth(),
        "rif_regression": bench_rif_regression(),
        "shift_share": bench_shift_share(),
        "entropy_balancing": bench_entropy_balancing(),
        "did_diagnostics": bench_did_diagnostics(),
        "honest_did": bench_honest_did(),
        "many_iv": bench_many_iv(),
        "fama_macbeth": bench_fama_macbeth(),
        "specification_curve": bench_specification_curve(),
        "sign_restricted_var": bench_sign_restricted_var(),
        "panel_quantile_fe": bench_panel_quantile_fe(),
        "arellano_bond": bench_arellano_bond(),
        "bvar_minnesota": bench_bvar_minnesota(),
        "mediation_analysis": bench_mediation_analysis(),
        "competing_risks": bench_competing_risks(),
        "stochastic_frontier": bench_stochastic_frontier(),
        "regression_kink": bench_regression_kink(),
        "spatial_econometrics": bench_spatial_econometrics(),
        "ordered_choice": bench_ordered_choice(),
        "triple_difference": bench_triple_difference(),
        "distribution_regression": bench_distribution_regression(),
        "simex": bench_simex(),
        "lp_did": bench_lp_did(),
        "control_function": bench_control_function(),
        "kernel_regression": bench_kernel_regression(),
        "censored_quantile": bench_censored_quantile(),
        "threshold_ar": bench_threshold_ar(),
        "fractional_response": bench_fractional_response(),
        "interval_censoring": bench_interval_censoring(),
        "maximum_score": bench_maximum_score(),
        "sieve_estimation": bench_sieve_estimation(),
        "nested_logit": bench_nested_logit(),
        "aft_model": bench_aft_model(),
        "distance_covariance": bench_distance_covariance(),
        "panel_unitroot": bench_panel_unitroot(),
        "mixed_logit": bench_mixed_logit(),
        "hurdle": bench_hurdle(),
        "sur_model": bench_sur_model(),
        "connectedness": bench_connectedness(),
        "nonparametric_iv": bench_nonparametric_iv(),
        "subsampling": bench_subsampling(),
        "frailty": bench_frailty(),
        "interrupted_ts": bench_interrupted_ts(),
        "lead_lag": bench_lead_lag(),
        "ppml": bench_ppml(),
        "event_study": bench_event_study(),
        "model_averaging": bench_model_averaging(),
        "variance_ratio": bench_variance_ratio(),
        "har_rv": bench_har_rv(),
        "clark_west": bench_clark_west(),
        "stambaugh": bench_stambaugh(),
        "roy_model": bench_roy_model(),
        "two_way_cluster": bench_two_way_cluster(),
        "bai_perron": bench_bai_perron(),
        "favar": bench_favar(),
        "panel_coint": bench_panel_coint(),
        "vuong_test": bench_vuong_test(),
        "merton_model": bench_merton_model(),
        "white_reality": bench_white_reality(),
        "johansen_vecm": bench_johansen_vecm(),
        "pin_model": bench_pin_model(),
        "kyle_lambda": bench_kyle_lambda(),
        "oster_bounds": bench_oster_bounds(),
        "storey_fdr": bench_storey_fdr(),
        "kiefer_vogelsang": bench_kiefer_vogelsang(),
        "conley_se": bench_conley_se(),
        "driscoll_kraay": bench_driscoll_kraay(),
        "pesaran_cce": bench_pesaran_cce(),
        "wald_sprt": bench_wald_sprt(),
        "lee_bounds": bench_lee_bounds(),
        "barrett_donald": bench_barrett_donald(),
        "blp_demand": bench_blp_demand(),
        "olley_pakes": bench_olley_pakes(),
        "rust_ddc": bench_rust_ddc(),
        "oaxaca_blinder": bench_oaxaca_blinder(),
        "binscatter": bench_binscatter(),
        "dfl_decomp": bench_dfl_decomp(),
        "rosenbaum_sensitivity": bench_rosenbaum_sensitivity(),
        "aipw_ate": bench_aipw_ate(),
        "cavi_gmm": bench_cavi_gmm(),
        "pesaran_cd": bench_pesaran_cd(),
        "hausman_tests": bench_hausman_tests(),
        "cusum_monitor": bench_cusum_monitor(),
        "tmle": bench_tmle(),
        "lewbel_iv": bench_lewbel_iv(),
        "proximal_causal": bench_proximal_causal(),
        "cover_up": bench_cover_up(),
        "vpin": bench_vpin(),
        "marginal_treatment": bench_marginal_treatment(),
        "eisenberg_noe": bench_eisenberg_noe(),
        "fire_sales": bench_fire_sales(),
        "delta_covar": bench_delta_covar(),
        "blanchard_quah": bench_blanchard_quah(),
        "tvp_var": bench_tvp_var(),
        "meta_analysis": bench_meta_analysis(),
        "acd_duration": bench_acd_duration(),
        "hjm": bench_hjm(),
        "affine_term": bench_affine_term(),
        "gil_pelaez": bench_gil_pelaez(),
        "hedonic": bench_hedonic(),
        "dea": bench_dea(),
        "bkm_moments": bench_bkm_moments(),
        "gsadf_bubble": bench_gsadf_bubble(),
        "pmg_ardl": bench_pmg_ardl(),
        "ross_recovery": bench_ross_recovery(),
        "ait_sahalia": bench_ait_sahalia(),
        "toda_yamamoto": bench_toda_yamamoto(),
        "diebold_mariano": bench_diebold_mariano(),
        "engle_granger": bench_engle_granger(),
        "glosten_milgrom": bench_glosten_milgrom(),
        "hasbrouck_is": bench_hasbrouck_is(),
        "bds": bench_bds(),
        "cochrane_piazzesi": bench_cochrane_piazzesi(),
        "engle_ng": bench_engle_ng(),
        "nardl": bench_nardl(),
        "growth_at_risk": bench_growth_at_risk(),
        "melick_thomas": bench_melick_thomas(),
        "bandi_russell": bench_bandi_russell(),
        "hong_li": bench_hong_li(),
        "beveridge_nelson": bench_beveridge_nelson(),
        "corradi_swanson": bench_corradi_swanson(),
        "engle_kroner_bekk": bench_engle_kroner_bekk(),
        "heston_qe": bench_heston_qe(),
        "model_confidence_set": bench_model_confidence_set(),
        "christoffersen_pelletier": bench_christoffersen_pelletier(),
        "sheppard_heavy": bench_sheppard_heavy(),
        "pesaran_timmermann": bench_pesaran_timmermann(),
        "giacomini_rossi": bench_giacomini_rossi(),
        "muller_watson": bench_muller_watson(),
        "romano_wolf": bench_romano_wolf(),
        "christensen_diebold_rudebusch": bench_christensen_diebold_rudebusch(),
        "danielsson_devries": bench_danielsson_devries(),
        "kpss": bench_kpss(),
        "ers_dfgls": bench_ers_dfgls(),
        "ng_perron": bench_ng_perron(),
        "phillips_perron": bench_phillips_perron(),
        "zivot_andrews": bench_zivot_andrews(),
        "lee_strazicich": bench_lee_strazicich(),
        "wavelet_modwt": bench_wavelet_modwt(),
        "geweke_spectral": bench_geweke_spectral(),
        "ivx": bench_ivx(),
        "bai_ng_ic": bench_bai_ng_ic(),
        "wooldridge_serial": bench_wooldridge_serial(),
        "lasso_pds": bench_lasso_pds(),
        "extremogram": bench_extremogram(),
        "echo_state": bench_echo_state(),
        "bates_svj": bench_bates_svj(),
        "stl_loess": bench_stl_loess(),
        "pelt_wbs": bench_pelt_wbs(),
        "spectral_pca": bench_spectral_pca(),
        "emd_hht": bench_emd_hht(),
        "gallant_snp": bench_gallant_snp(),
        "srisk": bench_srisk(),
        "wavelet_coherence": bench_wavelet_coherence(),
        "tar_coint": bench_tar_coint(),
        "extreme_qr": bench_extreme_qr(),
        "kalman_em": bench_kalman_em(),
        "fractional_coint": bench_fractional_coint(),
        "star_model": bench_star_model(),
        "garch_in_mean": bench_garch_in_mean(),
        "log_acd": bench_log_acd(),
        "wigner_ville": bench_wigner_ville(),
        "log_concave": bench_log_concave(),
        "dtw_warp": bench_dtw_warp(),
        "chow_lin": bench_chow_lin(),
        "chen_tiao_outliers": bench_chen_tiao_outliers(),
        "beta_ar": bench_beta_ar(),
        "ingarch": bench_ingarch(),
        "entropy_pooling": bench_entropy_pooling(),
        "narrative_svar": bench_narrative_svar(),
        "bfast": bench_bfast(),
        "stable_dist": bench_stable_dist(),
        "asian_option": bench_asian_option(),
        "black_litterman": bench_black_litterman(),
        "higham_corr": bench_higham_corr(),
        "lee_carter": bench_lee_carter(),
        "power_law": bench_power_law(),
        "convexity_adj": bench_convexity_adj(),
        "jln_uncertainty": bench_jln_uncertainty(),
        "first_passage": bench_first_passage(),
        "saddlepoint": bench_saddlepoint(),
        "mutual_info": bench_mutual_info(),
        "transfer_entropy": bench_transfer_entropy(),
        "brownian_bridge": bench_brownian_bridge(),
        "jarrow_turnbull": bench_jarrow_turnbull(),
        "campbell_shiller": bench_campbell_shiller(),
        "libor_market": bench_libor_market(),
        "cos_method": bench_cos_method(),
        "obizhaeva_wang": bench_obizhaeva_wang(),
        "spread_options": bench_spread_options(),
        "esscher": bench_esscher(),
        "shadow_rate": bench_shadow_rate(),
        "mlmc": bench_mlmc(),
        "black_karasinski": bench_black_karasinski(),
        "debtrank": bench_debtrank(),
        "ews_signals": bench_ews_signals(),
        "permutation_entropy": bench_permutation_entropy(),
        "svgd": bench_svgd(),
        "nested_sampling": bench_nested_sampling(),
        "smc_samplers": bench_smc_samplers(),
        "synthetic_likelihood": bench_synthetic_likelihood(),
        "state_dependent_lp": bench_state_dependent_lp(),
        "moment_inequalities": bench_moment_inequalities(),
        "euler_risk": bench_euler_risk(),
        "vix_replication": bench_vix_replication(),
        "mala": bench_mala(),
        "functional_pca": bench_functional_pca(),
        "particle_gibbs": bench_particle_gibbs(),
        "ripley_k": bench_ripley_k(),
        "synchrosqueezing": bench_synchrosqueezing(),
        "vmd": bench_vmd(),
        "stockwell": bench_stockwell(),
        "cca": bench_cca(),
        "isomap": bench_isomap(),
        "kriging": bench_kriging(),
        "innovations_ets": bench_innovations_ets(),
        "cyclostationary": bench_cyclostationary(),
        "empirical_wavelets": bench_empirical_wavelets(),
        "nonparametric_tests": bench_nonparametric_tests(),
        "polychoric": bench_polychoric(),
        "compositional": bench_compositional(),
        "sure_screening": bench_sure_screening(),
        "item_response": bench_item_response(),
        "latent_class": bench_latent_class(),
        "manova": bench_manova(),
        "procrustes": bench_procrustes(),
        "isotonic": bench_isotonic(),
        "hrp": bench_hrp(),
        "factor_analysis": bench_factor_analysis(),
        "slice_sampling": bench_slice_sampling(),
        "kuiper": bench_kuiper(),
        "p_spline": bench_p_spline(),
        "thin_plate": bench_thin_plate(),
        "james_stein": bench_james_stein(),
        "mardia": bench_mardia(),
        "mantel": bench_mantel(),
        "moran": bench_moran(),
        "friedman": bench_friedman(),
        "contingency": bench_contingency(),
        "hoeffding": bench_hoeffding(),
        "dispersion_tests": bench_dispersion_tests(),
        "median_tests": bench_median_tests(),
        "cochran_q": bench_cochran_q(),
        "quade": bench_quade(),
        "van_der_waerden": bench_van_der_waerden(),
        "dunn_test": bench_dunn_test(),
        "cronbach": bench_cronbach(),
        "icc": bench_icc(),
        "dif": bench_dif(),
        "g_theory": bench_g_theory(),
        "omega": bench_omega(),
        "rasch_fit": bench_rasch_fit(),
        "horvitz_thompson": bench_horvitz_thompson(),
        "poststrat": bench_poststrat(),
        "calibration_survey": bench_calibration_survey(),
        "fay_herriot": bench_fay_herriot(),
        "cluster_sampling": bench_cluster_sampling(),
        "design_effects": bench_design_effects(),
        "gee": bench_gee(),
        "lmm": bench_lmm(),
        "interrater": bench_interrater(),
        "isolation_forest": bench_isolation_forest(),
        "hegy": bench_hegy(),
        "mice": bench_mice(),
        "multiple_comparisons": bench_multiple_comparisons(),
        "rank_aggregation": bench_rank_aggregation(),
        "spc": bench_spc(),
        "risk_parity": bench_risk_parity(),
        "mst_topology": bench_mst_topology(),
        "e_divisive": bench_e_divisive(),
        "fastica": bench_fastica(),
        "barrier_options": bench_barrier_options(),
        "tost": bench_tost(),
        "msm_causal": bench_msm_causal(),
        "nmf": bench_nmf(),
        "auxiliary_pf": bench_auxiliary_pf(),
        "rmst": bench_rmst(),
        "ois_curve": bench_ois_curve(),
        "henze_zirkler": bench_henze_zirkler(),
        "epps_singleton": bench_epps_singleton(),
        "watson": bench_watson(),
        "energy_test": bench_energy_test(),
        "dirichlet_multinomial": bench_dirichlet_multinomial(),
        "vonmises_fisher": bench_vonmises_fisher(),
        "fkml": bench_fkml(),
        "gandh": bench_gandh(),
        "robbins_monro": bench_robbins_monro(),
        "pocs": bench_pocs(),
        "delong_auc": bench_delong_auc(),
        "passing_bablok": bench_passing_bablok(),
        "marginal_homogeneity": bench_marginal_homogeneity(),
        "circular_tests": bench_circular_tests(),
        "graded_irt": bench_graded_irt(),
        "welch_anova": bench_welch_anova(),
        "lin_ccc": bench_lin_ccc(),
        "influence": bench_influence(),
        "group_sequential": bench_group_sequential(),
        "mds": bench_mds(),
        "correspondence_analysis": bench_correspondence_analysis(),
        "circular_correlation": bench_circular_correlation(),
        "lmoments": bench_lmoments(),
        "sobol_sensitivity": bench_sobol_sensitivity(),
        "recurrent_events": bench_recurrent_events(),
        "dawid_skene": bench_dawid_skene(),
        "matrix_profile": bench_matrix_profile(),
        "hierarchical_reconciliation": bench_hierarchical_reconciliation(),
        "cma_es": bench_cma_es(),
        "sketches": bench_sketches(),
        "music_esprit": bench_music_esprit(),
        "chain_ladder": bench_chain_ladder(),
        "erlang_queueing": bench_erlang_queueing(),
        "inequality_indices": bench_inequality_indices(),
        "rainflow_fatigue": bench_rainflow_fatigue(),
        "bayesian_tracking": bench_bayesian_tracking(),
        "sbm_inference": bench_sbm_inference(),
        "pu_learning": bench_pu_learning(),
        "gr4j_hydrology": bench_gr4j_hydrology(),
        "brinson_attribution": bench_brinson_attribution(),
        "avellaneda_stoikov": bench_avellaneda_stoikov(),
        "gillespie_ssa": bench_gillespie_ssa(),
        "hamilton_filter": bench_hamilton_filter(),
        "corwin_schultz": bench_corwin_schultz(),
        "gwr_spatial": bench_gwr_spatial(),
        "pareto_nbd": bench_pareto_nbd(),
        "markov_discretization": bench_markov_discretization(),
        "lomb_scargle": bench_lomb_scargle(),
        "singular_spectrum": bench_singular_spectrum(),
        "beck_katz": bench_beck_katz(),
        "quandt_andrews": bench_quandt_andrews(),
        "friedman_supersmoother": bench_friedman_supersmoother(),
        "edf_tests": bench_edf_tests(),
        "normality_tests": bench_normality_tests(),
        "scale_homogeneity": bench_scale_homogeneity(),
        "score_scale": bench_score_scale(),
        "het_regressions": bench_het_regressions(),
        "serial_diagnostics": bench_serial_diagnostics(),
        "mars_regression": bench_mars_regression(),
        "ace_avas": bench_ace_avas(),
        "projection_pursuit": bench_projection_pursuit(),
        "marginal_likelihood": bench_marginal_likelihood(),
        "root_finders": bench_root_finders(),
        "blind_sources": bench_blind_sources(),
        "unconstrained_optimizers": bench_unconstrained_optimizers(),
        "clustering_methods": bench_clustering_methods(),
        "manifold_learning": bench_manifold_learning(),
        "robust_regression": bench_robust_regression(),
        "empirical_bayes": bench_empirical_bayes(),
        "design_experiments": bench_design_experiments(),
        "metaheuristic_optimizers": bench_metaheuristic_optimizers(),
        "bayesian_optimization": bench_bayesian_optimization(),
        "fuzzy_clustering": bench_fuzzy_clustering(),
        "self_organizing_maps": bench_self_organizing_maps(),
        "pagerank_topology": bench_pagerank_topology(),
        "hyperband_search": bench_hyperband_search(),
        "tree_ensembles": bench_tree_ensembles(),
        "metric_learning": bench_metric_learning(),
        "one_class_classification": bench_one_class_classification(),
        "gp_classification": bench_gp_classification(),
        "phase_retrieval": bench_phase_retrieval(),
        "factorization_machine": bench_factorization_machine(),
        "svm_classifiers": bench_svm_classifiers(),
        "discriminant_analysis": bench_discriminant_analysis(),
        "coordinate_descent_enet": bench_coordinate_descent_enet_family(),
        "kernel_methods": bench_kernel_methods_family(),
        "lda_topics": bench_lda_topics_family(),
        "online_convex": bench_online_convex_family(),
        "bayesian_linear": bench_bayesian_linear_family(),
        "graphical_models": bench_graphical_models_family(),
        "conjugate_gradient": bench_conjugate_gradient_family(),
        "frank_wolfe": bench_frank_wolfe_family(),
        "sparse_coding": bench_sparse_coding_family(),
        "evolution_strategies": bench_evolution_strategies_family(),
        "naive_bayes": bench_naive_bayes_family(),
        "adaboost": bench_adaboost_family(),
        "expectation_propagation": bench_expectation_propagation_family(),
        "collaborative_filtering": bench_collaborative_filtering_family(),
        "association_rules": bench_association_rules_family(),
        "nash_equilibrium": bench_nash_equilibrium_family(),
        "proximal_gradient": bench_proximal_gradient_family(),
        "quadrature": bench_quadrature_family(),
        "ode_solvers": bench_ode_solvers_family(),
        "semisupervised": bench_semisupervised_family(),
        "multiclass": bench_multiclass_family(),
        "sparse_pca": bench_sparse_pca_family(),
        "mdp_solvers": bench_mdp_solvers_family(),
        "td_learning": bench_td_learning_family(),
        "belief_propagation": bench_belief_propagation_family(),
        "extreme_learning": bench_extreme_learning_family(),
        "nearest_centroid": bench_nearest_centroid_family(),
        "cross_entropy_method": bench_cross_entropy_method_family(),
        "gibbs_sampler": bench_gibbs_sampler_family(),
        "laplace_approx": bench_laplace_approx_family(),
        "kde": bench_kde_family(),
        "tensor_power": bench_tensor_power_family(),
        "evidential": bench_evidential_family(),
        "multi_task": bench_multi_task_family(),
        "homotopy_continuation": bench_homotopy_continuation_family(),
        "anderson_accel": bench_anderson_accel_family(),
        "sequence_accel": bench_sequence_accel_family(),
        "iterative_ls": bench_iterative_ls_family(),
        "qmc_sequences": bench_qmc_sequences_family(),
        "symplectic_ode": bench_symplectic_ode_family(),
        "graph_traversal": bench_graph_traversal_family(),
        "shortest_paths": bench_shortest_paths_family(),
        "network_flow": bench_network_flow_family(),
        "assignment": bench_assignment_family(),
        "graph_components": bench_graph_components_family(),
        "exact_cover": bench_exact_cover_family(),
        "stochastic_bandits": bench_stochastic_bandits_family(),
        "kl_bandits": bench_kl_bandits_family(),
        "contextual_bandits": bench_contextual_bandits_family(),
        "adversarial_bandits": bench_adversarial_bandits_family(),
        "best_arm": bench_best_arm_family(),
        "nonstationary_bandits": bench_nonstationary_bandits_family(),
        "lanczos": bench_lanczos_family(),
        "arnoldi_gmres": bench_arnoldi_gmres_family(),
        "randomized_svd": bench_randomized_svd_family(),
        "nystrom": bench_nystrom_family(),
        "cur_decomp": bench_cur_decomp_family(),
        "interpolative_decomp": bench_interpolative_decomp_family(),
        "sinkhorn": bench_sinkhorn_family(),
        "emd_lp": bench_emd_lp_family(),
        "gromov_wasserstein": bench_gromov_wasserstein_family(),
        "unbalanced_ot": bench_unbalanced_ot_family(),
        "wasserstein_barycenter": bench_wasserstein_barycenter_family(),
        "fused_gromov": bench_fused_gromov_family(),
        "alpha_beta": bench_alpha_beta_family(),
        "mcts": bench_mcts_family(),
        "puct": bench_puct_family(),
        "negascout": bench_negascout_family(),
        "proof_number": bench_proof_number_family(),
        "dfpn": bench_dfpn_family(),
        "bdf": bench_bdf_family(),
        "adams": bench_adams_family(),
        "radau": bench_radau_family(),
        "strang": bench_strang_family(),
        "etdrk4": bench_etdrk4_family(),
        "crank_nicolson": bench_crank_nicolson_family(),
        "adi": bench_adi_family(),
        "lax_wendroff": bench_lax_wendroff_family(),
        "weno": bench_weno_family(),
        "level_set": bench_level_set_family(),
        "fast_marching": bench_fast_marching_family(),
        "godunov": bench_godunov_family(),
        "expm_pade": bench_expm_pade_family(),
        "matrix_sqrt": bench_matrix_sqrt_family(),
        "sylvester": bench_sylvester_family(),
        "riccati_care": bench_riccati_care_family(),
        "matrix_sign": bench_matrix_sign_family(),
        "toeplitz_solve": bench_toeplitz_solve_family(),
        "dubins": bench_dubins_family(),
        "rrt": bench_rrt_family(),
        "prm": bench_prm_family(),
        "dwa": bench_dwa_family(),
        "min_snap": bench_min_snap_family(),
        "frenet": bench_frenet_family(),
        "viterbi_decode": bench_viterbi_decode_family(),
        "gf256": bench_gf256_family(),
        "reed_solomon": bench_reed_solomon_family(),
        "costas": bench_costas_family(),
        "gardner": bench_gardner_family(),
        "rrc_filter": bench_rrc_filter_family(),
        "kabsch": bench_kabsch_family(),
        "icp": bench_icp_family(),
        "frechet": bench_frechet_family(),
        "hausdorff": bench_hausdorff_family(),
        "convex_hull": bench_convex_hull_family(),
        "delaunay": bench_delaunay_family(),
        "remez": bench_remez_family(),
        "iir_design": bench_iir_design_family(),
        "biquad": bench_biquad_family(),
        "filtfilt": bench_filtfilt_family(),
        "resample_poly": bench_resample_poly_family(),
        "farrow": bench_farrow_family(),
        "jonker_volgenant": bench_jonker_volgenant_family(),
        "jpda": bench_jpda_family(),
        "phd": bench_phd_family(),
        "mht": bench_mht_family(),
        "cov_int": bench_cov_int_family(),
        "tdoa": bench_tdoa_family(),
        "gold_code": bench_gold_code_family(),
        "klobuchar": bench_klobuchar_family(),
        "allan_variance": bench_allan_variance_family(),
        "strapdown": bench_strapdown_family(),
        "lambda_method": bench_lambda_method_family(),
        "rtk": bench_rtk_family(),
        "smolyak": bench_smolyak_family(),
        "pce": bench_pce_family(),
        "bayesian_quadrature": bench_bayesian_quadrature_family(),
        "kl_expand": bench_kl_expand_family(),
        "active_subspace": bench_active_subspace_family(),
        "mimc": bench_mimc_family(),
        "cfr": bench_cfr_family(),
        "lemke_howson": bench_lemke_howson_family(),
        "replicator": bench_replicator_family(),
        "wardrop": bench_wardrop_family(),
        "vcg": bench_vcg_family(),
        "nash_bargain": bench_nash_bargain_family(),
        "gae": bench_gae_family(),
        "vtrace": bench_vtrace_family(),
        "trpo": bench_trpo_family(),
        "ppo": bench_ppo_family(),
        "ddpg": bench_ddpg_family(),
        "td3": bench_td3_family(),
        "qmdp": bench_qmdp_family(),
        "grid_pomdp": bench_grid_pomdp_family(),
        "pbvi": bench_pbvi_family(),
        "perseus": bench_perseus_family(),
        "hsvi": bench_hsvi_family(),
        "pomcp": bench_pomcp_family(),
        "vdn": bench_vdn_family(),
        "qmix": bench_qmix_family(),
        "coma": bench_coma_family(),
        "maddpg": bench_maddpg_family(),
        "mappo": bench_mappo_family(),
        "mf_q": bench_mf_q_family(),
        "nucleolus": bench_nucleolus_family(),
        "banzhaf": bench_banzhaf_family(),
        "owen": bench_owen_family(),
        "myerson_auction": bench_myerson_auction_family(),
        "groves": bench_groves_family(),
        "envy_free": bench_envy_free_family(),
        "lil_ucb": bench_lil_ucb_family(),
        "sequential_halving": bench_sequential_halving_family(),
        "median_elim": bench_median_elim_family(),
        "ugape": bench_ugape_family(),
        "ttts": bench_ttts_family(),
        "track_stop": bench_track_stop_family(),
        "exec_rl": bench_exec_rl_family(),
        "smart_router": bench_smart_router_family(),
        "order_flow_imbalance": bench_order_flow_imbalance_family(),
        "pg_mm": bench_pg_mm_family(),
        "options_flow": bench_options_flow_family(),
        "dark_pool": bench_dark_pool_family(),
        "say_echo_do": bench_say_echo_do_family(),
        "multimodal_fusion": bench_multimodal_fusion_family(),
        "ts_diffusion": bench_ts_diffusion_family(),
        "synthetic_gan": bench_synthetic_gan_family(),
        "econ_calendar": bench_econ_calendar_family(),
        "quantcode_bench": bench_quantcode_bench_family(),
        "asset_gnn": bench_asset_gnn_family(),
        "counterparty_gnn": bench_counterparty_gnn_family(),
        "maml_portfolio": bench_maml_portfolio_family(),
        "continual_learning": bench_continual_learning_family(),
        "fed_avg": bench_fed_avg_family(),
        "insider_anomaly": bench_insider_anomaly_family(),
        "pinn_pricing": bench_pinn_pricing_family(),
        "qubo_portfolio": bench_qubo_portfolio_family(),
        "xai_shap": bench_xai_shap_family(),
        "adversarial_robust": bench_adversarial_robust_family(),
        "risk_flow": bench_risk_flow_family(),
        "causal_miner": bench_causal_miner_family(),
        "tft_forecaster": bench_tft_forecaster_family(),
        "patchtst": bench_patchtst_family(),
        "lob_transformer": bench_lob_transformer_family(),
        "set_transformer": bench_set_transformer_family(),
        "neural_ode": bench_neural_ode_family(),
        "world_model": bench_world_model_family(),
        "contrastive_repr": bench_contrastive_repr_family(),
        "hypernetwork_alloc": bench_hypernetwork_alloc_family(),
        "neural_thompson": bench_neural_thompson_family(),
        "bnn_ensemble": bench_bnn_ensemble_family(),
        "option_vae": bench_option_vae_family(),
        "diff_policy": bench_diff_policy_family(),
        "kan_forecaster": bench_kan_forecaster_family(),
        "ts_mixer": bench_ts_mixer_family(),
        "informer_attn": bench_informer_attn_family(),
        "cnn_alpha": bench_cnn_alpha_family(),
        "mask_autoencoder": bench_mask_autoencoder_family(),
        "graph_temporal": bench_graph_temporal_family(),
        "itransformer": bench_itransformer_family(),
        "tcn_forecaster": bench_tcn_forecaster_family(),
        "ft_transformer": bench_ft_transformer_family(),
        "nbeats_deep": bench_nbeats_deep_family(),
        "mambats": bench_mambats_family(),
        "crossformer": bench_crossformer_family(),
        "decision_transformer": bench_decision_transformer_family(),
        "cql_agent": bench_cql_agent_family(),
        "iql_agent": bench_iql_agent_family(),
        "trajectory_transformer": bench_trajectory_transformer_family(),
        "sac_agent": bench_sac_agent_family(),
        "gail_imitation": bench_gail_imitation_family(),
        "vq_vae_ts": bench_vq_vae_ts_family(),
        "flow_matching_ts": bench_flow_matching_ts_family(),
        "score_sde_ts": bench_score_sde_ts_family(),
        "consistency_ts": bench_consistency_ts_family(),
        "energy_ts": bench_energy_ts_family(),
        "perceiver_ts": bench_perceiver_ts_family(),
        "mahalanobis_ood": bench_mahalanobis_ood_family(),
        "max_softmax_ood": bench_max_softmax_ood_family(),
        "gradient_norm_ood": bench_gradient_norm_ood_family(),
        "energy_ood": bench_energy_ood_family(),
        "knn_ood": bench_knn_ood_family(),
        "vim_ood": bench_vim_ood_family(),
        "neural_process": bench_neural_process_family(),
        "attentive_np": bench_attentive_np_family(),
        "deep_kernel_gp": bench_deep_kernel_gp_family(),
        "convnp": bench_convnp_family(),
        "meta_uq": bench_meta_uq_family(),
        "llaplace_gp": bench_llaplace_gp_family(),
        "optnet_qp": bench_optnet_qp_family(),
        "cvxpy_layer": bench_cvxpy_layer_family(),
        "input_convex": bench_input_convex_family(),
        "deep_declarative": bench_deep_declarative_family(),
        "spd_net": bench_spd_net_family(),
        "diff_mpc": bench_diff_mpc_family(),
        "chebnet": bench_chebnet_family(),
        "graphsage": bench_graphsage_family(),
        "gin_gnn": bench_gin_gnn_family(),
        "graph_unet": bench_graph_unet_family(),
        "apnp_prop": bench_apnp_prop_family(),
        "jk_net": bench_jk_net_family(),
        "linformer_attn": bench_linformer_attn_family(),
        "performer_attn": bench_performer_attn_family(),
        "linear_attn": bench_linear_attn_family(),
        "sliding_attn": bench_sliding_attn_family(),
        "sinkhorn_attn": bench_sinkhorn_attn_family(),
        "nystrom_attn": bench_nystrom_attn_family(),
        "c51_dqn": bench_c51_dqn_family(),
        "qr_dqn": bench_qr_dqn_family(),
        "iqn_dqn": bench_iqn_dqn_family(),
        "noisy_net": bench_noisy_net_family(),
        "prioritized_replay": bench_prioritized_replay_family(),
        "bootstrapped_dqn": bench_bootstrapped_dqn_family(),
        "reformer_lsh": bench_reformer_lsh_family(),
        "memorizing_transformer": bench_memorizing_transformer_family(),
        "ntm_memory": bench_ntm_memory_family(),
        "dnc_memory": bench_dnc_memory_family(),
        "rssm_world": bench_rssm_world_family(),
        "mpc_planning": bench_mpc_planning_family(),
        "byol": bench_byol_family(),
        "barlow_twins": bench_barlow_twins_family(),
        "vicreg": bench_vicreg_family(),
        "tent_tta": bench_tent_tta_family(),
        "shot_tta": bench_shot_tta_family(),
        "ttt_layer": bench_ttt_layer_family(),
        "hyperbolic_nn": bench_hyperbolic_nn_family(),
        "capsule_dynamic": bench_capsule_dynamic_family(),
        "siren_inr": bench_siren_inr_family(),
        "equivar_gnn": bench_equivar_gnn_family(),
        "monotonic_net": bench_monotonic_net_family(),
        "sort_net": bench_sort_net_family(),
        "randomized_smoothing": bench_randomized_smoothing_family(),
        "ibp_bounds": bench_ibp_bounds_family(),
        "crown_bound": bench_crown_bound_family(),
        "lipschitz_net": bench_lipschitz_net_family(),
        "vector_neurons": bench_vector_neurons_family(),
        "gumbel_topk": bench_gumbel_topk_family(),
        "s4_ssm": bench_s4_ssm_family(),
        "rwkv_wkv": bench_rwkv_wkv_family(),
        "hyena_conv": bench_hyena_conv_family(),
        "retnet_decay": bench_retnet_decay_family(),
        "delta_net": bench_delta_net_family(),
        "mixture_of_depths": bench_mixture_of_depths_family(),
        "lora_ft": bench_lora_ft_family(),
        "qlora_nf4": bench_qlora_nf4_family(),
        "dora_weight": bench_dora_weight_family(),
        "prompt_tuning": bench_prompt_tuning_family(),
        "prefix_tuning": bench_prefix_tuning_family(),
        "task_vector_merge": bench_task_vector_merge_family(),
        "speculative_decoding": bench_speculative_decoding_family(),
        "paged_kv_cache": bench_paged_kv_cache_family(),
        "flash_attn": bench_flash_attn_family(),
        "gqa_attn": bench_gqa_attn_family(),
        "sliding_window_cache": bench_sliding_window_cache_family(),
        "ring_attn": bench_ring_attn_family(),
        "bm25_retriever": bench_bm25_retriever_family(),
        "dpr_retriever": bench_dpr_retriever_family(),
        "colbert_late": bench_colbert_late_family(),
        "hyde_retrieval": bench_hyde_retrieval_family(),
        "reranker_crossenc": bench_reranker_crossenc_family(),
        "rrf_fusion": bench_rrf_fusion_family(),
        "consistency_vote": bench_consistency_vote_family(),
        "verifier_prm": bench_verifier_prm_family(),
        "mcts_reason": bench_mcts_reason_family(),
        "debate_multiagent": bench_debate_multiagent_family(),
        "unlearn_ga": bench_unlearn_ga_family(),
        "knowledge_graph_embed": bench_knowledge_graph_embed_family(),
        "reward_model": bench_reward_model_family(),
        "dpo_train": bench_dpo_train_family(),
        "ipo_train": bench_ipo_train_family(),
        "kto_train": bench_kto_train_family(),
        "grpo_train": bench_grpo_train_family(),
        "rlhf_ppo": bench_rlhf_ppo_family(),
        "sae_feature": bench_sae_feature_family(),
        "activation_steering": bench_activation_steering_family(),
        "probe_linear": bench_probe_linear_family(),
        "logit_lens": bench_logit_lens_family(),
        "patch_activation": bench_patch_activation_family(),
        "circuit_ablation": bench_circuit_ablation_family(),
        "dataset_distillation": bench_dataset_distillation_family(),
        "coreset_herding": bench_coreset_herding_family(),
        "curriculum_magnitude": bench_curriculum_magnitude_family(),
        "label_smoothing": bench_label_smoothing_family(),
        "mixup_cutmix": bench_mixup_cutmix_family(),
        "sharpness_sam": bench_sharpness_sam_family(),
        "magnitude_pruning": bench_magnitude_pruning_family(),
        "lottery_ticket": bench_lottery_ticket_family(),
        "quant_int8": bench_quant_int8_family(),
        "kd_distill": bench_kd_distill_family(),
        "lowrank_factor": bench_lowrank_factor_family(),
        "fisher_prune": bench_fisher_prune_family(),
        "react_loop": bench_react_loop_family(),
        "toolformer_call": bench_toolformer_call_family(),
        "plan_search": bench_plan_search_family(),
        "reflexion_retry": bench_reflexion_retry_family(),
        "multi_agent_pipeline": bench_multi_agent_pipeline_family(),
        "judge_pairwise": bench_judge_pairwise_family(),
        "dp_sgd": bench_dp_sgd_family(),
        "secure_agg": bench_secure_agg_family(),
        "fedavg_hetero": bench_fedavg_hetero_family(),
        "pate_teacher": bench_pate_teacher_family(),
        "gradient_leakage": bench_gradient_leakage_family(),
        "canary_exposure": bench_canary_exposure_family(),
        "random_search_nas": bench_random_search_nas_family(),
        "evolution_nas": bench_evolution_nas_family(),
        "darts_nas": bench_darts_nas_family(),
        "enas_controller": bench_enas_controller_family(),
        "one_shot_nas": bench_one_shot_nas_family(),
        "arch_predictor": bench_arch_predictor_family(),
        "convnet_baseline": bench_convnet_baseline_family(),
        "vit_classifier": bench_vit_classifier_family(),
        "clip_align": bench_clip_align_family(),
        "simclr_views": bench_simclr_views_family(),
        "diffusion_ddim": bench_diffusion_ddim_family(),
        "attention_rollout": bench_attention_rollout_family(),
        "tarnet_ite": bench_tarnet_ite_family(),
        "dragonnet_dr": bench_dragonnet_dr_family(),
        "deep_iv": bench_deep_iv_family(),
        "cevae_latent": bench_cevae_latent_family(),
        "causal_rep": bench_causal_rep_family(),
        "policy_value": bench_policy_value_family(),
        "tokenizer_bpe": bench_tokenizer_bpe_family(),
        "tabular_resnet": bench_tabular_resnet_family(),
        "node_net": bench_node_net_family(),
        "grownet_boost": bench_grownet_boost_family(),
        "soft_tree": bench_soft_tree_family(),
        "tabm_mini": bench_tabm_mini_family(),
        "deep_svdd": bench_deep_svdd_family(),
        "dagmm": bench_dagmm_family(),
        "usad": bench_usad_family(),
        "anom_transformer": bench_anom_transformer_family(),
        "rrcf": bench_rrcf_family(),
        "tranad": bench_tranad_family(),
        "ranknet_ltr": bench_ranknet_ltr_family(),
        "listnet_ltr": bench_listnet_ltr_family(),
        "listmle_ltr": bench_listmle_ltr_family(),
        "lambdarank_ltr": bench_lambdarank_ltr_family(),
        "approx_ndcg_ltr": bench_approx_ndcg_ltr_family(),
        "neural_sort_ltr": bench_neural_sort_ltr_family(),
        "muon_opt": bench_muon_opt_family(),
        "lion_opt": bench_lion_opt_family(),
        "sophia_opt": bench_sophia_opt_family(),
        "lookahead_opt": bench_lookahead_opt_family(),
        "lamb_opt": bench_lamb_opt_family(),
        "adafactor_opt": bench_adafactor_opt_family(),
        "scaffold_fl": bench_scaffold_fl_family(),
        "fednova_fl": bench_fednova_fl_family(),
        "ditto_fl": bench_ditto_fl_family(),
        "moon_fl": bench_moon_fl_family(),
        "fedopt_adam": bench_fedopt_adam_family(),
        "mime_lite": bench_mime_lite_family(),
        "fno_1d": bench_fno_1d_family(),
        "deeponet": bench_deeponet_family(),
        "lowrank_op": bench_lowrank_op_family(),
        "pino_residual": bench_pino_residual_family(),
        "gno_lite": bench_gno_lite_family(),
        "cno_lite": bench_cno_lite_family(),
        "chronos_lite": bench_chronos_lite_family(),
        "timesfm_lite": bench_timesfm_lite_family(),
        "moirai_lite": bench_moirai_lite_family(),
        "lagllama_lite": bench_lagllama_lite_family(),
        "timer_lite": bench_timer_lite_family(),
        "moment_lite": bench_moment_lite_family(),
        "packnet_cl": bench_packnet_cl_family(),
        "lwf_cl": bench_lwf_cl_family(),
        "der_cl": bench_der_cl_family(),
        "agem_cl": bench_agem_cl_family(),
        "piggyback_cl": bench_piggyback_cl_family(),
        "hat_cl": bench_hat_cl_family(),
        "swag_diag": bench_swag_diag_family(),
        "mc_dropout": bench_mc_dropout_family(),
        "bbb_vi": bench_bbb_vi_family(),
        "snapshot_ens": bench_snapshot_ens_family(),
        "concrete_dropout": bench_concrete_dropout_family(),
        "vcl_online": bench_vcl_online_family(),
        "mdn_cond": bench_mdn_cond_family(),
        "flow_regression": bench_flow_regression_family(),
        "diffusion_regressor": bench_diffusion_regressor_family(),
        "het_gp": bench_het_gp_family(),
        "crps_net": bench_crps_net_family(),
        "kernel_mixture": bench_kernel_mixture_family(),
        "reptile": bench_reptile_family(),
        "protonet": bench_protonet_family(),
        "matching_net": bench_matching_net_family(),
        "anil_meta": bench_anil_meta_family(),
        "meta_sgd": bench_meta_sgd_family(),
        "r2d2_meta": bench_r2d2_meta_family(),
        "algo_reasoning": bench_algo_reasoning_family(),
        "pna_agg": bench_pna_agg_family(),
        "virtual_node": bench_virtual_node_family(),
        "gps_transformer": bench_gps_transformer_family(),
        "oversmooth_metric": bench_oversmooth_metric_family(),
        "dgn_directional": bench_dgn_directional_family(),
        "awac": bench_awac_family(),
        "redq": bench_redq_family(),
        "td7_lite": bench_td7_lite_family(),
        "crossq": bench_crossq_family(),
        "dr3_reg": bench_dr3_reg_family(),
        "ob2i": bench_ob2i_family(),
        "advi_bbvi": bench_advi_bbvi_family(),
        "iwae_bound": bench_iwae_bound_family(),
        "nf_vi": bench_nf_vi_family(),
        "sparse_gp_sv": bench_sparse_gp_sv_family(),
        "structured_vi": bench_structured_vi_family(),
        "vrnn_seq": bench_vrnn_seq_family(),
        "xlearner": bench_xlearner_family(),
        "rlearner": bench_rlearner_family(),
        "slearner_tlearner": bench_slearner_tlearner_family(),
        "causal_rep_bal": bench_causal_rep_bal_family(),
        "cate_distill": bench_cate_distill_family(),
        "net_drlearner": bench_net_drlearner_family(),
        "cqr_pred": bench_cqr_pred_family(),
        "survival_cp": bench_survival_cp_family(),
        "aps_cp": bench_aps_cp_family(),
        "ltt_cp": bench_ltt_cp_family(),
        "full_cp": bench_full_cp_family(),
        "risk_cp": bench_risk_cp_family(),
        "psrl": bench_psrl_family(),
        "gittins_index": bench_gittins_index_family(),
        "whittle_restless": bench_whittle_restless_family(),
        "cucb": bench_cucb_family(),
        "corrupt_bandit": bench_corrupt_bandit_family(),
        "neural_ucb": bench_neural_ucb_family(),
        "edm_karras": bench_edm_karras_family(),
        "rectified_flow": bench_rectified_flow_family(),
        "stoch_interp": bench_stoch_interp_family(),
        "ddim_ode": bench_ddim_ode_family(),
        "cold_diffusion": bench_cold_diffusion_family(),
        "diff_distill": bench_diff_distill_family(),
        "dcrnn_lite": bench_dcrnn_lite_family(),
        "stgcn_lite": bench_stgcn_lite_family(),
        "gwnet_lite": bench_gwnet_lite_family(),
        "astgcn": bench_astgcn_family(),
        "mtgnn_lite": bench_mtgnn_lite_family(),
        "agcrn": bench_agcrn_family(),
        "rope_attn": bench_rope_attn_family(),
        "alibi_attn": bench_alibi_attn_family(),
        "swiglu_ffn": bench_swiglu_ffn_family(),
        "rmsnorm_block": bench_rmsnorm_block_family(),
        "moe_router": bench_moe_router_family(),
        "mup_init": bench_mup_init_family(),
        "active_bald": bench_active_bald_family(),
        "data_cartography": bench_data_cartography_family(),
        "el2n_scoring": bench_el2n_scoring_family(),
        "forgetting_events": bench_forgetting_events_family(),
        "influence_func": bench_influence_func_family(),
        "proto_prune": bench_proto_prune_family(),
        "lif_neuron": bench_lif_neuron_family(),
        "stdp_learn": bench_stdp_learn_family(),
        "surrogate_snn": bench_surrogate_snn_family(),
        "izhikevich": bench_izhikevich_family(),
        "lsm_reservoir": bench_lsm_reservoir_family(),
        "temporal_code": bench_temporal_code_family(),
        "pcgrad": bench_pcgrad_family(),
        "mgda_mtl": bench_mgda_mtl_family(),
        "cagrad_mtl": bench_cagrad_mtl_family(),
        "gradnorm_bal": bench_gradnorm_bal_family(),
        "nash_mtl": bench_nash_mtl_family(),
        "imtl_g": bench_imtl_g_family(),
        "deepsurv": bench_deepsurv_family(),
        "deephit": bench_deephit_family(),
        "cox_time": bench_cox_time_family(),
        "nnet_surv": bench_nnet_surv_family(),
        "drsa_surv": bench_drsa_surv_family(),
        "pchazard": bench_pchazard_family(),
        "bouncy_particle": bench_bouncy_particle_family(),
        "zigzag_sampler": bench_zigzag_sampler_family(),
        "boomerang_sampler": bench_boomerang_sampler_family(),
        "kinetic_langevin": bench_kinetic_langevin_family(),
        "elliptical_slice": bench_elliptical_slice_family(),
        "riemannian_mala": bench_riemannian_mala_family(),
        "mamba2_ssd": bench_mamba2_ssd_family(),
        "xlstm_mlstm": bench_xlstm_mlstm_family(),
        "rwkv7": bench_rwkv7_family(),
        "titans_memory": bench_titans_memory_family(),
        "gated_deltanet": bench_gated_deltanet_family(),
        "longhorn_ssm": bench_longhorn_ssm_family(),
        "real_nvp": bench_real_nvp_family(),
        "glow_flow": bench_glow_flow_family(),
        "neural_spline_flow": bench_neural_spline_flow_family(),
        "maf_flow": bench_maf_flow_family(),
        "planar_flow": bench_planar_flow_family(),
        "iaf_flow": bench_iaf_flow_family(),
        "notears": bench_notears_family(),
        "dagma_lin": bench_dagma_lin_family(),
        "golem_ev": bench_golem_ev_family(),
        "notears_mlp": bench_notears_mlp_family(),
        "dag_gnn": bench_dag_gnn_family(),
        "cam_prune": bench_cam_prune_family(),
        "hessian_eig": bench_hessian_eig_family(),
        "ntk_kernel": bench_ntk_kernel_family(),
        "edge_stability": bench_edge_stability_family(),
        "mode_connectivity": bench_mode_connectivity_family(),
        "catapult_phase": bench_catapult_phase_family(),
        "neural_grok": bench_neural_grok_family(),
        "st_estimator": bench_st_estimator_family(),
        "gumbel_relax": bench_gumbel_relax_family(),
        "perturb_map": bench_perturb_map_family(),
        "implicit_diff": bench_implicit_diff_family(),
        "ode_adjoint": bench_ode_adjoint_family(),
        "smooth_argmax": bench_smooth_argmax_family(),
        "score_matching": bench_score_matching_family(),
        "denoising_sm": bench_denoising_sm_family(),
        "noise_contrastive": bench_noise_contrastive_family(),
        "contrastive_divergence": bench_contrastive_divergence_family(),
        "persistent_cd": bench_persistent_cd_family(),
        "adversarial_ebm": bench_adversarial_ebm_family(),
        "deepritz_pinn": bench_deepritz_pinn_family(),
        "weak_form_pinn": bench_weak_form_pinn_family(),
        "fbsde_solver": bench_fbsde_solver_family(),
        "spectral_pde": bench_spectral_pde_family(),
        "moc_lines": bench_moc_lines_family(),
        "feynman_kac_mc": bench_feynman_kac_mc_family(),
        "entropy_query": bench_entropy_query_family(),
        "margin_sampling": bench_margin_sampling_family(),
        "qbc_committee": bench_qbc_committee_family(),
        "coreset_kcenter": bench_coreset_kcenter_family(),
        "badge_embed": bench_badge_embed_family(),
        "egl_change": bench_egl_change_family(),
        "alphazero_lite": bench_alphazero_lite_family(),
        "expert_iteration": bench_expert_iteration_family(),
        "nfsp": bench_nfsp_family(),
        "psro": bench_psro_family(),
        "deep_cfr": bench_deep_cfr_family(),
        "mccfr_outcome": bench_mccfr_outcome_family(),
        "ges_search": bench_ges_search_family(),
        "fci_alg": bench_fci_alg_family(),
        "ica_lingam": bench_ica_lingam_family(),
        "direct_lingam": bench_direct_lingam_family(),
        "var_lingam": bench_var_lingam_family(),
        "mmmb_select": bench_mmmb_select_family(),
        "count_bonus": bench_count_bonus_family(),
        "rnd_explore": bench_rnd_explore_family(),
        "icm_explore": bench_icm_explore_family(),
        "ngu_explore": bench_ngu_explore_family(),
        "ride_explore": bench_ride_explore_family(),
        "go_explore": bench_go_explore_family(),
        "mmd_two_sample": bench_mmd_two_sample_family(),
        "hsic_independence": bench_hsic_independence_family(),
        "mine_mi": bench_mine_mi_family(),
        "nwj_mi": bench_nwj_mi_family(),
        "copula_mi": bench_copula_mi_family(),
        "lsd_deptest": bench_lsd_deptest_family(),
        "lqr_control": bench_lqr_control_family(),
        "ddp_solve": bench_ddp_solve_family(),
        "mppi_control": bench_mppi_control_family(),
        "pmp_bangbang": bench_pmp_bangbang_family(),
        "mpc_qp": bench_mpc_qp_family(),
        "lqg_control": bench_lqg_control_family(),
        "cir_sim": bench_cir_sim_family(),
        "ou_bridge": bench_ou_bridge_family(),
        "poisson_thinning": bench_poisson_thinning_family(),
        "hawkes_thinning": bench_hawkes_thinning_family(),
        "levy_jump": bench_levy_jump_family(),
        "gp_bridge": bench_gp_bridge_family(),
        "emcee_stretch": bench_emcee_stretch_family(),
        "de_mcmc": bench_de_mcmc_family(),
        "dram": bench_dram_family(),
        "rjmcmc": bench_rjmcmc_family(),
        "pcn_sampler": bench_pcn_sampler_family(),
        "indep_mh": bench_indep_mh_family(),
        "power_iter": bench_power_iter_family(),
        "inverse_iter": bench_inverse_iter_family(),
        "jacobi_eig": bench_jacobi_eig_family(),
        "qr_eig": bench_qr_eig_family(),
        "hessenberg_red": bench_hessenberg_red_family(),
        "bidiag_svd": bench_bidiag_svd_family(),
        "eoq_model": bench_eoq_model_family(),
        "newsvendor": bench_newsvendor_family(),
        "ss_policy": bench_ss_policy_family(),
        "wagner_whitin": bench_wagner_whitin_family(),
        "base_stock": bench_base_stock_family(),
        "clark_scarf": bench_clark_scarf_family(),
        "johnson_flowshop": bench_johnson_flowshop_family(),
        "neh_heuristic": bench_neh_heuristic_family(),
        "lpt_schedule": bench_lpt_schedule_family(),
        "knapsack_dp": bench_knapsack_dp_family(),
        "tsp_branchbound": bench_tsp_branchbound_family(),
        "spt_weighted": bench_spt_weighted_family(),
        "qaoa_maxcut": bench_qaoa_maxcut_family(),
        "vqe_ising": bench_vqe_ising_family(),
        "grover_search": bench_grover_search_family(),
        "qpe_phase": bench_qpe_phase_family(),
        "qkernel_svm": bench_qkernel_svm_family(),
        "quantum_walk": bench_quantum_walk_family(),
        "tt_svd": bench_tt_svd_family(),
        "dmrg_tfim": bench_dmrg_tfim_family(),
        "tensor_cross": bench_tensor_cross_family(),
        "tebd_quench": bench_tebd_quench_family(),
        "mps_fidelity": bench_mps_fidelity_family(),
        "tt_round": bench_tt_round_family(),
        "psor_american": bench_psor_american_family(),
        "crr_tree": bench_crr_tree_family(),
        "kushner_mca": bench_kushner_mca_family(),
        "hjb_penalty": bench_hjb_penalty_family(),
        "dual_american": bench_dual_american_family(),
        "exercise_boundary": bench_exercise_boundary_family(),
        "mfg_lq": bench_mfg_lq_family(),
        "mfg_flocking": bench_mfg_flocking_family(),
        "nash_cournot": bench_nash_cournot_family(),
        "stackelberg_game": bench_stackelberg_game_family(),
        "stochastic_game_vi": bench_stochastic_game_vi_family(),
        "potential_game": bench_potential_game_family(),
        "fisher_rao": bench_fisher_rao_family(),
        "natural_gradient": bench_natural_gradient_family(),
        "mirror_descent": bench_mirror_descent_family(),
        "bregman_nmf": bench_bregman_nmf_family(),
        "alpha_geodesic": bench_alpha_geodesic_family(),
        "jko_scheme": bench_jko_scheme_family(),
        "jackson_network": bench_jackson_network_family(),
        "bcmp_mva": bench_bcmp_mva_family(),
        "gordon_newell": bench_gordon_newell_family(),
        "ctmc_availability": bench_ctmc_availability_family(),
        "renewal_reward": bench_renewal_reward_family(),
        "vacation_queue": bench_vacation_queue_family(),
        "vickrey_auction": bench_vickrey_auction_family(),
        "first_price_auction": bench_first_price_auction_family(),
        "all_pay_auction": bench_all_pay_auction_family(),
        "ascending_clock": bench_ascending_clock_family(),
        "double_auction": bench_double_auction_family(),
        "gsp_auction": bench_gsp_auction_family(),
        "ucb_bound": bench_ucb_bound_family(),
        "mw_hedge": bench_mw_hedge_family(),
        "egreedy_decay": bench_egreedy_decay_family(),
        "pi_contraction": bench_pi_contraction_family(),
        "td_rate": bench_td_rate_family(),
        "qlearn_rate": bench_qlearn_rate_family(),
        "sha256_impl": bench_sha256_impl_family(),
        "aes_sbox": bench_aes_sbox_family(),
        "shamir_secret": bench_shamir_secret_family(),
        "pedersen_commit": bench_pedersen_commit_family(),
        "diffie_hellman": bench_diffie_hellman_family(),
        "ecc_secp256k1": bench_ecc_secp256k1_family(),
        "cwt_ridge": bench_cwt_ridge_family(),
        "cepstrum_pitch": bench_cepstrum_pitch_family(),
        "mvdr_beamformer": bench_mvdr_beamformer_family(),
        "hilbert_instant": bench_hilbert_instant_family(),
        "lpc_formant": bench_lpc_formant_family(),
        "goertzel_detect": bench_goertzel_detect_family(),
        "weibull_life": bench_weibull_life_family(),
        "fault_tree": bench_fault_tree_family(),
        "ram_markov": bench_ram_markov_family(),
        "fmea_rpn": bench_fmea_rpn_family(),
        "life_stress": bench_life_stress_family(),
        "redundancy_block": bench_redundancy_block_family(),
        "ldpc_decoder": bench_ldpc_decoder_family(),
        "turbo_decoder": bench_turbo_decoder_family(),
        "polar_code": bench_polar_code_family(),
        "bch_code": bench_bch_code_family(),
        "crc_check": bench_crc_check_family(),
        "conv_interleaver": bench_conv_interleaver_family(),
        "gomory_cut": bench_gomory_cut_family(),
        "column_generation": bench_column_generation_family(),
        "benders_decomp": bench_benders_decomp_family(),
        "lagrangian_relax": bench_lagrangian_relax_family(),
        "branch_and_cut": bench_branch_and_cut_family(),
        "held_karp": bench_held_karp_family(),
        "greedy_set_cover": bench_greedy_set_cover_family(),
        "primal_dual_vc": bench_primal_dual_vc_family(),
        "lp_rounding_sc": bench_lp_rounding_sc_family(),
        "fptas_knapsack": bench_fptas_knapsack_family(),
        "local_search_maxcut": bench_local_search_maxcut_family(),
        "christofides_tsp": bench_christofides_tsp_family(),
        "sdp_maxcut": bench_sdp_maxcut_family(),
        "eigenvalue_opt": bench_eigenvalue_opt_family(),
        "sos_certificate": bench_sos_certificate_family(),
        "qcqp_relax": bench_qcqp_relax_family(),
        "spectral_bisection": bench_spectral_bisection_family(),
        "hoffman_bound": bench_hoffman_bound_family(),
        "ski_rental": bench_ski_rental_family(),
        "marking_paging": bench_marking_paging_family(),
        "work_function_kserver": bench_work_function_kserver_family(),
        "ranking_matching": bench_ranking_matching_family(),
        "secretary_prophet": bench_secretary_prophet_family(),
        "online_gradient": bench_online_gradient_family(),
        "two_stage_lshaped": bench_two_stage_lshaped_family(),
        "scenario_tree": bench_scenario_tree_family(),
        "saa_consistency": bench_saa_consistency_family(),
        "chance_scenario": bench_chance_scenario_family(),
        "dro_wasserstein": bench_dro_wasserstein_family(),
        "robust_budget": bench_robust_budget_family(),
        "parallel_tempering": bench_parallel_tempering_family(),
        "wang_landau": bench_wang_landau_family(),
        "umbrella_sampling": bench_umbrella_sampling_family(),
        "metadynamics": bench_metadynamics_family(),
        "wham": bench_wham_family(),
        "thermo_integration": bench_thermo_integration_family(),
        "hinf_filter": bench_hinf_filter_family(),
        "cubature_kalman": bench_cubature_kalman_family(),
        "mhe": bench_mhe_family(),
        "variational_bayes": bench_variational_bayes_family(),
        "huber_filter": bench_huber_filter_family(),
        "particle_smoother": bench_particle_smoother_family(),
        "dupire_localvol": bench_dupire_localvol_family(),
        "sabr_calib": bench_sabr_calib_family(),
        "deep_hedge": bench_deep_hedge_family(),
        "heston_calib": bench_heston_calib_family(),
        "barrier_adjoint": bench_barrier_adjoint_family(),
        "andreasen_huge": bench_andreasen_huge_family(),
        "cdcl_solver": bench_cdcl_solver_family(),
        "walksat": bench_walksat_family(),
        "unit_propagation": bench_unit_propagation_family(),
        "twosat_scc": bench_twosat_scc_family(),
        "bdd_ops": bench_bdd_ops_family(),
        "ltl_mc": bench_ltl_mc_family(),
        "k_induction": bench_k_induction_family(),
        "ic3_pdr": bench_ic3_pdr_family(),
        "bmc_unroll": bench_bmc_unroll_family(),
        "invariant_synth": bench_invariant_synth_family(),
        "hoare_logic": bench_hoare_logic_family(),
        "ranking_function": bench_ranking_function_family(),
        "cegar_loop": bench_cegar_loop_family(),
        "buchberger": bench_buchberger_family(),
        "resultant": bench_resultant_family(),
        "poly_gcd": bench_poly_gcd_family(),
        "gf2_factor": bench_gf2_factor_family(),
        "lll_reduce": bench_lll_reduce_family(),
        "newton_interp": bench_newton_interp_family(),
        "miller_rabin": bench_miller_rabin_family(),
        "pollard_rho": bench_pollard_rho_family(),
        "tonelli_shanks": bench_tonelli_shanks_family(),
        "continued_fraction": bench_continued_fraction_family(),
        "crt_garner": bench_crt_garner_family(),
        "ec_scalar": bench_ec_scalar_family(),
        "paxos": bench_paxos_family(),
        "raft_election": bench_raft_election_family(),
        "vector_clock": bench_vector_clock_family(),
        "consistent_hash": bench_consistent_hash_family(),
        "gossip_epidemic": bench_gossip_epidemic_family(),
        "pbft_lite": bench_pbft_lite_family(),
        "aho_corasick": bench_aho_corasick_family(),
        "suffix_automaton": bench_suffix_automaton_family(),
        "kmp_search": bench_kmp_search_family(),
        "edit_distance": bench_edit_distance_family(),
        "lz77": bench_lz77_family(),
        "bwt_transform": bench_bwt_transform_family(),
        "nbody_leapfrog": bench_nbody_leapfrog_family(),
        "barnes_hut": bench_barnes_hut_family(),
        "sph_fluid": bench_sph_fluid_family(),
        "rigid_collision": bench_rigid_collision_family(),
        "verlet_cloth": bench_verlet_cloth_family(),
        "fem_truss": bench_fem_truss_family(),
        "regex_engine": bench_regex_engine_family(),
        "dfa_minimize": bench_dfa_minimize_family(),
        "cyk_parser": bench_cyk_parser_family(),
        "dominance_tree": bench_dominance_tree_family(),
        "liveness_dce": bench_liveness_dce_family(),
        "linscan_regalloc": bench_linscan_regalloc_family(),
        "huffman_codes": bench_huffman_codes_family(),
        "arithmetic_coding": bench_arithmetic_coding_family(),
        "lzw_compress": bench_lzw_compress_family(),
        "golomb_rice": bench_golomb_rice_family(),
        "rans_coder": bench_rans_coder_family(),
        "lz78_dict": bench_lz78_dict_family(),
        "gcounter": bench_gcounter_family(),
        "pncounter": bench_pncounter_family(),
        "orset": bench_orset_family(),
        "lww_map": bench_lww_map_family(),
        "twopset": bench_twopset_family(),
        "rga_sequence": bench_rga_sequence_family(),
        "bloom_filter": bench_bloom_filter_family(),
        "cuckoo_filter": bench_cuckoo_filter_family(),
        "xor_filter": bench_xor_filter_family(),
        "quotient_filter": bench_quotient_filter_family(),
        "minhash_lsh": bench_minhash_lsh_family(),
        "simhash": bench_simhash_family(),
        "ear_clipping": bench_ear_clipping_family(),
        "sutherland_hodgman": bench_sutherland_hodgman_family(),
        "segment_intersection": bench_segment_intersection_family(),
        "point_in_polygon": bench_point_in_polygon_family(),
        "closest_pair": bench_closest_pair_family(),
        "rotating_calipers": bench_rotating_calipers_family(),
        "cpu_pipeline": bench_cpu_pipeline_family(),
        "cache_sim": bench_cache_sim_family(),
        "branch_predictor": bench_branch_predictor_family(),
        "tomasulo_sim": bench_tomasulo_sim_family(),
        "paging_sim": bench_paging_sim_family(),
        "roofline_model": bench_roofline_model_family(),
        "merkle_tree": bench_merkle_tree_family(),
        "proof_of_work": bench_proof_of_work_family(),
        "utxo_set": bench_utxo_set_family(),
        "difficulty_retarget": bench_difficulty_retarget_family(),
        "fork_resolution": bench_fork_resolution_family(),
        "block_validator": bench_block_validator_family(),
        "ssa_construct": bench_ssa_construct_family(),
        "sccp_const": bench_sccp_const_family(),
        "gvn_elim": bench_gvn_elim_family(),
        "reg_coalesce": bench_reg_coalesce_family(),
        "instr_sched": bench_instr_sched_family(),
        "licm_hoist": bench_licm_hoist_family(),
        "btree_index": bench_btree_index_family(),
        "wal_recovery": bench_wal_recovery_family(),
        "join_algos": bench_join_algos_family(),
        "query_planner": bench_query_planner_family(),
        "mvcc_isolation": bench_mvcc_isolation_family(),
        "lsm_tree": bench_lsm_tree_family(),
        "round_robin_sched": bench_round_robin_sched_family(),
        "cfs_scheduler": bench_cfs_scheduler_family(),
        "demand_paging": bench_demand_paging_family(),
        "deadlock_detect": bench_deadlock_detect_family(),
        "disk_sched": bench_disk_sched_family(),
        "fs_journal": bench_fs_journal_family(),
        "raycaster": bench_raycaster_family(),
        "bresenham_line": bench_bresenham_line_family(),
        "scanline_fill": bench_scanline_fill_family(),
        "zbuffer_render": bench_zbuffer_render_family(),
        "quaternion_slerp": bench_quaternion_slerp_family(),
        "bsp_tree": bench_bsp_tree_family(),
        "mvp_transform": bench_mvp_transform_family(),
        "recursive_descent": bench_recursive_descent_family(),
        "pratt_parser": bench_pratt_parser_family(),
        "earley_parser": bench_earley_parser_family(),
        "slr_parser": bench_slr_parser_family(),
        "peg_packrat": bench_peg_packrat_family(),
        "ll1_table": bench_ll1_table_family(),
        "tcp_aimd": bench_tcp_aimd_family(),
        "sliding_window": bench_sliding_window_family(),
        "token_bucket": bench_token_bucket_family(),
        "rtt_estimator": bench_rtt_estimator_family(),
        "nat_table": bench_nat_table_family(),
        "http2_flow": bench_http2_flow_family(),
        "hm_inference": bench_hm_inference_family(),
        "tree_walk_interp": bench_tree_walk_interp_family(),
        "cps_transform": bench_cps_transform_family(),
        "macro_expand": bench_macro_expand_family(),
        "gc_marksweep": bench_gc_marksweep_family(),
        "simple_types": bench_simple_types_family(),
        "rsa_toy": bench_rsa_toy_family(),
        "winternitz_ots": bench_winternitz_ots_family(),
        "merkle_ots": bench_merkle_ots_family(),
        "blind_sig": bench_blind_sig_family(),
        "zkp_schnorr": bench_zkp_schnorr_family(),
        "commit_reveal": bench_commit_reveal_family(),
        "aries_recovery": bench_aries_recovery_family(),
        "two_phase_lock": bench_two_phase_lock_family(),
        "selinger_join": bench_selinger_join_family(),
        "mvcc_gc": bench_mvcc_gc_family(),
        "buffer_pool": bench_buffer_pool_family(),
        "blink_tree": bench_blink_tree_family(),
        "multi_paxos": bench_multi_paxos_family(),
        "epaxos": bench_epaxos_family(),
        "viewstamped": bench_viewstamped_family(),
        "zab_protocol": bench_zab_protocol_family(),
        "swim_gossip": bench_swim_gossip_family(),
        "two_three_pc": bench_two_three_pc_family(),
    }

    hyps = _build_hypotheses(families, rankers)
    scorecard = _benchmark_scorecard(families)
    n_trials, trial_names, trial_scores = _aligned_ranker_scores(rankers)
    overfitting = overfitting_diagnostics(
        trial_scores,
        n_trials=n_trials,
        names=trial_names or None,
        horizon_bars=_horizon_bars(label),
        embargo_bars=int(config.embargo_bars()),
    )

    synthetic = config.data.source == "synthetic"
    disclaimer = (
        "SYNTHETIC Dipcatcher proprietary research lab. Scores are proper rules (IC, QLIKE, pinball, "
        "CRPS, Brier, Kupiec, CQR/ACI/Mondrian coverage, e-process, CRC, "
        "Jackknife+, weighted CQR, interval caps, bandit regret, Northset). Not a live-P&L claim."
        if synthetic
        else "Dipcatcher proprietary research scores with HAC/DM/Kupiec uncertainty. No live P&L claim."
    )

    notebook = ResearchNotebook(
        schema_version=RESEARCH_RECEIPT_SCHEMA_VERSION,
        firm=__firm__,
        product="Dipcatcher",
        version=__version__,
        generated_at=datetime.now(tz=UTC).isoformat(),
        data_source="SYNTHETIC" if synthetic else str(config.data.source),
        synthetic=synthetic,
        disclaimer=disclaimer,
        ranking_target=label,
        claim="research_only",
        families=families,
        rankers=rankers,
        hypotheses=hyps,
        scorecard=scorecard,
        provenance=provenance,
        backtest_overfitting=overfitting,
    )

    dest_dir = Path(config.data.root) / "metadata" / "research"
    dest_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, Any] = {
        "disclaimer": disclaimer,
        "lab": {
            "firm": __firm__,
            "product": "Dipcatcher",
            "role": "Artificial Hedge proprietary research lab",
            "target": label,
            "source": notebook.data_source,
            "run_id": provenance["run_id"],
        },
        "provenance": provenance,
        "benchmark_scorecard": scorecard,
        "hypotheses": {
            h.id: (
                f"[{h.family}] {h.decision} | {h.test} "
                f"stat={h.statistic:.4g} "
                f"p={format_p_value(h.p_value)}"
                + (
                    f" meets_floor={h.meets_floor}"
                    if h.family == "bound"
                    else f" reject_fdr={h.reject_fdr}"
                )
            )
            for h in hyps
        },
        "ranking": {
            r["name"]: (
                f"set={r['feature_set']} IC={r['mean_ic']:.4f} RankIC={r.get('mean_rank_ic') or 0:.4f} "
                f"t={r['t_ic']:.2f} mono={r.get('decile_monotonicity') or 0:.3f}"
            )
            for r in rankers
            if not str(r.get("name", "")).startswith("_")
        },
        "data_snooping": _data_snooping_section(ranking_blob.get("data_snooping")),
        "backtest_overfitting": _overfitting_section(overfitting),
        "alpha": families["alpha"],
        "volatility": families["volatility"],
        "distribution": families["distribution"],
        "regime": families["regime"],
        "tail": families["tail"],
        "drawdown": families["drawdown"],
        "liquidity": families["liquidity"],
        "reinforcement": families["reinforcement"],
        "conformal": families["conformal"],
        "evalues": families["evalues"],
        "jackknife_plus": families["jackknife_plus"],
        "crc": families["crc"],
        "weighted_conformal": families["weighted_conformal"],
        "interval_risk": families["interval_risk"],
        "quantile_bandit": families["quantile_bandit"],
        "cv_plus": families["cv_plus"],
        "cpcv": families["cpcv"],
        "localized_conformal": families["localized_conformal"],
        "conformal_rank": families["conformal_rank"],
        "online_crc": families["online_crc"],
        "portfolio_conformal": families["portfolio_conformal"],
        "northset": families["northset"],
    }
    md_path = dest_dir / "latest.md"
    run_dir = dest_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(provenance["run_id"])
    run_md_path = run_dir / f"{run_id}.md"
    run_json_path = run_dir / f"{run_id}.json"
    write_report(
        md_path,
        f"Dipcatcher — {__firm__}'s proprietary research lab notebook",
        sections,
        synthetic=synthetic,
    )
    write_report(
        run_md_path,
        f"Dipcatcher — {__firm__}'s proprietary research lab notebook",
        sections,
        synthetic=synthetic,
    )
    write_report(
        latest_report_dir(Path(config.data.root)) / "latest.md",
        f"Dipcatcher — {__firm__}'s proprietary research lab notebook",
        sections,
        synthetic=synthetic,
    )
    notebook.artifacts = {
        # Persist paths relative to the receipt root.  The verifier resolves
        # relative artifacts from the receipt itself, so workspace-relative
        # paths would otherwise be duplicated (e.g. research/data/metadata/...).
        "json": "latest.json",
        "markdown": "latest.md",
        "immutable_json": f"runs/{run_id}.json",
        "immutable_markdown": f"runs/{run_id}.md",
        "immutable_markdown_sha256": hash_file(run_md_path),
    }
    # Bind the JSON receipt without creating a self-referential hash: the
    # digest covers the canonical payload before this digest field is added.
    digest_payload = notebook.to_dict()
    notebook.artifacts["immutable_json_sha256"] = hash_bytes(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    serialized = json.dumps(notebook.to_dict(), indent=2)
    _atomic_write_text(run_json_path, serialized)
    _atomic_write_text(dest_dir / "latest.json", serialized)
    return notebook
