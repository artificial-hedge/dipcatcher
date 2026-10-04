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
from quant_fund.research.benches_w453 import (
    bench_bar_cobar_family,
    bench_deligne_conj_family,
    bench_factor_homology_family,
    bench_hochschild_hom_family,
    bench_operad_koszul_family,
    bench_primitive_elts_family,
)
from quant_fund.research.benches_w454 import (
    bench_decomp_thm_family,
    bench_fourier_sato_family,
    bench_ic_stalk_family,
    bench_middle_ext_family,
    bench_riemann_hilbert_family,
    bench_vanishing_cycles_family,
)
from quant_fund.research.benches_w455 import (
    bench_cubical_path_family,
    bench_glue_types_family,
    bench_hcomp_fill_family,
    bench_interval_obj_family,
    bench_kan_op_family,
    bench_transport_coe_family,
)
from quant_fund.research.benches_w456 import (
    bench_chow_motive_family,
    bench_nori_motive_family,
    bench_num_equiv_family,
    bench_standard_conj_family,
    bench_tate_motive_family,
    bench_voev_motive_family,
)
from quant_fund.research.benches_w457 import (
    bench_companion_conj_family,
    bench_fibrant_double_family,
    bench_framed_bicat_family,
    bench_proarrow_family,
    bench_tabulation_family,
    bench_virtual_equip_family,
)
from quant_fund.research.benches_w458 import (
    bench_delta_matroid_family,
    bench_matroid_minor_family,
    bench_matroid_rep_family,
    bench_regular_mat_family,
    bench_transversal_mat_family,
    bench_tutte_poly_family,
)
from quant_fund.research.benches_w459 import (
    bench_chain_cond_family,
    bench_denotational_family,
    bench_fixed_points_ord_family,
    bench_galois_insertion_family,
    bench_scott_cpo_family,
    bench_way_below_family,
)
from quant_fund.research.benches_w460 import (
    bench_analytic_ring2_family,
    bench_clausen_scholze_family,
    bench_pyknotic_family,
    bench_solid_derived_family,
    bench_solid_tensor_family,
    bench_trace_class_family,
)
from quant_fund.research.benches_w461 import (
    bench_derived_critical_family,
    bench_lagrangian_int_family,
    bench_lie_algebroid_family,
    bench_moment_map_family,
    bench_quant_dag_family,
    bench_shifted_sympl_family,
)
from quant_fund.research.benches_w462 import (
    bench_cartesian_fib2_family,
    bench_cohesive_struct_family,
    bench_descent_cond_family,
    bench_lex_reflect_family,
    bench_n_localic_family,
    bench_shape_theory_family,
)
from quant_fund.research.benches_w463 import (
    bench_arithmetic_dm_family,
    bench_frobenius_dm_family,
    bench_holonomic_dm_family,
    bench_isocrystal_family,
    bench_overconv_dm_family,
    bench_rigid_dm_family,
)
from quant_fund.research.benches_w464 import (
    bench_chromatic_conv_family,
    bench_k_n_local_family,
    bench_morava_e_family,
    bench_nilpotence_dev_family,
    bench_telescopic_family,
    bench_tmf_spectrum_family,
)
from quant_fund.research.benches_w465 import (
    bench_affinoid_alg_family,
    bench_dagger_groth_family,
    bench_gauss_point_family,
    bench_kedlaya_renorm_family,
    bench_raynaud_gen_family,
    bench_weierstrass_prep_family,
)
from quant_fund.research.benches_w466 import (
    bench_kashiwara_schapira_family,
    bench_loc_system_family,
    bench_micro_supp_family,
    bench_perverse_2_family,
    bench_sheaf_homotopy_family,
    bench_stacky_sheaf_family,
)
from quant_fund.research.benches_w467 import (
    bench_constructible_l_family,
    bench_core_model_family,
    bench_large_card_family,
    bench_pcf_theory_family,
    bench_proper_forcing_family,
    bench_square_princ_family,
)
from quant_fund.research.benches_w468 import (
    bench_abstract_erc_family,
    bench_nip_theory_family,
    bench_nonforking_family,
    bench_o_minimal_family,
    bench_simple_theory_family,
    bench_tame_metric_family,
)
from quant_fund.research.benches_w469 import (
    bench_accessible_cat_family,
    bench_day_conv_family,
    bench_derivator2_family,
    bench_enriched_cat_family,
    bench_fibered_cat_family,
    bench_weight_lim_family,
)
from quant_fund.research.benches_w470 import (
    bench_e_infty2_family,
    bench_h_space_family,
    bench_james_constr_family,
    bench_obstruction_th_family,
    bench_power_op_family,
    bench_rational_htpy_family,
)
from quant_fund.research.benches_w471 import (
    bench_banach_colmez_family,
    bench_breuil_kisin_family,
    bench_diamond_geo_family,
    bench_drinfeld_tower_family,
    bench_integral_padic_family,
    bench_perfectoid2_family,
)
from quant_fund.research.benches_w472 import (
    bench_alg_cobordism_family,
    bench_hermitian_k_family,
    bench_motivic_stem2_family,
    bench_oriented_coh_family,
    bench_rostmotive_family,
    bench_slice_spec_family,
)
from quant_fund.research.benches_w473 import (
    bench_cartier_mod_family,
    bench_crystalline_stack_family,
    bench_cyclotomic2_family,
    bench_thh_2_family,
    bench_trt_functor_family,
    bench_witt_vec2_family,
)
from quant_fund.research.benches_w474 import (
    bench_cobordism_hyp_family,
    bench_heegaard_floer_family,
    bench_khovanov_family,
    bench_modular_cat_family,
    bench_reshet_turaev_family,
    bench_topological_order_family,
)
from quant_fund.research.benches_w475 import (
    bench_chiral_alg_family,
    bench_cyclic_hk_family,
    bench_dendroidal_family,
    bench_infty_operad_family,
    bench_seq_spectra_family,
    bench_sifted_cat_family,
)
from quant_fund.research.benches_w476 import (
    bench_adams_novikov_family,
    bench_bp_spectrum_family,
    bench_greek_letter_family,
    bench_landweber_exact_family,
    bench_picard_grp_family,
    bench_smith_toda_family,
)
from quant_fund.research.benches_w477 import (
    bench_arithmetic_ht_family,
    bench_berthelo_crys_family,
    bench_berthelot_rigid_family,
    bench_caro_dm_family,
    bench_dagger_dm_family,
    bench_spencer_dm_family,
)
from quant_fund.research.benches_w478 import (
    bench_beilinson_con_family,
    bench_cellular_motive_family,
    bench_levine_morel_family,
    bench_mgl_spec_family,
    bench_motivic_pi0_family,
    bench_quadratic_k_family,
)
from quant_fund.research.benches_w479 import (
    bench_ahb_ring_family,
    bench_drinfeld_sym_family,
    bench_fargues_diam_family,
    bench_prism_2_family,
    bench_scholze_diamond_family,
    bench_tilting_equiv_family,
)
from quant_fund.research.benches_w480 import (
    bench_classify_obj_family,
    bench_etale_geom_family,
    bench_exponentiable_family,
    bench_gros_topos_family,
    bench_local_homeo_family,
    bench_pi_infty_family,
)
from quant_fund.research.benches_w481 import (
    bench_fqmotive_family,
    bench_higher_chow2_family,
    bench_motivic_chern_family,
    bench_motivic_landin_family,
    bench_mtc_motive_family,
    bench_triang_motive_family,
)
from quant_fund.research.benches_w482 import (
    bench_blue_shift_family,
    bench_chromatic_fracture_family,
    bench_fgsl_group_family,
    bench_morava_stabilizer_family,
    bench_red_shift_family,
    bench_tate_spec_family,
)
from quant_fund.research.benches_w483 import (
    bench_bar_spec_family,
    bench_dyer_lashof_family,
    bench_free_loop_family,
    bench_loop_functor_family,
    bench_steenrod_ops_family,
    bench_sullivan_min_family,
)
from quant_fund.research.benches_w484 import (
    bench_formal_model_family,
    bench_internal_univ_family,
    bench_stein_space_family,
    bench_synth_stable_family,
    bench_univalent_found_family,
    bench_virtual_hodge_family,
)
from quant_fund.research.benches_w485 import (
    bench_banach_colmez2_family,
    bench_bc_space_family,
    bench_fargues_curve2_family,
    bench_local_shimura_family,
    bench_lubin_tate2_family,
    bench_scholze_weinstein_family,
)
from quant_fund.research.benches_w486 import (
    bench_karoubi_v_family,
    bench_kv_theory_family,
    bench_nk_theory_family,
    bench_plus_k_family,
    bench_vorst_stab_family,
    bench_waldhausen_k_family,
)
from quant_fund.research.benches_w487 import (
    bench_comma_cat_family,
    bench_compact_obj_family,
    bench_dualizable_cat_family,
    bench_endo_prof_family,
    bench_exact_cat_family,
    bench_prestack_family,
)
from quant_fund.research.benches_w488 import (
    bench_d_critical_family,
    bench_derived_quot_family,
    bench_intrinsic_be_family,
    bench_perfect_obstruction_family,
    bench_shifted_tangent_family,
    bench_virtual_pull_family,
)
from quant_fund.research.benches_w489 import (
    bench_bn_pair_family,
    bench_braid_grp_family,
    bench_building_toy_family,
    bench_coxeter_grp_family,
    bench_hecke_bm_family,
    bench_parabolic_grp_family,
)
from quant_fund.research.benches_w490 import (
    bench_adelic_curve_family,
    bench_arakelov_deg_family,
    bench_arith_rr_family,
    bench_arithmetic_chow_family,
    bench_faltings_metric_family,
    bench_height_arakelov_family,
)
from quant_fund.research.benches_w491 import (
    bench_arthur_param_family,
    bench_hecke_alg2_family,
    bench_l_function_family,
    bench_satake_param_family,
    bench_shimura_var_family,
    bench_theta_lift_family,
)
from quant_fund.research.benches_w492 import (
    bench_kato_fontaine_family,
    bench_log_crystalline_family,
    bench_log_derham_family,
    bench_log_etale_family,
    bench_log_smooth_family,
    bench_log_structure_family,
)
from quant_fund.research.benches_w493 import (
    bench_berkovich_an_family,
    bench_mikhalkin_family,
    bench_skeleton_trop_family,
    bench_tropical_curve_family,
    bench_tropical_cycle_family,
    bench_tropical_poly_family,
)
from quant_fund.research.benches_w494 import (
    bench_calabi_yau_alg_family,
    bench_connes_nc_family,
    bench_cyclic_coh_family,
    bench_ginzburg_dga_family,
    bench_hochschild_coh_family,
    bench_nc_scheme_family,
)
from quant_fund.research.benches_w495 import (
    bench_complicial_family,
    bench_globular_model_family,
    bench_opetopic_family,
    bench_theta_space_family,
    bench_verity_gray_family,
    bench_weak_infty_family,
)
from quant_fund.research.benches_w496 import (
    bench_dag_representation_family,
    bench_derived_deformation_family,
    bench_derived_moduli_family,
    bench_formal_deformation_family,
    bench_obstruction_2_family,
    bench_tangent_coh_family,
)
from quant_fund.research.benches_w497 import (
    bench_fano_mori_family,
    bench_flip_cone_family,
    bench_klt_pair_family,
    bench_minimal_model_family,
    bench_mmp_algorithm_family,
    bench_toric_flip_family,
)
from quant_fund.research.benches_w498 import (
    bench_gromov_witten_family,
    bench_hilbert_scheme2_family,
    bench_kuranishi_family,
    bench_m_bar_gn_family,
    bench_quot_scheme_family,
    bench_stable_map_family,
)
from quant_fund.research.benches_w499 import (
    bench_hodge_decomp_family,
    bench_l2_hodge_family,
    bench_limit_mhs_family,
    bench_mixed_hodge_family,
    bench_period_map_family,
    bench_vhs_polarized_family,
)
from quant_fund.research.benches_w500 import (
    bench_epsilon_factor_family,
    bench_harris_taylor_family,
    bench_l_packet_family,
    bench_langlands_functoriality_family,
    bench_local_langlands_family,
    bench_weil_group_family,
)
from quant_fund.research.benches_w501 import (
    bench_f_pure_family,
    bench_f_rational_family,
    bench_f_regular_family,
    bench_f_threshold_family,
    bench_test_ideal_family,
    bench_tight_closure_family,
)
from quant_fund.research.benches_w502 import (
    bench_dg_cat2_family,
    bench_dg_morita_family,
    bench_dg_nerve_family,
    bench_dg_quotient_family,
    bench_drinfeld_quotient_family,
    bench_keller_dg_family,
)
from quant_fund.research.benches_w503 import (
    bench_elliptic_surface_family,
    bench_kodaira_fiber_family,
    bench_mordell_weil2_family,
    bench_neron_model_family,
    bench_tate_algorithm_family,
    bench_weierstrass_eq_family,
)
from quant_fund.research.benches_w504 import (
    bench_adem_relations_family,
    bench_bar_resolution_family,
    bench_lambda_algebra_family,
    bench_serre_cartan_family,
    bench_steenrod_algebra_family,
    bench_unstable_modules_family,
)
from quant_fund.research.benches_w505 import (
    bench_cobordism_grp_family,
    bench_complex_cob_family,
    bench_framed_cob_family,
    bench_oriented_cob_family,
    bench_thom_cob_family,
    bench_unoriented_cob_family,
)
from quant_fund.research.benches_w506 import (
    bench_distality_family,
    bench_dp_rank_family,
    bench_forking_seq_family,
    bench_honest_def_family,
    bench_nip_formula_family,
    bench_uniform_def_family,
)
from quant_fund.research.benches_w507 import (
    bench_homotopy_coherent_family,
    bench_htc_colimit_family,
    bench_joyal_model_family,
    bench_marking_qcat_family,
    bench_nerve_quasi_family,
    bench_quasi_cat_family,
)
from quant_fund.research.benches_w508 import (
    bench_anabelian_geo_family,
    bench_etale_pi1_family,
    bench_fundamental_grp_family,
    bench_groth_tei_family,
    bench_section_conj_family,
    bench_tamagawa_mochi_family,
)
from quant_fund.research.benches_w509 import (
    bench_deligne_weil2_family,
    bench_etale_site2_family,
    bench_frobenius_action_family,
    bench_groth_lefschetz_family,
    bench_l_adic_sheaf_family,
    bench_purity_thm_family,
)
from quant_fund.research.benches_w510 import (
    bench_connective_e_ring_family,
    bench_elliptic_cohor_family,
    bench_spectral_alg_family,
    bench_spectral_scheme2_family,
    bench_spectral_stack_family,
    bench_taf_lurie_family,
)
from quant_fund.research.benches_w511 import (
    bench_bun_g_family,
    bench_fs_diamond_family,
    bench_geometric_satake_family,
    bench_hecke_stack_family,
    bench_v_sheaf_family,
    bench_y_diamond_family,
)
from quant_fund.research.benches_w512 import (
    bench_berezin_int_family,
    bench_odd_variables_family,
    bench_super_lie_family,
    bench_super_manifold_family,
    bench_super_scheme_family,
    bench_super_space_family,
)
from quant_fund.research.benches_w513 import (
    bench_categorify_family,
    bench_hecke_cat_family,
    bench_khovanov_hom_family,
    bench_rasmussen_inv_family,
    bench_soergel_bim_family,
    bench_uq_sl2_family,
)
from quant_fund.research.benches_w514 import (
    bench_beilinson_reg_family,
    bench_cheeger_simons_family,
    bench_deligne_cohom_family,
    bench_diff_cohom_family,
    bench_flat_bundle_family,
    bench_secondary_inv_family,
)
from quant_fund.research.benches_w515 import (
    bench_auslander_buchs_family,
    bench_betti_series_family,
    bench_green_koszul_family,
    bench_minimal_free_family,
    bench_quillen_suslin_family,
    bench_serre_conj_family,
)
from quant_fund.research.benches_w516 import (
    bench_affine_lie_family,
    bench_kac_moody_family,
    bench_moonshine_module_family,
    bench_vertex_alg_family,
    bench_weyl_kac_family,
    bench_zhu_algebra_family,
)
from quant_fund.research.benches_w517 import (
    bench_crystal_base_family,
    bench_jimbo_drin_family,
    bench_lusztig_can_family,
    bench_quantum_group_family,
    bench_quantum_rmatrix_family,
    bench_quantum_schur_family,
)
from quant_fund.research.benches_w518 import (
    bench_braid_rep_family,
    bench_quantum_double_family,
    bench_ribbon_cat_family,
    bench_rtt_formalism_family,
    bench_yang_baxter_family,
    bench_yangian_family,
)
from quant_fund.research.benches_w519 import (
    bench_cusp_form_family,
    bench_dedekind_eta_family,
    bench_eisenstein_srs2_family,
    bench_hecke_op2_family,
    bench_modular_form_family,
    bench_theta_func_family,
)
from quant_fund.research.benches_w520 import (
    bench_converse_thm_family,
    bench_gln_automorphic_family,
    bench_godement_jacq_family,
    bench_langlands_lfunc_family,
    bench_rankin_selberg_family,
    bench_whittaker_model_family,
)
from quant_fund.research.benches_w521 import (
    bench_chebyshev_bias_family,
    bench_dirichlet_l_family,
    bench_explicit_formula_family,
    bench_linnik_thm_family,
    bench_riemann_zeta_family,
    bench_zero_density_family,
)
from quant_fund.research.benches_w522 import (
    bench_freiman_thm_family,
    bench_gowers_norm_family,
    bench_green_tao_family,
    bench_plunnecke_family,
    bench_roth_thm_family,
    bench_szemeredi_family,
)
from quant_fund.research.benches_w523 import (
    bench_erdos_distinct_family,
    bench_ff_kakeya_family,
    bench_guth_katz_family,
    bench_joints_thm_family,
    bench_kakeya_family,
    bench_sz_trotter_family,
)
from quant_fund.research.benches_w524 import (
    bench_furstenberg_family,
    bench_gallai_thm_family,
    bench_hales_jewett_family,
    bench_hindman_family,
    bench_rado_thm_family,
    bench_schur_thm_family,
)
from quant_fund.research.benches_w525 import (
    bench_bernoulli_shift_family,
    bench_birkhoff_family,
    bench_entropy_ks_family,
    bench_mean_ergodic_family,
    bench_mixing_weak_family,
    bench_osceledets_family,
)
from quant_fund.research.benches_w526 import (
    bench_douady_hubbard_family,
    bench_fatou_set_family,
    bench_julia_set_family,
    bench_mandelbrot_set_family,
    bench_parabolic_impl_family,
    bench_sullivan_no_wander_family,
)
from quant_fund.research.benches_w527 import (
    bench_anosov_family,
    bench_bowen_spec_family,
    bench_horseshoe_family,
    bench_markov_partition_family,
    bench_srb_measure_family,
    bench_stable_mfld_family,
)
from quant_fund.research.benches_w528 import (
    bench_equilibrium_state_family,
    bench_lasota_yorke_family,
    bench_pressure_thm_family,
    bench_ruelle_zeta_family,
    bench_thermo_formal_family,
    bench_transfer_op_family,
)
from quant_fund.research.benches_w529 import (
    bench_arnold_diff_family,
    bench_aubry_mather_family,
    bench_cantorus_family,
    bench_greene_crit_family,
    bench_kam_theorem_family,
    bench_twist_map_family,
)
from quant_fund.research.benches_w530 import (
    bench_dominated_split_family,
    bench_katok_horseshoe_family,
    bench_lyapunov_chart_family,
    bench_nonuniform_hyp_family,
    bench_osceledets_reg_family,
    bench_pesin_theory_family,
)
from quant_fund.research.benches_w531 import (
    bench_bogdanov_takens_family,
    bench_homoclinic_bif_family,
    bench_hopf_bif_family,
    bench_neimark_sacker_family,
    bench_period_doubling_family,
    bench_saddle_node_family,
)
from quant_fund.research.benches_w532 import (
    bench_box_counting_family,
    bench_frostman_family,
    bench_hausdorff_dim_family,
    bench_iterated_function_family,
    bench_multifractal_formal_family,
    bench_self_similar_family,
)
from quant_fund.research.benches_w533 import (
    bench_besicovitch_family,
    bench_density_thm_family,
    bench_marstrand_family,
    bench_preiss_rect_family,
    bench_rectifiability_family,
    bench_tangent_measure_family,
)
from quant_fund.research.benches_w534 import (
    bench_balayage_family,
    bench_capacity_theory_family,
    bench_fine_topology_family,
    bench_green_fn_family,
    bench_harmonic_fn_family,
    bench_potential_thy_family,
)
from quant_fund.research.benches_w535 import (
    bench_elliptic_est_family,
    bench_fourier_io_family,
    bench_propagation_sing_family,
    bench_pseudodiff_op_family,
    bench_symbol_calc_family,
    bench_wavefront_set_family,
)
from quant_fund.research.benches_w536 import (
    bench_contact_geom_family,
    bench_gromov_nonsq_family,
    bench_hamiltonian_flow_family,
    bench_lagrangian_mfd_family,
    bench_poisson_bracket_family,
    bench_symplectic_form_family,
)
from quant_fund.research.benches_w537 import (
    bench_comparison_thm_family,
    bench_jacobi_field_family,
    bench_levi_civita_family,
    bench_ricci_scalar_family,
    bench_riemann_curvature_family,
    bench_riemann_metric_family,
)
from quant_fund.research.benches_w538 import (
    bench_degiorgi_nash_family,
    bench_harnack_thm_family,
    bench_poincare_ineq_family,
    bench_schauder_est_family,
    bench_sobolev_space_family,
    bench_trace_thm_family,
)
from quant_fund.research.benches_w539 import (
    bench_continued_frac2_family,
    bench_dirichlet_approx_family,
    bench_kronecker_thm_family,
    bench_liouville_number_family,
    bench_roth_thm2_family,
    bench_subspace_thm_family,
)
from quant_fund.research.benches_w540 import (
    bench_baker_thm_family,
    bench_gelfond_schneider_family,
    bench_hermite_lindemann_family,
    bench_lindemann_weier_family,
    bench_schanuel_conj_family,
    bench_siegel_shidlovskii_family,
)
from quant_fund.research.benches_w541 import (
    bench_jensen_formula_family,
    bench_montel_normal_family,
    bench_picard_thm_family,
    bench_riemann_mapping_family,
    bench_runge_approx_family,
    bench_schwarz_lemma_family,
)
from quant_fund.research.benches_w542 import (
    bench_d_bar_neumann_family,
    bench_domain_holo_family,
    bench_hartogs_thm_family,
    bench_levi_problem_family,
    bench_oka_coherence_family,
    bench_pseudoconvex_family,
)
from quant_fund.research.benches_w543 import (
    bench_abel_jacobi_family,
    bench_branched_cover_family,
    bench_fuchsian_group_family,
    bench_riemann_hurwitz_family,
    bench_riemann_surface_family,
    bench_teichmuller_space_family,
)
from quant_fund.research.benches_w544 import (
    bench_hyperbolic_3mfd_family,
    bench_jorgensen_thurston_family,
    bench_kleinian_group_family,
    bench_limit_set_family,
    bench_mostow_rigidity_family,
    bench_tameness_thm_family,
)
from quant_fund.research.benches_w545 import (
    bench_eight_geometries_family,
    bench_haken_mfd_family,
    bench_jsj_decomp_family,
    bench_ricci_flow_family,
    bench_seifert_fibered_family,
    bench_thurston_geometrization_family,
)
from quant_fund.research.benches_w546 import (
    bench_alexander_poly_family,
    bench_jones_poly_family,
    bench_knot_group_family,
    bench_knot_invariant_family,
    bench_knot_signature_family,
    bench_vassiliev_inv_family,
)
from quant_fund.research.benches_w547 import (
    bench_donaldson_thm_family,
    bench_exotic_r4_family,
    bench_four_mfd_family,
    bench_freedman_thm_family,
    bench_intersection_form_family,
    bench_seiberg_witten_family,
)
from quant_fund.research.benches_w548 import (
    bench_floer_homology_family,
    bench_fukaya_cat_family,
    bench_instanton_floer_family,
    bench_knot_floer_family,
    bench_lagrangian_floer_family,
    bench_monopole_floer_family,
)
from quant_fund.research.benches_w549 import (
    bench_analytic_torsion_family,
    bench_atiyah_singer_family,
    bench_dirac_op_family,
    bench_eta_invariant_family,
    bench_heat_kernel2_family,
    bench_signature_op_family,
)
from quant_fund.research.benches_w550 import (
    bench_chern_character_family,
    bench_chern_class_family,
    bench_euler_class_family,
    bench_hirzebruch_sig_family,
    bench_pontryagin_class_family,
    bench_todd_genus_family,
)
from quant_fund.research.benches_w551 import (
    bench_foliation_family,
    bench_godbillon_vey_family,
    bench_haefliger_struct_family,
    bench_holonomy_grp_family,
    bench_novikov_thm_family,
    bench_thurston_fol_family,
)
from quant_fund.research.benches_w552 import (
    bench_contact_form_family,
    bench_convex_surface_family,
    bench_giroux_corr_family,
    bench_legendrian_knot_family,
    bench_overtwisted_family,
    bench_tight_contact_family,
)
from quant_fund.research.benches_w553 import (
    bench_frobenius_mfd_family,
    bench_givental_j_family,
    bench_mirror_symmetry_family,
    bench_quantum_cohomology_family,
    bench_quintic_invariants_family,
    bench_toric_mirror_family,
)
from quant_fund.research.benches_w554 import (
    bench_donaldson_thomas_family,
    bench_gopakumar_vafa_family,
    bench_gw_descendant_family,
    bench_kontsevich_mgn_family,
    bench_mnop_conj_family,
    bench_pandharipande_thomas_family,
)
from quant_fund.research.benches_w555 import (
    bench_hms_conjecture_family,
    bench_landau_ginzburg_family,
    bench_mirror_functor_family,
    bench_syz_mirror_family,
    bench_torus_fibration_family,
    bench_wrapped_fukaya_family,
)
from quant_fund.research.benches_w556 import (
    bench_earthquake_map_family,
    bench_extremal_length_family,
    bench_mapping_class_family,
    bench_pseudo_anosov_family,
    bench_quadratic_diff_family,
    bench_weil_petersson_family,
)
from quant_fund.research.benches_w557 import (
    bench_dehn_surgery_family,
    bench_heegaard_splitting_family,
    bench_normal_surface_family,
    bench_sutured_mfd_family,
    bench_taut_foliation_family,
    bench_thin_position_family,
)
from quant_fund.research.benches_w558 import (
    bench_anti_self_dual_family,
    bench_higgs_bundle_family,
    bench_instanton_moduli_family,
    bench_kapustin_witten_family,
    bench_nahm_transform_family,
    bench_yang_mills_family,
)
from quant_fund.research.benches_w559 import (
    bench_calabi_conjecture_family,
    bench_calabi_yau_mfd_family,
    bench_csck_metric_family,
    bench_futaki_invariant_family,
    bench_k_stability_family,
    bench_kahler_einstein_family,
)
from quant_fund.research.benches_w560 import (
    bench_ancient_solution_family,
    bench_hamilton_ricci_family,
    bench_kahler_ricci_flow_family,
    bench_mean_curvature_flow_family,
    bench_perelman_entropy_family,
    bench_ricci_soliton_family,
)
from quant_fund.research.benches_w561 import (
    bench_almgren_pitts_family,
    bench_brakke_flow_family,
    bench_minimal_surface_family,
    bench_plateau_problem_family,
    bench_simon_regularity_family,
    bench_stable_minimal_family,
)
from quant_fund.research.benches_w562 import (
    bench_ekeland_hofer_family,
    bench_gromov_width_family,
    bench_hofer_metric_family,
    bench_mcduff_polterovich_family,
    bench_symplectic_capacity_family,
    bench_symplectic_packing_family,
)
from quant_fund.research.benches_w563 import (
    bench_bubbling_hm_family,
    bench_eells_sampson_family,
    bench_harmonic_map_family,
    bench_heat_flow_hm_family,
    bench_sacks_uhlenbeck_family,
    bench_schoen_uhlenbeck_family,
)
from quant_fund.research.benches_w564 import (
    bench_asymptotic_cone_family,
    bench_baumslag_solitar_family,
    bench_gromov_hyperbolic_family,
    bench_quasi_isometry_family,
    bench_thin_triangle_family,
    bench_word_problem_family,
)
from quant_fund.research.benches_w565 import (
    bench_bhargava_lic_family,
    bench_cohen_lenstra_family,
    bench_elliptic_rank_family,
    bench_malle_conj_family,
    bench_prime_gaps_family,
    bench_zhang_maynard_family,
)
from quant_fund.research.benches_w566 import (
    bench_gue_statistics_family,
    bench_keating_snaith_family,
    bench_montgomery_pair_family,
    bench_rudnick_sarnak_family,
    bench_selberg_trace2_family,
    bench_zero_spacing_family,
)
from quant_fund.research.benches_w567 import (
    bench_berenstein_zelevinsky_family,
    bench_honeycomb_tiling_family,
    bench_knuth_rsk_family,
    bench_littlewood_richardson_family,
    bench_macdonald_poly_family,
    bench_schubert_calc_family,
)
from quant_fund.research.benches_w568 import (
    bench_contact_homology3_family,
    bench_eliashberg_givental_family,
    bench_floer_homol_family,
    bench_reeb_orbit_family,
    bench_sft_algebra_family,
    bench_symplectic_field_family,
)
from quant_fund.research.benches_w569 import (
    bench_aubin_thm_family,
    bench_kazdan_warner_family,
    bench_nirenberg_problem_family,
    bench_prescribed_curvature_family,
    bench_trudinger_thm_family,
    bench_yamabe_problem_family,
)
from quant_fund.research.benches_w570 import (
    bench_bogomolov_ineq_family,
    bench_boundedness_moduli_family,
    bench_hodge_index_family,
    bench_kodaira_vanishing_family,
    bench_kollar_mori_family,
    bench_stability_sheaf_family,
)
from quant_fund.research.benches_w571 import (
    bench_bernstein_sato_family,
    bench_du_val_sing_family,
    bench_log_canonical_family,
    bench_milnor_fiber_family,
    bench_multiplier_ideal_family,
    bench_rational_sing_family,
)
from quant_fund.research.benches_w572 import (
    bench_abelian_variety_family,
    bench_faltings_thm_family,
    bench_isogeny_av_family,
    bench_mordell_weil_av_family,
    bench_shafarevich_conj_family,
    bench_tate_module_family,
)
from quant_fund.research.benches_w573 import (
    bench_absolute_hodge_family,
    bench_griffiths_transv_family,
    bench_hodge_class_family,
    bench_hodge_conj_family,
    bench_mumford_tate_family,
    bench_period_domain_family,
)
from quant_fund.research.benches_w574 import (
    bench_exotic_sphere_family,
    bench_immersion_thm_family,
    bench_kervaire_milnor_family,
    bench_smale_hcob_family,
    bench_surgery_theory_family,
    bench_whitney_trick_family,
)
from quant_fund.research.benches_w575 import (
    bench_airy_process_family,
    bench_beta_ensemble_family,
    bench_circular_law_family,
    bench_dyson_brownian_family,
    bench_sine_kernel_family,
    bench_tracy_widom_family,
)
from quant_fund.research.benches_w576 import (
    bench_free_convolution_family,
    bench_free_prob_family,
    bench_operator_valued_family,
    bench_r_transform_family,
    bench_s_transform_family,
    bench_voiculescu_thm_family,
)
from quant_fund.research.benches_w577 import (
    bench_char_cycle_family,
    bench_d_module2_family,
    bench_intersection_homology_family,
    bench_middle_perversity_family,
    bench_nearby_cycles_family,
    bench_perverse_sheaf_family,
)
from quant_fund.research.benches_w578 import (
    bench_bogomolov_conj_family,
    bench_canonical_height_family,
    bench_equidistribution_thm_family,
    bench_global_height_family,
    bench_nevanlinna_th_family,
    bench_vojta_conj_family,
)
from quant_fund.research.benches_w579 import (
    bench_git_quotient_family,
    bench_hilbert_mumford_family,
    bench_kirwan_strat_family,
    bench_luna_slice_family,
    bench_moment_polytope_family,
    bench_symplectic_quot_family,
)
from quant_fund.research.benches_w580 import (
    bench_cycle_index_family,
    bench_exponential_gf_family,
    bench_lagrange_inversion_family,
    bench_matrix_tree_family,
    bench_species_theory_family,
    bench_transfer_matrix_family,
)
from quant_fund.research.benches_w581 import (
    bench_bockstein_ss_family,
    bench_bousfield_ss_family,
    bench_cartan_ss_family,
    bench_eilenberg_moore_family,
    bench_lyndon_ss_family,
    bench_serre_ss4_family,
)
from quant_fund.research.benches_w582 import (
    bench_arinkin_gaitsgory_family,
    bench_derived_satake_family,
    bench_fusion_product_family,
    bench_geometric_satake2_family,
    bench_nilp_cone_family,
    bench_spectral_bung_family,
)
from quant_fund.research.benches_w583 import (
    bench_fulton_mclarty_family,
    bench_motivic_base_change_family,
    bench_motivic_homotopy2_family,
    bench_motivic_proper_family,
    bench_motivic_smooth_family,
    bench_six_op_motivic_family,
)
from quant_fund.research.benches_w584 import (
    bench_analytic_sheaf_family,
    bench_clausen_scholze2_family,
    bench_nuclear_space_family,
    bench_proetale_site2_family,
    bench_solid_cohom_family,
    bench_solid_tensor2_family,
)
from quant_fund.research.benches_w585 import (
    bench_canonical_bundle_family,
    bench_intersection_theory_family,
    bench_line_bundle_family,
    bench_macpherson_chern_family,
    bench_picard_group_family,
    bench_weil_divisor_family,
)
from quant_fund.research.benches_w586 import (
    bench_dualizing_cmplx_family,
    bench_dualizing_sheaf_family,
    bench_groth_duality_family,
    bench_relative_duality_family,
    bench_residue_thm_family,
    bench_verdier_duality_family,
)
from quant_fund.research.benches_w587 import (
    bench_abundance_conj_family,
    bench_bdd_fano_family,
    bench_canonical_sing2_family,
    bench_klt_mmp_family,
    bench_mmp_flip_family,
    bench_terminal_sing_family,
)
from quant_fund.research.benches_w588 import (
    bench_ehp_sequence_family,
    bench_freudenthal_susp_family,
    bench_james_period_family,
    bench_moore_space_family,
    bench_unstable_adams_family,
    bench_whitehead_prod_family,
)
from quant_fund.research.benches_w589 import (
    bench_cartesian_closed_family,
    bench_coherent_topos_family,
    bench_internal_logic_family,
    bench_power_object_family,
    bench_pretopos_family,
    bench_subobject_lattice_family,
)
from quant_fund.research.benches_w590 import (
    bench_bloch_k_family,
    bench_gersten_ss_family,
    bench_loday_k_family,
    bench_quillen_plus_family,
    bench_suslin_k_family,
    bench_volodin_k_family,
)
from quant_fund.research.benches_w591 import (
    bench_friedlander_voev_family,
    bench_motivic_descent_family,
    bench_motivic_eilenberg_family,
    bench_motivic_invert_family,
    bench_motivic_purity_family,
    bench_motivic_zeta_family,
)
from quant_fund.research.benches_w592 import (
    bench_braided_cat_family,
    bench_fusion_cat_family,
    bench_premodular_family,
    bench_rigid_cat_family,
    bench_spherical_cat_family,
    bench_tensor_cat_family,
)
from quant_fund.research.benches_w593 import (
    bench_harmonic_bdl_family,
    bench_higgs_bundle2_family,
    bench_hitchin_section_family,
    bench_hodge_moduli_family,
    bench_nonabelian_hodge_family,
    bench_simpson_corr_family,
)
from quant_fund.research.benches_w594 import (
    bench_artin_neighborhood_family,
    bench_etale_fund_family,
    bench_etale_homotopy_family,
    bench_galois_cat_family,
    bench_pro_etale_family,
    bench_shapiro_lemma_family,
)
from quant_fund.research.benches_w595 import (
    bench_conjugate_fil_family,
    bench_crys_cohom_family,
    bench_divided_power_family,
    bench_nygaard_filt_family,
    bench_pd_envelope_family,
    bench_syntomic_family,
)
from quant_fund.research.benches_w596 import (
    bench_breuil_mod_family,
    bench_etale_phi_family,
    bench_finite_height_family,
    bench_galois_lattice_family,
    bench_kisin_mod_family,
    bench_padic_hodge_family,
)
from quant_fund.research.benches_w597 import (
    bench_bousfield_kan_family,
    bench_curtis_lower_family,
    bench_dror_smith_family,
    bench_lannes_t_family,
    bench_periodicity_thm_family,
    bench_telescope_conj_family,
)
from quant_fund.research.benches_w598 import (
    bench_bloch_beilinson_family,
    bench_borel_regulator_family,
    bench_etale_ktheory_family,
    bench_lichtenbaum_k_family,
    bench_soul_elem_family,
    bench_thh_trace_family,
)
from quant_fund.research.benches_w599 import (
    bench_algebra_cat_family,
    bench_codensity_monad_family,
    bench_distributive_law_family,
    bench_klesli_cat_family,
    bench_monad_theorem_family,
    bench_monadicity_family,
)
from quant_fund.research.benches_w600 import (
    bench_a_infty_alg_family,
    bench_koszul_duality_family,
    bench_l_infty_alg_family,
    bench_minimal_model_op_family,
    bench_operad_cobar_family,
    bench_operadic_bar_family,
)
from quant_fund.research.benches_w601 import (
    bench_derived_loop_family,
    bench_derived_tangent_family,
    bench_dg_algebra_family,
    bench_e_infinity_ring_family,
    bench_structured_space_family,
    bench_virtual_fund_family,
)
from quant_fund.research.benches_w602 import (
    bench_cyclotomic_spec_family,
    bench_negative_cyclic_family,
    bench_periodic_cyclic_family,
    bench_tate_construction_family,
    bench_tc_spec_family,
    bench_tr_structure_family,
)
from quant_fund.research.benches_w603 import (
    bench_atomic_topos_family,
    bench_classifying_topos_family,
    bench_essential_morph_family,
    bench_giraud_axiom_family,
    bench_logical_morph_family,
    bench_slice_topos_family,
)
from quant_fund.research.benches_w604 import (
    bench_bokstedt_periodicity_family,
    bench_elliptic_k_family,
    bench_equivariant_cohomology2_family,
    bench_may_ss_family,
    bench_topo_k_theory_family,
    bench_unstable_cohomology_family,
)
from quant_fund.research.benches_w605 import (
    bench_dk_motive_family,
    bench_motivic_adem_family,
    bench_motivic_steenrod_family,
    bench_motivic_transfer_family,
    bench_power_operations_family,
    bench_simplicial_motive_family,
)
from quant_fund.research.benches_w606 import (
    bench_connective_k_family,
    bench_higher_k_family,
    bench_k_spectrum_family,
    bench_karoubi_k_family,
    bench_nil_k_family,
    bench_pedersen_weibel_family,
)
from quant_fund.research.benches_w607 import (
    bench_cocartesian_family,
    bench_homotopy_cat_family,
    bench_horn_filler_family,
    bench_kan_complex_family,
    bench_mapping_space_family,
    bench_nerve_cat_family,
)
from quant_fund.research.benches_w608 import (
    bench_etale_descent_family,
    bench_etale_morphism_family,
    bench_fppf_site_family,
    bench_fpqc_site_family,
    bench_ladic_sheaf_family,
    bench_lisse_sheaf_family,
)
from quant_fund.research.benches_w609 import (
    bench_brave_new_ring_family,
    bench_e_infty_space_family,
    bench_formal_moduli_family,
    bench_log_ring_family,
    bench_orient_cohom_family,
    bench_thom_constr_family,
)
from quant_fund.research.benches_w610 import (
    bench_first_order_family,
    bench_obstruction_def_family,
    bench_prorepresent_family,
    bench_schlessinger2_family,
    bench_semiuniversal_family,
    bench_versal_def_family,
)
from quant_fund.research.benches_w611 import (
    bench_mate_dual_family,
    bench_modification_family,
    bench_pasting_diag_family,
    bench_pseudo_naturality_family,
    bench_two_adjoint_family,
    bench_whisker_comp_family,
)
from quant_fund.research.benches_w612 import (
    bench_discrete_valuation_family,
    bench_factorial_ring_family,
    bench_gorenstein_ring_family,
    bench_jacobson_ring_family,
    bench_normal_ring_family,
    bench_regular_ring_family,
)
from quant_fund.research.benches_w613 import (
    bench_neron_smooth_family,
    bench_perfect_witt_family,
    bench_semistable_reduction_family,
    bench_verschiebung_witt_family,
    bench_witt_teich_family,
    bench_witt_vector_family,
)
from quant_fund.research.benches_w614 import (
    bench_finite_spectra_family,
    bench_moore_spec_family,
    bench_peterson_stein_family,
    bench_primary_op_family,
    bench_secondary_op_family,
    bench_steenrod_sq_family,
)
from quant_fund.research.benches_w615 import (
    bench_extremal_ray_family,
    bench_mori_bir_family,
    bench_motivic_adams_family,
    bench_motivic_classifying_family,
    bench_motivic_dg_family,
    bench_tate_object_family,
)
from quant_fund.research.benches_w616 import (
    bench_braided_functor_family,
    bench_center_cat_family,
    bench_ds_category_family,
    bench_fusion_ring_family,
    bench_multifusion_family,
    bench_premodular2_family,
)
from quant_fund.research.benches_w617 import (
    bench_berrick_k_family,
    bench_gersen_suslin_family,
    bench_gillet_thomason_family,
    bench_hermitian_quillen_family,
    bench_k_theory4_family,
    bench_khomo_k_family,
)
from quant_fund.research.benches_w618 import (
    bench_ad_period_family,
    bench_b_drb_family,
    bench_fontaine_curve_family,
    bench_perfectoid_c_family,
    bench_phi_mod_family,
    bench_untilt_family,
)
from quant_fund.research.benches_w619 import (
    bench_andersen_lannes_family,
    bench_chromatic_hopkins_family,
    bench_devissage_ss_family,
    bench_tame_htpy_family,
    bench_thick_spectrum_family,
    bench_unstable_htpy_family,
)
from quant_fund.research.benches_w620 import (
    bench_band_gerbe_family,
    bench_dm_stack2_family,
    bench_gerbe2_family,
    bench_inertia_stack_family,
    bench_rigid_stack_family,
    bench_root_stack_family,
)
from quant_fund.research.benches_w621 import (
    bench_motivic_borel_family,
    bench_motivic_chow_family,
    bench_motivic_class_family,
    bench_motivic_height_family,
    bench_motivic_homology_family,
    bench_motivic_k_family,
)
from quant_fund.research.benches_w622 import (
    bench_delta_ring_family,
    bench_hodge_tate_family,
    bench_nygaard2_family,
    bench_prism2_family,
    bench_prismatic_crystal_family,
    bench_prismatic_site_family,
)
from quant_fund.research.benches_w623 import (
    bench_a_infinity2_family,
    bench_cyclic_operad_family,
    bench_dendroidal2_family,
    bench_e_infinity3_family,
    bench_infty_operad2_family,
    bench_operadic_nerve_family,
)
from quant_fund.research.benches_w624 import (
    bench_adic_formal_family,
    bench_algebraization_family,
    bench_formal_completion_family,
    bench_formal_neighborhood_family,
    bench_groth_existence_family,
    bench_raynaud_formal_family,
)
from quant_fund.research.benches_w625 import (
    bench_complexity_spectrum_family,
    bench_simplicial_htpy_family,
    bench_small_spec_family,
    bench_spectrum_type_family,
    bench_stable_cohomology2_family,
    bench_woodward_op_family,
)
from quant_fund.research.benches_w626 import (
    bench_algebraic_stack2_family,
    bench_artin_stack_family,
    bench_gerbe_cohomology_family,
    bench_orbifold_stack_family,
    bench_quotient_stack2_family,
    bench_stacky_point_family,
)
from quant_fund.research.benches_w627 import (
    bench_condensed_coh_family,
    bench_condensed_ring_family,
    bench_discrete_liquid_family,
    bench_liquid_ring_family,
    bench_scholze_trace_family,
    bench_smith_project_family,
)
from quant_fund.research.benches_w628 import (
    bench_dendroidal_seg_family,
    bench_higher_operad_family,
    bench_moerdijk_weiss_family,
    bench_operad_cat2_family,
    bench_operad_infty3_family,
    bench_operad_module_family,
)
from quant_fund.research.benches_w629 import (
    bench_artinian_alg_family,
    bench_deform_functor2_family,
    bench_hull_deform_family,
    bench_rim_deform_family,
    bench_small_ext_family,
    bench_tangent_def_family,
)
from quant_fund.research.benches_w630 import (
    bench_cartesian_morphism_family,
    bench_fib_infty_family,
    bench_infty_functor_family,
    bench_inner_horn_family,
    bench_joyal_horn_family,
    bench_quasi_cat2_family,
)
from quant_fund.research.benches_w631 import (
    bench_constructible_sh_family,
    bench_etale_cover3_family,
    bench_etale_site3_family,
    bench_ql_sheaf_family,
    bench_torsion_sheaf_family,
    bench_weil_sheaf_family,
)
from quant_fund.research.benches_w632 import (
    bench_big_witt_family,
    bench_good_reduction_family,
    bench_odeur_zarba_family,
    bench_potential_reduction_family,
    bench_tate_curve_family,
    bench_witt_len2_family,
)
from quant_fund.research.benches_w633 import (
    bench_cone_theorem_family,
    bench_contr_rational_family,
    bench_motivic_abelian_family,
    bench_motivic_coho2_family,
    bench_motivic_compact_family,
    bench_motivic_landweber_family,
)
from quant_fund.research.benches_w634 import (
    bench_bicat2_family,
    bench_cat_3cell_family,
    bench_double_lim_family,
    bench_icon_cat_family,
    bench_two_transform_family,
    bench_vert_cat_family,
)
from quant_fund.research.benches_w635 import (
    bench_excellent_ring_family,
    bench_going_up_family,
    bench_integral_closure2_family,
    bench_lying_over_family,
    bench_weil_divisor2_family,
    bench_zariski_main_family,
)
from quant_fund.research.benches_w636 import (
    bench_ek_subfactor_family,
    bench_gyro_cat_family,
    bench_haagerup_sub_family,
    bench_sovereign_cat_family,
    bench_sylleptic_family,
    bench_yang_lee_cat_family,
)
from quant_fund.research.benches_w637 import (
    bench_cocartesian_diamond_family,
    bench_curve_padic_family,
    bench_diamond_mod_family,
    bench_etale_phiphi_family,
    bench_fargues_scholze2_family,
    bench_scholze_bc_family,
)
from quant_fund.research.benches_w638 import (
    bench_fundamental_cat_family,
    bench_grayson_s_family,
    bench_karoubi_v2_family,
    bench_quillen_ldev_family,
    bench_seg_street_family,
    bench_vorst_descent_family,
)
from quant_fund.research.benches_w639 import (
    bench_finite_chromatic_family,
    bench_finite_htpy_family,
    bench_homotopy_fiber2_family,
    bench_periodic_htpy_family,
    bench_rational_spec_family,
    bench_stable_htpy2_family,
)
from quant_fund.research.benches_w640 import (
    bench_azure_space_family,
    bench_elliptic_cohom2_family,
    bench_spectral_etale_family,
    bench_spectral_group_family,
    bench_spectral_scheme3_family,
    bench_spectral_smooth_family,
)
from quant_fund.research.benches_w641 import (
    bench_chromatic_htpy_family,
    bench_homotopy_colim_family,
    bench_periodic_fam_family,
    bench_smash_prod_family,
    bench_stable_stem2_family,
    bench_unstable_tower_family,
)
from quant_fund.research.benches_w642 import (
    bench_bhatt_scholze_family,
    bench_derived_prism_family,
    bench_prismatic_dieudonne_family,
    bench_prismatic_f_family,
    bench_q_crystal_family,
    bench_q_prism_family,
)
from quant_fund.research.benches_w643 import (
    bench_calc_tower_family,
    bench_goodwillie_deriv_family,
    bench_kervaire_inv_family,
    bench_mahowald_inv_family,
    bench_snaith_split_family,
    bench_toda_smith_family,
)
from quant_fund.research.benches_w644 import (
    bench_allday_k_family,
    bench_hall_alg_family,
    bench_residue_k_family,
    bench_s_multicat_family,
    bench_suslin_wagoner_family,
    bench_weibel_nil_family,
)
from quant_fund.research.benches_w645 import (
    bench_balmer_k_family,
    bench_hermitian_k3_family,
    bench_schlichting_k_family,
    bench_thomason_les_family,
    bench_vishik_k_family,
    bench_witt_k_family,
)
from quant_fund.research.benches_w646 import (
    bench_bdr_plus_family,
    bench_diamond_sheaf_family,
    bench_fargues_cat_family,
    bench_spatial_diamond_family,
    bench_untilt2_family,
    bench_v_stack_family,
)
from quant_fund.research.benches_w647 import (
    bench_arkowitz_htpy_family,
    bench_bochner_htpy_family,
    bench_kahn_priddy_family,
    bench_lin_htpy_family,
    bench_selick_htpy_family,
    bench_tits_building_family,
)
from quant_fund.research.benches_w648 import (
    bench_dupont_k_family,
    bench_guin_k_family,
    bench_kodaira_k_family,
    bench_lindenstrauss_k_family,
    bench_suslin_k2_family,
    bench_tsukada_k_family,
)
from quant_fund.research.benches_w649 import (
    bench_breuil_prism_family,
    bench_cartier_prism_family,
    bench_filtered_prism_family,
    bench_frobenius_prism_family,
    bench_prism_site2_family,
    bench_stacky_prism_family,
)
from quant_fund.research.benches_w650 import (
    bench_anick_htpy_family,
    bench_bousfield_htpy_family,
    bench_dror_htpy_family,
    bench_kane_htpy_family,
    bench_moore_htpy_family,
    bench_neisendorfer_htpy_family,
)
from quant_fund.research.benches_w651 import (
    bench_abelian_cat_family,
    bench_filtered_cat_family,
    bench_flat_functor_family,
    bench_malcev_cat_family,
    bench_regular_cat_family,
    bench_sifted_cat2_family,
)
from quant_fund.research.benches_w652 import (
    bench_asymptotic_motive_family,
    bench_exponential_motive_family,
    bench_log_motive_family,
    bench_numerical_motive_family,
    bench_sheaf_motive_family,
    bench_strict_motive_family,
)
from quant_fund.research.benches_w653 import (
    bench_ainf_cohom_family,
    bench_fargues_scholze3_family,
    bench_galois_padic_family,
    bench_hodge_tate_padic_family,
    bench_integral_padic2_family,
    bench_period_ring_family,
)
from quant_fund.research.benches_w654 import (
    bench_accessible_cat2_family,
    bench_compactly_generated_family,
    bench_flat_monad_family,
    bench_locally_presentable_family,
    bench_presentable_cat2_family,
    bench_regular_cat2_family,
)
from quant_fund.research.benches_w655 import (
    bench_bo_htpy_family,
    bench_chromatic_square_family,
    bench_devinatz_htpy_family,
    bench_hopkins_smith_family,
    bench_morava_stab_family,
    bench_telescope_tower_family,
)
from quant_fund.research.benches_w656 import (
    bench_admissible_cat_family,
    bench_cartesian_cat2_family,
    bench_cocomplete_cat_family,
    bench_definable_cat_family,
    bench_essentially_small_family,
    bench_finitely_accessible_family,
)
from quant_fund.research.benches_w657 import (
    bench_bousfield_period_family,
    bench_completion_htpy_family,
    bench_homotopy_cartesian_family,
    bench_p_local_htpy_family,
    bench_ravenel_htpy_family,
    bench_snake_constr_family,
)
from quant_fund.research.benches_w658 import (
    bench_beilinson_regulator_family,
    bench_f_motive_family,
    bench_hodge_motive_family,
    bench_motivic_galois_family,
    bench_period_realization_family,
    bench_tannakian_motive_family,
)
from quant_fund.research.benches_w659 import (
    bench_absolute_cohom_family,
    bench_motivic_pairing_family,
    bench_motivic_tate2_family,
    bench_motivic_weight_family,
    bench_norimotive2_family,
    bench_tate_triple_family,
)
from quant_fund.research.benches_w660 import (
    bench_chromatic_completion_family,
    bench_chromatic_l2_family,
    bench_morava_k2_family,
    bench_periodicity_height_family,
    bench_picard_spec_family,
    bench_telescope_tower2_family,
)
from quant_fund.research.benches_w661 import (
    bench_ambidexterity_family,
    bench_dieudonne_module_family,
    bench_higher_semiadditivity_family,
    bench_honda_formal_family,
    bench_raynaud_height_family,
    bench_tate_height_family,
)
from quant_fund.research.benches_w662 import (
    bench_bar_resolution2_family,
    bench_braces_higher_family,
    bench_deligne_conj2_family,
    bench_factor_homology2_family,
    bench_hochschild_hom2_family,
    bench_little_cubes_family,
)
from quant_fund.research.benches_w663 import (
    bench_dunn_additivity_family,
    bench_e2_algebra_family,
    bench_khovanov_2_family,
    bench_mckay_correspond_family,
    bench_swiss_cheese2_family,
    bench_tensor_factorization_family,
)
from quant_fund.research.benches_w664 import (
    bench_e_ring_moduli_family,
    bench_elliptic_spec2_family,
    bench_spectral_artstack_family,
    bench_spectral_moduli_family,
    bench_structured_spec_family,
    bench_tmf_stack_family,
)
from quant_fund.research.benches_w665 import (
    bench_cohen_moore2_family,
    bench_homotopy_decomp_family,
    bench_kervaire_inv2_family,
    bench_moore_space2_family,
    bench_unstable_vn_family,
    bench_whitehead_product_family,
)
from quant_fund.research.benches_w666 import (
    bench_cotangent_stack_family,
    bench_derived_abelian_family,
    bench_derived_bezout_family,
    bench_derived_bun_family,
    bench_derived_hecke_family,
    bench_simplicial_comm_family,
)
from quant_fund.research.benches_w667 import (
    bench_ab_cat_family,
    bench_coniveau_fil_family,
    bench_exact_cat2_family,
    bench_grothendieck_cat_family,
    bench_special_cat_family,
    bench_stable_cat2_family,
)
from quant_fund.research.benches_w668 import (
    bench_beilinson_regulator2_family,
    bench_hodge_motive2_family,
    bench_motivic_galois2_family,
    bench_norimotive3_family,
    bench_period_realization2_family,
    bench_tannakian_motive2_family,
)
from quant_fund.research.benches_w669 import (
    bench_f_motive2_family,
    bench_milnor_operations2_family,
    bench_motivic_bordism_family,
    bench_motivic_eilenberg2_family,
    bench_motivic_ss2_family,
    bench_slice_filtration2_family,
)
from quant_fund.research.benches_w670 import (
    bench_chromatic_l3_family,
    bench_morava_e2_family,
    bench_morava_k3_family,
    bench_picard_spec2_family,
    bench_red_shift2_family,
    bench_telescope_tower3_family,
)
from quant_fund.research.benches_w671 import (
    bench_adams_edge_family,
    bench_gray_periodic_family,
    bench_homotopy_exponent_family,
    bench_periodic_family_family,
    bench_stunted_proj_family,
    bench_unstable_adams2_family,
)
from quant_fund.research.benches_w672 import (
    bench_compact_cat_family,
    bench_monoidal_derived_family,
    bench_perverse_cat_family,
    bench_smashing_cat_family,
    bench_super_cat_family,
    bench_tannakian_cat_family,
)
from quant_fund.research.benches_w673 import (
    bench_derived_cohom_family,
    bench_derived_fiber2_family,
    bench_derived_intersection_family,
    bench_relative_trace_family,
    bench_spectral_deformation2_family,
    bench_virtual_class2_family,
)
from quant_fund.research.benches_w674 import (
    bench_analytic_spec_family,
    bench_derived_k3_family,
    bench_equivariant_spec_family,
    bench_graded_spec_family,
    bench_spectral_curve_family,
    bench_spectral_gm_family,
)
from quant_fund.research.benches_w675 import (
    bench_boards_operad_family,
    bench_cyclotomic_e_n_family,
    bench_e3_algebra_family,
    bench_getzler_jones_family,
    bench_surfaces_operad_family,
    bench_tadv_hochschild_family,
)
from quant_fund.research.benches_w676 import (
    bench_center_hochschild_family,
    bench_en_algebra2_family,
    bench_higher_brace2_family,
    bench_koszul_operad2_family,
    bench_operad_lie_family,
    bench_thom_transpose_family,
)
from quant_fund.research.benches_w677 import (
    bench_cat_dg_family,
    bench_cat_structure_family,
    bench_combinatorial_mc_family,
    bench_derivator_cat_family,
    bench_quillen_cat_family,
    bench_univalent_cat_family,
)
from quant_fund.research.benches_w678 import (
    bench_equipment_cat_family,
    bench_fibrant_cat_family,
    bench_homotopical_cat_family,
    bench_pointed_cat_family,
    bench_relative_cat_family,
    bench_simplicial_cat_family,
)
from quant_fund.research.benches_w679 import (
    bench_absolute_motive_family,
    bench_etale_motive_family,
    bench_motivic_heart_family,
    bench_motivic_realization_family,
    bench_motivic_thh_family,
    bench_relative_motive_family,
)
from quant_fund.research.benches_w680 import (
    bench_spectral_abelian_family,
    bench_spectral_crystal_family,
    bench_spectral_etale2_family,
    bench_spectral_perfect_family,
    bench_spectral_proper_family,
    bench_spectral_smooth2_family,
)
from quant_fund.research.benches_w681 import (
    bench_homotopy_factor_family,
    bench_homotopy_fixed_family,
    bench_homotopy_lift_family,
    bench_homotopy_orbit_family,
    bench_stable_operad_family,
    bench_stable_sheaf_family,
)
from quant_fund.research.benches_w682 import (
    bench_motivic_crystal_family,
    bench_motivic_cycle_family,
    bench_motivic_etale_family,
    bench_motivic_prism_family,
    bench_motivic_sphere3_family,
    bench_motivic_tower_family,
)
from quant_fund.research.benches_w683 import (
    bench_blue_shift2_family,
    bench_chromatic_fracture2_family,
    bench_fgsl_group2_family,
    bench_k_n_local2_family,
    bench_morava_stabilizer2_family,
    bench_tate_spec2_family,
)
from quant_fund.research.benches_w684 import (
    bench_motivic_base2_family,
    bench_motivic_frequency_family,
    bench_motivic_infinite_family,
    bench_motivic_suslin_family,
    bench_motivic_weight2_family,
    bench_motivic_wit_family,
)
from quant_fund.research.benches_w685 import (
    bench_homotopy_class2_family,
    bench_homotopy_limit_family,
    bench_homotopy_tower_family,
    bench_spectral_sequence5_family,
    bench_stable_bousfield_family,
    bench_stable_mapping_family,
)
from quant_fund.research.benches_w686 import (
    bench_braces_e4_family,
    bench_centralizer_alg_family,
    bench_delooping2_family,
    bench_e4_algebra_family,
    bench_factorization_hom2_family,
    bench_koszul_duality2_family,
)
from quant_fund.research.benches_w687 import (
    bench_cat_bicomplete_family,
    bench_cat_cofibrant_family,
    bench_cat_descent_family,
    bench_cat_fibrant_obj_family,
    bench_cat_glueable_family,
    bench_cat_univariant_family,
)
from quant_fund.research.benches_w688 import (
    bench_spectral_cellular_family,
    bench_spectral_cohomological_family,
    bench_spectral_field_family,
    bench_spectral_filtration_family,
    bench_spectral_finite_family,
    bench_spectral_lattice_family,
)
from quant_fund.research.benches_w689 import (
    bench_motivic_euler_family,
    bench_motivic_ext_family,
    bench_motivic_infinite2_family,
    bench_motivic_jouanolou_family,
    bench_motivic_norm_family,
    bench_motivic_ramified_family,
)
from quant_fund.research.benches_w690 import (
    bench_homotopy_model_family,
    bench_homotopy_sheaf_family,
    bench_stable_algebra_family,
    bench_stable_group_family,
    bench_stable_module_family,
    bench_stable_monoid_family,
)
from quant_fund.research.benches_w691 import (
    bench_cat_fusion_family,
    bench_cat_pretopos_family,
    bench_cat_ribbon_family,
    bench_cat_semiadd_family,
    bench_cat_semisimple_family,
    bench_cat_tannakian2_family,
)
from quant_fund.research.benches_w692 import (
    bench_centralizer_alg2_family,
    bench_e5_algebra_family,
    bench_factorization_hom3_family,
    bench_framed_discs_family,
    bench_little_cubes2_family,
    bench_swiss_cheese3_family,
)
from quant_fund.research.benches_w693 import (
    bench_derived_cartesian_family,
    bench_derived_etale_family,
    bench_derived_flat_family,
    bench_derived_quasi_coherent_family,
    bench_derived_represent_family,
    bench_derived_smooth2_family,
)
from quant_fund.research.benches_w694 import (
    bench_motivic_atiyah_family,
    bench_motivic_coniveau_family,
    bench_motivic_deligne_family,
    bench_motivic_residue_family,
    bench_motivic_trace_family,
    bench_motivic_transfer2_family,
)
from quant_fund.research.benches_w695 import (
    bench_homotopy_abelian_family,
    bench_homotopy_extended_family,
    bench_homotopy_finite_family,
    bench_homotopy_infinite_family,
    bench_stable_compact_family,
    bench_stable_synthetic_family,
)
from quant_fund.research.benches_w696 import (
    bench_cat_image_family,
    bench_cat_index_family,
    bench_cat_kernel_family,
    bench_cat_monotone_family,
    bench_cat_pullback_family,
    bench_cat_rank_family,
)
from quant_fund.research.benches_w697 import (
    bench_spectral_dedekind_family,
    bench_spectral_dvr_family,
    bench_spectral_excellent_family,
    bench_spectral_jacobson_family,
    bench_spectral_noether_family,
    bench_spectral_regular_family,
)
from quant_fund.research.benches_w698 import (
    bench_motivic_cartier_family,
    bench_motivic_frobenius_family,
    bench_motivic_hodge_family,
    bench_motivic_lax_family,
    bench_motivic_span_family,
    bench_motivic_street_family,
)
from quant_fund.research.benches_w699 import (
    bench_homotopy_general_family,
    bench_homotopy_rational_family,
    bench_stable_dual_family,
    bench_stable_lie_family,
    bench_stable_motivic_family,
    bench_stable_perf_family,
)
from quant_fund.research.benches_w700 import (
    bench_cat_lax_family,
    bench_cat_pushout_family,
    bench_cat_size_family,
    bench_cat_span_family,
    bench_cat_street_family,
    bench_cat_total_family,
)
from quant_fund.research.benches_w701 import (
    bench_derived_conn_family,
    bench_derived_integral_family,
    bench_derived_local_family,
    bench_derived_noether_family,
    bench_derived_normal_family,
    bench_derived_reduced_family,
)
from quant_fund.research.benches_w702 import (
    bench_motivic_degree_family,
    bench_motivic_diagonal_family,
    bench_motivic_field_family,
    bench_motivic_fundamental_family,
    bench_motivic_hochschild_family,
    bench_motivic_spark_family,
)
from quant_fund.research.benches_w703 import (
    bench_homotopy_fiber3_family,
    bench_homotopy_spectrum2_family,
    bench_homotopy_suspension2_family,
    bench_homotopy_vn_family,
    bench_stable_derivator_family,
    bench_stable_excisive_family,
)
from quant_fund.research.benches_w704 import (
    bench_braces_e5_family,
    bench_delooping3_family,
    bench_higher_algebra9_family,
    bench_koszul_duality3_family,
    bench_operad_infty5_family,
    bench_operad_swiss4_family,
)
from quant_fund.research.benches_w705 import (
    bench_derived_abelian2_family,
    bench_derived_cover_family,
    bench_derived_geometry7_family,
    bench_derived_morph_family,
    bench_derived_stack3_family,
    bench_derived_topos_family,
)
from quant_fund.research.benches_w706 import (
    bench_chromatic_base_family,
    bench_chromatic_layer_family,
    bench_chromatic_square2_family,
    bench_elliptic_morava_family,
    bench_lubin_tate3_family,
    bench_morava_maven_family,
)
from quant_fund.research.benches_w707 import (
    bench_spectral_coord_family,
    bench_spectral_ext_field_family,
    bench_spectral_level_family,
    bench_spectral_polynomial2_family,
    bench_spectral_prime_family,
    bench_spectral_residue_family,
)
from quant_fund.research.benches_w708 import (
    bench_cat_dold_kan_family,
    bench_cat_enriched_lim_family,
    bench_cat_hoc_family,
    bench_cat_pseudo_limit_family,
    bench_cat_reedy_cat_family,
    bench_cat_weak_eq_family,
)
from quant_fund.research.benches_w709 import (
    bench_cat_ab2_family,
    bench_cat_ab_loc_family,
    bench_cat_exact3_family,
    bench_cat_freyd_family,
    bench_cat_pro_object2_family,
    bench_cat_univariant2_family,
)
from quant_fund.research.benches_w710 import (
    bench_floyd_farey_family,
    bench_higher_algebra8_family,
    bench_little_discs3_family,
    bench_operad_infty4_family,
    bench_operad_swiss3_family,
    bench_operad_twisted_family,
)
from quant_fund.research.benches_w711 import (
    bench_homotopy_local_family,
    bench_homotopy_sheaf2_family,
    bench_homotopy_stable4_family,
    bench_stable_coalgebra_family,
    bench_stable_inf_cat_family,
    bench_stable_sheaf2_family,
)
from quant_fund.research.benches_w712 import (
    bench_motivic_additive_cat_family,
    bench_motivic_additive_family,
    bench_motivic_chern2_family,
    bench_motivic_cover_family,
    bench_motivic_filtration2_family,
    bench_motivic_gysin2_family,
)
from quant_fund.research.benches_w713 import (
    bench_derived_affine_family,
    bench_derived_projective_family,
    bench_spectral_artin_family,
    bench_spectral_dirac_family,
    bench_spectral_gal_family,
    bench_spectral_semi_family,
)
from quant_fund.research.benches_w714 import (
    bench_derived_proper2_family,
    bench_derived_separated2_family,
    bench_motivic_functor_family,
    bench_motivic_nerve_family,
    bench_motivic_partial_family,
    bench_motivic_total_family,
)
from quant_fund.research.benches_w715 import (
    bench_auslander_reiten_family,
    bench_cluster_algebra_family,
    bench_cluster_category_family,
    bench_quiver_mutation_family,
    bench_silting_object_family,
    bench_tilting_object_family,
)
from quant_fund.research.benches_w716 import (
    bench_exceptional_coll_family,
    bench_fourier_mukai_family,
    bench_semi_orthogonal_family,
    bench_serre_functor_family,
    bench_sod_decomp_family,
    bench_spherical_functor_family,
)
from quant_fund.research.benches_w717 import (
    bench_bondal_kapranov_family,
    bench_dg_enhancement_family,
    bench_enhanced_triangulated_family,
    bench_nc_k_theory_family,
    bench_nc_motive_family,
    bench_tabuada_motive_family,
)
from quant_fund.research.benches_w718 import (
    bench_der_bimodule_family,
    bench_helix_theory_family,
    bench_higher_auslander_family,
    bench_icy_paper_family,
    bench_mutation_class_family,
    bench_rep_finite_family,
)
from quant_fund.research.benches_w719 import (
    bench_calabi_yau_tri_family,
    bench_d_calabi_yau_family,
    bench_frobenius_cat_family,
    bench_gorenstein_proj_family,
    bench_orbit_category_family,
    bench_stable_category_family,
)
from quant_fund.research.benches_w720 import (
    bench_categorical_entropy_family,
    bench_cluster_tilting_family,
    bench_derived_morita_family,
    bench_preprojective_alg_family,
    bench_rouquier_dim_family,
    bench_serre_dim_family,
)
from quant_fund.research.benches_w721 import (
    bench_beilinson_height_family,
    bench_brown_motives_family,
    bench_mixed_elliptic_family,
    bench_motivic_pi_family,
    bench_mzc_motive_family,
    bench_zeta_element_family,
)
from quant_fund.research.benches_w722 import (
    bench_borel_motivic_family,
    bench_deligne_period_family,
    bench_motivic_multiple_zeta_family,
    bench_period_poly_family,
    bench_specialization_motive_family,
    bench_zagier_polylog_family,
)
from quant_fund.research.benches_w723 import (
    bench_euler_system_family,
    bench_gross_zagier_family,
    bench_iwasawa_motive_family,
    bench_kolyvagin_sys_family,
    bench_perrin_riou_family,
    bench_rubin_main_conj_family,
)
from quant_fund.research.benches_w724 import (
    bench_coates_wiles_family,
    bench_gan_gross_prasad_family,
    bench_greenberg_selmer_family,
    bench_heegner_cycle_family,
    bench_iwasawa_lfunc_family,
    bench_kurihara_iwasawa_family,
)
from quant_fund.research.benches_w725 import (
    bench_arithmetic_arnold_family,
    bench_bertolini_darmon_family,
    bench_darmon_point_family,
    bench_howard_main_family,
    bench_p_group_iwasawa_family,
    bench_shimura_period_family,
)
from quant_fund.research.benches_w726 import (
    bench_diamond_taylor_wiles_family,
    bench_jetchev_skinner_family,
    bench_kisin_crystalline_family,
    bench_mazur_deform_family,
    bench_wan_sss_family,
    bench_wiles_taylor_family,
)
from quant_fund.research.benches_w727 import (
    bench_breuil_meizard_family,
    bench_caruso_lebaron_family,
    bench_galdef_ring_family,
    bench_gee_kisin_family,
    bench_patching_arg_family,
    bench_taylor_wiles_family,
)
from quant_fund.research.benches_w728 import (
    bench_a1_degrees_family,
    bench_emerton_glass_family,
    bench_luan_yao_family,
    bench_morel_voev_family,
    bench_totaro_cycle_family,
    bench_voev_homotopy_family,
)
from quant_fund.research.benches_w729 import (
    bench_hauwas_nori_family,
    bench_jogiad_motive_family,
    bench_motivic_pipe_family,
    bench_roald_suslin_family,
    bench_thom_mgl2_family,
    bench_voev_suslin_family,
)
from quant_fund.research.benches_w730 import (
    bench_groth_tame_family,
    bench_grothendieck_muw_family,
    bench_kato_swan_family,
    bench_raynaud_pencil_family,
    bench_saito_epsilon_family,
    bench_swan_conductor_family,
)
from quant_fund.research.benches_w731 import (
    bench_brylinski_kato_family,
    bench_higher_ramif_family,
    bench_log_ramification_family,
    bench_neron_raynaud_family,
    bench_semi_stable_model_family,
    bench_temkin_alter_family,
)
from quant_fund.research.benches_w732 import (
    bench_hall_algebra_family,
    bench_joyce_hall_family,
    bench_lusztig_hall_family,
    bench_ringel_hall_family,
    bench_schiffmann_hall_family,
    bench_toen_hall_family,
)
from quant_fund.research.benches_w733 import (
    bench_bridgeland_hall_family,
    bench_calaque_hall_family,
    bench_green_hall_family,
    bench_kontsevich_soibelman_family,
    bench_morita_hall_family,
    bench_mozgovoy_hall_family,
)
from quant_fund.research.benches_w734 import (
    bench_garmadon_sle_family,
    bench_lawler_werner_family,
    bench_miller_sheffield_family,
    bench_osgood_schramm_family,
    bench_smirnov_parafermion_family,
    bench_werner_wilson_family,
)
from quant_fund.research.benches_w735 import (
    bench_beffara_sle_family,
    bench_benoist_sle_family,
    bench_holden_sle_family,
    bench_kemppainen_sle_family,
    bench_viklund_sle_family,
    bench_zykin_sle_family,
)
from quant_fund.research.benches_w736 import (
    bench_aru_powell_family,
    bench_berestycki_sheffield_family,
    bench_bisbisot_sheffield_family,
    bench_dhms_lqg_family,
    bench_huang_rhodes_family,
    bench_sheffield_gff_family,
)
from quant_fund.research.benches_w737 import (
    bench_ding_dupias_family,
    bench_gaines_sle_family,
    bench_gwynne_miller_family,
    bench_miller_wu_family,
    bench_rhoade_vargas_family,
    bench_sheffield_quantum_family,
)
from quant_fund.research.benches_w738 import (
    bench_abraham_bipartite_family,
    bench_bettinelli_jacob_family,
    bench_chapuy_dolega_family,
    bench_curien_legall_family,
    bench_le_gall_miermont_family,
    bench_marckert_mokkadem_family,
)
from quant_fund.research.benches_w739 import (
    bench_bernardi_bijection_family,
    bench_bonzom_combe_family,
    bench_bouttier_guiter_family,
    bench_caraceni_curien_family,
    bench_mullin_bijection_family,
    bench_schaeffer_bijection_family,
)
from quant_fund.research.benches_w740 import (
    bench_cardy_formula_family,
    bench_duminil_copin_family,
    bench_grimmett_percolation_family,
    bench_kesten_percolation_family,
    bench_russo_seymour_family,
    bench_smirnov_percolation_family,
)
from quant_fund.research.benches_w741 import (
    bench_aiten_chayes_family,
    bench_beffara_nolin_family,
    bench_gandre_liggett_family,
    bench_hara_slade_family,
    bench_heyman_redner_family,
    bench_newman_percolation_family,
)
from quant_fund.research.benches_w742 import (
    bench_deift_rmt_family,
    bench_erdos_yau_family,
    bench_forrester_rmt_family,
    bench_johansson_rmt_family,
    bench_mehta_rmt_family,
    bench_soshnikov_rmt_family,
)
from quant_fund.research.benches_w743 import (
    bench_baik_rmt_family,
    bench_borodin_olshanski_family,
    bench_bourgade_rmt_family,
    bench_chafai_rmt_family,
    bench_cipolloni_erdos_family,
    bench_tao_vu_family,
)
from quant_fund.research.benches_w744 import (
    bench_amir_corwin_family,
    bench_borodin_corwin_family,
    bench_calabrese_kpz_family,
    bench_corwin_kpz_family,
    bench_kardar_parisi_family,
    bench_quastel_spohn_family,
)
from quant_fund.research.benches_w745 import (
    bench_bernard_nicola_family,
    bench_dotsenko_kpz_family,
    bench_hairer_kpz_family,
    bench_imamura_sasamoto_family,
    bench_spohn_kpz_family,
    bench_tracy_widom_kpz_family,
)
from quant_fund.research.benches_w746 import (
    bench_derrida_tasep_family,
    bench_ferrari_tasep_family,
    bench_liggett_exclusion_family,
    bench_sasamoto_tasep_family,
    bench_spitzer_exclusion_family,
    bench_tracy_widom_tasep_family,
)
from quant_fund.research.benches_w747 import (
    bench_balazs_seppalainen_family,
    bench_bertini_giacomin_family,
    bench_gardina_asym_family,
    bench_quastel_valko_family,
    bench_schutz_tasep_family,
    bench_timar_tasep_family,
)
from quant_fund.research.benches_w748 import (
    bench_aggarwal_sixv_family,
    bench_baxter_vertex_family,
    bench_borodin_sixv_family,
    bench_corwin_petrov_family,
    bench_gowers_knot_family,
    bench_reshetikhin_vertex_family,
)
from quant_fund.research.benches_w749 import (
    bench_borodin_bufetov_family,
    bench_borodin_wheeler_family,
    bench_bufetov_sixv_family,
    bench_dimitrov_sixv_family,
    bench_kuan_sixv_family,
    bench_wheeler_zinn_family,
)
from quant_fund.research.benches_w750 import (
    bench_chelkak_ising_family,
    bench_duminil_copin2_family,
    bench_hongler_ising_family,
    bench_kenyon_dimers_family,
    bench_smirnov_ising_family,
    bench_thurston_tiling_family,
)
from quant_fund.research.benches_w751 import (
    bench_ciucu_dimers_family,
    bench_cohn_elkies_family,
    bench_durfee_arctic_family,
    bench_karl_dimers_family,
    bench_kassel_kenyon_family,
    bench_petrov_dimer_family,
)
from quant_fund.research.benches_w752 import (
    bench_benjamini_ust_family,
    bench_kirchhoff_matrix_family,
    bench_lawler_lerw_family,
    bench_pemantle_ust_family,
    bench_schramm_lerw_family,
    bench_wilson_ust_family,
)
from quant_fund.research.benches_w753 import (
    bench_aizenman_irf_family,
    bench_cardy_on_family,
    bench_fernandez_frohlich_family,
    bench_fradkin_sokal_family,
    bench_nienhuis_on_family,
    bench_pelissetto_vicari_family,
)
from quant_fund.research.benches_w754 import (
    bench_brydges_spencer_family,
    bench_caracciolo_pelissetto_family,
    bench_glasner_aizenman_family,
    bench_grimmett_rc_family,
    bench_hara_hara_family,
    bench_sokal_bcc_family,
)
from quant_fund.research.benches_w755 import (
    bench_barlow_ust_family,
    bench_kassel_wu_family,
    bench_kenyon_wilson_family,
    bench_lejan_loop_family,
    bench_lupu_loop_family,
    bench_lyons_peres_family,
)
from quant_fund.research.benches_w756 import (
    bench_berestycki_gff_family,
    bench_duplantier_sheffield_family,
    bench_houchmandzadeh_gff_family,
    bench_nick_gff_family,
    bench_sheffield_miller_family,
    bench_wiegmann_zabrodin_family,
)
from quant_fund.research.benches_w757 import (
    bench_camia_newman_family,
    bench_dubedat_cle_family,
    bench_kemppainen_werner_family,
    bench_miller_watson_cle_family,
    bench_rivera_cle_family,
    bench_sheffield_werner_cle_family,
)
from quant_fund.research.benches_w758 import (
    bench_aru_gff_family,
    bench_bolthausen_gff_family,
    bench_chatterjee_gff_family,
    bench_ding_zeitouni_family,
    bench_najafi_gff_family,
    bench_powell_gff_family,
)
from quant_fund.research.benches_w759 import (
    bench_apu_cle_family,
    bench_gwynne_cle_family,
    bench_hospitsky_cle_family,
    bench_nolin_cle_family,
    bench_sun_cle_family,
    bench_zhan_cle_family,
)
from quant_fund.research.benches_w760 import (
    bench_cameron_martin_family,
    bench_doob_bm_family,
    bench_gikhman_skorokhod_family,
    bench_ito_bm_family,
    bench_levy_bm_family,
    bench_wiener_bm_family,
)
from quant_fund.research.benches_w761 import (
    bench_alternating_renewal_family,
    bench_blackwell_renewal_family,
    bench_delayed_renewal_family,
    bench_excess_renewal_family,
    bench_key_renewal_family,
    bench_renewal_reward2_family,
)
from quant_fund.research.benches_w762 import (
    bench_cadlag_space_family,
    bench_doob_meyer_family,
    bench_martin_boundary_family,
    bench_prohorov_thm2_family,
    bench_skohorod_metric_family,
    bench_tightness_check_family,
)
from quant_fund.research.benches_w763 import (
    bench_boneschi_boal_family,
    bench_bradley_mixing_family,
    bench_hopf_chain_family,
    bench_ibagimov_mixing_family,
    bench_polya_urn_family,
    bench_rosenthal_mom_family,
)
from quant_fund.research.benches_w764 import (
    bench_donsker_class_family,
    bench_donsker_thm_family,
    bench_dz_invariance_family,
    bench_empirical_process_family,
    bench_osj_metric_family,
    bench_wiener_measure_family,
)
from quant_fund.research.benches_w765 import (
    bench_chung_lil_family,
    bench_glivenko_cantelli_family,
    bench_khintchine_lln_family,
    bench_kolmogorov_3series_family,
    bench_levy_convergence_family,
    bench_strassen_lil_family,
)
from quant_fund.research.benches_w766 import (
    bench_bounded_lip_family,
    bench_bracketing_ent_family,
    bench_dudley_theorem_family,
    bench_dvoretzky_thm_family,
    bench_varadarajan_thm_family,
    bench_vc_class_family,
)
from quant_fund.research.benches_w767 import (
    bench_borell_tis_family,
    bench_fernique_thm_family,
    bench_gordon_thm_family,
    bench_slepian_lemma_family,
    bench_sudakov_min_family,
    bench_talagrand_conc_family,
)
from quant_fund.research.benches_w768 import (
    bench_dw_ldp_family,
    bench_freidlin_wentzell_family,
    bench_mogulskii_thm_family,
    bench_sanov_thm_family,
    bench_schider_thm_family,
    bench_varadhan_ldp_family,
)
from quant_fund.research.benches_w769 import (
    bench_barbour_stein_family,
    bench_chatt_stein_family,
    bench_chen_stein_family,
    bench_ross_stein_family,
    bench_stein_equation_family,
    bench_stein_method_family,
)
from quant_fund.research.benches_w770 import (
    bench_frechet_domain_family,
    bench_gumbel_domain_family,
    bench_hill_est_family,
    bench_peak_over_family,
    bench_pickands_est_family,
    bench_weibull_domain_family,
)
from quant_fund.research.benches_w771 import (
    bench_clayton_copula_family,
    bench_copula_gauss_family,
    bench_copula_t_family,
    bench_frank_copula_family,
    bench_gumbel_copula_family,
    bench_joe_copula_family,
)
from quant_fund.research.benches_w772 import (
    bench_branching_imm_family,
    bench_crump_mode_family,
    bench_galton_watson_family,
    bench_kimmel_branch_family,
    bench_multi_type_branch_family,
    bench_sevastyanov_family,
)
from quant_fund.research.benches_w773 import (
    bench_bulk_queue_family,
    bench_gm_queue_family,
    bench_mg1_queue_family,
    bench_mm1_queue_family,
    bench_priority_queue_family,
    bench_retrial_queue_family,
)
from quant_fund.research.benches_w774 import (
    bench_fluctuation_rw_family,
    bench_ladder_epoch_family,
    bench_maxwell_rw_family,
    bench_sparc_rw_family,
    bench_spitzer_rw_family,
    bench_wiener_hopf_rw_family,
)
from quant_fund.research.benches_w775 import (
    bench_burkholder_davis_family,
    bench_doleans_meas_family,
    bench_gundy_mart_family,
    bench_local_mart_family,
    bench_predictable_proc_family,
    bench_square_bracket_family,
)
from quant_fund.research.benches_w776 import (
    bench_campbell_thm_family,
    bench_cox_process_family,
    bench_hawkes_point_family,
    bench_marked_point_family,
    bench_palm_dist_family,
    bench_self_excite_family,
)
from quant_fund.research.benches_w777 import (
    bench_diffusion_approx_family,
    bench_fluid_limit_family,
    bench_halfin_whitt_family,
    bench_heavy_traffic_family,
    bench_kingman_bound_family,
    bench_qed_regime_family,
)
from quant_fund.research.benches_w778 import (
    bench_borel_tanner_family,
    bench_engset_family,
    bench_erlang_b_family,
    bench_erlang_c_family,
    bench_pollaczek_khinchine_family,
    bench_takacs_vacation_family,
)
from quant_fund.research.benches_w779 import (
    bench_logarithmic_red_family,
    bench_matrix_geom_family,
    bench_neuts_map_family,
    bench_phase_type_family,
    bench_quasi_birth_family,
    bench_ramaswami_family,
)
from quant_fund.research.benches_w780 import (
    bench_bcmp_net_family,
    bench_convoy_net_family,
    bench_insensitive_thm_family,
    bench_kaufman_roberts_family,
    bench_mean_value_family,
    bench_orku_loss_family,
)
from quant_fund.research.benches_w781 import (
    bench_karlin_mcg_family,
    bench_keilson_stieltjes_family,
    bench_korolyuk_family,
    bench_palm_khinchin_family,
    bench_regen_proc_family,
    bench_wold_proc_family,
)
from quant_fund.research.benches_w782 import (
    bench_ffusion_lims_family,
    bench_filt_proc_family,
    bench_jacod_shiryaev_family,
    bench_kunita_watanabe_family,
    bench_pinsky_proc_family,
    bench_slivnyak_family,
)
from quant_fund.research.benches_w783 import (
    bench_cramer_wold_family,
    bench_hazard_order_family,
    bench_likelihood_order_family,
    bench_predictable_bracket_family,
    bench_semi_mart_family,
    bench_stricker_thm_family,
)
from quant_fund.research.benches_w784 import (
    bench_girsanov_thm2_family,
    bench_levy_khinchine_family,
    bench_levy_measure_family,
    bench_self_decomp_family,
    bench_stable_levy_family,
    bench_subordinator_family,
)
from quant_fund.research.benches_w785 import (
    bench_dolean_mart_family,
    bench_follmer_mart_family,
    bench_local_mart2_family,
    bench_protter_ito_family,
    bench_strong_sol_family,
    bench_usual_cond_family,
)
from quant_fund.research.benches_w786 import (
    bench_area_mart_family,
    bench_controlled_path_family,
    bench_hairspring_map_family,
    bench_lyons_lift_family,
    bench_rough_path_family,
    bench_signature_transform2_family,
)
from quant_fund.research.benches_w787 import (
    bench_gubinelli_sewing_family,
    bench_ito_signature_family,
    bench_lyons_extension_family,
    bench_step_signature_family,
    bench_tame_map_family,
    bench_young_integral_family,
)
from quant_fund.research.benches_w788 import (
    bench_clark_ocone_family,
    bench_divergence_op_family,
    bench_nourdin_peccati_family,
    bench_nualart_pardoux_family,
    bench_skorohod_int_family,
    bench_wiener_chaos_family,
)
from quant_fund.research.benches_w789 import (
    bench_benamou_brenier_family,
    bench_entropy_regular_family,
    bench_fokker_planck2_family,
    bench_gradient_flow_family,
    bench_jko_step_family,
    bench_wasserstein_grad_family,
)
from quant_fund.research.benches_w790 import (
    bench_kac_theorem_family,
    bench_mckean_vlasov_family,
    bench_mean_field_game2_family,
    bench_nonlinear_markov_family,
    bench_propagation_chaos_family,
    bench_self_stabilizing_family,
)
from quant_fund.research.benches_w791 import (
    bench_doering_mueller_family,
    bench_kpz_equation_family,
    bench_paracontrolled_spde_family,
    bench_quasilinear_spde_family,
    bench_spde_heat_family,
    bench_stochastic_burgers_family,
)
from quant_fund.research.benches_w792 import (
    bench_backward_sde_family,
    bench_bsde_solver_family,
    bench_fbsde_markov_family,
    bench_pardoux_peng_family,
    bench_reflected_bsde_family,
    bench_second_order_bsde_family,
)
from quant_fund.research.benches_w793 import (
    bench_cubature_wiener_family,
    bench_milstein_scheme_family,
    bench_rough_vol2_family,
    bench_stochastic_taylor_family,
    bench_wagner_platen_family,
    bench_wong_zakai_family,
)
from quant_fund.research.benches_w794 import (
    bench_dynamic_programming_family,
    bench_hamilton_jacobi_family,
    bench_impulsive_control_family,
    bench_quasi_variational_family,
    bench_verification_thm_family,
    bench_viscosity_solution_family,
)
from quant_fund.research.benches_w795 import (
    bench_differential_game_family,
    bench_dynkin_game_family,
    bench_isaacs_equation_family,
    bench_nonzero_sum_game_family,
    bench_stochastic_game2_family,
    bench_zero_sum_game_family,
)
from quant_fund.research.benches_w796 import (
    bench_coupled_fbsde_family,
    bench_decoupling_field2_family,
    bench_four_step_scheme_family,
    bench_quasi_bsde_family,
    bench_random_bsde_family,
    bench_time_bsde_family,
)
from quant_fund.research.benches_w797 import (
    bench_doubly_bsde_family,
    bench_obstacle_bsde_family,
    bench_quadratic_bsde_family,
    bench_reflected_bsde2_family,
    bench_second_bsde_family,
    bench_super_linear_family,
)
from quant_fund.research.benches_w798 import (
    bench_anticipating_sde_family,
    bench_delayed_sde_family,
    bench_forward_sde_family,
    bench_functional_sde_family,
    bench_neutral_sde_family,
    bench_random_sde_family,
)
from quant_fund.research.benches_w799 import (
    bench_dupire_functional_family,
    bench_functional_ito_family,
    bench_kolmogorov_path_family,
    bench_path_dependent_pde_family,
    bench_path_sobolev_family,
    bench_viscosity_path_family,
)
from quant_fund.research.benches_w800 import (
    bench_bates_model_family,
    bench_heston_model_family,
    bench_rough_heston_family,
    bench_sabr_model_family,
    bench_scott_vol_family,
    bench_three_two_vol_family,
)
from quant_fund.research.benches_w801 import (
    bench_fractional_heston_family,
    bench_multifactor_rough_family,
    bench_rough_bergomi_family,
    bench_rough_sabr_family,
    bench_rough_variance_family,
    bench_volterra_sde_family,
)
from quant_fund.research.benches_w802 import (
    bench_latent_sde_family,
    bench_logsig_rde_family,
    bench_neural_cde_family,
    bench_neural_rde_family,
    bench_sde_gan_family,
    bench_sde_matching_family,
)
from quant_fund.research.benches_w803 import (
    bench_expected_sig_family,
    bench_pde_signature_family,
    bench_sig_inversion_family,
    bench_signature_gan2_family,
    bench_signature_kernel_family,
    bench_truncated_sig_family,
)
from quant_fund.research.benches_w804 import (
    bench_absolute_cont_family,
    bench_density_bound_family,
    bench_malliavin_cov_family,
    bench_nualart_zakai_family,
    bench_smoothness_h_family,
    bench_watanabe_map_family,
)
from quant_fund.research.benches_w805 import (
    bench_doss_sussmann_family,
    bench_follmer_strat_family,
    bench_ito_isometry_family,
    bench_skorohod_lemma_family,
    bench_stratonovich_conv_family,
    bench_tanaka_meyer_family,
)
from quant_fund.research.benches_w806 import (
    bench_compound_poisson_family,
    bench_excursion_theory_family,
    bench_jump_diffusion_family,
    bench_kou_model_family,
    bench_marked_hawkes_family,
    bench_merton_jump_family,
)
from quant_fund.research.benches_w807 import (
    bench_cayley_moser_family,
    bench_chow_robbins_family,
    bench_free_boundary_family,
    bench_markov_stopping_family,
    bench_secretary_dp_family,
    bench_snell_envelope_family,
)
from quant_fund.research.benches_w808 import (
    bench_bene_filter_family,
    bench_hidden_markov_filter_family,
    bench_kalman_bucy_family,
    bench_kushner_strat_family,
    bench_particle_filter2_family,
    bench_zakai_eq_family,
)
from quant_fund.research.benches_w809 import (
    bench_cutoff_phenomenon_family,
    bench_doeblin_coupling_family,
    bench_drift_lyapunov_family,
    bench_ergodic_markov_family,
    bench_harris_recurrent_family,
    bench_mixing_time_family,
)
from quant_fund.research.benches_w810 import (
    bench_anytime_valid,
    bench_distributional_ml,
    bench_energy_score,
    bench_leakage_redteam,
    bench_regime_eval,
    bench_ts_conformal,
)
from quant_fund.research.benches_w811 import (
    bench_cambrian_pp_family,
    bench_ergodic_pp_family,
    bench_gneding_metric_family,
    bench_j_function_family,
    bench_papangelou_family,
    bench_void_prob_family,
)
from quant_fund.research.benches_w812 import (
    bench_epsilon_coupling_family,
    bench_nummelin_family,
    bench_petite_set_family,
    bench_regenerative_family,
    bench_small_set_family,
    bench_split_chain_family,
)
from quant_fund.research.benches_w813 import (
    bench_diffusion_semigroup_family,
    bench_feller_boundary_family,
    bench_kreyn_resolvent_family,
    bench_scale_measure_family,
    bench_speed_measure_family,
    bench_yosida_op_family,
)
from quant_fund.research.benches_w814 import (
    bench_karal_flow_family,
    bench_kunita_flow_family,
    bench_liouville_flow_family,
    bench_meyers_process_family,
    bench_stochastic_damping_family,
    bench_stochastic_flow_family,
)
from quant_fund.research.benches_w815 import (
    bench_excursion_proc_family,
    bench_inverse_local_family,
    bench_knight_theorem_family,
    bench_mazza_yor_family,
    bench_pitman_thm_family,
    bench_ray_knight_family,
)
from quant_fund.research.benches_w816 import (
    bench_cadlag_markov_family,
    bench_characteristic_markov_family,
    bench_generator_markov_family,
    bench_hunt_process_family,
    bench_resolvent_markov_family,
    bench_transition_semigroup_family,
)
from quant_fund.research.benches_w817 import (
    bench_ladder_height_family,
    bench_levy_fluct_family,
    bench_overshoot_levy_family,
    bench_renewal_measure_family,
    bench_spitzer_levy_family,
    bench_wiener_hopf_f_family,
)
from quant_fund.research.benches_w818 import (
    bench_bounded_var_family,
    bench_covariation_family,
    bench_ito_integral_family,
    bench_mart_meas_family,
    bench_stochastic_int2_family,
    bench_vector_mart_family,
)
from quant_fund.research.benches_w819 import (
    bench_accessible_time_family,
    bench_debuts_theorem_family,
    bench_first_hitting_family,
    bench_last_exit_family,
    bench_progressive_set_family,
    bench_stopping_sigma_family,
)
from quant_fund.research.benches_w820 import (
    bench_canonical_decomp_family,
    bench_doom_decomp_family,
    bench_pcdt_family,
    bench_sem_loc_char_family,
    bench_special_sem_family,
    bench_triplet_char_family,
)
from quant_fund.research.benches_w821 import (
    bench_compensator_rm_family,
    bench_integer_measure_family,
    bench_jump_measure_family,
    bench_poisson_rm_family,
    bench_random_measure_family,
    bench_sato_measure_family,
)
from quant_fund.research.benches_w822 import (
    bench_enlargement_f_family,
    bench_initial_enlarg_family,
    bench_natural_filtration_family,
    bench_progressive_enlarg_family,
    bench_right_continuous_f_family,
    bench_usual_aug_family,
)
from quant_fund.research.benches_w823 import (
    bench_cylindrical_law_family,
    bench_finite_dim_family,
    bench_law_convergence_family,
    bench_polish_law_family,
    bench_support_law_family,
    bench_tight_law_family,
)
from quant_fund.research.benches_w824 import (
    bench_convex_order_family,
    bench_first_order_dom_family,
    bench_hazard_rate_order_family,
    bench_second_order_dom_family,
    bench_supermodular_order_family,
    bench_usual_stoch_order_family,
)
from quant_fund.research.benches_w825 import (
    bench_castaing_rep_family,
    bench_integrand_map_family,
    bench_kura_ryll_family,
    bench_measur_select_family,
    bench_measurable_graph_family,
    bench_stoch_open_family,
)
from quant_fund.research.benches_w826 import (
    bench_bj_ineq_family,
    bench_doob_ineq_family,
    bench_etemadi_ineq_family,
    bench_kolmogorov_ineq_family,
    bench_levy_ineq_family,
    bench_max_ineq_family,
)
from quant_fund.research.benches_w827 import (
    bench_chung_series_family,
    bench_ito_nisio_family,
    bench_kolmogorov_two_family,
    bench_ortega_series_family,
    bench_salem_zygmund_family,
    bench_three_series_family,
)
from quant_fund.research.benches_w828 import (
    bench_cross_section_family,
    bench_dellacherie_section_family,
    bench_maharam_lift_family,
    bench_projection_theorem_family,
    bench_uniform_section_family,
    bench_von_neumann_sel_family,
)
from quant_fund.research.benches_w829 import (
    bench_bounded_mart_family,
    bench_cadlag_mart_family,
    bench_decomp_mart_family,
    bench_fv_mart_family,
    bench_local_time_process_family,
    bench_locator_proc_family,
)
from quant_fund.research.benches_w830 import (
    bench_azuma_ineq_family,
    bench_bounded_diff_family,
    bench_efron_stein_family,
    bench_hoeffding_ineq_family,
    bench_mcdiarmid_ineq_family,
    bench_talagrand_ineq_family,
)
from quant_fund.research.benches_w831 import (
    bench_covering_number_family,
    bench_entropy_integral_family,
    bench_metric_entropy_family,
    bench_rademacher_cplx_family,
    bench_symmetrization_family,
    bench_uniform_clt_family,
)
from quant_fund.research.benches_w832 import (
    bench_continuous_map_family,
    bench_delta_method_family,
    bench_empirical_bridge_family,
    bench_kmt_approx_family,
    bench_porte_manteau_family,
    bench_skorohod_embed_family,
)
from quant_fund.research.benches_w833 import (
    bench_brownian_approx_family,
    bench_donsker_invariance_family,
    bench_fclt_invariance_family,
    bench_martingale_fclt_family,
    bench_stable_limit_family,
    bench_strassen_flln_family,
)
from quant_fund.research.benches_w834 import (
    bench_dirichlet_form_family,
    bench_hypercontractive_family,
    bench_log_sobolev_sem_family,
    bench_markov_semigroup_family,
    bench_poincare_semigroup_family,
    bench_spectral_gap_sem_family,
)
from quant_fund.research.benches_w835 import (
    bench_boolean_model_family,
    bench_germ_grain_family,
    bench_intrinsic_volumes_family,
    bench_miles_matheron_family,
    bench_poisson_voronoi_family,
    bench_steiner_formula_family,
)
from quant_fund.research.benches_w836 import (
    bench_buffon_needle_family,
    bench_crofton_formula_family,
    bench_hadwiger_chars_family,
    bench_kinematic_measure_family,
    bench_kubota_mean_width_family,
    bench_santalo_measure_family,
)
from quant_fund.research.benches_w837 import (
    bench_alexandrov_fenchel_family,
    bench_brunn_minkowski_family,
    bench_helly_theorem_family,
    bench_isoperimetric_ineq_family,
    bench_minkowski_sum_family,
    bench_mixed_volume_family,
)
from quant_fund.research.benches_w838 import (
    bench_caratheodory_thm_family,
    bench_farkas_lemma_family,
    bench_lattice_point_family,
    bench_radon_theorem_family,
    bench_separation_thm_family,
    bench_tverberg_thm_family,
)
from quant_fund.research.benches_w839 import (
    bench_bernstein_poly_family,
    bench_chebyshev_alternation_family,
    bench_fourier_decay_family,
    bench_jackson_direct_family,
    bench_kolmogorov_nwidth_family,
    bench_markov_brothers_family,
)
from quant_fund.research.benches_w840 import (
    bench_loewner_interp_family,
    bench_nevanlinna_pick_family,
    bench_pade_approx_family,
    bench_rational_chebyshev_family,
    bench_schur_continued_family,
    bench_stieltjes_fraction_family,
)
from quant_fund.research.benches_w841 import (
    bench_chebyshev_t_family,
    bench_gegenbauer_poly_family,
    bench_hermite_poly_family,
    bench_jacobi_poly_family,
    bench_laguerre_poly_family,
    bench_legendre_poly_family,
)
from quant_fund.research.benches_w842 import (
    bench_chebyshev_collocation_family,
    bench_chebyshev_grid_family,
    bench_dealiasing_family,
    bench_fourier_galerkin_family,
    bench_legendre_tau_family,
    bench_spectral_deriv_family,
)
from quant_fund.research.benches_w843 import (
    bench_b_spline_family,
    bench_blossoming_family,
    bench_box_spline_family,
    bench_cardinal_spline_family,
    bench_de_boor_family,
    bench_knot_insertion_family,
)
from quant_fund.research.benches_w844 import (
    bench_asymptotic_series_family,
    bench_borel_resum_family,
    bench_poincare_expansion_family,
    bench_stationary_phase_family,
    bench_steepest_descent_family,
    bench_wkb_approx_family,
)
from quant_fund.research.benches_w845 import (
    bench_abel_transform_family,
    bench_hankel_transform_family,
    bench_hilbert_transform_family,
    bench_laplace_transform_family,
    bench_mellin_transform_family,
    bench_z_transform_family,
)
from quant_fund.research.benches_w846 import (
    bench_airy_fn_family,
    bench_bessel_fn_family,
    bench_beta_fn_family,
    bench_error_fn_family,
    bench_gamma_fn_family,
    bench_hypergeometric_fn_family,
)
from quant_fund.research.benches_w847 import (
    bench_boundary_layer_family,
    bench_lindstedt_poincare_family,
    bench_matched_asymptotic_family,
    bench_multiple_scales_family,
    bench_regular_perturbation_family,
    bench_singular_perturbation_family,
)
from quant_fund.research.benches_w848 import (
    bench_dof_management_family,
    bench_edge_elements_family,
    bench_fem_assembly_family,
    bench_isoparametric_map_family,
    bench_quadrature_rules_family,
    bench_triangular_basis_family,
)
from quant_fund.research.benches_w849 import (
    bench_compact_scheme_family,
    bench_crank_nicholson2_family,
    bench_fdm_grid_family,
    bench_flux_splitting_family,
    bench_muscl_reconstruct_family,
    bench_upwind_scheme_family,
)
from quant_fund.research.benches_w850 import (
    bench_ausm_flux_family,
    bench_godunov_exact_family,
    bench_hllc_solver_family,
    bench_lax_friedrichs_family,
    bench_osher_solver_family,
    bench_roe_solver_family,
)
from quant_fund.research.benches_w851 import (
    bench_dg_discretization_family,
    bench_limiter_tvb_family,
    bench_modal_basis_family,
    bench_numerical_flux_dg_family,
    bench_penalty_dg_family,
    bench_rkdg_step_family,
)
from quant_fund.research.benches_w852 import (
    bench_bem_kernel_family,
    bench_fast_multipole_family,
    bench_fredholm_solve_family,
    bench_galerkin_bem_family,
    bench_nystrom_method_family,
    bench_singular_integrals_family,
)
from quant_fund.research.benches_w853 import (
    bench_gll_nodes_family,
    bench_hp_refinement_family,
    bench_mortar_method_family,
    bench_sem_grid_family,
    bench_spectral_element_family,
    bench_tensor_product_sem_family,
)
from quant_fund.research.benches_w854 import (
    bench_gaussian_rbf_family,
    bench_kansa_collocation_family,
    bench_multiquadric_rbf_family,
    bench_rbf_finite_diff_family,
    bench_rbf_interp_family,
    bench_wendland_rbf_family,
)
from quant_fund.research.benches_w855 import (
    bench_adapt_wavelet_family,
    bench_coiflet_basis_family,
    bench_daubechies_basis_family,
    bench_spline_wavelet_family,
    bench_wavelet_collocation_family,
    bench_wavelet_galerkin_family,
)
from quant_fund.research.benches_w856 import (
    bench_halton_seq_family,
    bench_latin_hypercube_family,
    bench_monte_carlo_quad_family,
    bench_quasi_mc_family,
    bench_sobol_seq_family,
    bench_stratified_mc_family,
)
from quant_fund.research.benches_w857 import (
    bench_clenshaw_curtis_family,
    bench_fejer_quad_family,
    bench_gauss_chebyshev_family,
    bench_gauss_kronrod_family,
    bench_gauss_legendre_family,
    bench_newton_cotes_family,
)
from quant_fund.research.benches_w858 import (
    bench_adaptive_simpsons_family,
    bench_double_exp_quad_family,
    bench_filon_quad_family,
    bench_levin_quad_family,
    bench_osc_singular_family,
    bench_tanh_sinh_family,
)
from quant_fund.research.benches_w859 import (
    bench_diffuse_element_family,
    bench_hp_clouds_family,
    bench_meshless_local_family,
    bench_mls_shape_family,
    bench_moving_least_sq_family,
    bench_point_cloud_interp_family,
)
from quant_fund.research.benches_w860 import (
    bench_cut_cell_family,
    bench_fictitious_domain_family,
    bench_immersed_boundary_family,
    bench_iso_geom_family,
    bench_nurbs_elem_family,
    bench_xfem_family,
)
from quant_fund.research.benches_w861 import (
    bench_ars_imex_family,
    bench_dirk_scheme_family,
    bench_exponential_euler_family,
    bench_imex_rk_family,
    bench_rosenbrock_w_family,
    bench_ssp_rk_family,
)
from quant_fund.research.benches_w862 import (
    bench_anisotropic_quad_family,
    bench_combination_technique_family,
    bench_dimension_adaptive_family,
    bench_gerstner_griebel_family,
    bench_smolyak_grid_family,
    bench_sparse_tensor_family,
)
from quant_fund.research.benches_w863 import (
    bench_bddc_lite_family,
    bench_coarse_correction_family,
    bench_feti_lite_family,
    bench_mortar_dd_family,
    bench_schwarz_add_family,
    bench_schwarz_mult_family,
)
from quant_fund.research.benches_w864 import (
    bench_intrusive_pce_family,
    bench_nonintrusive_pce_family,
    bench_poly_chaos_uq_family,
    bench_stochastic_colloc_family,
    bench_stochastic_fem_family,
    bench_stochastic_galerkin_family,
)
from quant_fund.research.benches_w865 import (
    bench_dual_weighted_res_family,
    bench_equilibrated_flux_family,
    bench_goal_oriented_family,
    bench_recovery_error_family,
    bench_residual_estimator_family,
    bench_zienkiewicz_zhu_family,
)
from quant_fund.research.benches_w866 import (
    bench_deim_point_family,
    bench_eim_interp_family,
    bench_greedy_rb_family,
    bench_pod_galerkin_family,
    bench_proper_gen_family,
    bench_reduced_basis_family,
)
from quant_fund.research.benches_w867 import (
    bench_bayes_inverse_family,
    bench_iter_regularize_family,
    bench_l_curve_opt_family,
    bench_morozov_dp_family,
    bench_tikhonov_reg_family,
    bench_tv_denoise_family,
)
from quant_fund.research.benches_w868 import (
    bench_arc_continuation_family,
    bench_bifurcation_track_family,
    bench_davidenko_ode_family,
    bench_deflation_method_family,
    bench_homotopy_solver_family,
    bench_pseudo_arclength_family,
)
from quant_fund.research.benches_w869 import (
    bench_antithetic_var_family,
    bench_common_random_family,
    bench_conditional_mc_family,
    bench_control_variate_family,
    bench_importance_sampling_family,
    bench_stratified_var_family,
)
from quant_fund.research.benches_w870 import (
    bench_augmented_lagrangian_family,
    bench_conjugate_opt_family,
    bench_grad_descent_nest_family,
    bench_interior_point2_family,
    bench_newton_method_family,
    bench_quasi_newton_lbfgs_family,
)
from quant_fund.research.benches_w871 import (
    bench_arnoldi_eig_family,
    bench_bicg_solver_family,
    bench_cg_solver_family,
    bench_gmres_solver_family,
    bench_lanczos_eig_family,
    bench_lsqr_solver_family,
)
from quant_fund.research.benches_w872 import (
    bench_amg_precond_family,
    bench_ic_precond_family,
    bench_ilut_precond_family,
    bench_jacobi_precond_family,
    bench_polynomial_precond_family,
    bench_ssor_precond_family,
)
from quant_fund.research.benches_w873 import (
    bench_asm_precond_family,
    bench_baldding_dd_family,
    bench_dd_partition_family,
    bench_feti_dp_family,
    bench_neumann_dd_family,
    bench_subspace_dd_family,
)
from quant_fund.research.benches_w874 import (
    bench_first_order_rel_family,
    bench_line_sampling_family,
    bench_metamodel_rel_family,
    bench_sorm_method_family,
    bench_subset_sim_family,
    bench_uq_reliability_family,
)
from quant_fund.research.benches_w875 import (
    bench_adaptive_finite_family,
    bench_adaptive_marking_family,
    bench_convergence_theory_family,
    bench_dorfler_marking_family,
    bench_goal_adaptive_family,
    bench_hierarchical_est_family,
)
from quant_fund.research.benches_w876 import (
    bench_anderson_mixing_family,
    bench_conjugate_grad_ls_family,
    bench_gauss_newton_family,
    bench_landweber_iter_family,
    bench_levenberg_marq_family,
    bench_moore_penrose_family,
)
from quant_fund.research.benches_w877 import (
    bench_discrete_ordinates_family,
    bench_moc_transport_family,
    bench_pn_closure_family,
    bench_spherical_harmonics_family,
    bench_spn_equations_family,
    bench_transport_sn_family,
)
from quant_fund.research.benches_w878 import (
    bench_block_low_rank_family,
    bench_h_matrix_family,
    bench_hss_matrix_family,
    bench_kronecker_approx_family,
    bench_low_rank_svd_family,
    bench_randomized_nystrom_family,
)
from quant_fund.research.benches_w879 import (
    bench_etd_rk4_classic_family,
    bench_expm_int_family,
    bench_expokit_family,
    bench_krylov_subspace_time_family,
    bench_leja_point_family,
    bench_phi_function_family,
)
from quant_fund.research.benches_w880 import (
    bench_adaptive_quad2_family,
    bench_cubature_rule_family,
    bench_empirical_interp_family,
    bench_gq_adaptive_family,
    bench_pod_deim_family,
    bench_tensor_interp_family,
)
from quant_fund.research.benches_w881 import (
    bench_bicgstab2_family,
    bench_block_cg_family,
    bench_cgs_solver_family,
    bench_minres_solver_family,
    bench_qmr_solver_family,
    bench_tfqmr_family,
)
from quant_fund.research.benches_w882 import (
    bench_balanced_dd_family,
    bench_diagonal_scale_family,
    bench_nonoverlap_dd_family,
    bench_overlap_dd_family,
    bench_restrictive_dd_family,
    bench_spai_precond_family,
)
from quant_fund.research.benches_w883 import (
    bench_cv_optimal_family,
    bench_is_drift_family,
    bench_min_var_closure_family,
    bench_nest_accel_family,
    bench_subgradient_descent_family,
    bench_tangent_predictor_family,
)
from quant_fund.research.benches_w884 import (
    bench_boundary_element_family,
    bench_marquina_flux_family,
    bench_multidomain_sem_family,
    bench_nodal_dg_family,
    bench_second_gen_wavelet_family,
    bench_wavelet_matrix_family,
)
from quant_fund.research.benches_w885 import (
    bench_averaging_method_family,
    bench_entropy_stable_dg_family,
    bench_hyperasymptotic_family,
    bench_laplace_method_family,
    bench_ldg_flux_family,
    bench_wkb_turning_family,
)
from quant_fund.research.benches_w886 import (
    bench_faure_seq_family,
    bench_gauss_hermite_family,
    bench_gauss_laguerre_family,
    bench_hiot_decomp_family,
    bench_importance_mc_family,
    bench_tensor_train_family,
)
from quant_fund.research.benches_w887 import (
    bench_adjoint_sparse_family,
    bench_diffusion_approx_sp_family,
    bench_element_free_family,
    bench_epi_rk_family,
    bench_gauss_rk_family,
    bench_importance_rel_family,
)
from quant_fund.research.benches_w888 import (
    bench_galerkin_projection_family,
    bench_periodic_spline_family,
    bench_polyharmonic_rbf_family,
    bench_thin_plate_spline_family,
    bench_trefethen_diff_family,
    bench_zernike_poly_family,
)
from quant_fund.research.benches_w889 import (
    bench_chebyshev_u_family,
    bench_epsilon_algo_family,
    bench_hexahedral_basis_family,
    bench_quadrilateral_basis_family,
    bench_spline_theory_family,
    bench_walsh_table_family,
)
from quant_fund.research.benches_w890 import (
    bench_clenshaw_quad_family,
    bench_elliptic_fn_family,
    bench_fejer_nested_family,
    bench_hartley_transform_family,
    bench_radon_transform_family,
    bench_zeta_fn_family,
)
from quant_fund.research.benches_w891 import (
    bench_bfgs_update_family,
    bench_iga_colloc_family,
    bench_lebesgue_const_family,
    bench_newton_armijo_family,
    bench_trimmed_cad_family,
    bench_trust_region_dogleg_family,
)
from quant_fund.research.benches_w892 import (
    bench_form_analysis_family,
    bench_greedy_marking_family,
    bench_hp_adaptive_family,
    bench_residual_marking_family,
    bench_space_time_adapt_family,
    bench_wavelet_adapt_family,
)
from quant_fund.research.benches_w893 import (
    bench_coarsening_mark_family,
    bench_covello_est_family,
    bench_dual_goal_est_family,
    bench_galerkin_least_sq_family,
    bench_pseudospectral_coll_family,
    bench_tau_method_family,
)
from quant_fund.research.benches_w894 import (
    bench_barycentric_wts_family,
    bench_divid_diff_table_family,
    bench_floater_hormann_family,
    bench_hermite_interp_family,
    bench_lagrange_interp_family,
    bench_neville_interp_family,
)
from quant_fund.research.benches_w895 import (
    bench_aitken_steffensen_family,
    bench_bulirsch_stoer_family,
    bench_muller_root_family,
    bench_regula_falsi_family,
    bench_richardson_limit_family,
    bench_secant_root_family,
)
from quant_fund.research.benches_w896 import (
    bench_backward_euler_family,
    bench_bogacki_shampine_family,
    bench_cash_karp_family,
    bench_dormand_prince_family,
    bench_fehlberg_rk_family,
    bench_predictor_corrector_family,
)
from quant_fund.research.benches_w897 import (
    bench_dahlquist_test_family,
    bench_explicit_midpoint_family,
    bench_heun_method_family,
    bench_linear_multistep_family,
    bench_order_barrier_family,
    bench_trapezoid_rule_family,
)
from quant_fund.research.benches_w898 import (
    bench_collocation_bvp_family,
    bench_finite_diff_bvp_family,
    bench_multiple_shooting_family,
    bench_relaxation_bvp_family,
    bench_riccati_bvp_family,
    bench_shooting_bvp_family,
)
from quant_fund.research.benches_w899 import (
    bench_bernstein_form_family,
    bench_cardinal_interp_family,
    bench_chebyshev_interp_family,
    bench_osculating_interp_family,
    bench_rational_interp_family,
    bench_shanks_trans_family,
)
from quant_fund.research.benches_w900 import (
    bench_ball_tree_family,
    bench_cover_tree_family,
    bench_kd_tree_family,
    bench_quad_tree_family,
    bench_r_tree_family,
    bench_vp_tree_family,
)
from quant_fund.research.benches_w901 import (
    bench_binary_heap_family,
    bench_binomial_heap_family,
    bench_fibonacci_heap_family,
    bench_leftist_heap_family,
    bench_pairing_heap_family,
    bench_skew_heap_family,
)
from quant_fund.research.benches_w902 import (
    bench_crit_bit_tree_family,
    bench_patricia_trie_family,
    bench_radix_trie_family,
    bench_suffix_trie_family,
    bench_ternary_trie_family,
    bench_trie_family,
)
from quant_fund.research.benches_w903 import (
    bench_cuckoo_hash_family,
    bench_hopscotch_hash_family,
    bench_open_addr_hash_family,
    bench_perfect_hash_family,
    bench_robin_hood_hash_family,
    bench_swiss_table_family,
)
from quant_fund.research.benches_w904 import (
    bench_aa_tree_family,
    bench_avl_tree_family,
    bench_red_black_tree_family,
    bench_scapegoat_tree_family,
    bench_splay_tree_family,
    bench_treap_family,
)
from quant_fund.research.benches_w905 import (
    bench_heapsort_family,
    bench_introsort_family,
    bench_mergesort_family,
    bench_quicksort_family,
    bench_radix_sort_family,
    bench_timsort_family,
)
from quant_fund.research.benches_w906 import (
    bench_finger_tree_family,
    bench_persistent_array_family,
    bench_pure_queue_family,
    bench_rope_string_family,
    bench_skip_list_family,
    bench_vlist_family,
)
from quant_fund.research.benches_w907 import (
    bench_dsu_rollback_family,
    bench_interval_heap_family,
    bench_potential_dsu_family,
    bench_union_find_family,
    bench_van_emde_boas_family,
    bench_weak_heap_family,
)
from quant_fund.research.benches_w908 import (
    bench_fenwick_tree_family,
    bench_merge_sort_tree_family,
    bench_segment_tree_family,
    bench_sparse_table_family,
    bench_sqrt_decomp_family,
    bench_wavelet_tree_family,
)
from quant_fund.research.benches_w909 import (
    bench_deque_array_family,
    bench_doubly_linked_list_family,
    bench_gap_buffer_family,
    bench_piece_table_family,
    bench_unrolled_list_family,
    bench_xor_linked_list_family,
)
from quant_fund.research.benches_w910 import (
    bench_b_plus_tree_family,
    bench_b_star_tree_family,
    bench_b_tree_family,
    bench_tango_tree_family,
    bench_wavl_tree_family,
    bench_weight_balanced_tree_family,
)
from quant_fund.research.benches_w911 import (
    bench_bezier_eval_family,
    bench_chan_hull_family,
    bench_cohen_sutherland_family,
    bench_gift_wrap_family,
    bench_liang_barsky_family,
    bench_monotone_chain_family,
)
from quant_fund.research.benches_w912 import (
    bench_hilbert_curve_family,
    bench_morton_order_family,
    bench_octree_index_family,
    bench_range_tree_family,
    bench_rstar_tree_family,
    bench_z_curve_family,
)
from quant_fund.research.benches_w913 import (
    bench_akima_interp_family,
    bench_makima_interp_family,
    bench_monotone_interp_family,
    bench_pchip_interp_family,
    bench_scattered_interp_family,
    bench_spline_interp_family,
)
from quant_fund.research.benches_w914 import (
    bench_green_function_bvp_family,
    bench_invariant_imbedding_family,
    bench_ralston_rk_family,
    bench_ralston_second_family,
    bench_runge_kutta4_family,
    bench_verner_rk_family,
)
from quant_fund.research.benches_w915 import (
    bench_bvp_eigen_family,
    bench_continuation_bvp_family,
    bench_fusion_tree_family,
    bench_loser_tree_family,
    bench_robbins_bvp_family,
    bench_superposition_bvp_family,
)
from quant_fund.research.benches_w916 import (
    bench_da_trie_family,
    bench_fst_index_family,
    bench_hollow_dsu_family,
    bench_hollow_heap_family,
    bench_rank_pairing_family,
    bench_soft_heap_family,
)
from quant_fund.research.benches_w917 import (
    bench_bucket_sort_family,
    bench_chained_hash_family,
    bench_linear_probe_family,
    bench_rand_access_list_family,
    bench_shell_sort_family,
    bench_skew_list_family,
)
from quant_fund.research.benches_w918 import (
    bench_fractional_cascade_family,
    bench_free_list_family,
    bench_halfplane_isect_family,
    bench_object_pool_family,
    bench_range_min_query_family,
    bench_welzl_circle_family,
)
from quant_fund.research.benches_w919 import (
    bench_convex_layers_family,
    bench_delaunay_flip_family,
    bench_polygon_offset_family,
    bench_rotating_sweep_family,
    bench_visibility_graph_family,
    bench_voronoi_lite_family,
)
from quant_fund.research.benches_w920 import (
    bench_alpha_shape_family,
    bench_diameter_pair_family,
    bench_min_area_rect_family,
    bench_minkowski_sum_poly_family,
    bench_monotone_partition_family,
    bench_polygon_triangulate_family,
)
from quant_fund.research.benches_w921 import (
    bench_atomic_bcast_family,
    bench_avalanche_consensus_family,
    bench_honey_badger_family,
    bench_isis_bcast_family,
    bench_snowball_consensus_family,
    bench_virtual_synchrony_family,
)
from quant_fund.research.benches_w922 import (
    bench_abcast_lite_family,
    bench_cap_theorem_family,
    bench_cbc_bcast_family,
    bench_lake_wisc_family,
    bench_slush_consensus_family,
    bench_snowflake_consensus_family,
)
from quant_fund.research.benches_w923 import (
    bench_chinese_restaurant_family,
    bench_dirichlet_process_family,
    bench_hierarchical_dp_family,
    bench_indian_buffet_family,
    bench_pitman_yor_family,
    bench_stick_breaking_family,
)
from quant_fund.research.benches_w924 import (
    bench_beta_bernoulli_family,
    bench_crp_table_family,
    bench_dp_mm_family,
    bench_gem_distribution_family,
    bench_gibbs_type_family,
    bench_neutral_process_family,
)
from quant_fund.research.benches_w925 import (
    bench_bondesson_shot_family,
    bench_exchangeable_pf_family,
    bench_kingman_paintbox_family,
    bench_nggp_process_family,
    bench_normalized_rm_family,
    bench_sigma_stable_family,
)
from quant_fund.research.benches_w926 import (
    bench_alpha_divergence_family,
    bench_amari_connection_family,
    bench_csiszar_div_family,
    bench_dual_connection_family,
    bench_f_divergence_family,
    bench_tsallis_entropy_family,
)
from quant_fund.research.benches_w927 import (
    bench_ebanch_diverge_family,
    bench_expectation_param_family,
    bench_fisher_metric2_family,
    bench_potential_fn_family,
    bench_renyi_div_family,
    bench_shannon_gibbs_family,
)
from quant_fund.research.benches_w928 import (
    bench_beta_skeleton_family,
    bench_convex_hull_3d_family,
    bench_medial_axis_family,
    bench_polygon_boolean_family,
    bench_polygon_centroid_family,
    bench_shape_context_family,
)
from quant_fund.research.benches_w929 import (
    bench_bhat_distance_family,
    bench_chi_square_div_family,
    bench_d_total_var_family,
    bench_hellinger_dist_family,
    bench_jeffreys_div_family,
    bench_mahalanobis_div_family,
)
from quant_fund.research.benches_w930 import (
    bench_grasp_meta_family,
    bench_iterated_local_family,
    bench_lin_kernighan_family,
    bench_tabu_search_family,
    bench_three_opt_move_family,
    bench_two_opt_move_family,
)
from quant_fund.research.benches_w931 import (
    bench_ant_colony_family,
    bench_diff_evolution_family,
    bench_firefly_algo_family,
    bench_genetic_tsp_family,
    bench_harmony_search_family,
    bench_pso_swarm_family,
)
from quant_fund.research.benches_w932 import (
    bench_guided_local_family,
    bench_large_neighborhood_family,
    bench_path_relinking_family,
    bench_ruin_recreate_family,
    bench_simulated_annealing_family,
    bench_vns_search_family,
)
from quant_fund.research.benches_w933 import (
    bench_bregman_proj_family,
    bench_conjugate_fn_family,
    bench_fenchel_dual_family,
    bench_moreau_env_family,
    bench_proximal_map_family,
    bench_subgradient_proj_family,
)
from quant_fund.research.benches_w934 import (
    bench_inf_convolution_family,
    bench_legendre_transform_family,
    bench_normal_cone_family,
    bench_perspective_fn_family,
    bench_polar_cone_family,
    bench_support_fn_family,
)
from quant_fund.research.benches_w935 import (
    bench_analytic_center_family,
    bench_cvx_reform_family,
    bench_dik_ellipsoid_family,
    bench_kkt_solve_family,
    bench_logbarrier_fn_family,
    bench_self_concordant_family,
)
from quant_fund.research.benches_w936 import (
    bench_bundle_level_family,
    bench_clarke_subdiff_family,
    bench_epigraph_proj_family,
    bench_gauge_duality_family,
    bench_gauge_fn_family,
    bench_subdiff_compute_family,
)
from quant_fund.research.benches_w937 import (
    bench_chambolle_pock_family,
    bench_davis_yin_family,
    bench_douglas_rachford_family,
    bench_forward_backward_family,
    bench_peaceman_rachford_family,
    bench_tseng_split_family,
)
from quant_fund.research.benches_w938 import (
    bench_averaged_operator_family,
    bench_cocoercive_family,
    bench_fejer_monotone_family,
    bench_firmly_nonexpansive_family,
    bench_monotone_inclusion_family,
    bench_quasinonexpansive_family,
)
from quant_fund.research.benches_w939 import (
    bench_cq_algorithm_family,
    bench_dykstra_proj_family,
    bench_halpern_iter_family,
    bench_haugazeau_proj_family,
    bench_parallel_prox_family,
    bench_split_feasibility_family,
)
from quant_fund.research.benches_w940 import (
    bench_forward_reflected_family,
    bench_korpelevich_eg_family,
    bench_popov_alg_family,
    bench_reflected_golden_family,
    bench_subgradient_extragradient_family,
    bench_tseng_fb_family,
)
from quant_fund.research.benches_w941 import (
    bench_augmented_lagr_family,
    bench_ekeland_var_family,
    bench_limiting_subdiff_family,
    bench_monteiro_semismooth_family,
    bench_proximal_subdiff_family,
    bench_semismooth_newton_family,
)
from quant_fund.research.benches_w942 import (
    bench_backward_forward_family,
    bench_ishikawa_iter_family,
    bench_malitsky_golden_family,
    bench_mann_iter_family,
    bench_primal_dual_hybrid_family,
    bench_vu_condat_family,
)
from quant_fund.research.benches_w943 import (
    bench_gershgorin_disc_family,
    bench_kadison_ineq_family,
    bench_loewner_matrix_family,
    bench_operator_convex_family,
    bench_ostrowski_bound_family,
    bench_wielandt_ineq_family,
)
from quant_fund.research.benches_w944 import (
    bench_cauchy_binet_family,
    bench_fan_inequality_family,
    bench_horn_inequality_family,
    bench_majorization_vec_family,
    bench_schur_complement_family,
    bench_weyl_ineq_family,
)
from quant_fund.research.benches_w945 import (
    bench_hankel_op_family,
    bench_kyfan_norm_family,
    bench_matrix_det_family,
    bench_numerical_radius_family,
    bench_pfaffian_poly_family,
    bench_schatten_norm_family,
)
from quant_fund.research.benches_w946 import (
    bench_cholesky_piv_family,
    bench_douglas_factor_family,
    bench_matrix_square_root_family,
    bench_perron_frobenius_family,
    bench_polar_decomp_family,
    bench_sylvester_matrix_family,
)
from quant_fund.research.benches_w947 import (
    bench_bezout_matrix_family,
    bench_cauchy_interlace_family,
    bench_haynsworth_inertia_family,
    bench_min_max_eig_family,
    bench_sturm_sequence_family,
    bench_sylvester_law_family,
)
from quant_fund.research.benches_w948 import (
    bench_circulant_matrix_family,
    bench_companion_matrix_family,
    bench_hankel_matrix_family,
    bench_hessenberg_form_family,
    bench_krylov_matrix_family,
    bench_vandermonde_matrix_family,
)
from quant_fund.research.benches_w949 import (
    bench_deflating_subspace_family,
    bench_invariant_subspace_family,
    bench_jordan_form_family,
    bench_kronecker_canonical_family,
    bench_matrix_pencil_family,
    bench_rational_canonical_family,
)
from quant_fund.research.benches_w950 import (
    bench_hadamard_product_family,
    bench_khatri_rao_family,
    bench_kron_product_family,
    bench_outer_product_family,
    bench_tensor_contraction_family,
    bench_tensor_unfold_family,
)
from quant_fund.research.benches_w951 import (
    bench_determinant_cofactor_family,
    bench_frechet_derivative_family,
    bench_kronecker_sum_family,
    bench_matrix_exponential_family,
    bench_permanent_matrix_family,
    bench_vec_operator_family,
)
from quant_fund.research.benches_w952 import (
    bench_eigval_bounds_family,
    bench_power_deflation_family,
    bench_qr_iteration_family,
    bench_schur_decomp_family,
    bench_spectral_gap_family,
    bench_spectral_radius_family,
)
from quant_fund.research.benches_w953 import (
    bench_bounded_operator_family,
    bench_isometry_operator_family,
    bench_operator_adjoint_family,
    bench_operator_norm_family,
    bench_positive_operator_family,
    bench_projection_operator_family,
)
from quant_fund.research.benches_w954 import (
    bench_condition_number_family,
    bench_low_rank_approx_family,
    bench_matrix_truncate_family,
    bench_nuclear_norm_family,
    bench_rank_estimate_family,
    bench_spectral_threshold_family,
)
from quant_fund.research.benches_w955 import (
    bench_back_substitution_family,
    bench_forward_substitution_family,
    bench_givens_rotation_family,
    bench_gram_determinant_family,
    bench_gram_matrix_family,
    bench_householder_reflect_family,
)
from quant_fund.research.benches_w956 import (
    bench_cp_rank_family,
    bench_mode_n_product_family,
    bench_tensor_norm_family,
    bench_tensor_symmetry_family,
    bench_tensor_trace_family,
    bench_tucker_rank_family,
)
from quant_fund.research.benches_w957 import (
    bench_fredholm_op_family,
    bench_multiplication_op_family,
    bench_normal_operator_family,
    bench_selfadjoint_op_family,
    bench_shift_operator_family,
    bench_unitary_operator_family,
)
from quant_fund.research.benches_w958 import (
    bench_araki_lieb_thirring_family,
    bench_hadamard_fischer_family,
    bench_ky_fan_family,
    bench_lidskii_thm_family,
    bench_pinching_ineq_family,
    bench_von_neumann_trace_family,
)
from quant_fund.research.benches_w959 import (
    bench_accretive_op_family,
    bench_contraction_op_family,
    bench_differential_op_family,
    bench_integral_op_family,
    bench_sectorial_op_family,
    bench_toeplitz_op_family,
)
from quant_fund.research.benches_w960 import (
    bench_banach_algebra_family,
    bench_c_star_algebra_family,
    bench_gelfand_transform_family,
    bench_holomorphic_calculus_family,
    bench_positive_functional_family,
    bench_spectrum_algebra_family,
)
from quant_fund.research.benches_w961 import (
    bench_analytic_semigroup_family,
    bench_c0_semigroup_family,
    bench_cosine_family_family,
    bench_hille_yosida_family,
    bench_lumer_phillips_family,
    bench_trotter_kato_family,
)
from quant_fund.research.benches_w962 import (
    bench_double_commutant_family,
    bench_jones_index_family,
    bench_normal_state_family,
    bench_predual_space_family,
    bench_tomita_takesaki_family,
    bench_von_neumann_alg_family,
)
from quant_fund.research.benches_w963 import (
    bench_atkinson_thm_family,
    bench_browder_operator_family,
    bench_essential_spectrum_family,
    bench_fredholm_index_family,
    bench_riesz_schauder_family,
    bench_weyl_theorem_family,
)
from quant_fund.research.benches_w964 import (
    bench_adjoint_unbounded_family,
    bench_closed_operator_family,
    bench_domain_dense_family,
    bench_resolvent_op_family,
    bench_spectral_measure_family,
    bench_unbounded_operator_family,
)
from quant_fund.research.benches_w965 import (
    bench_compact_normal_family,
    bench_hilbert_schmidt_op_family,
    bench_polar_operator_family,
    bench_schmidt_decomp_family,
    bench_singular_value_op_family,
    bench_trace_class_op_family,
)
from quant_fund.research.benches_w966 import (
    bench_cb_map_family,
    bench_complete_contraction_family,
    bench_injective_space_family,
    bench_noncommutative_lp_family,
    bench_oh_emb_family,
    bench_operator_space_family,
)
from quant_fund.research.benches_w967 import (
    bench_crossed_product_family,
    bench_cstar_dynamics_family,
    bench_kirchberg_absorb_family,
    bench_rokhlin_action_family,
    bench_taf_dim_family,
    bench_z_stability_family,
)
from quant_fund.research.benches_w968 import (
    bench_baaj_julg_family,
    bench_cuntz_picture_family,
    bench_ext_functor_family,
    bench_kasparov_prod_family,
    bench_kk_duality_family,
    bench_kk_theory_family,
)
from quant_fund.research.benches_w969 import (
    bench_bott_periodicity_k_family,
    bench_elliott_invariant_family,
    bench_k0_algebra_family,
    bench_k1_algebra_family,
    bench_pimsner_voicul_family,
    bench_six_term_exact_family,
)
from quant_fund.research.benches_w970 import (
    bench_fusion_algebra_family,
    bench_paragroup_family,
    bench_planar_algebra_family,
    bench_principal_graph_family,
    bench_standard_invariant_family,
    bench_subfactor_family,
)
from quant_fund.research.benches_w971 import (
    bench_connes_metric_family,
    bench_differential_form_nc_family,
    bench_geodesic_nc_family,
    bench_hochschild_cycle_family,
    bench_index_pairing_family,
    bench_spectral_triple_family,
)
from quant_fund.research.benches_w972 import (
    bench_free_berg_family,
    bench_free_cumulant_family,
    bench_free_entropy_family,
    bench_free_fisher_info_family,
    bench_freeness_check_family,
    bench_matrix_model_free_family,
)
from quant_fund.research.benches_w973 import (
    bench_banach_mazur_family,
    bench_djt_space_family,
    bench_gl_property_family,
    bench_kalton_loc_family,
    bench_schauder_basis_family,
    bench_type_cotype_family,
)
from quant_fund.research.benches_w974 import (
    bench_complex_interp_family,
    bench_lorentz_space_family,
    bench_marcinkiewicz_interp_family,
    bench_peetre_kfunctor_family,
    bench_real_interp_k_family,
    bench_reiteration_thm_family,
)
from quant_fund.research.benches_w975 import (
    bench_dist_convolution_family,
    bench_paley_wiener_family,
    bench_schwartz_dist_family,
    bench_sing_support_family,
    bench_sobolev_trace_family,
    bench_temper_dist_family,
)
from quant_fund.research.benches_w976 import (
    bench_diam_dim_family,
    bench_frechet_nuclear_family,
    bench_gelfand_triple_family,
    bench_hilbert_schmidt_emb_family,
    bench_nuclear_map_family,
    bench_trace_duality_family,
)
from quant_fund.research.benches_w977 import (
    bench_bochner_integral_family,
    bench_bochner_meas_family,
    bench_lusin_rep_family,
    bench_norm_integrable_family,
    bench_pettis_weak_family,
    bench_radon_nikodym_prop_family,
)
from quant_fund.research.benches_w978 import (
    bench_bochner_riesz_family,
    bench_hausdorff_young_family,
    bench_lp_multiplier_family,
    bench_oscillatory_int_family,
    bench_restriction_est_family,
    bench_strichartz_est_family,
)
from quant_fund.research.benches_w979 import (
    bench_disjointness_dyn_family,
    bench_horocycle_flow_family,
    bench_ratner_thm_family,
    bench_unipotent_ergodic_family,
    bench_van_der_corput_family,
    bench_weyl_equidist_family,
)
from quant_fund.research.benches_w980 import (
    bench_de_boor_stable_family,
    bench_faber_schauder_family,
    bench_haar_system_family,
    bench_korovkin_thm_family,
    bench_walsh_series_family,
    bench_whitney_ext_family,
)
from quant_fund.research.benches_w981 import (
    bench_grothendieck_const_family,
    bench_john_ellipsoid_family,
    bench_kadison_singer_family,
    bench_loewner_ellipsoid_family,
    bench_milman_isotropic_family,
    bench_milman_rev_thm_family,
)
from quant_fund.research.benches_w982 import (
    bench_boundary_regular_family,
    bench_capacitary_pot_family,
    bench_dirichlet_problem_family,
    bench_energy_principle_family,
    bench_equilibrium_measure_family,
    bench_thin_set_family,
)
from quant_fund.research.benches_w983 import (
    bench_atoms_decomp_family,
    bench_besov_embed_family,
    bench_besov_space_family,
    bench_hardy_littlewood_max_family,
    bench_triebel_lizorkin_family,
    bench_wavelet_char_family,
)
from quant_fund.research.benches_w984 import (
    bench_ambiguity_fn_family,
    bench_feichtinger_alg_family,
    bench_gabor_frame_family,
    bench_modulation_space_family,
    bench_short_time_ft_family,
    bench_wigner_dist_family,
)
from quant_fund.research.benches_w985 import (
    bench_atomic_h1_family,
    bench_bmo_space_family,
    bench_carleson_measure_family,
    bench_fefferman_stein_family,
    bench_hardy_h1_family,
    bench_john_nirenberg_family,
)
from quant_fund.research.benches_w986 import (
    bench_ap_weight_family,
    bench_calderon_zygmund_family,
    bench_cotlar_ineq_family,
    bench_cz_decomp_family,
    bench_good_lambda_family,
    bench_reverse_holder_family,
)
from quant_fund.research.benches_w987 import (
    bench_davies_gaffney_family,
    bench_gaussian_upper_family,
    bench_grad_est_family,
    bench_li_yau_family,
    bench_nash_ineq_family,
    bench_parabolic_harnack_family,
)
from quant_fund.research.benches_w988 import (
    bench_cheeger_ineq_family,
    bench_heat_invariants_family,
    bench_isospectral_family,
    bench_nodal_domain_family,
    bench_spectral_geometry_family,
    bench_weyl_law_family,
)
from quant_fund.research.benches_w989 import (
    bench_fbi_transform_family,
    bench_melrose_bdy_family,
    bench_parametrix_family,
    bench_propagation_thm_family,
    bench_sg_calculus_family,
    bench_wave_eq_group_family,
)
from quant_fund.research.benches_w990 import (
    bench_euler_lagrange_family,
    bench_geodesic_var_family,
    bench_isoperimetric_var_family,
    bench_jacobi_eq_family,
    bench_legendre_cond_family,
    bench_soap_film_family,
)
from quant_fund.research.benches_w991 import (
    bench_bloch_decomp_family,
    bench_gamma_convergence_family,
    bench_h_convergence_family,
    bench_homogenization_family,
    bench_mosco_conv_family,
    bench_two_scale_conv_family,
)
from quant_fund.research.benches_w992 import (
    bench_limiting_absorption_family,
    bench_radiation_cond_family,
    bench_resonances_thy_family,
    bench_scattering_matrix_family,
    bench_trace_class_scatt_family,
    bench_wave_operators_family,
)
from quant_fund.research.benches_w993 import (
    bench_degree_theory_family,
    bench_krein_rutman_family,
    bench_maximal_monotone_family,
    bench_minty_browder_family,
    bench_monotone_op_family,
    bench_schauder_fixed_family,
)
from quant_fund.research.benches_w994 import (
    bench_borg_levinson_family,
    bench_gelfand_levitan_family,
    bench_inverse_scattering_family,
    bench_kdv_isospectral_family,
    bench_marchenko_eq_family,
    bench_trace_formulas_family,
)
from quant_fund.research.benches_w995 import (
    bench_calogero_moser_family,
    bench_kp_hierarchy_family,
    bench_nls_soliton_family,
    bench_painleve_eq_family,
    bench_sine_gordon_family,
    bench_toda_lattice_family,
)
from quant_fund.research.benches_w996 import (
    bench_dbar_method_family,
    bench_deift_zhou_family,
    bench_fokas_unified_family,
    bench_isomonodromy_family,
    bench_orthogonal_poly_rh_family,
    bench_small_norm_rh_family,
)
from quant_fund.research.benches_w997 import (
    bench_bilinear_estimates_family,
    bench_i_method_family,
    bench_kdv_dispersion_family,
    bench_local_smoothing_family,
    bench_nls_dispersion_family,
    bench_strichartz_estimates_family,
)
from quant_fund.research.benches_w998 import (
    bench_fujita_exponent_family,
    bench_matched_asymptotic_pde_family,
    bench_regularity_critical_family,
    bench_self_similar_blowup_family,
    bench_semilinear_heat_family,
    bench_singularity_formation_family,
)
from quant_fund.research.benches_w999 import (
    bench_brakke_varifolds_family,
    bench_currents_theory_family,
    bench_flat_chains_family,
    bench_integral_currents_family,
    bench_rectifiable_measures_family,
    bench_varifold_theory_family,
)
from quant_fund.research.benches_w1000 import (
    bench_contact_mechanics_family,
    bench_fracture_mechanics_family,
    bench_homogenized_elasticity_family,
    bench_kirchhoff_plate_family,
    bench_mindlin_reissner_family,
    bench_navier_elasticity_family,
)
from quant_fund.research.benches_w1001 import (
    bench_beale_kato_majda_family,
    bench_euler_equations_family,
    bench_ladyzhenskaya_weak_family,
    bench_leray_theory_family,
    bench_navier_stokes_family,
    bench_vorticity_form_family,
)
from quant_fund.research.benches_w1002 import (
    bench_energy_spectrum_family,
    bench_intermittency_models_family,
    bench_kolmogorov_theory_family,
    bench_reynolds_decomp_family,
    bench_taylor_series_hyp_family,
    bench_wall_turbulence_family,
)
from quant_fund.research.benches_w1003 import (
    bench_alfven_waves_family,
    bench_elsaesser_vars_family,
    bench_frozen_flux_family,
    bench_magnetic_reconnection_family,
    bench_mhd_equations_family,
    bench_parker_solar_wind_family,
)
from quant_fund.research.benches_w1004 import (
    bench_bgk_model_family,
    bench_boltzmann_eq_family,
    bench_chapman_enskog_family,
    bench_h_theorem_family,
    bench_landau_damping_family,
    bench_vlasov_eq_family,
)
from quant_fund.research.benches_w1005 import (
    bench_einstein_equations_family,
    bench_friedmann_eq_family,
    bench_gr_birkhoff_family,
    bench_kerr_metric_family,
    bench_penrose_diagrams_family,
    bench_schwarzschild_metric_family,
)
from quant_fund.research.benches_w1006 import (
    bench_fock_space_family,
    bench_harmonic_oscillator_family,
    bench_hydrogen_atom_family,
    bench_schrodinger_eq_family,
    bench_spin_half_family,
    bench_wigner_wick_family,
)
from quant_fund.research.benches_w1007 import (
    bench_bose_einstein_family,
    bench_fermi_dirac_family,
    bench_free_energy_family,
    bench_gibbs_measure_family,
    bench_ising_model_family,
    bench_partition_function_family,
)
from quant_fund.research.benches_w1008 import (
    bench_carnot_cycle_family,
    bench_critical_phenomena_family,
    bench_entropy_production_family,
    bench_fluctuation_dissipation_family,
    bench_maxwell_relations_family,
    bench_phase_transitions_family,
)
from quant_fund.research.benches_w1009 import (
    bench_dipole_radiation_family,
    bench_fresnel_eq_family,
    bench_lorentz_lorenz_family,
    bench_maxwell_equations_family,
    bench_poynting_vector_family,
    bench_wave_guides_family,
)
from quant_fund.research.benches_w1010 import (
    bench_canonical_quantization_family,
    bench_dirac_equation_family,
    bench_feynman_rules_family,
    bench_klein_gordon_family,
    bench_path_integral_qm_family,
    bench_renormalization_group_family,
)
from quant_fund.research.benches_w1011 import (
    bench_band_structure_family,
    bench_bloch_theorem_family,
    bench_hubbard_model_family,
    bench_kondo_effect_family,
    bench_phonon_spectrum_family,
    bench_tight_binding_family,
)
from quant_fund.research.benches_w1012 import (
    bench_bcs_theory_family,
    bench_cabibbo_km_family,
    bench_nuclear_liquid_drop_family,
    bench_nuclear_shell_model_family,
    bench_parton_model_family,
    bench_quark_model_family,
)
from quant_fund.research.benches_w1013 import (
    bench_born_oppenheimer_family,
    bench_hartree_fock_family,
    bench_molecular_orbitals_family,
    bench_rotational_spectra_family,
    bench_vibrational_spectra_family,
    bench_zeeman_effect_family,
)
from quant_fund.research.benches_w1014 import (
    bench_cmb_anisotropy_family,
    bench_dark_matter_family,
    bench_hubble_law_family,
    bench_jeans_instability_family,
    bench_stellar_evolution_family,
    bench_stellar_structure_family,
)
from quant_fund.research.benches_w1015 import (
    bench_acoustic_wave_eq_family,
    bench_doppler_effect_family,
    bench_helmholtz_eq_family,
    bench_rayleigh_scattering_family,
    bench_room_acoustics_family,
    bench_sound_absorption_family,
)
from quant_fund.research.benches_w1016 import (
    bench_coherence_theory_family,
    bench_diffraction_grating_family,
    bench_fourier_optics_family,
    bench_holography_family,
    bench_interference_fringes_family,
    bench_polarization_states_family,
)
from quant_fund.research.benches_w1017 import (
    bench_navier_cauchy_family,
    bench_plasticity_family,
    bench_poroelasticity_family,
    bench_rheology_family,
    bench_stress_tensor_family,
    bench_viscoelasticity_family,
)
from quant_fund.research.benches_w1018 import (
    bench_four_vectors_family,
    bench_geodesic_motion_family,
    bench_gravitational_lensing_family,
    bench_gravitational_waves_family,
    bench_lorentz_transformation_family,
    bench_spacetime_interval_family,
)
from quant_fund.research.benches_w1019 import (
    bench_earthquake_magnitude_family,
    bench_geomagnetism_family,
    bench_gravity_anomaly_family,
    bench_heat_flow_geo_family,
    bench_plate_tectonics_family,
    bench_seismic_waves_family,
)
from quant_fund.research.benches_w1020 import (
    bench_food_web_family,
    bench_island_biogeography_family,
    bench_logistic_growth_family,
    bench_lotka_volterra_family,
    bench_neutral_theory_family,
    bench_predator_prey_family,
)
from quant_fund.research.benches_w1021 import (
    bench_branching_epidemic_family,
    bench_herd_immunity_family,
    bench_r0_estimation_family,
    bench_seir_epidemic_family,
    bench_sir_epidemic_family,
    bench_sis_epidemic_family,
)
from quant_fund.research.benches_w1022 import (
    bench_auction_theory2_family,
    bench_growth_theory_family,
    bench_mechanism_design_family,
    bench_overlapping_gens_family,
    bench_real_business_family,
    bench_search_matching_family,
)
from quant_fund.research.benches_w1023 import (
    bench_arbitrage_pricing_family,
    bench_black_scholes_family,
    bench_capm_model_family,
    bench_corporate_finance_family,
    bench_default_risk_family,
    bench_yield_curve_family,
)
from quant_fund.research.benches_w1024 import (
    bench_dna_sequencing_family,
    bench_gene_expression_family,
    bench_metabolomics_family,
    bench_phylogenetics_family,
    bench_protein_folding_family,
    bench_systems_biology_family,
)
from quant_fund.research.benches_w1025 import (
    bench_behavioral_econ_family,
    bench_cognitive_science_family,
    bench_game_theory2_family,
    bench_linguistics_family,
    bench_political_science_family,
    bench_sociology_net_family,
)
from quant_fund.research.benches_w1026 import (
    bench_atmospheric_chem_family,
    bench_carbon_cycle_family,
    bench_climate_model_family,
    bench_ecosystem_model_family,
    bench_hydrology_family,
    bench_ocean_circulation_family,
)
from quant_fund.research.benches_w1027 import (
    bench_ceramics_family,
    bench_crystal_structure_family,
    bench_metallurgy_family,
    bench_nanomaterials_family,
    bench_polymer_physics_family,
    bench_superconductivity_family,
)
from quant_fund.research.benches_w1028 import (
    bench_fluid_dynamics2_family,
    bench_heat_exchanger_family,
    bench_process_control_family,
    bench_reaction_kinetics_family,
    bench_separation_proc_family,
    bench_thermo_props_family,
)
from quant_fund.research.benches_w1029 import (
    bench_fatigue_life_family,
    bench_kinematics_family,
    bench_machine_design_family,
    bench_solid_mechanics_family,
    bench_tribology_family,
    bench_vibration_analysis_family,
)
from quant_fund.research.benches_w1030 import (
    bench_circuit_analysis_family,
    bench_control_systems_family,
    bench_electromagnetics_family,
    bench_power_systems_family,
    bench_semiconductor_family,
    bench_signal_processing2_family,
)
from quant_fund.research.benches_w1031 import (
    bench_construction_mgmt_family,
    bench_geotechnics_family,
    bench_structural_analysis_family,
    bench_surveying_family,
    bench_transportation_eng_family,
    bench_water_resources_family,
)
from quant_fund.research.benches_w1032 import (
    bench_aerodynamics_family,
    bench_airfoil_theory_family,
    bench_flight_dynamics_family,
    bench_orbital_mechanics2_family,
    bench_propulsion_family,
    bench_spacecraft_design_family,
)
from quant_fund.research.benches_w1033 import (
    bench_bioinstrumentation_family,
    bench_biomechanics_family,
    bench_biomedical_imaging2_family,
    bench_medical_devices_family,
    bench_physiological_modeling_family,
    bench_tissue_engineering_family,
)
from quant_fund.research.benches_w1034 import (
    bench_ergonomics_family,
    bench_facility_layout_family,
    bench_manufacturing_sys_family,
    bench_operations_research_family,
    bench_quality_control_family,
    bench_supply_chain_family,
)
from quant_fund.research.benches_w1035 import (
    bench_isotope_production_family,
    bench_nuclear_fuel_cycle_family,
    bench_nuclear_safety_family,
    bench_radiation_protection_family,
    bench_reactor_physics_family,
    bench_thermal_hydraulics_family,
)
from quant_fund.research.benches_w1036 import (
    bench_drilling_engineering_family,
    bench_enhanced_recovery_family,
    bench_formation_evaluation_family,
    bench_production_engineering_family,
    bench_reservoir_engineering_family,
    bench_well_testing_family,
)
from quant_fund.research.benches_w1037 import (
    bench_agronomy_family,
    bench_animal_science_family,
    bench_crop_science_family,
    bench_horticulture_family,
    bench_pest_management_family,
    bench_soil_science_family,
)
from quant_fund.research.benches_w1038 import (
    bench_cardiology_family,
    bench_human_physiology_family,
    bench_immunology_family,
    bench_neuroscience_med_family,
    bench_pathology_family,
    bench_pharmacokinetics_family,
)
from quant_fund.research.benches_w1039 import (
    bench_air_pollution_control_family,
    bench_environmental_remediation_family,
    bench_noise_control_family,
    bench_waste_management_family,
    bench_wastewater_engineering_family,
    bench_water_treatment_family,
)
from quant_fund.research.benches_w1040 import (
    bench_actuator_design_family,
    bench_motion_control_family,
    bench_path_planning_family,
    bench_robot_dynamics_family,
    bench_robot_kinematics_family,
    bench_sensor_fusion_family,
)
from quant_fund.research.benches_w1041 import (
    bench_coastal_engineering_family,
    bench_marine_propulsion_family,
    bench_naval_architecture_family,
    bench_ocean_waves_family,
    bench_offshore_engineering_family,
    bench_submarine_systems_family,
)
from quant_fund.research.benches_w1042 import (
    bench_food_chemistry_family,
    bench_food_microbiology_family,
    bench_food_processing_family,
    bench_food_safety_family,
    bench_nutrition_science_family,
    bench_sensory_evaluation_family,
)
from quant_fund.research.benches_w1043 import (
    bench_dendrology_family,
    bench_forest_ecology_family,
    bench_forest_economics_family,
    bench_silviculture_family,
    bench_timber_harvesting_family,
    bench_wildfire_management_family,
)
from quant_fund.research.benches_w1044 import (
    bench_blasting_engineering_family,
    bench_mine_design_family,
    bench_mine_ventilation_family,
    bench_mineral_processing_family,
    bench_ore_reserve_estimation_family,
    bench_rock_mechanics_family,
)
from quant_fund.research.benches_w1045 import (
    bench_geochemistry_family,
    bench_geochronology_family,
    bench_paleontology_family,
    bench_petrology_family,
    bench_stratigraphy_family,
    bench_structural_geology_family,
)
from quant_fund.research.benches_w1046 import (
    bench_atmospheric_dynamics_family,
    bench_climate_dynamics_family,
    bench_cloud_physics_family,
    bench_mesoscale_meteorology_family,
    bench_numerical_weather_family,
    bench_synoptic_meteorology_family,
)
from quant_fund.research.benches_w1047 import (
    bench_aquaculture_family,
    bench_benthic_biology_family,
    bench_coral_reef_ecology_family,
    bench_fisheries_science_family,
    bench_marine_ecology_family,
    bench_plankton_dynamics_family,
)
from quant_fund.research.benches_w1048 import (
    bench_animal_surgery_family,
    bench_equine_medicine_family,
    bench_veterinary_anatomy_family,
    bench_veterinary_epidemiology_family,
    bench_veterinary_pathology_family,
    bench_veterinary_pharmacology_family,
)
from quant_fund.research.benches_w1049 import (
    bench_dental_anatomy_family,
    bench_endodontics_family,
    bench_oral_pathology_family,
    bench_orthodontics_family,
    bench_periodontology_family,
    bench_prosthodontics_family,
)
from quant_fund.research.benches_w1050 import (
    bench_clinical_pharmacology_family,
    bench_drug_metabolism_family,
    bench_neuropharmacology_family,
    bench_pharmacodynamics_family,
    bench_pharmacokinetics_2_family,
    bench_toxicology_family,
)
from quant_fund.research.benches_w1051 import (
    bench_biostatistics_2_family,
    bench_epidemiology_2_family,
    bench_global_health_family,
    bench_health_policy_family,
    bench_occupational_health_family,
    bench_preventive_medicine_family,
)
from quant_fund.research.benches_w1052 import (
    bench_clinical_nutrition_family,
    bench_dietary_assessment_family,
    bench_metabolic_health_family,
    bench_nutritional_biochemistry_family,
    bench_nutritional_epidemiology_family,
    bench_sports_nutrition_family,
)
from quant_fund.research.benches_w1053 import (
    bench_behavioral_neuroscience_family,
    bench_clinical_psychology_family,
    bench_cognitive_psychology_family,
    bench_developmental_psychology_family,
    bench_psychometrics_family,
    bench_social_psychology_family,
)
from quant_fund.research.benches_w1054 import (
    bench_criminology_family,
    bench_demography_family,
    bench_economic_sociology_family,
    bench_social_networks_family,
    bench_social_stratification_family,
    bench_urban_sociology_family,
)
from quant_fund.research.benches_w1055 import (
    bench_archaeology_family,
    bench_cultural_anthropology_family,
    bench_ethnography_family,
    bench_linguistic_anthropology_family,
    bench_physical_anthropology_family,
    bench_primatology_family,
)
from quant_fund.research.benches_w1056 import (
    bench_comparative_politics_family,
    bench_electoral_systems_family,
    bench_international_relations_family,
    bench_political_economy_family,
    bench_political_theory_family,
    bench_public_administration_family,
)
from quant_fund.research.benches_w1057 import (
    bench_morphology_family,
    bench_phonetics_family,
    bench_phonology_family,
    bench_pragmatics_family,
    bench_semantics_family,
    bench_syntax_theory_family,
)
from quant_fund.research.benches_w1058 import (
    bench_aesthetics_family,
    bench_epistemology_family,
    bench_ethics_philosophy_family,
    bench_logic_philosophy_family,
    bench_metaphysics_family,
    bench_philosophy_of_science_family,
)
from quant_fund.research.benches_w1059 import (
    bench_ancient_history_family,
    bench_economic_history_family,
    bench_historiography_family,
    bench_intellectual_history_family,
    bench_medieval_history_family,
    bench_modern_history_family,
)
from quant_fund.research.benches_w1060 import (
    bench_assessment_theory_family,
    bench_curriculum_design_family,
    bench_educational_psychology_family,
    bench_educational_technology_family,
    bench_learning_sciences_family,
    bench_pedagogy_family,
)
from quant_fund.research.benches_w1061 import (
    bench_administrative_law_family,
    bench_constitutional_law_family,
    bench_contract_law_family,
    bench_criminal_law_family,
    bench_international_law_family,
    bench_tort_law_family,
)
from quant_fund.research.benches_w1062 import (
    bench_biblical_studies_family,
    bench_buddhist_studies_family,
    bench_comparative_religion_family,
    bench_islamic_studies_family,
    bench_religious_ethics_family,
    bench_theology_family,
)
from quant_fund.research.benches_w1063 import (
    bench_communication_theory_family,
    bench_digital_media_family,
    bench_journalism_family,
    bench_media_studies_family,
    bench_public_relations_family,
    bench_rhetoric_family,
)
from quant_fund.research.benches_w1064 import (
    bench_disability_studies_family,
    bench_ethnic_studies_family,
    bench_gender_studies_family,
    bench_public_policy_family,
    bench_social_work_family,
    bench_urban_studies_family,
)
from quant_fund.research.benches_w1065 import (
    bench_cartography_family,
    bench_climatology_family,
    bench_geomorphology_family,
    bench_human_geography_family,
    bench_physical_geography_family,
    bench_remote_sensing_family,
)
from quant_fund.research.benches_w1066 import (
    bench_african_studies_family,
    bench_asian_studies_family,
    bench_european_studies_family,
    bench_latin_american_studies_family,
    bench_middle_eastern_studies_family,
    bench_slavic_studies_family,
)
from quant_fund.research.benches_w1067 import (
    bench_archival_studies_family,
    bench_digital_humanities_family,
    bench_information_science_family,
    bench_knowledge_organization_family,
    bench_library_science_family,
    bench_museum_studies_family,
)
from quant_fund.research.benches_w1068 import (
    bench_conflict_resolution_family,
    bench_defense_studies_family,
    bench_intelligence_studies_family,
    bench_military_science_family,
    bench_peace_studies_family,
    bench_strategic_studies_family,
)
from quant_fund.research.benches_w1069 import (
    bench_criminal_justice_family,
    bench_criminal_procedure_family,
    bench_forensic_science_family,
    bench_penology_family,
    bench_policing_studies_family,
    bench_victimology_family,
)
from quant_fund.research.benches_w1070 import (
    bench_athletic_training_family,
    bench_exercise_physiology_family,
    bench_sports_analytics_family,
    bench_sports_biomechanics_family,
    bench_sports_psychology_family,
    bench_sports_science_family,
)
from quant_fund.research.benches_w1071 import (
    bench_ethnomusicology_family,
    bench_music_cognition_family,
    bench_music_history_family,
    bench_music_theory_family,
    bench_musicology_family,
    bench_organology_family,
)
from quant_fund.research.benches_w1072 import (
    bench_cinema_studies_family,
    bench_documentary_studies_family,
    bench_film_history_family,
    bench_film_studies_family,
    bench_film_theory_family,
    bench_screenwriting_family,
)
from quant_fund.research.benches_w1073 import (
    bench_biblical_exegesis_family,
    bench_church_history_family,
    bench_liturgical_studies_family,
    bench_missiology_family,
    bench_pastoral_theology_family,
    bench_systematic_theology_family,
)
from quant_fund.research.benches_w1074 import (
    bench_baking_science_family,
    bench_culinary_arts_family,
    bench_fermentation_science_family,
    bench_flavor_science_family,
    bench_food_studies_family,
    bench_gastronomy_family,
)
from quant_fund.research.benches_w1075 import (
    bench_architecture_theory_family,
    bench_building_science_family,
    bench_industrial_design_family,
    bench_interior_design_family,
    bench_landscape_architecture_family,
    bench_urban_design_family,
)
from quant_fund.research.benches_w1076 import (
    bench_archaeometry_family,
    bench_bioarchaeology_family,
    bench_experimental_archaeology_family,
    bench_field_archaeology_family,
    bench_landscape_archaeology_family,
    bench_underwater_archaeology_family,
)
from quant_fund.research.benches_w1077 import (
    bench_art_conservation_family,
    bench_art_history_family,
    bench_painting_techniques_family,
    bench_printmaking_family,
    bench_sculpture_methods_family,
    bench_visual_culture_family,
)
from quant_fund.research.benches_w1078 import (
    bench_choreography_family,
    bench_dance_studies_family,
    bench_dramaturgy_family,
    bench_performance_theory_family,
    bench_stage_design_family,
    bench_theater_studies_family,
)
from quant_fund.research.benches_w1079 import (
    bench_ancient_greek_family,
    bench_classical_archaeology_family,
    bench_classical_studies_family,
    bench_latin_language_family,
    bench_papyrology_family,
    bench_philology_family,
)
from quant_fund.research.benches_w1080 import (
    bench_byzantine_studies_family,
    bench_codicology_family,
    bench_hagiography_family,
    bench_medieval_studies_family,
    bench_numismatics_family,
    bench_paleography_family,
)
from quant_fund.research.benches_w1081 import (
    bench_baroque_studies_family,
    bench_early_modern_family,
    bench_enlightenment_studies_family,
    bench_humanism_family,
    bench_reformation_studies_family,
    bench_renaissance_studies_family,
)
from quant_fund.research.benches_w1082 import (
    bench_hebrew_language_family,
    bench_jewish_philosophy_family,
    bench_jewish_studies_family,
    bench_kabbalah_family,
    bench_rabbinics_family,
    bench_talmudic_studies_family,
)
from quant_fund.research.benches_w1083 import (
    bench_hermeneutics_family,
    bench_narratology_family,
    bench_phenomenology_family,
    bench_poststructuralism_family,
    bench_semiotics_family,
    bench_structuralism_family,
)
from quant_fund.research.benches_w1084 import (
    bench_comparative_literature_family,
    bench_critical_theory_family,
    bench_literary_theory_family,
    bench_postcolonial_studies_family,
    bench_translation_studies_family,
    bench_world_literature_family,
)
from quant_fund.research.benches_w1085 import (
    bench_medieval_literature_family,
    bench_modernism_family,
    bench_postmodernism_family,
    bench_renaissance_literature_family,
    bench_romanticism_family,
    bench_victorian_studies_family,
)
from quant_fund.research.benches_w1086 import (
    bench_assyriology_family,
    bench_egyptology_family,
    bench_indology_family,
    bench_iranian_studies_family,
    bench_ottoman_studies_family,
    bench_sinology_family,
)
from quant_fund.research.benches_w1087 import (
    bench_diplomatics_family,
    bench_epigraphy_family,
    bench_genealogy_studies_family,
    bench_heraldry_family,
    bench_onomastics_family,
    bench_sigillography_family,
)
from quant_fund.research.benches_w1088 import (
    bench_analytic_philosophy_family,
    bench_ancient_philosophy_family,
    bench_continental_philosophy_family,
    bench_existentialism_family,
    bench_medieval_philosophy_family,
    bench_pragmatism_family,
)
from quant_fund.research.benches_w1089 import (
    bench_computational_linguistics_family,
    bench_corpus_linguistics_family,
    bench_dialectology_family,
    bench_historical_linguistics_family,
    bench_psycholinguistics_family,
    bench_sociolinguistics_family,
)
from quant_fund.research.benches_w1090 import (
    bench_history_of_science_family,
    bench_information_history_family,
    bench_media_archaeology_family,
    bench_philosophy_of_technology_family,
    bench_sts_studies_family,
    bench_technology_studies_family,
)
from quant_fund.research.benches_w1091 import (
    bench_comparative_education_family,
    bench_distance_learning_family,
    bench_higher_education_family,
    bench_literacy_studies_family,
    bench_special_education_family,
    bench_vocational_education_family,
)
from quant_fund.research.benches_w1092 import (
    bench_connoisseurship_family,
    bench_curation_practice_family,
    bench_formal_analysis_family,
    bench_iconography_family,
    bench_iconology_family,
    bench_provenance_studies_family,
)
from quant_fund.research.benches_w1093 import (
    bench_canon_law_family,
    bench_civil_law_family,
    bench_common_law_family,
    bench_maritime_law_family,
    bench_procedural_law_family,
    bench_property_law_family,
)
from quant_fund.research.benches_w1094 import (
    bench_dermatology_family,
    bench_neurology_family,
    bench_oncology_family,
    bench_orthopedics_family,
    bench_psychiatry_family,
    bench_radiology_family,
)
from quant_fund.research.benches_w1095 import (
    bench_deviance_studies_family,
    bench_family_sociology_family,
    bench_medical_sociology_family,
    bench_organization_theory_family,
    bench_rural_sociology_family,
    bench_social_movements_family,
)
from quant_fund.research.benches_w1096 import (
    bench_conservation_biology_family,
    bench_environmental_toxicology_family,
    bench_landscape_ecology_family,
    bench_marine_conservation_family,
    bench_pollution_science_family,
    bench_urban_ecology_family,
)
from quant_fund.research.benches_w1097 import (
    bench_eastern_philosophy_family,
    bench_moral_philosophy_family,
    bench_philosophy_of_language_family,
    bench_philosophy_of_law_family,
    bench_philosophy_of_mind_family,
    bench_political_philosophy_family,
)
from quant_fund.research.benches_w1098 import (
    bench_biological_anthropology_family,
    bench_economic_anthropology_family,
    bench_medical_anthropology_family,
    bench_paleoanthropology_family,
    bench_political_anthropology_family,
    bench_urban_anthropology_family,
)
from quant_fund.research.benches_w1099 import (
    bench_abnormal_psychology_family,
    bench_forensic_psychology_family,
    bench_health_psychology_family,
    bench_neuropsychology_family,
    bench_organizational_psychology_family,
    bench_personality_psychology_family,
)
from quant_fund.research.benches_w1100 import (
    bench_analytical_chemistry_family,
    bench_biochemistry_family,
    bench_electrochemistry_family,
    bench_inorganic_chemistry_family,
    bench_organic_chemistry_family,
    bench_physical_chemistry_family,
)
from quant_fund.research.benches_w1101 import (
    bench_botany_family,
    bench_cell_biology_family,
    bench_genetics_family,
    bench_microbiology_family,
    bench_molecular_biology_family,
    bench_zoology_family,
)
from quant_fund.research.benches_w1102 import (
    bench_geophysics_applied_family,
    bench_hydrogeology_family,
    bench_mineralogy_family,
    bench_sedimentology_family,
    bench_tectonics_family,
    bench_volcanology_family,
)
from quant_fund.research.benches_w1103 import (
    bench_boundary_layer_meteorology_family,
    bench_micrometeorology_family,
    bench_polar_meteorology_family,
    bench_radar_meteorology_family,
    bench_severe_weather_family,
    bench_tropical_meteorology_family,
)
from quant_fund.research.benches_w1104 import (
    bench_american_politics_family,
    bench_policy_analysis_family,
    bench_political_behavior_family,
    bench_political_methodology_family,
    bench_public_law_family,
    bench_security_studies_family,
)
from quant_fund.research.benches_w1105 import (
    bench_bounded_rationality_family,
    bench_experimental_economics_family,
    bench_financial_behavior_family,
    bench_neuroeconomics_family,
    bench_nudge_theory_family,
    bench_prospect_theory_family,
)
from quant_fund.research.benches_w1106 import (
    bench_biomaterials_family,
    bench_characterization_methods_family,
    bench_composite_materials_family,
    bench_phase_diagrams_family,
    bench_semiconductors_materials_family,
    bench_thin_films_family,
)
from quant_fund.research.benches_w1107 import (
    bench_endocrinology_family,
    bench_gastroenterology_family,
    bench_hematology_family,
    bench_infectious_diseases_family,
    bench_nephrology_family,
    bench_pulmonology_family,
)
from quant_fund.research.benches_w1108 import (
    bench_cultural_sociology_family,
    bench_environmental_sociology_family,
    bench_industrial_sociology_family,
    bench_political_sociology_family,
    bench_sociology_of_education_family,
    bench_sociology_of_religion_family,
)
from quant_fund.research.benches_w1109 import (
    bench_anthropological_linguistics_family,
    bench_applied_linguistics_family,
    bench_discourse_analysis_family,
    bench_evolutionary_linguistics_family,
    bench_forensic_linguistics_family,
    bench_neurolinguistics_family,
)
from quant_fund.research.benches_w1110 import (
    bench_phenomenology_2_family,
    bench_philosophy_of_biology_family,
    bench_philosophy_of_history_family,
    bench_philosophy_of_mathematics_family,
    bench_philosophy_of_religion_family,
    bench_process_philosophy_family,
)
from quant_fund.research.benches_w1111 import (
    bench_medicinal_chemistry_family,
    bench_photochemistry_family,
    bench_quantum_chemistry_family,
    bench_spectroscopy_family,
    bench_stereochemistry_family,
    bench_supramolecular_chemistry_family,
)
from quant_fund.research.benches_w1112 import (
    bench_biophysics_family,
    bench_comparative_anatomy_family,
    bench_developmental_biology_family,
    bench_ethology_family,
    bench_evolutionary_biology_family,
    bench_neurobiology_family,
)
from quant_fund.research.benches_w1113 import (
    bench_anesthesiology_family,
    bench_emergency_medicine_family,
    bench_family_medicine_family,
    bench_obstetrics_gynecology_family,
    bench_pediatrics_family,
    bench_surgery_family,
)
from quant_fund.research.benches_w1114 import (
    bench_classical_mechanics_family,
    bench_condensed_matter_2_family,
    bench_nuclear_physics_family,
    bench_plasma_physics_family,
    bench_quantum_mechanics_2_family,
    bench_statistical_mechanics_2_family,
)
from quant_fund.research.benches_w1115 import (
    bench_financial_economics_family,
    bench_industrial_organization_family,
    bench_international_economics_family,
    bench_labor_economics_family,
    bench_monetary_economics_family,
    bench_public_economics_family,
)
from quant_fund.research.benches_w1116 import (
    bench_comparative_psychology_family,
    bench_environmental_psychology_family,
    bench_evolutionary_psychology_family,
    bench_experimental_psychology_family,
    bench_psychopathology_family,
    bench_sport_psychology_family,
)
from quant_fund.research.benches_w1117 import (
    bench_historical_sociology_family,
    bench_legal_sociology_family,
    bench_mathematical_sociology_family,
    bench_military_sociology_family,
    bench_science_studies_family,
    bench_sociology_of_knowledge_family,
)
from quant_fund.research.benches_w1118 import (
    bench_field_linguistics_family,
    bench_language_acquisition_family,
    bench_linguistic_typology_family,
    bench_sign_linguistics_family,
    bench_theoretical_linguistics_family,
    bench_translation_theory_family,
)
from quant_fund.research.benches_w1119 import (
    bench_cultural_history_family,
    bench_diplomatic_history_family,
    bench_history_of_medicine_family,
    bench_history_of_technology_family,
    bench_military_history_family,
    bench_social_history_family,
)
from quant_fund.research.benches_w1120 import (
    bench_applied_anthropology_family,
    bench_digital_anthropology_family,
    bench_environmental_anthropology_family,
    bench_forensic_anthropology_family,
    bench_psychological_anthropology_family,
    bench_visual_anthropology_family,
)
from quant_fund.research.benches_w1121 import (
    bench_economic_geography_family,
    bench_gis_science_family,
    bench_health_geography_family,
    bench_political_geography_family,
    bench_population_geography_family,
    bench_regional_geography_family,
)
from quant_fund.research.benches_w1122 import (
    bench_archaeogenetics_family,
    bench_ceramic_analysis_family,
    bench_geoarchaeology_family,
    bench_lithic_analysis_family,
    bench_paleoethnobotany_family,
    bench_zooarchaeology_family,
)
from quant_fund.research.benches_w1123 import (
    bench_contact_linguistics_family,
    bench_descriptive_linguistics_family,
    bench_dialectometry_family,
    bench_etymology_family,
    bench_lexicography_family,
    bench_philological_studies_family,
)
from quant_fund.research.benches_w1124 import (
    bench_sociology_of_aging_family,
    bench_sociology_of_emotions_family,
    bench_sociology_of_food_family,
    bench_sociology_of_media_family,
    bench_sociology_of_sport_family,
    bench_sociology_of_work_family,
)
from quant_fund.research.benches_w1125 import (
    bench_agricultural_economics_family,
    bench_development_economics_family,
    bench_energy_economics_family,
    bench_environmental_economics_family,
    bench_health_economics_family,
    bench_urban_economics_family,
)
from quant_fund.research.benches_w1126 import (
    bench_community_psychology_family,
    bench_consumer_psychology_family,
    bench_cross_cultural_psychology_family,
    bench_political_psychology_family,
    bench_positive_psychology_family,
    bench_social_cognition_family,
)
from quant_fund.research.benches_w1127 import (
    bench_digital_history_family,
    bench_environmental_history_family,
    bench_global_history_family,
    bench_maritime_history_family,
    bench_oral_history_family,
    bench_public_history_family,
)
from quant_fund.research.benches_w1128 import (
    bench_computational_stylistics_family,
    bench_corpus_phonology_family,
    bench_language_documentation_family,
    bench_lexical_semantics_family,
    bench_stylistics_family,
    bench_translation_technology_family,
)
from quant_fund.research.benches_w1129 import (
    bench_african_philosophy_family,
    bench_bioethics_family,
    bench_environmental_philosophy_family,
    bench_feminist_philosophy_family,
    bench_philosophy_of_education_family,
    bench_philosophy_of_medicine_family,
)
from quant_fund.research.benches_w1130 import (
    bench_anthropology_of_religion_family,
    bench_cognitive_anthropology_family,
    bench_kinship_studies_family,
    bench_material_culture_family,
    bench_museum_anthropology_family,
    bench_social_anthropology_family,
)
from quant_fund.research.benches_w1131 import (
    bench_digital_sociology_family,
    bench_sociology_of_disaster_family,
    bench_sociology_of_housing_family,
    bench_sociology_of_migration_family,
    bench_sociology_of_risk_family,
    bench_sociology_of_the_body_family,
)
from quant_fund.research.benches_w1132 import (
    bench_history_of_capitalism_family,
    bench_history_of_emotions_family,
    bench_history_of_religions_family,
    bench_history_of_sexuality_family,
    bench_history_of_the_book_family,
    bench_microhistory_family,
)
from quant_fund.research.benches_w1133 import (
    bench_conformal_field_theory_family,
    bench_holography_ads_family,
    bench_lattice_field_theory_family,
    bench_loop_quantum_gravity_family,
    bench_statistical_field_theory_family,
    bench_string_theory_math_family,
)
from quant_fund.research.benches_w1134 import (
    bench_adaptive_method_theory_family,
    bench_finite_element_theory_family,
    bench_high_performance_numerics_family,
    bench_reduced_order_modeling_family,
    bench_spectral_theory_numerics_family,
    bench_uncertainty_quantification_2_family,
)
from quant_fund.research.benches_w1135 import (
    bench_ophthalmology_family,
    bench_otolaryngology_family,
    bench_palliative_medicine_family,
    bench_rehabilitation_medicine_family,
    bench_sports_medicine_family,
    bench_urology_family,
)
from quant_fund.research.benches_w1136 import (
    bench_environmental_law_family,
    bench_evidence_law_family,
    bench_family_law_family,
    bench_immigration_law_family,
    bench_labor_law_family,
    bench_tax_law_family,
)
from quant_fund.research.benches_w1137 import (
    bench_adult_education_family,
    bench_bilingual_education_family,
    bench_early_childhood_education_family,
    bench_educational_leadership_family,
    bench_gifted_education_family,
    bench_instructional_design_family,
)
from quant_fund.research.benches_w1138 import (
    bench_behavioral_economics_family,
    bench_econ_neuroscience_family,
    bench_evolutionary_economics_family,
    bench_experimental_economics_2_family,
    bench_institutional_economics_family,
    bench_political_economy_2_family,
)
from quant_fund.research.benches_w1139 import (
    bench_dentistry_2_family,
    bench_dietetics_family,
    bench_occupational_therapy_family,
    bench_optometry_family,
    bench_physiotherapy_family,
    bench_podiatry_family,
)
from quant_fund.research.benches_w1140 import (
    bench_glaciology_family,
    bench_hydrology_2_family,
    bench_oceanography_family,
    bench_paleoclimatology_family,
    bench_seismology_family,
    bench_volcanology_2_family,
)
from quant_fund.research.benches_w1141 import (
    bench_astrobiology_family,
    bench_astrochemistry_family,
    bench_cosmology_2_family,
    bench_exoplanet_science_family,
    bench_galactic_dynamics_family,
    bench_helio_seismology_family,
)
from quant_fund.research.benches_w1142 import (
    bench_quantum_chemistry_2_family,
    bench_quantum_computing_family,
    bench_quantum_error_2_family,
    bench_quantum_information_2_family,
    bench_quantum_optics_family,
    bench_quantum_sensing_family,
)
from quant_fund.research.benches_w1143 import (
    bench_acoustics_2_family,
    bench_biophysics_2_family,
    bench_condensed_matter_3_family,
    bench_nanotechnology_family,
    bench_optics_3_family,
    bench_thermodynamics_2_family,
)
from quant_fund.research.benches_w1144 import (
    bench_asteroid_science_family,
    bench_astrophotonics_family,
    bench_comet_science_family,
    bench_grav_waves_2_family,
    bench_planetology_family,
    bench_space_weather_family,
)
from quant_fund.research.benches_w1145 import (
    bench_bioinformatics_4_family,
    bench_epidemiology_3_family,
    bench_genomicsciences_family,
    bench_proteomics_family,
    bench_synthetic_biology_family,
    bench_systems_biology_2_family,
)
from quant_fund.research.benches_w1146 import (
    bench_entomology_2_family,
    bench_limnology_family,
    bench_mycology_family,
    bench_parasitology_family,
    bench_virology_family,
    bench_wildlife_biology_family,
)
from quant_fund.research.benches_w1147 import (
    bench_anatomy_family,
    bench_cardiology_2_family,
    bench_endocrinology_2_family,
    bench_immunology_2_family,
    bench_neuroscience_2_family,
    bench_physiology_2_family,
)
from quant_fund.research.benches_w1148 import (
    bench_geochronology_2_family,
    bench_geology_3_family,
    bench_geomorphology_2_family,
    bench_mineralogy_2_family,
    bench_petrology_2_family,
    bench_stratigraphy_2_family,
)
from quant_fund.research.benches_w1149 import (
    bench_dermatology_2_family,
    bench_hematology_2_family,
    bench_hepatology_2_family,
    bench_nephrology_2_family,
    bench_pulmonology_2_family,
    bench_toxicology_2_family,
)
from quant_fund.research.benches_w1150 import (
    bench_bacteriology_family,
    bench_epigenetics_family,
    bench_immunogenetics_family,
    bench_microbiology_2_family,
    bench_molecular_genetics_family,
    bench_virology_2_family,
)
from quant_fund.research.benches_w1151 import (
    bench_analytical_chemistry_2_family,
    bench_chemistry_3_family,
    bench_electrochemistry_2_family,
    bench_inorganic_chemistry_2_family,
    bench_organic_chemistry_2_family,
    bench_physical_chemistry_2_family,
)
from quant_fund.research.benches_w1152 import (
    bench_biochemistry_2_family,
    bench_cell_biology_2_family,
    bench_genetics_2_family,
    bench_molecular_biology_2_family,
    bench_pharmacology_2_family,
    bench_toxicology_3_family,
)
from quant_fund.research.benches_w1153 import (
    bench_astrophysics_3_family,
    bench_cosmology_3_family,
    bench_geophysics_3_family,
    bench_mechanics_family,
    bench_physics_6_family,
    bench_thermodynamics_3_family,
)
from quant_fund.research.benches_w1154 import (
    bench_electromagnetism_family,
    bench_nuclear_physics_2_family,
    bench_optics_4_family,
    bench_particle_physics_family,
    bench_quantum_physics_family,
    bench_relativity_3_family,
)
from quant_fund.research.benches_w1155 import (
    bench_atmospheric_science_family,
    bench_earth_system_science_family,
    bench_environmental_science_2_family,
    bench_hydrology_3_family,
    bench_oceanography_2_family,
    bench_soil_science_2_family,
)
from quant_fund.research.benches_w1156 import (
    bench_applied_mathematics_family,
    bench_bioinformatics_5_family,
    bench_computational_science_family,
    bench_data_science_family,
    bench_probability_4_family,
    bench_statistics_2_family,
)
from quant_fund.research.benches_w1157 import (
    bench_artificial_intelligence_family,
    bench_computer_science_2_family,
    bench_data_engineering_family,
    bench_information_theory_2_family,
    bench_machine_learning_2_family,
    bench_software_engineering_family,
)
from quant_fund.research.benches_w1158 import (
    bench_anthropology_6_family,
    bench_economics_6_family,
    bench_linguistics_7_family,
    bench_political_science_3_family,
    bench_psychology_5_family,
    bench_sociology_6_family,
)
from quant_fund.research.benches_w1159 import (
    bench_aerospace_engineering_2_family,
    bench_biomedical_engineering_2_family,
    bench_chemical_engineering_2_family,
    bench_civil_engineering_2_family,
    bench_electrical_engineering_2_family,
    bench_mechanical_engineering_2_family,
)
from quant_fund.research.benches_w1160 import (
    bench_area_studies_2_family,
    bench_classics_2_family,
    bench_history_5_family,
    bench_humanities_2_family,
    bench_philosophy_6_family,
    bench_religious_studies_2_family,
)
from quant_fund.research.benches_w1161 import (
    bench_dentistry_3_family,
    bench_medicine_7_family,
    bench_nursing_2_family,
    bench_pharmacy_2_family,
    bench_public_health_2_family,
    bench_veterinary_medicine_2_family,
)
from quant_fund.research.benches_w1162 import (
    bench_criminology_2_family,
    bench_international_relations_2_family,
    bench_law_5_family,
    bench_military_science_2_family,
    bench_political_science_4_family,
    bench_public_administration_2_family,
)
from quant_fund.research.benches_w1163 import (
    bench_accounting_2_family,
    bench_business_administration_family,
    bench_entrepreneurship_2_family,
    bench_finance_4_family,
    bench_management_2_family,
    bench_marketing_2_family,
)
from quant_fund.research.benches_w1164 import (
    bench_communication_studies_2_family,
    bench_education_5_family,
    bench_information_science_2_family,
    bench_journalism_2_family,
    bench_library_science_2_family,
    bench_media_studies_2_family,
)
from quant_fund.research.benches_w1165 import (
    bench_agriculture_2_family,
    bench_fisheries_2_family,
    bench_food_science_2_family,
    bench_forestry_2_family,
    bench_horticulture_2_family,
    bench_veterinary_science_2_family,
)
from quant_fund.research.benches_w1166 import (
    bench_architecture_2_family,
    bench_graphic_design_2_family,
    bench_industrial_design_2_family,
    bench_interior_design_2_family,
    bench_landscape_architecture_2_family,
    bench_urban_planning_2_family,
)
from quant_fund.research.benches_w1167 import (
    bench_art_history_2_family,
    bench_dance_2_family,
    bench_film_studies_3_family,
    bench_music_2_family,
    bench_performance_studies_2_family,
    bench_theater_2_family,
)
from quant_fund.research.benches_w1168 import (
    bench_aviation_2_family,
    bench_logistics_2_family,
    bench_maritime_studies_2_family,
    bench_supply_chain_2_family,
    bench_transportation_2_family,
    bench_warehousing_2_family,
)
from quant_fund.research.benches_w1169 import (
    bench_disability_studies_2_family,
    bench_ethnic_studies_2_family,
    bench_gender_studies_2_family,
    bench_public_policy_2_family,
    bench_social_work_2_family,
    bench_urban_studies_2_family,
)
from quant_fund.research.benches_w1170 import (
    bench_criminology_3_family,
    bench_forensic_science_2_family,
    bench_intelligence_studies_2_family,
    bench_penology_2_family,
    bench_security_studies_2_family,
    bench_victimology_2_family,
)
from quant_fund.research.benches_w1171 import (
    bench_demography_2_family,
    bench_geography_2_family,
    bench_gis_science_2_family,
    bench_land_use_family,
    bench_regional_science_family,
    bench_urbanization_family,
)
from quant_fund.research.benches_w1172 import (
    bench_accounting_3_family,
    bench_entrepreneurship_3_family,
    bench_finance_5_family,
    bench_management_3_family,
    bench_marketing_3_family,
    bench_organizational_behavior_family,
)
from quant_fund.research.benches_w1173 import (
    bench_communication_3_family,
    bench_digital_media_2_family,
    bench_information_science_3_family,
    bench_journalism_3_family,
    bench_media_studies_3_family,
    bench_rhetoric_2_family,
)
from quant_fund.research.benches_w1174 import (
    bench_hospitality_family,
    bench_leisure_studies_family,
    bench_recreation_family,
    bench_recreation_therapy_family,
    bench_sports_management_family,
    bench_tourism_family,
)
from quant_fund.research.benches_w1175 import (
    bench_automotive_technology_family,
    bench_carpentry_trades_family,
    bench_electrical_trades_family,
    bench_plumbing_hvac_family,
    bench_refrigeration_technology_family,
    bench_welding_technology_family,
)
from quant_fund.research.benches_w1176 import (
    bench_biblical_studies_2_family,
    bench_buddhist_studies_2_family,
    bench_comparative_religion_2_family,
    bench_islamic_studies_2_family,
    bench_religious_studies_3_family,
    bench_theology_3_family,
)
from quant_fund.research.benches_w1177 import (
    bench_axiomatic_systems_family,
    bench_formal_ontology_family,
    bench_formal_sciences_family,
    bench_mathematical_logic_family,
    bench_model_checking_2_family,
    bench_proof_calculus_family,
)
from quant_fund.research.benches_w1178 import (
    bench_cognitive_science_2_family,
    bench_complexity_science_family,
    bench_futures_studies_family,
    bench_human_computer_interaction_family,
    bench_interdisciplinary_studies_family,
    bench_systems_science_family,
)
from quant_fund.research.benches_w1179 import (
    bench_conflict_studies_family,
    bench_intelligence_analysis_family,
    bench_military_history_2_family,
    bench_peace_research_family,
    bench_strategic_analysis_family,
    bench_war_studies_family,
)
from quant_fund.research.benches_w1180 import (
    bench_allied_health_family,
    bench_midwifery_family,
    bench_nursing_studies_family,
    bench_occupational_science_family,
    bench_paramedicine_family,
    bench_speech_pathology_family,
)
from quant_fund.research.benches_w1181 import (
    bench_acting_studies_family,
    bench_directing_studies_family,
    bench_performing_arts_2_family,
    bench_playwriting_family,
    bench_scenography_family,
    bench_theater_arts_family,
)
from quant_fund.research.benches_w1182 import (
    bench_ballet_studies_family,
    bench_choreography_2_family,
    bench_dance_pedagogy_family,
    bench_dance_science_family,
    bench_movement_studies_family,
    bench_somatic_practices_family,
)
from quant_fund.research.benches_w1183 import (
    bench_composition_studies_family,
    bench_ethnomusicology_2_family,
    bench_music_cognition_2_family,
    bench_music_theory_2_family,
    bench_musicology_2_family,
    bench_organology_2_family,
)
from quant_fund.research.benches_w1184 import (
    bench_animation_studies_family,
    bench_cinematography_studies_family,
    bench_documentary_production_family,
    bench_film_editing_family,
    bench_film_production_family,
    bench_sound_design_family,
)
from quant_fund.research.benches_w1185 import (
    bench_esports_studies_family,
    bench_game_design_family,
    bench_game_development_family,
    bench_game_studies_family,
    bench_interactive_media_family,
    bench_ludology_family,
)
from quant_fund.research.benches_w1186 import (
    bench_accessibility_studies_family,
    bench_hci_studies_family,
    bench_information_architecture_family,
    bench_interaction_design_family,
    bench_service_design_family,
    bench_ux_design_family,
)
from quant_fund.research.benches_w1187 import (
    bench_apparel_studies_family,
    bench_costume_design_family,
    bench_fashion_studies_family,
    bench_footwear_design_family,
    bench_jewelry_design_family,
    bench_textile_studies_family,
)
from quant_fund.research.benches_w1188 import (
    bench_brewing_science_family,
    bench_culinary_science_family,
    bench_enology_family,
    bench_fermentation_studies_family,
    bench_gastronomy_2_family,
    bench_pastry_arts_family,
)
from quant_fund.research.benches_w1189 import (
    bench_event_management_family,
    bench_hospitality_studies_family,
    bench_hotel_management_family,
    bench_leisure_science_family,
    bench_recreation_management_family,
    bench_tourism_studies_family,
)
from quant_fund.research.benches_w1190 import (
    bench_graphic_design_family,
    bench_motion_graphics_family,
    bench_photography_studies_family,
    bench_print_media_family,
    bench_typography_studies_family,
    bench_web_design_family,
)
from quant_fund.research.benches_w1191 import (
    bench_advertising_studies_family,
    bench_broadcasting_studies_family,
    bench_journalism_studies_family,
    bench_news_media_family,
    bench_public_relations_studies_family,
    bench_publishing_studies_family,
)
from quant_fund.research.benches_w1192 import (
    bench_acupuncture_studies_family,
    bench_chiropractic_studies_family,
    bench_herbal_medicine_family,
    bench_homeopathy_family,
    bench_naturopathy_family,
    bench_osteopathy_studies_family,
)
from quant_fund.research.benches_w1193 import (
    bench_audiology_studies_family,
    bench_clinical_psychology_2_family,
    bench_midwifery_studies_family,
    bench_opticianry_family,
    bench_orthoptics_family,
    bench_prosthetics_orthotics_family,
)
from quant_fund.research.benches_w1194 import (
    bench_genetic_counseling_family,
    bench_lactation_consulting_family,
    bench_perfusion_technology_family,
    bench_podiatric_medicine_family,
    bench_radiation_therapy_family,
    bench_respiratory_therapy_family,
)
from quant_fund.research.benches_w1195 import (
    bench_clinical_laboratory_family,
    bench_medical_imaging_studies_family,
    bench_mortuary_science_family,
    bench_phlebotomy_studies_family,
    bench_sterile_processing_family,
    bench_surgical_technology_family,
)
from quant_fund.research.benches_w1196 import (
    bench_disaster_management_family,
    bench_emergency_medical_technician_family,
    bench_fire_science_studies_family,
    bench_industrial_hygiene_family,
    bench_occupational_safety_family,
    bench_paramedic_studies_family,
)
from quant_fund.research.benches_w1197 import (
    bench_addiction_counseling_family,
    bench_genetic_screening_family,
    bench_neonatology_studies_family,
    bench_pediatric_therapeutics_family,
    bench_prenatal_studies_family,
    bench_rehabilitation_counseling_family,
)
from quant_fund.research.benches_w1198 import (
    bench_electrodiagnostic_studies_family,
    bench_hyperbaric_medicine_family,
    bench_infusion_therapy_family,
    bench_pain_management_family,
    bench_sleep_medicine_family,
    bench_wound_care_family,
)
from quant_fund.research.benches_w1199 import (
    bench_cardiac_electrophysiology_family,
    bench_dialysis_technology_family,
    bench_hepatobiliary_studies_family,
    bench_interventional_radiology_family,
    bench_nuclear_cardiology_family,
    bench_transplant_studies_family,
)
from quant_fund.research.benches_w1200 import (
    bench_cardiovascular_technology_family,
    bench_dosimetry_studies_family,
    bench_medical_physics_studies_family,
    bench_nuclear_medicine_technology_family,
    bench_radiation_dosimetry_family,
    bench_radiopharmacy_family,
)
from quant_fund.research.benches_w1201 import (
    bench_biomedical_informatics_family,
    bench_clinical_informatics_family,
    bench_health_data_science_family,
    bench_health_informatics_family,
    bench_health_information_family,
    bench_medical_records_family,
)
from quant_fund.research.benches_w1202 import (
    bench_entomology_medical_family,
    bench_medical_microbiology_family,
    bench_mycology_studies_family,
    bench_parasitology_studies_family,
    bench_public_health_microbiology_family,
    bench_vector_borne_diseases_family,
)
from quant_fund.research.benches_w1203 import (
    bench_genomic_medicine_family,
    bench_laboratory_medicine_family,
    bench_molecular_diagnostics_family,
    bench_precision_medicine_family,
    bench_travel_medicine_family,
    bench_tropical_medicine_family,
)
from quant_fund.research.benches_w1204 import (
    bench_aging_research_family,
    bench_geriatric_medicine_family,
    bench_gerontology_studies_family,
    bench_hospice_care_family,
    bench_longevity_medicine_family,
    bench_palliative_care_family,
)
from quant_fund.research.benches_w1205 import (
    bench_aerospace_medicine_family,
    bench_diving_medicine_family,
    bench_high_altitude_medicine_family,
    bench_hyperbaric_oxygen_family,
    bench_space_physiology_family,
    bench_wilderness_medicine_family,
)
from quant_fund.research.benches_w1206 import (
    bench_immunosuppression_family,
    bench_organ_donation_family,
    bench_regenerative_medicine_family,
    bench_stem_cell_therapy_family,
    bench_transplantation_medicine_family,
    bench_xenotransplantation_family,
)
from quant_fund.research.benches_w1207 import (
    bench_art_therapy_family,
    bench_behavioral_therapy_cognitive_family,
    bench_music_therapy_family,
    bench_play_therapy_family,
    bench_psychoanalysis_studies_family,
    bench_psychotherapy_studies_family,
)
from quant_fund.research.benches_w1208 import (
    bench_child_adolescent_therapy_family,
    bench_couples_therapy_family,
    bench_family_therapy_family,
    bench_group_therapy_family,
    bench_marriage_family_therapy_family,
    bench_trauma_therapy_family,
)
from quant_fund.research.benches_w1209 import (
    bench_addiction_medicine_family,
    bench_community_psychiatry_family,
    bench_consultation_liaison_family,
    bench_eating_disorders_family,
    bench_psychosomatic_medicine_family,
    bench_sleep_disorders_family,
)
from quant_fund.research.benches_w1210 import (
    bench_anxiety_disorders_family,
    bench_forensic_psychiatry_family,
    bench_geriatric_psychiatry_family,
    bench_mood_disorders_family,
    bench_personality_disorders_family,
    bench_psychotic_disorders_family,
)
from quant_fund.research.benches_w1211 import (
    bench_epilepsy_studies_family,
    bench_headache_medicine_family,
    bench_movement_disorders_family,
    bench_neurodevelopmental_disorders_family,
    bench_neuropsychiatry_studies_family,
    bench_pediatric_neurology_family,
)
from quant_fund.research.benches_w1212 import (
    bench_neuro_ophthalmology_family,
    bench_neurocritical_care_family,
    bench_neurogenetics_family,
    bench_neuroimmunology_family,
    bench_neuromuscular_medicine_family,
    bench_neurovascular_studies_family,
)
from quant_fund.research.benches_w1213 import (
    bench_neurorehabilitation_family,
    bench_neurosurgery_studies_family,
    bench_neurotoxicology_family,
    bench_neurotrauma_family,
    bench_neurovascular_surgery_family,
    bench_spinal_cord_medicine_family,
)
from quant_fund.research.benches_w1214 import (
    bench_cardiology_studies_family,
    bench_cardiovascular_imaging_family,
    bench_electrophysiology_studies_family,
    bench_heart_failure_medicine_family,
    bench_interventional_cardiology_family,
    bench_preventive_cardiology_family,
)
from quant_fund.research.benches_w1215 import (
    bench_adult_congenital_family,
    bench_cardiac_surgery_family,
    bench_structural_heart_family,
    bench_thoracic_surgery_family,
    bench_transplant_cardiology_family,
    bench_vascular_surgery_family,
)
from quant_fund.research.benches_w1216 import (
    bench_critical_care_medicine_family,
    bench_gastroenterology_studies_family,
    bench_hepatology_studies_family,
    bench_hospital_medicine_family,
    bench_internal_medicine_family,
    bench_pulmonary_medicine_family,
)
from quant_fund.research.benches_w1217 import (
    bench_adrenal_medicine_family,
    bench_bone_metabolism_family,
    bench_diabetes_medicine_family,
    bench_endocrinology_studies_family,
    bench_metabolic_medicine_family,
    bench_thyroid_medicine_family,
)
from quant_fund.research.benches_w1218 import (
    bench_acid_base_medicine_family,
    bench_dialysis_medicine_family,
    bench_hypertension_medicine_family,
    bench_nephrology_studies_family,
    bench_renal_transplant_family,
    bench_urology_studies_family,
)
from quant_fund.research.benches_w1219 import (
    bench_hematologic_malignancies_family,
    bench_hematology_studies_family,
    bench_oncology_studies_family,
    bench_radiation_oncology_family,
    bench_solid_tumor_oncology_family,
    bench_transfusion_medicine_family,
)
from quant_fund.research.benches_w1220 import (
    bench_allergy_immunology_family,
    bench_antimicrobial_stewardship_family,
    bench_hiv_medicine_family,
    bench_immunology_studies_family,
    bench_infectious_disease_medicine_family,
    bench_rheumatology_studies_family,
)
from quant_fund.research.benches_w1221 import (
    bench_audiology_medicine_family,
    bench_dermatology_studies_family,
    bench_dermatopathology_family,
    bench_ophthalmology_studies_family,
    bench_optometry_studies_family,
    bench_otolaryngology_studies_family,
)
from quant_fund.research.benches_w1222 import (
    bench_hand_surgery_family,
    bench_joint_replacement_family,
    bench_musculoskeletal_medicine_family,
    bench_orthopedics_studies_family,
    bench_spine_surgery_family,
    bench_sports_medicine_orthopedics_family,
)
from quant_fund.research.benches_w1223 import (
    bench_airway_management_family,
    bench_anesthesiology_studies_family,
    bench_pain_medicine_studies_family,
    bench_perioperative_medicine_family,
    bench_regional_anesthesia_family,
    bench_sedation_medicine_family,
)
from quant_fund.research.benches_w1224 import (
    bench_fetal_medicine_family,
    bench_gynecologic_oncology_family,
    bench_gynecology_studies_family,
    bench_maternal_fetal_medicine_family,
    bench_obstetrics_studies_family,
    bench_reproductive_endocrinology_family,
)
from quant_fund.research.benches_w1225 import (
    bench_colorectal_surgery_family,
    bench_general_surgery_studies_family,
    bench_hepatobiliary_surgery_family,
    bench_minimally_invasive_surgery_family,
    bench_surgical_oncology_studies_family,
    bench_trauma_surgery_family,
)
from quant_fund.research.benches_w1226 import (
    bench_anatomical_pathology_family,
    bench_clinical_pathology_family,
    bench_cytopathology_family,
    bench_histopathology_studies_family,
    bench_molecular_pathology_family,
    bench_pathology_studies_family,
)
from quant_fund.research.benches_w1227 import (
    bench_body_imaging_family,
    bench_diagnostic_imaging_family,
    bench_interventional_neuroradiology_family,
    bench_musculoskeletal_imaging_family,
    bench_pediatric_imaging_family,
    bench_radiology_studies_family,
)
from quant_fund.research.benches_w1228 import (
    bench_dental_studies_family,
    bench_endodontic_studies_family,
    bench_oral_surgery_studies_family,
    bench_orthodontic_studies_family,
    bench_pediatric_dentistry_family,
    bench_periodontal_studies_family,
)
from quant_fund.research.benches_w1229 import (
    bench_adolescent_medicine_studies_family,
    bench_developmental_pediatrics_family,
    bench_neonatal_medicine_studies_family,
    bench_pediatric_cardiology_family,
    bench_pediatric_oncology_family,
    bench_pediatrics_studies_family,
)
from quant_fund.research.benches_w1230 import (
    bench_acute_care_studies_family,
    bench_disaster_medicine_family,
    bench_emergency_medicine_studies_family,
    bench_resuscitation_medicine_family,
    bench_toxicology_medicine_family,
    bench_trauma_medicine_family,
)
from quant_fund.research.benches_w1231 import (
    bench_allergy_studies_family,
    bench_autoimmunity_studies_family,
    bench_hematopoietic_transplant_family,
    bench_immunodeficiency_studies_family,
    bench_immunology_medicine_family,
    bench_transplant_medicine_studies_family,
)
from quant_fund.research.benches_w1232 import (
    bench_caregiver_medicine_family,
    bench_falls_prevention_studies_family,
    bench_frailty_medicine_family,
    bench_geriatrics_studies_family,
    bench_memory_clinic_studies_family,
    bench_polypharmacy_studies_family,
)
from quant_fund.research.benches_w1233 import (
    bench_dysmorphology_studies_family,
    bench_genetic_diagnostics_family,
    bench_lysosomal_medicine_family,
    bench_medical_genetics_studies_family,
    bench_mitochondrial_medicine_family,
    bench_pharmacogenomics_studies_family,
)
from quant_fund.research.benches_w1234 import (
    bench_breast_medicine_family,
    bench_contraception_studies_family,
    bench_infertility_studies_family,
    bench_menopause_medicine_family,
    bench_pelvic_health_studies_family,
    bench_urogynecology_studies_family,
)
from quant_fund.research.benches_w1235 import (
    bench_connective_tissue_studies_family,
    bench_inflammatory_arthritis_studies_family,
    bench_myositis_studies_family,
    bench_osteoarthritis_studies_family,
    bench_rheumatology_medicine_family,
    bench_spondyloarthritis_studies_family,
)
from quant_fund.research.benches_w1236 import (
    bench_chronic_pain_studies_family,
    bench_fibromyalgia_studies_family,
    bench_headache_studies_family,
    bench_interventional_pain_studies_family,
    bench_neuropathic_pain_studies_family,
    bench_opioid_stewardship_studies_family,
)
from quant_fund.research.benches_w1237 import (
    bench_clinical_pharmacy_studies_family,
    bench_compounding_pharmacy_family,
    bench_hospital_pharmacy_studies_family,
    bench_medication_therapy_mgmt_family,
    bench_pharmacovigilance_studies_family,
    bench_pharmacy_practice_studies_family,
)
from quant_fund.research.benches_w1238 import (
    bench_aortic_medicine_studies_family,
    bench_lymphatic_medicine_family,
    bench_peripheral_artery_studies_family,
    bench_phlebology_studies_family,
    bench_vascular_lab_studies_family,
    bench_vascular_medicine_studies_family,
)
from quant_fund.research.benches_w1239 import (
    bench_celiac_studies_family,
    bench_gi_endoscopy_studies_family,
    bench_hepatology_medicine_family,
    bench_ibd_studies_family,
    bench_motility_studies_family,
    bench_pancreatic_medicine_family,
)
from quant_fund.research.benches_w1240 import (
    bench_asthma_studies_family,
    bench_bronchiectasis_studies_family,
    bench_copd_studies_family,
    bench_interstitial_lung_studies_family,
    bench_respiratory_studies_family,
    bench_sleep_breathing_studies_family,
)
from quant_fund.research.benches_w1241 import (
    bench_anemia_studies_family,
    bench_bleeding_disorders_family,
    bench_coagulation_studies_family,
    bench_hemoglobin_studies_family,
    bench_marrow_studies_family,
    bench_thrombosis_medicine_family,
)
from quant_fund.research.benches_w1242 import (
    bench_fracture_studies_family,
    bench_osteoporosis_studies_family,
    bench_physiatry_studies_family,
    bench_physical_therapy_studies_family,
    bench_rehabilitation_studies_family,
    bench_sports_injury_studies_family,
)
from quant_fund.research.benches_w1243 import (
    bench_healthcare_infection_studies_family,
    bench_mycosis_studies_family,
    bench_opportunistic_studies_family,
    bench_sepsis_studies_family,
    bench_sexually_transmitted_studies_family,
    bench_tuberculosis_studies_family,
)
from quant_fund.research.benches_w1244 import (
    bench_ct_imaging_studies_family,
    bench_mammography_studies_family,
    bench_mri_studies_family,
    bench_neuroradiology_studies_family,
    bench_pet_imaging_studies_family,
    bench_ultrasound_studies_family,
)
from quant_fund.research.benches_w1245 import (
    bench_breast_oncology_studies_family,
    bench_gi_oncology_studies_family,
    bench_immuno_oncology_studies_family,
    bench_medical_oncology_studies_family,
    bench_targeted_therapy_studies_family,
    bench_thoracic_oncology_studies_family,
)
from quant_fund.research.benches_w1246 import (
    bench_bariatric_surgery_studies_family,
    bench_burn_surgery_studies_family,
    bench_endocrine_surgery_studies_family,
    bench_pediatric_surgery_studies_family,
    bench_plastic_surgery_studies_family,
    bench_transplant_surgery_studies_family,
)
from quant_fund.research.benches_w1247 import (
    bench_aki_studies_family,
    bench_ckd_studies_family,
    bench_electrolyte_studies_family,
    bench_glomerular_studies_family,
    bench_stones_studies_family,
    bench_tubulointerstitial_studies_family,
)
from quant_fund.research.benches_w1248 import (
    bench_cataract_studies_family,
    bench_corneal_studies_family,
    bench_glaucoma_studies_family,
    bench_macular_studies_family,
    bench_refractive_studies_family,
    bench_retinal_studies_family,
)
from quant_fund.research.benches_w1249 import (
    bench_acne_studies_family,
    bench_alopecia_studies_family,
    bench_eczema_studies_family,
    bench_psoriasis_studies_family,
    bench_skin_cancer_studies_family,
    bench_vitiligo_studies_family,
)
from quant_fund.research.benches_w1250 import (
    bench_cochlear_studies_family,
    bench_head_neck_surgery_studies_family,
    bench_laryngology_studies_family,
    bench_otology_studies_family,
    bench_rhinology_studies_family,
    bench_sinus_studies_family,
)
from quant_fund.research.benches_w1251 import (
    bench_andrology_studies_family,
    bench_bladder_studies_family,
    bench_bph_studies_family,
    bench_erectile_studies_family,
    bench_incontinence_studies_family,
    bench_prostate_studies_family,
)
from quant_fund.research.benches_w1252 import (
    bench_hypothalamic_studies_family,
    bench_lipid_studies_family,
    bench_metabolic_syndrome_studies_family,
    bench_obesity_studies_family,
    bench_parathyroid_studies_family,
    bench_pituitary_studies_family,
)
from quant_fund.research.benches_w1253 import (
    bench_antibody_studies_family,
    bench_chemokine_studies_family,
    bench_complement_studies_family,
    bench_cytokine_studies_family,
    bench_interferon_studies_family,
    bench_lymphocyte_studies_family,
)
from quant_fund.research.benches_w1254 import (
    bench_allele_studies_family,
    bench_cnv_studies_family,
    bench_haplotype_studies_family,
    bench_pedigree_studies_family,
    bench_penetrance_studies_family,
    bench_snp_studies_family,
)
from quant_fund.research.benches_w1255 import (
    bench_als_studies_family,
    bench_alzheimer_studies_family,
    bench_dementia_studies_family,
    bench_huntington_studies_family,
    bench_ms_studies_family,
    bench_parkinson_studies_family,
)
from quant_fund.research.benches_w1256 import (
    bench_community_health_studies_family,
    bench_health_disparities_studies_family,
    bench_outbreak_studies_family,
    bench_screening_studies_family,
    bench_surveillance_studies_family,
    bench_vaccination_studies_family,
)
from quant_fund.research.benches_w1257 import (
    bench_culture_studies_family,
    bench_flow_cytometry_studies_family,
    bench_immunoassay_studies_family,
    bench_microscopy_studies_family,
    bench_pcr_studies_family,
    bench_serology_studies_family,
)
from quant_fund.research.benches_w1258 import (
    bench_interactome_studies_family,
    bench_metabolome_studies_family,
    bench_methylome_studies_family,
    bench_microbiome_studies_family,
    bench_proteome_studies_family,
    bench_transcriptome_studies_family,
)
from quant_fund.research.benches_w1259 import (
    bench_admet_studies_family,
    bench_de_novo_design_studies_family,
    bench_docking_studies_family,
    bench_lead_optimization_studies_family,
    bench_qsar_studies_family,
    bench_virtual_screening_studies_family,
)
from quant_fund.research.benches_w1260 import (
    bench_adaptive_trial_studies_family,
    bench_clinical_trial_studies_family,
    bench_comparative_effectiveness_studies_family,
    bench_meta_analysis_studies_family,
    bench_outcomes_research_studies_family,
    bench_rwe_studies_family,
)
from quant_fund.research.benches_w1261 import (
    bench_biostatistics_methods_studies_family,
    bench_epidemiology_methods_studies_family,
    bench_heor_studies_family,
    bench_regulatory_science_studies_family,
    bench_survival_trial_studies_family,
    bench_translational_studies_family,
)
from quant_fund.research.benches_w1262 import (
    bench_dynamic_borrowing_studies_family,
    bench_e_value_studies_family,
    bench_master_protocol_studies_family,
    bench_stepped_wedge_studies_family,
    bench_target_trial_emulation_studies_family,
    bench_win_ratio_studies_family,
)
from quant_fund.research.benches_w1263 import (
    bench_diagnostic_meta_studies_family,
    bench_fragility_index_studies_family,
    bench_individual_patient_meta_studies_family,
    bench_network_meta_studies_family,
    bench_trial_sequential_studies_family,
    bench_umbrella_review_studies_family,
)
from quant_fund.research.benches_w1264 import (
    bench_external_control_studies_family,
    bench_negative_control_studies_family,
    bench_probabilistic_bias_studies_family,
    bench_self_controlled_studies_family,
    bench_structural_nested_studies_family,
    bench_transportability_studies_family,
)
from quant_fund.research.benches_w1265 import (
    bench_colocalization_studies_family,
    bench_genetic_correlation_studies_family,
    bench_heritability_ldscore_studies_family,
    bench_mendelian_randomization_studies_family,
    bench_pleiotropy_robust_studies_family,
    bench_polygenic_score_studies_family,
)
from quant_fund.research.benches_w1266 import (
    bench_diffusion_lm_studies_family,
    bench_kv_compression_studies_family,
    bench_medusa_speculation_studies_family,
    bench_moe_shared_expert_studies_family,
    bench_rope_scaling_studies_family,
    bench_sparse_attention_studies_family,
)
from quant_fund.research.benches_w1267 import (
    bench_alpha_tensor_studies_family,
    bench_differentiable_sat_studies_family,
    bench_neural_theorem_studies_family,
    bench_program_synthesis_studies_family,
    bench_sketch_programming_studies_family,
    bench_symbolic_regression_dl_studies_family,
)
from quant_fund.research.benches_w1268 import (
    bench_curiosity_diversity_studies_family,
    bench_hindsight_relabel_studies_family,
    bench_occupancy_measure_studies_family,
    bench_option_discovery_studies_family,
    bench_skill_chain_studies_family,
    bench_successor_feature_studies_family,
)
from quant_fund.research.benches_w1269 import (
    bench_best_of_n_studies_family,
    bench_cdpo_studies_family,
    bench_constitutional_ai_studies_family,
    bench_orpo_studies_family,
    bench_simpo_studies_family,
    bench_sppo_studies_family,
)
from quant_fund.research.benches_w1270 import (
    bench_attribution_graph_studies_family,
    bench_causal_tracing_studies_family,
    bench_circuit_discovery_studies_family,
    bench_feature_geometry_studies_family,
    bench_gated_sae_studies_family,
    bench_transcoder_studies_family,
)
from quant_fund.research.benches_w1271 import (
    bench_arena_battle_studies_family,
    bench_bigbench_studies_family,
    bench_capability_elicitation_studies_family,
    bench_contamination_detect_studies_family,
    bench_helm_eval_studies_family,
    bench_llm_judge_studies_family,
)
from quant_fund.research.benches_w1272 import (
    bench_audio_encoder_studies_family,
    bench_document_ai_studies_family,
    bench_omni_modal_studies_family,
    bench_unified_tokenizer_studies_family,
    bench_video_llm_studies_family,
    bench_visual_grounding_studies_family,
)
from quant_fund.research.benches_w1273 import (
    bench_alignment_eval_studies_family,
    bench_guardrail_studies_family,
    bench_hallucination_detect_studies_family,
    bench_jailbreak_defense_studies_family,
    bench_red_team_studies_family,
    bench_sleeper_agent_studies_family,
)
from quant_fund.research.benches_w1274 import (
    bench_agent_memory_studies_family,
    bench_code_agent_studies_family,
    bench_computer_use_studies_family,
    bench_mcp_protocol_studies_family,
    bench_skill_library_studies_family,
    bench_web_agent_studies_family,
)
from quant_fund.research.benches_w1275 import (
    bench_deliberate_search_studies_family,
    bench_latent_reasoning_studies_family,
    bench_self_improvement_studies_family,
    bench_test_time_scaling_studies_family,
    bench_tree_thought_studies_family,
    bench_verifier_gated_studies_family,
)
from quant_fund.research.benches_w1276 import (
    bench_attribution_patching_studies_family,
    bench_causal_scrubbing_studies_family,
    bench_function_vector_studies_family,
    bench_induction_head_studies_family,
    bench_monosemantic_studies_family,
    bench_superposition_studies_family,
)
from quant_fund.research.benches_w1277 import (
    bench_debate_alignment_studies_family,
    bench_deliberative_alignment_studies_family,
    bench_iterated_amplification_studies_family,
    bench_recursive_reward_studies_family,
    bench_scalable_oversight_studies_family,
    bench_weak_to_strong_studies_family,
)
from quant_fund.research.benches_w1278 import (
    bench_chunked_prefill_studies_family,
    bench_continuous_batching_studies_family,
    bench_disaggregated_serving_studies_family,
    bench_early_exit_studies_family,
    bench_prefix_caching_studies_family,
    bench_tensor_parallel_studies_family,
)
from quant_fund.research.benches_w1279 import (
    bench_activation_checkpoint_studies_family,
    bench_fsdp_sharding_studies_family,
    bench_hybrid_parallel_studies_family,
    bench_pipeline_schedule_studies_family,
    bench_sequence_parallel_studies_family,
    bench_zero_optimizer_studies_family,
)
from quant_fund.research.benches_w1280 import (
    bench_data_mixture_studies_family,
    bench_data_quality_studies_family,
    bench_dedup_pipeline_studies_family,
    bench_domain_filtering_studies_family,
    bench_synthetic_data_studies_family,
    bench_token_budget_studies_family,
)
from quant_fund.research.benches_w1281 import (
    bench_analogical_prompting_studies_family,
    bench_graph_of_thought_studies_family,
    bench_least_to_most_studies_family,
    bench_plan_and_solve_studies_family,
    bench_step_back_studies_family,
    bench_tree_of_thought_studies_family,
)
from quant_fund.research.benches_w1282 import (
    bench_affordance_map_studies_family,
    bench_embodied_agent_studies_family,
    bench_spatial_reasoning_studies_family,
    bench_video_diffusion_studies_family,
    bench_vla_model_studies_family,
    bench_world_sim_studies_family,
)
from quant_fund.research.benches_w1283 import (
    bench_context_compression_studies_family,
    bench_episodic_memory_studies_family,
    bench_memory_bank_studies_family,
    bench_retrieval_memory_studies_family,
    bench_semantic_memory_studies_family,
    bench_working_memory_studies_family,
)
from quant_fund.research.benches_w1284 import (
    bench_citation_check_studies_family,
    bench_claim_verifier_studies_family,
    bench_entailment_studies_family,
    bench_factuality_score_studies_family,
    bench_grounding_verify_studies_family,
    bench_self_reflect_studies_family,
)
from quant_fund.research.benches_w1285 import (
    bench_audio_lm_studies_family,
    bench_chart_reasoning_studies_family,
    bench_doc_vqa_studies_family,
    bench_gui_agent_studies_family,
    bench_video_understanding_studies_family,
    bench_vision_pretraining_studies_family,
)
from quant_fund.research.benches_w1286 import (
    bench_beacon_context_studies_family,
    bench_hierarchical_context_studies_family,
    bench_infini_attention_studies_family,
    bench_landmark_attention_studies_family,
    bench_ntk_scaling_studies_family,
    bench_yarn_scaling_studies_family,
)
from quant_fund.research.benches_w1287 import (
    bench_capability_eval_studies_family,
    bench_control_eval_studies_family,
    bench_deception_eval_studies_family,
    bench_prompt_injection_studies_family,
    bench_sandbox_escape_studies_family,
    bench_tool_call_verify_studies_family,
)
from quant_fund.research.benches_w1288 import (
    bench_adversarial_irl_studies_family,
    bench_behavior_cloning_studies_family,
    bench_dagger_studies_family,
    bench_offline_distill_studies_family,
    bench_preference_irl_studies_family,
    bench_skill_extraction_studies_family,
)
from quant_fund.research.benches_w1289 import (
    bench_analogical_prompt_studies_family,
    bench_cot_studies_family,
    bench_reflexion_studies_family,
    bench_scratchpad_studies_family,
    bench_self_consistency_studies_family,
    bench_stepwise_verify_studies_family,
)
from quant_fund.research.benches_w1290 import (
    bench_awq_studies_family,
    bench_entropy_code_quant_studies_family,
    bench_gptq_studies_family,
    bench_kv_cache_quant_studies_family,
    bench_smoothquant_studies_family,
    bench_weight_share_studies_family,
)
from quant_fund.research.benches_w1291 import (
    bench_data_mix_studies_family,
    bench_dedup_minhash_studies_family,
    bench_dedup_studies_family,
    bench_domain_classifier_studies_family,
    bench_perplexity_filter_studies_family,
    bench_quality_filter_studies_family,
)
from quant_fund.research.benches_w1292 import (
    bench_grpo_studies_family,
    bench_math_reward_studies_family,
    bench_outcome_reward_studies_family,
    bench_process_reward_studies_family,
    bench_rlvr_studies_family,
    bench_verifiable_reward_studies_family,
)
from quant_fund.research.benches_w1293 import (
    bench_cai_critique_studies_family,
    bench_constitutional_studies_family,
    bench_harmlessness_rl_studies_family,
    bench_principle_eval_studies_family,
    bench_rlaif_studies_family,
    bench_sleeper_eval_studies_family,
)
from quant_fund.research.benches_w1294 import (
    bench_activation_oracle_studies_family,
    bench_concept_vector_studies_family,
    bench_feature_ablation_studies_family,
    bench_honesty_vector_studies_family,
    bench_reading_vector_studies_family,
    bench_refusal_vector_studies_family,
)
from quant_fund.research.benches_w1295 import (
    bench_activation_patch_studies_family,
    bench_circuit_tracer_studies_family,
    bench_feature_dashboard_studies_family,
    bench_jailbreak_detect_studies_family,
    bench_mech_anomaly_studies_family,
    bench_sae_linter_studies_family,
)
from quant_fund.research.benches_w1296 import (
    bench_ensemble_rm_studies_family,
    bench_judge_reward_studies_family,
    bench_margin_reward_studies_family,
    bench_reward_hacking_studies_family,
    bench_reward_uncertainty_studies_family,
    bench_rm_btd_studies_family,
)
from quant_fund.research.benches_w1297 import (
    bench_benchmark_gaming_studies_family,
    bench_benchmark_saturate_studies_family,
    bench_contamination_studies_family,
    bench_eval_coverage_studies_family,
    bench_eval_reliability_studies_family,
    bench_lm_eval_harness_studies_family,
)
from quant_fund.research.benches_w1298 import (
    bench_browse_eval_studies_family,
    bench_os_world_studies_family,
    bench_swe_bench_studies_family,
    bench_terminal_bench_studies_family,
    bench_tool_use_eval_studies_family,
    bench_web_arena_studies_family,
)
from quant_fund.research.benches_w1299 import (
    bench_agent_harm_studies_family,
    bench_harm_bench_studies_family,
    bench_jailbreak_bench_studies_family,
    bench_prompt_inject_studies_family,
    bench_safety_bench_studies_family,
    bench_xstest_studies_family,
)
from quant_fund.research.benches_w1300 import (
    bench_adversarial_eval_studies_family,
    bench_autoattack_studies_family,
    bench_corruption_studies_family,
    bench_imagenet_c_studies_family,
    bench_imagenet_r_studies_family,
    bench_robust_bench_studies_family,
)
from quant_fund.research.benches_w1301 import (
    bench_backdoor_studies_family,
    bench_clean_label_studies_family,
    bench_data_poison_studies_family,
    bench_neural_cleanse_studies_family,
    bench_spectral_signature_studies_family,
    bench_trojan_studies_family,
)
from quant_fund.research.benches_w1302 import (
    bench_canary_infer_studies_family,
    bench_deep_leak_studies_family,
    bench_gradient_leak_studies_family,
    bench_lira_studies_family,
    bench_membership_infer_studies_family,
    bench_shadow_model_studies_family,
)
from quant_fund.research.benches_w1303 import (
    bench_bbh_studies_family,
    bench_gsm8k_studies_family,
    bench_humaneval_studies_family,
    bench_ifeval_studies_family,
    bench_mmlu_studies_family,
    bench_mt_bench_studies_family,
)
from quant_fund.research.benches_w1304 import (
    bench_boolq_studies_family,
    bench_copa_studies_family,
    bench_hellaswag_studies_family,
    bench_openbookqa_studies_family,
    bench_piqa_studies_family,
    bench_siqa_studies_family,
)
from quant_fund.research.benches_w1305 import (
    bench_coqa_studies_family,
    bench_drop_studies_family,
    bench_hotpotqa_studies_family,
    bench_nq_studies_family,
    bench_squad_studies_family,
    bench_triviaqa_studies_family,
)
from quant_fund.research.benches_w1306 import (
    bench_glue_studies_family,
    bench_mnli_studies_family,
    bench_qnli_studies_family,
    bench_rte_studies_family,
    bench_super_glue_studies_family,
    bench_wnli_studies_family,
)
from quant_fund.research.benches_w1307 import (
    bench_lambada_studies_family,
    bench_record_studies_family,
    bench_story_cloze_studies_family,
    bench_winogender_studies_family,
    bench_winograd_studies_family,
    bench_wsc_studies_family,
)
from quant_fund.research.benches_w1308 import (
    bench_halu_eval_studies_family,
    bench_infinite_bench_studies_family,
    bench_longmem_studies_family,
    bench_needle_haystack_studies_family,
    bench_ruler_studies_family,
    bench_truthful_qa_studies_family,
)
from quant_fund.research.benches_w1309 import (
    bench_frontier_math_studies_family,
    bench_gpqa_studies_family,
    bench_hle_studies_family,
    bench_mmlu_pro_studies_family,
    bench_tau_bench_studies_family,
    bench_workarena_studies_family,
)
from quant_fund.research.benches_w1310 import (
    bench_alpaca_eval_studies_family,
    bench_attribution_eval_studies_family,
    bench_citation_eval_studies_family,
    bench_diversity_eval_studies_family,
    bench_factscore_studies_family,
    bench_self_bleu_studies_family,
)
from quant_fund.research.benches_w1311 import (
    bench_aegis_studies_family,
    bench_air_bench_studies_family,
    bench_overkill_studies_family,
    bench_salad_bench_studies_family,
    bench_sorry_bench_studies_family,
    bench_wildguard_studies_family,
)
from quant_fund.research.benches_w1312 import (
    bench_bio_risk_eval_studies_family,
    bench_chem_risk_eval_studies_family,
    bench_cyber_sec_eval_studies_family,
    bench_lab_bench_studies_family,
    bench_malicious_instruct_studies_family,
    bench_wmdp_studies_family,
)
from quant_fund.research.benches_w1313 import (
    bench_abs_scan_studies_family,
    bench_activation_cluster_studies_family,
    bench_fine_pruning_studies_family,
    bench_sleepless_studies_family,
    bench_strip_defense_studies_family,
    bench_watermark_studies_family,
)
from quant_fund.research.benches_w1314 import (
    bench_attribute_infer_studies_family,
    bench_extraction_studies_family,
    bench_inversion_studies_family,
    bench_model_stealing_studies_family,
    bench_property_infer_studies_family,
    bench_reconstruction_studies_family,
)
from quant_fund.research.benches_w1315 import (
    bench_imagenet_a_studies_family,
    bench_imagenet_e_studies_family,
    bench_imagenet_o_studies_family,
    bench_imagenet_sketch_studies_family,
    bench_imagenet_v2_studies_family,
    bench_stylized_studies_family,
)
from quant_fund.research.benches_w1316 import (
    bench_backgrounds_studies_family,
    bench_cue_conflict_studies_family,
    bench_geirhos_studies_family,
    bench_imagenet_bg_studies_family,
    bench_shape_bias_studies_family,
    bench_texture_bias_studies_family,
)
from quant_fund.research.benches_w1317 import (
    bench_arc_eval_studies_family,
    bench_do_anything_studies_family,
    bench_step_eval_studies_family,
    bench_strong_reject_studies_family,
    bench_verifier_reward_studies_family,
    bench_winogrande_studies_family,
)
from quant_fund.research.benches_w1318 import (
    bench_cb_studies_family,
    bench_cola_studies_family,
    bench_qqp_studies_family,
    bench_squad_v2_studies_family,
    bench_sst2_studies_family,
    bench_wic_studies_family,
)
from quant_fund.research.benches_w1319 import (
    bench_math_bench_studies_family,
    bench_multirc_studies_family,
    bench_ninco_studies_family,
    bench_objectnet_studies_family,
    bench_ood_bench_studies_family,
    bench_wild_bench_studies_family,
)
from quant_fund.research.benches_w1320 import (
    bench_gaia_bench_studies_family,
    bench_mmbench_agent_studies_family,
    bench_osworld_studies_family,
    bench_screen_eval_studies_family,
    bench_vsi_bench_studies_family,
    bench_webvoyager_studies_family,
)
from quant_fund.research.benches_w1321 import (
    bench_chart_gqa_studies_family,
    bench_mathvista_studies_family,
    bench_mkqa_studies_family,
    bench_mmmlu_studies_family,
    bench_mmmu_studies_family,
    bench_videomme_studies_family,
)
from quant_fund.research.benches_w1322 import (
    bench_logic_bench_studies_family,
    bench_minif2f_studies_family,
    bench_olympiad_bench_studies_family,
    bench_putnam_studies_family,
    bench_truthfulqa_studies_family,
    bench_zebra_logic_studies_family,
)
from quant_fund.research.benches_w1323 import (
    bench_bigcodebench_studies_family,
    bench_ds1000_studies_family,
    bench_humaneval_plus_studies_family,
    bench_livecodebench_studies_family,
    bench_mbpp_plus_studies_family,
    bench_swe_perf_studies_family,
)
from quant_fund.research.benches_w1324 import (
    bench_alpacaeval_studies_family,
    bench_arena_hard_studies_family,
    bench_judge_bench_studies_family,
    bench_mt_bench_judge_studies_family,
    bench_prometheus_eval_studies_family,
    bench_reward_bench_studies_family,
)
from quant_fund.research.benches_w1325 import (
    bench_babilong_studies_family,
    bench_infinitebench_studies_family,
    bench_longbench_studies_family,
    bench_lv_eval_studies_family,
    bench_ruler_bench_studies_family,
    bench_zero_scrolls_studies_family,
)
from quant_fund.research.benches_w1326 import (
    bench_attribute_inference_studies_family,
    bench_canary_memorization_studies_family,
    bench_extraction_attack_studies_family,
    bench_membership_inference_studies_family,
    bench_model_inversion_studies_family,
    bench_privacy_meter_studies_family,
)
from quant_fund.research.benches_w1327 import (
    bench_bbq_bias_studies_family,
    bench_bold_bias_studies_family,
    bench_crowspairs_studies_family,
    bench_holist_bias_studies_family,
    bench_realtoxicity_studies_family,
    bench_toxigen_eval_studies_family,
)
from quant_fund.research.benches_w1328 import (
    bench_apps_bench_studies_family,
    bench_class_eval_studies_family,
    bench_code_contests_studies_family,
    bench_multipl_e_studies_family,
    bench_polyglot_bench_studies_family,
    bench_repobench_studies_family,
)
from quant_fund.research.benches_w1329 import (
    bench_gov_report_studies_family,
    bench_looogle_studies_family,
    bench_lost_middle_studies_family,
    bench_marathon_eval_studies_family,
    bench_niah_v2_studies_family,
    bench_passkey_retrieval_studies_family,
)
from quant_fund.research.benches_w1330 import (
    bench_beaver_safe_studies_family,
    bench_do_not_answer_studies_family,
    bench_hh_rlhf_studies_family,
    bench_honest_eval_studies_family,
    bench_safe_rlhf_studies_family,
    bench_sos_bench_studies_family,
)
from quant_fund.research.benches_w1331 import (
    bench_codescope_studies_family,
    bench_concode_eval_studies_family,
    bench_crosscodeeval_studies_family,
    bench_mer_bench_studies_family,
    bench_project_eval_studies_family,
    bench_swe_bench_verified_studies_family,
)
from quant_fund.research.benches_w1332 import (
    bench_decontaminate_studies_family,
    bench_eval_bias_studies_family,
    bench_fair_eval_studies_family,
    bench_g_eval_studies_family,
    bench_ngram_overlap_studies_family,
    bench_pandalm_studies_family,
)
from quant_fund.research.benches_w1333 import (
    bench_fava_studies_family,
    bench_polyglo_tox_studies_family,
    bench_regard_eval_studies_family,
    bench_unqover_studies_family,
    bench_vlur_studies_family,
    bench_xlsum_studies_family,
)
from quant_fund.research.benches_w1334 import (
    bench_books_qa_studies_family,
    bench_lcc_codebase_studies_family,
    bench_multi_news_eval_studies_family,
    bench_narrative_qa_studies_family,
    bench_needle_multi_studies_family,
    bench_qmsum_eval_studies_family,
)
from quant_fund.research.benches_w1335 import (
    bench_code_rag_studies_family,
    bench_codegen_universal_studies_family,
    bench_long_code_bench_studies_family,
    bench_odex_eval_studies_family,
    bench_swe_dev_studies_family,
    bench_swe_multimodal_studies_family,
)
from quant_fund.research.benches_w1336 import (
    bench_assistantbench_studies_family,
    bench_mind2web_studies_family,
    bench_miniwob_studies_family,
    bench_visual_web_studies_family,
    bench_web_nav_studies_family,
    bench_webarena_studies_family,
)
from quant_fund.research.benches_w1337 import (
    bench_android_env_studies_family,
    bench_api_bank_studies_family,
    bench_gaia_level_studies_family,
    bench_video_game_studies_family,
    bench_voyager_minecraft_studies_family,
    bench_web_shopping_studies_family,
)
from quant_fund.research.benches_w1338 import (
    bench_arith_qa_studies_family,
    bench_gsm_hard_studies_family,
    bench_math_reason_studies_family,
    bench_mini_f2f_studies_family,
    bench_proof_pile_studies_family,
    bench_theorem_qa_studies_family,
)
from quant_fund.research.benches_w1339 import (
    bench_aqua_rat_studies_family,
    bench_geo_qa_studies_family,
    bench_hol_step_studies_family,
    bench_math_odyssey_studies_family,
    bench_tab_math_studies_family,
    bench_uni_math_studies_family,
)
from quant_fund.research.benches_w1340 import (
    bench_arc_challenge_studies_family,
    bench_bio_qa_studies_family,
    bench_med_qa_studies_family,
    bench_openbook_qa_studies_family,
    bench_pubmed_qa_studies_family,
    bench_sci_q_studies_family,
)
from quant_fund.research.benches_w1341 import (
    bench_bold_eval_studies_family,
    bench_crow_s_pairs_studies_family,
    bench_hate_speech_eval_studies_family,
    bench_holo_bias_studies_family,
    bench_real_toxicity_studies_family,
    bench_stereo_set_studies_family,
)
from quant_fund.research.benches_w1342 import (
    bench_fairness_eval_studies_family,
    bench_gender_bias_studies_family,
    bench_jigsaw_tox_studies_family,
    bench_nlp_bias_studies_family,
    bench_pronoun_bias_studies_family,
    bench_regard_metric_studies_family,
)
from quant_fund.research.benches_w1343 import (
    bench_coqa_qa_studies_family,
    bench_drop_qa_studies_family,
    bench_news_qa_studies_family,
    bench_quac_qa_studies_family,
    bench_quail_qa_studies_family,
    bench_quoref_qa_studies_family,
)
from quant_fund.research.benches_w1344 import (
    bench_boolq_qa_studies_family,
    bench_dream_qa_studies_family,
    bench_duorc_qa_studies_family,
    bench_mctest_qa_studies_family,
    bench_qasper_qa_studies_family,
    bench_race_qa_studies_family,
)
from quant_fund.research.benches_w1345 import (
    bench_complex_qa_studies_family,
    bench_entity_quests_studies_family,
    bench_freebase_qa_studies_family,
    bench_nq_open_studies_family,
    bench_trivia_qa_studies_family,
    bench_web_qa_studies_family,
)
from quant_fund.research.benches_w1346 import (
    bench_grail_qa_studies_family,
    bench_graph_questions_studies_family,
    bench_kqa_pro_studies_family,
    bench_lc_quad_studies_family,
    bench_mintaka_qa_studies_family,
    bench_spinach_qa_studies_family,
)
from quant_fund.research.benches_w1347 import (
    bench_abductive_nli_studies_family,
    bench_conseq_log_studies_family,
    bench_logiqa_log_studies_family,
    bench_lsat_log_studies_family,
    bench_reason_mc_studies_family,
    bench_recli_log_studies_family,
)
from quant_fund.research.benches_w1348 import (
    bench_anli_r1_studies_family,
    bench_anli_r2_studies_family,
    bench_anli_r3_studies_family,
    bench_mnli_match_studies_family,
    bench_scitail_lite_studies_family,
    bench_snli_lite_studies_family,
)
from quant_fund.research.benches_w1349 import (
    bench_cola_lite_studies_family,
    bench_qnli_lite_studies_family,
    bench_qqp_lite_studies_family,
    bench_sst2_lite_studies_family,
    bench_stsb_lite_studies_family,
    bench_wnli_lite_studies_family,
)
from quant_fund.research.benches_w1350 import (
    bench_dyck_lang_studies_family,
    bench_hops_add_studies_family,
    bench_lcmc_lite_studies_family,
    bench_mco_lite_studies_family,
    bench_scan_cfsp_studies_family,
    bench_shuffle_expr_studies_family,
)
from quant_fund.research.benches_w1351 import (
    bench_ethic_jiminy_studies_family,
    bench_moral_exc_studies_family,
    bench_moral_found_studies_family,
    bench_principlism_toy_studies_family,
    bench_scruples_lite_studies_family,
    bench_virtue_ethics_studies_family,
)
from quant_fund.research.benches_w1352 import (
    bench_bias_bench_studies_family,
    bench_crowsp_lite_studies_family,
    bench_honesty_lie_studies_family,
    bench_social_iqa2_studies_family,
    bench_stereo_lite_studies_family,
    bench_wino_bias_studies_family,
)
from quant_fund.research.benches_w1353 import (
    bench_creak_lite_studies_family,
    bench_entailment_bn_studies_family,
    bench_hans_lite_studies_family,
    bench_prove_it_studies_family,
    bench_strategy_qa_studies_family,
    bench_sup_nli_studies_family,
)
from quant_fund.research.benches_w1354 import (
    bench_arc_easy2_studies_family,
    bench_boolq_lite_studies_family,
    bench_cosmos_qa_studies_family,
    bench_race_lite_studies_family,
    bench_sciq_lite_studies_family,
    bench_social_qa_studies_family,
)
from quant_fund.research.benches_w1355 import (
    bench_arc_hard2_studies_family,
    bench_csqa_lite_studies_family,
    bench_hellaswag_lite_studies_family,
    bench_piqa_lite_studies_family,
    bench_prost_lite_studies_family,
    bench_swag_lite_studies_family,
)
from quant_fund.research.benches_w1356 import (
    bench_bigbench_lite_studies_family,
    bench_entity_qa_studies_family,
    bench_mmlu_lite_studies_family,
    bench_natural_qa_studies_family,
    bench_pop_qa_studies_family,
    bench_triviaqa_lite_studies_family,
)
from quant_fund.research.benches_w1357 import (
    bench_hendrycks_test_studies_family,
    bench_hotpot_lite_studies_family,
    bench_multirc_lite_studies_family,
    bench_quoref_lite_studies_family,
    bench_record_lite_studies_family,
    bench_squad_lite2_studies_family,
)
from quant_fund.research.benches_w1358 import (
    bench_adver_qa_studies_family,
    bench_coqa_lite_studies_family,
    bench_drop_lite_studies_family,
    bench_duo_rc_studies_family,
    bench_quac_lite_studies_family,
    bench_trivia_web_studies_family,
)
from quant_fund.research.benches_w1359 import (
    bench_argu_ana_studies_family,
    bench_babi_lite_studies_family,
    bench_curious_qa_studies_family,
    bench_qasper_lite_studies_family,
    bench_scifact_lite_studies_family,
    bench_web_questions_studies_family,
)
from quant_fund.research.benches_w1360 import (
    bench_bioasq_lite_studies_family,
    bench_cite_worth_studies_family,
    bench_climate_fever_studies_family,
    bench_fever_lite_studies_family,
    bench_touch_e_studies_family,
    bench_verdict_qa_studies_family,
)
from quant_fund.research.benches_w1361 import (
    bench_covid_lies_studies_family,
    bench_evidence_inf_studies_family,
    bench_hoax_detect_studies_family,
    bench_liar_lite_studies_family,
    bench_rumor_eval_studies_family,
    bench_scidtb_lite_studies_family,
)
from quant_fund.research.benches_w1362 import (
    bench_check_that_studies_family,
    bench_claim_buster_studies_family,
    bench_emergent_lite_studies_family,
    bench_fake_news_studies_family,
    bench_snopes_lite_studies_family,
    bench_stance_detect_studies_family,
)
from quant_fund.research.benches_w1363 import (
    bench_age_bias_studies_family,
    bench_curry_qa_studies_family,
    bench_cw_qa2_studies_family,
    bench_dialect_bias_studies_family,
    bench_politi_fact_studies_family,
    bench_rumor_twitter_studies_family,
)
from quant_fund.research.benches_w1364 import (
    bench_anli_lite_studies_family,
    bench_mnli_lite_studies_family,
    bench_mrpc_lite_studies_family,
    bench_paws_lite_studies_family,
    bench_quora_dup_studies_family,
    bench_rte_lite_studies_family,
)
from quant_fund.research.benches_w1365 import (
    bench_arxiv_sum_studies_family,
    bench_cnn_dailymail_studies_family,
    bench_dialogsum_lite_studies_family,
    bench_multi_news_studies_family,
    bench_pubmed_sum_studies_family,
    bench_samsum_lite_studies_family,
)
from quant_fund.research.benches_w1366 import (
    bench_align_score_studies_family,
    bench_dice_eval_studies_family,
    bench_factcc_lite_studies_family,
    bench_faith_eval_studies_family,
    bench_quest_eval_studies_family,
    bench_summa_eval_studies_family,
)
from quant_fund.research.benches_w1367 import (
    bench_bert_score_studies_family,
    bench_bleu_rouge_studies_family,
    bench_bleurt_lite_studies_family,
    bench_comet_mt_studies_family,
    bench_meteor_lite_studies_family,
    bench_rouge_lite_studies_family,
)
from quant_fund.research.benches_w1368 import (
    bench_chr_f_studies_family,
    bench_mover_score_studies_family,
    bench_nist_metric_studies_family,
    bench_prism_mt_studies_family,
    bench_sacrebleu_lite_studies_family,
    bench_ter_lite_studies_family,
)
from quant_fund.research.benches_w1369 import (
    bench_aime_eval_studies_family,
    bench_asdiv_lite_studies_family,
    bench_math500_lite_studies_family,
    bench_mgsm_lite_studies_family,
    bench_minerva_math_studies_family,
    bench_svamp_lite_studies_family,
)
from quant_fund.research.benches_w1370 import (
    bench_bbh_lite_studies_family,
    bench_gpqa_lite_studies_family,
    bench_if_eval_studies_family,
    bench_live_bench_studies_family,
    bench_olympic_bench_studies_family,
    bench_trivia_qa_lite_studies_family,
)
from quant_fund.research.benches_w1371 import (
    bench_ethos_lite_studies_family,
    bench_moral_stories_studies_family,
    bench_mutual_lite_studies_family,
    bench_prosocial_lite_studies_family,
    bench_scruples_studies_family,
    bench_siqa_lite_studies_family,
)
from quant_fund.research.benches_w1372 import (
    bench_billsum_lite_studies_family,
    bench_booksum_lite_studies_family,
    bench_elm_lite_studies_family,
    bench_govreport_lite_studies_family,
    bench_qmsum_lite_studies_family,
    bench_wikisum_lite_studies_family,
)
from quant_fund.research.benches_w1373 import (
    bench_commonsense_lite_studies_family,
    bench_logi_qa_studies_family,
    bench_mr_lite_studies_family,
    bench_muin_lite_studies_family,
    bench_qasc_sci2_studies_family,
    bench_winogrande_lite_studies_family,
)
from quant_fund.research.benches_w1374 import (
    bench_deduc_lite_studies_family,
    bench_entail_bank_studies_family,
    bench_folio_lite_studies_family,
    bench_logic_nli_studies_family,
    bench_proof_writer_studies_family,
    bench_rule_taker_studies_family,
)
from quant_fund.research.benches_w1375 import (
    bench_facet_sum_studies_family,
    bench_ms2_lite_studies_family,
    bench_patent_sum_studies_family,
    bench_sci_lay_studies_family,
    bench_scitldr_lite_studies_family,
    bench_spectrum_sum_studies_family,
)
from quant_fund.research.benches_w1376 import (
    bench_blender_bot_studies_family,
    bench_conv_ai2_studies_family,
    bench_daily_dialog_studies_family,
    bench_dstc_lite_studies_family,
    bench_empathy_dialog_studies_family,
    bench_persona_chat_studies_family,
)
from quant_fund.research.benches_w1377 import (
    bench_abduction_lite_studies_family,
    bench_board_game_qa_studies_family,
    bench_conv_finqa_studies_family,
    bench_dream_lite_studies_family,
    bench_equiv_lite_studies_family,
    bench_wsc_lite_studies_family,
)
from quant_fund.research.benches_w1378 import (
    bench_aqua_lite_studies_family,
    bench_fin_qa_studies_family,
    bench_math_qa_studies_family,
    bench_num_glue_studies_family,
    bench_tab_fact_studies_family,
    bench_tat_qa_studies_family,
)
from quant_fund.research.benches_w1379 import (
    bench_bary_score_studies_family,
    bench_cider_lite_studies_family,
    bench_gleu_lite_studies_family,
    bench_kl_div_eval_studies_family,
    bench_rouge_we_studies_family,
    bench_wmt_metric_studies_family,
)
from quant_fund.research.benches_w1380 import (
    bench_art_nli_studies_family,
    bench_para_paws_studies_family,
    bench_recast_lite_studies_family,
    bench_snips_lite_studies_family,
    bench_social_lite_studies_family,
    bench_subj_lite_studies_family,
)
from quant_fund.research.benches_w1381 import (
    bench_begins_lite_studies_family,
    bench_diamonds_lite_studies_family,
    bench_faithful_dial_studies_family,
    bench_multi_woz_studies_family,
    bench_top_dialog_studies_family,
    bench_wow_lite_studies_family,
)
from quant_fund.research.benches_w1382 import (
    bench_fact_score_studies_family,
    bench_gpt_score_studies_family,
    bench_helm_lite_studies_family,
    bench_lmsys_eval_studies_family,
    bench_nugget_eval_studies_family,
    bench_vicuna_bench_studies_family,
)
from quant_fund.research.benches_w1383 import (
    bench_aime24_studies_family,
    bench_gpqa_diamond_studies_family,
    bench_hle_lite_studies_family,
    bench_mmmlu_lite_studies_family,
    bench_olympiadbench_studies_family,
    bench_super_gpqa_studies_family,
)
from quant_fund.research.benches_w1384 import (
    bench_aider_polyglot_studies_family,
    bench_hum_eval_studies_family,
    bench_livebench_arena_studies_family,
    bench_mbti_eval_studies_family,
    bench_olmes_lite_studies_family,
    bench_plus_eval_studies_family,
)
from quant_fund.research.benches_w1385 import (
    bench_airtasks_studies_family,
    bench_browsergym_studies_family,
    bench_maze_eval_studies_family,
    bench_mmind2web_studies_family,
    bench_screenqa_studies_family,
    bench_weblinx_studies_family,
)
from quant_fund.research.benches_w1386 import (
    bench_alfworld_lite_studies_family,
    bench_babyai_lite_studies_family,
    bench_crafter_lite_studies_family,
    bench_jericho_lite_studies_family,
    bench_scienceworld_studies_family,
    bench_textworld_lite_studies_family,
)
from quant_fund.research.benches_w1387 import (
    bench_api_blend_studies_family,
    bench_bfcl_v3_studies_family,
    bench_gorilla_eval_studies_family,
    bench_gta_bench_studies_family,
    bench_seal_tools_studies_family,
    bench_stabletoolbench_studies_family,
)
from quant_fund.research.benches_w1388 import (
    bench_corpus_qa_studies_family,
    bench_crag_bench_studies_family,
    bench_domain_rag_studies_family,
    bench_freshqa_studies_family,
    bench_ragas_lite_studies_family,
    bench_rgb_eval_studies_family,
)
from quant_fund.research.benches_w1389 import (
    bench_book_sum_studies_family,
    bench_fanout_qa_studies_family,
    bench_infinitesum_studies_family,
    bench_marlense_studies_family,
    bench_narra_sum_studies_family,
    bench_quote_sum_studies_family,
)
from quant_fund.research.benches_w1390 import (
    bench_episum_lite_studies_family,
    bench_fsum_lite_studies_family,
    bench_mds_news_studies_family,
    bench_sqcs_lite_studies_family,
    bench_summon_fce_studies_family,
    bench_wcep_lite_studies_family,
)
from quant_fund.research.benches_w1391 import (
    bench_archer_qa_studies_family,
    bench_argue_eval_studies_family,
    bench_expert_qa_studies_family,
    bench_mintaka_lite_studies_family,
    bench_musique_lite_studies_family,
    bench_wiki2_qa_studies_family,
)
from quant_fund.research.benches_w1392 import (
    bench_meta_tool_studies_family,
    bench_nest_tools_studies_family,
    bench_toolbench2_studies_family,
    bench_toolqa_lite_studies_family,
    bench_ultra_tool_studies_family,
    bench_work_plus_studies_family,
)
from quant_fund.research.benches_w1393 import (
    bench_hamming_mcp_studies_family,
    bench_mcp_bench_studies_family,
    bench_net_hack_studies_family,
    bench_tool_sandbox_studies_family,
    bench_videoweb_studies_family,
    bench_webshop_lite_studies_family,
)
from quant_fund.research.benches_w1394 import (
    bench_api_eval_studies_family,
    bench_apps_lite_studies_family,
    bench_livecode_studies_family,
    bench_mbpp_lite_studies_family,
    bench_restbench_studies_family,
    bench_swe_gym_studies_family,
)
from quant_fund.research.benches_w1395 import (
    bench_bamboogle_studies_family,
    bench_fine_qa_studies_family,
    bench_hotpot2_studies_family,
    bench_kwik_qa_studies_family,
    bench_quest_qa_studies_family,
    bench_tatqa2_studies_family,
)
from quant_fund.research.benches_w1396 import (
    bench_agnews_lite_studies_family,
    bench_dialsum_lite_studies_family,
    bench_facet_lite_studies_family,
    bench_medsum_lite_studies_family,
    bench_oposum_lite_studies_family,
    bench_qsum_lite_studies_family,
)
from quant_fund.research.benches_w1397 import (
    bench_coma_qa_studies_family,
    bench_gaia_lite_studies_family,
    bench_simple_qa_studies_family,
    bench_sqa_lite_studies_family,
    bench_tqa_lite_studies_family,
    bench_tydiqa_lite_studies_family,
)
from quant_fund.research.benches_w1398 import (
    bench_doc2dial_studies_family,
    bench_finqa_lite_studies_family,
    bench_hybridqa_lite_studies_family,
    bench_infotabs_studies_family,
    bench_ottqa_lite_studies_family,
    bench_tab_cwq_studies_family,
)
from quant_fund.research.benches_w1399 import (
    bench_asqa_lite_studies_family,
    bench_eli5_lite_studies_family,
    bench_fresh_qa_studies_family,
    bench_nq_lite_studies_family,
    bench_trivia_lite_studies_family,
    bench_xor_tydi_studies_family,
)
from quant_fund.research.benches_w1400 import (
    bench_cronqa_lite_studies_family,
    bench_cwq_lite_studies_family,
    bench_grailqa_studies_family,
    bench_kgqa_lite_studies_family,
    bench_pweb_qa_studies_family,
    bench_qald_lite_studies_family,
)
from quant_fund.research.benches_w1401 import (
    bench_alg514_lite_studies_family,
    bench_dolphin_lite_studies_family,
    bench_draw_lite_studies_family,
    bench_lila_lite_studies_family,
    bench_math_doc_studies_family,
    bench_math_eval_studies_family,
)
from quant_fund.research.benches_w1402 import (
    bench_ai2_arc_lite_studies_family,
    bench_arc_da_lite_studies_family,
    bench_drug_qa_lite_studies_family,
    bench_emrqa_lite_studies_family,
    bench_head_qa_lite_studies_family,
    bench_medmcqa_lite_studies_family,
)
from quant_fund.research.benches_w1403 import (
    bench_ai2d_lite_studies_family,
    bench_chart_qa_lite_studies_family,
    bench_docvqa_lite_studies_family,
    bench_infovqa_lite_studies_family,
    bench_mmqa_lite_studies_family,
    bench_ocrvqa_lite_studies_family,
)
from quant_fund.research.benches_w1404 import (
    bench_activitynet_qa_studies_family,
    bench_how2qa_lite_studies_family,
    bench_movie_qa_lite_studies_family,
    bench_msrvtt_qa_studies_family,
    bench_nextqa_lite_studies_family,
    bench_star_qa_lite_studies_family,
)
from quant_fund.research.benches_w1405 import (
    bench_ambi_qa_studies_family,
    bench_audio_qa_lite_studies_family,
    bench_avsd_lite_studies_family,
    bench_clotho_qa_studies_family,
    bench_esc_qa_studies_family,
    bench_music_avqa_studies_family,
)
from quant_fund.research.benches_w1406 import (
    bench_menat_qa_studies_family,
    bench_syndq_lite_studies_family,
    bench_teas_qa_studies_family,
    bench_time_qa_studies_family,
    bench_timedial_qa_studies_family,
    bench_timetravel_lite_studies_family,
)
from quant_fund.research.benches_w1407 import (
    bench_canard_lite_studies_family,
    bench_clarq_lite_studies_family,
    bench_doqa_lite_studies_family,
    bench_duread_qa_studies_family,
    bench_orchid_qa_studies_family,
    bench_qrecc_lite_studies_family,
)
from quant_fund.research.benches_w1408 import (
    bench_abduct_qa_studies_family,
    bench_analogy_qa_studies_family,
    bench_arct_lite_studies_family,
    bench_entailment_qa_studies_family,
    bench_fusion_qa_studies_family,
    bench_proof_qa_studies_family,
)
from quant_fund.research.benches_w1409 import (
    bench_bamboogle_lite_studies_family,
    bench_beerqa_lite_studies_family,
    bench_cider_qa_studies_family,
    bench_ensem_qa_studies_family,
    bench_fanqa_lite_studies_family,
    bench_hops_qa_studies_family,
)
from quant_fund.research.benches_w1410 import (
    bench_causal_qa_studies_family,
    bench_ecare_lite_studies_family,
    bench_event2mind_lite_studies_family,
    bench_event_qa_studies_family,
    bench_hippo_qa_studies_family,
    bench_intent_qa_studies_family,
)
from quant_fund.research.benches_w1411 import (
    bench_fakeqa_lite_studies_family,
    bench_flame_qa_studies_family,
    bench_hate_qa_studies_family,
    bench_ironic_qa_studies_family,
    bench_offensive_qa_studies_family,
    bench_politeness_qa_studies_family,
)
from quant_fund.research.benches_w1412 import (
    bench_affect_qa_studies_family,
    bench_anger_qa_studies_family,
    bench_comfort_qa_studies_family,
    bench_distress_qa_studies_family,
    bench_emotion_qa_studies_family,
    bench_empathy_qa_studies_family,
)
from quant_fund.research.benches_w1413 import (
    bench_anaphora_qa_studies_family,
    bench_coherence_qa_studies_family,
    bench_dialogue_act_studies_family,
    bench_discourse_qa_studies_family,
    bench_hedge_qa_studies_family,
    bench_implicit_qa_studies_family,
)
from quant_fund.research.benches_w1414 import (
    bench_afford_qa_studies_family,
    bench_counter_qa_studies_family,
    bench_custom_qa_studies_family,
    bench_everyday_qa_studies_family,
    bench_folk_qa_studies_family,
    bench_moral_qa_studies_family,
)
from quant_fund.research.benches_w1415 import (
    bench_case_qa_studies_family,
    bench_clause_qa_studies_family,
    bench_contract_qa_studies_family,
    bench_lawqa_lite_studies_family,
    bench_legal_qa_studies_family,
    bench_statute_qa_studies_family,
)
from quant_fund.research.benches_w1416 import (
    bench_analyst_qa_studies_family,
    bench_audit_qa_studies_family,
    bench_bank_qa_studies_family,
    bench_broker_qa_studies_family,
    bench_credit_qa_studies_family,
    bench_earnings_qa_studies_family,
)
from quant_fund.research.benches_w1417 import (
    bench_checklist_qa_studies_family,
    bench_flow_qa_studies_family,
    bench_guide_qa_studies_family,
    bench_howto_qa_studies_family,
    bench_instruct_qa_studies_family,
    bench_lesson_qa_studies_family,
)
from quant_fund.research.benches_w1418 import (
    bench_almanac_qa_studies_family,
    bench_atlas_qa_studies_family,
    bench_idiom_qa_studies_family,
    bench_jeopardy_qa_studies_family,
    bench_misc_qa_studies_family,
    bench_myth_qa_studies_family,
)
from quant_fund.research.benches_w1419 import (
    bench_anecdote_qa_studies_family,
    bench_ballad_qa_studies_family,
    bench_biography_qa_studies_family,
    bench_chronicle_qa_studies_family,
    bench_epic_qa_studies_family,
    bench_fable_qa_studies_family,
)
from quant_fund.research.benches_w1420 import (
    bench_cause_qa_studies_family,
    bench_claim_qa_studies_family,
    bench_conclusion_qa_studies_family,
    bench_deduction_qa_studies_family,
    bench_effect_qa_studies_family,
    bench_fallacy_qa_studies_family,
)
from quant_fund.research.benches_w1421 import (
    bench_geospatial_qa_studies_family,
    bench_itinerary_qa_studies_family,
    bench_journey_qa_studies_family,
    bench_route_qa_studies_family,
    bench_spatial_qa_studies_family,
    bench_terrain_qa_studies_family,
)
from quant_fund.research.benches_w1422 import (
    bench_calendar_qa_studies_family,
    bench_century_qa_studies_family,
    bench_date_qa_studies_family,
    bench_decade_qa_studies_family,
    bench_epoch_qa_studies_family,
    bench_era_qa_studies_family,
)
from quant_fund.research.benches_w1423 import (
    bench_class_qa_studies_family,
    bench_course_qa_studies_family,
    bench_exam_qa_studies_family,
    bench_homework_qa_studies_family,
    bench_lecture_qa_studies_family,
    bench_seminar_qa_studies_family,
)
from quant_fund.research.benches_w1424 import (
    bench_blueprint_qa_studies_family,
    bench_design_qa_studies_family,
    bench_format_qa_studies_family,
    bench_layout_qa_studies_family,
    bench_pattern_qa_studies_family,
    bench_schema_qa_studies_family,
)
from quant_fund.research.benches_w1425 import (
    bench_article_qa_studies_family,
    bench_broadcast_qa_studies_family,
    bench_column_qa_studies_family,
    bench_debate_qa_studies_family,
    bench_editorial_qa_studies_family,
    bench_headline_qa_studies_family,
)
from quant_fund.research.benches_w1426 import (
    bench_challenge_qa_studies_family,
    bench_contest_qa_studies_family,
    bench_game_qa_studies_family,
    bench_hobby_qa_studies_family,
    bench_leisure_qa_studies_family,
    bench_match_qa_studies_family,
)
from quant_fund.research.benches_w1427 import (
    bench_canyon_qa_studies_family,
    bench_coast_qa_studies_family,
    bench_desert_qa_studies_family,
    bench_field_qa_studies_family,
    bench_forest_qa_studies_family,
    bench_glacier_qa_studies_family,
)
from quant_fund.research.benches_w1428 import (
    bench_agency_qa_studies_family,
    bench_bureau_qa_studies_family,
    bench_cabinet_qa_studies_family,
    bench_election_qa_studies_family,
    bench_government_qa_studies_family,
    bench_ministry_qa_studies_family,
)
from quant_fund.research.benches_w1429 import (
    bench_atom_qa_studies_family,
    bench_electron_qa_studies_family,
    bench_ion_qa_studies_family,
    bench_molecule_qa_studies_family,
    bench_neutron_qa_studies_family,
    bench_photon_qa_studies_family,
)
from quant_fund.research.benches_w1430 import (
    bench_animal_qa_studies_family,
    bench_bird_qa_studies_family,
    bench_ecosystem_qa_studies_family,
    bench_fish_qa_studies_family,
    bench_habitat_qa_studies_family,
    bench_insect_qa_studies_family,
)
from quant_fund.research.benches_w1431 import (
    bench_aircraft_qa_studies_family,
    bench_bike_qa_studies_family,
    bench_bus_qa_studies_family,
    bench_car_qa_studies_family,
    bench_engine_qa_studies_family,
    bench_plane_qa_studies_family,
)
from quant_fund.research.benches_w1432 import (
    bench_beverage_qa_studies_family,
    bench_cuisine_qa_studies_family,
    bench_dessert_qa_studies_family,
    bench_dish_qa_studies_family,
    bench_fruit_qa_studies_family,
    bench_ingredient_qa_studies_family,
)
from quant_fund.research.benches_w1433 import (
    bench_cloud_qa_studies_family,
    bench_frost_qa_studies_family,
    bench_hurricane_qa_studies_family,
    bench_rain_qa_studies_family,
    bench_storm_qa_studies_family,
    bench_wind_qa_studies_family,
)
from quant_fund.research.benches_w1434 import (
    bench_alloy_qa_studies_family,
    bench_ceramic_qa_studies_family,
    bench_glass_qa_studies_family,
    bench_iron_qa_studies_family,
    bench_steel_qa_studies_family,
    bench_wood_qa_studies_family,
)
from quant_fund.research.benches_w1435 import (
    bench_blood_qa_studies_family,
    bench_bone_qa_studies_family,
    bench_brain_qa_studies_family,
    bench_heart_qa_studies_family,
    bench_muscle_qa_studies_family,
    bench_nerve_qa_studies_family,
)
from quant_fund.research.benches_w1436 import (
    bench_comet_qa_studies_family,
    bench_galaxy_qa_studies_family,
    bench_moon_qa_studies_family,
    bench_nebula_qa_studies_family,
    bench_planet_qa_studies_family,
    bench_star_qa_studies_family,
)
from quant_fund.research.benches_w1437 import (
    bench_deity_qa_studies_family,
    bench_dragon_qa_studies_family,
    bench_hero_qa_studies_family,
    bench_olympus_qa_studies_family,
    bench_phoenix_qa_studies_family,
    bench_titan_qa_studies_family,
)
from quant_fund.research.benches_w1438 import (
    bench_coral_qa_studies_family,
    bench_dolphin_qa_studies_family,
    bench_reef_qa_studies_family,
    bench_shark_qa_studies_family,
    bench_turtle_qa_studies_family,
    bench_whale_qa_studies_family,
)
from quant_fund.research.benches_w1439 import (
    bench_cliff_qa_studies_family,
    bench_crater_qa_studies_family,
    bench_dune_qa_studies_family,
    bench_fjord_qa_studies_family,
    bench_gorge_qa_studies_family,
    bench_mesa_qa_studies_family,
)
from quant_fund.research.benches_w1440 import (
    bench_bamboo_qa_studies_family,
    bench_cactus_qa_studies_family,
    bench_fern_qa_studies_family,
    bench_moss_qa_studies_family,
    bench_pine_qa_studies_family,
    bench_vine_qa_studies_family,
)
from quant_fund.research.benches_w1441 import (
    bench_crane_qa_studies_family,
    bench_eagle_qa_studies_family,
    bench_falcon_qa_studies_family,
    bench_owl_qa_studies_family,
    bench_raven_qa_studies_family,
    bench_swan_qa_studies_family,
)
from quant_fund.research.benches_w1442 import (
    bench_ant_qa_studies_family,
    bench_bee_qa_studies_family,
    bench_beetle_qa_studies_family,
    bench_butterfly_qa_studies_family,
    bench_cricket_qa_studies_family,
    bench_moth_qa_studies_family,
)
from quant_fund.research.benches_w1443 import (
    bench_amber_qa_studies_family,
    bench_amethyst_qa_studies_family,
    bench_crystal_qa_studies_family,
    bench_diamond_qa_studies_family,
    bench_emerald_qa_studies_family,
    bench_jade_qa_studies_family,
)
from quant_fund.research.benches_w1444 import (
    bench_brook_qa_studies_family,
    bench_creek_qa_studies_family,
    bench_delta_qa_studies_family,
    bench_estuary_qa_studies_family,
    bench_marsh_qa_studies_family,
    bench_pond_qa_studies_family,
)
from quant_fund.research.benches_w1445 import (
    bench_birch_qa_studies_family,
    bench_cedar_qa_studies_family,
    bench_elm_qa_studies_family,
    bench_maple_qa_studies_family,
    bench_oak_qa_studies_family,
    bench_willow_qa_studies_family,
)
from quant_fund.research.benches_w1446 import (
    bench_cello_qa_studies_family,
    bench_drum_qa_studies_family,
    bench_flute_qa_studies_family,
    bench_guitar_qa_studies_family,
    bench_piano_qa_studies_family,
    bench_violin_qa_studies_family,
)
from quant_fund.research.benches_w1447 import (
    bench_bear_qa_studies_family,
    bench_cheetah_qa_studies_family,
    bench_fox_qa_studies_family,
    bench_leopard_qa_studies_family,
    bench_lion_qa_studies_family,
    bench_wolf_qa_studies_family,
)
from quant_fund.research.benches_w1448 import (
    bench_cobra_qa_studies_family,
    bench_frog_qa_studies_family,
    bench_gecko_qa_studies_family,
    bench_iguana_qa_studies_family,
    bench_python_qa_studies_family,
    bench_viper_qa_studies_family,
)
from quant_fund.research.benches_w1449 import (
    bench_beluga_qa_studies_family,
    bench_manatee_qa_studies_family,
    bench_narwhal_qa_studies_family,
    bench_orca_qa_studies_family,
    bench_otter_qa_studies_family,
    bench_walrus_qa_studies_family,
)
from quant_fund.research.benches_w1450 import (
    bench_barn_qa_studies_family,
    bench_cow_qa_studies_family,
    bench_goat_qa_studies_family,
    bench_horse_qa_studies_family,
    bench_pig_qa_studies_family,
    bench_sheep_qa_studies_family,
)
from quant_fund.research.benches_w1451 import (
    bench_apple_qa_studies_family,
    bench_cherry_qa_studies_family,
    bench_grape_qa_studies_family,
    bench_lemon_qa_studies_family,
    bench_mango_qa_studies_family,
    bench_peach_qa_studies_family,
)
from quant_fund.research.benches_w1452 import (
    bench_carrot_qa_studies_family,
    bench_cucumber_qa_studies_family,
    bench_garlic_qa_studies_family,
    bench_onion_qa_studies_family,
    bench_potato_qa_studies_family,
    bench_tomato_qa_studies_family,
)
from quant_fund.research.benches_w1453 import (
    bench_condor_qa_studies_family,
    bench_harrier_qa_studies_family,
    bench_kestrel_qa_studies_family,
    bench_kite_qa_studies_family,
    bench_osprey_qa_studies_family,
    bench_vulture_qa_studies_family,
)
from quant_fund.research.benches_w1454 import (
    bench_aphid_qa_studies_family,
    bench_hornet_qa_studies_family,
    bench_locust_qa_studies_family,
    bench_mosquito_qa_studies_family,
    bench_scarab_qa_studies_family,
    bench_termite_qa_studies_family,
)
from quant_fund.research.benches_w1455 import (
    bench_axolotl_qa_studies_family,
    bench_bullfrog_qa_studies_family,
    bench_newt_qa_studies_family,
    bench_salamander_qa_studies_family,
    bench_toad_qa_studies_family,
    bench_tree_frog_qa_studies_family,
)
from quant_fund.research.benches_w1456 import (
    bench_barracuda_qa_studies_family,
    bench_catfish_qa_studies_family,
    bench_cod_qa_studies_family,
    bench_piranha_qa_studies_family,
    bench_salmon_qa_studies_family,
    bench_tuna_qa_studies_family,
)
from quant_fund.research.benches_w1457 import (
    bench_badger_qa_studies_family,
    bench_beaver_qa_studies_family,
    bench_bison_qa_studies_family,
    bench_cougar_qa_studies_family,
    bench_elk_qa_studies_family,
    bench_lynx_qa_studies_family,
)
from quant_fund.research.benches_w1458 import (
    bench_arroyo_qa_studies_family,
    bench_butte_qa_studies_family,
    bench_camel_qa_studies_family,
    bench_caravan_qa_studies_family,
    bench_mirage_qa_studies_family,
    bench_oasis_qa_studies_family,
)
from quant_fund.research.benches_w1459 import (
    bench_arctic_fox_qa_studies_family,
    bench_caribou_qa_studies_family,
    bench_musk_ox_qa_studies_family,
    bench_penguin_qa_studies_family,
    bench_polar_bear_qa_studies_family,
    bench_reindeer_qa_studies_family,
)
from quant_fund.research.benches_w1460 import (
    bench_baboon_qa_studies_family,
    bench_elephant_qa_studies_family,
    bench_gazelle_qa_studies_family,
    bench_giraffe_qa_studies_family,
    bench_wildebeest_qa_studies_family,
    bench_zebra_qa_studies_family,
)
from quant_fund.research.benches_w1461 import (
    bench_gorilla_qa_studies_family,
    bench_jaguar_qa_studies_family,
    bench_macaw_qa_studies_family,
    bench_orangutan_qa_studies_family,
    bench_sloth_qa_studies_family,
    bench_toucan_qa_studies_family,
)
from quant_fund.research.benches_w1462 import (
    bench_crab_qa_studies_family,
    bench_jellyfish_qa_studies_family,
    bench_octopus_qa_studies_family,
    bench_seahorse_qa_studies_family,
    bench_squid_qa_studies_family,
    bench_stingray_qa_studies_family,
)
from quant_fund.research.benches_w1463 import (
    bench_acorn_qa_studies_family,
    bench_blossom_qa_studies_family,
    bench_canopy_qa_studies_family,
    bench_firefly_qa_studies_family,
    bench_sprout_qa_studies_family,
    bench_truffle_qa_studies_family,
)
from quant_fund.research.benches_w1464 import (
    bench_abyss_qa_studies_family,
    bench_beacon_qa_studies_family,
    bench_blizzard_qa_studies_family,
    bench_monolith_qa_studies_family,
    bench_spire_qa_studies_family,
    bench_tempest_qa_studies_family,
)
from quant_fund.research.benches_w1465 import (
    bench_citadel_qa_studies_family,
    bench_forge_qa_studies_family,
    bench_grotto_qa_studies_family,
    bench_lighthouse_qa_studies_family,
    bench_quarry_qa_studies_family,
    bench_vault_qa_studies_family,
)
from quant_fund.research.benches_w1466 import (
    bench_basalt_qa_studies_family,
    bench_cathedral_qa_studies_family,
    bench_chasm_qa_studies_family,
    bench_crag_qa_studies_family,
    bench_plateau_qa_studies_family,
    bench_ravine_qa_studies_family,
)
from quant_fund.research.benches_w1467 import (
    bench_arch_qa_studies_family,
    bench_steppe_qa_studies_family,
    bench_summit_qa_studies_family,
    bench_tundra_qa_studies_family,
    bench_valley_qa_studies_family,
    bench_volcano_qa_studies_family,
)
from quant_fund.research.benches_w1468 import (
    bench_dale_qa_studies_family,
    bench_fen_qa_studies_family,
    bench_glen_qa_studies_family,
    bench_heath_qa_studies_family,
    bench_knoll_qa_studies_family,
    bench_moor_qa_studies_family,
)
from quant_fund.research.benches_w1469 import (
    bench_atoll_qa_studies_family,
    bench_bluff_qa_studies_family,
    bench_cove_qa_studies_family,
    bench_headland_qa_studies_family,
    bench_inlet_qa_studies_family,
    bench_islet_qa_studies_family,
)
from quant_fund.research.benches_w1470 import (
    bench_aspen_qa_studies_family,
    bench_fir_qa_studies_family,
    bench_holly_qa_studies_family,
    bench_juniper_qa_studies_family,
    bench_redwood_qa_studies_family,
    bench_sequoia_qa_studies_family,
)
from quant_fund.research.benches_w1471 import (
    bench_crocus_qa_studies_family,
    bench_daffodil_qa_studies_family,
    bench_daisy_qa_studies_family,
    bench_foxglove_qa_studies_family,
    bench_iris_qa_studies_family,
    bench_poppy_qa_studies_family,
)
from quant_fund.research.benches_w1472 import (
    bench_basil_qa_studies_family,
    bench_cardamom_qa_studies_family,
    bench_chervil_qa_studies_family,
    bench_cinnamon_qa_studies_family,
    bench_coriander_qa_studies_family,
    bench_cumin_qa_studies_family,
)
from quant_fund.research.benches_w1473 import (
    bench_clove_qa_studies_family,
    bench_dill_qa_studies_family,
    bench_fennel_qa_studies_family,
    bench_lemongrass_qa_studies_family,
    bench_mint_qa_studies_family,
    bench_nutmeg_qa_studies_family,
)
from quant_fund.research.benches_w1474 import (
    bench_bittern_qa_studies_family,
    bench_cormorant_qa_studies_family,
    bench_curlew_qa_studies_family,
    bench_ibis_qa_studies_family,
    bench_kingfisher_qa_studies_family,
    bench_loon_qa_studies_family,
)
from quant_fund.research.benches_w1475 import (
    bench_cicada_qa_studies_family,
    bench_dragonfly_qa_studies_family,
    bench_grasshopper_qa_studies_family,
    bench_ladybug_qa_studies_family,
    bench_mantis_qa_studies_family,
    bench_scorpion_qa_studies_family,
)
from quant_fund.research.benches_w1476 import (
    bench_caracal_qa_studies_family,
    bench_jaguarundi_qa_studies_family,
    bench_margay_qa_studies_family,
    bench_ocelot_qa_studies_family,
    bench_puma_qa_studies_family,
    bench_serval_qa_studies_family,
)
from quant_fund.research.benches_w1477 import (
    bench_albatross_qa_studies_family,
    bench_gannet_qa_studies_family,
    bench_petrel_qa_studies_family,
    bench_puffin_qa_studies_family,
    bench_shearwater_qa_studies_family,
    bench_skua_qa_studies_family,
)
from quant_fund.research.benches_w1478 import (
    bench_antelope_qa_studies_family,
    bench_eland_qa_studies_family,
    bench_impala_qa_studies_family,
    bench_kudu_qa_studies_family,
    bench_oryx_qa_studies_family,
    bench_springbok_qa_studies_family,
)
from quant_fund.research.benches_w1479 import (
    bench_agouti_qa_studies_family,
    bench_armadillo_qa_studies_family,
    bench_capybara_qa_studies_family,
    bench_coati_qa_studies_family,
    bench_peccary_qa_studies_family,
    bench_tapir_qa_studies_family,
)
from quant_fund.research.benches_w1480 import (
    bench_adder_qa_studies_family,
    bench_boa_qa_studies_family,
    bench_krait_qa_studies_family,
    bench_mamba_qa_studies_family,
    bench_monitor_qa_studies_family,
    bench_taipan_qa_studies_family,
)
from quant_fund.research.benches_w1481 import (
    bench_earwig_qa_studies_family,
    bench_katydid_qa_studies_family,
    bench_mayfly_qa_studies_family,
    bench_stonefly_qa_studies_family,
    bench_wasp_qa_studies_family,
    bench_weevil_qa_studies_family,
)
from quant_fund.research.benches_w1482 import (
    bench_ermine_qa_studies_family,
    bench_fisher_qa_studies_family,
    bench_marten_qa_studies_family,
    bench_mink_qa_studies_family,
    bench_polecat_qa_studies_family,
    bench_wolverine_qa_studies_family,
)
from quant_fund.research.benches_w1483 import (
    bench_coyote_qa_studies_family,
    bench_ferret_qa_studies_family,
    bench_jackal_qa_studies_family,
    bench_marmot_qa_studies_family,
    bench_moose_qa_studies_family,
    bench_raccoon_qa_studies_family,
)
from quant_fund.research.benches_w1484 import (
    bench_bandicoot_qa_studies_family,
    bench_koala_qa_studies_family,
    bench_numbat_qa_studies_family,
    bench_quokka_qa_studies_family,
    bench_wallaby_qa_studies_family,
    bench_wombat_qa_studies_family,
)
from quant_fund.research.benches_w1485 import (
    bench_avocet_qa_studies_family,
    bench_egret_qa_studies_family,
    bench_heron_qa_studies_family,
    bench_plover_qa_studies_family,
    bench_sandpiper_qa_studies_family,
    bench_tern_qa_studies_family,
)
from quant_fund.research.benches_w1486 import (
    bench_flamingo_qa_studies_family,
    bench_godwit_qa_studies_family,
    bench_grebe_qa_studies_family,
    bench_pelican_qa_studies_family,
    bench_spoonbill_qa_studies_family,
    bench_stork_qa_studies_family,
)
from quant_fund.research.benches_w1487 import (
    bench_auk_qa_studies_family,
    bench_fulmar_qa_studies_family,
    bench_gull_qa_studies_family,
    bench_jaeger_qa_studies_family,
    bench_kittiwake_qa_studies_family,
    bench_tropicbird_qa_studies_family,
)
from quant_fund.research.benches_w1488 import (
    bench_bobcat_qa_studies_family,
    bench_dingo_qa_studies_family,
    bench_kodkod_qa_studies_family,
    bench_oncilla_qa_studies_family,
    bench_panther_qa_studies_family,
    bench_tiger_qa_studies_family,
)
from quant_fund.research.benches_w1489 import (
    bench_clover_qa_studies_family,
    bench_heather_qa_studies_family,
    bench_lavender_qa_studies_family,
    bench_lilac_qa_studies_family,
    bench_marigold_qa_studies_family,
    bench_primrose_qa_studies_family,
)
from quant_fund.research.benches_w1490 import (
    bench_camellia_qa_studies_family,
    bench_dahlia_qa_studies_family,
    bench_sage_qa_studies_family,
    bench_thyme_qa_studies_family,
    bench_violet_qa_studies_family,
    bench_zinnia_qa_studies_family,
)
from quant_fund.research.benches_w1491 import (
    bench_cypress_qa_studies_family,
    bench_eucalyptus_qa_studies_family,
    bench_hemlock_qa_studies_family,
    bench_laurel_qa_studies_family,
    bench_magnolia_qa_studies_family,
    bench_spruce_qa_studies_family,
)
from quant_fund.research.benches_w1492 import (
    bench_bilby_qa_studies_family,
    bench_echidna_qa_studies_family,
    bench_platypus_qa_studies_family,
    bench_possum_qa_studies_family,
    bench_quoll_qa_studies_family,
    bench_thylacine_qa_studies_family,
)
from quant_fund.research.benches_w1493 import (
    bench_bongo_qa_studies_family,
    bench_duiker_qa_studies_family,
    bench_hartebeest_qa_studies_family,
    bench_nyala_qa_studies_family,
    bench_topi_qa_studies_family,
    bench_waterbuck_qa_studies_family,
)
from quant_fund.research.benches_w1494 import (
    bench_anole_qa_studies_family,
    bench_chameleon_qa_studies_family,
    bench_hognose_qa_studies_family,
    bench_skink_qa_studies_family,
    bench_terrapin_qa_studies_family,
    bench_tuatara_qa_studies_family,
)
from quant_fund.research.benches_w1495 import (
    bench_grison_qa_studies_family,
    bench_sable_qa_studies_family,
    bench_stoat_qa_studies_family,
    bench_tayra_qa_studies_family,
    bench_weasel_qa_studies_family,
    bench_zorilla_qa_studies_family,
)
from quant_fund.research.benches_w1496 import (
    bench_jacana_qa_studies_family,
    bench_lapwing_qa_studies_family,
    bench_moorhen_qa_studies_family,
    bench_railbird_qa_studies_family,
    bench_snipe_qa_studies_family,
    bench_turnstone_qa_studies_family,
)
from quant_fund.research.benches_w1497 import (
    bench_bettong_qa_studies_family,
    bench_cuscus_qa_studies_family,
    bench_numbat2_qa_studies_family,
    bench_pademelon_qa_studies_family,
    bench_potoroo_qa_studies_family,
    bench_woylie_qa_studies_family,
)
from quant_fund.research.benches_w1498 import (
    bench_auklet_qa_studies_family,
    bench_booby_qa_studies_family,
    bench_frigatebird_qa_studies_family,
    bench_guillemot_qa_studies_family,
    bench_murrelet_qa_studies_family,
    bench_razorbill_qa_studies_family,
)
from quant_fund.research.benches_w1499 import (
    bench_oregano_qa_studies_family,
    bench_parsley_qa_studies_family,
    bench_rosemary_qa_studies_family,
    bench_saffron_qa_studies_family,
    bench_tarragon_qa_studies_family,
    bench_turmeric_qa_studies_family,
)
from quant_fund.research.benches_w1500 import (
    bench_caddisfly_qa_studies_family,
    bench_centipede_qa_studies_family,
    bench_horntail_qa_studies_family,
    bench_lacewing_qa_studies_family,
    bench_millipede_qa_studies_family,
    bench_spider_qa_studies_family,
)
from quant_fund.research.benches_w1501 import (
    bench_acacia_qa_studies_family,
    bench_alder_qa_studies_family,
    bench_baobab_qa_studies_family,
    bench_olive_qa_studies_family,
    bench_palm_qa_studies_family,
    bench_sycamore_qa_studies_family,
)
from quant_fund.research.benches_w1502 import (
    bench_anteater_qa_studies_family,
    bench_coatimundi_qa_studies_family,
    bench_kinkajou_qa_studies_family,
    bench_opossum_qa_studies_family,
    bench_paca_qa_studies_family,
    bench_tamandua_qa_studies_family,
)
from quant_fund.research.benches_w1503 import (
    bench_garter_qa_studies_family,
    bench_keelback_qa_studies_family,
    bench_kingsnake_qa_studies_family,
    bench_mockviper_qa_studies_family,
    bench_racer_qa_studies_family,
    bench_sidewinder_qa_studies_family,
)
from quant_fund.research.benches_w1504 import (
    bench_murre_qa_studies_family,
    bench_noddie_qa_studies_family,
    bench_prion_qa_studies_family,
    bench_shag_qa_studies_family,
    bench_skimmer_qa_studies_family,
    bench_storm_petrel_qa_studies_family,
)
from quant_fund.research.benches_w1505 import (
    bench_gadwall_qa_studies_family,
    bench_pintail_qa_studies_family,
    bench_pochard_qa_studies_family,
    bench_shoveler_qa_studies_family,
    bench_teal_qa_studies_family,
    bench_wigeon_qa_studies_family,
)
from quant_fund.research.benches_w1506 import (
    bench_bufflehead_qa_studies_family,
    bench_canvasback_qa_studies_family,
    bench_eider_qa_studies_family,
    bench_mallard_qa_studies_family,
    bench_merganser_qa_studies_family,
    bench_scoter_qa_studies_family,
)
from quant_fund.research.benches_w1507 import (
    bench_buzzard_qa_studies_family,
    bench_caracara_qa_studies_family,
    bench_goshawk_qa_studies_family,
    bench_merlin_qa_studies_family,
    bench_peregrine_qa_studies_family,
    bench_sparrowhawk_qa_studies_family,
)
from quant_fund.research.benches_w1508 import (
    bench_chickadee_qa_studies_family,
    bench_finch_qa_studies_family,
    bench_sparrow_qa_studies_family,
    bench_thrush_qa_studies_family,
    bench_warbler_qa_studies_family,
    bench_wren_qa_studies_family,
)
from quant_fund.research.benches_w1509 import (
    bench_bunting_qa_studies_family,
    bench_grosbeak_qa_studies_family,
    bench_nuthatch_qa_studies_family,
    bench_tanager_qa_studies_family,
    bench_titmouse_qa_studies_family,
    bench_vireo_qa_studies_family,
)
from quant_fund.research.benches_w1510 import (
    bench_chough_qa_studies_family,
    bench_crow_qa_studies_family,
    bench_jackdaw_qa_studies_family,
    bench_jay_qa_studies_family,
    bench_magpie_qa_studies_family,
    bench_rook_qa_studies_family,
)
from quant_fund.research.benches_w1511 import (
    bench_brilliant_qa_studies_family,
    bench_hermit_qa_studies_family,
    bench_hummingbird_qa_studies_family,
    bench_sapphire_qa_studies_family,
    bench_topaz_qa_studies_family,
    bench_woodstar_qa_studies_family,
)
from quant_fund.research.benches_w1512 import (
    bench_dunlin_qa_studies_family,
    bench_knot_qa_studies_family,
    bench_oystercatcher_qa_studies_family,
    bench_phalarope_qa_studies_family,
    bench_stilt_qa_studies_family,
    bench_whimbrel_qa_studies_family,
)
from quant_fund.research.benches_w1513 import (
    bench_barnowl_qa_studies_family,
    bench_barred_owl_qa_studies_family,
    bench_eagle_owl_qa_studies_family,
    bench_screech_owl_qa_studies_family,
    bench_snowy_owl_qa_studies_family,
    bench_tawny_owl_qa_studies_family,
)
from quant_fund.research.benches_w1514 import (
    bench_blue_morpho_qa_studies_family,
    bench_cabbage_white_qa_studies_family,
    bench_fritillary_qa_studies_family,
    bench_monarch_qa_studies_family,
    bench_painted_lady_qa_studies_family,
    bench_swallowtail_qa_studies_family,
)
from quant_fund.research.benches_w1515 import (
    bench_click_beetle_qa_studies_family,
    bench_dung_beetle_qa_studies_family,
    bench_ground_beetle_qa_studies_family,
    bench_rhino_beetle_qa_studies_family,
    bench_stag_beetle_qa_studies_family,
    bench_tiger_beetle_qa_studies_family,
)
from quant_fund.research.benches_w1516 import (
    bench_atlas_moth_qa_studies_family,
    bench_gypsy_moth_qa_studies_family,
    bench_hawk_moth_qa_studies_family,
    bench_luna_moth_qa_studies_family,
    bench_tussock_moth_qa_studies_family,
    bench_underwing_qa_studies_family,
)
from quant_fund.research.benches_w1517 import (
    bench_clubtail_qa_studies_family,
    bench_damselfly_qa_studies_family,
    bench_darner_qa_studies_family,
    bench_forktail_qa_studies_family,
    bench_hawker_qa_studies_family,
    bench_spreadwing_qa_studies_family,
)
from quant_fund.research.benches_w1518 import (
    bench_coquette_qa_studies_family,
    bench_fairy_qa_studies_family,
    bench_jacobin_qa_studies_family,
    bench_lancebill_qa_studies_family,
    bench_sabrewing_qa_studies_family,
    bench_sheartail_qa_studies_family,
)
from quant_fund.research.benches_w1519 import (
    bench_empusa_qa_studies_family,
    bench_ghost_mantis_qa_studies_family,
    bench_mantidfly_qa_studies_family,
    bench_orchid_mantis_qa_studies_family,
    bench_praying_mantis_qa_studies_family,
    bench_shield_mantis_qa_studies_family,
)
from quant_fund.research.benches_w1520 import (
    bench_agaric_qa_studies_family,
    bench_bolete_qa_studies_family,
    bench_chanterelle_qa_studies_family,
    bench_inkcap_qa_studies_family,
    bench_morel_qa_studies_family,
    bench_puffball_qa_studies_family,
)
from quant_fund.research.benches_w1521 import (
    bench_cattleya_qa_studies_family,
    bench_cymbidium_qa_studies_family,
    bench_dendrobium_qa_studies_family,
    bench_oncidium_qa_studies_family,
    bench_paphiopedilum_qa_studies_family,
    bench_phalaenopsis_qa_studies_family,
)
from quant_fund.research.benches_w1522 import (
    bench_agave_qa_studies_family,
    bench_aloe_qa_studies_family,
    bench_echeveria_qa_studies_family,
    bench_haworthia_qa_studies_family,
    bench_lithops_qa_studies_family,
    bench_sedum_qa_studies_family,
)
from quant_fund.research.benches_w1523 import (
    bench_bluegrass_qa_studies_family,
    bench_fescue_qa_studies_family,
    bench_miscanthus_qa_studies_family,
    bench_pampas_qa_studies_family,
    bench_ryegrass_qa_studies_family,
    bench_switchgrass_qa_studies_family,
)
from quant_fund.research.benches_w1524 import (
    bench_bulrush_qa_studies_family,
    bench_carex_qa_studies_family,
    bench_cattail_qa_studies_family,
    bench_cottongrass_qa_studies_family,
    bench_reed_qa_studies_family,
    bench_rush_qa_studies_family,
)
from quant_fund.research.benches_w1525 import (
    bench_clubmoss_qa_studies_family,
    bench_haircap_qa_studies_family,
    bench_hornwort_qa_studies_family,
    bench_liverwort_qa_studies_family,
    bench_quillwort_qa_studies_family,
    bench_sphagnum_qa_studies_family,
)
from quant_fund.research.benches_w1526 import (
    bench_bracken_qa_studies_family,
    bench_horsetail_qa_studies_family,
    bench_maidenhair_qa_studies_family,
    bench_staghorn_qa_studies_family,
    bench_swordfern_qa_studies_family,
    bench_treefern_qa_studies_family,
)
from quant_fund.research.benches_w1527 import (
    bench_crustose_qa_studies_family,
    bench_foliose_qa_studies_family,
    bench_fruticose_qa_studies_family,
    bench_oakmoss_qa_studies_family,
    bench_usnea_qa_studies_family,
    bench_xanthoria_qa_studies_family,
)
from quant_fund.research.benches_w1528 import (
    bench_calcite_qa_studies_family,
    bench_feldspar_qa_studies_family,
    bench_fluorite_qa_studies_family,
    bench_gypsum_qa_studies_family,
    bench_olivine_qa_studies_family,
    bench_quartz_qa_studies_family,
)
from quant_fund.research.benches_w1529 import (
    bench_aquamarine_qa_studies_family,
    bench_garnet_qa_studies_family,
    bench_opal_qa_studies_family,
    bench_ruby_qa_studies_family,
    bench_tanzanite_qa_studies_family,
    bench_tourmaline_qa_studies_family,
)
from quant_fund.research.benches_w1530 import (
    bench_amalgam_qa_studies_family,
    bench_brass_qa_studies_family,
    bench_bronze_qa_studies_family,
    bench_nichrome_qa_studies_family,
    bench_pewter_qa_studies_family,
    bench_solder_qa_studies_family,
)
from quant_fund.research.benches_w1531 import (
    bench_downy_qa_studies_family,
    bench_flicker_qa_studies_family,
    bench_pileated_qa_studies_family,
    bench_sapsucker_qa_studies_family,
    bench_woodpecker_qa_studies_family,
    bench_wryneck_qa_studies_family,
)
from quant_fund.research.benches_w1532 import (
    bench_bee_eater_qa_studies_family,
    bench_jacamar_qa_studies_family,
    bench_kookaburra_qa_studies_family,
    bench_motmot_qa_studies_family,
    bench_roller_qa_studies_family,
    bench_tody_qa_studies_family,
)
from quant_fund.research.benches_w1533 import (
    bench_aracari_qa_studies_family,
    bench_barbet_qa_studies_family,
    bench_honeyguide_qa_studies_family,
    bench_hornbill_qa_studies_family,
    bench_quetzal_qa_studies_family,
    bench_trogon_qa_studies_family,
)
from quant_fund.research.benches_w1534 import (
    bench_cuckoo_qa_studies_family,
    bench_frogmouth_qa_studies_family,
    bench_koel_qa_studies_family,
    bench_nighthawk_qa_studies_family,
    bench_nightjar_qa_studies_family,
    bench_roadrunner_qa_studies_family,
)
from quant_fund.research.benches_w1535 import (
    bench_martin_qa_studies_family,
    bench_needletail_qa_studies_family,
    bench_swallow_qa_studies_family,
    bench_swift_qa_studies_family,
    bench_swiftlet_qa_studies_family,
    bench_treeswift_qa_studies_family,
)
from quant_fund.research.benches_w1536 import (
    bench_collared_dove_qa_studies_family,
    bench_dove_qa_studies_family,
    bench_mourning_dove_qa_studies_family,
    bench_pigeon_qa_studies_family,
    bench_turtle_dove_qa_studies_family,
    bench_woodpigeon_qa_studies_family,
)
from quant_fund.research.benches_w1537 import (
    bench_coot_qa_studies_family,
    bench_crake_qa_studies_family,
    bench_dabchick_qa_studies_family,
    bench_gallinule_qa_studies_family,
    bench_rail_qa_studies_family,
    bench_waterhen_qa_studies_family,
)
from quant_fund.research.benches_w1538 import (
    bench_crowned_crane_qa_studies_family,
    bench_demoiselle_qa_studies_family,
    bench_finfoot_qa_studies_family,
    bench_limpkin_qa_studies_family,
    bench_trumpeter_qa_studies_family,
    bench_whooping_qa_studies_family,
)
from quant_fund.research.benches_w1539 import (
    bench_goliath_heron_qa_studies_family,
    bench_green_heron_qa_studies_family,
    bench_grey_heron_qa_studies_family,
    bench_night_heron_qa_studies_family,
    bench_purple_heron_qa_studies_family,
    bench_tiger_heron_qa_studies_family,
)
from quant_fund.research.benches_w1540 import (
    bench_cattle_egret_qa_studies_family,
    bench_glossy_ibis_qa_studies_family,
    bench_great_egret_qa_studies_family,
    bench_sacred_ibis_qa_studies_family,
    bench_snowy_egret_qa_studies_family,
    bench_squacco_qa_studies_family,
)
from quant_fund.research.benches_w1541 import (
    bench_anhinga_qa_studies_family,
    bench_darter_qa_studies_family,
    bench_diving_petrel_qa_studies_family,
    bench_gadfly_qa_studies_family,
    bench_manx_qa_studies_family,
    bench_mollymawk_qa_studies_family,
)
from quant_fund.research.benches_w1542 import (
    bench_accipiter_qa_studies_family,
    bench_bateleur_qa_studies_family,
    bench_falconet_qa_studies_family,
    bench_harpy_qa_studies_family,
    bench_lammergeier_qa_studies_family,
    bench_seriema_qa_studies_family,
)
from quant_fund.research.benches_w1543 import (
    bench_oilbird_qa_studies_family,
    bench_owlet_nightjar_qa_studies_family,
    bench_pauraque_qa_studies_family,
    bench_poorwill_qa_studies_family,
    bench_potoo_qa_studies_family,
    bench_whip_poor_will_qa_studies_family,
)
from quant_fund.research.benches_w1544 import (
    bench_hoopoe_qa_studies_family,
    bench_nunbird_qa_studies_family,
    bench_nunlet_qa_studies_family,
    bench_puffbird_qa_studies_family,
    bench_toco_qa_studies_family,
    bench_woodhoopoe_qa_studies_family,
)
from quant_fund.research.benches_w1545 import (
    bench_amazon_qa_studies_family,
    bench_cockatoo_qa_studies_family,
    bench_conure_qa_studies_family,
    bench_kakapo_qa_studies_family,
    bench_kea_qa_studies_family,
    bench_lorikeet_qa_studies_family,
)
from quant_fund.research.benches_w1546 import (
    bench_corncrake_qa_studies_family,
    bench_flufftail_qa_studies_family,
    bench_sora_qa_studies_family,
    bench_sungrebe_qa_studies_family,
    bench_swamphen_qa_studies_family,
    bench_takhe_qa_studies_family,
)
from quant_fund.research.benches_w1547 import (
    bench_crowned_pigeon_qa_studies_family,
    bench_cuckoo_dove_qa_studies_family,
    bench_emerald_dove_qa_studies_family,
    bench_fruit_dove_qa_studies_family,
    bench_ground_dove_qa_studies_family,
    bench_quail_dove_qa_studies_family,
)
from quant_fund.research.benches_w1548 import (
    bench_ani_qa_studies_family,
    bench_coua_qa_studies_family,
    bench_guira_qa_studies_family,
    bench_hoatzin_qa_studies_family,
    bench_malkoha_qa_studies_family,
    bench_turaco_qa_studies_family,
)
from quant_fund.research.benches_w1549 import (
    bench_cassowary_qa_studies_family,
    bench_emu_qa_studies_family,
    bench_kiwi_qa_studies_family,
    bench_ostrich_qa_studies_family,
    bench_rhea_qa_studies_family,
    bench_tinamou_qa_studies_family,
)
from quant_fund.research.benches_w1550 import (
    bench_agama_qa_studies_family,
    bench_chuckwalla_qa_studies_family,
    bench_frilled_lizard_qa_studies_family,
    bench_monitor_lizard_qa_studies_family,
    bench_tegu_qa_studies_family,
    bench_uromastyx_qa_studies_family,
)
from quant_fund.research.benches_w1551 import (
    bench_bushmaster_qa_studies_family,
    bench_copperhead_qa_studies_family,
    bench_coral_snake_qa_studies_family,
    bench_cottonmouth_qa_studies_family,
    bench_fer_de_lance_qa_studies_family,
    bench_rattlesnake_qa_studies_family,
)
from quant_fund.research.benches_w1552 import (
    bench_box_turtle_qa_studies_family,
    bench_map_turtle_qa_studies_family,
    bench_painted_turtle_qa_studies_family,
    bench_slider_qa_studies_family,
    bench_snapping_turtle_qa_studies_family,
    bench_tortoise_qa_studies_family,
)
from quant_fund.research.benches_w1553 import (
    bench_dart_frog_qa_studies_family,
    bench_horned_frog_qa_studies_family,
    bench_leopard_frog_qa_studies_family,
    bench_spring_peeper_qa_studies_family,
    bench_treefrog_qa_studies_family,
    bench_wood_frog_qa_studies_family,
)
from quant_fund.research.benches_w1554 import (
    bench_black_widow_qa_studies_family,
    bench_huntsman_qa_studies_family,
    bench_jumping_spider_qa_studies_family,
    bench_orb_weaver_qa_studies_family,
    bench_tarantula_qa_studies_family,
    bench_wolf_spider_qa_studies_family,
)
from quant_fund.research.benches_w1555 import (
    bench_harvestman_qa_studies_family,
    bench_pseudoscorpion_qa_studies_family,
    bench_solifuge_qa_studies_family,
    bench_tick_qa_studies_family,
    bench_vinegaroon_qa_studies_family,
    bench_whip_scorpion_qa_studies_family,
)
from quant_fund.research.benches_w1556 import (
    bench_bristletail_qa_studies_family,
    bench_pillbug_qa_studies_family,
    bench_silverfish_qa_studies_family,
    bench_springtail_qa_studies_family,
    bench_velvet_worm_qa_studies_family,
    bench_woodlouse_qa_studies_family,
)
from quant_fund.research.benches_w1557 import (
    bench_arapaima_qa_studies_family,
    bench_electric_eel_qa_studies_family,
    bench_knifefish_qa_studies_family,
    bench_oscar_qa_studies_family,
    bench_pacu_qa_studies_family,
    bench_tetra_qa_studies_family,
)
from quant_fund.research.benches_w1558 import (
    bench_char_qa_studies_family,
    bench_dolly_varden_qa_studies_family,
    bench_grayling_qa_studies_family,
    bench_sockeye_qa_studies_family,
    bench_steelhead_qa_studies_family,
    bench_whitefish_qa_studies_family,
)
from quant_fund.research.benches_w1559 import (
    bench_anchovy_qa_studies_family,
    bench_bonito_qa_studies_family,
    bench_herring_qa_studies_family,
    bench_kingfish_qa_studies_family,
    bench_mackerel_qa_studies_family,
    bench_sardine_qa_studies_family,
)
from quant_fund.research.benches_w1560 import (
    bench_butterflyfish_qa_studies_family,
    bench_damselfish_qa_studies_family,
    bench_grouper_qa_studies_family,
    bench_parrotfish_qa_studies_family,
    bench_snapper_qa_studies_family,
    bench_wrasse_qa_studies_family,
)
from quant_fund.research.benches_w1561 import (
    bench_angelfish_qa_studies_family,
    bench_blenny_qa_studies_family,
    bench_goby_qa_studies_family,
    bench_lionfish_qa_studies_family,
    bench_surgeonfish_qa_studies_family,
    bench_triggerfish_qa_studies_family,
)
from quant_fund.research.benches_w1562 import (
    bench_boxfish_qa_studies_family,
    bench_clownfish_qa_studies_family,
    bench_dragonet_qa_studies_family,
    bench_mandarinfish_qa_studies_family,
    bench_pipefish_qa_studies_family,
    bench_pufferfish_qa_studies_family,
)
from quant_fund.research.benches_w1563 import (
    bench_cleaner_shrimp_qa_studies_family,
    bench_decorator_crab_qa_studies_family,
    bench_hermit_crab_qa_studies_family,
    bench_mantis_shrimp_qa_studies_family,
    bench_pistol_shrimp_qa_studies_family,
    bench_porcelain_crab_qa_studies_family,
)
from quant_fund.research.benches_w1564 import (
    bench_bobtail_squid_qa_studies_family,
    bench_cuttlefish_qa_studies_family,
    bench_nautilus_qa_studies_family,
    bench_nudibranch_qa_studies_family,
    bench_sea_slug_qa_studies_family,
    bench_vampire_squid_qa_studies_family,
)
from quant_fund.research.benches_w1565 import (
    bench_earthworm_qa_studies_family,
    bench_feather_duster_qa_studies_family,
    bench_leech_qa_studies_family,
    bench_lugworm_qa_studies_family,
    bench_polychaete_qa_studies_family,
    bench_ragworm_qa_studies_family,
)
from quant_fund.research.benches_w1566 import (
    bench_bluegill_qa_studies_family,
    bench_crappie_qa_studies_family,
    bench_perch_qa_studies_family,
    bench_pike_qa_studies_family,
    bench_sturgeon_qa_studies_family,
    bench_walleye_qa_studies_family,
)
from quant_fund.research.benches_w1567 import (
    bench_barbel_qa_studies_family,
    bench_bream_qa_studies_family,
    bench_carp_qa_studies_family,
    bench_minnow_qa_studies_family,
    bench_roach_qa_studies_family,
    bench_tench_qa_studies_family,
)
from quant_fund.research.benches_w1568 import (
    bench_conger_qa_studies_family,
    bench_garden_eel_qa_studies_family,
    bench_hagfish_qa_studies_family,
    bench_lamprey_qa_studies_family,
    bench_moray_qa_studies_family,
    bench_ribbon_eel_qa_studies_family,
)
from quant_fund.research.benches_w1569 import (
    bench_eagle_ray_qa_studies_family,
    bench_guitarfish_qa_studies_family,
    bench_manta_qa_studies_family,
    bench_sawfish_qa_studies_family,
    bench_thornback_qa_studies_family,
    bench_torpedo_ray_qa_studies_family,
)
from quant_fund.research.benches_w1570 import (
    bench_fiddler_crab_qa_studies_family,
    bench_ghost_crab_qa_studies_family,
    bench_horseshoe_qa_studies_family,
    bench_mud_crab_qa_studies_family,
    bench_porcelain_qa_studies_family,
    bench_spider_crab_qa_studies_family,
)
from quant_fund.research.benches_w1571 import (
    bench_clam_qa_studies_family,
    bench_conch_qa_studies_family,
    bench_mussel_qa_studies_family,
    bench_oyster_qa_studies_family,
    bench_scallop_qa_studies_family,
    bench_whelk_qa_studies_family,
)
from quant_fund.research.benches_w1572 import (
    bench_aster_qa_studies_family,
    bench_bluebell_qa_studies_family,
    bench_buttercup_qa_studies_family,
    bench_columbine_qa_studies_family,
    bench_cornflower_qa_studies_family,
    bench_lupine_qa_studies_family,
)
from quant_fund.research.benches_w1573 import (
    bench_addax_qa_studies_family,
    bench_fennec_qa_studies_family,
    bench_jerboa_qa_studies_family,
    bench_meerkat_qa_studies_family,
    bench_onager_qa_studies_family,
    bench_pangolin_qa_studies_family,
)
from quant_fund.research.benches_w1574 import (
    bench_gibbon_qa_studies_family,
    bench_langur_qa_studies_family,
    bench_lemur_qa_studies_family,
    bench_macaque_qa_studies_family,
    bench_marmoset_qa_studies_family,
    bench_tamarin_qa_studies_family,
)
from quant_fund.research.benches_w1575 import (
    bench_chinchilla_qa_studies_family,
    bench_degu_qa_studies_family,
    bench_gerbil_qa_studies_family,
    bench_hamster_qa_studies_family,
    bench_lemming_qa_studies_family,
    bench_vole_qa_studies_family,
)
from quant_fund.research.benches_w1576 import (
    bench_binturong_qa_studies_family,
    bench_fossa_qa_studies_family,
    bench_honey_badger_qa_studies_family,
    bench_kusimanse_qa_studies_family,
    bench_maned_wolf_qa_studies_family,
    bench_sun_bear_qa_studies_family,
)
from quant_fund.research.benches_w1577 import (
    bench_civet_qa_studies_family,
    bench_genet_qa_studies_family,
    bench_manul_qa_studies_family,
    bench_mongoose_qa_studies_family,
    bench_sloth_bear_qa_studies_family,
    bench_suricate_qa_studies_family,
)
from quant_fund.research.benches_w1578 import (
    bench_gerenuk_qa_studies_family,
    bench_markhor_qa_studies_family,
    bench_nilgai_qa_studies_family,
    bench_okapi_qa_studies_family,
    bench_saiga_qa_studies_family,
    bench_takin_qa_studies_family,
)
from quant_fund.research.benches_w1579 import (
    bench_dikdik_qa_studies_family,
    bench_grysbok_qa_studies_family,
    bench_klipspringer_qa_studies_family,
    bench_rhebok_qa_studies_family,
    bench_steenbok_qa_studies_family,
    bench_suni_qa_studies_family,
)
from quant_fund.research.benches_w1580 import (
    bench_chital_qa_studies_family,
    bench_fallow_qa_studies_family,
    bench_muntjac_qa_studies_family,
    bench_pudu_qa_studies_family,
    bench_roe_qa_studies_family,
    bench_sika_qa_studies_family,
)
from quant_fund.research.benches_w1581 import (
    bench_elephant_seal_qa_studies_family,
    bench_fur_seal_qa_studies_family,
    bench_harp_seal_qa_studies_family,
    bench_leopard_seal_qa_studies_family,
    bench_monk_seal_qa_studies_family,
    bench_weddell_qa_studies_family,
)
from quant_fund.research.benches_w1582 import (
    bench_flying_fox_qa_studies_family,
    bench_horseshoe_bat_qa_studies_family,
    bench_leaf_nosed_qa_studies_family,
    bench_noctule_qa_studies_family,
    bench_pipistrelle_qa_studies_family,
    bench_vampire_qa_studies_family,
)
from quant_fund.research.benches_w1583 import (
    bench_bowhead_qa_studies_family,
    bench_fin_whale_qa_studies_family,
    bench_humpback_qa_studies_family,
    bench_minke_qa_studies_family,
    bench_pilot_whale_qa_studies_family,
    bench_sperm_whale_qa_studies_family,
)
from quant_fund.research.benches_w1584 import (
    bench_cottontail_qa_studies_family,
    bench_hare_qa_studies_family,
    bench_hedgehog_qa_studies_family,
    bench_hyrax_qa_studies_family,
    bench_jackrabbit_qa_studies_family,
    bench_pika_qa_studies_family,
)
from quant_fund.research.benches_w1585 import (
    bench_aardvark_qa_studies_family,
    bench_elephant_shrew_qa_studies_family,
    bench_golden_mole_qa_studies_family,
    bench_gymnure_qa_studies_family,
    bench_solenodon_qa_studies_family,
    bench_tenrec_qa_studies_family,
)
from quant_fund.research.benches_w1586 import (
    bench_beira_qa_studies_family,
    bench_gemsbok_qa_studies_family,
    bench_madoqua_qa_studies_family,
    bench_oribi_qa_studies_family,
    bench_reedbuck_qa_studies_family,
    bench_tsessebe_qa_studies_family,
)
from quant_fund.research.benches_w1587 import (
    bench_barasingha_qa_studies_family,
    bench_brocket_qa_studies_family,
    bench_huemul_qa_studies_family,
    bench_mule_deer_qa_studies_family,
    bench_sambar_qa_studies_family,
    bench_taruca_qa_studies_family,
)
from quant_fund.research.benches_w1588 import (
    bench_bearded_seal_qa_studies_family,
    bench_crabeater_qa_studies_family,
    bench_hooded_seal_qa_studies_family,
    bench_ribbon_seal_qa_studies_family,
    bench_ringed_seal_qa_studies_family,
    bench_ross_seal_qa_studies_family,
)
from quant_fund.research.benches_w1589 import (
    bench_porpoise_qa_studies_family,
    bench_right_whale_qa_studies_family,
    bench_rissos_qa_studies_family,
    bench_river_dolphin_qa_studies_family,
    bench_spinner_qa_studies_family,
    bench_vaquita_qa_studies_family,
)
from quant_fund.research.benches_w1590 import (
    bench_black_footed_qa_studies_family,
    bench_fishing_cat_qa_studies_family,
    bench_jungle_cat_qa_studies_family,
    bench_pallas_qa_studies_family,
    bench_rusty_spotted_qa_studies_family,
    bench_sand_cat_qa_studies_family,
)
from quant_fund.research.benches_w1591 import (
    bench_capuchin_qa_studies_family,
    bench_saki_qa_studies_family,
    bench_squirrel_monkey_qa_studies_family,
    bench_titi_qa_studies_family,
    bench_uakari_qa_studies_family,
    bench_woolly_qa_studies_family,
)
from quant_fund.research.benches_w1592 import (
    bench_bushbaby_qa_studies_family,
    bench_galago_qa_studies_family,
    bench_indri_qa_studies_family,
    bench_loris_qa_studies_family,
    bench_potto_qa_studies_family,
    bench_tarsier_qa_studies_family,
)
from quant_fund.research.benches_w1593 import (
    bench_colobus_qa_studies_family,
    bench_drill_qa_studies_family,
    bench_gelada_qa_studies_family,
    bench_guenon_qa_studies_family,
    bench_mandrill_qa_studies_family,
    bench_mangabey_qa_studies_family,
)
from quant_fund.research.benches_w1594 import (
    bench_bonobo_qa_studies_family,
    bench_chimpanzee_qa_studies_family,
    bench_douc_qa_studies_family,
    bench_proboscis_qa_studies_family,
    bench_siamang_qa_studies_family,
    bench_snub_nosed_qa_studies_family,
)
from quant_fund.research.benches_w1595 import (
    bench_andean_cat_qa_studies_family,
    bench_bay_cat_qa_studies_family,
    bench_flat_headed_qa_studies_family,
    bench_geoffroys_qa_studies_family,
    bench_marbled_cat_qa_studies_family,
    bench_pampas_cat_qa_studies_family,
)
from quant_fund.research.benches_w1596 import (
    bench_bottlenose_qa_studies_family,
    bench_dusky_dolphin_qa_studies_family,
    bench_false_killer_qa_studies_family,
    bench_melon_head_qa_studies_family,
    bench_pygmy_whale_qa_studies_family,
    bench_sea_lion_qa_studies_family,
)
from quant_fund.research.benches_w1597 import (
    bench_bharal_qa_studies_family,
    bench_chamois_qa_studies_family,
    bench_goral_qa_studies_family,
    bench_ibex_qa_studies_family,
    bench_serow_qa_studies_family,
    bench_tahr_qa_studies_family,
)
from quant_fund.research.benches_w1598 import (
    bench_bontebok_qa_studies_family,
    bench_bushbuck_qa_studies_family,
    bench_greater_kudu_qa_studies_family,
    bench_lesser_kudu_qa_studies_family,
    bench_mountain_nyala_qa_studies_family,
    bench_sitatunga_qa_studies_family,
)
from quant_fund.research.benches_w1599 import (
    bench_aye_aye_qa_studies_family,
    bench_howler_qa_studies_family,
    bench_mouse_lemur_qa_studies_family,
    bench_night_monkey_qa_studies_family,
    bench_ring_tailed_qa_studies_family,
    bench_spider_monkey_qa_studies_family,
)
from quant_fund.research.benches_w1600 import (
    bench_abalone_qa_studies_family,
    bench_chiton_qa_studies_family,
    bench_cockle_qa_studies_family,
    bench_cowrie_qa_studies_family,
    bench_limpet_qa_studies_family,
    bench_periwinkle_qa_studies_family,
)
from quant_fund.research.benches_w1601 import (
    bench_blossom_bat_qa_studies_family,
    bench_bulldog_bat_qa_studies_family,
    bench_free_tailed_qa_studies_family,
    bench_fruit_bat_qa_studies_family,
    bench_mouse_eared_qa_studies_family,
    bench_tent_bat_qa_studies_family,
)
from quant_fund.research.benches_w1602 import (
    bench_desman_qa_studies_family,
    bench_marsupial_mole_qa_studies_family,
    bench_moles_lite_qa_studies_family,
    bench_monotreme_qa_studies_family,
    bench_moonrat_qa_studies_family,
    bench_sengi_qa_studies_family,
)
from quant_fund.research.benches_w1603 import (
    bench_axis_qa_studies_family,
    bench_marsh_deer_qa_studies_family,
    bench_musk_deer_qa_studies_family,
    bench_pampas_deer_qa_studies_family,
    bench_tufted_qa_studies_family,
    bench_water_deer_qa_studies_family,
)
from quant_fund.research.benches_w1604 import (
    bench_dassie_qa_studies_family,
    bench_gopher_qa_studies_family,
    bench_mole_qa_studies_family,
    bench_rabbit_qa_studies_family,
    bench_shrew_qa_studies_family,
    bench_springhare_qa_studies_family,
)
from quant_fund.research.benches_w1605 import (
    bench_decorator_qa_studies_family,
    bench_fiddler_qa_studies_family,
    bench_rock_crab_qa_studies_family,
    bench_sea_snake_qa_studies_family,
    bench_skate_qa_studies_family,
    bench_wobbegong_qa_studies_family,
)
from quant_fund.research.benches_w1606 import (
    bench_alpaca_qa_studies_family,
    bench_aoudad_qa_studies_family,
    bench_dromedary_qa_studies_family,
    bench_guanaco_qa_studies_family,
    bench_salt_qa_studies_family,
    bench_vicuna_qa_studies_family,
)
from quant_fund.research.benches_w1607 import (
    bench_bamboo_lemur_qa_studies_family,
    bench_bearded_saki_qa_studies_family,
    bench_owl_monkey_qa_studies_family,
    bench_pale_titi_qa_studies_family,
    bench_uakari_2_qa_studies_family,
    bench_woolly_lemur_qa_studies_family,
)
from quant_fund.research.benches_w1608 import (
    bench_buffalo_qa_studies_family,
    bench_kob_qa_studies_family,
    bench_lechwe_qa_studies_family,
    bench_rhino_qa_studies_family,
    bench_roan_qa_studies_family,
    bench_warthog_qa_studies_family,
)
from quant_fund.research.benches_w1609 import (
    bench_black_lemur_qa_studies_family,
    bench_brown_lemur_qa_studies_family,
    bench_dwarf_lemur_qa_studies_family,
    bench_mongoose_lemur_qa_studies_family,
    bench_ruffed_qa_studies_family,
    bench_sportive_lemur_qa_studies_family,
)
from quant_fund.research.benches_w1610 import (
    bench_cavy_qa_studies_family,
    bench_coypu_qa_studies_family,
    bench_dhole_qa_studies_family,
    bench_mara_qa_studies_family,
    bench_porcupine_qa_studies_family,
    bench_ratel_qa_studies_family,
)
from quant_fund.research.benches_w1611 import (
    bench_crowned_lemur_qa_studies_family,
    bench_fat_tailed_qa_studies_family,
    bench_fork_marked_qa_studies_family,
    bench_needle_clawed_qa_studies_family,
    bench_ringtail_qa_studies_family,
    bench_sifaka_qa_studies_family,
)
from quant_fund.research.benches_w1612 import (
    bench_argali_qa_studies_family,
    bench_bighorn_qa_studies_family,
    bench_dall_qa_studies_family,
    bench_llama_qa_studies_family,
    bench_mouflon_qa_studies_family,
    bench_urial_qa_studies_family,
)
from quant_fund.research.benches_w1613 import (
    bench_aurochs_qa_studies_family,
    bench_banteng_qa_studies_family,
    bench_gaur_qa_studies_family,
    bench_saola_qa_studies_family,
    bench_tamaraw_qa_studies_family,
    bench_yak_qa_studies_family,
)
from quant_fund.research.benches_w1614 import (
    bench_hog_deer_qa_studies_family,
    bench_kouprey_qa_studies_family,
    bench_mule_qa_studies_family,
    bench_pere_david_qa_studies_family,
    bench_red_deer_qa_studies_family,
    bench_wapiti_qa_studies_family,
)
from quant_fund.research.benches_w1615 import (
    bench_golden_brown_qa_studies_family,
    bench_gray_mouse_qa_studies_family,
    bench_pygmy_qa_studies_family,
    bench_slender_qa_studies_family,
    bench_slow_qa_studies_family,
    bench_thin_spined_qa_studies_family,
)
from quant_fund.research.benches_w1616 import (
    bench_amphipod_qa_studies_family,
    bench_barnacle_qa_studies_family,
    bench_copepod_qa_studies_family,
    bench_isopod_qa_studies_family,
    bench_krill_qa_studies_family,
    bench_sandhopper_qa_studies_family,
)
from quant_fund.research.benches_w1617 import (
    bench_amber_mountain_qa_studies_family,
    bench_anosy_qa_studies_family,
    bench_daraina_qa_studies_family,
    bench_red_bellied_qa_studies_family,
    bench_russet_qa_studies_family,
    bench_white_footed_qa_studies_family,
)
from quant_fund.research.benches_w1618 import (
    bench_anglerfish_qa_studies_family,
    bench_bristlemouth_qa_studies_family,
    bench_grenadier_qa_studies_family,
    bench_hatchetfish_qa_studies_family,
    bench_lanternfish_qa_studies_family,
    bench_viperfish_qa_studies_family,
)
from quant_fund.research.benches_w1619 import (
    bench_blobfish_qa_studies_family,
    bench_dragonfish_qa_studies_family,
    bench_dumbo_qa_studies_family,
    bench_fangtooth_qa_studies_family,
    bench_gulper_qa_studies_family,
    bench_tripodfish_qa_studies_family,
)
from quant_fund.research.benches_w1620 import (
    bench_boomslang_qa_studies_family,
    bench_death_adder_qa_studies_family,
    bench_gaboon_qa_studies_family,
    bench_inland_taipan_qa_studies_family,
    bench_saw_scaled_qa_studies_family,
    bench_sea_krait_qa_studies_family,
)
from quant_fund.research.benches_w1621 import (
    bench_cave_beetle_qa_studies_family,
    bench_cave_cricket_qa_studies_family,
    bench_cave_fish_qa_studies_family,
    bench_mudpuppy_qa_studies_family,
    bench_olm_qa_studies_family,
    bench_troglobite_qa_studies_family,
)
from quant_fund.research.benches_w1622 import (
    bench_barbary_qa_studies_family,
    bench_blue_sheep_qa_studies_family,
    bench_himalayan_qa_studies_family,
    bench_nilgiri_qa_studies_family,
    bench_snow_leopard_qa_studies_family,
    bench_snowcock_qa_studies_family,
)
from quant_fund.research.benches_w1623 import (
    bench_altai_qa_studies_family,
    bench_blood_pheasant_qa_studies_family,
    bench_chukar_qa_studies_family,
    bench_monal_qa_studies_family,
    bench_snow_partridge_qa_studies_family,
    bench_wallcreeper_qa_studies_family,
)
from quant_fund.research.benches_w1624 import (
    bench_arctic_hare_qa_studies_family,
    bench_gyrfalcon_qa_studies_family,
    bench_pallas_manul_qa_studies_family,
    bench_ptarmigan_qa_studies_family,
    bench_snowshoe_qa_studies_family,
    bench_tundra_swan_qa_studies_family,
)
from quant_fund.research.benches_w1625 import (
    bench_blind_salamander_qa_studies_family,
    bench_cave_shrimp_qa_studies_family,
    bench_cave_spider_qa_studies_family,
    bench_cave_swiftlet_qa_studies_family,
    bench_grotto_salamander_qa_studies_family,
    bench_proteus_qa_studies_family,
)
from quant_fund.research.benches_w1626 import (
    bench_bondolo_qa_studies_family,
    bench_madame_berthe_qa_studies_family,
    bench_mittermeier_qa_studies_family,
    bench_northern_qa_studies_family,
    bench_southern_qa_studies_family,
    bench_western_qa_studies_family,
)
from quant_fund.research.benches_w1627 import (
    bench_cave_crayfish_qa_studies_family,
    bench_cave_scorpion_qa_studies_family,
    bench_cave_springtail_qa_studies_family,
    bench_cave_worm_qa_studies_family,
    bench_stygobite_qa_studies_family,
    bench_troglofish_qa_studies_family,
)
from quant_fund.research.benches_w1628 import (
    bench_colugo_qa_studies_family,
    bench_geoffroy_qa_studies_family,
    bench_mandarin_qa_studies_family,
    bench_moray_eel_qa_studies_family,
    bench_pangolin_2_qa_studies_family,
    bench_satyr_qa_studies_family,
)
from quant_fund.research.benches_w1629 import (
    bench_chupacabra_qa_studies_family,
    bench_jersey_devil_qa_studies_family,
    bench_kraken_2_qa_studies_family,
    bench_mothman_qa_studies_family,
    bench_thunderbird_qa_studies_family,
    bench_yeti_2_qa_studies_family,
)
from quant_fund.research.benches_w1630 import (
    bench_bigfoot_qa_studies_family,
    bench_bunyip_qa_studies_family,
    bench_loch_ness_qa_studies_family,
    bench_rougarou_qa_studies_family,
    bench_skinwalker_qa_studies_family,
    bench_wendigo_qa_studies_family,
)
from quant_fund.research.benches_w1631 import (
    bench_basilisk_qa_studies_family,
    bench_chimera_qa_studies_family,
    bench_gorgon_qa_studies_family,
    bench_griffin_2_qa_studies_family,
    bench_hydra_2_qa_studies_family,
    bench_manticore_qa_studies_family,
)
from quant_fund.research.benches_w1632 import (
    bench_cerberus_qa_studies_family,
    bench_dragon_2_qa_studies_family,
    bench_minotaur_qa_studies_family,
    bench_pegasus_qa_studies_family,
    bench_phoenix_2_qa_studies_family,
    bench_unicorn_2_qa_studies_family,
)
from quant_fund.research.benches_w1633 import (
    bench_air_sylph_qa_studies_family,
    bench_earth_golem_qa_studies_family,
    bench_fire_spirit_qa_studies_family,
    bench_frost_wight_qa_studies_family,
    bench_storm_jinn_qa_studies_family,
    bench_water_sprite_qa_studies_family,
)
from quant_fund.research.benches_w1634 import (
    bench_gnome_2_qa_studies_family,
    bench_ifrit_qa_studies_family,
    bench_marid_qa_studies_family,
    bench_salamander_2_qa_studies_family,
    bench_sylph_2_qa_studies_family,
    bench_undine_qa_studies_family,
)
from quant_fund.research.benches_w1635 import (
    bench_kappa_qa_studies_family,
    bench_kitsune_2_qa_studies_family,
    bench_oni_qa_studies_family,
    bench_tanuki_2_qa_studies_family,
    bench_tengu_qa_studies_family,
    bench_tsukumogami_qa_studies_family,
)
from quant_fund.research.benches_w1636 import (
    bench_gashadokuro_qa_studies_family,
    bench_jorogumo_qa_studies_family,
    bench_kodama_qa_studies_family,
    bench_namahage_qa_studies_family,
    bench_nue_2_qa_studies_family,
    bench_tsuchinoko_qa_studies_family,
)
from quant_fund.research.benches_w1637 import (
    bench_abura_sumashi_qa_studies_family,
    bench_azukiarai_qa_studies_family,
    bench_betobeto_2_qa_studies_family,
    bench_futakuchi_qa_studies_family,
    bench_rokurokubi_qa_studies_family,
    bench_shirime_qa_studies_family,
)
from quant_fund.research.benches_w1638 import (
    bench_akaname_qa_studies_family,
    bench_hitodama_qa_studies_family,
    bench_ittanmomen_qa_studies_family,
    bench_nurikabe_qa_studies_family,
    bench_shikigami_qa_studies_family,
    bench_ubume_qa_studies_family,
)
from quant_fund.research.benches_w1639 import (
    bench_dorotabo_qa_studies_family,
    bench_kitsune_3_qa_studies_family,
    bench_tanuki_3_qa_studies_family,
    bench_tengu_2_qa_studies_family,
    bench_yukionna_qa_studies_family,
    bench_zashiki_warashi_qa_studies_family,
)
from quant_fund.research.benches_w1640 import (
    bench_byakko_qa_studies_family,
    bench_genbu_qa_studies_family,
    bench_kirin_2_qa_studies_family,
    bench_kohryu_qa_studies_family,
    bench_seiryu_qa_studies_family,
    bench_suzaku_qa_studies_family,
)
from quant_fund.research.benches_w1641 import (
    bench_bixie_qa_studies_family,
    bench_fenghuang_qa_studies_family,
    bench_hundun_qa_studies_family,
    bench_qiongqi_qa_studies_family,
    bench_taotie_qa_studies_family,
    bench_taowu_qa_studies_family,
)
from quant_fund.research.benches_w1642 import (
    bench_basilisk_2_qa_studies_family,
    bench_chimera_2_qa_studies_family,
    bench_cockatrice_qa_studies_family,
    bench_manticore_2_qa_studies_family,
    bench_sphinx_2_qa_studies_family,
    bench_wyvern_2_qa_studies_family,
)
from quant_fund.research.benches_w1643 import (
    bench_centaur_2_qa_studies_family,
    bench_gryphon_qa_studies_family,
    bench_harpy_2_qa_studies_family,
    bench_hippogryph_qa_studies_family,
    bench_minotaur_2_qa_studies_family,
    bench_satyr_2_qa_studies_family,
)
from quant_fund.research.benches_w1644 import (
    bench_charybdis_qa_studies_family,
    bench_cyclops_2_qa_studies_family,
    bench_hydra_3_qa_studies_family,
    bench_medusa_2_qa_studies_family,
    bench_scylla_qa_studies_family,
    bench_siren_2_qa_studies_family,
)
from quant_fund.research.benches_w1645 import (
    bench_argus_qa_studies_family,
    bench_cerberus_2_qa_studies_family,
    bench_nemean_qa_studies_family,
    bench_orthrus_qa_studies_family,
    bench_pegasus_2_qa_studies_family,
    bench_typhon_qa_studies_family,
)
from quant_fund.research.benches_w1646 import (
    bench_fenrir_2_qa_studies_family,
    bench_garm_qa_studies_family,
    bench_jormungandr_qa_studies_family,
    bench_nidhogg_qa_studies_family,
    bench_ratatoskr_qa_studies_family,
    bench_sleipnir_qa_studies_family,
)
from quant_fund.research.benches_w1647 import (
    bench_akhekh_qa_studies_family,
    bench_ammit_qa_studies_family,
    bench_apophis_qa_studies_family,
    bench_bes_qa_studies_family,
    bench_sekhmet_qa_studies_family,
    bench_sphairo_qa_studies_family,
)
from quant_fund.research.benches_w1648 import (
    bench_aitvaras_qa_studies_family,
    bench_bilwis_qa_studies_family,
    bench_indus_qa_studies_family,
    bench_kudlak_qa_studies_family,
    bench_viy_qa_studies_family,
    bench_zilant_qa_studies_family,
)
from quant_fund.research.benches_w1649 import (
    bench_aswang_qa_studies_family,
    bench_bakunawa_qa_studies_family,
    bench_berbalang_qa_studies_family,
    bench_kapre_qa_studies_family,
    bench_sigbin_qa_studies_family,
    bench_tikbalang_qa_studies_family,
)
from quant_fund.research.benches_w1650 import (
    bench_adjule_qa_studies_family,
    bench_agogwe_qa_studies_family,
    bench_biloko_qa_studies_family,
    bench_kongamato_qa_studies_family,
    bench_popobawa_qa_studies_family,
    bench_rompo_qa_studies_family,
)
from quant_fund.research.benches_w1651 import (
    bench_awgy_qa_studies_family,
    bench_kuritja_qa_studies_family,
    bench_minka_qa_studies_family,
    bench_papin_qa_studies_family,
    bench_yara_qa_studies_family,
    bench_yowie_qa_studies_family,
)
from quant_fund.research.benches_w1652 import (
    bench_cuco_qa_studies_family,
    bench_dahu_qa_studies_family,
    bench_gargouille_qa_studies_family,
    bench_lavellan_qa_studies_family,
    bench_muscaliet_qa_studies_family,
    bench_tarasque_qa_studies_family,
)
from quant_fund.research.benches_w1653 import (
    bench_alion_qa_studies_family,
    bench_catoblepas_qa_studies_family,
    bench_jasconius_qa_studies_family,
    bench_pard_qa_studies_family,
    bench_peluda_qa_studies_family,
    bench_zaratan_qa_studies_family,
)
from quant_fund.research.benches_w1654 import (
    bench_amphisbaena_qa_studies_family,
    bench_bonnacon_qa_studies_family,
    bench_cerastes_qa_studies_family,
    bench_leucrotta_qa_studies_family,
    bench_parandrus_qa_studies_family,
    bench_questing_qa_studies_family,
)
from quant_fund.research.benches_w1655 import (
    bench_basiliskcock_qa_studies_family,
    bench_calygreyhound_qa_studies_family,
    bench_cocatrix_qa_studies_family,
    bench_gryps_qa_studies_family,
    bench_mantygre_qa_studies_family,
    bench_opinicus_qa_studies_family,
)
from quant_fund.research.benches_w1656 import (
    bench_aatxe_qa_studies_family,
    bench_achiyalabopa_qa_studies_family,
    bench_afanc_qa_studies_family,
    bench_akhlut_qa_studies_family,
    bench_amarok_qa_studies_family,
    bench_eachuisge_qa_studies_family,
)
from quant_fund.research.benches_w1657 import (
    bench_gargoyle_qa_studies_family,
    bench_guivre_qa_studies_family,
    bench_melusine_qa_studies_family,
    bench_quinotaur_qa_studies_family,
    bench_tarascon_qa_studies_family,
    bench_tarrasque_qa_studies_family,
)
from quant_fund.research.benches_w1658 import (
    bench_ahuizotl_qa_studies_family,
    bench_alicanto_qa_studies_family,
    bench_cadejo_qa_studies_family,
    bench_cipactli_qa_studies_family,
    bench_jinn_qa_studies_family,
    bench_quetzalcoat_qa_studies_family,
)
from quant_fund.research.benches_w1659 import (
    bench_manananggal_qa_studies_family,
    bench_minokawa_qa_studies_family,
    bench_nuno_qa_studies_family,
    bench_siyokoy_qa_studies_family,
    bench_tiyanak_qa_studies_family,
    bench_wakwak_qa_studies_family,
)
from quant_fund.research.benches_w1660 import (
    bench_centaur_qa_studies_family,
    bench_cyclops_qa_studies_family,
    bench_griffin_qa_studies_family,
    bench_hydra_qa_studies_family,
    bench_medusa_qa_studies_family,
    bench_sphinx_qa_studies_family,
)
from quant_fund.research.benches_w1661 import (
    bench_draugr_qa_studies_family,
    bench_fenrir_qa_studies_family,
    bench_gullinbursti_qa_studies_family,
    bench_hraesvelgr_qa_studies_family,
    bench_huginn_qa_studies_family,
    bench_muninn_qa_studies_family,
)
from quant_fund.research.benches_w1662 import (
    bench_banshee_qa_studies_family,
    bench_dullahan_qa_studies_family,
    bench_kelpie_qa_studies_family,
    bench_leprechaun_qa_studies_family,
    bench_puca_qa_studies_family,
    bench_selkie_qa_studies_family,
)
from quant_fund.research.benches_w1663 import (
    bench_barghest_qa_studies_family,
    bench_black_dog_qa_studies_family,
    bench_cat_sith_qa_studies_family,
    bench_church_grim_qa_studies_family,
    bench_cwn_annwn_qa_studies_family,
    bench_grimalkin_qa_studies_family,
)
from quant_fund.research.benches_w1664 import (
    bench_kraken_qa_studies_family,
    bench_krampus_qa_studies_family,
    bench_roc_qa_studies_family,
    bench_simurgh_qa_studies_family,
    bench_siren_qa_studies_family,
    bench_wyvern_qa_studies_family,
)
from quant_fund.research.benches_w1665 import (
    bench_einherjar_qa_studies_family,
    bench_hati_qa_studies_family,
    bench_lindworm_qa_studies_family,
    bench_skoll_qa_studies_family,
    bench_vargbroder_qa_studies_family,
    bench_vedrfolnir_qa_studies_family,
)
from quant_fund.research.benches_w1666 import (
    bench_agta_qa_studies_family,
    bench_berberoka_qa_studies_family,
    bench_bungisngis_qa_studies_family,
    bench_dalaketnon_qa_studies_family,
    bench_ekek_qa_studies_family,
    bench_engkanto_qa_studies_family,
)
from quant_fund.research.benches_w1667 import (
    bench_centauride_qa_studies_family,
    bench_dryad_qa_studies_family,
    bench_faun_qa_studies_family,
    bench_hamadryad_qa_studies_family,
    bench_nereid_qa_studies_family,
    bench_nymph_qa_studies_family,
)
from quant_fund.research.benches_w1668 import (
    bench_berserkr_qa_studies_family,
    bench_fafnir_qa_studies_family,
    bench_jotun_qa_studies_family,
    bench_regin_qa_studies_family,
    bench_ulfhednar_qa_studies_family,
    bench_vargr_qa_studies_family,
)
from quant_fund.research.benches_w1669 import (
    bench_ghouling_qa_studies_family,
    bench_ikugan_qa_studies_family,
    bench_kataw_qa_studies_family,
    bench_lambana_qa_studies_family,
    bench_sarimanok_qa_studies_family,
    bench_tamahaling_qa_studies_family,
)
from quant_fund.research.benches_w1670 import (
    bench_alseid_qa_studies_family,
    bench_gnome_volk_qa_studies_family,
    bench_meliae_qa_studies_family,
    bench_napaea_qa_studies_family,
    bench_oread_qa_studies_family,
    bench_sylph_qa_studies_family,
)
from quant_fund.research.benches_w1671 import (
    bench_drakk_qa_studies_family,
    bench_grimr_qa_studies_family,
    bench_hildr_qa_studies_family,
    bench_mare_qa_studies_family,
    bench_nisse_qa_studies_family,
    bench_sigrun_qa_studies_family,
)
from quant_fund.research.benches_w1672 import (
    bench_domovoi_qa_studies_family,
    bench_kikimora_qa_studies_family,
    bench_leshy_qa_studies_family,
    bench_polevik_qa_studies_family,
    bench_rusalka_qa_studies_family,
    bench_vodianoi_qa_studies_family,
)
from quant_fund.research.benches_w1673 import (
    bench_apsara_qa_studies_family,
    bench_asura_qa_studies_family,
    bench_gandharva_qa_studies_family,
    bench_naga_qa_studies_family,
    bench_rakshasa_qa_studies_family,
    bench_yaksha_qa_studies_family,
)
from quant_fund.research.benches_w1674 import (
    bench_genii_qa_studies_family,
    bench_lares_qa_studies_family,
    bench_larvae_qa_studies_family,
    bench_lemures_qa_studies_family,
    bench_manes_qa_studies_family,
    bench_penates_qa_studies_family,
)
from quant_fund.research.benches_w1675 import (
    bench_chaneque_qa_studies_family,
    bench_cihuateteo_qa_studies_family,
    bench_nagual_qa_studies_family,
    bench_tlalocan_qa_studies_family,
    bench_tzitzimitl_qa_studies_family,
    bench_xiuhcoatl_qa_studies_family,
)
from quant_fund.research.benches_w1676 import (
    bench_kinnara_qa_studies_family,
    bench_pisacha_qa_studies_family,
    bench_uraga_qa_studies_family,
    bench_vetala_qa_studies_family,
    bench_vidyadhara_qa_studies_family,
    bench_yakshini_qa_studies_family,
)
from quant_fund.research.benches_w1677 import (
    bench_antheia_qa_studies_family,
    bench_aurae_qa_studies_family,
    bench_camenae_qa_studies_family,
    bench_fauns_qa_studies_family,
    bench_limoniad_qa_studies_family,
    bench_numina_qa_studies_family,
)
from quant_fund.research.benches_w1678 import (
    bench_bannik_qa_studies_family,
    bench_dvorovoi_qa_studies_family,
    bench_mora_qa_studies_family,
    bench_ovinnik_qa_studies_family,
    bench_poludnica_qa_studies_family,
    bench_vila_qa_studies_family,
)
from quant_fund.research.benches_w1679 import (
    bench_alfheim_qa_studies_family,
    bench_bergrisi_qa_studies_family,
    bench_geirahod_qa_studies_family,
    bench_huldra_qa_studies_family,
    bench_troll_qa_studies_family,
    bench_vaetter_qa_studies_family,
)
from quant_fund.research.benches_w1680 import (
    bench_coatlicue_qa_studies_family,
    bench_mictlan_qa_studies_family,
    bench_mixcoatl_qa_studies_family,
    bench_tlaloc_qa_studies_family,
    bench_tonatiuh_qa_studies_family,
    bench_xipe_qa_studies_family,
)
from quant_fund.research.benches_w1681 import (
    bench_anansi_qa_studies_family,
    bench_impundulu_qa_studies_family,
    bench_kalulu_qa_studies_family,
    bench_mamiwata_qa_studies_family,
    bench_sasabonsam_qa_studies_family,
    bench_tokoloshe_qa_studies_family,
)
from quant_fund.research.benches_w1682 import (
    bench_alkonost_qa_studies_family,
    bench_gamayun_qa_studies_family,
    bench_sirin_qa_studies_family,
    bench_veles_qa_studies_family,
    bench_zhaba_qa_studies_family,
    bench_zmei_qa_studies_family,
)
from quant_fund.research.benches_w1683 import (
    bench_danava_qa_studies_family,
    bench_gana_qa_studies_family,
    bench_gandharva_qa_studies_family,
    bench_kalakeya_qa_studies_family,
    bench_kimpurusha_qa_studies_family,
    bench_rakshasa_qa_studies_family,
)
from quant_fund.research.benches_w1684 import (
    bench_cihuacoatl_qa_studies_family,
    bench_mayahuel_qa_studies_family,
    bench_oyohualli_qa_studies_family,
    bench_quetzalli_qa_studies_family,
    bench_teteoinnan_qa_studies_family,
    bench_yaotl_qa_studies_family,
)
from quant_fund.research.benches_w1685 import (
    bench_ettin_qa_studies_family,
    bench_fylgja_qa_studies_family,
    bench_landvaettir_qa_studies_family,
    bench_nokken_qa_studies_family,
    bench_seidr_qa_studies_family,
    bench_vette_qa_studies_family,
)
from quant_fund.research.benches_w1686 import (
    bench_abada_qa_studies_family,
    bench_adze_qa_studies_family,
    bench_ilomba_qa_studies_family,
    bench_nbanda_qa_studies_family,
    bench_ninki_qa_studies_family,
    bench_okubi_qa_studies_family,
)
from quant_fund.research.benches_w1687 import (
    bench_abti_qa_studies_family,
    bench_akh_qa_studies_family,
    bench_apep_qa_studies_family,
    bench_bastet_qa_studies_family,
    bench_khonsu_qa_studies_family,
    bench_sobek_qa_studies_family,
)
from quant_fund.research.benches_w1688 import (
    bench_indiges_qa_studies_family,
    bench_lar_qa_studies_family,
    bench_numen_qa_studies_family,
    bench_penates_qa_studies_family,
    bench_terminus_qa_studies_family,
    bench_vertumnus_qa_studies_family,
)
from quant_fund.research.benches_w1689 import (
    bench_apsara_qa_studies_family,
    bench_bhairava_qa_studies_family,
    bench_bhuta_qa_studies_family,
    bench_pretas_qa_studies_family,
    bench_vetal_qa_studies_family,
    bench_yaksha_qa_studies_family,
)
from quant_fund.research.benches_w1690 import (
    bench_kladenets_qa_studies_family,
    bench_kostroma_qa_studies_family,
    bench_leshii_qa_studies_family,
    bench_morozko_qa_studies_family,
    bench_vedmak_qa_studies_family,
    bench_yarilo_qa_studies_family,
)
from quant_fund.research.benches_w1691 import (
    bench_abzu_qa_studies_family,
    bench_enki_qa_studies_family,
    bench_enlil_qa_studies_family,
    bench_nanna_qa_studies_family,
    bench_tiamat_qa_studies_family,
    bench_utu_qa_studies_family,
)
from quant_fund.research.benches_w1692 import (
    bench_alfar_qa_studies_family,
    bench_draugar_qa_studies_family,
    bench_hulder_qa_studies_family,
    bench_muspell_qa_studies_family,
    bench_svartalf_qa_studies_family,
    bench_ymir_qa_studies_family,
)
from quant_fund.research.benches_w1693 import (
    bench_asag_qa_studies_family,
    bench_edimmu_qa_studies_family,
    bench_galla_qa_studies_family,
    bench_lamassu_qa_studies_family,
    bench_shedu_qa_studies_family,
    bench_utukku_qa_studies_family,
)
from quant_fund.research.benches_w1694 import (
    bench_duwende_qa_studies_family,
    bench_karibusa_qa_studies_family,
    bench_mambabarang_qa_studies_family,
    bench_mangkukulam_qa_studies_family,
    bench_sokoy_qa_studies_family,
    bench_tiktik_qa_studies_family,
)
from quant_fund.research.benches_w1695 import (
    bench_dijiang_qa_studies_family,
    bench_huli_qa_studies_family,
    bench_jiangshi_qa_studies_family,
    bench_mogwai_qa_studies_family,
    bench_yaoguai_qa_studies_family,
    bench_zhuyin_qa_studies_family,
)
from quant_fund.research.benches_w1696 import (
    bench_dongwanggong_qa_studies_family,
    bench_fuxi_qa_studies_family,
    bench_kuafu_qa_studies_family,
    bench_nuwa_qa_studies_family,
    bench_shennong_qa_studies_family,
    bench_xiwangmu_qa_studies_family,
)
from quant_fund.research.benches_w1697 import (
    bench_aoqin_qa_studies_family,
    bench_guandi_qa_studies_family,
    bench_houyi_qa_studies_family,
    bench_wenchang_qa_studies_family,
    bench_yutu_qa_studies_family,
    bench_zao_qa_studies_family,
)
from quant_fund.research.benches_w1698 import (
    bench_maui_qa_studies_family,
    bench_menahune_qa_studies_family,
    bench_pele_qa_studies_family,
    bench_rangi_qa_studies_family,
    bench_tane_qa_studies_family,
    bench_tangaroa_qa_studies_family,
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
