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
