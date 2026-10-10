"""Proper-score ensemble forecast combination.

Combines multiple forecast members into one distributional forecast, scored
exclusively with proper scores (pinball/CRPS, QLIKE, log-score). Includes
quantile averaging (vincentization), Granger–Ramanathan regression weights,
QLIKE-optimal variance combination with shrinkage, online (exponentiated
gradient / fixed-share) combination, trimming, split-conformal ensembling,
and pseudo-Bayesian averaging.

Honesty: every evaluation uses proper scores on supplied data; nothing here
is a live-performance headline.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "adaptive_combine",
    "adaptive_vs_static",
    "band_coverage",
    "bma_combine",
    "bonferroni_union",
    "combined_quantiles",
    "cqr_correct",
    "cqr_coverage",
    "cqr_intervals",
    "cqr_scores",
    "crps_weights",
    "dm_test",
    "diversity_report",
    "empirical_coverage",
    "exceedance_rates",
    "encompassing_test",
    "error_correlation",
    "ewma_variance",
    "exponentiated_gradient",
    "fit_adaptive_weights",
    "fixed_share",
    "ambiguity_decomposition",
    "diversity_bonus",
    "effective_ensemble_size",
    "granger_ramanathan",
    "greedy_prune",
    "log_score_weights",
    "long_run_variance",
    "member_health_report",
    "member_loss_drift",
    "member_pinball",
    "leave_one_out_contribution",
    "median_ensemble",
    "min_var_weights",
    "pinball",
    "pinball_optimal_all_levels",
    "bias_correct_quantiles",
    "pinball_decomposition",
    "pinball_optimal_weights",
    "pruned_weights",
    "pruning_gain",
    "probability_average_gaussian",
    "qlike_losses",
    "quantile_average",
    "quantile_average_gaussian",
    "quantile_band_average",
    "regret_bound",
    "rolling_combination",
    "rolling_mad_variance",
    "shrink_to_equal",
    "split_conformal_intervals",
    "trimmed_mean_ensemble",
    "winkler_score",
    "winsorize_members",
]

_EXPORTS: dict[str, str] = {
    "quantile_average": "quant_fund.combination.quantile_average",
    "quantile_average_gaussian": "quant_fund.combination.quantile_average",
    "probability_average_gaussian": "quant_fund.combination.quantile_average",
    "pinball": "quant_fund.combination.quantile_average",
    "granger_ramanathan": "quant_fund.combination.granger_ramanathan",
    "rolling_combination": "quant_fund.combination.granger_ramanathan",
    "min_var_weights": "quant_fund.combination.qlike_optimal",
    "shrink_to_equal": "quant_fund.combination.qlike_optimal",
    "crps_weights": "quant_fund.combination.qlike_optimal",
    "exponentiated_gradient": "quant_fund.combination.online",
    "fixed_share": "quant_fund.combination.online",
    "regret_bound": "quant_fund.combination.online",
    "trimmed_mean_ensemble": "quant_fund.combination.trimming",
    "median_ensemble": "quant_fund.combination.trimming",
    "winsorize_members": "quant_fund.combination.trimming",
    "split_conformal_intervals": "quant_fund.combination.conformal_combo",
    "empirical_coverage": "quant_fund.combination.conformal_combo",
    "log_score_weights": "quant_fund.combination.bma",
    "bma_combine": "quant_fund.combination.bma",
    "ewma_variance": "quant_fund.combination.variance_ensemble",
    "rolling_mad_variance": "quant_fund.combination.variance_ensemble",
    "long_run_variance": "quant_fund.combination.variance_ensemble",
    "qlike_losses": "quant_fund.combination.variance_ensemble",
    "dm_test": "quant_fund.combination.variance_ensemble",
    "bonferroni_union": "quant_fund.combination.interval_ensemble",
    "quantile_band_average": "quant_fund.combination.interval_ensemble",
    "band_coverage": "quant_fund.combination.interval_ensemble",
    "error_correlation": "quant_fund.combination.ensemble_diag",
    "encompassing_test": "quant_fund.combination.ensemble_diag",
    "diversity_report": "quant_fund.combination.ensemble_diag",
    "exceedance_rates": "quant_fund.combination.quantile_bias",
    "pinball_decomposition": "quant_fund.combination.quantile_bias",
    "bias_correct_quantiles": "quant_fund.combination.quantile_bias",
    "cqr_scores": "quant_fund.combination.conformal_quantiles",
    "cqr_correct": "quant_fund.combination.conformal_quantiles",
    "cqr_intervals": "quant_fund.combination.conformal_quantiles",
    "cqr_coverage": "quant_fund.combination.conformal_quantiles",
    "winkler_score": "quant_fund.combination.interval_ensemble",
    "fit_adaptive_weights": "quant_fund.combination.adaptive_combo",
    "adaptive_combine": "quant_fund.combination.adaptive_combo",
    "adaptive_vs_static": "quant_fund.combination.adaptive_combo",
    "pinball_optimal_weights": "quant_fund.combination.quantile_regression_combo",
    "pinball_optimal_all_levels": "quant_fund.combination.quantile_regression_combo",
    "combined_quantiles": "quant_fund.combination.quantile_regression_combo",
    "member_health_report": "quant_fund.combination.member_health",
    "member_loss_drift": "quant_fund.combination.member_health",
    "member_pinball": "quant_fund.combination.member_health",
    "leave_one_out_contribution": "quant_fund.combination.member_health",
    "ambiguity_decomposition": "quant_fund.combination.diversity_metrics",
    "diversity_bonus": "quant_fund.combination.diversity_metrics",
    "effective_ensemble_size": "quant_fund.combination.diversity_metrics",
    "greedy_prune": "quant_fund.combination.ensemble_pruning",
    "pruned_weights": "quant_fund.combination.ensemble_pruning",
    "pruning_gain": "quant_fund.combination.ensemble_pruning",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
