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
    "bma_combine",
    "crps_weights",
    "empirical_coverage",
    "exponentiated_gradient",
    "fixed_share",
    "granger_ramanathan",
    "log_score_weights",
    "median_ensemble",
    "min_var_weights",
    "pinball",
    "probability_average_gaussian",
    "quantile_average",
    "quantile_average_gaussian",
    "regret_bound",
    "rolling_combination",
    "shrink_to_equal",
    "split_conformal_intervals",
    "trimmed_mean_ensemble",
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
