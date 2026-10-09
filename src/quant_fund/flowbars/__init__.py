"""Information-driven bar construction and sampling tools.

Implements the *Advances in Financial ML* toolkit: dollar/volume/imbalance/run
bars, bar-return statistics, fractional differentiation, and sequential
bootstrap sample weights. Everything operates on synthetic or ingested trade
tapes; no market-performance claims are made anywhere in this package.

Honesty: every ``synth_*`` generator produces labeled synthetic data used
only for correctness tests. Bar statistics are diagnostics, never evidence
of live-trading performance.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "adf_pvalue",
    "aggregate_ofi",
    "ar_signed_flow",
    "avg_uniqueness",
    "bar_feature_frame",
    "bar_ohlc",
    "bar_return_stats",
    "bars_per_period",
    "bipower_variation",
    "bns_ratio_test",
    "compare_bar_returns",
    "cross_impact_bootstrap_p",
    "cross_impact_matrix",
    "diagonal_dominance",
    "bar_builder_audit",
    "invariant_monotone_ids",
    "invariant_ohlc_consistent",
    "invariant_returns_telescope",
    "invariant_volume_conserved",
    "dollar_bar_ids",
    "dollar_imbalance_bar_ids",
    "dollar_run_bar_ids",
    "deseasonalize_returns",
    "effective_spread_from_quotes",
    "entropy_weights",
    "frac_diff_apply",
    "frac_diff_weights",
    "huang_tauchen_z",
    "information_bar_ids",
    "integrated_jump_component",
    "kyle_lambda_ofi",
    "label_overlap_matrix",
    "medianrv",
    "min_stationary_d",
    "minrv",
    "ofi_events",
    "realized_kernel",
    "realized_variance",
    "rolling_shannon_entropy",
    "roll_implied_spread",
    "roll_implied_spread_full",
    "seasonal_strength",
    "seasonality_curve",
    "run_bar_ids",
    "sequential_bootstrap",
    "shannon_entropy",
    "synth_frac_integrated",
    "synth_tape",
    "tape_quality_report",
    "discretization_grid",
    "gap_time_stats",
    "outlier_trade_rate",
    "zero_change_runs",
    "tick_bar_ids",
    "tick_imbalance_bar_ids",
    "select_threshold",
    "threshold_profile",
    "tick_rule_signs",
    "tick_run_bar_ids",
    "uniqueness_weights",
    "volume_bar_ids",
    "volume_imbalance_bar_ids",
    "volume_run_bar_ids",
]

_EXPORTS: dict[str, str] = {
    "dollar_bar_ids": "quant_fund.flowbars.bars",
    "volume_bar_ids": "quant_fund.flowbars.bars",
    "tick_bar_ids": "quant_fund.flowbars.bars",
    "bar_ohlc": "quant_fund.flowbars.bars",
    "synth_tape": "quant_fund.flowbars.bars",
    "tick_rule_signs": "quant_fund.flowbars.imbalance",
    "tick_imbalance_bar_ids": "quant_fund.flowbars.imbalance",
    "volume_imbalance_bar_ids": "quant_fund.flowbars.imbalance",
    "dollar_imbalance_bar_ids": "quant_fund.flowbars.imbalance",
    "run_bar_ids": "quant_fund.flowbars.runs",
    "tick_run_bar_ids": "quant_fund.flowbars.runs",
    "volume_run_bar_ids": "quant_fund.flowbars.runs",
    "dollar_run_bar_ids": "quant_fund.flowbars.runs",
    "information_bar_ids": "quant_fund.flowbars.information",
    "shannon_entropy": "quant_fund.flowbars.entropy",
    "rolling_shannon_entropy": "quant_fund.flowbars.entropy",
    "entropy_weights": "quant_fund.flowbars.entropy",
    "bar_return_stats": "quant_fund.flowbars.stats",
    "bars_per_period": "quant_fund.flowbars.stats",
    "compare_bar_returns": "quant_fund.flowbars.stats",
    "frac_diff_weights": "quant_fund.flowbars.fracdiff",
    "frac_diff_apply": "quant_fund.flowbars.fracdiff",
    "adf_pvalue": "quant_fund.flowbars.fracdiff",
    "min_stationary_d": "quant_fund.flowbars.fracdiff",
    "synth_frac_integrated": "quant_fund.flowbars.fracdiff",
    "label_overlap_matrix": "quant_fund.flowbars.sample_weights",
    "avg_uniqueness": "quant_fund.flowbars.sample_weights",
    "sequential_bootstrap": "quant_fund.flowbars.sample_weights",
    "uniqueness_weights": "quant_fund.flowbars.sample_weights",
    "realized_variance": "quant_fund.flowbars.realized",
    "bipower_variation": "quant_fund.flowbars.realized",
    "minrv": "quant_fund.flowbars.realized",
    "medianrv": "quant_fund.flowbars.realized",
    "realized_kernel": "quant_fund.flowbars.realized",
    "bns_ratio_test": "quant_fund.flowbars.realized",
    "huang_tauchen_z": "quant_fund.flowbars.realized",
    "integrated_jump_component": "quant_fund.flowbars.realized",
    "ofi_events": "quant_fund.flowbars.ofi_impact",
    "aggregate_ofi": "quant_fund.flowbars.ofi_impact",
    "kyle_lambda_ofi": "quant_fund.flowbars.ofi_impact",
    "ar_signed_flow": "quant_fund.flowbars.ofi_impact",
    "cross_impact_matrix": "quant_fund.flowbars.cross_impact",
    "diagonal_dominance": "quant_fund.flowbars.cross_impact",
    "cross_impact_bootstrap_p": "quant_fund.flowbars.cross_impact",
    "bar_feature_frame": "quant_fund.flowbars.bar_features",
    "seasonality_curve": "quant_fund.flowbars.seasonality",
    "deseasonalize_returns": "quant_fund.flowbars.seasonality",
    "seasonal_strength": "quant_fund.flowbars.seasonality",
    "roll_implied_spread": "quant_fund.flowbars.spread_estimators",
    "roll_implied_spread_full": "quant_fund.flowbars.spread_estimators",
    "effective_spread_from_quotes": "quant_fund.flowbars.spread_estimators",
    "bar_builder_audit": "quant_fund.flowbars.roundtrip",
    "invariant_monotone_ids": "quant_fund.flowbars.roundtrip",
    "invariant_ohlc_consistent": "quant_fund.flowbars.roundtrip",
    "invariant_returns_telescope": "quant_fund.flowbars.roundtrip",
    "invariant_volume_conserved": "quant_fund.flowbars.roundtrip",
    "tape_quality_report": "quant_fund.flowbars.tape_quality",
    "discretization_grid": "quant_fund.flowbars.tape_quality",
    "gap_time_stats": "quant_fund.flowbars.tape_quality",
    "outlier_trade_rate": "quant_fund.flowbars.tape_quality",
    "zero_change_runs": "quant_fund.flowbars.tape_quality",
    "threshold_profile": "quant_fund.flowbars.threshold_selection",
    "select_threshold": "quant_fund.flowbars.threshold_selection",
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
