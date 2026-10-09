"""Alpha/signal decay analytics.

Quantifies how a cross-sectional signal's predictive content decays with
horizon: IC time series, half-life estimation, horizon-IC curves, turnover
and cost-adjusted net IC, significance testing with multiplicity control,
and decay-aware signal combination. All statistics are proper diagnostics
(information coefficients and proper scores), never performance headlines.

Honesty: IC here is a rank-correlation diagnostic on the supplied panels.
Nothing in this package is a Sharpe/Sortino/P&L claim.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ar1_half_life",
    "block_bootstrap_ic_pvalue",
    "bootstrap_ir",
    "bootstrap_ir_delta",
    "breakeven_cost",
    "bucket_returns",
    "bucket_trend_ic",
    "classify_decay",
    "clip_negative_weights",
    "combine_signals",
    "compound_forward_returns",
    "cost_adjusted_ic",
    "curve_auc",
    "decay_profile_weights",
    "deflated_excess_t",
    "effective_sample_tstat",
    "expected_max_null_tstat",
    "efficiency_frontier",
    "ic_turnover_points",
    "trade_off_slope",
    "forward_returns_matrix",
    "group_ic_series",
    "group_ic_summary",
    "cusum_break",
    "group_spread_permutation",
    "ic_curve",
    "ic_factor_structure",
    "ic_matrix",
    "ic_summary",
    "ic_tstat",
    "information_ratio",
    "fused_signal",
    "fusion_beats_single",
    "horizon_ic_profile",
    "ic_weight_fuse",
    "lag_ic",
    "monotonicity_score",
    "net_ic_curve",
    "pearson_ic",
    "peak_lag",
    "regime_ic_difference",
    "regime_ic_dispersion",
    "optimal_holding_lag",
    "lag_breakeven_table",
    "regime_ic_summary",
    "rolling_ic_stats",
    "signal_clusters",
    "signal_ic_correlation",
    "signal_turnover",
    "signal_weights",
    "spearman_ic",
    "spread_curve",
    "stability_score",
    "winsorized_ic_weights",
]

_EXPORTS: dict[str, str] = {
    "spearman_ic": "quant_fund.decay.ic_series",
    "pearson_ic": "quant_fund.decay.ic_series",
    "ic_summary": "quant_fund.decay.ic_series",
    "ic_tstat": "quant_fund.decay.ic_series",
    "effective_sample_tstat": "quant_fund.decay.ic_series",
    "ar1_half_life": "quant_fund.decay.half_life",
    "classify_decay": "quant_fund.decay.half_life",
    "forward_returns_matrix": "quant_fund.decay.ic_curve",
    "compound_forward_returns": "quant_fund.decay.ic_curve",
    "ic_curve": "quant_fund.decay.ic_curve",
    "lag_ic": "quant_fund.decay.ic_curve",
    "curve_auc": "quant_fund.decay.ic_curve",
    "peak_lag": "quant_fund.decay.ic_curve",
    "signal_weights": "quant_fund.decay.turnover",
    "signal_turnover": "quant_fund.decay.turnover",
    "cost_adjusted_ic": "quant_fund.decay.turnover",
    "breakeven_cost": "quant_fund.decay.turnover",
    "expected_max_null_tstat": "quant_fund.decay.significance",
    "deflated_excess_t": "quant_fund.decay.significance",
    "block_bootstrap_ic_pvalue": "quant_fund.decay.significance",
    "winsorized_ic_weights": "quant_fund.decay.ensemble_weights",
    "clip_negative_weights": "quant_fund.decay.ensemble_weights",
    "decay_profile_weights": "quant_fund.decay.ensemble_weights",
    "combine_signals": "quant_fund.decay.ensemble_weights",
    "group_ic_series": "quant_fund.decay.ic_attribution",
    "group_ic_summary": "quant_fund.decay.ic_attribution",
    "group_spread_permutation": "quant_fund.decay.ic_attribution",
    "net_ic_curve": "quant_fund.decay.holding_period",
    "optimal_holding_lag": "quant_fund.decay.holding_period",
    "lag_breakeven_table": "quant_fund.decay.holding_period",
    "rolling_ic_stats": "quant_fund.decay.signal_stability",
    "cusum_break": "quant_fund.decay.signal_stability",
    "stability_score": "quant_fund.decay.signal_stability",
    "bucket_returns": "quant_fund.decay.quantile_spread",
    "spread_curve": "quant_fund.decay.quantile_spread",
    "monotonicity_score": "quant_fund.decay.quantile_spread",
    "bucket_trend_ic": "quant_fund.decay.quantile_spread",
    "efficiency_frontier": "quant_fund.decay.frontier",
    "ic_turnover_points": "quant_fund.decay.frontier",
    "trade_off_slope": "quant_fund.decay.frontier",
    "fused_signal": "quant_fund.decay.lag_weighting",
    "fusion_beats_single": "quant_fund.decay.lag_weighting",
    "horizon_ic_profile": "quant_fund.decay.lag_weighting",
    "ic_weight_fuse": "quant_fund.decay.lag_weighting",
    "ic_matrix": "quant_fund.decay.ic_matrix",
    "signal_ic_correlation": "quant_fund.decay.ic_matrix",
    "ic_factor_structure": "quant_fund.decay.ic_matrix",
    "signal_clusters": "quant_fund.decay.ic_matrix",
    "information_ratio": "quant_fund.decay.bootstrap_ir",
    "bootstrap_ir": "quant_fund.decay.bootstrap_ir",
    "bootstrap_ir_delta": "quant_fund.decay.bootstrap_ir",
    "regime_ic_summary": "quant_fund.decay.regime_ic",
    "regime_ic_difference": "quant_fund.decay.regime_ic",
    "regime_ic_dispersion": "quant_fund.decay.regime_ic",
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
