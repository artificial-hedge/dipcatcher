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
    "breakeven_cost",
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
    "forward_returns_matrix",
    "ic_curve",
    "ic_summary",
    "ic_tstat",
    "lag_ic",
    "pearson_ic",
    "peak_lag",
    "signal_turnover",
    "signal_weights",
    "spearman_ic",
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
