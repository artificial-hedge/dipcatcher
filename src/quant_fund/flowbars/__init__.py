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
    "avg_uniqueness",
    "bar_ohlc",
    "bar_return_stats",
    "bars_per_period",
    "compare_bar_returns",
    "dollar_bar_ids",
    "dollar_imbalance_bar_ids",
    "dollar_run_bar_ids",
    "entropy_weights",
    "frac_diff_apply",
    "frac_diff_weights",
    "information_bar_ids",
    "label_overlap_matrix",
    "min_stationary_d",
    "rolling_shannon_entropy",
    "run_bar_ids",
    "sequential_bootstrap",
    "shannon_entropy",
    "synth_frac_integrated",
    "synth_tape",
    "tick_bar_ids",
    "tick_imbalance_bar_ids",
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
