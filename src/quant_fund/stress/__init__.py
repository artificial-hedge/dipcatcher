"""Research stress-testing and scenario engine.

Simulation and diagnostics only. This package does not submit orders, talk to
a broker, or rewrite sealed research receipts.

Heavy numeric dependencies stay behind attribute access so ``dipcatcher --help``
does not import them.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "CRISIS_CATALOG": ("quant_fund.stress.catalog", "CRISIS_CATALOG"),
    "Crisis": ("quant_fund.stress.catalog", "Crisis"),
    "crisis_by_id": ("quant_fund.stress.catalog", "crisis_by_id"),
    "replay_crisis": ("quant_fund.stress.replay", "replay_crisis"),
    "replay_portfolio": ("quant_fund.stress.replay", "replay_portfolio"),
    "build_stress_report": ("quant_fund.stress.report", "build_stress_report"),
    "render_html": ("quant_fund.stress.report", "render_html"),
    "render_markdown": ("quant_fund.stress.report", "render_markdown"),
    "reverse_stress": ("quant_fund.stress.reverse", "reverse_stress"),
    "worst_linear_scenario": ("quant_fund.stress.reverse", "worst_linear_scenario"),
    "ResearchStrategy": ("quant_fund.stress.strategy", "ResearchStrategy"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
