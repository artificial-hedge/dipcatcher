"""Leakage Hunter: AST linter + proximity headline matcher + runtime watchdog.

``report`` and ``watchdog`` pull PROOFCORE contracts. The CLI imports this
package only to mount ``leakage.cli``, so those surfaces load on first use.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from quant_fund.leakage.patterns import (
    FORBIDDEN_HEADLINE_TOKENS,
    PROXIMITY_WINDOW,
    find_forbidden_headline,
)
from quant_fund.leakage.rules import RULE_REGISTRY, RuleSpec

__all__ = [
    "FORBIDDEN_HEADLINE_TOKENS",
    "PROXIMITY_WINDOW",
    "RULE_REGISTRY",
    "LeakageError",
    "LeakageWatchdog",
    "RuleSpec",
    "WatchdogProtocol",
    "find_forbidden_headline",
    "report_to_json",
    "report_to_text",
    "scan_paths",
]

_EXPORTS: dict[str, str] = {
    "LeakageError": "quant_fund.leakage.watchdog",
    "LeakageWatchdog": "quant_fund.leakage.watchdog",
    "WatchdogProtocol": "quant_fund.leakage.watchdog",
    "report_to_json": "quant_fund.leakage.report",
    "report_to_text": "quant_fund.leakage.report",
    "scan_paths": "quant_fund.leakage.report",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
