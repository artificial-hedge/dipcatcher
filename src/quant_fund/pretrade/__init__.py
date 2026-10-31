"""Standalone pre-trade risk engine.

This package does not submit orders and is not imported by the simulated
broker or the paper loop. ``quant_fund.live`` calls it and submits only
after a decision allows the order. See ``docs/PRETRADE_RISK.md``.

``shadow`` pulls the simulated broker and its numeric stack. Those names
load on first use so importing this package does not import them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from quant_fund.pretrade.codes import (
    HARD_LIMIT_MASK,
    decision_allowed,
    reason_names,
)
from quant_fund.pretrade.config import (
    PretradeConfig,
    load_pretrade_config,
    sign_config,
)
from quant_fund.pretrade.engine import (
    Decision,
    OrderView,
    PretradeEngine,
)

__all__ = [
    "HARD_LIMIT_MASK",
    "Decision",
    "OrderView",
    "PretradeConfig",
    "PretradeEngine",
    "ShadowRiskAdapter",
    "business_session_id",
    "decision_allowed",
    "load_pretrade_config",
    "reason_names",
    "sign_config",
]

_SHADOW = "quant_fund.pretrade.shadow"
_EXPORTS: dict[str, str] = {
    "ShadowRiskAdapter": _SHADOW,
    "business_session_id": _SHADOW,
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
