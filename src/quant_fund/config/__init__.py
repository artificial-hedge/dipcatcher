"""Typed config package.

Public names load on first use. Importing a submodule such as
``quant_fund.config.yaml_safe`` must not pull ``numpy`` through this
package, because the CLI cold-start budget forbids heavy libraries.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from quant_fund.config.loader import dump_resolved as dump_resolved
    from quant_fund.config.loader import load_config as load_config
    from quant_fund.config.models import AppConfig as AppConfig
    from quant_fund.config.models import RuntimeMode as RuntimeMode

__all__ = ["AppConfig", "RuntimeMode", "dump_resolved", "load_config"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AppConfig": ("quant_fund.config.models", "AppConfig"),
    "RuntimeMode": ("quant_fund.config.models", "RuntimeMode"),
    "dump_resolved": ("quant_fund.config.loader", "dump_resolved"),
    "load_config": ("quant_fund.config.loader", "load_config"),
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    value = cast(object, getattr(importlib.import_module(module_name), attr))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
