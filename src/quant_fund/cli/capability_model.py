"""Shared capability-contract types and portable value helpers.

No research registry or lab CLI import. Document builders re-export the
public names from ``quant_fund.cli.capability_cmds``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*( [a-z0-9][a-z0-9-]*)*$")

_ARG_ORDER = (
    "name",
    "kind",
    "type",
    "required",
    "default",
    "repeatable",
    "help",
    "choices",
    "min",
    "max",
    "item_type",
    "path_role",
)


@dataclass
class LabCommand:
    """One invocable lab command, already introspected."""

    argv: tuple[str, ...]
    help_text: str
    hidden: bool
    args: tuple[dict[str, Any], ...]

    @property
    def id(self) -> str:
        return " ".join(self.argv)


def find_repo_root(start: Path | None = None) -> Path | None:
    """Return the checkout root, or None when this file is not inside one."""
    here = (start or Path(__file__)).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "quant_fund").is_dir():
            return parent
    return None


def dumps_document(doc: Mapping[str, Any]) -> str:
    """Serialize one document with stable key order and a trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _nonportable(text: str) -> bool:
    if "://" in text:
        return True
    normalized = text.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    return len(normalized) >= 3 and normalized[0].isalpha() and normalized[1:3] == ":/"


def _order_arg(arg: Mapping[str, Any]) -> dict[str, Any]:
    return {key: arg[key] for key in _ARG_ORDER if key in arg}


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_timeout(value: object) -> bool:
    if value is None:
        return True
    return _nonnegative_int(value)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _number_or_none(value: object) -> int | float | None:
    if not _is_number(value):
        return None
    return _as_number(value)


def _as_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a number")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
