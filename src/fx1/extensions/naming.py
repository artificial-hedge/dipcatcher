"""Stable names for the first-party fx-1 extension modules."""

from __future__ import annotations

import re
from typing import Literal

ExtensionKind = Literal["skill", "plugin", "feature"]

_DIRECTORY_BY_KIND: dict[ExtensionKind, str] = {
    "skill": "skills",
    "plugin": "plugins",
    "feature": "features",
}


def module_basename(owner: str) -> str:
    """Return the safe Python filename stem for a registered owner id."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", owner):
        raise ValueError(f"unsafe extension owner {owner!r}")
    return owner.replace("-", "_")


def module_path(kind: ExtensionKind, owner: str) -> str:
    """Return the only import path permitted for a registered extension."""
    try:
        directory = _DIRECTORY_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"unknown extension kind {kind!r}") from exc
    return f"fx1.extensions.{directory}.{module_basename(owner)}"


def module_directory(kind: ExtensionKind) -> str:
    """Return the package directory for an extension kind."""
    try:
        return _DIRECTORY_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"unknown extension kind {kind!r}") from exc
