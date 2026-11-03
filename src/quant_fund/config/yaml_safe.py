"""YAML documents through PyYAML's safe loader.

Arbitrary tags, parser errors, and nesting that overflows the interpreter
become ``ValueError``. Callers never receive a constructed object.
"""

from __future__ import annotations

from typing import Any

import yaml

_MAX_YAML_BYTES = 1_048_576


def load_yaml_document(text: str) -> Any:
    """Parse ``text`` with ``yaml.safe_load`` and fail closed."""
    if not isinstance(text, str):
        raise ValueError("YAML document must be text")
    if len(text.encode("utf-8")) > _MAX_YAML_BYTES:
        raise ValueError("YAML document exceeds 1 MiB")
    try:
        return yaml.safe_load(text)
    except (yaml.YAMLError, RecursionError, ValueError) as exc:
        raise ValueError("YAML document is invalid") from exc
