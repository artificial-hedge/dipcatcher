"""Load YAML configs with inheritance and dump the resolved object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from quant_fund.config.models import AppConfig


class _ConfigLoader(yaml.SafeLoader):
    def __init__(self, stream: str, path: Path) -> None:
        super().__init__(stream)
        self.config_path = path


def _construct_unique_mapping(
    loader: _ConfigLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = (
            key_node.value
            if key_node.tag == "tag:yaml.org,2002:merge"
            else loader.construct_object(key_node, deep=deep)
        )
        try:
            hash(key)
        except TypeError:
            continue
        if key in seen:
            raise ValueError(
                f"Config {loader.config_path}: duplicate mapping key "
                f"{key!r} at line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_ConfigLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay onto base. Overlay wins on conflicts."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if len(text.encode("utf-8")) > 1_048_576:
        raise ValueError(f"Config {path} is not valid YAML")
    loader = _ConfigLoader(text, path)
    try:
        try:
            raw = loader.get_single_data()
        except RecursionError as exc:
            raise ValueError(f"Config {path} is not valid YAML") from exc
    finally:
        loader.dispose()
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return raw


def load_config(
    path: str | Path,
    *,
    _seen: frozenset[Path] | None = None,
    _root: Path | None = None,
) -> AppConfig:
    """Load a YAML file, following `inherit:` relative to the file's directory."""
    path = Path(path).resolve()
    root = path.parent if _root is None else _root
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Config inheritance escapes config root: {path}") from exc
    seen = _seen or frozenset()
    if path in seen:
        raise ValueError(f"Config inheritance cycle at {path}")
    data = _read_yaml(path)
    inherit = data.pop("inherit", None)
    if inherit is not None:
        if not isinstance(inherit, str) or not inherit.strip():
            raise ValueError(f"Config {path}: inherit must be null or a non-empty string")
        parent_path = (path.parent / inherit.strip()).resolve()
        parent = load_config(parent_path, _seen=seen | {path}, _root=root)
        merged = deep_merge(parent.model_dump(mode="python"), data)
        return AppConfig.model_validate(merged)
    return AppConfig.model_validate(data)


def dump_resolved(config: AppConfig, dest: Path) -> None:
    from quant_fund.utils.atomicio import atomic_write_text

    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(dest, json.dumps(config.dump(), indent=2, sort_keys=True))
