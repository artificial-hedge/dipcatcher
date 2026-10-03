"""Direct lookup of first-party, separately loadable fx-1 extensions."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from fx1.extensions.contracts import ExtensionModule
from fx1.extensions.naming import ExtensionKind, module_path


def _owners(kind: ExtensionKind) -> tuple[str, ...]:
    if kind == "skill":
        from fx1.harness import HARNESS_REGISTRY

        return tuple(command.name for command in HARNESS_REGISTRY)
    if kind == "plugin":
        from fx1.data.sources.registry import list_sources

        return tuple(source.name for source in list_sources())
    if kind == "feature":
        from fx1.extensions.feature_catalog import list_feature_definitions

        return tuple(feature.name for feature in list_feature_definitions())
    raise ValueError(f"unknown extension kind {kind!r}")


def list_extensions(kind: ExtensionKind | None = None) -> list[dict[str, object]]:
    """List approved extension modules without importing their large card arrays."""
    kinds: tuple[ExtensionKind, ...] = (
        (kind,)
        if kind is not None
        else (
            "skill",
            "plugin",
            "feature",
        )
    )
    return [
        {
            "schema": "fx1.extension-module/v1",
            "kind": extension_kind,
            "owner": owner,
            "module": module_path(extension_kind, owner),
            "market_evidence": False,
        }
        for extension_kind in kinds
        for owner in _owners(extension_kind)
    ]


def get_extension(kind: ExtensionKind, owner: str) -> ExtensionModule:
    """Load one approved extension module, never an arbitrary import path."""
    if owner not in _owners(kind):
        raise KeyError(f"unknown {kind} extension {owner!r}; known: {list(_owners(kind))}")
    expected_path = module_path(kind, owner)
    imported = import_module(expected_path)
    extension = getattr(imported, "MODULE", None)
    if not isinstance(extension, ExtensionModule):
        raise TypeError(f"{expected_path} does not export an ExtensionModule named MODULE")
    if extension.kind != kind or extension.owner != owner or extension.module != expected_path:
        raise ValueError(f"{expected_path} does not match its registered extension identity")
    return extension


def extension_manifest(kind: ExtensionKind, owner: str) -> dict[str, object]:
    """Read the exact module manifest plus its concrete binding metadata."""
    extension = get_extension(kind, owner)
    manifest = extension.manifest()
    if kind == "skill":
        command = cast(Any, extension).command()
        manifest["command"] = command.name
        manifest["role"] = command.role.value
        manifest["description"] = command.description
        manifest["operations"] = ["run"]
    elif kind == "plugin":
        spec = cast(Any, extension).adapter().spec
        manifest["source"] = spec.name
        manifest["display"] = spec.display
        manifest["markets"] = spec.markets
        manifest["assets"] = spec.assets
        manifest["availability"] = cast(Any, extension).probe().model_dump(mode="json")
        manifest["operations"] = ["probe", "describe", "fetch"]
    else:
        metadata = cast(Any, extension).metadata()
        manifest["feature"] = metadata.name
        manifest["feature_family"] = metadata.family
        manifest["lookback"] = metadata.lookback
        manifest["point_in_time_safe"] = metadata.point_in_time_safe
        manifest["synthetic_only"] = metadata.synthetic_only
        manifest["operations"] = ["metadata", "build"]
    return manifest
