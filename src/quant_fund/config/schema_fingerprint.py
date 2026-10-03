"""Schema-level config provenance: fingerprint + compat checking.

``config.impact`` answers "which receipts does this *value* change
void?". This module answers the schema-level question: "was this
receipt measured under the same config *schema* we have now, and would
its declared config still load?"

``model_schema_tree`` reduces a pydantic model tree to a canonical
dict — dotted field paths with {type, required, has_default} leaves —
and ``schema_fingerprint`` pins it to a sha256 a receipt can record as
``config_schema_sha256``. When the schema later evolves (field added,
renamed, required-ness or type changed), every receipt pinned to the
old digest is detectably stale without re-running anything.

``compat_report`` checks a config dict against a model tree:
extra keys, missing required fields, and leaf type drift (where the
declared value can't coerce to the field's annotation) — each named
with its dotted path so an operator sees exactly what breaks.
"""

from __future__ import annotations

import types
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _type_name(annotation: Any) -> str:
    """Stable type label for a field annotation (unions/enums reduced)."""
    if annotation is None:
        return "none"
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return "union[" + ",".join(sorted(_type_name(a) for a in get_args(annotation))) + "]"
    if origin in (list, dict, tuple, set, frozenset):
        args = ",".join(_type_name(a) for a in get_args(annotation)) or "any"
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def model_schema_tree(model: type[BaseModel]) -> dict[str, Any]:
    """Canonical {dotted.path: leaf-spec} tree for a pydantic model."""
    tree: dict[str, Any] = {}

    def walk(cls: type[BaseModel], prefix: str) -> None:
        for name, field in cls.model_fields.items():
            path = f"{prefix}{name}"
            ann = field.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                walk(ann, path + ".")
                continue
            entry: dict[str, Any] = {
                "type": _type_name(ann),
                "required": field.is_required(),
                "has_default": not field.is_required(),
            }
            tree[path] = entry

    walk(model, "")
    return tree


def schema_fingerprint(model: type[BaseModel]) -> str:
    """sha256 of the canonical schema tree — pin this into a receipt."""
    return hash_bytes(canonical_json_bytes(model_schema_tree(model)))


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in config.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, path + "."))
        else:
            out[path] = v
    return out


def _type_compatible(value: Any, type_label: str) -> bool:
    """Consistent coercion check mirroring pydantic's lax mode."""
    if "float" in type_label and isinstance(value, int):
        return True  # pydantic coerces int -> float
    if type_label.startswith("int") and isinstance(value, bool):
        return False
    py: dict[str, Any] = {"int": int, "float": (int, float), "str": str, "bool": bool}
    for prim, cls in py.items():
        if type_label.startswith(prim):
            return isinstance(value, cls)
    return True  # enums/unions/lists: leave detailed checks to validation


def compat_report(
    config: dict[str, Any] | Path,
    model: type[BaseModel],
) -> dict[str, Any]:
    """Check a config mapping (or YAML path) against a model tree."""
    if isinstance(config, Path):
        config = yaml.safe_load(config.read_text()) or {}
    flat = _flatten(dict(config))
    tree = model_schema_tree(model)
    extra = sorted(k for k in flat if k not in tree)
    missing = sorted(k for k, spec in tree.items() if spec["required"] and k not in flat)
    mismatched = sorted(
        k for k in flat if k in tree and not _type_compatible(flat[k], tree[k]["type"])
    )
    verdict = (
        "loads_cleanly"
        if not extra and not missing and not mismatched
        else "extra_keys"
        if not missing and not mismatched
        else "missing_required"
        if missing
        else "type_drift"
    )
    return {
        "schema_fingerprint": schema_fingerprint(model),
        "n_config_keys": len(flat),
        "n_schema_keys": len(tree),
        "extra_keys": extra,
        "missing_required": missing,
        "type_mismatches": mismatched,
        "verdict": verdict,
    }


def schema_fingerprint_bench() -> dict[str, Any]:
    """Fingerprint ``AppConfig`` and compat-check a planted legacy config."""
    from quant_fund.config.models import AppConfig

    fp = schema_fingerprint(AppConfig)
    tree = model_schema_tree(AppConfig)
    # plant a "legacy" config: a few valid keys + one retired + one drifted
    valid_keys = [k for k, s in tree.items() if not s["required"]][:4]
    legacy: dict[str, Any] = {k.split(".")[0]: {} for k in valid_keys}
    for k in valid_keys:
        legacy.setdefault(k.split(".")[0], {})[k.split(".", 1)[1]] = 1
    legacy["retired_section"] = {"old_key": True}
    legacy.setdefault("costs", {})["half_spread_bps"] = "two bucks"
    report = compat_report(legacy, AppConfig)
    interpretation = {
        "schema_fingerprint": fp,
        "n_schema_keys": len(tree),
        "legacy_report": {
            "verdict": report["verdict"],
            "extra_keys": report["extra_keys"],
            "type_mismatches": report["type_mismatches"],
            "missing_required": len(report["missing_required"]),
        },
    }
    payload: dict[str, Any] = {
        "kind": "schema_fingerprint",
        "schema": "schema_fingerprint.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "schema drift is detectable from the fingerprint + compat report",
            "verdict": "ok"
            if report["verdict"] != "loads_cleanly" and report["extra_keys"]
            else "degenerate",
        },
        "interpretation": interpretation,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
