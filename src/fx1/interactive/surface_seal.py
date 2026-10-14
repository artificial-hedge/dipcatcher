"""Sealed digest of the interactive orchestration surface.

Forks that strip harness commands, add unregistered lab surfaces, or rewrite
/superpower planning triggers drift this digest. The seal is recomputed from
live registries — no secrets — and is meant to be checked by ``doctor`` and
fx-1 gate tests so tampering fails closed before a model session starts.

Provenance only; not a market or performance claim.
"""

from __future__ import annotations

import json
from typing import Any

from fx1.harness import HARNESS_REGISTRY
from fx1.interactive.workflow import WORKFLOW_SCHEMA
from quant_fund.utils.hashing import hash_bytes

SURFACE_SEAL_SCHEMA = "interactive_surface.v1"


def _canonical_harness_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in HARNESS_REGISTRY:
        rows.append(
            {
                "name": command.name,
                "role": command.role.value,
                "timeout_s": command.timeout_s,
                "argv": list(command.argv),
                "description": command.description,
            }
        )
    rows.sort(key=lambda row: row["name"])
    return rows


def surface_payload(
    *,
    capabilities: dict[str, dict[str, Any]],
    triggers: dict[str, tuple[str, ...]],
    web_triggers: tuple[str, ...],
) -> dict[str, Any]:
    cap_rows = [
        {
            "id": cap_id,
            "kind": meta.get("kind"),
            "role": meta.get("role"),
            "consequential": bool(meta.get("consequential")),
            "timeout_s": meta.get("timeout_s"),
            "description": meta.get("description"),
        }
        for cap_id, meta in sorted(capabilities.items())
    ]
    trigger_rows = [
        {"cap_id": cap_id, "terms": list(terms)} for cap_id, terms in sorted(triggers.items())
    ]
    return {
        "schema": SURFACE_SEAL_SCHEMA,
        "workflow": WORKFLOW_SCHEMA,
        "harness": _canonical_harness_rows(),
        "capabilities": cap_rows,
        "triggers": trigger_rows,
        "web_triggers": list(web_triggers),
    }


def surface_digest(payload: dict[str, Any] | None = None, *, raw: str | None = None) -> str:
    if raw is None:
        if payload is None:
            raise ValueError("payload or raw required")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hash_bytes(raw.encode("utf-8"))


def build_surface_seal(
    *,
    capabilities: dict[str, dict[str, Any]],
    triggers: dict[str, tuple[str, ...]],
    web_triggers: tuple[str, ...],
) -> dict[str, Any]:
    payload = surface_payload(
        capabilities=capabilities,
        triggers=triggers,
        web_triggers=web_triggers,
    )
    digest = surface_digest(payload)
    return {"schema": SURFACE_SEAL_SCHEMA, "digest": digest, "workflow": WORKFLOW_SCHEMA}
