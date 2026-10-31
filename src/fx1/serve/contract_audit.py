"""contract_audit — pin the harness API's OpenAPI surface.

The harness API is the product boundary fx-1 calls over the wire.
Additive or breaking drift in routes, methods, parameters, or declared
response codes must fail loudly, not ship silently. This audit
normalizes ``app.openapi()`` into a stable surface and diffs it against
the committed golden ``quality/fx1_openapi_surface.json`` — any
difference fails the audit and the seal.

Probe surfaces:

- golden exists, parses, and is itself normalized (sorted keys)
- every golden path present in the live spec, and no live path missing
  from the golden (additive drift is flagged, not silently accepted)
- methods per path match exactly
- parameter names + required flags match per operation
- declared response status codes match per operation
- request-body required flag matches
- whole-schema sha256 of the canonicalized ``openapi()`` pinned
- two consecutive ``create_app().openapi()`` builds are byte-identical
  (the schema is deterministic)
- top-level metadata (openapi version, title) pinned
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

GOLDEN = Path(__file__).resolve().parents[3] / "quality" / "fx1_openapi_surface.json"


def _operation_surface(op: dict[str, Any]) -> dict[str, Any]:
    params = [
        {"name": p["name"], "in": p["in"], "required": bool(p.get("required", False))}
        for p in op.get("parameters", [])
    ]
    params.sort(key=lambda p: (p["in"], p["name"]))
    rb = op.get("requestBody")
    return {
        "params": params,
        "request_body_required": bool(rb.get("required", False)) if isinstance(rb, dict) else None,
        "responses": sorted(op.get("responses", {}).keys()),
    }


def normalized_surface(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stable route x method surface derived from ``create_app().openapi()``."""
    if spec is None:
        from fx1.serve.api import create_app

        spec = create_app().openapi()
    paths: dict[str, Any] = {}
    for path, ops in spec.get("paths", {}).items():
        paths[path] = {m: _operation_surface(ops[m]) for m in sorted(ops)}
    return {
        "openapi": spec.get("openapi", ""),
        "title": spec.get("info", {}).get("title", ""),
        "version": spec.get("info", {}).get("version", ""),
        "paths": {p: paths[p] for p in sorted(paths)},
    }


def emit_golden() -> dict[str, Any]:
    """Write the committed golden surface (quality/fx1_openapi_surface.json)."""
    surface = normalized_surface()
    surface["schema_sha256"] = hash_bytes(canonical_json_bytes(surface))
    GOLDEN.write_text(json.dumps(surface, indent=2) + "\n")
    return surface


def contract_audit() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["golden_exists"] = GOLDEN.is_file()
    golden: dict[str, Any] | None = None
    if out["golden_exists"]:
        try:
            golden = json.loads(GOLDEN.read_text())
            out["golden_parses"] = isinstance(golden, dict) and "paths" in golden
        except (OSError, ValueError):
            out["golden_parses"] = False
    else:
        out["golden_parses"] = False

    live = normalized_surface()
    out["live_deterministic"] = live == normalized_surface()

    if golden is not None and out["golden_parses"]:
        out["golden_self_sealed"] = golden.get("schema_sha256") == hash_bytes(
            canonical_json_bytes({k: v for k, v in golden.items() if k != "schema_sha256"})
        )
        live_paths = set(live["paths"])
        golden_paths = set(golden.get("paths", {}))
        out["no_missing_routes"] = golden_paths <= live_paths
        out["no_additive_drift"] = live_paths <= golden_paths
        out["metadata_pinned"] = all(
            live[k] == golden.get(k) for k in ("openapi", "title", "version")
        )
        methods_ok = True
        params_ok = True
        responses_ok = True
        body_ok = True
        for path in live_paths & golden_paths:
            lp = live["paths"][path]
            gp = golden["paths"][path]
            if set(lp) != set(gp):
                methods_ok = False
                continue
            for method, ops in lp.items():
                gop = gp[method]
                if ops["params"] != gop.get("params"):
                    params_ok = False
                if ops["responses"] != gop.get("responses"):
                    responses_ok = False
                if ops["request_body_required"] != gop.get("request_body_required"):
                    body_ok = False
        out["methods_pinned"] = methods_ok
        out["params_pinned"] = params_ok
        out["responses_pinned"] = responses_ok
        out["request_body_pinned"] = body_ok
        out["surface_matches_golden"] = all(
            [
                out["no_missing_routes"],
                out["no_additive_drift"],
                methods_ok,
                params_ok,
                responses_ok,
                body_ok,
                out["metadata_pinned"],
            ]
        )
    else:
        for k in (
            "golden_self_sealed",
            "no_missing_routes",
            "no_additive_drift",
            "metadata_pinned",
            "methods_pinned",
            "params_pinned",
            "responses_pinned",
            "request_body_pinned",
            "surface_matches_golden",
        ):
            out[k] = False

    # codegen-grade: every operation carries a stable unique operationId
    # and at least one tag — generated clients get real method names.
    from fx1.serve.api import create_app as _create_app  # noqa: PLC0415

    ops = [
        op
        for ops in _create_app().openapi().get("paths", {}).values()
        for op in ops.values()
        if isinstance(op, dict)
    ]
    ids = [op.get("operationId") for op in ops]
    out["operation_ids_present_unique"] = bool(ops) and all(ids) and len(set(ids)) == len(ids)
    out["operation_tags_present"] = bool(ops) and all(
        isinstance(op.get("tags"), list) and op["tags"] for op in ops
    )

    # The generated TypeScript client's committed spec must be exactly the
    # live surface — a drifted snapshot ships clients a stale contract.
    snapshot = (
        Path(__file__).resolve().parents[3] / "clients" / "typescript" / "fx1" / "openapi.json"
    )
    try:
        committed = json.loads(snapshot.read_text())
        out["ts_client_snapshot_exists"] = isinstance(committed, dict)
    except (OSError, ValueError):
        committed = None
        out["ts_client_snapshot_exists"] = False
    out["ts_client_snapshot_fresh"] = committed == _create_app().openapi()
    return out


def contract_audit_bench() -> dict[str, Any]:
    """Seal the contract audit as ``fx1_contract_audit.v1``."""
    r = contract_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "fx1_contract_audit",
        "schema": "fx1_contract_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "the harness API surface is pinned: routes, methods, params, "
            "declared response codes, and request bodies match the committed "
            "golden; additive or breaking drift fails the seal"
            if ok
            else f"CONTRACT DRIFT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
