"""contract_probe — which committed receipts does the verifier deep-check?

A sha256 seal proves a receipt was not tampered with *after* writing; lane
contracts go further and re-derive claims inside the payload so numbers
fabricated before sealing still fail. The dispatch web in ``receipt_v2``
(fleet_eval / vol_bench / quantile_ladder / evalue / capacity / rankic /
lane_contracts / script_receipts / hedge_lab / data manifest / hstep /
calibration_eval / cost_calibration) is intricate enough that the honest way
to measure coverage is empirical: forge a claim, reseal honestly, verify.

Per committed receipt this module applies up to three fabrication probes:

- ``numeric``: the first numeric leaf in the sealed body (outside identity /
  environment / digest fields) is inflated — catches contracts that re-derive
  or bound magnitudes.
- ``boolean``: the first non-honesty bool leaf is flipped — catches
  verdict/flag contracts.
- ``truncate``: the last element of the first non-empty list field is
  dropped — catches row-count / completeness contracts.

A receipt is ``contracted`` when at least one forged reseal is rejected and
``envelope_only`` when every probe verifies clean — meaning its claims are
self-attested under the seal alone. The classification is a *probe-relative*
lower bound: an envelope_only verdict means these specific fabrications
escaped, not that no fabrication would.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.schemas.receipt import seal_receipt, verify_receipt_payload
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["contract_probe_bench", "probe_receipt", "scan_corpus"]

#: Leaves whose mutation attests to envelope fields, not lane claims —
#: excluded so probes measure deep-contract coverage rather than the
#: envelope's own digest/env checks.
_SKIP_KEYS = frozenset(
    {
        "receipt_sha256",
        "dataset_hash",
        "params_hash",
        "config_hash",
        "code_sha256",
        "git_revision",
        "sha256",
        "n",
        "seed",
    }
)
_SKIP_SUBTREES = frozenset({"environment", "code_files", "env_fingerprint"})
_HONESTY_KEYS = frozenset({"live_pnl_claim", "research_only", "simulated_only"})


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _walk_leaves(node: object, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], Any]]:
    """Deterministic leaf enumeration: dict keys sorted, lists in order."""
    if path and path[0] in _SKIP_SUBTREES:
        return []
    if isinstance(node, Mapping):
        leaves: list[tuple[tuple[str, ...], Any]] = []
        for key in sorted(node):
            if path == () and str(key) in _SKIP_KEYS | _SKIP_SUBTREES:
                continue
            leaves.extend(_walk_leaves(node[key], path + (str(key),)))
        return leaves
    if isinstance(node, list):
        leaves = []
        for index, item in enumerate(node):
            leaves.extend(_walk_leaves(item, path + (f"[{index}]",)))
        return leaves
    return [(path, node)]


def _set_path(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Return a deep copy of ``payload`` with the leaf at ``path`` replaced."""
    import copy

    body = copy.deepcopy(payload)
    node: Any = body
    for part in path[:-1]:
        if part.startswith("[") and part.endswith("]"):
            node = node[int(part[1:-1])]
        else:
            node = node[part]
    last = path[-1]
    if last.startswith("[") and last.endswith("]"):
        node[int(last[1:-1])] = value
    else:
        node[last] = value
    return body


def _drop_last(payload: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    import copy

    body = copy.deepcopy(payload)
    node: Any = body
    for part in path:
        if part.startswith("[") and part.endswith("]"):
            node = node[int(part[1:-1])]
        else:
            node = node[part]
    node.pop()
    return body


def _list_fields(node: object, path: tuple[str, ...]) -> list[tuple[str, ...]]:
    if path and path[0] in _SKIP_SUBTREES:
        return []
    found: list[tuple[str, ...]] = []
    if isinstance(node, Mapping):
        for key in sorted(node):
            child = node[key]
            if isinstance(child, list) and child:
                found.append(path + (str(key),))
            else:
                found.extend(_list_fields(child, path + (str(key),)))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_list_fields(item, path + (f"[{index}]",)))
    return found


def probe_receipt(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Forge-and-reseal probes against one parsed receipt body.

    Returns the per-receipt row: identity, which probes ran, which escaped,
    and the ``contracted``/``envelope_only`` classification.
    """
    leaves = _walk_leaves(payload, ())
    probes: dict[str, dict[str, Any]] = {}

    numeric_leaf = next(
        (
            (p, v)
            for p, v in leaves
            if _is_number(v) and p and not any(part.endswith("_sha256") for part in p)
        ),
        None,
    )
    if numeric_leaf is not None:
        leaf_path, value = numeric_leaf
        forged = seal_receipt(
            _set_path(dict(payload), leaf_path, float(value) + max(1.0, abs(float(value))))
        )
        verdict = verify_receipt_payload(forged, path)
        probes["numeric"] = {
            "leaf": ".".join(leaf_path),
            "escaped": bool(verdict["valid"]),
        }

    bool_leaf = next(
        ((p, v) for p, v in leaves if isinstance(v, bool) and p[-1] not in _HONESTY_KEYS),
        None,
    )
    if bool_leaf is not None:
        leaf_path, value = bool_leaf
        forged = seal_receipt(_set_path(dict(payload), leaf_path, not value))
        verdict = verify_receipt_payload(forged, path)
        probes["boolean"] = {
            "leaf": ".".join(leaf_path),
            "escaped": bool(verdict["valid"]),
        }

    list_fields = _list_fields(payload, ())
    if list_fields:
        list_path = list_fields[0]
        forged = seal_receipt(_drop_last(dict(payload), list_path))
        verdict = verify_receipt_payload(forged, path)
        probes["truncate"] = {
            "leaf": ".".join(list_path),
            "escaped": bool(verdict["valid"]),
        }

    contracted = bool(probes) and any(not p["escaped"] for p in probes.values())
    return {
        "file": path.name,
        "kind": payload.get("kind"),
        "schema": payload.get("schema", payload.get("schema_version")),
        "probes": probes,
        "coverage": "contracted" if contracted else "envelope_only",
    }


def scan_corpus(receipts_dir: Path) -> list[dict[str, Any]]:
    """Probe every committed ``receipts/*.json`` (sorted for determinism)."""
    import json

    rows: list[dict[str, Any]] = []
    for path in sorted(receipts_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            rows.append(
                {
                    "file": path.name,
                    "kind": None,
                    "schema": None,
                    "probes": {},
                    "coverage": "unparseable",
                }
            )
            continue
        if not isinstance(payload, Mapping) or "receipt_sha256" not in payload:
            rows.append(
                {
                    "file": path.name,
                    "kind": payload.get("kind") if isinstance(payload, Mapping) else None,
                    "schema": None,
                    "probes": {},
                    "coverage": "unsealed",
                }
            )
            continue
        rows.append(probe_receipt(payload, path))
    return rows


def contract_probe_bench(receipts_dir: Path | None = None) -> dict[str, Any]:
    """Coverage map over the committed corpus, sealed as ``contract_probe.v1``."""
    root = receipts_dir or Path(__file__).resolve().parents[3] / "receipts"
    rows = scan_corpus(root)
    n = len(rows)
    n_contracted = sum(1 for r in rows if r["coverage"] == "contracted")
    per_kind: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row["kind"] or row["schema"] or "?")
        cell = per_kind.setdefault(key, {"contracted": 0, "envelope_only": 0, "other": 0})
        cell_key = (
            row["coverage"] if row["coverage"] in ("contracted", "envelope_only") else "other"
        )
        cell[cell_key] += 1
    envelope_only = sorted(r["file"] for r in rows if r["coverage"] == "envelope_only")
    payload: dict[str, Any] = {
        "kind": "contract_probe",
        "schema": "contract_probe.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "deep-verification coverage measured empirically via forge-and-reseal probes",
            "n_receipts": n,
            "n_contracted": n_contracted,
            "n_envelope_only": len(envelope_only),
            "per_kind": per_kind,
            "envelope_only_files": envelope_only,
        },
        "interpretation": (
            f"{n_contracted}/{n} committed receipts reject at least one "
            "fabricated-claim probe (deep contract present); "
            f"{len(envelope_only)} verify clean under every probe — their "
            "claims are self-attested under the seal alone. Coverage is "
            "probe-relative: envelope_only bounds, not measures, the gap."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
