"""Provenance citation-graph audit over a committed receipt corpus.

Every sealed receipt commits digests: ``inputs_sha256`` over its measured
data, ``code_files``/``code_sha256`` over the writer source, per-row file
digests in audit lanes. Some of those digests are *citations* — a 64-hex
value naming another corpus member's file bytes or sealed
``receipt_sha256`` — and filename fields (``prev_epoch_receipt``,
``corpus_epoch_receipt``) name member files outright. ``verify-receipt``
proves each artifact individually; nothing proves the corpus's *citation
graph* is coherent: a receipt may cite a member that was never committed
(dangling), filename references may cycle, and a member may sit orphaned
from every chain.

Procedure:

- Read every ``*.json`` member — recursively, matching the epoch chain's
  member semantics and quarantined subdirs excluded (a nested member is
  still corpus evidence, not a blind spot). Index two identities per
  member: the sha256 of its file bytes and its sealed ``receipt_sha256``
  (absent for unsealed or unreadable files).
- Walk each receipt body — the whole document for v1, and for ``receipt.v2``
  envelopes both the envelope provenance (``code_files``) and the wrapped
  ``payload`` — harvesting two reference types: *digest refs* (fields whose
  name ends ``sha256``/``digest`` holding a 64-hex value) and *filename
  refs* (fields whose name ends ``receipt``/``receipts`` holding ``.json``
  names, plus ``code_files`` paths). A receipt's own ``receipt_sha256``
  seal is its identity, never a citation, so it is skipped.
- Resolve each reference: digests against member file-bytes and sealed
  digests, filenames against member names. Edges record the citing file,
  the dotted field path, and the target member (or ``null``).
- Classify: *dangling* = an unresolved digest ref whose field name implies
  corpus membership (``prev_*``, ``prior_*``, ``cited_*``, ``parent_*``,
  ``next_*``, ``child_*``, or any ``*receipt*`` digest field — e.g.
  ``prev_epoch_sha256`` pairs with ``prev_epoch_receipt``); *unresolvable*
  = a ``*receipt`` filename ref naming no member; *cycles* = strongly
  connected components of the member→member filename graph (digest refs
  cannot cycle — content-addressed values name fixed content);
  *orphans* = members referenced by nothing and referencing nothing
  (informational; corpus-meta kinds are exempt).

The output is itself a sealed ``receipt_graph.v1`` receipt so the audit is
verifiable by the same contract machinery it extends. Structural only —
no statistical claims.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import SHA256_HEX_LENGTH, canonical_json_bytes, hash_bytes
from quant_fund.utils.receipt import verified_corpus_files
from quant_fund.utils.reproducibility import git_revision

GRAPH_SCHEMA = "receipt_graph.v1"

# Field-name prefixes that imply a digest is meant to name a corpus member.
_MEMBERSHIP_PREFIXES = ("prev_", "prior_", "cited_", "parent_", "next_", "child_")

# Corpus-meta audit kinds exempt from orphan classification: they describe
# the corpus itself, so standing alone carries no provenance signal.
_EPOCH_ADJACENT_KINDS = frozenset({"corpus_epoch.v1", "receipt_lattice.v1", GRAPH_SCHEMA})

# A member's own seal is its identity, not a citation of another member.
_SEAL_PATHS = frozenset({"receipt_sha256", "payload.receipt_sha256"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


def _field_name(field_path: str) -> str:
    """Last mapping-key segment of a dotted path (``a.b[0].c`` -> ``c``)."""
    return field_path.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def _implies_membership(field_name: str) -> bool:
    """Whether a digest field's name claims the value names a corpus member."""
    if field_name.startswith(_MEMBERSHIP_PREFIXES):
        return True
    return any(t in ("receipt", "receipts") for t in field_name.split("_"))


def _is_filename_ref_name(field_name: str) -> bool:
    return field_name.endswith("receipt") or field_name.endswith("receipts")


def _walk_fields(node: Any, path: str) -> list[tuple[str, str, Any]]:
    """Scalar leaves of a receipt body as ``(field_path, field_name, value)``."""
    out: list[tuple[str, str, Any]] = []
    if isinstance(node, Mapping):
        for key, val in node.items():
            child = f"{path}.{key}" if path else str(key)
            out.extend(_walk_fields(val, child))
    elif isinstance(node, list):
        for i, val in enumerate(node):
            child = f"{path}[{i}]"
            name = _field_name(path)
            if name.endswith("receipts") and isinstance(val, str):
                out.append((child, name, val))
            else:
                out.extend(_walk_fields(val, child))
    else:
        out.append((path, _field_name(path), node))
    return out


def _member_kind(doc: Mapping[str, Any]) -> str | None:
    kind = doc.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    inner = doc.get("payload")
    if isinstance(inner, Mapping):
        inner_kind = inner.get("kind") or inner.get("schema")
        if isinstance(inner_kind, str) and inner_kind:
            return inner_kind
    return None


def _member_seal(doc: Mapping[str, Any]) -> str | None:
    seal = doc.get("receipt_sha256")
    return seal if _is_sha256(seal) else None


def _harvest_edges(
    name: str,
    doc: Mapping[str, Any],
    by_file_bytes: dict[str, str],
    by_seal: dict[str, str],
    member_names: set[str],
    unique_basenames: dict[str, str],
) -> list[dict[str, Any]]:
    """Typed reference edges from one member to the rest of the corpus."""
    edges: list[dict[str, Any]] = []

    def resolve_digest(value: str) -> tuple[str | None, str | None]:
        if value in by_seal and by_seal[value] != name:
            return by_seal[value], "receipt_sha256"
        if value in by_file_bytes and by_file_bytes[value] != name:
            return by_file_bytes[value], "file_bytes"
        return None, None

    def resolve_filename(value: str) -> str | None:
        base = value.rsplit("/", 1)[-1]
        for candidate in (value, base):
            if candidate in member_names and candidate != name:
                return candidate
        # a member sitting in a non-quarantined subdir resolves through its
        # basename only when that basename names exactly one member
        if base in unique_basenames and unique_basenames[base] != name:
            return unique_basenames[base]
        return None

    for field_path, field_name, value in _walk_fields(doc, ""):
        if field_path in _SEAL_PATHS:
            continue
        if (
            isinstance(value, str)
            and _is_sha256(value)
            and (field_name.endswith("sha256") or field_name.endswith("digest"))
        ):
            target, via = resolve_digest(value)
            edges.append(
                {
                    "file": name,
                    "field_path": field_path,
                    "ref": "digest",
                    "value": value,
                    "target": target,
                    "target_via": via,
                }
            )
        elif (
            isinstance(value, str) and _is_filename_ref_name(field_name) and value.endswith(".json")
        ):
            target = resolve_filename(value)
            edges.append(
                {
                    "file": name,
                    "field_path": field_path,
                    "ref": "filename",
                    "value": value,
                    "target": target,
                    "target_via": "filename" if target else None,
                }
            )

    # ``code_files`` maps path -> content digest: a path reference whose
    # paired digest is the claimed member identity.
    def code_file_edges(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, val in node.items():
                child = f"{path}.{key}" if path else str(key)
                if key == "code_files" and isinstance(val, Mapping):
                    for ref_path, ref_digest in val.items():
                        if not isinstance(ref_path, str):
                            continue
                        base = ref_path.rsplit("/", 1)[-1]
                        if base.endswith(".json"):
                            target = resolve_filename(ref_path)
                            via = "filename" if target else None
                            if target is None and _is_sha256(ref_digest):
                                target, via = resolve_digest(str(ref_digest))
                            edges.append(
                                {
                                    "file": name,
                                    "field_path": f"{child}.{ref_path}",
                                    "ref": "filename",
                                    "value": ref_path,
                                    "target": target,
                                    "target_via": via,
                                }
                            )
                        elif _is_sha256(ref_digest):
                            target, via = resolve_digest(str(ref_digest))
                            edges.append(
                                {
                                    "file": name,
                                    "field_path": f"{child}.{ref_path}",
                                    "ref": "digest",
                                    "value": str(ref_digest),
                                    "target": target,
                                    "target_via": via,
                                }
                            )
                else:
                    code_file_edges(val, child)
        elif isinstance(node, list):
            for i, val in enumerate(node):
                code_file_edges(val, f"{path}[{i}]")

    code_file_edges(doc, "")
    edges.sort(key=lambda e: (e["field_path"], e["value"]))
    return edges


def _filename_cycles(adjacency: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    """SCCs of size > 1 plus self-loops in the member->member filename graph."""
    nodes = set(adjacency)
    for targets in adjacency.values():
        nodes |= targets
    reach: dict[str, set[str]] = {}
    for node in nodes:
        seen: set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for nxt in adjacency.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reach[node] = seen
    cycles: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for node in sorted(nodes):
        if node in consumed:
            continue
        scc = {m for m in reach[node] if node in reach[m]}
        if len(scc) > 1:
            cycles.append({"members": sorted(scc)})
            consumed |= scc
        elif node in adjacency.get(node, set()):
            cycles.append({"members": [node]})
            consumed.add(node)
    return cycles


def _adjacency(edges: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        if edge["ref"] == "filename" and edge["target"] is not None:
            adjacency.setdefault(str(edge["file"]), set()).add(str(edge["target"]))
    return adjacency


def _expected_orphans(members: Mapping[str, Any], edges: Sequence[Mapping[str, Any]]) -> list[str]:
    referenced = {e["target"] for e in edges if e["target"] is not None}
    referencing = {e["file"] for e in edges if e["target"] is not None}
    return sorted(
        name
        for name in members
        if name not in referenced
        and name not in referencing
        and members[name].get("kind") not in _EPOCH_ADJACENT_KINDS
    )


def _expected_dangling(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"file": str(e["file"]), "field_path": str(e["field_path"]), "value": str(e["value"])}
        for e in edges
        if e["ref"] == "digest"
        and e["target"] is None
        and _implies_membership(_field_name(str(e["field_path"])))
    ]


def _expected_unresolvable(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"file": str(e["file"]), "field_path": str(e["field_path"]), "value": str(e["value"])}
        for e in edges
        if e["ref"] == "filename"
        and e["target"] is None
        and _is_filename_ref_name(_field_name(str(e["field_path"])))
    ]


def _verdict(n_parse_errors: int, n_dangling: int, n_unresolvable: int, n_cycles: int) -> str:
    if n_parse_errors:
        return "partially_unreadable"
    if n_dangling or n_unresolvable:
        return "dangling"
    if n_cycles:
        return "cyclic"
    return "clean"


def receipt_graph(
    corpus_dir: Path | str,
    *,
    glob: str = "*.json",
) -> dict[str, Any]:
    """Audit a receipt corpus's provenance citation graph.

    Returns a ``receipt_graph.v1`` receipt dict. Fails closed on a missing
    directory; unreadable members are recorded and downgrade the verdict to
    at most ``"partially_unreadable"`` while still serving as edge targets
    (their file-bytes digest is computable).
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"corpus dir {root} does not exist")

    member_files = verified_corpus_files(root, pattern=glob)
    members: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    docs: dict[str, Mapping[str, Any]] = {}
    for path in member_files:
        rel = path.relative_to(root).as_posix()
        entry: dict[str, Any] = {"file_sha256": hash_bytes(path.read_bytes())}
        try:
            doc = json.loads(path.read_bytes())
            if not isinstance(doc, Mapping):
                raise ValueError("receipt root is not an object")
        except (OSError, ValueError) as exc:
            # Narrowed from `except Exception` (quality ratchet): read_bytes
            # faults are OSError; non-object roots are the ValueError raised
            # above. Recorded, never skipped.
            errors.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
            entry["kind"] = None
            entry["receipt_sha256"] = None
        else:
            docs[rel] = doc
            entry["kind"] = _member_kind(doc)
            entry["receipt_sha256"] = _member_seal(doc)
        members[rel] = entry

    # Members keyed by rel path: a basename resolves a nested member only
    # when it names exactly one.
    basename_count: dict[str, int] = {}
    for member_name in members:
        base = member_name.rsplit("/", 1)[-1]
        basename_count[base] = basename_count.get(base, 0) + 1
    unique_basenames = {
        member_name.rsplit("/", 1)[-1]: member_name
        for member_name in members
        if basename_count[member_name.rsplit("/", 1)[-1]] == 1
    }

    by_file_bytes = {m["file_sha256"]: name for name, m in members.items()}
    by_seal = {
        m["receipt_sha256"]: name for name, m in members.items() if m["receipt_sha256"] is not None
    }
    member_names = set(members)

    edges: list[dict[str, Any]] = []
    for name in sorted(docs):
        edges.extend(
            _harvest_edges(name, docs[name], by_file_bytes, by_seal, member_names, unique_basenames)
        )

    dangling = _expected_dangling(edges)
    unresolvable = _expected_unresolvable(edges)
    cycles = _filename_cycles(_adjacency(edges))
    orphans = _expected_orphans(members, edges)
    verdict = _verdict(len(errors), len(dangling), len(unresolvable), len(cycles))

    params = {"glob": glob}
    digests = {name: m["file_sha256"] for name, m in members.items()}
    inputs_sha256 = hash_bytes(canonical_json_bytes({"digests": digests, "params": params}))
    return {
        "kind": GRAPH_SCHEMA,
        "schema": GRAPH_SCHEMA,
        "data_label": "CORPUS",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": inputs_sha256,
        "params": params,
        "verdict": verdict,
        "n_members": len(members),
        "members": members,
        "n_parse_errors": len(errors),
        "parse_errors": errors,
        "n_edges": len(edges),
        "edges": edges,
        "n_dangling": len(dangling),
        "dangling": dangling,
        "n_unresolvable": len(unresolvable),
        "unresolvable": unresolvable,
        "n_cycles": len(cycles),
        "cycles": cycles,
        "n_orphans": len(orphans),
        "orphans": orphans,
        "evidence": [
            "digest_reference_resolution",
            "filename_reference_chains",
            "dangling_and_cycle_detection",
        ],
    }


def _member_entry_errors(name: str, entry: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(entry, Mapping):
        return [f"members[{name}]_not_object"]
    if not _is_sha256(entry.get("file_sha256")):
        errors.append(f"members[{name}].file_sha256")
    seal = entry.get("receipt_sha256")
    if seal is not None and not _is_sha256(seal):
        errors.append(f"members[{name}].receipt_sha256")
    kind = entry.get("kind")
    if kind is not None and not isinstance(kind, str):
        errors.append(f"members[{name}].kind")
    return errors


def _edge_errors(edge: object, members: Mapping[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(edge, Mapping):
        return [f"edges[{index}]_not_object"]
    if edge.get("file") not in members:
        errors.append(f"edges[{index}].file")
    if not isinstance(edge.get("field_path"), str) or not str(edge["field_path"]).strip():
        errors.append(f"edges[{index}].field_path")
    ref = edge.get("ref")
    if ref not in ("digest", "filename"):
        errors.append(f"edges[{index}].ref")
    value = edge.get("value")
    if not isinstance(value, str) or not value:
        errors.append(f"edges[{index}].value")
    target = edge.get("target")
    via = edge.get("target_via")
    if target is None:
        if via is not None:
            errors.append(f"edges[{index}].target_via")
    else:
        # Resolved edges must name an existing member.
        if target not in members:
            errors.append(f"edges[{index}].target")
        if ref == "digest" and via not in ("receipt_sha256", "file_bytes"):
            errors.append(f"edges[{index}].target_via")
        if ref == "filename" and via not in ("filename", "file_bytes", "receipt_sha256"):
            errors.append(f"edges[{index}].target_via")
        if ref == "filename" and via == "filename" and isinstance(value, str):
            base = value.rsplit("/", 1)[-1]
            # mirrors resolve_filename: exact member name, member basename,
            # or the unique member carrying that basename in a subdir
            ok = value == target or base == target
            if not ok and isinstance(target, str) and target.rsplit("/", 1)[-1] == base:
                ok = sum(1 for m in members if str(m).rsplit("/", 1)[-1] == base) == 1
            if not ok:
                errors.append(f"edges[{index}].value")
    if ref == "digest" and isinstance(value, str) and not _is_sha256(value):
        errors.append(f"edges[{index}].digest_shape")
    return errors


def graph_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Re-derive the accounting invariants of a ``receipt_graph.v1`` body.

    Beyond shape checks this recomputes the classified lists *from the
    reported edges* — a receipt whose dangling/cycle/orphan claims were
    edited before sealing still fails.
    """
    if payload.get("schema") != GRAPH_SCHEMA and payload.get("kind") != GRAPH_SCHEMA:
        return ["schema"]
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")

    members = payload.get("members")
    if not isinstance(members, Mapping):
        return errors + ["members_missing"]
    for name in sorted(members):
        errors.extend(_member_entry_errors(str(name), members[name]))
    if payload.get("n_members") != len(members):
        errors.append("n_members")

    edges = payload.get("edges")
    if not isinstance(edges, list):
        return errors + ["edges_missing"]
    for index, edge in enumerate(edges):
        errors.extend(_edge_errors(edge, members, index))
    if payload.get("n_edges") != len(edges):
        errors.append("n_edges")

    def _ref_list(key: str) -> list[Mapping[str, Any]]:
        raw = payload.get(key)
        if not isinstance(raw, list):
            errors.append(f"{key}_missing")
            return []
        bad = [
            item
            for item in raw
            if not (
                isinstance(item, Mapping)
                and item.get("file") in members
                and isinstance(item.get("field_path"), str)
                and isinstance(item.get("value"), str)
            )
        ]
        if bad:
            errors.append(f"{key}_malformed")
        return [item for item in raw if isinstance(item, Mapping)]

    dangling = _ref_list("dangling")
    for item in dangling:
        if not _is_sha256(item["value"]):
            errors.append("dangling_digest_shape")
            break
    unresolvable = _ref_list("unresolvable")
    for item in unresolvable:
        value = item["value"]
        base = value.rsplit("/", 1)[-1] if isinstance(value, str) else ""
        resolved = value if value in members else (base if base in members else None)
        if resolved is None and base:
            candidates = [m for m in members if str(m).rsplit("/", 1)[-1] == base]
            if len(candidates) == 1:
                resolved = candidates[0]
        if resolved is not None and resolved != item.get("file"):
            errors.append("unresolvable_names_member")
            break

    sane_edges = [e for e in edges if isinstance(e, Mapping)]
    expected_dangling = _expected_dangling(sane_edges)
    if sorted(json.dumps(d, sort_keys=True) for d in dangling) != sorted(
        json.dumps(d, sort_keys=True) for d in expected_dangling
    ):
        errors.append("dangling")
    expected_unresolvable = _expected_unresolvable(sane_edges)
    if sorted(json.dumps(d, sort_keys=True) for d in unresolvable) != sorted(
        json.dumps(d, sort_keys=True) for d in expected_unresolvable
    ):
        errors.append("unresolvable")

    cycles = payload.get("cycles")
    if not isinstance(cycles, list):
        errors.append("cycles_missing")
        cycles = []
    reported_cycles = set()
    for index, cycle in enumerate(cycles):
        names = cycle.get("members") if isinstance(cycle, Mapping) else None
        if not (
            isinstance(names, list)
            and names
            and all(isinstance(n, str) and n in members for n in names)
            and names == sorted(names)
        ):
            errors.append(f"cycles[{index}].members")
            continue
        reported_cycles.add(frozenset(names))
    expected_cycles = {frozenset(c["members"]) for c in _filename_cycles(_adjacency(sane_edges))}
    if reported_cycles != expected_cycles:
        errors.append("cycles")

    orphans = payload.get("orphans")
    if not isinstance(orphans, list) or not all(
        isinstance(o, str) and o in members for o in orphans
    ):
        errors.append("orphans")
    else:
        expected_orphans = _expected_orphans(members, sane_edges)
        if sorted(orphans) != expected_orphans:
            errors.append("orphans_mismatch")

    for key, listed in (
        ("n_dangling", dangling),
        ("n_unresolvable", unresolvable),
        ("n_cycles", cycles),
        ("n_orphans", orphans if isinstance(orphans, list) else []),
        ("n_parse_errors", payload.get("parse_errors") or []),
    ):
        if payload.get(key) != len(listed):
            errors.append(key)

    n_dangling = payload.get("n_dangling", len(dangling))
    n_unresolvable = payload.get("n_unresolvable", len(unresolvable))
    n_cycles = payload.get("n_cycles", len(cycles))
    n_parse = payload.get("n_parse_errors", 0)
    expected_verdict = _verdict(
        int(n_parse) if isinstance(n_parse, int) else 0,
        int(n_dangling) if isinstance(n_dangling, int) else 0,
        int(n_unresolvable) if isinstance(n_unresolvable, int) else 0,
        int(n_cycles) if isinstance(n_cycles, int) else 0,
    )
    if payload.get("verdict") != expected_verdict:
        errors.append("verdict")

    params = payload.get("params")
    if isinstance(params, Mapping):
        try:
            derived = hash_bytes(
                canonical_json_bytes(
                    {
                        "digests": {n: m.get("file_sha256") for n, m in members.items()},
                        "params": dict(params),
                    }
                )
            )
        except (TypeError, ValueError, AttributeError):
            derived = None
        if derived is not None and payload.get("inputs_sha256") != derived:
            errors.append("inputs_sha256")
    return errors


def write_graph_receipt(
    receipt: Mapping[str, Any],
    corpus_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a receipt_graph receipt and write ``receipt_graph_<hash>.json``.

    Filename digest = canonical ``receipt_sha256``. Atomic, fail-closed on
    a malformed receipt. ``receipt_version=2`` wraps the same body in the
    unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if (
        receipt.get("kind") != GRAPH_SCHEMA
        or receipt.get("schema") != GRAPH_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("inputs_sha256"), str)
        or not isinstance(receipt.get("params"), Mapping)
    ):
        raise ValueError("receipt_graph receipt violates its contract")
    errors = graph_contract_errors(receipt)
    if errors:
        raise ValueError(f"receipt_graph receipt violates its contract: {errors}")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass" if receipt.get("verdict") == "clean" else "fail",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(corpus_dir) / f"receipt_graph_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
