"""Bind a research receipt to the audit ledger without modifying the receipt.

Two digest conventions exist in this repo and they disagree (docs/SOTA/12 §2.4,
N7):

* ``legacy_full_document`` — ``quant_fund.research.verify._receipt_digest``,
  the hash ``verify-research`` uses. It covers the *whole* document, the
  self-referential ``receipt_sha256`` field included, serialized with
  ``ensure_ascii=True`` and ``allow_nan=True``.
* ``seal_excluding_self`` — ``quant_fund.research.receipt_v2.seal_receipt``,
  the convention the seal a receipt *publishes* is computed under. It excludes
  ``receipt_sha256`` and serializes with ``ensure_ascii=False``,
  ``allow_nan=False`` (non-finite floats become ``null``).

For any sealed receipt the two can never be equal, so an external auditor
holding the receipt and the ledger could not connect them. This module records
and matches **both**, versioned by an explicit convention name, so:

* ledger entries written before this change (which embed only the legacy
  digest in ``payload.receipt_sha256``) keep verifying and keep tracing;
* new entries also carry ``receipt_seal_sha256`` and the seal the receipt
  advertises, so a published ``receipt_sha256`` reconciles to its ledger entry;
* ``trace_published_number`` links on either convention and reports which one
  matched.

A published number is located by a dotted path inside the receipt. The link is
the ledger entry whose recorded digest matches, plus an inclusion proof under
the signed Merkle root. Code identity is the git revision and worktree hash
already stored on the receipt; this module does not re-seal them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.audit.errors import AuditError
from quant_fund.audit.ledger import AuditLedger, LedgerEntry
from quant_fund.audit.merkle import hash_leaf, verify_inclusion
from quant_fund.audit.verify import verify_ledger
from quant_fund.research.verify import _receipt_digest
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

LINK_FIELDS = (
    "run_id",
    "git_revision",
    "git_worktree_sha256",
    "config_sha256",
    "dataset_sha256",
    "dataset_content_sha256",
    "northset_inputs_sha256",
    "code_sha256",
)

#: ``verify-research`` convention: whole document, seal included, ASCII-escaped.
LEGACY_DIGEST_CONVENTION = "legacy_full_document"
#: ``receipt_v2.seal_receipt`` convention: document minus ``receipt_sha256``.
SEAL_DIGEST_CONVENTION = "seal_excluding_self"
#: The field name a sealed receipt publishes its own digest under.
SEAL_FIELD = "receipt_sha256"


def receipt_digest(notebook: dict[str, Any]) -> str:
    """Hash a receipt the same way the research verifier hashes it.

    This is the ``legacy_full_document`` convention. It is unchanged and stays
    the value recorded in ``payload.receipt_sha256`` so committed ledger
    entries keep verifying. Use :func:`receipt_seal_digest` for the digest a
    sealed receipt *publishes*.
    """
    return _receipt_digest(notebook)


def receipt_seal_digest(notebook: dict[str, Any]) -> str:
    """Hash a receipt under the canonical seal convention (``seal_excluding_self``).

    Equal to the ``receipt_sha256`` a receipt advertises when it was sealed
    with ``quant_fund.research.receipt_v2.seal_receipt``: the document minus
    its own seal field, serialized by ``quant_fund.utils.hashing.
    canonical_json_bytes`` (sorted keys, tight separators, ``ensure_ascii=False``,
    ``allow_nan=False`` — non-finite floats become ``null``).

    The primitives are shared with ``seal_receipt`` rather than copied, so the
    two cannot drift; ``tests/unit/audit/test_trace.py`` pins the equality
    against ``seal_receipt`` itself.
    """
    body = {key: value for key, value in notebook.items() if key != SEAL_FIELD}
    return hash_bytes(canonical_json_bytes(body))


def receipt_digests(notebook: dict[str, Any]) -> dict[str, str]:
    """Both conventions, keyed by name. The seal convention never raises here.

    ``canonical_json_bytes`` maps non-finite floats to ``null`` instead of
    emitting the non-JSON tokens ``NaN`` / ``Infinity`` that the legacy
    convention would hash (docs/SOTA/12 N16), so a NaN metric yields a digest
    whose preimage is valid JSON.
    """
    return {
        LEGACY_DIGEST_CONVENTION: receipt_digest(notebook),
        SEAL_DIGEST_CONVENTION: receipt_seal_digest(notebook),
    }


def advertised_seal(notebook: dict[str, Any]) -> str | None:
    """The ``receipt_sha256`` a receipt publishes, or ``None`` when unsealed."""
    seal = notebook.get(SEAL_FIELD)
    return seal if isinstance(seal, str) and seal else None


def _entry_digest_candidates(entry_payload: dict[str, Any]) -> set[str]:
    """Every digest a ledger entry may have committed, under either convention."""
    candidates: set[str] = set()
    for field in (SEAL_FIELD, "receipt_seal_sha256"):
        value = entry_payload.get(field)
        if isinstance(value, str) and value:
            candidates.add(value)
    return candidates


def lookup_path(document: Any, metric_path: str) -> Any:
    """Walk a dotted path. Numeric parts index lists. Missing steps raise KeyError."""
    if not metric_path or metric_path.startswith(".") or metric_path.endswith("."):
        raise KeyError(metric_path)
    current = document
    for part in metric_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise KeyError(metric_path)
            current = current[index]
            continue
        raise KeyError(metric_path)
    return current


def trace_published_number(
    receipt_path: Path | str,
    metric_path: str,
    ledger: AuditLedger,
    *,
    trust_public_key: bytes | None = None,
    verify_receipt: bool = False,
    compare_worktree: bool = False,
) -> dict[str, Any]:
    """Trace one number inside a receipt to its ledger entry and code hashes.

    ``linked`` is true only when the ledger verifies, the receipt digest matches
    an entry, the provenance fields match, the number exists, and the inclusion
    proof rebuilds the signed root. ``verify_receipt`` embeds the existing
    ``verify-research`` result and does not change ``linked``.
    """
    path = Path(receipt_path)
    try:
        before = path.read_bytes()
        parsed = json.loads(before)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _unlinked(metric_path, [f"receipt_unreadable:{exc}"])
    if not isinstance(parsed, dict):
        return _unlinked(metric_path, ["receipt_not_object"])
    notebook: dict[str, Any] = parsed
    try:
        digest = receipt_digest(notebook)
    except (TypeError, ValueError) as exc:
        return _unlinked(metric_path, [f"receipt_digest_failed:{exc}"])
    try:
        seal_digest = receipt_seal_digest(notebook)
    except (TypeError, ValueError, AuditError) as exc:
        seal_digest = None
        seal_error: str | None = f"receipt_seal_digest_failed:{exc}"
    else:
        seal_error = None

    errors: list[str] = []
    if seal_error is not None:
        errors.append(seal_error)
    try:
        value = lookup_path(notebook, metric_path)
    except KeyError:
        value = None
        errors.append("metric_not_found")

    report = verify_ledger(ledger.root, trust_public_key=trust_public_key)
    if not report["fully_signed"]:
        errors.append("ledger_not_fully_signed")
    if not report["valid"]:
        errors.append("ledger_invalid")
        errors.extend(str(item) for item in report["errors"])

    provenance = notebook.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        errors.append("provenance_missing")

    try:
        entries = ledger.entries()
    except AuditError as exc:
        entries = []
        errors.append("ledger_unreadable")
        errors.extend(exc.errors)

    # Match on EITHER convention: the legacy whole-document digest an entry
    # recorded before the convention split, or the canonical seal digest the
    # receipt publishes. Pre-split entries carry only payload.receipt_sha256
    # and keep tracing; post-split entries carry both.
    matches: list[tuple[LedgerEntry, str]] = []
    for candidate in entries:
        if candidate.kind != "research_run":
            continue
        recorded = _entry_digest_candidates(candidate.payload)
        if seal_digest is not None and seal_digest in recorded:
            matches.append((candidate, SEAL_DIGEST_CONVENTION))
        elif digest in recorded:
            matches.append((candidate, LEGACY_DIGEST_CONVENTION))
    if not matches:
        errors.append("receipt_not_in_ledger")
        entry = None
        digest_convention: str | None = None
    else:
        entry, digest_convention = matches[-1]
        for field in LINK_FIELDS:
            recorded_value = entry.payload.get(field)
            current = provenance.get(field)
            if current is None and field == "code_sha256":
                current = notebook.get("code_sha256")
            if recorded_value != current:
                errors.append(f"provenance_mismatch:{field}")

    inclusion: dict[str, Any] | None = None
    if entry is not None and report.get("merkle_root") and report.get("tree_size"):
        try:
            proof = ledger.inclusion_proof(entry.index)
            root = bytes.fromhex(str(report["merkle_root"]))
            inclusion_ok = verify_inclusion(
                hash_leaf(entry.preimage()),
                entry.index,
                proof,
                int(report["tree_size"]),
                root,
            )
        except (AuditError, ValueError):
            inclusion_ok = False
            proof = []
        if not inclusion_ok:
            errors.append("inclusion_failed")
        inclusion = {
            "index": entry.index,
            "tree_size": report["tree_size"],
            "merkle_root": report["merkle_root"],
            "proof": [node.hex() for node in proof],
            "valid": inclusion_ok,
        }

    checkout_revision = git_revision()
    recorded_revision = provenance.get("git_revision")
    result: dict[str, Any] = {
        "linked": not errors,
        "metric_path": metric_path,
        "value": value,
        # Legacy convention, kept under its historical key so existing
        # consumers (and the audit-trace CLI output) do not move.
        "receipt_sha256": digest,
        "digest_convention": LEGACY_DIGEST_CONVENTION,
        # The canonical seal digest the receipt publishes, plus the seal it
        # actually advertises. Equal iff the receipt was sealed with
        # receipt_v2.seal_receipt; both are reported so a mismatch is visible
        # rather than inferred.
        "receipt_seal_sha256": seal_digest,
        "advertised_receipt_sha256": advertised_seal(notebook),
        "seal_matches_advertised": (
            seal_digest is not None and seal_digest == advertised_seal(notebook)
        ),
        "matched_digest_convention": digest_convention,
        "ledger_index": None if entry is None else entry.index,
        "entry_hash": None if entry is None else entry.entry_hash,
        "git_revision": recorded_revision,
        "git_worktree_sha256": provenance.get("git_worktree_sha256"),
        "config_sha256": provenance.get("config_sha256"),
        "dataset_sha256": provenance.get("dataset_sha256"),
        "dataset_content_sha256": provenance.get("dataset_content_sha256"),
        "northset_inputs_sha256": provenance.get("northset_inputs_sha256"),
        "code_sha256": provenance.get("code_sha256", notebook.get("code_sha256")),
        "checkout_git_revision": checkout_revision,
        "checkout_matches_recorded_revision": checkout_revision == recorded_revision,
        "inclusion": inclusion,
        "ledger": report,
        "errors": errors,
        "live_pnl_claim": False,
    }
    if compare_worktree:
        checkout_tree = git_worktree_sha256()
        result["checkout_git_worktree_sha256"] = checkout_tree
        result["checkout_matches_recorded_worktree"] = checkout_tree == provenance.get(
            "git_worktree_sha256"
        )
    if verify_receipt:
        from quant_fund.research.verify import verify_research_artifact

        result["research_verifier"] = verify_research_artifact(path)
    try:
        unchanged = path.read_bytes() == before
    except OSError:
        unchanged = False
    if not unchanged:
        result["linked"] = False
        result["errors"] = [*errors, "receipt_mutated"]
    return result


def _unlinked(metric_path: str, errors: list[str]) -> dict[str, Any]:
    return {
        "linked": False,
        "metric_path": metric_path,
        "value": None,
        "receipt_sha256": None,
        "digest_convention": None,
        "receipt_seal_sha256": None,
        "advertised_receipt_sha256": None,
        "seal_matches_advertised": False,
        "matched_digest_convention": None,
        "errors": errors,
        "live_pnl_claim": False,
    }
