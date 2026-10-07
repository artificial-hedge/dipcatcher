"""Condition-1 QUALIFYING feed gate for the licensed-vendor adapter.

`docs/INSTITUTIONAL_READINESS.md` condition 1 requires *"a licensed,
point-in-time data source with release and ingestion timestamps"* before any
live claim. This module is the machine-checkable half of that condition: it
takes normalized vendor rows plus the two dataset-level attestations and
decides whether the feed is **QUALIFYING** evidence.

The rule is deliberately fail-closed and all-or-nothing. ``QUALIFYING`` is
``True`` **only** when every condition-1 field is present *and* consistent:

Per-row causal chain (hard-rejected on violation)::

    event_time <= available_time <= ingested_time
    available_time <= decision_time        # enforced at every consumer

Per-row identity fields (missing -> non-qualifying): ``source_id``,
``revision_id``. Dataset attestations (missing/unattested -> non-qualifying):
``universe_completeness``, ``adjustment_lineage``.

Outcomes are one of:

* ``QUALIFYING``     — every field present and consistent. Only this may be
  used as condition-1 evidence.
* ``NON_QUALIFYING`` — no hard violation, but one or more required fields or
  attestations are missing/unattested. Correctness diagnostics only.
* ``REJECTED``       — a hard point-in-time violation: a causal-chain break, a
  decision-time leak, or a **late revision** (a revision of an already-seen
  ``(security_id, event_key)`` surfacing out of order or after the cutoff).
* ``UNAVAILABLE``    — no entitlement configured. This is the documented
  fail-closed path; there is **no silent synthetic substitution** and
  ``synthetic_substitution`` is always ``False``.

SYNTHETIC data is used only in tests and is always labelled; a SYNTHETIC feed
is never market evidence. Nothing here claims live connectivity or live P&L
(``live_pnl_claim=False`` throughout).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

SOURCE = "vendor_http"

#: Per-row condition-1 fields. Every one must be present and consistent.
CONDITION1_ROW_FIELDS: tuple[str, ...] = (
    "event_time",
    "available_time",
    "ingested_time",
    "source_id",
    "revision_id",
)

#: Dataset-level attestations required before the feed may be QUALIFYING.
CONDITION1_ATTESTATIONS: tuple[str, ...] = (
    "universe_completeness",
    "adjustment_lineage",
)

STATUS_QUALIFYING = "QUALIFYING"
STATUS_NON_QUALIFYING = "NON_QUALIFYING"
STATUS_REJECTED = "REJECTED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

#: Reasons that are hard point-in-time violations -> status REJECTED.
_REJECTED_PREFIXES: tuple[str, ...] = (
    "causal_chain_violation",
    "decision_time_leak",
    "late_revision",
    "future_ingest",
    "duplicate_key",
)


class QualifyingFeedError(ValueError):
    """Malformed input to the qualifying gate (never a data-quality verdict)."""


@dataclass
class _CausalTriple:
    event_time: datetime
    available_time: datetime
    ingested_time: datetime


@dataclass
class _Accumulator:
    """Per-row reason + receipt accumulators (keeps each guard tiny)."""

    decision_time: datetime
    clock: datetime
    reasons: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    seen: dict[tuple[str | None, str], datetime] = field(default_factory=dict)
    source_ids: set[str] = field(default_factory=set)
    revision_ids: set[str] = field(default_factory=set)
    receipt_rows: list[dict[str, Any]] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejected.append(reason)

    def soft(self, reason: str) -> None:
        self.reasons.append(reason)


# -- tiny parsing / extraction guards --


def _parse_utc(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise QualifyingFeedError(f"{field_name} must be timezone-aware: {value!r}")
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        raise QualifyingFeedError(f"{field_name} must be an ISO-8601 timestamp, got {value!r}")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise QualifyingFeedError(f"{field_name} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise QualifyingFeedError(f"{field_name} must be timezone-aware (UTC): {value!r}")
    return parsed.astimezone(UTC)


def _nonempty_str(row: Mapping[str, object], field_name: str) -> str | None:
    raw = row.get(field_name)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _opt_datetime(row: Mapping[str, object], field_name: str) -> datetime | None:
    raw = row.get(field_name)
    return None if raw is None else _parse_utc(raw, field_name=field_name)


# -- per-row guards (each intentionally tiny) --


def _guard_identity(row: Mapping[str, object], idx: int, acc: _Accumulator) -> None:
    source_id = _nonempty_str(row, "source_id")
    revision_id = _nonempty_str(row, "revision_id")
    if source_id is None:
        acc.soft(f"missing_field:{idx}:source_id")
    else:
        acc.source_ids.add(source_id)
    if revision_id is None:
        acc.soft(f"missing_field:{idx}:revision_id")
    else:
        acc.revision_ids.add(revision_id)


def _guard_timestamps(
    row: Mapping[str, object], idx: int, acc: _Accumulator
) -> _CausalTriple | None:
    parsed: dict[str, datetime | None] = {}
    for name in ("event_time", "available_time", "ingested_time"):
        parsed[name] = _opt_datetime(row, name)
        if parsed[name] is None:
            acc.soft(f"missing_field:{idx}:{name}")
    if any(v is None for v in parsed.values()):
        return None
    return _CausalTriple(
        event_time=parsed["event_time"],  # type: ignore[arg-type]
        available_time=parsed["available_time"],  # type: ignore[arg-type]
        ingested_time=parsed["ingested_time"],  # type: ignore[arg-type]
    )


def _guard_causal_chain(idx: int, triple: _CausalTriple, acc: _Accumulator) -> None:
    if not (triple.event_time <= triple.available_time <= triple.ingested_time):
        acc.reject(f"causal_chain_violation:{idx}")
    if triple.ingested_time > acc.clock:
        acc.reject(f"future_ingest:{idx}")


def _guard_availability(
    idx: int,
    key: tuple[str | None, str],
    available_time: datetime,
    acc: _Accumulator,
) -> bool:
    """Enforce ``available_time <= decision_time``; return True if a revision."""
    is_revision = key in acc.seen
    if available_time > acc.decision_time:
        acc.reject(f"{'late_revision' if is_revision else 'decision_time_leak'}:{idx}")
    elif is_revision and available_time < acc.seen[key]:
        acc.reject(f"late_revision:{idx}")
    return is_revision


def _track_revision(
    key: tuple[str | None, str], available_time: datetime, acc: _Accumulator
) -> None:
    prior = acc.seen.get(key)
    if prior is None or available_time > prior:
        acc.seen[key] = available_time


def _receipt_row(
    row: Mapping[str, object], triple: _CausalTriple, source_id: str | None, revision_id: str | None
) -> dict[str, Any]:
    return {
        "security_id": _nonempty_str(row, "security_id"),
        "event_time": triple.event_time.isoformat(),
        "available_time": triple.available_time.isoformat(),
        "ingested_time": triple.ingested_time.isoformat(),
        "source_id": source_id,
        "revision_id": revision_id,
    }


def _evaluate_row(row: object, idx: int, acc: _Accumulator) -> None:
    if not isinstance(row, Mapping):
        raise QualifyingFeedError(f"row {idx} is not a mapping")
    mapping = dict(row)
    _guard_identity(mapping, idx, acc)
    triple = _guard_timestamps(mapping, idx, acc)
    if triple is None:
        return
    _guard_causal_chain(idx, triple, acc)
    source_id = _nonempty_str(mapping, "source_id")
    revision_id = _nonempty_str(mapping, "revision_id")
    key = (
        _nonempty_str(mapping, "security_id"),
        _nonempty_str(mapping, "event_key") or triple.event_time.isoformat(),
    )
    _guard_availability(idx, key, triple.available_time, acc)
    _track_revision(key, triple.available_time, acc)
    acc.receipt_rows.append(_receipt_row(mapping, triple, source_id, revision_id))


# -- dataset-level attestation guards --


def _guard_attestation(
    acc: _Accumulator, attestation: Mapping[str, object] | None, name: str
) -> None:
    if attestation is None:
        acc.soft(f"missing_attestation:{name}")
        return
    if attestation.get("attested") is not True:
        acc.soft(f"unattested:{name}")
    ident = attestation.get("attestation_id")
    if not isinstance(ident, str) or not ident.strip():
        acc.soft(f"missing_attestation_id:{name}")


# -- verdict assembly --


def _is_rejected_reason(reason: str) -> bool:
    key = reason.split(":", 1)[0]
    return any(key.startswith(prefix) for prefix in _REJECTED_PREFIXES)


def _verdict(acc: _Accumulator) -> dict[str, Any]:
    if len(acc.source_ids) > 1:
        acc.soft("source_id_inconsistent")
    hard = list(acc.rejected)
    soft = [r for r in acc.reasons if _is_rejected_reason(r)]
    hard.extend(soft)
    remaining = [r for r in acc.reasons if r not in hard]
    all_reasons = hard + remaining
    status = (
        STATUS_REJECTED if hard else (STATUS_NON_QUALIFYING if all_reasons else STATUS_QUALIFYING)
    )
    return {
        "status": status,
        "qualifying": status == STATUS_QUALIFYING,
        "rejected": bool(hard),
        "reasons": all_reasons,
        "n_rows": len(acc.receipt_rows),
        "source_ids": sorted(acc.source_ids),
        "revision_ids": sorted(acc.revision_ids),
        "decision_time": acc.decision_time.isoformat(),
        "pit_receipt": pit_receipt(acc.receipt_rows),
        "synthetic_substitution": False,
        "research_only": True,
        "live_pnl_claim": False,
    }


def evaluate_feed(
    rows: Sequence[Mapping[str, object]],
    *,
    decision_time: datetime,
    universe_completeness: Mapping[str, object] | None = None,
    adjustment_lineage: Mapping[str, object] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decide whether ``rows`` are QUALIFYING condition-1 evidence.

    ``decision_time`` is the consumer's information cutoff: every row must
    satisfy ``available_time <= decision_time`` or it is a decision-time leak
    (or a late revision when the ``(security_id, event_key)`` was already
    seen). ``now`` bounds ingestion (a future ``ingested_time`` is a hard
    violation); it defaults to the wall clock.

    Returns a JSON-serializable verdict dict carrying ``qualifying`` (the
    flag), ``status``, ``rejected``, ``reasons``, and a hashed PIT receipt.
    """
    if not isinstance(decision_time, datetime) or decision_time.tzinfo is None:
        raise QualifyingFeedError("decision_time must be a timezone-aware datetime")
    acc = _Accumulator(
        decision_time=decision_time.astimezone(UTC),
        clock=now.astimezone(UTC) if now is not None else datetime.now(tz=UTC),
    )
    for idx, row in enumerate(rows):
        _evaluate_row(row, idx, acc)
    _guard_attestation(acc, universe_completeness, "universe_completeness")
    _guard_attestation(acc, adjustment_lineage, "adjustment_lineage")
    return _verdict(acc)


def pit_receipt(rows: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    """Return a hashed, immutable point-in-time receipt over normalized rows.

    The receipt binds each row's full causal chain and identity to a
    content-addressed digest so later silent edits are detectable. It is a
    research/provenance artifact (``research_only=True``), never market
    evidence and never a live claim.
    """
    normalized = [dict(row) for row in rows]
    row_hashes = [hash_bytes(canonical_json_bytes(row)) for row in normalized]
    body = {"schema": "pit_receipt.v1", "rows": row_hashes}
    return {
        "schema": "pit_receipt.v1",
        "n_rows": len(normalized),
        "row_hashes": row_hashes,
        "receipt_sha256": hash_bytes(canonical_json_bytes(body)),
        "synthetic_substitution": False,
        "research_only": True,
        "live_pnl_claim": False,
    }


def unavailable_feed(reason: str) -> dict[str, Any]:
    """Documented fail-closed path when no entitlement is configured.

    Returns an ``UNAVAILABLE`` verdict with ``qualifying=False`` and
    ``synthetic_substitution=False``. There is deliberately **no synthetic
    fallback**: a missing entitlement is an evidence gap, not a row to fill.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise QualifyingFeedError("unavailable reason must be a non-empty string")
    return {
        "status": STATUS_UNAVAILABLE,
        "qualifying": False,
        "rejected": False,
        "reasons": [f"unavailable:{reason.strip()}"],
        "n_rows": 0,
        "source_ids": [],
        "revision_ids": [],
        "decision_time": None,
        "pit_receipt": pit_receipt([]),
        "entitlement": "absent",
        "synthetic_substitution": False,
        "research_only": True,
        "live_pnl_claim": False,
    }


def resolve_feed(
    *,
    entitled: bool,
    rows: Sequence[Mapping[str, object]] = (),
    decision_time: datetime | None = None,
    universe_completeness: Mapping[str, object] | None = None,
    adjustment_lineage: Mapping[str, object] | None = None,
    unavailable_reason: str = "no_entitlement_configured",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Resolve a feed to a verdict, fail-closed on a missing entitlement.

    When ``entitled`` is False the documented ``UNAVAILABLE`` path is returned
    and **no rows are evaluated and no synthetic data is substituted**.
    """
    if not entitled:
        return unavailable_feed(unavailable_reason)
    if decision_time is None:
        raise QualifyingFeedError("decision_time is required when the feed is entitled")
    return evaluate_feed(
        rows,
        decision_time=decision_time,
        universe_completeness=universe_completeness,
        adjustment_lineage=adjustment_lineage,
        now=now,
    )


__all__ = [
    "CONDITION1_ATTESTATIONS",
    "CONDITION1_ROW_FIELDS",
    "QualifyingFeedError",
    "SOURCE",
    "STATUS_NON_QUALIFYING",
    "STATUS_QUALIFYING",
    "STATUS_REJECTED",
    "STATUS_UNAVAILABLE",
    "evaluate_feed",
    "pit_receipt",
    "resolve_feed",
    "unavailable_feed",
]
