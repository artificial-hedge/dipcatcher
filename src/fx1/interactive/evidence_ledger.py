"""Harness-authored evidence ledger.

Receipt verdicts come from the sealed-receipt verifier. Web, flash, and model
items are never ``VERIFIED``: unverified prose must not inherit verified
status. A command exit code is not an input to this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.redact import redact_text

RECEIPT_MARKER = "receipt="
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_PATH_CHARS = 512
_MAX_LOCATE_CHARS = 4096
_DETAIL_LIMIT = 200
_STEP_LIMIT = 128
_EXTRACT_CAP = 20

_BLOCK_HEADER = "Evidence (harness-verified — authored by dipcatcher, not by the model)"
_CITED_HEADER = "Model interpretation (cites harness-verified evidence below):"
_PLAIN_HEADER = "Model interpretation (not independently verified):"
_WITHHELD_HEADER = "Model interpretation withheld by the honesty gate:"
_WEB_DETAIL = "HEURISTIC WEB VERIFICATION — not a sealed receipt"
_WEB_ROW = "(heuristic web verification — not a sealed receipt)"
_FLASH_ROW = "(flash entry is not a sealed receipt)"
_MODEL_ROW = "(model interpretation — not a sealed receipt)"
_SYNTHETIC_NOTE = "SYNTHETIC correctness evidence only — not market evidence"
_NON_RECEIPT_NOTE = "non-receipt evidence cannot be VERIFIED"

_CANONICAL_LABELS = {
    "SYNTHETIC": "SYNTHETIC",
    "USER_SUPPLIED": "USER_SUPPLIED",
    "REAL": "REAL",
    "UNKNOWN": "UNKNOWN",
}
_HEX64 = re.compile(r"[0-9a-fA-F]{64}")
_HASH_RE = re.compile(r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{64})(?![0-9A-Fa-f])")
_JSON_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])([A-Za-z0-9_./\\-]*[A-Za-z0-9_-]\.json)(?![A-Za-z0-9_./\\-])"
)
_VERIFIED_LANGUAGE_RE = re.compile(
    r"\bverified\b|\bproven\b|confirmed by receipt",
    re.IGNORECASE,
)
_DATA_LABEL_LINE = re.compile(r"^DATA_LABEL\s*=\s*(.*)$")
_ITEM_ID_RE = re.compile(r"E\d+")
_LABEL_TOKEN = re.compile(r"[A-Z0-9_-]{1,64}")
_SYNTHETIC_WORD = re.compile(r"\bSYNTHETIC\b")

AnswerStatus = Literal["verified", "partial", "unverified", "no-evidence", "withheld"]


class Verdict(StrEnum):
    VERIFIED = "VERIFIED"
    INVALID = "INVALID"
    UNVERIFIABLE = "UNVERIFIABLE"
    MISSING = "MISSING"


class EvidenceKind(StrEnum):
    RECEIPT = "RECEIPT"
    WEB = "WEB"
    FLASH = "FLASH"
    MODEL = "MODEL"


_FAILURE_LABELS = (
    (Verdict.INVALID, "INVALID receipt"),
    (Verdict.MISSING, "MISSING receipt"),
    (Verdict.UNVERIFIABLE, "UNVERIFIED receipt"),
)
_NONRECEIPT_LABELS = (
    (EvidenceKind.WEB, "heuristic web source", "heuristic web sources"),
    (EvidenceKind.FLASH, "flash entry", "flash entries"),
    (EvidenceKind.MODEL, "model item", "model items"),
)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    item_id: str
    kind: EvidenceKind
    ref: str
    sha256: str | None
    seal: str | None
    verdict: Verdict
    data_label: str
    schema: str | None
    detail: str
    step: str | None

    def __post_init__(self) -> None:
        _normalize_item(self)


@dataclass(frozen=True, slots=True)
class CitationProblem:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class ComposedAnswer:
    text: str
    status: AnswerStatus
    problems: tuple[CitationProblem, ...]


@dataclass(frozen=True, slots=True)
class _Located:
    path: Path | None
    verdict: Verdict | None
    detail: str
    ref: str


@dataclass(frozen=True, slots=True)
class _Checked:
    verdict: Verdict
    detail: str
    seal: str | None
    schema: str | None


def _normalize_item(item: EvidenceItem) -> None:
    label = item.data_label.strip() or "UNKNOWN"
    if label != item.data_label:
        object.__setattr__(item, "data_label", label)
    if item.kind is EvidenceKind.RECEIPT or item.verdict is not Verdict.VERIFIED:
        return
    _downgrade_non_receipt(item)


def _downgrade_non_receipt(item: EvidenceItem) -> None:
    object.__setattr__(item, "verdict", Verdict.UNVERIFIABLE)
    if _NON_RECEIPT_NOTE in item.detail:
        return
    object.__setattr__(item, "detail", _join_note(item.detail, _NON_RECEIPT_NOTE))


def _join_note(detail: str, note: str) -> str:
    if not detail:
        return note
    return f"{detail}; {note}"


def _canon_label(raw: str) -> str:
    cleaned = raw.strip()
    if _label_rejected(cleaned):
        return "UNKNOWN"
    upper = cleaned.upper()
    if _SYNTHETIC_WORD.search(upper):
        return "SYNTHETIC"
    token = upper.split(None, 1)[0].strip("\"'")
    mapped = _CANONICAL_LABELS.get(token)
    if mapped is not None:
        return mapped
    if _LABEL_TOKEN.fullmatch(upper):
        return upper
    return "UNKNOWN"


def _label_rejected(cleaned: str) -> bool:
    if not cleaned or len(cleaned) > 64:
        return True
    return "\x00" in cleaned


def _collapse_labels(found: Sequence[str]) -> str:
    if not found:
        return "UNKNOWN"
    if any(label == "SYNTHETIC" for label in found):
        return "SYNTHETIC"
    unique = list(dict.fromkeys(found))
    if len(unique) == 1:
        return unique[0]
    return "UNKNOWN"


def _markers(marker: str) -> tuple[str, ...]:
    if marker.endswith("="):
        return (marker, marker[:-1] + ":")
    return (marker,)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _path_token_rejected(value: str) -> bool:
    if not value or len(value) > _MAX_PATH_CHARS:
        return True
    if "\x00" in value or "\n" in value or "\r" in value:
        return True
    return ".." in value


def _clean_path_token(raw: str) -> str | None:
    value = _unquote(raw.strip())
    if _path_token_rejected(value):
        return None
    return value


def _extract_line(line: str, markers: Sequence[str]) -> str | None:
    best: tuple[int, str] | None = None
    for marker in markers:
        match = re.search(rf"(?:^|\s){re.escape(marker)}\s*(.*)$", line)
        if match is None:
            continue
        if best is None or match.start() < best[0]:
            best = (match.start(), match.group(1))
    if best is None:
        return None
    return _clean_path_token(best[1])


def extract_receipt_paths(stdout: str, marker: str = RECEIPT_MARKER) -> list[str]:
    """Pull receipt paths from harness stdout without touching the filesystem.

    Accepts ``receipt=<path>`` and ``receipt: <path>`` (or *marker*). Empty
    values, NUL, newlines, ``..``, and values longer than 512 characters are
    dropped. One marker per line (the rest of the line is the value).
    Duplicates collapse; at most 20 paths are returned, in order.
    """
    markers = _markers(marker)
    found: list[str] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        token = _extract_line(line, markers)
        if token is None or token in seen:
            continue
        seen.add(token)
        found.append(token)
        if len(found) >= _EXTRACT_CAP:
            break
    return found


def _label_on_line(line: str) -> str | None:
    stripped = line.strip()
    match = _DATA_LABEL_LINE.match(stripped)
    if match:
        raw = match.group(1).strip()
        if not raw or "\x00" in raw:
            return None
        return _canon_label(raw)
    if stripped == "SYNTHETIC":
        return "SYNTHETIC"
    return None


def extract_data_label(stdout: str) -> str:
    """Read ``DATA_LABEL=`` lines and a bare ``SYNTHETIC`` line.

    Any SYNTHETIC label wins. Disagreeing non-synthetic labels become
    ``UNKNOWN``. Absent markers are ``UNKNOWN``.
    """
    found = [label for line in stdout.splitlines() if (label := _label_on_line(line)) is not None]
    return _collapse_labels(found)


def _hex64(value: object) -> str | None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return value.lower()
    return None


def _path_text(path: str | Path) -> str | None:
    try:
        value = os.fspath(path)
    except TypeError:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _path_text_rejected(text: str) -> bool:
    if not text or len(text) > _MAX_LOCATE_CHARS:
        return True
    return "\x00" in text or "\n" in text or "\r" in text


def _as_path(path: str | Path) -> Path | None:
    text = _path_text(path)
    if text is None or _path_text_rejected(text):
        return None
    try:
        return Path(text)
    except (TypeError, ValueError):
        return None


def _safe_ref(path: str | Path) -> str:
    text = _path_text(path)
    if text is None:
        return ""
    return redact_text(text.replace("\x00", ""))[:_MAX_PATH_CHARS]


def _resolve(path: Path) -> Path | None:
    try:
        return path.resolve(strict=False)
    except OSError:
        return None


def _ref_under(resolved: Path, root: Path) -> str | None:
    try:
        relative = resolved.relative_to(Path(root).resolve(strict=False))
    except (OSError, ValueError):
        return None
    return relative.as_posix()


def _ref_under_any(resolved: Path, roots: Sequence[Path]) -> str | None:
    for root in roots:
        ref = _ref_under(resolved, root)
        if ref is not None:
            return ref
    return None


def _candidate_paths(path: Path, roots: Sequence[Path]) -> list[Path]:
    if path.is_absolute():
        return [path]
    return [Path(root) / path for root in roots]


def _contained_targets(path: Path, roots: Sequence[Path]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for candidate in _candidate_paths(path, roots):
        resolved = _resolve(candidate)
        if resolved is None:
            continue
        ref = _ref_under_any(resolved, roots)
        if ref is not None:
            found.append((resolved, ref))
    return found


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _first_file(contained: Sequence[tuple[Path, str]]) -> tuple[Path, str] | None:
    for path, ref in contained:
        if _is_file(path):
            return path, ref
    return None


def _pick_contained(contained: Sequence[tuple[Path, str]]) -> _Located:
    chosen = _first_file(contained)
    if chosen is not None:
        path, ref = chosen
        return _Located(path, None, "", ref)
    for path, ref in contained:
        if _exists(path):
            return _Located(path, Verdict.UNVERIFIABLE, "not a regular file", ref)
    path, ref = contained[0]
    return _Located(path, Verdict.MISSING, "receipt file not found", ref)


def _locate(path: str | Path, roots: Sequence[Path]) -> _Located:
    raw = _as_path(path)
    if raw is None:
        return _Located(None, Verdict.UNVERIFIABLE, "path rejected", "")
    contained = _contained_targets(raw, roots)
    if not contained:
        return _Located(None, Verdict.UNVERIFIABLE, "outside allowed roots", _safe_ref(raw))
    return _pick_contained(contained)


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _read_fd(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        block = os.read(fd, min(remaining, 1024 * 1024))
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _use_nofollow() -> bool:
    """POSIX can refuse to reopen a swapped symlink. Windows has no O_NOFOLLOW."""
    return os.name == "posix"


def _read_limited(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if _use_nofollow():
        flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        return _read_fd(fd, max_bytes + 1)
    finally:
        os.close(fd)


def _read_capped(path: Path, max_bytes: int) -> tuple[bytes | None, str | None]:
    size = _file_size(path)
    if size is None:
        return None, "receipt unreadable"
    if size > max_bytes:
        return None, "receipt exceeds size limit"
    try:
        data = _read_limited(path, max_bytes)
    except OSError:
        return None, "receipt unreadable"
    if len(data) > max_bytes:
        return None, "receipt exceeds size limit"
    return data, None


def _as_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _json_object(data: bytes) -> dict[str, Any] | None:
    try:
        parsed: object = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError, MemoryError):
        return None
    return _as_object(parsed)


def _string_label(payload: Mapping[str, Any]) -> str | None:
    raw = payload.get("data_label")
    if isinstance(raw, str) and raw.strip():
        return _canon_label(raw)
    return None


def _nested_label(payload: Mapping[str, Any]) -> str | None:
    inner = _as_object(payload.get("payload"))
    if inner is None:
        return None
    return _string_label(inner)


def _label_from_payload(payload: Mapping[str, Any]) -> str:
    labels = [
        label for label in (_string_label(payload), _nested_label(payload)) if label is not None
    ]
    return _collapse_labels(labels)


def _schema_fields(payload: Mapping[str, Any]) -> str | None:
    for key in ("schema", "kind"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _schema_from_payload(payload: Mapping[str, Any]) -> str | None:
    found = _schema_fields(payload)
    if found is not None:
        return found
    inner = _as_object(payload.get("payload"))
    if inner is None:
        return None
    return _schema_fields(inner)


def _claims_from_bytes(data: bytes) -> tuple[str, str | None, str | None]:
    payload = _json_object(data)
    if payload is None:
        return "UNKNOWN", None, None
    return (
        _label_from_payload(payload),
        _hex64(payload.get("receipt_sha256")),
        _schema_from_payload(payload),
    )


def _clip(text: str, limit: int = _DETAIL_LIMIT) -> str:
    cleaned = redact_text(text).replace("\n", " ").replace("\r", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _short_reason(text: str) -> str:
    return _clip(text) or "receipt invalid"


def _invalid_detail(result: Mapping[str, Any]) -> str:
    errors = result.get("errors")
    if not isinstance(errors, list):
        return "receipt invalid"
    parts = _error_parts(errors)
    if not parts:
        return "receipt invalid"
    return _short_reason("; ".join(parts))


def _error_parts(errors: list[Any]) -> list[str]:
    parts: list[str] = []
    for item in errors:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
        if len(parts) >= 8:
            break
    return parts


def _fault_detail(exc: Exception) -> str:
    return _short_reason(f"verifier failed: {exc.__class__.__name__}: {exc}")


def _seal_from_verifier(result: Mapping[str, Any]) -> str | None:
    for key in ("seal", "digest", "receipt_sha256"):
        found = _hex64(result.get(key))
        if found is not None:
            return found
    return None


def _schema_from_verifier(result: Mapping[str, Any]) -> str | None:
    value = result.get("schema")
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "unknown":
        return None
    return cleaned


def _checked_from_mapping(result: object, seal: str | None, schema: str | None) -> _Checked:
    if not isinstance(result, Mapping):
        return _Checked(Verdict.UNVERIFIABLE, "verifier returned no verdict", seal, schema)
    body: Mapping[str, Any] = result
    reported_seal = _seal_from_verifier(body) or seal
    reported_schema = _schema_from_verifier(body) or schema
    valid = body.get("valid")
    if valid is True:
        return _Checked(Verdict.VERIFIED, "sealed receipt verified", reported_seal, reported_schema)
    if valid is False:
        return _Checked(Verdict.INVALID, _invalid_detail(body), reported_seal, reported_schema)
    return _Checked(
        Verdict.UNVERIFIABLE, "verifier returned no verdict", reported_seal, reported_schema
    )


def _verify_default(data: bytes, path: Path) -> Mapping[str, Any]:
    from quant_fund.schemas.receipt_v2_leaf import verify_receipt_bytes

    return verify_receipt_bytes(data, path)


def _apply_verifier(
    path: Path,
    data: bytes,
    verifier: Callable[[Path], Mapping[str, Any]] | None,
    seal: str | None,
    schema: str | None,
) -> _Checked:
    try:
        if verifier is None:
            result: object = _verify_default(data, path)
        else:
            result = verifier(path)
    except Exception as exc:  # noqa: BLE001 — a verifier fault is a verdict, never VERIFIED
        return _Checked(Verdict.UNVERIFIABLE, _fault_detail(exc), seal, schema)
    return _checked_from_mapping(result, seal, schema)


def _receipt_item(
    *,
    ref: str,
    verdict: Verdict,
    detail: str,
    data_label: str = "UNKNOWN",
    sha256: str | None = None,
    seal: str | None = None,
    schema: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        item_id="E0",
        kind=EvidenceKind.RECEIPT,
        ref=redact_text(ref)[:_MAX_PATH_CHARS],
        sha256=sha256,
        seal=seal,
        verdict=verdict,
        data_label=data_label or "UNKNOWN",
        schema=schema,
        detail=detail,
        step=None,
    )


def verify_receipt(
    path: str | Path,
    *,
    roots: Sequence[Path],
    max_bytes: int = MAX_RECEIPT_BYTES,
    verifier: Callable[[Path], Mapping[str, Any]] | None = None,
) -> EvidenceItem:
    """Verify one receipt file that resolves under ``roots``.

    The file is read only after the resolved path is inside a root. The
    default check is ``verify_receipt_bytes`` on those bytes — the check
    ``verify_receipt_file`` runs after its own read — so the sha256 and the
    verdict describe one byte string. ``item_id`` is ``E0`` until a ledger
    assigns one. A verifier exception yields ``UNVERIFIABLE``.
    """
    located = _locate(path, roots)
    if located.verdict is not None or located.path is None:
        return _receipt_item(
            ref=located.ref,
            verdict=located.verdict or Verdict.UNVERIFIABLE,
            detail=located.detail or "path unreadable",
        )
    data, read_error = _read_capped(located.path, max_bytes)
    if data is None:
        return _receipt_item(
            ref=located.ref,
            verdict=Verdict.UNVERIFIABLE,
            detail=read_error or "receipt unreadable",
        )
    label, seal, schema = _claims_from_bytes(data)
    checked = _apply_verifier(located.path, data, verifier, seal, schema)
    return _receipt_item(
        ref=located.ref,
        verdict=checked.verdict,
        detail=checked.detail,
        data_label=label,
        sha256=hashlib.sha256(data).hexdigest(),
        seal=checked.seal,
        schema=checked.schema,
    )


def _clean_step(step: str | None) -> str | None:
    if step is None:
        return None
    cleaned = redact_text(step).strip()
    if not cleaned:
        return None
    return cleaned[:_STEP_LIMIT]


def _verdict_word(verdict: Verdict) -> str:
    if verdict is Verdict.UNVERIFIABLE:
        return "UNVERIFIED"
    return verdict.value


def _class_phrase(kind: EvidenceKind) -> str:
    if kind is EvidenceKind.WEB:
        return _WEB_ROW
    if kind is EvidenceKind.FLASH:
        return _FLASH_ROW
    return _MODEL_ROW


def _short_hash(value: str | None) -> str | None:
    if value is None or len(value) < 8:
        return None
    return f"{value[:4]}…{value[-4:]}"


def _join_fields(fields: Sequence[str]) -> str:
    return "  " + "  ".join(fields)


def _append_if(fields: list[str], value: str | None) -> None:
    if value:
        fields.append(value)


def _append_hash(fields: list[str], sha256: str | None) -> None:
    short = _short_hash(sha256)
    if short is not None:
        fields.append(f"sha256 {short}")


def _append_step(fields: list[str], step: str | None) -> None:
    if step:
        fields.append(f"step {step}")


def _append_failure_detail(fields: list[str], item: EvidenceItem) -> None:
    if item.verdict is Verdict.VERIFIED or not item.detail:
        return
    fields.append(f"({item.detail})")


def _receipt_row(item: EvidenceItem) -> str:
    fields = [
        item.item_id,
        item.kind.value.lower(),
        _verdict_word(item.verdict),
        item.data_label or "UNKNOWN",
    ]
    _append_if(fields, item.schema)
    _append_hash(fields, item.sha256)
    fields.append(item.ref)
    _append_step(fields, item.step)
    _append_failure_detail(fields, item)
    return _join_fields(fields)


def _other_row(item: EvidenceItem) -> str:
    fields = [
        item.item_id,
        item.kind.value.lower(),
        _verdict_word(item.verdict),
        _class_phrase(item.kind),
        item.ref,
    ]
    _append_step(fields, item.step)
    return _join_fields(fields)


def _row(item: EvidenceItem) -> str:
    if item.kind is EvidenceKind.RECEIPT:
        return _receipt_row(item)
    return _other_row(item)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        word = singular
    elif plural is not None:
        word = plural
    else:
        word = f"{singular}s"
    return f"{count} {word}"


def _synthetic_count(items: Sequence[EvidenceItem]) -> int:
    return sum(1 for item in items if item.data_label == "SYNTHETIC")


def _verified_clause(verified: Sequence[EvidenceItem]) -> str:
    phrase = _plural(len(verified), "verified receipt")
    synthetic = _synthetic_count(verified)
    if synthetic == 0:
        return phrase
    if synthetic == len(verified):
        return f"{phrase} ({_SYNTHETIC_NOTE})"
    return f"{phrase} ({synthetic} {_SYNTHETIC_NOTE})"


def _count_receipts(items: Sequence[EvidenceItem], verdict: Verdict) -> int:
    return sum(1 for item in items if item.kind is EvidenceKind.RECEIPT and item.verdict is verdict)


def _failure_clauses(items: Sequence[EvidenceItem]) -> list[str]:
    clauses: list[str] = []
    for verdict, label in _FAILURE_LABELS:
        count = _count_receipts(items, verdict)
        if count:
            clauses.append(_plural(count, label))
    return clauses


def _kind_count(items: Sequence[EvidenceItem], kind: EvidenceKind) -> int:
    return sum(1 for item in items if item.kind is kind)


def _nonreceipt_clauses(items: Sequence[EvidenceItem]) -> list[str]:
    clauses: list[str] = []
    for kind, singular, plural in _NONRECEIPT_LABELS:
        count = _kind_count(items, kind)
        if count:
            clauses.append(_plural(count, singular, plural))
    return clauses


def _verified_only(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    return [
        item
        for item in items
        if item.kind is EvidenceKind.RECEIPT and item.verdict is Verdict.VERIFIED
    ]


def _summary(items: Sequence[EvidenceItem]) -> str:
    if not items:
        return "Evidence class summary: no evidence items."
    parts = [_verified_clause(_verified_only(items))]
    parts.extend(_failure_clauses(items))
    parts.extend(_nonreceipt_clauses(items))
    return "Evidence class summary: " + "; ".join(parts) + "."


def _wrap_line(line: str, width: int) -> list[str]:
    limit = width if width >= 16 else 16
    if len(line) <= limit:
        return [line]
    return _wrap_long(line, limit)


def _wrap_long(line: str, limit: int) -> list[str]:
    lines: list[str] = []
    rest = line
    pad = ""
    while rest:
        rest, pad = _wrap_step(lines, rest, pad, limit)
    return lines


def _wrap_step(lines: list[str], rest: str, pad: str, limit: int) -> tuple[str, str]:
    budget = max(limit - len(pad), 8)
    if len(rest) <= budget:
        lines.append(pad + rest)
        return "", "    "
    cut = rest.rfind(" ", 0, budget)
    if cut <= 0:
        cut = budget
    chunk = rest[:cut].rstrip()
    nxt = rest[cut:].lstrip()
    if chunk:
        lines.append(pad + chunk)
    return nxt, "    "


def _wrap_lines(lines: Sequence[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_line(line, width))
    return wrapped


def _item_has_hash(item: EvidenceItem, key: str) -> bool:
    return any(value is not None and value.lower() == key for value in (item.sha256, item.seal))


def _matching_items(token: str, items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    key = token.lower()
    return [item for item in items if _item_has_hash(item, key)]


def _any_verified(items: Sequence[EvidenceItem]) -> bool:
    return any(item.verdict is Verdict.VERIFIED for item in items)


def _one_hash_problem(token: str, items: Sequence[EvidenceItem]) -> CitationProblem | None:
    matched = _matching_items(token, items)
    if _any_verified(matched):
        return None
    key = token.lower()
    if matched:
        item = matched[0]
        return CitationProblem(
            code="unverified_hash",
            detail=f"hash {key} matches {item.item_id} ({item.verdict.value})",
        )
    return CitationProblem(code="unknown_hash", detail=f"hash {key} is not in the ledger")


def _hash_problems(answer: str, items: Sequence[EvidenceItem]) -> list[CitationProblem]:
    problems: list[CitationProblem] = []
    seen: set[str] = set()
    for match in _HASH_RE.finditer(answer):
        token = match.group(1)
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        problem = _one_hash_problem(token, items)
        if problem is not None:
            problems.append(problem)
    return problems


def _norm_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _basename_match(left: str, right: str) -> bool:
    left_name = left.rsplit("/", 1)[-1]
    right_name = right.rsplit("/", 1)[-1]
    if "/" not in left and left == right_name:
        return True
    return "/" not in right and right == left_name


def _matches_ref(mention: str, ref: str) -> bool:
    left = _norm_path(mention)
    right = _norm_path(ref)
    if not left or not right:
        return False
    if left == right:
        return True
    if left.endswith("/" + right) or right.endswith("/" + left):
        return True
    return _basename_match(left, right)


def _path_known(mention: str, items: Sequence[EvidenceItem]) -> bool:
    return any(_matches_ref(mention, item.ref) for item in items)


def _path_problems(answer: str, items: Sequence[EvidenceItem]) -> list[CitationProblem]:
    problems: list[CitationProblem] = []
    seen: set[str] = set()
    for match in _JSON_PATH_RE.finditer(answer):
        mention = match.group(1)
        if mention in seen or _path_known(mention, items):
            continue
        seen.add(mention)
        problems.append(
            CitationProblem(
                code="unknown_receipt_path", detail=f"path {mention} is not in the ledger"
            )
        )
    return problems


def _language_problem(answer: str, items: Sequence[EvidenceItem]) -> CitationProblem | None:
    if _any_verified(items):
        return None
    match = _VERIFIED_LANGUAGE_RE.search(answer)
    if match is None:
        return None
    return CitationProblem(
        code="verified_language_without_receipt",
        detail=f"answer says {match.group(0)!r} without a verified receipt",
    )


def _has_synthetic(items: Sequence[EvidenceItem]) -> bool:
    return any(item.data_label == "SYNTHETIC" for item in items)


def _label_problem(answer: str, items: Sequence[EvidenceItem]) -> CitationProblem | None:
    if not _has_synthetic(items):
        return None
    if _SYNTHETIC_WORD.search(answer):
        return None
    return CitationProblem(
        code="label_missing",
        detail="SYNTHETIC evidence requires an explicit SYNTHETIC label",
    )


def check_citations(answer: str, ledger: EvidenceLedger) -> list[CitationProblem]:
    """Check hashes, receipt paths, verified-language, and the SYNTHETIC label.

    A 64-hex string must equal some item sha256 or seal (case-insensitive)
    and that item must be ``VERIFIED``. Shorter hex is ignored. Path-like
    ``*.json`` mentions must match a ledger ref. ``verified`` / ``proven`` /
    ``confirmed by receipt`` with no verified item is
    ``verified_language_without_receipt``. A SYNTHETIC ledger item requires
    an explicit uppercase ``SYNTHETIC`` label in the answer.
    """
    items = ledger.items()
    problems = _hash_problems(answer, items)
    problems.extend(_path_problems(answer, items))
    language = _language_problem(answer, items)
    if language is not None:
        problems.append(language)
    label = _label_problem(answer, items)
    if label is not None:
        problems.append(label)
    return problems


def _fully_verified(items: Sequence[EvidenceItem], problems: Sequence[CitationProblem]) -> bool:
    if problems:
        return False
    if not items:
        return False
    return all(item.verdict is Verdict.VERIFIED for item in items)


def _compose_status(ledger: EvidenceLedger, problems: Sequence[CitationProblem]) -> AnswerStatus:
    items = ledger.items()
    if not items:
        return "no-evidence"
    if _fully_verified(items, problems):
        return "verified"
    if ledger.has_verified():
        return "partial"
    return "unverified"


def _compose_header(ledger: EvidenceLedger, problems: Sequence[CitationProblem]) -> str:
    if ledger.has_verified() and not problems:
        return _CITED_HEADER
    return _PLAIN_HEADER


def _warning_block(problems: Sequence[CitationProblem]) -> str:
    if not problems:
        return ""
    lines = ["Citation warnings:"]
    for problem in problems:
        lines.append(f"- {problem.code}: {problem.detail}")
    return "\n".join(lines)


def _join_answer(header: str, body: str, problems: Sequence[CitationProblem], block: str) -> str:
    parts = [header]
    if body:
        parts.append(body)
    warning = _warning_block(problems)
    if warning:
        parts.append(warning)
    parts.append(block)
    return "\n\n".join(parts)


def _run_validate(model_text: str, validate: Callable[[str], str] | None) -> str:
    checker = validate_fx1_output if validate is None else validate
    return checker(model_text)


def _withheld(model_text: str, ledger: EvidenceLedger, exc: Fx1HonestyError) -> ComposedAnswer:
    problems = tuple(check_citations(model_text, ledger))
    text = _join_answer(_WITHHELD_HEADER, _short_reason(str(exc)), problems, ledger.render_block())
    return ComposedAnswer(text=text, status="withheld", problems=problems)


def compose_answer(
    model_text: str,
    ledger: EvidenceLedger,
    *,
    validate: Callable[[str], str] | None = None,
) -> ComposedAnswer:
    """Attach the harness evidence block to model prose.

    ``validate`` defaults to :func:`fx1.honesty.validate_fx1_output`. If it
    raises :class:`Fx1HonestyError`, the model prose is replaced by the gate
    reason and ``status`` is ``withheld``; the evidence block is still shown.
    ``verified`` requires every ledger item to be a verified receipt and no
    citation problems. A verified receipt mixed with anything else is
    ``partial`` so unverified items do not inherit verified status.
    """
    try:
        shown = _run_validate(model_text, validate)
    except Fx1HonestyError as exc:
        return _withheld(model_text, ledger, exc)
    problems = tuple(check_citations(shown, ledger))
    text = _join_answer(
        _compose_header(ledger, problems), shown.strip(), problems, ledger.render_block()
    )
    return ComposedAnswer(text=text, status=_compose_status(ledger, problems), problems=problems)


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"evidence field {field} must be a string")
    if len(value) > _MAX_LOCATE_CHARS:
        raise ValueError(f"evidence field {field} is too long")
    return value


def _require_id(value: object) -> str:
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError("evidence item_id is invalid")
    if _ITEM_ID_RE.fullmatch(value) is None:
        raise ValueError("evidence item_id is invalid")
    return value


def _require_kind(value: object) -> EvidenceKind:
    try:
        return EvidenceKind(_require_str(value, "kind"))
    except ValueError as exc:
        raise ValueError("evidence kind is invalid") from exc


def _require_verdict(value: object) -> Verdict:
    try:
        return Verdict(_require_str(value, "verdict"))
    except ValueError as exc:
        raise ValueError("evidence verdict is invalid") from exc


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    text = _require_str(value, field).strip()
    return text or None


def _optional_hex(value: object, field: str) -> str | None:
    if value is None:
        return None
    found = _hex64(value)
    if found is None:
        raise ValueError(f"evidence field {field} must be a sha256 hex digest or null")
    return found


def _item_from_mapping(raw: object) -> EvidenceItem:
    if not isinstance(raw, dict):
        raise ValueError("evidence item must be an object")
    body = _as_object(raw)
    if body is None:
        raise ValueError("evidence item must be an object")
    return EvidenceItem(
        item_id=_require_id(body.get("item_id")),
        kind=_require_kind(body.get("kind")),
        ref=redact_text(_require_str(body.get("ref"), "ref"))[:_MAX_PATH_CHARS],
        sha256=_optional_hex(body.get("sha256"), "sha256"),
        seal=_optional_hex(body.get("seal"), "seal"),
        verdict=_require_verdict(body.get("verdict")),
        data_label=_require_str(body.get("data_label"), "data_label"),
        schema=_optional_text(body.get("schema"), "schema"),
        detail=redact_text(_require_str(body.get("detail"), "detail")),
        step=_clean_step(_optional_text(body.get("step"), "step")),
    )


def _restore_items(raw_items: list[Any]) -> list[EvidenceItem]:
    items = [_item_from_mapping(raw) for raw in raw_items]
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence item_id")
    return items


def _next_index(items: Sequence[EvidenceItem]) -> int:
    highest = 0
    for item in items:
        match = re.fullmatch(r"E(\d+)", item.item_id)
        if match is None:
            continue
        highest = max(highest, int(match.group(1)))
    return highest + 1


def _item_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "kind": item.kind.value,
        "ref": item.ref,
        "sha256": item.sha256,
        "seal": item.seal,
        "verdict": item.verdict.value,
        "data_label": item.data_label,
        "schema": item.schema,
        "detail": item.detail,
        "step": item.step,
    }


class EvidenceLedger:
    """Append-only, thread-safe evidence for one run.

    Ids are ``E1``, ``E2``, ... in append order. Web, flash, and model items
    stay unverified even if a flag or a serialized verdict says ``VERIFIED``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[EvidenceItem] = []
        self._next = 1

    def _append(self, item: EvidenceItem) -> EvidenceItem:
        with self._lock:
            numbered = replace(item, item_id=f"E{self._next}")
            self._next += 1
            self._items.append(numbered)
            return numbered

    def add_receipt(
        self,
        path: str | Path,
        *,
        roots: Sequence[Path],
        step: str | None = None,
        verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    ) -> EvidenceItem:
        item = verify_receipt(path, roots=roots, verifier=verifier)
        if step is not None:
            item = replace(item, step=_clean_step(step))
        return self._append(item)

    def add_web(self, url: str, *, step: str | None = None, detail: str = "") -> EvidenceItem:
        text = _WEB_DETAIL
        extra = redact_text(detail).strip()
        if extra:
            text = f"{text}; {extra}"
        item = EvidenceItem(
            item_id="E0",
            kind=EvidenceKind.WEB,
            ref=redact_text(url).strip(),
            sha256=None,
            seal=None,
            verdict=Verdict.UNVERIFIABLE,
            data_label="UNKNOWN",
            schema=None,
            detail=text,
            step=_clean_step(step),
        )
        return self._append(item)

    def add_artifact(
        self,
        path: str | Path,
        *,
        detail: str,
        roots: Sequence[Path] | None = None,
        step: str | None = None,
        data_label: str = "UNKNOWN",
    ) -> EvidenceItem:
        """Record a non-receipt file as ``UNVERIFIABLE`` without reading bytes.

        A path outside ``roots`` is not opened. Callers use this for partial
        family records (``json=``), which are not sealed research receipts
        and must not become ``VERIFIED``.
        """
        note = _artifact_note(path, detail, () if roots is None else roots)
        item = EvidenceItem(
            item_id="E0",
            kind=EvidenceKind.RECEIPT,
            ref=note[0],
            sha256=None,
            seal=None,
            verdict=Verdict.UNVERIFIABLE,
            data_label=data_label or "UNKNOWN",
            schema=None,
            detail=note[1],
            step=_clean_step(step),
        )
        return self._append(item)

    def add_flash(self, entry_id: str, *, verified_flag: bool, detail: str = "") -> EvidenceItem:
        text = _flash_detail(verified_flag)
        extra = redact_text(detail).strip()
        if extra:
            text = f"{text}; {extra}"
        item = EvidenceItem(
            item_id="E0",
            kind=EvidenceKind.FLASH,
            ref=redact_text(entry_id).strip(),
            sha256=None,
            seal=None,
            verdict=Verdict.UNVERIFIABLE,
            data_label="UNKNOWN",
            schema=None,
            detail=text,
            step=None,
        )
        return self._append(item)

    def items(self) -> list[EvidenceItem]:
        with self._lock:
            return list(self._items)

    def verified_receipts(self) -> list[EvidenceItem]:
        return _verified_only(self.items())

    def has_verified(self) -> bool:
        return _any_verified(self.items())

    def labels(self) -> set[str]:
        return {item.data_label for item in self.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"items": [_item_dict(item) for item in self.items()]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceLedger:
        if not isinstance(payload, Mapping):
            raise ValueError("evidence ledger payload must be an object")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("evidence ledger payload missing items")
        items = _restore_items(raw_items)
        ledger = cls()
        with ledger._lock:
            ledger._items = items
            ledger._next = _next_index(items)
        return ledger

    def render_block(self, *, width: int = 100) -> str:
        """Plain-text evidence block. No ANSI. Does not trust model prose."""
        items = self.items()
        lines = [_BLOCK_HEADER, *(_row(item) for item in items), _summary(items)]
        return "\n".join(_wrap_lines(lines, width))


def _artifact_note(path: str | Path, detail: str, roots: Sequence[Path]) -> tuple[str, str]:
    located = _locate(path, roots)
    ref = located.ref or _safe_ref(path)
    text = _clip(detail) or "non-receipt artifact"
    if located.detail:
        text = _join_note(text, located.detail)
    return redact_text(ref)[:_MAX_PATH_CHARS], text


def _flash_detail(verified_flag: bool) -> str:
    if verified_flag:
        return "flash verified_flag set; not a sealed receipt"
    return "flash entry is not a sealed receipt"


__all__ = [
    "MAX_RECEIPT_BYTES",
    "RECEIPT_MARKER",
    "AnswerStatus",
    "CitationProblem",
    "ComposedAnswer",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceLedger",
    "Verdict",
    "check_citations",
    "compose_answer",
    "extract_data_label",
    "extract_receipt_paths",
    "verify_receipt",
]
