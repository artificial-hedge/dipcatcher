"""Hash chain for new paper journal and audit lines.

Lines written before this chain are a legacy prefix and are left in place.
A new line stores ``prev_hash`` and ``line_hash``. Startup rejects a torn
tail, a short file relative to the tip, a broken link, or an edited body.
No signing key is read or written.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from quant_fund.observe.order_guard import note_audit_tamper

GENESIS = "0" * 64
TIP_SCHEMA = "paper_log_tip.v1"
_STREAMS = {"journal": "journal/events.jsonl", "audit": "audit/audit.jsonl"}
_CACHE: dict[str, tuple[str, int]] = {}
_BAD = object()


class AuditTamperError(RuntimeError):
    """Startup refused a paper journal or audit log."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _Chain:
    code: str | None
    last: str
    count: int
    hashes: tuple[str, ...]


def forget_runtime(state: Path) -> None:
    """Drop cached tips for ``state``. The next append reads the files."""
    prefix = f"{state.resolve()}:"
    for key in list(_CACHE):
        if key.startswith(prefix):
            _CACHE.pop(key, None)


def append_runtime(state: Path, stream: str, row: Mapping[str, object]) -> None:
    """Append one chained line and record the new tip. The caller holds the lease."""
    path = _stream_path(state, stream)
    prev, count = _cached(state, stream, path)
    sealed = _seal(row, prev)
    from quant_fund.paper.durable_book import append_json_line

    append_json_line(path, sealed)
    digest = sealed["line_hash"]
    if not isinstance(digest, str):
        raise AuditTamperError("edited")
    _CACHE[_cache_key(state, stream)] = (digest, count + 1)
    _write_tip(state, stream, digest, count + 1)


def verify_book(state: Path) -> None:
    """Halt when the journal or the audit log was truncated, reordered, or edited."""
    code = verify_state(state)
    if code is None:
        return
    note_audit_tamper(code)
    raise AuditTamperError(code)


def verify_state(state: Path) -> str | None:
    """Return ``truncated``, ``reordered``, ``edited``, or ``None``."""
    loaded = _load_document(state)
    if loaded is _BAD:
        return "truncated"
    document = loaded if isinstance(loaded, dict) else None
    for stream in ("journal", "audit"):
        tip = _stream_tip(document, stream)
        code = verify_stream(_stream_path(state, stream), tip)
        if code is not None:
            return code
    return None


def verify_stream(path: Path, tip: Mapping[str, object] | None) -> str | None:
    text = _read(path)
    if text is None:
        if tip is None:
            return None
        return "truncated"
    if text and not text.endswith("\n"):
        return "truncated"
    walked = _walk(text)
    if walked.code is not None:
        return walked.code
    return _tip_code(tip, walked)


def _cache_key(state: Path, stream: str) -> str:
    return f"{state.resolve()}:{stream}"


def _stream_path(state: Path, stream: str) -> Path:
    relative = _STREAMS.get(stream)
    if relative is None:
        raise AuditTamperError("edited")
    return state / relative


def _seal(row: Mapping[str, object], prev: str) -> dict[str, object]:
    body = {key: value for key, value in row.items() if key != "line_hash"}
    body["prev_hash"] = prev
    body["line_hash"] = _digest(body)
    return body


def _digest(row: Mapping[str, object]) -> str:
    body = {key: value for key, value in row.items() if key != "line_hash"}
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached(state: Path, stream: str, path: Path) -> tuple[str, int]:
    key = _cache_key(state, stream)
    found = _CACHE.get(key)
    if found is not None:
        return found
    code = verify_stream(path, None)
    if code is not None:
        note_audit_tamper(code)
        raise AuditTamperError(code)
    walked = _walk(_read(path) or "")
    _CACHE[key] = (walked.last, walked.count)
    return walked.last, walked.count


def _read(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "x"


def _walk(text: str) -> _Chain:
    prev = GENESIS
    count = 0
    seen = False
    hashes: list[str] = []
    last = GENESIS
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _row(line)
        if row is None:
            return _Chain("truncated", last, count, tuple(hashes))
        kind = _legacy(row, seen)
        if kind == "legacy":
            continue
        if kind is not None:
            return _Chain(kind, last, count, tuple(hashes))
        code = _linked(row, prev)
        if code is not None:
            return _Chain(code, last, count, tuple(hashes))
        last = str(row["line_hash"])
        prev = last
        hashes.append(last)
        seen = True
        count += 1
    return _Chain(None, last, count, tuple(hashes))


def _legacy(row: Mapping[str, object], seen: bool) -> str | None:
    if "line_hash" in row or "prev_hash" in row:
        return None
    if seen:
        return "reordered"
    return "legacy"


def _linked(row: Mapping[str, object], prev: str) -> str | None:
    if "line_hash" not in row or "prev_hash" not in row:
        return "edited"
    if row.get("prev_hash") != prev:
        return "reordered"
    if _digest(row) != row.get("line_hash"):
        return "edited"
    return None


def _row(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    found: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            return None
        found[key] = value
    return found


def _tip_code(tip: Mapping[str, object] | None, walked: _Chain) -> str | None:
    if tip is None:
        return None
    stored_count = tip.get("count")
    stored_hash = tip.get("hash")
    if isinstance(stored_count, bool) or not isinstance(stored_count, int):
        return "truncated"
    if not isinstance(stored_hash, str):
        return "truncated"
    if stored_count > walked.count:
        return "truncated"
    if stored_count == walked.count:
        if stored_hash != walked.last:
            return "edited"
        return None
    return _behind(stored_count, stored_hash, walked.hashes)


def _behind(count: int, digest: str, hashes: tuple[str, ...]) -> str | None:
    if count == 0 and digest == GENESIS:
        return None
    if count < 1 or count > len(hashes):
        return "truncated"
    if hashes[count - 1] != digest:
        return "truncated"
    return None


def _load_document(state: Path) -> dict[str, object] | None | object:
    path = state / "log_tip.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _BAD
    found = _row(json.dumps(payload)) if isinstance(payload, dict) else None
    if found is None or found.get("schema") != TIP_SCHEMA:
        return _BAD
    return found


def _stream_tip(document: Mapping[str, object] | None, stream: str) -> dict[str, object] | None:
    if document is None:
        return None
    streams = document.get("streams")
    if not isinstance(streams, dict):
        return None
    found = streams.get(stream)
    if not isinstance(found, dict):
        return None
    return _row(json.dumps(found))


def _write_tip(state: Path, stream: str, digest: str, count: int) -> None:
    current = _load_document(state)
    if current is _BAD:
        note_audit_tamper("truncated")
        raise AuditTamperError("truncated")
    document: dict[str, object] = current if isinstance(current, dict) else {}
    streams = document.get("streams")
    if not isinstance(streams, dict):
        streams = {}
    streams[stream] = {"count": count, "hash": digest}
    document["schema"] = TIP_SCHEMA
    document["streams"] = streams
    payload = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    _atomic(state / "log_tip.json", payload)


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
