"""Carry ledger, journal, rate-store, and audit bytes across a config bump.

N+1 writes ``config.json`` schema 2 and does not rewrite the four state
files. Downgrade to N rewrites that config only when the state schema is
still 1. A newer state schema fails closed and leaves every byte in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_FILES = (
    "ledger/broker_state.json",
    "journal/events.jsonl",
    "rate/window.json",
    "audit/audit.jsonl",
)
_STATE_OK = 1
_CONFIG_N = 1
_CONFIG_NEXT = 2


def content_digests(root: Path) -> dict[str, str]:
    """SHA-256 of the four state files. ``config.json`` is not included."""
    found: dict[str, str] = {}
    for relative in _FILES:
        path = root / relative
        found[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def upgrade(root: Path) -> dict[str, object]:
    """Move config schema 1 to 2. State bytes stay put."""
    return _step(root, _CONFIG_NEXT)


def downgrade(root: Path) -> dict[str, object]:
    """Move a compatible config 2 back to 1, or refuse a newer state schema."""
    return _step(root, _CONFIG_N)


def _step(root: Path, target: int) -> dict[str, object]:
    before = content_digests(root)
    config = _read_config(root)
    state = _schema_int(config.get("state_schema"))
    current = _schema_int(config.get("config_schema"))
    if state != _STATE_OK:
        return _closed(
            before, f"state schema {state} is newer than this process; refusing to rewrite state"
        )
    if current is None or current > _CONFIG_NEXT:
        shown = current if current is not None else "missing"
        return _closed(
            before, f"config schema {shown} is newer than this process; refusing to rewrite state"
        )
    if current != target:
        _write_config(root, _body(target))
    after = content_digests(root)
    ok = after == before
    return {
        "config_schema": target,
        "digests": after,
        "error": "" if ok else "digest",
        "ok": ok,
    }


def _closed(before: dict[str, str], message: str) -> dict[str, object]:
    return {"config_schema": None, "digests": before, "error": message, "ok": False}


def _body(config_schema: int) -> dict[str, object]:
    body: dict[str, object] = {
        "config_schema": config_schema,
        "order_path": "paper",
        "state_schema": _STATE_OK,
    }
    if config_schema == _CONFIG_NEXT:
        body["rate_limit_rps"] = 50
    return body


def _read_config(root: Path) -> dict[str, object]:
    payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_config(root: Path, body: dict[str, object]) -> None:
    text = json.dumps(body, sort_keys=True) + "\n"
    (root / "config.json").write_text(text, encoding="utf-8")


def _schema_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
