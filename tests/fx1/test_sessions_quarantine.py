"""list() quarantines corrupt session JSON and leaves a newer schema in place."""

from __future__ import annotations

import json
from pathlib import Path

from fx1.interactive.sessions import SESSION_SCHEMA, SessionStore


def _record(session_id: str, *, schema: int = SESSION_SCHEMA) -> dict[str, object]:
    return {
        "schema": schema,
        "id": session_id,
        "model": "fx1",
        "status": "open",
        "messages": [],
        "job": {"state": "idle", "id": "", "steps_completed": [], "resume": "none"},
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }


def _write(directory: Path, record: dict[str, object]) -> Path:
    path = directory / f"{record['id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_list_quarantines_corrupt_json(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    good = "a" * 32
    corrupt = "b" * 32
    _write(tmp_path, _record(good))
    bad = tmp_path / f"{corrupt}.json"
    bad.write_text("{", encoding="utf-8")
    summaries, warnings = store.list()
    assert [item["id"] for item in summaries] == [good]
    assert any("invalid session JSON" in warning for warning in warnings)
    assert not bad.exists()
    moved = [path for path in tmp_path.iterdir() if path.name.startswith(f"{bad.name}.corrupt-")]
    assert len(moved) == 1
    assert moved[0].read_text(encoding="utf-8") == "{"
    _again, later = store.list()
    assert not any("invalid session JSON" in warning for warning in later)


def test_list_leaves_a_newer_schema_unquarantined(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    future = "c" * 32
    path = _write(tmp_path, _record(future, schema=9))
    original = path.read_bytes()
    summaries, warnings = store.list()
    assert summaries == []
    assert any("newer" in warning and "left unchanged" in warning for warning in warnings)
    assert path.is_file()
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.corrupt-*")) == []
