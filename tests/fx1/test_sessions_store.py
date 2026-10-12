"""Production behavior for the dict session store. Offline; no network."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import fx1.interactive.sessions as sessions
from fx1.interactive.atomicstore import FileLock
from fx1.interactive.sessions import (
    DETAIL_MAX_CHARS,
    SESSION_SCHEMA,
    SessionConflict,
    SessionError,
    SessionStore,
    scrub_secret_text,
)

GOLDEN = """\
# dipcatcher session

- id: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- model: fx1-lite
- status: open
- revision: 2
- created_at: 2020-01-01T00:00:00+00:00
- updated_at: 2020-01-02T00:00:00+00:00

## Job

- state: interrupted
- resume: manual
- id: j1
- steps: score

## Turn 0 · fx1 · ok

- served_model: fx1-2026
- started_at: 2020-01-01T00:00:01+00:00
- usage: max_tokens=32, total_tokens=11

**user**

hello

**assistant** · fx1

hi

**tool** · `score`

args: `{"n": 1}`

**tool result**

RESULT

tool_call_id: `call_1`

> model switch: fx1 -> fx1-lite

## Turn 1 · fx1-lite · ok

- started_at: 2020-01-01T00:00:02+00:00
- usage: total_tokens=4

**user**

switch

**assistant** · fx1-lite

lite reply
"""

_HOSTILE = (
    "../x",
    "..",
    ".",
    "",
    "a/b",
    "a\\b",
    "/tmp/x",
    "A" * 32,
    "g" * 32,
    "s-../../../etc/passwd",
    "s-20261012T024500Z-zzzzzzzz",
    "s-20261012T024500Z-abcdef1",
    "id\x00bad",
    "s-20261012T024500Z-abcdef12.json",
)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def _write(directory: Path, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    session_id = str(record["id"])
    (directory / f"{session_id}.json").write_text(json.dumps(record), encoding="utf-8")


def _shell(
    session_id: str, updated_at: str = "2020-01-01T00:00:00+00:00", **extra: Any
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "id": session_id,
        "model": "fx1",
        "status": "open",
        "messages": [],
        "job": {"state": "idle", "id": "", "steps_completed": [], "resume": "none"},
        "created_at": updated_at,
        "updated_at": updated_at,
    }
    record.update(extra)
    return record


def _join(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()


def test_create_revision_zero_then_save_bumps(store: SessionStore) -> None:
    record = store.create(model="fx1")
    assert record["revision"] == 0
    assert record["turns"] == []
    assert record["truncated"] is False
    store.save(record)
    assert record["revision"] == 1
    assert store.load(record["id"])["revision"] == 1


def test_frozen_clock_stamps_match(tmp_path: Path) -> None:
    when = datetime(2024, 1, 2, tzinfo=UTC)
    store = SessionStore(tmp_path, clock=lambda: when)
    record = store.create(model="fx1")
    assert record["created_at"] == "2024-01-02T00:00:00+00:00"
    assert record["updated_at"] == record["created_at"]


def test_conflict_and_force(store: SessionStore) -> None:
    created = store.create(model="fx1")
    first = store.load(created["id"])
    second = store.load(created["id"])
    first["messages"] = [{"role": "user", "content": "first"}]
    store.save(first)
    second["messages"] = [{"role": "user", "content": "second"}]
    with pytest.raises(SessionConflict) as caught:
        store.save(second)
    assert caught.value.expected == 0
    assert caught.value.found == 1
    assert issubclass(SessionConflict, SessionError)
    assert store.load(created["id"])["messages"][0]["content"] == "first"
    store.save(second, force=True)
    loaded = store.load(created["id"])
    assert loaded["messages"][0]["content"] == "second"
    assert loaded["revision"] == 2


def test_conflict_leaves_caller_revision(store: SessionStore) -> None:
    record = store.create(model="fx1")
    stale = store.load(record["id"])
    store.save(record)
    stale["status"] = "closed"
    with pytest.raises(SessionConflict):
        store.save(stale)
    assert stale["revision"] == 0
    assert stale["status"] == "closed"
    assert store.load(record["id"])["revision"] == 1


def test_missing_file_with_advanced_revision_conflicts(store: SessionStore) -> None:
    record = store.create(model="fx1")
    store.save(record)
    store.save(record)
    assert record["revision"] == 2
    assert store.delete(record["id"]) is True
    record["status"] = "closed"
    with pytest.raises(SessionConflict) as caught:
        store.save(record)
    assert caught.value.found is None
    assert caught.value.expected == 2
    store.save(record, force=True)
    assert store.load(record["id"])["revision"] == 3
    assert store.load(record["id"])["status"] == "closed"


def test_future_schema_is_never_overwritten(store: SessionStore) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "safe"}])
    path = store.directory / f"{record['id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    with pytest.raises(SessionError, match="left unchanged"):
        store.load(str(record["id"]))
    fresh = dict(record)
    fresh["schema"] = 1
    fresh["revision"] = 0
    with pytest.raises(SessionError, match="left unchanged"):
        store.save(fresh, force=True)
    assert path.read_text(encoding="utf-8") == before
    assert not list(store.directory.glob("*.corrupt-*"))


def test_corrupt_file_is_not_overwritten(store: SessionStore) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "kept"}])
    path = store.directory / f"{record['id']}.json"
    path.write_text("{not json", encoding="utf-8")
    record["messages"].append({"role": "user", "content": "nope"})
    with pytest.raises(SessionError):
        store.save(record)
    with pytest.raises(SessionError):
        store.save(record, force=True)
    assert path.read_text(encoding="utf-8") == "{not json"


def test_quarantine_moves_corrupt_json(store: SessionStore) -> None:
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(SessionError, match="invalid session JSON"):
        store.load(str(record["id"]), quarantine=True)
    assert not path.exists()
    assert list(store.directory.glob(f"{path.name}.corrupt-*"))


def test_load_without_quarantine_keeps_corrupt_file(store: SessionStore) -> None:
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(SessionError):
        store.load(str(record["id"]))
    assert path.read_text(encoding="utf-8") == "{"


def test_schema_zero_migrates_on_load_and_save(store: SessionStore) -> None:
    session_id = "d" * 32
    raw = _shell(
        session_id,
        schema=0,
        messages=[{"role": "user", "content": "legacy", "api_key": "leak"}],
    )
    _write(store.directory, raw)
    loaded = store.load(session_id)
    assert loaded["schema"] == 1
    assert loaded["revision"] == 0
    assert loaded["turns"] == []
    assert "api_key" not in loaded["messages"][0]
    disk = json.loads((store.directory / f"{session_id}.json").read_text(encoding="utf-8"))
    assert disk["schema"] == 0
    assert "api_key" in json.dumps(disk)
    store.save(loaded)
    disk = json.loads((store.directory / f"{session_id}.json").read_text(encoding="utf-8"))
    assert disk["schema"] == 1
    assert disk["revision"] == 1
    assert "api_key" not in json.dumps(disk)
    assert "leak" not in json.dumps(disk)


def test_old_format_without_revision_or_turns(store: SessionStore) -> None:
    session_id = "e" * 32
    raw = _shell(session_id, messages=[{"role": "user", "content": "legacy"}])
    assert "revision" not in raw
    assert "turns" not in raw
    _write(store.directory, raw)
    loaded = store.load(session_id)
    assert loaded["revision"] == 0
    assert loaded["turns"] == []
    assert loaded["truncated"] is False
    assert loaded["messages"][0]["content"] == "legacy"
    store.save(loaded)
    again = store.load(session_id)
    assert again["revision"] == 1
    assert again["messages"][0]["content"] == "legacy"
    assert again["turns"] == []


def test_bool_revision_is_rejected(store: SessionStore) -> None:
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    before = path.read_text(encoding="utf-8")
    record["revision"] = True
    with pytest.raises(SessionError, match="revision"):
        store.save(record)
    assert path.read_text(encoding="utf-8") == before


def test_nan_does_not_destroy_file(store: SessionStore) -> None:
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    before = path.read_text(encoding="utf-8")
    record["budget"] = {"max_seconds": float("nan")}
    with pytest.raises(SessionError):
        store.save(record)
    assert path.read_text(encoding="utf-8") == before


def test_non_json_value_does_not_destroy_file(store: SessionStore) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "kept"}])
    path = store.directory / f"{record['id']}.json"
    before = path.read_text(encoding="utf-8")
    record["messages"] = [{"role": "user", "content": object()}]
    with pytest.raises(SessionError):
        store.save(record)
    assert path.read_text(encoding="utf-8") == before
    assert store.load(str(record["id"]))["messages"][0]["content"] == "kept"


def test_redaction_keeps_numbers_and_drops_secret_strings(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_API_KEY", "registered-secret-value")
    created = store.create(
        model="fx1",
        messages=[
            {
                "role": "user",
                "content": "plain",
                "token": "abc",
                "api_key": "super-secret",
                "max_tokens": 16,
                "total_tokens": 4,
                "enabled": False,
                "note": "see registered-secret-value now",
            }
        ],
    )
    created["usage"] = {"max_tokens": 128, "total_tokens": 40, "prompt_tokens": 30}
    created["budget"] = {"max_steps": 5, "max_seconds": 1.5, "include": False}
    store.save(created)
    message = store.load(str(created["id"]))["messages"][0]
    assert "token" not in message
    assert "api_key" not in message
    assert message["max_tokens"] == 16
    assert message["total_tokens"] == 4
    assert message["enabled"] is False
    assert message["note"] == "[redacted]"
    loaded = store.load(str(created["id"]))
    assert loaded["usage"] == {"max_tokens": 128, "total_tokens": 40, "prompt_tokens": 30}
    assert loaded["budget"]["max_steps"] == 5
    assert loaded["budget"]["max_seconds"] == 1.5
    assert loaded["budget"]["include"] is False
    raw = (store.directory / f"{created['id']}.json").read_text(encoding="utf-8")
    assert "registered-secret-value" not in raw
    assert "super-secret" not in raw


def test_redaction_shapes_bearer_url_and_env_secret(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERVICE_TOKEN", "bearer-env-secret")
    token = "b" * 24
    record = store.create(
        model="fx1-lite",
        messages=[
            {
                "role": "user",
                "content": f"Bearer {token} https://alice:s3cret@example.com/v1",
            }
        ],
    )
    blob = json.dumps(store.load(str(record["id"])))
    assert token not in blob
    assert "s3cret" not in blob
    assert "alice" not in blob
    assert "Bearer [REDACTED]" in blob
    assert "https://[REDACTED]@" in blob
    assert record["messages"][0]["content"] != f"Bearer {token}"


def test_scrub_secret_text_pins_substring_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_API_KEY", "env-secret-value")
    monkeypatch.setenv("FX1_PASSWORD", "shortpw")
    assert scrub_secret_text("see env-secret-value now") == "see [redacted] now"
    assert scrub_secret_text("shortpw") == "shortpw"
    assert scrub_secret_text("shortpw-extra") == "shortpw-extra"


def test_directory_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FX1_SESSION_DIR", raising=False)
    monkeypatch.delenv("FX1_CONFIG_DIR", raising=False)
    monkeypatch.setattr(sessions.Path, "home", lambda: tmp_path)
    assert SessionStore().directory == tmp_path / ".fx1" / "sessions"
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    assert SessionStore().directory == tmp_path / "cfg" / "sessions"
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "explicit"))
    chosen = SessionStore()
    assert chosen.directory == tmp_path / "explicit"
    created = chosen.create(model="fx1-lite")
    assert (tmp_path / "explicit" / f"{created['id']}.json").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_posix_file_and_directory_modes(store: SessionStore) -> None:
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700


@pytest.mark.parametrize("bad", _HOSTILE)
def test_hostile_ids_do_not_escape(store: SessionStore, bad: str, tmp_path: Path) -> None:
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    with pytest.raises(SessionError):
        store.load(bad)
    with pytest.raises(SessionError):
        store.delete(bad)
    with pytest.raises(SessionError):
        store.save({"schema": SESSION_SCHEMA, "id": bad, "model": "fx1"})
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after == before
    assert not (tmp_path / "x.json").exists()


def test_kernel_session_id_roundtrip(store: SessionStore) -> None:
    session_id = "s-20261012T024500Z-abcdef12"
    record = _shell(session_id, messages=[{"role": "user", "content": "kernel id"}])
    path = store.save(record)
    assert path.name == f"{session_id}.json"
    loaded = store.load(session_id)
    assert loaded["revision"] == 1
    assert loaded["messages"][0]["content"] == "kernel id"
    assert store.resolve_prefix(session_id) == session_id
    assert store.resolve_prefix("s-20261012T024500Z-abcdef") == session_id


def test_list_orders_newest_and_warns(store: SessionStore) -> None:
    older = "a" * 32
    newer = "b" * 32
    future = "c" * 32
    _write(store.directory, _shell(older, "2020-01-01T00:00:00+00:00"))
    _write(store.directory, _shell(newer, "2024-06-01T00:00:00+00:00", model="fx1-lite"))
    _write(store.directory, _shell(future, "2030-01-01T00:00:00+00:00", schema=9))
    (store.directory / "notes.json").write_text("{}", encoding="utf-8")
    (store.directory / f"{'d' * 32}.json").write_text("{", encoding="utf-8")
    summaries, warnings = store.list()
    assert [item["id"] for item in summaries] == [newer, older]
    assert summaries[0]["model"] == "fx1-lite"
    assert summaries[0]["n_messages"] == 0
    assert any("notes.json" in warning and "malformed" in warning for warning in warnings)
    assert any("left unchanged" in warning for warning in warnings)
    assert any("invalid session JSON" in warning for warning in warnings)
    limited, again = store.list(limit=1)
    assert [item["id"] for item in limited] == [newer]
    assert len(again) == len(warnings)


def test_list_rejects_negative_and_honors_zero(store: SessionStore) -> None:
    _write(store.directory, _shell("a" * 32))
    (store.directory / "notes.json").write_text("{", encoding="utf-8")
    summaries, warnings = store.list(limit=0)
    assert summaries == []
    assert warnings
    with pytest.raises(SessionError, match="limit"):
        store.list(limit=-1)


def test_summaries_omit_message_bodies(store: SessionStore) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "secret-body"}])
    summaries, _warnings = store.list()
    assert summaries[0]["id"] == record["id"]
    assert "messages" not in summaries[0]
    assert "secret-body" not in json.dumps(summaries)


def test_delete_roundtrip(store: SessionStore) -> None:
    record = store.create(model="fx1")
    assert store.delete(str(record["id"])) is True
    assert store.delete(str(record["id"])) is False
    with pytest.raises(SessionError, match="missing"):
        store.load(str(record["id"]))
    with pytest.raises(SessionError):
        store.delete("../x")


def test_prune_by_age(store: SessionStore) -> None:
    old = "a" * 32
    mid = "b" * 32
    fresh = "c" * 32
    _write(store.directory, _shell(old, "2020-01-01T00:00:00+00:00"))
    _write(store.directory, _shell(mid, "2025-12-01T00:00:00+00:00"))
    _write(store.directory, _shell(fresh, "2026-01-09T00:00:00+00:00"))
    removed = store.prune(
        max_sessions=10,
        max_age_days=30,
        clock=lambda: datetime(2026, 1, 10, tzinfo=UTC),
    )
    assert removed == 2
    assert store.load(fresh)["id"] == fresh
    with pytest.raises(SessionError):
        store.load(old)


def test_prune_by_count(store: SessionStore) -> None:
    kept: list[str] = []
    for day in range(1, 6):
        session_id = str(day) * 32
        _write(store.directory, _shell(session_id, f"2026-01-0{day}T00:00:00+00:00"))
        if day >= 4:
            kept.append(session_id)
    removed = store.prune(
        max_sessions=2,
        max_age_days=3650,
        clock=lambda: datetime(2026, 1, 10, tzinfo=UTC),
    )
    assert removed == 3
    summaries, _warnings = store.list()
    assert [item["id"] for item in summaries] == list(reversed(kept))


def test_prune_rejects_negative_bounds(store: SessionStore) -> None:
    with pytest.raises(SessionError):
        store.prune(max_sessions=-1)
    with pytest.raises(SessionError):
        store.prune(max_age_days=-1)


def test_resolve_prefix_last_unique_and_ambiguous(store: SessionStore) -> None:
    id1 = "ab" + "0" * 30
    id2 = "ac" + "0" * 30
    id3 = "b" + "0" * 31
    _write(store.directory, _shell(id1, "2020-01-01T00:00:00+00:00"))
    _write(store.directory, _shell(id2, "2021-01-01T00:00:00+00:00"))
    _write(store.directory, _shell(id3, "2024-01-01T00:00:00+00:00", model="fx1-lite"))
    assert store.resolve_prefix("last") == id3
    assert store.resolve_prefix(id1) == id1
    assert store.resolve_prefix("ab") == id1
    assert store.resolve_prefix("b") == id3
    with pytest.raises(SessionError, match="ambiguous"):
        store.resolve_prefix("a")


def test_resolve_empty_and_unknown(store: SessionStore, tmp_path: Path) -> None:
    with pytest.raises(SessionError, match="empty"):
        store.resolve_prefix("   ")
    with pytest.raises(SessionError, match="no session"):
        store.resolve_prefix("zzz")
    with pytest.raises(SessionError, match="no sessions"):
        SessionStore(tmp_path / "empty").resolve_prefix("last")


def _golden_record() -> dict[str, Any]:
    return {
        "schema": 1,
        "id": "a" * 32,
        "model": "fx1-lite",
        "status": "open",
        "revision": 2,
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-02T00:00:00+00:00",
        "truncated": False,
        "job": {
            "state": "interrupted",
            "id": "j1",
            "steps_completed": ["score"],
            "resume": "manual",
        },
        "turns": [
            {
                "index": 0,
                "model": "fx1",
                "served_model": "fx1-2026",
                "usage": {"max_tokens": 32, "total_tokens": 11},
                "outcome": "ok",
                "started_at": "2020-01-01T00:00:01+00:00",
                "detail": "",
            },
            {
                "index": 1,
                "model": "fx1-lite",
                "served_model": None,
                "usage": {"total_tokens": 4},
                "outcome": "ok",
                "started_at": "2020-01-01T00:00:02+00:00",
                "detail": "",
            },
        ],
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "hi",
                "tool_calls": [{"function": {"name": "score", "arguments": '{"n": 1}'}}],
            },
            {"role": "tool", "content": "RESULT", "tool_call_id": "call_1"},
            {"role": "user", "content": "switch"},
            {"role": "assistant", "content": "lite reply"},
        ],
    }


def test_export_markdown_golden(store: SessionStore) -> None:
    assert store.export_markdown(_golden_record()) == GOLDEN


def test_export_switch_comes_from_turn_metadata(store: SessionStore) -> None:
    record = store.create(model="fx1")
    store.append_turn(
        record,
        model="fx1",
        served_model="fx1",
        usage={"total_tokens": 9, "max_tokens": 32},
        outcome="ok",
    )
    store.append_turn(record, model="fx1-lite", usage={"total_tokens": 3})
    text = store.export_markdown(store.load(str(record["id"])))
    assert "> model switch: fx1 -> fx1-lite" in text
    assert "## Turn 0 · fx1 · ok" in text
    assert "## Turn 1 · fx1-lite · ok" in text
    assert "total_tokens=9" in text
    assert "max_tokens=32" in text


def test_export_without_turns_does_not_invent_a_switch(store: SessionStore) -> None:
    record = store.create(
        model="fx1",
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "from fx1"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "from lite"},
        ],
    )
    text = store.export_markdown(record)
    assert "model switch" not in text
    assert "**assistant** · fx1" in text
    assert "**assistant** · fx1-lite" not in text


def test_export_json_keeps_numbers_and_drops_secrets(store: SessionStore) -> None:
    record = _golden_record()
    record["messages"][0]["api_key"] = "sekrit-value-1"
    record["usage"] = {"max_tokens": 128, "total_tokens": 40}
    text = store.export_json(record)
    parsed = json.loads(text)
    assert text.endswith("\n")
    assert "sekrit-value-1" not in text
    assert "api_key" not in text
    assert parsed["usage"]["max_tokens"] == 128
    assert parsed["usage"]["total_tokens"] == 40
    assert parsed["turns"][0]["usage"]["max_tokens"] == 32


def test_export_does_not_mutate_or_include_tool_secrets(store: SessionStore) -> None:
    record = store.create(model="fx1")
    record["messages"] = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "function": {
                        "name": "score",
                        "arguments": '{"api_key": "sekrit-value-1", "n": 1}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "R" * 600, "tool_call_id": "call_1"},
    ]
    before = copy.deepcopy(record)
    text = store.export_markdown(record)
    assert record == before
    assert "sekrit-value-1" not in text
    assert "**tool** · `score`" in text
    assert "[truncated: 600 chars]" in text
    assert "R" * 600 not in text
    store.export_json(record)
    assert record == before


def test_append_turn_metadata_persists_usage(store: SessionStore) -> None:
    record = store.create(model="fx1")
    updated = store.append_turn(
        record,
        model="fx1",
        served_model="fx1",
        usage={"total_tokens": 9, "max_tokens": 32},
        outcome="ok",
        detail="done",
        started_at="2020-01-01T00:00:03+00:00",
    )
    turn = updated["turns"][0]
    assert turn["index"] == 0
    assert turn["model"] == "fx1"
    assert turn["served_model"] == "fx1"
    assert turn["usage"] == {"total_tokens": 9, "max_tokens": 32}
    assert turn["started_at"] == "2020-01-01T00:00:03+00:00"
    assert turn["detail"] == "done"
    loaded = store.load(str(record["id"]))
    assert loaded["turns"][0]["usage"]["max_tokens"] == 32
    assert loaded["revision"] == 1


def test_detail_is_capped_and_redacted(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = store.create(model="fx1")
    store.append_turn(record, model="fx1", detail="d" * 600)
    assert len(store.load(str(record["id"]))["turns"][0]["detail"]) == DETAIL_MAX_CHARS
    monkeypatch.setenv("FX1_API_KEY", "detail-secret-value")
    store.append_turn(record, model="fx1-lite", detail="leak detail-secret-value please")
    blob = json.dumps(store.load(str(record["id"])))
    assert "detail-secret-value" not in blob
    assert store.load(str(record["id"]))["turns"][1]["model"] == "fx1-lite"


def test_size_policy_truncates_old_tool_results(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, max_bytes=1800)
    old_tool = "T" * 4000
    latest_tool = "LATEST_TOOL_" + "Z" * 80
    record = store.create(
        model="fx1",
        messages=[
            {"role": "user", "content": "keep-user"},
            {"role": "assistant", "content": "old-answer"},
            {"role": "tool", "content": old_tool, "tool_call_id": "c1"},
            {"role": "user", "content": "keep-latest"},
            {"role": "assistant", "content": "latest-answer"},
            {"role": "tool", "content": latest_tool, "tool_call_id": "c2"},
        ],
    )
    assert record["truncated"] is True
    assert record["messages"][0]["content"] == "keep-user"
    assert record["messages"][2]["content"] == "[truncated: 4000 chars]"
    assert record["messages"][3]["content"] == "keep-latest"
    assert record["messages"][5]["content"] == latest_tool
    path = store.directory / f"{record['id']}.json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["messages"][5]["content"] == latest_tool
    assert old_tool not in path.read_text(encoding="utf-8")


def test_size_policy_keeps_latest_turn_and_valid_json(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, max_bytes=400)
    huge = "U" * 5000
    record = store.create(model="fx1", messages=[{"role": "user", "content": huge}])
    assert record["truncated"] is False
    assert record["messages"][0]["content"] == huge
    path = store.directory / f"{record['id']}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["messages"][0]["content"] == huge
    assert path.stat().st_size > 400


def test_size_policy_skips_unfinished_older_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, max_bytes=800)
    record = store.create(model="fx1")
    record["messages"] = [
        {"role": "user", "content": "old"},
        {"role": "tool", "content": "T" * 3000, "tool_call_id": "c"},
        {"role": "user", "content": "new"},
    ]
    record["turns"] = [
        {"index": 0, "model": "fx1", "outcome": "", "started_at": "2020-01-01T00:00:00+00:00"},
        {"index": 1, "model": "fx1", "outcome": "ok", "started_at": "2020-01-01T00:00:01+00:00"},
    ]
    store.save(record)
    assert record["truncated"] is False
    assert record["messages"][1]["content"] == "T" * 3000
    parsed = json.loads((store.directory / f"{record['id']}.json").read_text(encoding="utf-8"))
    assert parsed["messages"][1]["content"] == "T" * 3000


def test_threads_append_without_lost_updates(store: SessionStore) -> None:
    record = store.create(model="fx1")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker(text: str) -> None:
        try:
            barrier.wait(2)
            store.append_turn(str(record["id"]), {"role": "user", "content": text})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(text,)) for text in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    _join(threads)
    assert errors == []
    loaded = store.load(str(record["id"]))
    assert {item["content"] for item in loaded["messages"]} == {"alpha", "beta"}


def test_threads_save_one_conflict(store: SessionStore) -> None:
    record = store.create(model="fx1")
    left = store.load(str(record["id"]))
    right = store.load(str(record["id"]))
    left["messages"] = [{"role": "user", "content": "left"}]
    right["messages"] = [{"role": "user", "content": "right"}]
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    saved: list[str] = []

    def worker(payload: dict[str, Any], label: str) -> None:
        try:
            barrier.wait(2)
            store.save(payload)
        except SessionConflict as exc:
            errors.append(exc)
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            return
        saved.append(label)

    threads = [
        threading.Thread(target=worker, args=(left, "left")),
        threading.Thread(target=worker, args=(right, "right")),
    ]
    for thread in threads:
        thread.start()
    _join(threads)
    assert len(saved) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], SessionConflict)
    loaded = store.load(str(record["id"]))
    assert loaded["messages"][0]["content"] == saved[0]


def test_processes_append_without_lost_updates(store: SessionStore) -> None:
    record = store.create(model="fx1")
    src = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, os.environ['WP2C_SRC'])\n"
        "from fx1.interactive.sessions import SessionStore\n"
        "store = SessionStore(Path(os.environ['WP2C_DIR']))\n"
        "store.append_turn(os.environ['WP2C_ID'],"
        " {'role': 'user', 'content': os.environ['WP2C_TEXT']})\n"
    )
    procs: list[subprocess.Popen[str]] = []
    for text in ("alpha", "beta"):
        env = os.environ.copy()
        env["WP2C_SRC"] = str(src)
        env["WP2C_DIR"] = str(store.directory)
        env["WP2C_ID"] = str(record["id"])
        env["WP2C_TEXT"] = text
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for proc in procs:
        _out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, err
    loaded = store.load(str(record["id"]))
    assert {item["content"] for item in loaded["messages"]} == {"alpha", "beta"}


def test_lock_timeout(store: SessionStore) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "kept"}])
    path = store.directory / f"{record['id']}.json"
    before = path.read_text(encoding="utf-8")
    slow = SessionStore(store.directory, lock_timeout_s=0.05)
    with FileLock(path, timeout_s=1), pytest.raises(SessionError, match="lock"):
        slow.save(record)
    assert path.read_text(encoding="utf-8") == before


def test_distinct_sessions_append_concurrently(store: SessionStore) -> None:
    first = store.create(model="fx1")
    second = store.create(model="fx1-lite")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker(session_id: str, text: str) -> None:
        try:
            barrier.wait(2)
            store.append_turn(session_id, {"role": "user", "content": text})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(str(first["id"]), "one")),
        threading.Thread(target=worker, args=(str(second["id"]), "two")),
    ]
    for thread in threads:
        thread.start()
    _join(threads)
    assert errors == []
    assert store.load(str(first["id"]))["messages"][0]["content"] == "one"
    assert store.load(str(second["id"]))["messages"][0]["content"] == "two"


def test_single_writer_save_loop(store: SessionStore) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "keep me"}])
    record["messages"] = [
        {"role": "user", "content": "keep me"},
        {"role": "assistant", "content": "ok"},
    ]
    record["model"] = "fx1-lite"
    record["budget"] = {"max_steps": 2, "max_seconds": 3.5, "max_retries": 1}
    record["job"]["state"] = "running"
    store.save(record)
    record["job"]["state"] = "finished"
    record["job"]["steps_completed"] = ["score"]
    store.save(record)
    loaded = store.load(str(record["id"]))
    assert loaded["revision"] == 2
    assert loaded["model"] == "fx1-lite"
    assert loaded["budget"]["max_steps"] == 2
    assert loaded["budget"]["max_seconds"] == 3.5
    assert loaded["job"]["state"] == "finished"
    assert loaded["job"]["steps_completed"] == ["score"]
    assert [item["content"] for item in loaded["messages"]] == ["keep me", "ok"]


def test_emit_events_omit_message_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_API_KEY", "emit-secret-value")
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(name: str, fields: dict[str, Any]) -> None:
        events.append((name, dict(fields)))

    store = SessionStore(tmp_path, emit=emit)
    record = store.create(model="fx1", messages=[{"role": "user", "content": "emit-secret-value"}])
    stale = store.load(str(record["id"]))
    store.save(record)
    stale["status"] = "closed"
    with pytest.raises(SessionConflict):
        store.save(stale)
    blob = json.dumps(events)
    assert "emit-secret-value" not in blob
    assert [name for name, _fields in events] == [
        "session.created",
        "session.saved",
        "session.conflict",
    ]
    conflict = events[-1][1]
    assert conflict["expected"] == 0
    assert conflict["found"] == 1
    assert "content" not in conflict


def test_malformed_filename_is_warned(store: SessionStore) -> None:
    (store.directory).mkdir(parents=True, exist_ok=True)
    (store.directory / "notes.json").write_text("{}", encoding="utf-8")
    summaries, warnings = store.list()
    assert summaries == []
    assert any("notes.json" in warning for warning in warnings)
    assert store.latest() is None
