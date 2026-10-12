"""Durable dipcatcher session store. Offline; no network."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

import fx1.interactive.sessions as sessions
from fx1.interactive.sessions import SESSION_SCHEMA, SessionError, SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def _write_raw(directory: Path, record: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    session_id = str(record["id"])
    path = directory / f"{session_id}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def _record(session_id: str, updated_at: str, *, model: str = "fx1") -> dict[str, object]:
    return {
        "schema": SESSION_SCHEMA,
        "id": session_id,
        "model": model,
        "status": "open",
        "messages": [],
        "job": {"state": "idle", "id": "", "steps_completed": [], "resume": "none"},
        "created_at": updated_at,
        "updated_at": updated_at,
    }


def test_create_load_roundtrip(store: SessionStore) -> None:
    messages = [{"role": "user", "content": "hello"}]
    created = store.create(model="fx1", messages=messages)
    messages[0]["content"] = "changed"
    assert SESSION_SCHEMA == 1
    assert created["schema"] == SESSION_SCHEMA
    assert len(created["id"]) == 32
    assert created["id"] == created["id"].lower()
    assert created["model"] == "fx1"
    assert created["status"] == "open"
    assert created["messages"] == [{"role": "user", "content": "hello"}]
    assert created["messages"] is not messages
    assert created["job"] == {
        "state": "idle",
        "id": "",
        "steps_completed": [],
        "resume": "none",
    }
    assert created["created_at"].endswith("+00:00")
    assert created["updated_at"] == created["created_at"]
    assert store.load(created["id"]) == created

    lite = store.create(model="fx1-lite")
    assert lite["model"] == "fx1-lite"
    assert lite["id"] != created["id"]
    assert store.load(lite["id"])["messages"] == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_session_file_mode_is_0600(store: SessionStore) -> None:
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = store.save(record)
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600


def test_strips_secret_keys_and_redacts_sk_prefix(store: SessionStore) -> None:
    messages = [
        {
            "role": "user",
            "content": "sk-123456789",
            "api_key": "super-secret",
            "token": "abc",
            "nested": {"password": "hunter2", "note": "sk-abcdef", "short": "sk-12345"},
        }
    ]
    created = store.create(model="fx1", messages=messages)
    assert "api_key" in messages[0]
    loaded = store.load(created["id"])
    message = loaded["messages"][0]
    assert "api_key" not in message
    assert "token" not in message
    assert message["content"] == "[redacted]"
    assert "password" not in message["nested"]
    assert message["nested"]["note"] == "[redacted]"
    assert message["nested"]["short"] == "sk-12345"
    raw = (store.directory / f"{created['id']}.json").read_text(encoding="utf-8")
    assert "api_key" not in raw
    assert "super-secret" not in raw
    assert "sk-123456789" not in raw
    assert "sk-abcdef" not in raw


def test_load_strips_secrets_already_on_disk(store: SessionStore) -> None:
    session_id = "d" * 32
    _write_raw(
        store.directory,
        {
            "schema": 1,
            "id": session_id,
            "model": "fx1",
            "messages": [{"role": "user", "api_key": "leak", "content": "sk-abcdefgh"}],
            "job": {"state": "idle", "id": "", "steps_completed": [], "resume": "none"},
            "created_at": "2020-01-01T00:00:00+00:00",
            "updated_at": "2020-01-01T00:00:00+00:00",
        },
    )
    loaded = store.load(session_id)
    assert "api_key" not in loaded["messages"][0]
    assert loaded["messages"][0]["content"] == "[redacted]"


def test_environment_secrets_are_not_stored(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_API_KEY", "env-secret-value")
    monkeypatch.setenv("SERVICE_TOKEN", "super-secret-token")
    monkeypatch.setenv("FX1_PASSWORD", "shortpw")
    created = store.create(
        model="fx1",
        messages=[
            {"role": "user", "content": "see env-secret-value now"},
            {"role": "user", "content": "token=super-secret-token;"},
            {"role": "user", "content": "shortpw"},
            {"role": "user", "content": "shortpw-extra"},
        ],
    )
    contents = [message["content"] for message in created["messages"]]
    assert contents == ["[redacted]", "[redacted]", "[redacted]", "shortpw-extra"]
    raw = (store.directory / f"{created['id']}.json").read_text(encoding="utf-8")
    assert "env-secret-value" not in raw
    assert "super-secret-token" not in raw
    assert "shortpw" not in raw.split("shortpw-extra")[0]


def test_mark_interrupted_preserves_steps(store: SessionStore) -> None:
    record = store.create(model="fx1")
    record["job"]["state"] = "running"
    record["job"]["steps_completed"] = ["load", "score"]
    store.save(record)
    updated = store.mark_interrupted(record["id"])
    assert updated["job"]["state"] == "interrupted"
    assert updated["job"]["resume"] == "manual"
    assert updated["job"]["steps_completed"] == ["load", "score"]
    again = store.mark_interrupted(record["id"])
    assert again["job"]["state"] == "interrupted"
    assert again["job"]["resume"] == "manual"
    assert again["job"]["steps_completed"] == ["load", "score"]
    assert store.load(record["id"])["job"]["state"] == "interrupted"


def test_mark_interrupted_leaves_idle_and_finished(store: SessionStore) -> None:
    idle = store.create(model="fx1")
    idle_out = store.mark_interrupted(idle["id"])
    assert idle_out["job"]["state"] == "idle"
    assert idle_out["job"]["resume"] == "none"
    assert idle_out["job"]["steps_completed"] == []

    finished = store.create(model="fx1-lite")
    finished["job"]["state"] = "finished"
    finished["job"]["steps_completed"] = ["score"]
    finished["job"]["resume"] = "none"
    store.save(finished)
    finished_out = store.mark_interrupted(finished["id"])
    assert finished_out["job"]["state"] == "finished"
    assert finished_out["job"]["resume"] == "none"
    assert finished_out["job"]["steps_completed"] == ["score"]


def test_latest_picks_newer_and_skips_corrupt(store: SessionStore) -> None:
    older = "a" * 32
    newer = "b" * 32
    corrupt = "c" * 32
    bad_schema = "e" * 32
    _write_raw(store.directory, _record(older, "2020-01-01T00:00:00+00:00"))
    _write_raw(store.directory, _record(newer, "2024-06-01T00:00:00+00:00", model="fx1-lite"))
    _write_raw(store.directory, _record(bad_schema, "2030-01-01T00:00:00+00:00"))
    schema_path = store.directory / f"{bad_schema}.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    payload["schema"] = 2
    schema_path.write_text(json.dumps(payload), encoding="utf-8")
    (store.directory / f"{corrupt}.json").write_text("{not json", encoding="utf-8")
    (store.directory / "notes.json").write_text("{", encoding="utf-8")

    found = store.latest()
    assert found is not None
    assert found["id"] == newer
    with pytest.raises(SessionError):
        store.load(corrupt)
    with pytest.raises(SessionError):
        store.load(bad_schema)


def test_latest_missing_directory_is_none(tmp_path: Path) -> None:
    assert SessionStore(tmp_path / "absent").latest() is None


def test_path_traversal_id_is_refused(store: SessionStore) -> None:
    with pytest.raises(SessionError):
        store.load("../x")
    with pytest.raises(SessionError):
        store.save({"schema": SESSION_SCHEMA, "id": "../x"})
    with pytest.raises(SessionError):
        store.save({"schema": True, "id": "a" * 32})
    with pytest.raises(SessionError):
        store.save({"id": "a" * 32})
    assert list(store.directory.glob("*.json")) == []
    assert not (store.directory.parent / "x.json").exists()


@pytest.mark.parametrize("model", ["", "FX1", "fx1-mini", "gpt-4"])
def test_unknown_model_raises(store: SessionStore, model: str) -> None:
    with pytest.raises(SessionError):
        store.create(model=model)


def test_atomic_save_leaves_no_temp_files(
    store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = store.create(model="fx1", messages=[{"role": "user", "content": "kept"}])
    assert [path.name for path in store.directory.iterdir()] == [f"{record['id']}.json"]

    def fail_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(sessions.os, "replace", fail_replace)
    record["messages"].append({"role": "user", "content": "nope"})
    with pytest.raises(SessionError):
        store.save(record)
    names = [path.name for path in store.directory.iterdir()]
    assert names == [f"{record['id']}.json"]
    assert all(not name.endswith(".tmp") for name in names)
    assert all(not name.startswith(".session-") for name in names)
    loaded = store.load(record["id"])
    assert loaded["messages"] == [{"role": "user", "content": "kept"}]


def test_append_turn_redacts_and_saves(store: SessionStore) -> None:
    record = store.create(model="fx1-lite")
    updated = store.append_turn(
        record["id"],
        {"role": "user", "content": "sk-123456789", "api_key": "nope", "note": "hi"},
    )
    assert updated["messages"] == [{"role": "user", "content": "[redacted]", "note": "hi"}]
    assert store.load(record["id"])["messages"] == updated["messages"]


def test_default_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FX1_SESSION_DIR", raising=False)
    monkeypatch.setattr(sessions.Path, "home", lambda: tmp_path)
    assert SessionStore().directory == tmp_path / ".fx1" / "sessions"
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "from-env"))
    chosen = SessionStore()
    assert chosen.directory == tmp_path / "from-env"
    created = chosen.create(model="fx1")
    assert (tmp_path / "from-env" / f"{created['id']}.json").is_file()
