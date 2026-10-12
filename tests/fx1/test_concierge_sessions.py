"""Session list, resume, export, and per-turn metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.interactive.model_client import ModelIdentity
from fx1.interactive.sessions import SessionError, SessionStore
from fx1.serve.backends import ToolCompletion
from tests.fx1.test_concierge import FakeBackend, make_concierge


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def test_turns_record_model_switch_and_served_model(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del env
    monkeypatch.setenv("MOONSHOT_API_KEY", "turn-secret-key")
    store = SessionStore(tmp_path / "sessions")
    backend = FakeBackend(
        [
            ToolCompletion(content="pinball loss is 0.11", tool_calls=None, finish_reason="stop"),
            ToolCompletion(content="pinball loss is 0.12", tool_calls=None, finish_reason="stop"),
        ]
    )
    backend.last_identity = ModelIdentity(
        "fx1", "fx1-served", None, "models.example.test", None, True
    )  # type: ignore[attr-defined]
    concierge, _out = make_concierge(tmp_path, backend=backend)
    concierge._session_store = store  # noqa: SLF001
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    profiles.set_endpoint("fx1-lite", "test-key-lite", "https://lite.example.test/v1")
    concierge.handle_line("please hide turn-secret-key")
    concierge.handle_line("/use fx1-lite")
    record = concierge._session  # noqa: SLF001
    assert record is not None
    turns = record["turns"]
    assert [turn["model"] for turn in turns] == ["fx1", "fx1-lite"]
    assert turns[0]["served_model"] == "fx1-served"
    assert turns[0]["outcome"] == "ok"
    assert "turn-secret-key" not in json.dumps(record)
    blob = Path(store.directory, f"{record['id']}.json").read_text()
    assert "turn-secret-key" not in blob


def test_sessions_resume_prefix_and_export_refuse_overwrite(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del env
    monkeypatch.setenv("MOONSHOT_API_KEY", "export-secret-key")
    store = SessionStore(tmp_path / "sessions")
    concierge, out = make_concierge(tmp_path)
    concierge._session_store = store  # noqa: SLF001
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    backend = FakeBackend(
        [ToolCompletion(content="pinball loss is 0.3", tool_calls=None, finish_reason="stop")]
    )
    concierge._backend_factory = lambda **_kwargs: backend  # noqa: SLF001
    concierge.handle_line("please hide export-secret-key")
    record = concierge._session  # noqa: SLF001
    assert record is not None
    prefix = str(record["id"])[:8]
    out.clear()
    concierge.handle_line("/sessions 5")
    assert prefix in _joined(out)
    fresh, _resumed = make_concierge(tmp_path)
    fresh._session_store = SessionStore(tmp_path / "sessions")  # noqa: SLF001
    fresh.handle_line(f"/resume {prefix}")
    assert any("pinball" in str(item.get("content")) for item in fresh.messages())
    assert str(fresh._session["id"]) == record["id"]  # noqa: SLF001
    out.clear()
    concierge.handle_line("/export md")
    text = _joined(out)
    assert "exported session" in text
    assert "export-secret-key" not in text
    written = next((tmp_path / "cfg" / "exports").iterdir())
    body = written.read_text()
    assert "export-secret-key" not in body
    assert written.stat().st_mode & 0o077 == 0
    out.clear()
    concierge.handle_line(f"/export md {written}")
    assert "refusing to overwrite" in _joined(out)


def test_session_conflict_warns_and_does_not_force_save(env: Path, tmp_path: Path) -> None:
    del env
    store = SessionStore(tmp_path / "sessions")
    concierge, out = make_concierge(tmp_path)
    concierge._session_store = store  # noqa: SLF001
    concierge.handle_line("/help")
    record = concierge._session  # noqa: SLF001
    assert record is not None
    other = SessionStore(tmp_path / "sessions")
    newer = other.load(str(record["id"]))
    newer["messages"].append({"role": "user", "content": "from the other writer"})
    other.save(newer)
    out.clear()
    concierge.handle_line("/model")
    assert "not saved over the newer revision" in _joined(out)
    loaded = store.load(str(record["id"]))
    assert loaded["revision"] == newer["revision"]
    assert any(
        isinstance(item, dict) and item.get("content") == "from the other writer"
        for item in loaded["messages"]
    )


def test_resume_of_an_interrupted_job_is_not_restarted(env: Path, tmp_path: Path) -> None:
    del env
    store = SessionStore(tmp_path / "sessions")
    first, _out = make_concierge(tmp_path)
    first._session_store = store  # noqa: SLF001
    first.handle_line("/help")
    session = first._session  # noqa: SLF001
    assert session is not None
    session["job"]["state"] = "running"
    store.save(session)
    second, out = make_concierge(tmp_path)
    second._session_store = store  # noqa: SLF001
    second.handle_line("/resume last")
    text = _joined(out)
    assert "not restarted" in text
    assert "/evidence" in text and "/sessions" in text
    assert second.job_running is False
    loaded = store.load(str(session["id"]))
    assert loaded["job"]["state"] == "interrupted"


def test_new_session_keeps_the_model(env: Path, tmp_path: Path) -> None:
    del env
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1-lite", "lite-key", "https://lite.example.test/v1")
    store = SessionStore(tmp_path / "sessions")
    concierge, out = make_concierge(tmp_path)
    concierge._session_store = store  # noqa: SLF001
    concierge.handle_line("/use fx1-lite")
    concierge.handle_line("/new")
    assert concierge.model == "fx1-lite"
    assert concierge.messages() == []
    assert "new session" in _joined(out)
    assert len(list(store.directory.glob("*.json"))) == 2


def test_restore_prefix_unknown_id_and_running_job(env: Path, tmp_path: Path) -> None:
    del env
    store = SessionStore(tmp_path / "sessions")
    first, _out = make_concierge(tmp_path)
    first._session_store = store  # noqa: SLF001
    first._messages.append({"role": "user", "content": "kept line"})  # noqa: SLF001
    first._persist_session()  # noqa: SLF001
    record = first._session  # noqa: SLF001
    assert record is not None
    record["job"]["state"] = "running"
    store.save(record)
    session_id = str(record["id"])

    resumed, _ignored = make_concierge(tmp_path)
    resumed._session_store = SessionStore(tmp_path / "sessions")  # noqa: SLF001
    resumed.restore(session_id[:8], preferred_model="fx1-lite")
    assert resumed.messages()[0]["content"] == "kept line"
    assert resumed.model == "fx1-lite"
    assert session_id in resumed.resume_notice
    assert "not restarted" in resumed.resume_notice
    assert resumed.job_running is False
    assert store.load(session_id)["job"]["state"] == "interrupted"

    planted = "sk-" + ("abCD12" * 4)
    with pytest.raises(SessionError) as caught:
        resumed.restore(planted)
    assert planted not in str(caught.value)

    reloaded = store.load(session_id)
    reloaded["job"]["state"] = "running"
    store.save(reloaded)
    latest, _again = make_concierge(tmp_path)
    latest._session_store = SessionStore(tmp_path / "sessions")  # noqa: SLF001
    latest.restore_latest()
    assert latest.messages()[0]["content"] == "kept line"
    assert "not restarted" in latest.resume_notice
    assert latest.job_running is False
    assert store.load(session_id)["job"]["state"] == "interrupted"

    by_last, _last = make_concierge(tmp_path)
    by_last._session_store = SessionStore(tmp_path / "sessions")  # noqa: SLF001
    by_last.restore("last")
    assert by_last.messages()[0]["content"] == "kept line"
    assert str(by_last._session["id"]) == session_id  # noqa: SLF001

    unconfigured, _none = make_concierge(tmp_path)
    unconfigured.restore(session_id)
    assert unconfigured.resume_notice == "session store is not configured"
    unconfigured.restore_latest()
    assert unconfigured.resume_notice == "session store is not configured"
    empty, _empty = make_concierge(tmp_path)
    empty._session_store = SessionStore(tmp_path / "empty-sessions")  # noqa: SLF001
    empty.restore_latest()
    assert empty.resume_notice == "no saved session to resume"
