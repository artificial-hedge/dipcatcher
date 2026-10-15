"""Per-session journal — append-only chat log, redaction, resume. Offline."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import fx1
from fx1.interactive.session import (
    SCHEMA_VERSION,
    SessionEvent,
    SessionJournal,
    list_sessions,
    load_header,
    load_session,
    new_session_id,
    reconcile_interrupted,
    redact,
    sessions_dir,
    transcript,
)

_REDACTED = "[REDACTED …{tail}]"


def _mask(tail: str) -> str:
    return _REDACTED.format(tail=tail)


def _journal(directory: Path, session_id: str | None = None, **kwargs: Any) -> SessionJournal:
    kwargs.setdefault("code_revision", "abc1234")
    return SessionJournal(session_id, sessions_dir=directory, **kwargs)


def _header(journal: SessionJournal) -> dict[str, Any]:
    raw = json.loads(journal.path.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(raw, dict)
    return raw


def _write_session(
    directory: Path,
    session_id: str,
    started_at: str,
    events: list[dict[str, Any]],
    *,
    model: str = "fx1",
    endpoint_host: str = "api.example",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.jsonl"
    header = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "started_at": started_at,
        "model": model,
        "endpoint_host": endpoint_host,
        "code_revision": "abc1234",
        "fx1_version": fx1.__version__,
    }
    lines = [json.dumps(header), *[json.dumps(event) for event in events]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _event(
    seq: int,
    kind: str,
    data: dict[str, Any],
    ts: str = "2026-10-12T00:00:00+00:00",
) -> dict[str, Any]:
    return {"seq": seq, "ts": ts, "kind": kind, "data": data}


def test_header_records_schema_model_and_version(tmp_path: Path) -> None:
    journal = _journal(
        tmp_path,
        "header-session",
        model="fx1-lite",
        endpoint_host="api.example",
    )
    header = _header(journal)
    assert list(header) == [
        "schema_version",
        "session_id",
        "started_at",
        "model",
        "endpoint_host",
        "code_revision",
        "fx1_version",
    ]
    assert header["schema_version"] == SCHEMA_VERSION == 1
    assert header["session_id"] == "header-session"
    assert header["model"] == "fx1-lite"
    assert header["endpoint_host"] == "api.example"
    assert header["code_revision"] == "abc1234"
    assert header["fx1_version"] == fx1.__version__
    started = datetime.fromisoformat(header["started_at"])
    assert started.utcoffset() == timedelta(0)
    assert journal.path == tmp_path / "header-session.jsonl"
    assert journal.session_id == "header-session"
    assert journal.write_errors == []

    defaults = _journal(tmp_path, "defaults")
    default_header = _header(defaults)
    assert default_header["model"] == "fx1"
    assert default_header["endpoint_host"] == "unconfigured"


def test_header_redacts_secret_shaped_model_and_host(tmp_path: Path) -> None:
    secret = "sk-" + ("z" * 20)
    journal = _journal(
        tmp_path,
        "secret-header",
        model=secret,
        endpoint_host="Bearer abcdefghijklmnop",
    )
    blob = journal.path.read_text(encoding="utf-8")
    assert secret not in blob
    assert "abcdefghijklmnop" not in blob
    assert "sk-" not in blob
    header = _header(journal)
    assert header["model"] == _mask("zzzz")
    assert header["endpoint_host"] == _mask("mnop")
    assert header["code_revision"] == "abc1234"

    leaky_rev = "sk-" + ("q" * 24) + "9xyz"
    leaky = _journal(tmp_path, "leaky-rev", code_revision=leaky_rev)
    leaky_blob = leaky.path.read_text(encoding="utf-8")
    assert leaky_rev not in leaky_blob
    assert "sk-" not in leaky_blob
    assert _header(leaky)["code_revision"] == _mask("9xyz")


def test_new_session_id_format_timezone_and_uniqueness() -> None:
    stamped = new_session_id(datetime(2026, 10, 12, 3, 4, 5, tzinfo=UTC))
    prefix, suffix = stamped.split("-", 1)
    assert prefix == "20261012T030405Z"
    assert re.fullmatch(r"[0-9a-f]{8}", suffix)

    eastern = timezone(timedelta(hours=-4))
    shifted = new_session_id(datetime(2026, 10, 12, 1, 2, 3, tzinfo=eastern))
    assert shifted.startswith("20261012T050203Z-")

    naive = new_session_id(datetime(2026, 1, 2, 3, 4, 5))
    assert naive.startswith("20260102T030405Z-")

    minted = {new_session_id() for _ in range(32)}
    assert len(minted) == 32


def test_generated_session_id_and_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = SessionJournal(sessions_dir=tmp_path, code_revision="abc1234")
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", journal.session_id)
    assert journal.path.name == f"{journal.session_id}.jsonl"

    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    assert sessions_dir() == tmp_path / "cfg" / "sessions"
    via_env = SessionJournal(session_id="env-session", code_revision="abc1234")
    assert via_env.path == tmp_path / "cfg" / "sessions" / "env-session.jsonl"

    monkeypatch.setenv("FX1_CONFIG_DIR", "~/fx1-from-home")
    assert sessions_dir() == Path.home() / "fx1-from-home" / "sessions"

    monkeypatch.setenv("FX1_CONFIG_DIR", "")
    assert sessions_dir() == Path.home() / ".fx1" / "sessions"
    monkeypatch.delenv("FX1_CONFIG_DIR", raising=False)
    assert sessions_dir() == Path.home() / ".fx1" / "sessions"


@pytest.mark.parametrize(
    "session_id",
    ["", ".", "..", "../escape", "a/b", "a\\b", "has\nnewline", "has\rreturn", "nul\x00byte"],
)
def test_rejects_unsafe_session_ids(session_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid session id"):
        SessionJournal(session_id=session_id, sessions_dir=tmp_path, code_revision="abc1234")
    assert list(tmp_path.glob("*.jsonl")) == []
    assert not (tmp_path.parent / "escape.jsonl").exists()


@pytest.mark.parametrize("session_id", ["", ".", "..", "../escape", "a/b", "a\\b"])
def test_resume_lookup_rejects_path_tokens(session_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid session id"):
        load_session(session_id, directory=tmp_path)
    with pytest.raises(ValueError, match="invalid session id"):
        load_header(session_id, directory=tmp_path)


def test_record_events_roundtrip_keeps_seq(tmp_path: Path) -> None:
    journal = _journal(tmp_path, "round")
    first = journal.turn("user", "hello")
    second = journal.turn("assistant", "world")
    assert first.seq == 1
    assert second.seq == 2
    assert journal.events() == [first, second]
    assert journal.read_errors() == []
    raw_lines = journal.path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
    event_line = json.loads(raw_lines[1])
    assert list(event_line) == ["seq", "ts", "kind", "data"]
    assert event_line["data"] == {"role": "user", "text": "hello"}


def test_record_does_not_reread_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = _journal(tmp_path, "mem-seq")
    journal.record("ping")

    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("record() must not read the journal")

    monkeypatch.setattr(Path, "read_text", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)
    event = journal.record("ping", n=2)
    assert event.seq == 2
    assert event.data == {"n": 2}
    assert journal.write_errors == []


def test_typed_recorders_kinds_and_data(tmp_path: Path) -> None:
    journal = _journal(tmp_path, "typed")
    recorded = [
        journal.turn("user", "hi"),
        journal.turn("assistant", "hello"),
        journal.turn("system", "note"),
        journal.model_switch("fx1", "fx1-lite", "api.example"),
        journal.job_started("research-01", "pinball", "estimate pinball", ["fam-a", "fam-b"]),
        journal.job_step("research-01", "fit", "ok", detail="fold 1"),
        journal.job_finished("research-01", "done", detail="pinball 0.12", artifacts=["out.json"]),
        journal.job_finished("research-02", "failed"),
        journal.approval("consequential", "run it?", True),
        journal.close(),
        journal.close("shutdown"),
    ]
    assert [event.seq for event in recorded] == list(range(1, 12))
    assert journal.events() == recorded
    assert recorded[0].kind == "turn"
    assert recorded[0].data == {"role": "user", "text": "hi"}
    assert recorded[1].data["role"] == "assistant"
    assert recorded[2].data["role"] == "system"
    assert recorded[3].kind == "model_switch"
    assert recorded[3].data == {"old": "fx1", "new": "fx1-lite", "endpoint_host": "api.example"}
    assert recorded[4].kind == "job_started"
    assert recorded[4].data == {
        "job_id": "research-01",
        "label": "pinball",
        "goal": "estimate pinball",
        "plan_ids": ["fam-a", "fam-b"],
    }
    assert recorded[5].kind == "job_step"
    assert recorded[5].data == {
        "job_id": "research-01",
        "step": "fit",
        "status": "ok",
        "detail": "fold 1",
    }
    assert recorded[6].kind == "job_finished"
    assert recorded[6].data == {
        "job_id": "research-01",
        "status": "done",
        "detail": "pinball 0.12",
        "artifacts": ["out.json"],
    }
    assert recorded[7].data["artifacts"] == []
    assert recorded[7].data["detail"] == ""
    assert recorded[7].data["status"] == "failed"
    assert recorded[8].kind == "approval"
    assert recorded[8].data == {
        "approval_kind": "consequential",
        "question": "run it?",
        "approved": True,
    }
    assert recorded[9].kind == "session_end"
    assert recorded[9].data == {"reason": "exit"}
    assert recorded[10].data == {"reason": "shutdown"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see Bearer abcdefghijklmnop now", "see [REDACTED …mnop] now"),
        ("token sk-abcdefghijklmnopqrstuvwxyz", "token [REDACTED …wxyz]"),
        ("token sk_abcdefghijklmnopqrstuvwxyz", "token [REDACTED …wxyz]"),
        ("token key-abcdefghijklmnopqrstuvwxyz", "token [REDACTED …wxyz]"),
        ("export MOONSHOT_API_KEY=abcdefghijklmnop", "export [REDACTED …mnop]"),
        ("export FX1_API_KEY=abcdefghijklmnop", "export [REDACTED …mnop]"),
        ("FX1_BYOK_API_KEY=abcdefghijklmnop", "[REDACTED …mnop]"),
        ("FX1_SIGNING_KEY=abcdefghijklmnop", "[REDACTED …mnop]"),
        ("bearer AAAAaaaaBBBBbbbb", "[REDACTED …bbbb]"),
        ("sk-abcdefgh", "[REDACTED …efgh]"),
        ("Bearer aaaaaaaabbbb sk-ccccccccdddd", "[REDACTED …bbbb] [REDACTED …dddd]"),
        ("The pinball score was 0.18 on fold 3.", "The pinball score was 0.18 on fold 3."),
        ("flash id deadbeef", "flash id deadbeef"),
        ("flash id a1b2c3d4", "flash id a1b2c3d4"),
        ("https://artificialhedge.co/fx1", "https://artificialhedge.co/fx1"),
        ("ask-abcdefghijklmnop", "ask-abcdefghijklmnop"),
        ("sk-abcdefg", "sk-abcdefg"),
        ("MOONSHOT_API_KEY", "MOONSHOT_API_KEY"),
        ("b" * 40, "b" * 40),
        ("c" * 27 + "1234", "c" * 27 + "1234"),
    ],
)
def test_redact_value_patterns(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_redact_long_blob_keeps_last_four_and_one_span() -> None:
    blob = ("A" * 28) + "1234"
    assert len(blob) == 32
    assert redact(f"prefix {blob} suffix") == f"prefix {_mask('1234')} suffix"
    almost = "d" * 31 + "7"
    assert redact(almost) == _mask("ddd7")
    token = "sk-" + ("b" * 20) + "1234567890"
    assert redact(token) == _mask(token[-4:])
    assert redact(token).count("[REDACTED") == 1
    encoded = "abcdEFGH1234+/ijklMNOP5678==qrstUV"
    assert len(encoded) >= 32
    assert redact(encoded) == _mask(encoded[-4:])


def test_redact_secret_keys_nested_and_does_not_mutate() -> None:
    original: dict[str, Any] = {
        "api_key": "short",
        "APIKEY": "abcd",
        "note": "pinball 0.2",
        "api_key_id": "not-a-secret",
        "nested": {"Password": "hunter2", "ok": "deadbeef"},
        "items": [
            "Bearer zzzzzzzzzz9999",
            "hello",
            {"token": "cat", "Secret": "abcdefghij"},
        ],
    }
    snapshot = json.loads(json.dumps(original))
    redacted = redact(original)
    assert original == snapshot
    assert redacted["api_key"] == _mask("hort")
    assert redacted["APIKEY"] == _mask("abcd")
    assert redacted["note"] == "pinball 0.2"
    assert redacted["api_key_id"] == "not-a-secret"
    assert redacted["nested"]["Password"] == _mask("ter2")
    assert redacted["nested"]["ok"] == "deadbeef"
    assert redacted["items"][0] == _mask("9999")
    assert redacted["items"][1] == "hello"
    assert redacted["items"][2]["token"] == _mask("cat")
    assert redacted["items"][2]["Secret"] == _mask("ghij")
    forced = redact({"authorization": {"nested": "value"}})
    assert isinstance(forced["authorization"], str)
    assert forced["authorization"].startswith("[REDACTED …")
    assert "value" not in forced["authorization"]


def test_record_strips_secrets_before_they_hit_disk(tmp_path: Path) -> None:
    journal = _journal(tmp_path, "quiet")
    secret = "sk-" + ("a" * 24) + "wxyz"
    journal.turn("user", f"use {secret} please")
    journal.record(
        "note",
        text="Bearer abcdefghijklmnop",
        api_key="not-shown",
        nested={"password": "hunter2", "ok": "deadbeef"},
        plan="export MOONSHOT_API_KEY=abcdefghijklmnop",
    )
    blob = journal.path.read_text(encoding="utf-8")
    assert secret not in blob
    assert "sk-" not in blob
    assert "abcdefghijklmnop" not in blob
    assert "not-shown" not in blob
    assert "hunter2" not in blob
    assert "MOONSHOT_API_KEY=" not in blob
    assert "deadbeef" in blob
    assert _mask("wxyz") in blob
    assert _mask("mnop") in blob
    assert _mask("cret") not in blob
    stored = journal.events()[-1]
    assert stored.data["api_key"] == _mask("hown")
    assert stored.data["nested"]["password"] == _mask("ter2")


def test_torn_and_invalid_lines_are_reported(tmp_path: Path) -> None:
    journal = _journal(tmp_path, "torn")
    journal.turn("user", "before")
    assert journal.read_errors() == []
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
        handle.write("{not-json\n")
        handle.write("[]\n")
        handle.write('{"seq": true, "ts": "t", "kind": "turn", "data": {}}\n')
        handle.write('{"seq": "nope", "ts": "t", "kind": "turn", "data": {}}\n')
    journal.turn("user", "after")
    with journal.path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 99, "ts": "torn"')
    events = journal.events()
    assert [event.data["text"] for event in events] == ["before", "after"]
    errors = journal.read_errors()
    assert errors
    assert any("not JSON" in error for error in errors)
    assert any("not an object" in error for error in errors)
    assert any("not an event" in error for error in errors)
    assert events[0].seq == 1
    assert events[1].seq == 2


def test_list_sessions_orders_limits_and_counts(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    _write_session(
        directory,
        "alpha",
        "2026-10-01T00:00:00+00:00",
        [
            _event(1, "turn", {"role": "user", "text": "one"}),
            _event(2, "turn", {"role": "assistant", "text": "two"}),
            _event(3, "job_started", {"job_id": "job-a", "goal": "alpha-goal"}),
            _event(4, "job_finished", {"job_id": "job-a", "status": "done", "detail": "ok"}),
            _event(5, "session_end", {"reason": "exit"}, ts="2026-10-01T01:00:00+00:00"),
        ],
        model="fx1",
        endpoint_host="h1",
    )
    _write_session(
        directory,
        "middle",
        "2026-10-02T00:00:00+00:00",
        [
            _event(1, "turn", {"role": "user", "text": "a"}),
            _event(2, "turn", {"role": "user", "text": "b"}),
            _event(3, "turn", {"role": "system", "text": "c"}),
            _event(4, "session_end", {"reason": "exit"}, ts="2026-10-02T02:00:00+00:00"),
        ],
    )
    _write_session(
        directory,
        "zebra",
        "2026-10-03T00:00:00+00:00",
        [
            _event(1, "turn", {"role": "user", "text": "later"}),
            _event(2, "job_started", {"job_id": "job-b", "goal": "beta"}),
            _event(3, "job_started", {"job_id": "job-c", "goal": "gamma"}),
            _event(4, "job_finished", {"job_id": "job-b", "status": "done", "detail": "ok"}),
        ],
        model="fx1-lite",
        endpoint_host="h2",
    )
    _write_session(directory, "tie-aaa", "2026-10-04T00:00:00+00:00", [])
    _write_session(directory, "tie-zzz", "2026-10-04T00:00:00+00:00", [])
    (directory / "garbage.jsonl").write_text("{not json\n", encoding="utf-8")
    (directory / "notes.txt").write_text("ignore me", encoding="utf-8")
    (directory / "nested.jsonl").mkdir()
    (directory / "bad-encoding.jsonl").write_bytes(b"\xff\xfe\xfa")

    rows = list_sessions(directory, limit=50)
    assert [row.session_id for row in rows] == ["tie-zzz", "tie-aaa", "zebra", "middle", "alpha"]

    by_id = {row.session_id: row for row in rows}
    alpha = by_id["alpha"]
    assert alpha.turns == 2
    assert alpha.jobs_total == 1
    assert alpha.jobs_open == 0
    assert alpha.last_goal == "alpha-goal"
    assert alpha.ended_at == "2026-10-01T01:00:00+00:00"
    assert alpha.model == "fx1"
    assert alpha.endpoint_host == "h1"
    assert alpha.path == directory / "alpha.jsonl"

    zebra = by_id["zebra"]
    assert zebra.turns == 1
    assert zebra.jobs_total == 2
    assert zebra.jobs_open == 1
    assert zebra.last_goal == "gamma"
    assert zebra.ended_at is None
    assert zebra.model == "fx1-lite"
    assert zebra.endpoint_host == "h2"

    middle = by_id["middle"]
    assert middle.turns == 3
    assert middle.jobs_total == 0
    assert middle.jobs_open == 0
    assert middle.last_goal is None
    assert middle.ended_at == "2026-10-02T02:00:00+00:00"

    assert [row.session_id for row in list_sessions(directory, limit=2)] == ["tie-zzz", "tie-aaa"]
    assert list_sessions(directory, limit=0) == []
    assert list_sessions(directory, limit=-3) == []

    for index in range(22):
        _write_session(
            directory,
            f"n{index:02d}",
            f"2026-11-01T00:{index:02d}:00+00:00",
            [],
        )
    newest = [f"n{index:02d}" for index in range(21, 1, -1)]
    assert [row.session_id for row in list_sessions(directory)] == newest


def test_list_and_load_missing_directory_does_not_create_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert list_sessions(missing) == []
    assert load_session("anything", missing) is None
    assert load_header("anything", missing) is None
    assert reconcile_interrupted(missing) == []
    assert not missing.exists()


def test_load_session_prefix_exact_ambiguous_and_header(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    _write_session(
        directory,
        "abc",
        "2026-10-01T00:00:00+00:00",
        [_event(1, "turn", {"role": "user", "text": "short"})],
        model="fx1",
        endpoint_host="h-short",
    )
    _write_session(
        directory,
        "abcdef",
        "2026-10-02T00:00:00+00:00",
        [_event(1, "turn", {"role": "assistant", "text": "long"})],
        model="fx1-lite",
        endpoint_host="h-long",
    )
    _write_session(
        directory,
        "qqqq",
        "2026-10-03T00:00:00+00:00",
        [_event(1, "turn", {"role": "system", "text": "other"})],
    )

    exact = load_session("abc", directory)
    assert exact is not None
    assert len(exact) == 1
    assert exact[0].data["text"] == "short"
    assert exact[0].kind == "turn"

    prefixed = load_session("abcde", directory)
    assert prefixed is not None
    assert prefixed[0].data["text"] == "long"

    other = load_session("qq", directory)
    assert other is not None
    assert other[0].data["role"] == "system"

    assert load_session("missing", directory) is None
    with pytest.raises(ValueError, match="ambiguous") as raised:
        load_session("ab", directory)
    assert "abc" in str(raised.value)
    assert "abcdef" in str(raised.value)

    header = load_header("abcdef", directory)
    assert header is not None
    assert header["schema_version"] == 1
    assert header["model"] == "fx1-lite"
    assert header["endpoint_host"] == "h-long"
    assert header["session_id"] == "abcdef"
    assert load_header("abcdef", directory) == header
    assert load_header("nope", directory) is None
    with pytest.raises(ValueError, match="ambiguous"):
        load_header("ab", directory)


def test_reconcile_interrupted_marks_open_jobs_and_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    opened = _journal(directory, "a-open")
    opened.job_started("job-1", "one", "goal one", ["p1"])
    opened.job_started("job-2", "two", "goal two", [])
    opened.job_finished("job-2", "done", detail="ok")
    opened.job_started("job-3", "three", "goal three", [])

    also_open = _journal(directory, "m-open")
    also_open.job_started("job-9", "nine", "goal nine", [])

    current = _journal(directory, "z-current")
    current.job_started("job-c", "cur", "current goal", [])
    current_bytes = current.path.read_bytes()

    closed = _journal(directory, "b-closed")
    closed.job_started("job-x", "x", "still open but session ended", [])
    closed.close("exit")
    closed_bytes = closed.path.read_bytes()

    settled = _journal(directory, "c-settled")
    settled.job_started("job-s", "s", "done already", [])
    settled.job_finished("job-s", "done", detail="scored")
    settled_bytes = settled.path.read_bytes()

    idle = _journal(directory, "d-idle")
    idle.turn("user", "hello")
    idle_bytes = idle.path.read_bytes()

    marked = reconcile_interrupted(directory, exclude_session_id="z-current")
    assert marked == [("a-open", "job-1"), ("a-open", "job-3"), ("m-open", "job-9")]

    events = load_session("a-open", directory)
    assert events is not None
    assert [event.kind for event in events] == [
        "job_started",
        "job_started",
        "job_finished",
        "job_started",
        "job_finished",
        "job_finished",
        "session_end",
    ]
    assert [event.seq for event in events] == [1, 2, 3, 4, 5, 6, 7]
    assert events[4].data == {
        "job_id": "job-1",
        "status": "interrupted",
        "detail": "process exited before the job settled",
        "artifacts": [],
    }
    assert events[5].data["job_id"] == "job-3"
    assert events[5].data["status"] == "interrupted"
    assert events[6].kind == "session_end"
    assert events[6].data == {"reason": "interrupted"}

    more = load_session("m-open", directory)
    assert more is not None
    assert [event.seq for event in more] == [1, 2, 3]
    assert more[-1].data["reason"] == "interrupted"

    assert current.path.read_bytes() == current_bytes
    assert closed.path.read_bytes() == closed_bytes
    assert settled.path.read_bytes() == settled_bytes
    assert idle.path.read_bytes() == idle_bytes

    closed_events = load_session("b-closed", directory)
    assert closed_events is not None
    assert closed_events[-1].data["reason"] == "exit"
    current_events = load_session("z-current", directory)
    assert current_events is not None
    assert [event.kind for event in current_events] == ["job_started"]

    rows = {row.session_id: row for row in list_sessions(directory, limit=50)}
    assert rows["a-open"].jobs_open == 0
    assert rows["a-open"].jobs_total == 3
    assert rows["a-open"].last_goal == "goal three"
    assert rows["a-open"].ended_at is not None
    assert rows["z-current"].jobs_open == 1
    assert rows["z-current"].ended_at is None

    assert reconcile_interrupted(directory, exclude_session_id="z-current") == []


def test_reconcile_reopens_after_finish_and_continues_seq(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    journal = _journal(directory, "retry")
    journal.job_started("job-1", "label", "first", [])
    journal.job_finished("job-1", "failed", detail="retry")
    journal.job_started("job-1", "label", "second", [])
    marked = reconcile_interrupted(directory)
    assert marked == [("retry", "job-1")]
    events = load_session("retry", directory)
    assert events is not None
    assert events[-2].data["status"] == "interrupted"
    assert events[-2].seq == 4
    assert events[-1].seq == 5
    assert list_sessions(directory)[0].jobs_open == 0
    assert list_sessions(directory)[0].jobs_total == 2
    assert list_sessions(directory)[0].last_goal == "second"

    gapped = directory / "gapped.jsonl"
    _write_session(
        directory,
        "gapped",
        "2026-10-01T00:00:00+00:00",
        [
            _event(2, "job_started", {"job_id": "job-gap", "goal": "gap"}),
            _event(9, "turn", {"role": "user", "text": "still going"}),
        ],
    )
    assert reconcile_interrupted(directory) == [("gapped", "job-gap")]
    reloaded = load_session("gapped", directory)
    assert reloaded is not None
    assert [event.seq for event in reloaded] == [2, 9, 10, 11]
    assert reloaded[-2].kind == "job_finished"
    assert reloaded[-1].kind == "session_end"
    assert gapped.is_file()


def test_reconcile_repairs_a_torn_trailing_line(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    directory.mkdir()
    path = directory / "torn-open.jsonl"
    header = {
        "schema_version": 1,
        "session_id": "torn-open",
        "started_at": "2026-10-12T00:00:00+00:00",
        "model": "fx1",
        "endpoint_host": "api.example",
        "code_revision": "abc1234",
        "fx1_version": fx1.__version__,
    }
    event = _event(4, "job_started", {"job_id": "job-7", "goal": "g"})
    path.write_text(
        json.dumps(header) + "\n" + json.dumps(event) + "\n" + '{"seq":', encoding="utf-8"
    )
    assert reconcile_interrupted(directory) == [("torn-open", "job-7")]
    text = path.read_text(encoding="utf-8")
    assert '{"seq":\n' in text
    finished = json.loads(text.splitlines()[-2])
    ended = json.loads(text.splitlines()[-1])
    assert finished["seq"] == 5
    assert finished["kind"] == "job_finished"
    assert finished["data"]["detail"] == "process exited before the job settled"
    assert ended["seq"] == 6
    assert ended["data"]["reason"] == "interrupted"
    assert reconcile_interrupted(directory) == []


def test_transcript_format_and_truncation() -> None:
    events = [
        SessionEvent(1, "t", "turn", {"role": "user", "text": "hello"}),
        SessionEvent(2, "t", "turn", {"role": "assistant", "text": "hi there"}),
        SessionEvent(3, "t", "turn", {"role": "system", "text": "note"}),
        SessionEvent(
            4,
            "t",
            "model_switch",
            {"old": "fx1", "new": "fx1-lite", "endpoint_host": "api.example"},
        ),
        SessionEvent(5, "t", "job_started", {"job_id": "research-01", "goal": "estimate pinball"}),
        SessionEvent(
            6,
            "t",
            "job_step",
            {"job_id": "research-01", "step": "fit", "status": "ok", "detail": ""},
        ),
        SessionEvent(
            7,
            "t",
            "job_finished",
            {"job_id": "research-01", "status": "done", "detail": "pinball 0.12"},
        ),
        SessionEvent(
            8,
            "t",
            "approval",
            {"approval_kind": "consequential", "question": "run the family?", "approved": True},
        ),
        SessionEvent(
            9,
            "t",
            "approval",
            {"approval_kind": "private_data", "question": "send the note?", "approved": False},
        ),
        SessionEvent(10, "t", "session_end", {"reason": "exit"}),
        SessionEvent(11, "t", "mystery", {"x": "y"}),
    ]
    expected = "\n".join(
        [
            "you › hello",
            "dip › hi there",
            "sys › note",
            "[model] fx1 → fx1-lite @ api.example",
            "[job research-01] started: estimate pinball",
            "[job research-01] fit: ok",
            "[job research-01] done: pinball 0.12",
            "[approval] consequential: run the family? (approved)",
            "[approval] private_data: send the note? (denied)",
            "[session] exit",
        ]
    )
    assert transcript(events) == expected
    assert transcript([]) == ""
    folded = transcript([SessionEvent(1, "t", "turn", {"role": "user", "text": "hello\nworld"})])
    assert folded == "you › hello world"

    many = [
        SessionEvent(index, "t", "turn", {"role": "user", "text": f"line-{index:02d}"})
        for index in range(1, 31)
    ]
    full = transcript(many)
    assert "line-01" in full
    assert "line-30" in full
    assert transcript(many, max_chars=len(full)) == full
    short = transcript(many, max_chars=80)
    assert short.startswith("… (earlier lines omitted)\n")
    assert len(short) <= 80
    assert full.endswith(short.split("\n", 1)[1])
    assert "line-30" in short
    assert "line-01" not in short
    clipped = transcript(many, max_chars=len(full) - 1)
    assert clipped.startswith("… (earlier lines omitted)\n")
    assert len(clipped) <= len(full) - 1
    assert transcript(many, max_chars=10) == "… (earlier lines omitted)\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_journal_file_and_directory_modes(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    previous = os.umask(0)
    try:
        journal = _journal(directory, "private")
        journal.turn("user", "hi")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal.path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_journal_repairs_permissive_modes(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)
    path = directory / "loose.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o666)
    journal = _journal(directory, "loose")
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal.path.stat().st_mode) == 0o600
    assert journal.path.read_text(encoding="utf-8").splitlines()[0]


def test_concurrent_records_are_complete_lines(tmp_path: Path) -> None:
    journal = _journal(tmp_path, "threads")
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        for _ in range(50):
            journal.record("ping", ok=True)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert journal.write_errors == []
    lines = [line for line in journal.path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 201
    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["schema_version"] == 1
    assert "seq" not in payloads[0]
    seqs = [payload["seq"] for payload in payloads[1:]]
    assert seqs == list(range(1, 201))
    assert journal.path.read_text(encoding="utf-8").endswith("\n")


def test_record_swallows_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = _journal(tmp_path, "disk")
    real_open = Path.open
    raised = {"done": False}

    def flaky(self: Path, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        if not isinstance(mode, str):
            mode = "r"
        if self == journal.path and "a" in mode and not raised["done"]:
            raised["done"] = True
            raise OSError("disk full")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky)
    event = journal.record("note", token="supersecret")
    assert event.seq == 1
    assert event.kind == "note"
    assert event.data == {"token": _mask("cret")}
    assert journal.write_errors
    assert "disk full" in journal.write_errors[0]
    assert "supersecret" not in journal.path.read_text(encoding="utf-8")

    follow = journal.record("note", text="again")
    assert follow.seq == 2
    assert len(journal.write_errors) == 1
    stored = journal.events()
    assert [item.seq for item in stored] == [2]
    assert stored[0].data == {"text": "again"}


def test_code_revision_injected_and_git_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injected = _journal(tmp_path, "injected", code_revision="deadbee")
    assert _header(injected)["code_revision"] == "deadbee"

    seen: dict[str, Any] = {}

    def succeed(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="cafebabe\n", stderr="")

    monkeypatch.setattr("fx1.interactive.session.subprocess.run", succeed)
    resolved = SessionJournal(session_id="git-ok", sessions_dir=tmp_path)
    assert _header(resolved)["code_revision"] == "cafebabe"
    assert seen["cmd"] == ["git", "rev-parse", "--short", "HEAD"]
    assert seen["kwargs"]["timeout"] == 2
    assert seen["kwargs"]["cwd"] == Path.cwd()

    def nonzero(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="cafebabe\n", stderr="no")

    monkeypatch.setattr("fx1.interactive.session.subprocess.run", nonzero)
    failed = SessionJournal(session_id="git-bad", sessions_dir=tmp_path)
    assert _header(failed)["code_revision"] == "unknown"

    def blank(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="ab cd\n", stderr="")

    monkeypatch.setattr("fx1.interactive.session.subprocess.run", blank)
    spaced = SessionJournal(session_id="git-space", sessions_dir=tmp_path)
    assert _header(spaced)["code_revision"] == "unknown"


@pytest.mark.parametrize(
    "boom",
    [
        OSError("git not found"),
        subprocess.TimeoutExpired(cmd=["git"], timeout=2),
        RuntimeError("broken"),
    ],
)
def test_code_revision_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boom: BaseException
) -> None:
    def explode(*_args: object, **_kwargs: object) -> object:
        raise boom

    monkeypatch.setattr("fx1.interactive.session.subprocess.run", explode)
    journal = SessionJournal(session_id=f"git-down-{type(boom).__name__}", sessions_dir=tmp_path)
    assert _header(journal)["code_revision"] == "unknown"
