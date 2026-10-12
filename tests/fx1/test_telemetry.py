"""Local JSONL telemetry: shape, redaction, rotation, and failure isolation."""

from __future__ import annotations

import json
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive import telemetry as telemetry_mod
from fx1.interactive.redact import REDACTED, SecretRegistry
from fx1.interactive.runctx import new_id
from fx1.interactive.telemetry import (
    MAX_VALUE_CHARS,
    Level,
    Telemetry,
    parse_level,
    sanitize_value,
)

NOW = datetime(2026, 10, 12, 2, 45, tzinfo=UTC)
SID = new_id("s", now=NOW, rand="abc12345")
API_KEY = "sk-" + "AbCdEfGhIjKlMnOpQrSt"
SENTINEL = "zzPromptSentinelzz"
CONTENT_FIELDS = (
    "content",
    "messages",
    "prompt",
    "answer",
    "text",
    "body",
    "arguments",
    "stdout",
    "stderr",
)


def _open(tmp_path: Path, **overrides: Any) -> Telemetry:
    options: dict[str, Any] = {
        "session_id": SID,
        "directory": tmp_path / "logs",
        "clock": lambda: NOW,
    }
    options.update(overrides)
    return Telemetry(**options)


def _blob(tel: Telemetry) -> str:
    return json.dumps(tel.tail(1000))


def test_event_shape_order_and_timestamp(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    assert tel.log_files() == []
    tel.emit("model.response", {"tokens": 12})
    tel.emit("model.response", {"tokens": 13})
    text = tel.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[0]["ts"] == "2026-10-12T02:45:00.000000Z"
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[0]["level"] == "info"
    assert rows[0]["event"] == "model.response"
    assert rows[0]["session_id"] == SID
    assert rows[0]["tokens"] == 12
    assert list(rows[0])[:4] == ["ts", "seq", "level", "event"]
    assert tel.log_files() == [tel.path]
    assert stat.S_IMODE(tel.path.stat().st_mode) == 0o600


def test_init_does_not_create_directory(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    Telemetry(session_id=SID, directory=logs, clock=lambda: NOW)
    assert not logs.exists()


def test_redacts_api_keys_nested_dicts_and_urls(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    url = f"https://alicezz:{SENTINEL}@files.example/v1?token={API_KEY}"
    tel.emit(
        "model.response",
        {
            "api_key": API_KEY,
            "note": f"Authorization: Bearer {API_KEY} {url}",
            "meta": {"api_key": API_KEY, "url": url},
            "prompt": f"hello {SENTINEL}",
        },
    )
    blob = _blob(tel)
    assert API_KEY not in blob
    assert SENTINEL not in blob
    assert "alicezz" not in blob
    assert "files.example" in blob
    assert REDACTED in blob
    assert "[omitted:" in blob


def test_registry_scrubs_exact_secret(tmp_path: Path) -> None:
    registry = SecretRegistry()
    registry.register("plain-hunter-secret")
    tel = _open(tmp_path, registry=registry)
    tel.emit("app.note", {"note": "see plain-hunter-secret please"})
    blob = _blob(tel)
    assert "plain-hunter-secret" not in blob
    assert REDACTED in blob


@pytest.mark.parametrize("field", CONTENT_FIELDS)
def test_content_fields_record_length_only(field: str, tmp_path: Path) -> None:
    payload = f"{field}-{SENTINEL}"
    tel = _open(tmp_path)
    tel.emit("tool.result", {field: payload, "ok": True})
    row = tel.tail(1)[0]
    blob = json.dumps(row)
    assert SENTINEL not in blob
    assert payload not in blob
    assert row[field] == f"[omitted: {len(payload)} chars]"
    assert row["ok"] is True


def test_allow_list_keeps_named_field_and_still_guards_nested(tmp_path: Path) -> None:
    tel = _open(tmp_path, allow_content_fields={"messages", "prompt"})
    child = tel.bind(model="fx1")
    child.emit(
        "model.request",
        {
            "prompt": "hello",
            "answer": "xyz",
            "messages": [{"role": "user", "content": SENTINEL}],
            "wrapper": {"stdout": "hidden-stdout"},
        },
    )
    row = child.tail(1)[0]
    assert row["prompt"] == "hello"
    assert row["answer"] == "[omitted: 3 chars]"
    assert row["messages"][0]["role"] == "user"
    assert str(row["messages"][0]["content"]).startswith("[omitted:")
    assert row["wrapper"]["stdout"] == "[omitted: 13 chars]"
    assert row["model"] == "fx1"
    blob = json.dumps(row)
    assert SENTINEL not in blob
    assert "hidden-stdout" not in blob


def test_long_values_are_truncated_and_idempotent(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    long = "a" * 800
    tel.emit("app.note", {"detail": long})
    val = tel.tail(1)[0]["detail"]
    assert isinstance(val, str)
    assert len(val) == MAX_VALUE_CHARS
    assert "truncated 800 chars" in val
    assert long not in _blob(tel)
    assert sanitize_value(tel.tail(1)[0])["detail"] == val
    once = sanitize_value({"prompt": "hello", "note": "b" * 700, "api_key": API_KEY})
    assert sanitize_value(once) == once
    assert API_KEY not in json.dumps(once)


def test_nonserializable_repr_is_redacted(tmp_path: Path) -> None:
    class Holder:
        def __repr__(self) -> str:
            return f"Holder({API_KEY})"

    tel = _open(tmp_path)
    tel.emit("app.note", {"obj": Holder()})
    blob = _blob(tel)
    assert API_KEY not in blob
    assert "Holder" in blob
    assert len(tel.tail(1)[0]["obj"]) <= MAX_VALUE_CHARS


def test_bytes_never_logged(tmp_path: Path) -> None:
    secret = b"sk-" + b"AbCdEfGhIjKlMnOp"
    tel = _open(tmp_path)
    tel.emit("app.blob", {"blob": secret, "stdout": secret})
    blob = _blob(tel)
    assert b"AbCdEfGhIjKlMnOp".decode() not in blob
    row = tel.tail(1)[0]
    assert str(row["blob"]).startswith("[bytes:")
    assert str(row["stdout"]).startswith("[omitted:")


def test_non_finite_numbers_become_null(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    tel.emit("app.tick", {"x": float("nan"), "y": float("inf"), "z": 1.5, "ok": False, "n": 0})
    row = tel.tail(1)[0]
    assert row["x"] is None
    assert row["y"] is None
    assert row["z"] == 1.5
    assert row["ok"] is False
    assert row["n"] == 0


def test_depth_and_cycles_do_not_leak() -> None:
    outer: dict[str, Any] = {"prompt": API_KEY + SENTINEL}
    for _ in range(20):
        outer = {"wrap": outer}
    assert API_KEY not in json.dumps(sanitize_value(outer))
    assert SENTINEL not in json.dumps(sanitize_value(outer))
    cycle: dict[str, Any] = {"prompt": SENTINEL}
    cycle["self"] = cycle
    safe = sanitize_value(cycle)
    assert SENTINEL not in json.dumps(safe)
    assert str(safe["prompt"]).startswith("[omitted:")


@pytest.mark.parametrize("name", ["", "single", "Bad.Name", "app.tick ", "../etc/passwd"])
def test_invalid_event_names_keep_length_only(name: str, tmp_path: Path) -> None:
    tel = _open(tmp_path)
    tel.emit(name, {"prompt": SENTINEL, "note": SENTINEL})
    row = tel.tail(1)[0]
    assert row["event"] == "telemetry.bad_event"
    assert row["name_len"] == len(name)
    blob = json.dumps(row)
    assert SENTINEL not in blob
    if name not in {"", "single"}:
        assert name not in blob


def test_valid_dotted_names_pass(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    tel.emit("a.b", {"n": 1})
    tel.emit("research.family.start", {"n": 2})
    assert [row["event"] for row in tel.tail()] == ["a.b", "research.family.start"]


def test_level_filtering_and_debug(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    tel = _open(tmp_path, level="error")
    tel.emit("app.trace", {}, level=Level.DEBUG)
    tel.emit("app.tick", {})
    assert not logs.exists()
    tel.emit("app.boom", {"status": "bad"}, level=Level.ERROR)
    rows = tel.tail()
    assert [row["event"] for row in rows] == ["app.boom"]
    assert rows[0]["status"] == "bad"
    debug = _open(tmp_path, directory=tmp_path / "debug", level=Level.DEBUG)
    debug.emit("app.trace", {"n": 1}, level=Level.DEBUG)
    debug.emit("app.tick", {"n": 2})
    assert [row["level"] for row in debug.tail()] == ["debug", "info"]


def test_level_off_creates_nothing(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    tel = Telemetry(session_id=SID, directory=logs, level="off", clock=lambda: NOW)
    tel.emit("app.tick", {"prompt": SENTINEL}, level=Level.ERROR)
    tel.emit("app.trace", {"prompt": SENTINEL}, level=Level.DEBUG)
    assert tel.level is Level.OFF
    assert not logs.exists()
    assert tel.tail() == []
    assert tel.dropped == 0
    assert tel.log_files() == []


def test_level_from_env_and_invalid_strings(tmp_path: Path) -> None:
    debug = Telemetry(
        session_id=SID,
        directory=tmp_path / "a",
        environ={"DIPCATCHER_LOG": "DeBuG"},
    )
    assert debug.level is Level.DEBUG
    fallback = Telemetry(
        session_id=SID,
        directory=tmp_path / "b",
        environ={"DIPCATCHER_LOG": "verbose"},
    )
    assert fallback.level is Level.INFO
    forced = Telemetry(
        session_id=SID,
        directory=tmp_path / "c",
        level=Level.ERROR,
        environ={"DIPCATCHER_LOG": "debug"},
    )
    assert forced.level is Level.ERROR
    assert parse_level(" OFF ") is Level.OFF
    with pytest.raises(ValueError, match="unknown telemetry level"):
        Telemetry(session_id=SID, directory=tmp_path / "d", level="verbose")


def test_rotation_keeps_max_files(tmp_path: Path) -> None:
    tel = _open(tmp_path, max_bytes=1, max_files=3)
    for i in range(1, 7):
        tel.emit("app.tick", {"i": i})
    files = tel.log_files()
    assert len(files) == 3
    assert not tel.path.with_name(tel.path.name + ".3").exists()
    rows = tel.tail(10)
    assert [row["seq"] for row in rows] == [4, 5, 6]
    assert [row["i"] for row in rows] == [4, 5, 6]
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_max_files_one_retains_only_active(tmp_path: Path) -> None:
    tel = _open(tmp_path, max_bytes=1, max_files=1)
    tel.emit("app.tick", {"i": 1})
    tel.emit("app.tick", {"i": 2})
    tel.emit("app.tick", {"i": 3})
    assert tel.log_files() == [tel.path]
    rows = tel.tail(10)
    assert len(rows) == 1
    assert rows[0]["i"] == 3


def test_bind_inherits_context_and_restores_valid_ids(tmp_path: Path) -> None:
    tel = _open(tmp_path, context={"family": "volatility"})
    run = new_id("r", now=NOW, rand="aabbccdd")
    job = new_id("j", now=NOW, rand="00112233")
    child = tel.bind(run_id=run, model="fx1-lite", family="tail")
    grand = child.bind(job_id=job)
    grand.emit("step.done", {"step": f"{run}.000"})
    row = tel.tail(1)[0]
    assert row["session_id"] == SID
    assert row["run_id"] == run
    assert row["job_id"] == job
    assert row["model"] == "fx1-lite"
    assert row["family"] == "tail"
    assert row["step"] == f"{run}.000"
    assert grand.path == tel.path
    grand.emit("step.done", {"run_id": "nope", "job_id": "../x", "session_id": "../y"})
    bad = tel.tail(10)[-1]
    assert bad["session_id"] == SID
    assert bad["run_id"] == run
    assert bad["job_id"] == job
    assert bad["id_rejected"] == ["session_id", "run_id", "job_id"]
    blob = json.dumps(bad)
    assert "nope" not in blob
    assert "../x" not in blob
    assert "../y" not in blob


def test_bad_bound_session_keeps_root_id(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    child = tel.bind(session_id="../evil")
    child.emit("app.tick", {})
    row = tel.tail(1)[0]
    assert row["session_id"] == SID
    assert "session_id" in row["id_rejected"]
    assert "../evil" not in json.dumps(row)
    assert tel.path.name == f"{SID}.jsonl"


def test_bind_rejects_non_strings(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    with pytest.raises(TypeError, match="strings"):
        tel.bind(model=1)  # type: ignore[arg-type]


def test_thread_safety_strictly_increasing_seq(tmp_path: Path) -> None:
    tel = _open(tmp_path, max_bytes=10**9)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            for n in range(30):
                tel.emit("worker.event", {"i": index, "n": n})
        except BaseException as exc:  # noqa: BLE001 - the test asserts emit does not raise
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    rows = tel.tail(1000)
    assert [row["seq"] for row in rows] == list(range(1, 8 * 30 + 1))
    assert {row["event"] for row in rows} == {"worker.event"}


def test_rotation_under_threads(tmp_path: Path) -> None:
    tel = _open(tmp_path, max_bytes=1, max_files=3)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for n in range(8):
                tel.emit("worker.event", {"i": index, "n": n})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == []
    assert tel.dropped == 0
    assert len(tel.log_files()) <= 3
    rows = tel.tail(10)
    assert [row["seq"] for row in rows] == [30, 31, 32]
    for path in tel.log_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed["seq"], int)


def test_failing_sink_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    os.chmod(logs, 0o555)
    tel = Telemetry(session_id=SID, directory=logs, clock=lambda: NOW)
    tel.emit("app.tick", {"prompt": SENTINEL, "api_key": API_KEY})
    if tel.dropped == 0:

        def boom(_path: Path, _line: bytes) -> None:
            raise OSError(28, "no space")

        monkeypatch.setattr(telemetry_mod, "_append_line", boom)
        tel.emit("app.tick", {"prompt": SENTINEL})
    assert tel.dropped >= 1
    err = tel.sink_error
    assert err is not None
    assert err["event"] == "telemetry.sink_error"
    assert err["level"] == "error"
    blob = json.dumps(err)
    assert SENTINEL not in blob
    assert API_KEY not in blob
    reason = err["reason"]
    tel.emit("app.tick", {"prompt": SENTINEL})
    assert tel.dropped >= 2
    assert tel.sink_error is not None
    assert tel.sink_error["reason"] == reason
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
    if tel.path.exists() and not tel.path.is_symlink():
        text = tel.path.read_text(encoding="utf-8")
        assert SENTINEL not in text
        assert API_KEY not in text
    assert all(row["event"] != "telemetry.sink_error" for row in tel.tail())


def test_failed_write_does_not_consume_seq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tel = _open(tmp_path)
    real = telemetry_mod._append_line
    calls = {"n": 0}

    def flaky(path: Path, line: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "nospace")
        real(path, line)

    monkeypatch.setattr(telemetry_mod, "_append_line", flaky)
    tel.emit("app.tick", {"i": 1})
    tel.emit("app.tick", {"i": 2})
    rows = tel.tail()
    assert [row["seq"] for row in rows] == [1]
    assert rows[0]["i"] == 2
    assert tel.dropped == 1
    assert tel.sink_error is not None


def test_symlink_log_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text("SECRET-OUTSIDE\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"{SID}.jsonl").symlink_to(outside)
    tel = Telemetry(session_id=SID, directory=logs, clock=lambda: NOW)
    tel.emit("app.tick", {"prompt": SENTINEL})
    assert tel.dropped == 1
    assert outside.read_text(encoding="utf-8") == "SECRET-OUTSIDE\n"
    assert tel.sink_error is not None
    assert SENTINEL not in json.dumps(tel.sink_error)


def test_symlink_directory_is_not_followed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "logs"
    link.symlink_to(real, target_is_directory=True)
    tel = Telemetry(session_id=SID, directory=link, clock=lambda: NOW)
    tel.emit("app.tick", {})
    assert tel.dropped == 1
    assert list(real.iterdir()) == []


def test_file_modes_and_preexisting_dir(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.chmod(0o755)
    tel = _open(tmp_path)
    tel.emit("app.ready", {})
    assert stat.S_IMODE(tel.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(logs.stat().st_mode) == 0o700


def test_windows_branch_does_not_chmod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = os.chmod

    def spy(path: Path | str, mode: int) -> None:
        calls.append(mode)
        real(path, mode)

    monkeypatch.setattr(os, "chmod", spy)
    monkeypatch.setattr(os, "name", "nt")
    tel = _open(tmp_path)
    tel.emit("app.ready", {})
    assert tel.path.is_file()
    assert calls == []


def test_tail_returns_most_recent_in_order(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    assert tel.tail() == []
    for i in range(10):
        tel.emit("app.tick", {"i": i})
    assert [row["i"] for row in tel.tail(3)] == [7, 8, 9]
    assert tel.tail(0) == []
    assert tel.tail(-1) == []
    assert len(tel.tail(100)) == 10


def test_default_directory_follows_config_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    tel = Telemetry(session_id=SID, clock=lambda: NOW)
    tel.emit("app.ready", {})
    assert tel.path == tmp_path / "cfg" / "logs" / f"{SID}.jsonl"
    assert tel.path.is_file()


def test_malformed_session_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="malformed"):
        Telemetry(session_id="../etc/passwd", directory=tmp_path)
    with pytest.raises(ValueError, match="malformed"):
        Telemetry(session_id="r-20261012T024500Z-abc12345", directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_bad_limits_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        Telemetry(session_id=SID, directory=tmp_path, max_bytes=0)
    with pytest.raises(ValueError, match="max_files"):
        Telemetry(session_id=SID, directory=tmp_path, max_files=0)
    with pytest.raises(ValueError, match="max_bytes"):
        Telemetry(session_id=SID, directory=tmp_path, max_bytes=True)  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_as_emit_respects_level(tmp_path: Path) -> None:
    tel = _open(tmp_path, level="info")
    tel.as_emit()("app.ready", {"ok": True})
    tel.as_emit(Level.DEBUG)("app.trace", {"ok": True})
    rows = tel.tail()
    assert [row["event"] for row in rows] == ["app.ready"]
    assert rows[0]["ok"] is True


def test_close_is_idempotent(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    tel.emit("app.tick", {"i": 1})
    tel.close()
    tel.close()
    tel.emit("app.tick", {"i": 2})
    assert [row["i"] for row in tel.tail()] == [1]
    assert tel.dropped == 0


def test_non_mapping_fields_still_record_event(tmp_path: Path) -> None:
    tel = _open(tmp_path)
    tel.emit("app.tick", ["not", "a", "mapping"])  # type: ignore[arg-type]
    row = tel.tail(1)[0]
    assert row["event"] == "app.tick"
    assert row["session_id"] == SID
    assert "not" not in json.dumps(row)


def test_clock_failure_still_writes(tmp_path: Path) -> None:
    def boom() -> datetime:
        raise RuntimeError("clock")

    tel = _open(tmp_path, clock=boom)
    tel.emit("app.tick", {})
    assert tel.dropped == 0
    assert "T" in tel.tail(1)[0]["ts"]


def test_repr_has_no_secrets(tmp_path: Path) -> None:
    tel = _open(tmp_path, registry=SecretRegistry())
    text = repr(tel)
    assert SID in text
    assert API_KEY not in text
    assert SENTINEL not in text
    assert "INFO" in text
