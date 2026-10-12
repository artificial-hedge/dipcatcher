"""Non-interactive ``dipcatcher sessions`` and ``dipcatcher resume``."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SECRET = "sk-plantedkeyvalue123456"
ENV_SECRET = "env-planted-secret-value-999"
PHRASE = "ordinary-body-phrase"
SCHEMA = "dipcatcher.sessions.v1"
OLDER = "a" * 32
NEWER = "b" * 32
MID = "c" * 32
FROZEN = datetime(2026, 1, 10, tzinfo=UTC)
_HOSTILE = ("../x", "..", "a/b", "s-../../../etc/passwd", "id\x00bad")


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "cfg"
    directory = tmp_path / "sessions"
    config.mkdir()
    directory.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(config))
    monkeypatch.setenv("FX1_SESSION_DIR", str(directory))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FX1_PLANTED_API_KEY", raising=False)
    return directory


def _code(exc: SystemExit) -> int:
    return exc.code if isinstance(exc.code, int) else 0


def _captured(capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    captured = capsys.readouterr()
    return captured.out, captured.err


def _app(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    from dipcatcher_cli.app import main

    try:
        main(args)
    except SystemExit as exc:
        code = _code(exc)
    else:
        code = 0
    out, err = _captured(capsys)
    return code, out, err


def _cli(
    args: list[str],
    capsys: pytest.CaptureFixture[str],
    **kwargs: Any,
) -> tuple[int, str, str]:
    from dipcatcher_cli.sessions_cmd import sessions_main

    try:
        sessions_main(args, **kwargs)
    except SystemExit as exc:
        code = _code(exc)
    else:
        code = 0
    out, err = _captured(capsys)
    return code, out, err


def _resume(
    args: list[str],
    capsys: pytest.CaptureFixture[str],
    **kwargs: Any,
) -> tuple[int, str, str]:
    from dipcatcher_cli.sessions_cmd import resume_main

    try:
        resume_main(args, **kwargs)
    except SystemExit as exc:
        code = _code(exc)
    else:
        code = 0
    out, err = _captured(capsys)
    return code, out, err


def _write(directory: Path, session_id: str, updated_at: str, **extra: Any) -> None:
    record: dict[str, Any] = {
        "schema": 1,
        "id": session_id,
        "model": "fx1",
        "status": "open",
        "messages": [],
        "job": {"state": "idle", "id": "", "steps_completed": [], "resume": "none"},
        "created_at": updated_at,
        "updated_at": updated_at,
        "revision": 0,
        "turns": [],
        "truncated": False,
    }
    record.update(extra)
    (directory / f"{session_id}.json").write_text(json.dumps(record), encoding="utf-8")


def _turn(**extra: Any) -> dict[str, Any]:
    turn = {
        "index": 0,
        "model": "fx1",
        "served_model": "fx1-2026",
        "usage": {"max_tokens": 32, "total_tokens": 11},
        "outcome": "ok",
        "started_at": "2020-01-01T00:00:01+00:00",
        "detail": "hidden-detail-text",
    }
    turn.update(extra)
    return turn


def _secret_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"{PHRASE} {SECRET}", "api_key": ENV_SECRET},
        {"role": "user", "content": ENV_SECRET},
    ]


def _plant(monkeypatch: pytest.MonkeyPatch, directory: Path, session_id: str = OLDER) -> None:
    monkeypatch.setenv("FX1_PLANTED_API_KEY", ENV_SECRET)
    _write(
        directory,
        session_id,
        "2024-06-01T00:00:00+00:00",
        messages=_secret_messages(),
        turns=[_turn()],
        model="fx1-lite",
    )


def _idle_job() -> dict[str, Any]:
    return {"state": "idle", "id": "", "steps_completed": [], "resume": "none"}


def _running_job() -> dict[str, Any]:
    return {"state": "running", "id": "job-1", "steps_completed": ["score"], "resume": "none"}


def test_list_is_newest_first(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        sessions, OLDER, "2020-01-01T00:00:00+00:00", messages=[{"role": "user", "content": PHRASE}]
    )
    _write(
        sessions,
        NEWER,
        "2024-06-01T00:00:00+00:00",
        model="fx1-lite",
        status="open",
        turns=[_turn(), _turn(index=1)],
    )
    code, out, err = _app(["sessions"], capsys)
    assert code == 0
    assert err == ""
    lines = out.splitlines()
    assert lines[0] == "id\tmodel\tstatus\tturns\tupdated"
    assert lines[1] == f"{NEWER}\tfx1-lite\topen\t2\t2024-06-01T00:00:00+00:00"
    assert lines[2] == f"{OLDER}\tfx1\topen\t0\t2020-01-01T00:00:00+00:00"
    assert PHRASE not in out


def test_list_limit_and_zero(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00")
    _write(sessions, NEWER, "2024-06-01T00:00:00+00:00")
    code, out, _err = _app(["sessions", "--limit", "1"], capsys)
    assert code == 0
    assert NEWER in out
    assert OLDER not in out
    code, out, err = _app(["sessions", "list", "--limit", "0"], capsys)
    assert code == 0
    assert err == ""
    assert out == "id\tmodel\tstatus\tturns\tupdated\n"


def test_list_json_is_one_document(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        sessions, OLDER, "2020-01-01T00:00:00+00:00", messages=[{"role": "user", "content": PHRASE}]
    )
    _write(sessions, NEWER, "2024-06-01T00:00:00+00:00", model="fx1-lite", turns=[_turn()])
    code, out, err = _app(["sessions", "list", "--json"], capsys)
    assert code == 0
    assert err == ""
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert payload["schema"] == SCHEMA
    assert payload["command"] == "list"
    assert payload["warnings"] == []
    assert [row["id"] for row in payload["sessions"]] == [NEWER, OLDER]
    assert payload["sessions"][0]["model"] == "fx1-lite"
    assert payload["sessions"][0]["turns"] == 1
    assert payload["sessions"][0]["status"] == "open"
    assert payload["sessions"][0]["updated"] == "2024-06-01T00:00:00+00:00"
    assert set(payload["sessions"][0]) == {"id", "model", "status", "turns", "updated"}
    assert PHRASE not in out


def test_list_empty_text_and_json(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _app(["sessions", "list"], capsys)
    assert code == 0
    assert err == ""
    assert out == "id\tmodel\tstatus\tturns\tupdated\n"
    code, out, err = _app(["sessions", "--json"], capsys)
    assert code == 0
    assert err == ""
    assert json.loads(out) == {
        "command": "list",
        "schema": SCHEMA,
        "sessions": [],
        "warnings": [],
    }


def test_corrupt_files_warn_on_stderr_and_do_not_crash(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, NEWER, "2024-06-01T00:00:00+00:00")
    (sessions / f"{'d' * 32}.json").write_text("{", encoding="utf-8")
    (sessions / "notes.json").write_text("{}", encoding="utf-8")
    code, out, err = _app(["sessions", "list"], capsys)
    assert code == 0
    assert NEWER in out
    assert "invalid session JSON" in err
    assert "malformed" in err
    assert "invalid" not in out
    assert "malformed" not in out
    assert "skipped" not in out
    code, out, err = _app(["sessions", "list", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert [row["id"] for row in payload["sessions"]] == [NEWER]
    assert payload["warnings"]
    assert "invalid session JSON" in err
    assert any("invalid session JSON" in warning for warning in payload["warnings"])


def test_negative_limit_is_usage_error(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _app(["sessions", "--limit", "-1"], capsys)
    assert code == 2
    assert out == ""
    assert "non-negative" in err
    code, _out, err = _app(["sessions", "--limit", "nope"], capsys)
    assert code == 2
    assert "invalid" in err.lower()


def test_show_turn_table_omits_message_bodies(
    sessions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(monkeypatch, sessions)
    code, out, err = _app(["sessions", "show", "last"], capsys)
    assert code == 0
    assert err == ""
    assert f"id: {OLDER}" in out
    assert "model: fx1-lite" in out
    assert "fx1-2026" in out
    assert "max_tokens=32,total_tokens=11" in out
    assert "outcome" in out
    assert PHRASE not in out
    assert SECRET not in out
    assert ENV_SECRET not in out
    assert "hidden-detail-text" not in out
    assert "messages:" not in out


def test_show_messages_keeps_phrase_and_drops_secrets(
    sessions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(monkeypatch, sessions)
    code, out, err = _app(["sessions", "show", OLDER, "--messages"], capsys)
    assert code == 0
    assert err == ""
    assert PHRASE in out
    assert SECRET not in out
    assert ENV_SECRET not in out
    assert "hidden-detail-text" in out
    assert SECRET in (sessions / f"{OLDER}.json").read_text(encoding="utf-8")


def test_show_json_omits_bodies_unless_requested(
    sessions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(monkeypatch, sessions)
    code, out, err = _app(["sessions", "show", OLDER[:8], "--json"], capsys)
    assert code == 0
    assert err == ""
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert payload["schema"] == SCHEMA
    assert payload["command"] == "show"
    session = payload["session"]
    assert session["id"] == OLDER
    assert "messages" not in session
    turn = session["turns"][0]
    assert turn["served_model"] == "fx1-2026"
    assert turn["model"] == "fx1"
    assert turn["outcome"] == "ok"
    assert turn["usage"]["total_tokens"] == 11
    assert turn["usage"]["max_tokens"] == 32
    assert "detail" not in turn
    assert PHRASE not in out
    assert SECRET not in out
    code, out, _err = _app(["sessions", "show", "last", "--json", "--messages"], capsys)
    assert code == 0
    body = json.loads(out)
    assert PHRASE in out
    assert SECRET not in out
    assert ENV_SECRET not in out
    assert body["session"]["messages"]
    assert "detail" in body["session"]["turns"][0]


def test_show_interrupted_note_and_running_is_read_only(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        sessions,
        OLDER,
        "2020-01-01T00:00:00+00:00",
        job={"state": "interrupted", "id": "j1", "steps_completed": ["score"], "resume": "manual"},
    )
    running = sessions / f"{NEWER}.json"
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00", job=_running_job())
    before = running.read_text(encoding="utf-8")
    code, out, err = _app(["sessions", "show", OLDER], capsys)
    assert code == 0
    assert err == ""
    assert "note: an interrupted job is not restarted" in out
    assert "job_steps: score" in out
    code, out, err = _app(["sessions", "show", NEWER], capsys)
    assert code == 0
    assert "note: a running job is not restarted" in out
    assert running.read_text(encoding="utf-8") == before


def test_show_ambiguous_unknown_and_missing(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = "ab" + "0" * 30
    right = "ac" + "0" * 30
    _write(sessions, left, "2020-01-01T00:00:00+00:00")
    _write(sessions, right, "2021-01-01T00:00:00+00:00")
    code, out, err = _app(["sessions", "show", "a"], capsys)
    assert code == 2
    assert out == ""
    assert "ambiguous" in err
    code, out, err = _app(["sessions", "show", "ab"], capsys)
    assert code == 0
    assert left in out
    code, out, err = _app(["sessions", "show", "zzz"], capsys)
    assert code == 2
    assert out == ""
    assert "no session" in err
    code, out, err = _app(["sessions", "show"], capsys)
    assert code == 2
    assert out == ""


def test_show_last_empty_store(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _app(["sessions", "show", "last"], capsys)
    assert code == 2
    assert out == ""
    assert "no sessions" in err


def test_show_last_is_newest(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00")
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00", model="fx1-lite")
    code, out, _err = _app(["sessions", "show", "last"], capsys)
    assert code == 0
    assert f"id: {NEWER}" in out
    assert "model: fx1-lite" in out


@pytest.mark.parametrize("bad", _HOSTILE)
def test_hostile_ids_do_not_escape(
    sessions: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path, bad: str
) -> None:
    canary = tmp_path / "x"
    canary.write_text("canary", encoding="utf-8")
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")
    for args in (
        ["sessions", "show", bad],
        ["sessions", "export", bad],
        ["sessions", "rm", bad, "--yes"],
    ):
        code, out, err = _app(args, capsys)
        assert code == 2
        assert out == ""
        assert err
        assert canary.read_text(encoding="utf-8") == "canary"
        assert (sessions / f"{NEWER}.json").is_file()


def test_kernel_style_prefix_resolves(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    session_id = "s-20261012T024500Z-abcdef12"
    _write(sessions, session_id, "2024-01-01T00:00:00+00:00")
    code, out, err = _app(["sessions", "show", "s-20261012T024500Z-abcdef"], capsys)
    assert code == 0
    assert err == ""
    assert session_id in out


def test_export_markdown_matches_store_and_drops_secrets(
    sessions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(monkeypatch, sessions)
    from fx1.interactive.sessions import SessionStore

    code, out, err = _app(["sessions", "export", "last"], capsys)
    assert code == 0
    assert err == ""
    expected = SessionStore().export_markdown(SessionStore().load(OLDER))
    assert out == expected
    assert PHRASE in out
    assert SECRET not in out
    assert ENV_SECRET not in out
    assert "# dipcatcher session" in out
    assert SECRET in (sessions / f"{OLDER}.json").read_text(encoding="utf-8")


def test_export_json_is_one_schema_document(
    sessions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(monkeypatch, sessions)
    from fx1.interactive.sessions import SessionStore

    code, out, err = _app(["sessions", "export", OLDER, "--format", "json"], capsys)
    assert code == 0
    assert err == ""
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert payload["schema"] == SCHEMA
    assert payload["command"] == "export"
    assert payload["session"] == json.loads(SessionStore().export_json(SessionStore().load(OLDER)))
    assert payload["session"]["schema"] == 1
    assert PHRASE in out
    assert SECRET not in out
    assert ENV_SECRET not in out
    assert "api_key" not in out


def test_export_out_is_private_and_refuses_overwrite(
    sessions: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _plant(monkeypatch, sessions)
    dest = tmp_path / "fresh" / "out.md"
    code, out, err = _app(["sessions", "export", "last", "--out", str(dest)], capsys)
    assert code == 0
    assert out == ""
    assert "wrote" in err
    assert SECRET not in err
    assert PHRASE in dest.read_text(encoding="utf-8")
    assert SECRET not in dest.read_text(encoding="utf-8")
    assert ENV_SECRET not in dest.read_text(encoding="utf-8")
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    assert stat.S_IMODE(dest.parent.stat().st_mode) == 0o700
    before = dest.read_text(encoding="utf-8")
    code, out, err = _app(
        ["sessions", "export", "last", "--format", "json", "--out", str(dest)], capsys
    )
    assert code == 2
    assert out == ""
    assert "refusing to overwrite" in err
    assert dest.read_text(encoding="utf-8") == before
    kept = tmp_path / "keep.md"
    kept.write_text("keep-me", encoding="utf-8")
    code, out, err = _app(["sessions", "export", OLDER, "--out", str(kept)], capsys)
    assert code == 2
    assert kept.read_text(encoding="utf-8") == "keep-me"


def test_export_json_out_and_runtime_failure(
    sessions: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _plant(monkeypatch, sessions)
    dest = tmp_path / "export.json"
    code, out, err = _app(
        ["sessions", "export", OLDER, "--format", "json", "--out", str(dest)], capsys
    )
    assert code == 0
    assert out == ""
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert SECRET not in dest.read_text(encoding="utf-8")
    blocker = tmp_path / "not-dir"
    blocker.write_text("x", encoding="utf-8")
    code, out, err = _app(["sessions", "export", OLDER, "--out", str(blocker / "out.md")], capsys)
    assert code == 1
    assert out == ""
    assert SECRET not in err
    assert ENV_SECRET not in err
    assert blocker.read_text(encoding="utf-8") == "x"
    code, out, err = _app(["sessions", "export", OLDER, "--out", " "], capsys)
    assert code == 2
    assert "empty" in err
    code, _out, _err = _app(["sessions", "export", OLDER, "--format", "xml"], capsys)
    assert code == 2


def test_rm_without_yes_refuses_on_a_non_tty(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")
    code, out, err = _cli(["rm", NEWER], capsys, isatty=lambda: False)
    assert code == 2
    assert out == ""
    assert "without confirmation" in err
    assert "[y/N]" not in err
    assert (sessions / f"{NEWER}.json").is_file()


def test_rm_yes_deletes_last(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00")
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")
    code, out, err = _app(["sessions", "rm", "last", "--yes"], capsys)
    assert code == 0
    assert err == ""
    assert out == f"deleted {NEWER}\n"
    assert not (sessions / f"{NEWER}.json").exists()
    assert (sessions / f"{OLDER}.json").is_file()


@pytest.mark.parametrize(
    ("answer", "deleted"),
    [
        ("y\n", True),
        ("Y\n", True),
        ("yes\n", True),
        ("n\n", False),
        ("\n", False),
        ("yeah\n", False),
    ],
)
def test_rm_tty_confirmation(
    sessions: Path, capsys: pytest.CaptureFixture[str], answer: str, deleted: bool
) -> None:
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")
    code, out, err = _cli(["rm", "last"], capsys, isatty=lambda: True, read_line=lambda: answer)
    assert "[y/N]" in err
    path = sessions / f"{NEWER}.json"
    if deleted:
        assert code == 0
        assert out == f"deleted {NEWER}\n"
        assert not path.exists()
    else:
        assert code == 2
        assert "without confirmation" in err
        assert path.is_file()


def test_rm_yes_does_not_prompt(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")

    def refuse() -> str:
        raise AssertionError("prompt should not be read")

    code, out, err = _cli(["rm", NEWER, "--yes"], capsys, isatty=lambda: True, read_line=refuse)
    assert code == 0
    assert "[y/N]" not in err
    assert out == f"deleted {NEWER}\n"


def test_rm_ambiguous_deletes_nothing(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    left = "ab" + "0" * 30
    right = "ac" + "0" * 30
    _write(sessions, left, "2020-01-01T00:00:00+00:00")
    _write(sessions, right, "2021-01-01T00:00:00+00:00")
    code, out, err = _app(["sessions", "rm", "a", "--yes"], capsys)
    assert code == 2
    assert out == ""
    assert "ambiguous" in err
    assert (sessions / f"{left}.json").is_file()
    assert (sessions / f"{right}.json").is_file()


def test_prune_dry_run_reports_ids_without_deleting(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00")
    _write(sessions, MID, "2025-12-01T00:00:00+00:00")
    _write(sessions, NEWER, "2026-01-09T00:00:00+00:00")
    (sessions / f"{'d' * 32}.json").write_text("{", encoding="utf-8")
    code, out, err = _cli(
        ["prune", "--dry-run", "--max-sessions", "10", "--max-age-days", "30"],
        capsys,
        clock=lambda: FROZEN,
    )
    assert code == 0
    assert out.splitlines()[0] == "dry-run: would delete 2"
    assert MID in out
    assert OLDER in out
    assert NEWER not in out.splitlines()[1:]
    assert "invalid session JSON" in err
    assert (sessions / f"{OLDER}.json").is_file()
    assert (sessions / f"{MID}.json").is_file()
    assert (sessions / f"{NEWER}.json").is_file()
    assert (sessions / f"{'d' * 32}.json").is_file()


def test_prune_deletes_and_json_matches_plan(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00")
    _write(sessions, MID, "2025-12-01T00:00:00+00:00")
    _write(sessions, NEWER, "2026-01-09T00:00:00+00:00")
    code, out, _err = _cli(
        ["prune", "--max-sessions", "10", "--max-age-days", "30", "--json"],
        capsys,
        clock=lambda: FROZEN,
    )
    assert code == 0
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert payload["schema"] == SCHEMA
    assert payload["command"] == "prune"
    assert payload["dry_run"] is False
    assert payload["deleted"] == 2
    assert set(payload["planned"]) == {OLDER, MID}
    assert payload["max_sessions"] == 10
    assert payload["max_age_days"] == 30
    assert not (sessions / f"{OLDER}.json").exists()
    assert not (sessions / f"{MID}.json").exists()
    assert (sessions / f"{NEWER}.json").is_file()


def test_prune_keeps_newest_by_count(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    kept = "1" * 32
    drop_a = "2" * 32
    drop_b = "3" * 32
    _write(sessions, kept, "2026-01-09T00:00:00+00:00")
    _write(sessions, drop_a, "2026-01-08T00:00:00+00:00")
    _write(sessions, drop_b, "2026-01-07T00:00:00+00:00")
    code, out, _err = _cli(
        ["prune", "--dry-run", "--max-sessions", "1", "--max-age-days", "3650"],
        capsys,
        clock=lambda: FROZEN,
    )
    assert code == 0
    assert "dry-run: would delete 2" in out
    assert drop_a in out
    assert drop_b in out
    assert kept not in out.splitlines()[1:]
    assert (sessions / f"{drop_a}.json").is_file()
    code, out, _err = _cli(
        ["prune", "--max-sessions", "1", "--max-age-days", "3650"],
        capsys,
        clock=lambda: FROZEN,
    )
    assert code == 0
    assert out == "deleted 2\n"
    assert (sessions / f"{kept}.json").is_file()
    assert not (sessions / f"{drop_a}.json").exists()
    assert not (sessions / f"{drop_b}.json").exists()


def test_prune_rejects_negative_bounds(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(sessions, NEWER, "2026-01-09T00:00:00+00:00")
    code, out, err = _cli(["prune", "--max-age-days", "-3"], capsys)
    assert code == 2
    assert out == ""
    assert "non-negative" in err
    assert (sessions / f"{NEWER}.json").is_file()
    code, _out, err = _cli(["prune", "--max-sessions", "-1"], capsys)
    assert code == 2
    assert "non-negative" in err


def test_sessions_help_lists_commands(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _app(["sessions", "--help"], capsys)
    assert code == 0
    for name in ("list", "show", "export", "rm", "prune"):
        assert name in out
    assert SCHEMA in out
    assert "not restarted" in out


def test_resume_help_without_a_terminal(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _resume(["--help"], capsys, isatty=lambda: False)
    assert code == 0
    assert "not restarted" in out
    assert "fx1-lite" in out
    assert "chat --resume" in out


def test_resume_refuses_a_non_tty(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _resume([], capsys, isatty=lambda: False)
    assert code == 2
    assert out == ""
    assert "requires an interactive terminal" in err


def test_resume_without_a_session_uses_chat_resume(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)

    code, _out, _err = _resume([], capsys, run_console=fake, isatty=lambda: True)
    assert code == 0
    assert seen == [{"model": "fx1", "resume": True, "model_explicit": False}]


def test_resume_model_flag_is_explicit(sessions: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)

    code, _out, _err = _resume(
        ["--model", "fx1-lite"], capsys, run_console=fake, isatty=lambda: True
    )
    assert code == 0
    assert seen[0]["model"] == "fx1-lite"
    assert seen[0]["model_explicit"] is True
    assert seen[0]["resume"] is True
    code, out, err = _resume(["--model", "other"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 2
    assert out == ""
    assert err


def test_resume_last_uses_the_latest_chat_path(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00")
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)

    code, _out, _err = _resume(["last"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 0
    assert seen[0]["resume"] is True
    assert "concierge_factory" not in seen[0]
    code, _out, _err = _resume([NEWER], capsys, run_console=fake, isatty=lambda: True)
    assert code == 0
    assert seen[1]["resume"] is True
    assert "concierge_factory" not in seen[1]


def test_resume_unknown_ambiguous_and_hostile_do_not_open_chat(
    sessions: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    left = "ab" + "0" * 30
    right = "ac" + "0" * 30
    _write(sessions, left, "2020-01-01T00:00:00+00:00")
    _write(sessions, right, "2021-01-01T00:00:00+00:00")
    canary = tmp_path / "x"
    canary.write_text("canary", encoding="utf-8")
    opened: list[str] = []

    def fake(**_kwargs: Any) -> None:
        opened.append("opened")

    code, out, err = _resume(["zzz"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 2
    assert out == ""
    assert "no session" in err
    code, out, err = _resume(["a"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 2
    assert "ambiguous" in err
    code, out, err = _resume(["../x"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 2
    assert opened == []
    assert canary.read_text(encoding="utf-8") == "canary"
    code, out, err = _resume(["last"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 0
    assert opened == ["opened"]


def test_resume_specific_session_restores_transcript(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        sessions,
        NEWER,
        "2024-01-01T00:00:00+00:00",
        messages=[{"role": "user", "content": "latest-body"}],
    )
    _write(
        sessions,
        OLDER,
        "2020-01-01T00:00:00+00:00",
        model="fx1-lite",
        messages=[{"role": "user", "content": "older-body"}],
        job=_idle_job(),
    )
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> None:
        seen.update(kwargs)

    code, _out, _err = _resume([OLDER], capsys, run_console=fake, isatty=lambda: True)
    assert code == 0
    assert "resume" not in seen
    concierge = seen["concierge_factory"]()
    assert concierge.model == "fx1-lite"
    assert concierge._messages == [{"role": "user", "content": "older-body"}]
    assert "resumed session" in concierge.resume_notice
    assert OLDER in concierge.resume_notice
    assert "not restarted" not in concierge.resume_notice
    assert concierge._job is None


def test_resume_does_not_restart_a_running_job(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00", job=_idle_job())
    _write(
        sessions,
        OLDER,
        "2020-01-01T00:00:00+00:00",
        model="fx1",
        messages=[{"role": "user", "content": "older-body"}],
        job=_running_job(),
    )
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> None:
        seen.update(kwargs)

    code, _out, _err = _resume(["a"], capsys, run_console=fake, isatty=lambda: True)
    assert code == 0
    concierge = seen["concierge_factory"]()
    assert concierge._job is None
    assert "not restarted" in concierge.resume_notice
    assert concierge._messages[0]["content"] == "older-body"
    saved = json.loads((sessions / f"{OLDER}.json").read_text(encoding="utf-8"))
    assert saved["job"]["state"] == "interrupted"
    assert saved["job"]["resume"] == "manual"
    assert saved["job"]["steps_completed"] == ["score"]
    assert saved["job"]["id"] == "job-1"
    latest = json.loads((sessions / f"{NEWER}.json").read_text(encoding="utf-8"))
    assert latest["job"]["state"] == "idle"
    names = sorted(path.name for path in sessions.glob("*.json"))
    assert names == [f"{OLDER}.json", f"{NEWER}.json"]


def test_resume_model_override_stays_in_memory(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(sessions, NEWER, "2024-01-01T00:00:00+00:00")
    _write(sessions, OLDER, "2020-01-01T00:00:00+00:00", model="fx1-lite", job=_idle_job())
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> None:
        seen.update(kwargs)

    code, _out, _err = _resume(
        [OLDER, "--model", "fx1"], capsys, run_console=fake, isatty=lambda: True
    )
    assert code == 0
    concierge = seen["concierge_factory"]()
    assert concierge.model == "fx1"
    assert "on model fx1" in concierge.resume_notice
    disk = json.loads((sessions / f"{OLDER}.json").read_text(encoding="utf-8"))
    assert disk["model"] == "fx1-lite"


def test_public_restore_hook_is_used_when_present() -> None:
    from dipcatcher_cli.sessions_cmd import _restore_specific

    class Host:
        def __init__(self, notice: str = "") -> None:
            self.model = "fx1"
            self.resume_notice = notice
            self.calls: list[str] = []
            self._session = {
                "id": "abc",
                "model": "fx1",
                "job": {"state": "interrupted"},
            }

        def restore(self, session_id: str) -> None:
            self.calls.append(session_id)

    host = Host()
    _restore_specific(host, "abc", None)
    assert host.calls == ["abc"]
    assert host.resume_notice == "the previous job was not restarted"
    already = Host("already not restarted")
    _restore_specific(already, "abc", "fx1-lite")
    assert already.calls == ["abc"]
    assert already.model == "fx1-lite"
    assert already._session["model"] == "fx1-lite"
    assert already.resume_notice == "already not restarted"


def test_model_choices_match_profile_ids() -> None:
    from dipcatcher_cli.sessions_cmd import _MODEL_IDS
    from fx1.interactive.profiles import MODELS

    assert tuple(MODELS) == _MODEL_IDS


def test_session_limits_match_the_store() -> None:
    from dipcatcher_cli.sessions_cmd import (
        _DEFAULT_LIST_LIMIT,
        _DEFAULT_MAX_AGE_DAYS,
        _DEFAULT_MAX_SESSIONS,
    )
    from fx1.interactive.sessions import (
        DEFAULT_LIST_LIMIT,
        DEFAULT_MAX_AGE_DAYS,
        DEFAULT_MAX_SESSIONS,
    )

    assert _DEFAULT_LIST_LIMIT == DEFAULT_LIST_LIMIT
    assert _DEFAULT_MAX_SESSIONS == DEFAULT_MAX_SESSIONS
    assert _DEFAULT_MAX_AGE_DAYS == DEFAULT_MAX_AGE_DAYS


def test_help_list_and_resume_do_not_import_concierge(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NO_COLOR": "1"}
    env["FX1_CONFIG_DIR"] = str(tmp_path / "cfg")
    env["FX1_SESSION_DIR"] = str(tmp_path / "sessions")
    script = """
import sys
from dipcatcher_cli.app import main
assert "dipcatcher_cli.sessions_cmd" not in sys.modules
assert "fx1.interactive.concierge" not in sys.modules
for args in (["sessions", "--help"], ["resume", "--help"]):
    try:
        main(args)
    except SystemExit as exc:
        assert exc.code == 0, args
    assert "fx1.interactive.concierge" not in sys.modules, args
try:
    main(["resume"])
except SystemExit as exc:
    assert exc.code == 2
else:
    raise SystemExit("resume should refuse a non-tty")
assert "fx1.interactive.concierge" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["FX1_CONFIG_DIR"]).exists()


def test_module_entrypoint_lists_json(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    directory.mkdir()
    _write(directory, NEWER, "2024-01-01T00:00:00+00:00", model="fx1-lite", turns=[_turn()])
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NO_COLOR": "1"}
    env["FX1_CONFIG_DIR"] = str(tmp_path / "cfg")
    env["FX1_SESSION_DIR"] = str(directory)
    result = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "sessions", "--limit", "5", "--json"],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == SCHEMA
    assert payload["sessions"][0]["id"] == NEWER
    assert payload["sessions"][0]["model"] == "fx1-lite"
    assert payload["sessions"][0]["turns"] == 1
    assert result.stderr == ""
    assert not Path(env["FX1_CONFIG_DIR"]).exists()


def test_app_dispatch_resume_reaches_console(
    sessions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.interactive import concierge

    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)

    monkeypatch.setattr(concierge, "run_console", fake)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    code, _out, _err = _app(["resume", "--model", "fx1-lite"], capsys)
    assert code == 0
    assert seen[0]["model"] == "fx1-lite"
    assert seen[0]["resume"] is True
    assert seen[0]["model_explicit"] is True


def test_importing_the_launcher_does_not_import_sessions(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "NO_COLOR": "1"}
    env["FX1_CONFIG_DIR"] = str(tmp_path / "cfg")
    script = """
import sys
import dipcatcher_cli.app
assert "dipcatcher_cli.sessions_cmd" not in sys.modules
assert "fx1.interactive.concierge" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
