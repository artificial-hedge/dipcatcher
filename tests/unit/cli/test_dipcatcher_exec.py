"""Scripted ``dipcatcher exec`` and the models command surface."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fx1.stub_model_server import Reply, StubModelServer

ROOT = Path(__file__).resolve().parents[3]
FX1_KEY = "sk-test-fx1-secret-value-aaa111"
LITE_KEY = "sk-test-lite-secret-value-bbb222"
EXEC_KEYS = {
    "schema",
    "status",
    "model",
    "served_model",
    "answer",
    "usage",
    "latency_s",
    "evidence",
    "session_id",
    "warnings",
    "errors",
    "duration_s",
}


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "cfg"
    sessions = tmp_path / "sessions"
    config.mkdir()
    sessions.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(config))
    monkeypatch.setenv("FX1_SESSION_DIR", str(sessions))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for name in (
        "MOONSHOT_API_KEY",
        "FX1_BASE_URL",
        "MOONSHOT_API_URL",
        "DIPCATCHER_FX1_API_KEY",
        "DIPCATCHER_FX1_BASE_URL",
        "DIPCATCHER_FX1_LITE_API_KEY",
        "DIPCATCHER_FX1_LITE_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _configure(model: str, key: str, url: str) -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint(model, key, url)


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    from dipcatcher_cli.app import main

    try:
        main(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    env["NO_COLOR"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    return env


@pytest.fixture
def both_models(isolated: Path) -> Iterator[StubModelServer]:
    del isolated
    server = StubModelServer({"fx1": FX1_KEY, "fx1-lite": LITE_KEY})
    server.start()
    _configure("fx1", FX1_KEY, server.url)
    _configure("fx1-lite", LITE_KEY, server.url)
    try:
        yield server
    finally:
        server.close()


@pytest.mark.parametrize("model", ["fx1", "fx1-lite"])
def test_exec_plain_answers_for_both_models(
    model: str, both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    del both_models
    code, out, err = _run(["exec", "--model", model, "hello"], capsys)
    assert code == 0
    assert out.strip() == f"stub {model} reply"
    assert f"stub {model} reply" not in err
    assert FX1_KEY not in out + err
    assert LITE_KEY not in out + err


@pytest.mark.parametrize("model", ["fx1", "fx1-lite"])
def test_exec_json_document_for_both_models(
    model: str, both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    del both_models
    code, out, err = _run(["exec", "--json", "--model", model, "hello"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert set(payload) == EXEC_KEYS
    assert payload["schema"] == "dipcatcher.exec.v1"
    assert payload["status"] == "ok"
    assert payload["model"] == model
    assert payload["served_model"] == model
    assert payload["answer"] == f"stub {model} reply"
    assert isinstance(payload["usage"], dict)
    assert isinstance(payload["latency_s"], float)
    assert payload["session_id"] is None
    assert FX1_KEY not in out + err
    assert LITE_KEY not in err


def test_exec_no_endpoint_is_exit_5(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, out, err = _run(["exec", "hello"], capsys)
    assert code == 5
    assert out.strip() == ""
    assert "/keys set fx1" in err
    assert "MOONSHOT_API_KEY" in err


def test_exec_json_no_endpoint(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, out, _err = _run(["exec", "--json", "hello"], capsys)
    payload = json.loads(out)
    assert code == 5
    assert payload["status"] == "no_endpoint"
    assert payload["answer"] == ""
    assert payload["errors"]


def test_exec_rejects_unknown_model(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, _out, err = _run(["exec", "--model", "other", "hello"], capsys)
    assert code == 2
    assert "unknown model" in err


def test_exec_rejects_negative_timeout(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, _out, err = _run(["exec", "--timeout-s", "-1", "hello"], capsys)
    assert code == 2
    assert "timeout-s" in err


def test_exec_zero_timeout_does_not_contact(
    both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _err = _run(["exec", "--json", "--timeout-s", "0", "hello"], capsys)
    payload = json.loads(out)
    assert code == 4
    assert payload["status"] == "timeout"
    assert both_models.requests == []


def test_exec_timeout_on_slow_reply(
    both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    both_models.queue(Reply.delay(8, Reply.ok("late", served_model="fx1")))
    code, out, _err = _run(["exec", "--json", "--timeout-s", "0.35", "slow"], capsys)
    payload = json.loads(out)
    assert code == 4
    assert payload["status"] == "timeout"
    assert payload["answer"] == ""


def test_exec_withheld_by_honesty_gate(
    both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    both_models.queue(Reply.ok("The strategy Sharpe ratio is 2.4 after costs.", served_model="fx1"))
    code, out, _err = _run(["exec", "headline"], capsys)
    assert code == 3
    assert "withheld" in out.casefold()
    assert "2.4" not in out


def test_exec_json_withheld(
    both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    both_models.queue(Reply.ok("The strategy Sharpe ratio is 2.4 after costs.", served_model="fx1"))
    code, out, _err = _run(["exec", "--json", "headline"], capsys)
    payload = json.loads(out)
    assert code == 3
    assert payload["status"] == "withheld"
    assert "withheld" in payload["answer"].casefold()


def test_exec_http_401(both_models: StubModelServer, capsys: pytest.CaptureFixture[str]) -> None:
    _configure("fx1", "sk-wrong-key-value-not-matching", both_models.url)
    code, out, err = _run(["exec", "--json", "hello"], capsys)
    payload = json.loads(out)
    assert code == 1
    assert payload["status"] == "error"
    assert payload["answer"] == ""
    blob = out + err
    assert FX1_KEY not in blob
    assert "sk-wrong-key-value-not-matching" not in blob


def test_exec_redacts_planted_secret(
    both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    both_models.queue(Reply.ok(f"echo {FX1_KEY}", served_model="fx1"))
    code, out, err = _run(["exec", f"please ignore {FX1_KEY}"], capsys)
    assert code == 0
    assert FX1_KEY not in out
    assert FX1_KEY not in err


def test_exec_reads_prompt_from_stdin_dash(
    both_models: StubModelServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("hello from stdin\n"))
    code, out, _err = _run(["exec", "-"], capsys)
    assert code == 0
    assert out.strip() == "stub fx1 reply"
    assert both_models.requests


def test_exec_reads_piped_prompt_without_dash(
    both_models: StubModelServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("piped prompt\n"))
    code, out, _err = _run(["exec"], capsys)
    assert code == 0
    assert out.strip() == "stub fx1 reply"


def test_exec_refuses_to_block_on_a_terminal(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    code, _out, err = _run(["exec"], capsys)
    assert code == 2
    assert "terminal" in err


def test_exec_session_none_writes_nothing(
    both_models: StubModelServer, isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del both_models
    code, _out, _err = _run(["exec", "--session", "none", "hello"], capsys)
    assert code == 0
    assert list((isolated / "sessions").glob("*.json")) == []


def test_exec_session_new_persists_0600(
    both_models: StubModelServer, isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del both_models
    code, out, _err = _run(["exec", "--json", "--session", "new", "hello"], capsys)
    payload = json.loads(out)
    assert code == 0
    assert isinstance(payload["session_id"], str) and payload["session_id"]
    files = list((isolated / "sessions").glob("*.json"))
    assert len(files) == 1
    assert files[0].stat().st_mode & 0o777 == 0o600
    stored = files[0].read_text(encoding="utf-8")
    assert FX1_KEY not in stored


def test_exec_session_last_resumes_context(
    both_models: StubModelServer, isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.interactive.sessions import SessionStore

    store = SessionStore(isolated / "sessions")
    store.create(model="fx1", messages=[{"role": "user", "content": "earlier context token ALPHA"}])
    code, _out, _err = _run(["exec", "--session", "last", "continue"], capsys)
    assert code == 0
    body = json.dumps(both_models.requests[-1].body)
    assert "ALPHA" in body


def test_exec_session_last_missing_is_usage_error(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    code, _out, err = _run(["exec", "--session", "last", "hello"], capsys)
    assert code == 2
    assert err.strip()


def test_exec_rejects_bad_session_id(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, _out, err = _run(["exec", "--session", "../etc/passwd", "hello"], capsys)
    assert code == 2
    assert "session" in err


def test_exec_output_last_message_is_private(
    both_models: StubModelServer, isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del both_models
    path = isolated / "nested" / "last.txt"
    code, out, _err = _run(
        ["exec", "--output-last-message", str(path), "hello"],
        capsys,
    )
    assert code == 0
    assert path.read_text(encoding="utf-8") == out.strip()
    assert path.stat().st_mode & 0o777 == 0o600


def test_exec_help_works_offline(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, out, _err = _run(["exec", "/help"], capsys)
    assert code == 0
    assert "/superpower" in out


def test_exec_exit_is_a_noop(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _err = _run(["exec", "--session", "new", "/exit"], capsys)
    assert code == 0
    assert out.strip() == ""
    assert list((isolated / "sessions").glob("*.json")) == []


def test_exec_refuses_keys_set(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del isolated
    code, out, err = _run(["exec", "/keys", "set", "fx1"], capsys)
    assert code == 2
    assert out.strip() == ""
    assert "refuses /keys set" in err


def test_exec_denies_consequential_harness_without_approval(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    code, out, err = _run(["exec", "/harness", "run", "train"], capsys)
    assert code == 1
    assert out.strip() == ""
    assert "denied harness:train" in err


def test_exec_json_keys_set_is_one_document(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    code, out, _err = _run(["exec", "--json", "/keys", "set"], capsys)
    payload = json.loads(out)
    assert code == 2
    assert payload["status"] == "error"
    assert set(payload) == EXEC_KEYS


@pytest.mark.parametrize("sig,expected", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_exec_signal_exits(isolated: Path, sig: int, expected: int) -> None:
    from tests.fx1.stub_model_server import Reply, StubModelServer

    server = StubModelServer({"fx1": FX1_KEY})
    server.start()
    proc: subprocess.Popen[str] | None = None
    try:
        _configure("fx1", FX1_KEY, server.url)
        server.queue(Reply.delay(20, Reply.ok("late", served_model="fx1")))
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dipcatcher_cli",
                "exec",
                "--json",
                "--timeout-s",
                "15",
                "--model",
                "fx1",
                "ping",
            ],
            cwd=ROOT,
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert server.wait_for_requests(1, timeout=4)
        assert proc.pid is not None
        proc.send_signal(sig)
        out, _err = proc.communicate(timeout=3)
    finally:
        server.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=2)
    assert proc is not None
    assert proc.returncode == expected
    if out.strip():
        payload = json.loads(out)
        assert payload["status"] == "cancelled"
        assert FX1_KEY not in out


def test_models_list_json_is_presence_only(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    code, out, err = _run(["models", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0
    assert payload["schema"] == "dipcatcher.models.v1"
    assert [row["model"] for row in payload["models"]] == ["fx1", "fx1-lite"]
    assert FX1_KEY not in out + err
    assert "available" not in out.casefold()


def test_models_check_offline_says_nothing_was_contacted(
    both_models: StubModelServer, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _err = _run(["models", "check"], capsys)
    assert code == 0
    assert "nothing was contacted" in out
    assert both_models.requests == []


def test_models_check_missing_endpoint_exits_5(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    code, out, _err = _run(["models", "check", "--json"], capsys)
    payload = json.loads(out)
    assert code == 5
    assert payload["schema"] == "dipcatcher.models_check.v1"
    assert payload["contacted"] is False
    assert "nothing was contacted" in payload["note"]


def test_models_check_unknown_model_exits_2(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del isolated
    code, _out, err = _run(["models", "check", "other"], capsys)
    assert code == 2
    assert err.strip()


def test_plain_mode_output_cannot_inject_terminal_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from dipcatcher_cli import exec_cmd

    outcome = SimpleNamespace(
        answer="ok \x1b]0;pwned\x07 \x1b[2J done",
        warnings=["w \x9b31m"],
        errors=[],
        as_json=lambda: {"answer": "x"},
    )
    exec_cmd._emit(outcome, json_mode=False)
    captured = capsys.readouterr()
    assert "\x1b" not in captured.out and "\x07" not in captured.out
    assert "\x9b" not in captured.err
    assert "ok" in captured.out and "done" in captured.out
