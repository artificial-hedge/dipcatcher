"""Golden subprocess contract for ``dipcatcher exec`` exit codes and JSON."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fx1.stub_model_server import Reply, StubModelServer
from tests.fx1.ux_pty import hermetic_network, prepare, run_pty

ROOT = Path(__file__).resolve().parents[3]
_PY = sys.executable
_KEY = "sk-exec-contract-secret-aaa111"
_PLANTED = "sk-exec-planted-secret-value"
_EXEC_KEYS = {
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


def _env(tmp_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not _sensitive(key)}
    cfg = tmp_path / "cfg"
    sessions = tmp_path / "sessions"
    home = tmp_path / "home"
    for path in (cfg, sessions, home):
        path.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "FX1_CONFIG_DIR": str(cfg),
            "FX1_SESSION_DIR": str(sessions),
            "HOME": str(home),
            "NO_COLOR": "1",
            "TERM": "dumb",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.update(hermetic_network())
    return env


def _sensitive(key: str) -> bool:
    folded = key.casefold()
    return folded.endswith(("_api_key", "_token", "_secret")) or "password" in folded


def _configure(url: str) -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint("fx1", _KEY, url)


def _run(
    args: list[str],
    env: dict[str, str],
    *,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PY, "-m", "dipcatcher_cli", *args],
        cwd=ROOT,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StubModelServer:
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    stub = StubModelServer({"fx1": _KEY})
    stub.start()
    _configure(stub.url)
    yield stub
    stub.close()


def test_dipcatcher_contract_exec_exit_codes(tmp_path: Path, server: StubModelServer) -> None:
    env = _env(tmp_path)
    ok = _run(["exec", "--json", "hello"], env)
    assert ok.returncode == 0
    payload = json.loads(ok.stdout)
    assert set(payload) == _EXEC_KEYS
    assert payload["schema"] == "dipcatcher.exec.v1"
    assert payload["status"] == "ok"
    assert _KEY not in ok.stdout + ok.stderr

    server.queue(Reply.ok("echo " + _KEY, served_model="fx1"))
    leaked = _run(["exec", "ping"], env)
    assert leaked.returncode == 0
    assert _KEY not in leaked.stdout + leaked.stderr

    from fx1.interactive.profiles import set_endpoint

    set_endpoint("fx1", "sk-wrong-exec-key-value", server.url)
    runtime = _run(["exec", "--json", "hello"], env)
    assert runtime.returncode == 1
    assert json.loads(runtime.stdout)["status"] == "error"
    assert "sk-wrong-exec-key-value" not in runtime.stdout + runtime.stderr
    _configure(server.url)

    usage = _run(["exec", "--model", "nope", "hello"], env)
    assert usage.returncode == 2
    assert "unknown model" in usage.stderr

    server.queue(Reply.ok("The strategy Sharpe ratio is 2.4 after costs.", served_model="fx1"))
    withheld = _run(["exec", "--json", "headline"], env)
    assert withheld.returncode == 3
    assert json.loads(withheld.stdout)["status"] == "withheld"
    assert "2.4" not in withheld.stdout

    before = len(server.requests)
    timed = _run(["exec", "--json", "--timeout-s", "0", "hello"], env)
    assert timed.returncode == 4
    assert json.loads(timed.stdout)["status"] == "timeout"
    assert len(server.requests) == before

    bare = _env(tmp_path / "empty")
    missing = _run(["exec", "--json", "hello"], bare)
    assert missing.returncode == 5
    assert json.loads(missing.stdout)["status"] == "no_endpoint"


def test_dipcatcher_contract_exec_signals(tmp_path: Path, server: StubModelServer) -> None:
    env = _env(tmp_path)
    for sig, code in ((signal.SIGINT, 130), (signal.SIGTERM, 143), (signal.SIGHUP, 130)):
        seen = len(server.requests)
        server.queue(Reply.delay(30, Reply.ok("late", served_model="fx1")))
        proc = subprocess.Popen(
            [_PY, "-m", "dipcatcher_cli", "exec", "--json", "--timeout-s", "45", "ping"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert server.wait_for_requests(seen + 1, timeout=60)
            assert proc.pid is not None
            proc.send_signal(sig)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=5)
        assert proc.returncode == code
        assert _KEY not in out + err


def test_dipcatcher_contract_exec_stdin_and_tty(tmp_path: Path, server: StubModelServer) -> None:
    env = _env(tmp_path)
    piped = _run(["exec"], env, stdin="piped prompt\n")
    assert piped.returncode == 0
    assert "stub fx1 reply" in piped.stdout
    dashed = _run(["exec", "-"], env, stdin="dashed prompt\n")
    assert dashed.returncode == 0
    root = prepare(tmp_path, "tty")
    refused = run_pty(root, [("wait", b"terminal")], args=("exec",), wrap=False, extra=env)
    assert refused.status == 2


def test_dipcatcher_contract_exec_output_and_sessions(
    tmp_path: Path, server: StubModelServer
) -> None:
    del server
    env = _env(tmp_path)
    last = tmp_path / "nested" / "last.txt"
    wrote = _run(["exec", "--output-last-message", str(last), "--session", "none", "hello"], env)
    assert wrote.returncode == 0
    assert last.read_text(encoding="utf-8").strip() == wrote.stdout.strip()
    assert stat.S_IMODE(last.stat().st_mode) == 0o600
    assert list(last.parent.glob("*.tmp")) == []
    assert list((tmp_path / "sessions").glob("*.json")) == []

    created = _run(["exec", "--json", "--session", "new", "hello"], env)
    body = json.loads(created.stdout)
    assert created.returncode == 0
    session_id = body["session_id"]
    assert isinstance(session_id, str) and session_id
    files = list((tmp_path / "sessions").glob("*.json"))
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600

    resumed = _run(["exec", "--session", "last", "continue"], env)
    assert resumed.returncode == 0
    by_id = _run(["exec", "--json", "--session", session_id, "again"], env)
    assert by_id.returncode == 0
    assert json.loads(by_id.stdout)["session_id"] == session_id
    assert _PLANTED not in created.stdout
    help_text = _run(["exec", "--help"], env)
    assert help_text.returncode == 0
    for snippet in ("0    ok", "130  interrupted", "143  terminated", "SIGHUP", "no_endpoint"):
        assert snippet in help_text.stdout
