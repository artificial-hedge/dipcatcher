"""Non-interactive exec driver: approvals, signals, and owned-process cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fx1.interactive.approvals import ApprovalRequest
from fx1.interactive.headless import (
    EXIT_BUDGET,
    EXIT_INTERRUPTED,
    EXIT_NO_ENDPOINT,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_TERMINATED,
    EXIT_USAGE,
    EXIT_WITHHELD,
    ExecRequest,
    cleanup_owned,
    exec_approvals,
    exit_code_for,
    install_exec_signals,
    run_exec,
)
from fx1.interactive.procs import LiveProc, ProcessRegistry

ROOT = Path(__file__).resolve().parents[2]
FX1_KEY = "sk-test-fx1-secret-value-aaa111"


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
    for name in ("MOONSHOT_API_KEY", "FX1_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_exec_approvals_honor_only_listed_keys() -> None:
    gate = exec_approvals(("harness:research",))
    allowed = ApprovalRequest("harness:research", "consequential", "run research?")
    denied = ApprovalRequest("harness:train", "consequential", "run train?")
    assert gate.policy.decide(allowed).value == "once"
    assert gate.policy.decide(denied).value == "deny"
    assert gate.ask("run research?", kind="consequential", key="harness:research") is True
    assert gate.ask("run train?", kind="consequential", key="harness:train") is False
    assert gate.ask("run train?", kind="consequential") is False


@pytest.mark.parametrize(
    ("status", "signum", "code"),
    [
        ("ok", None, EXIT_OK),
        ("error", None, EXIT_RUNTIME),
        ("withheld", None, EXIT_WITHHELD),
        ("timeout", None, EXIT_BUDGET),
        ("budget", None, EXIT_BUDGET),
        ("no_endpoint", None, EXIT_NO_ENDPOINT),
        ("cancelled", None, EXIT_INTERRUPTED),
        ("ok", int(signal.SIGINT), EXIT_INTERRUPTED),
        ("ok", int(signal.SIGTERM), EXIT_TERMINATED),
    ],
)
def test_exit_code_matrix(status: str, signum: int | None, code: int) -> None:
    assert exit_code_for(status, signum) == code
    assert EXIT_USAGE == 2


def test_install_signals_skips_missing_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(signal, "SIGTERM", raising=False)
    installed = install_exec_signals(lambda _signum, _frame: None)
    try:
        assert any(sig == signal.SIGINT for sig, _handler in installed)
        assert all(sig != getattr(signal, "SIGTERM", None) for sig, _handler in installed)
    finally:
        for sig, handler in installed:
            signal.signal(sig, handler)


def test_run_exec_restores_sigint(isolated: Path) -> None:
    del isolated
    previous = signal.getsignal(signal.SIGINT)
    outcome = run_exec(ExecRequest(prompt="/help", timeout_s=2), install_signals=True)
    assert outcome.status == "ok"
    assert "/superpower" in outcome.answer
    assert signal.getsignal(signal.SIGINT) == previous


def test_cleanup_owned_reaps_a_sleeper() -> None:
    registry = ProcessRegistry()
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        registry.add(
            LiveProc(
                pid=sleeper.pid,
                pgid=os.getpgid(sleeper.pid),
                started_at=time.monotonic(),
                argv_summary=["sleeper"],
                step="cleanup",
            )
        )
        cleanup_owned(registry=registry, grace_s=0.2)
        assert sleeper.wait(timeout=1) is not None
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait(timeout=1)


def test_approve_key_runs_harness_and_omission_denies(isolated: Path) -> None:
    del isolated

    class _Result:
        stdout = "synthetic harness marker"
        stderr = ""
        exit_code = 0

    class _Harness:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(
            self,
            name: str,
            args: list[str],
            cancel_event: object = None,
            env_extra: object = None,
        ) -> _Result:
            del args, cancel_event, env_extra
            self.calls.append(name)
            return _Result()

    denied = _Harness()
    blocked = run_exec(
        ExecRequest(prompt="/harness run train", timeout_s=5),
        harness=denied,
        install_signals=False,
    )
    assert blocked.exit_code == EXIT_RUNTIME
    assert denied.calls == []
    assert any("harness:train" in item for item in blocked.errors)

    allowed = _Harness()
    ran = run_exec(
        ExecRequest(
            prompt="/harness run train",
            timeout_s=5,
            approve_keys=("harness:train",),
        ),
        harness=allowed,
        install_signals=False,
    )
    assert allowed.calls == ["train"]
    assert ran.exit_code == EXIT_OK
    assert "synthetic harness marker" in ran.answer


def test_signal_cleans_up_owned_child(isolated: Path) -> None:
    from fx1.interactive.profiles import set_endpoint
    from tests.fx1.stub_model_server import Reply, StubModelServer

    pid_file = isolated / "child.pid"
    state_file = isolated / "child.state"
    server = StubModelServer({"fx1": FX1_KEY})
    server.start()
    proc: subprocess.Popen[str] | None = None
    try:
        set_endpoint("fx1", FX1_KEY, server.url)
        server.queue(Reply.delay(20, Reply.ok("late", served_model="fx1")))
        env = os.environ.copy()
        src = str(ROOT / "src")
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src if not current else src + os.pathsep + current
        env["CHILD_PID_FILE"] = str(pid_file)
        env["CHILD_STATE_FILE"] = str(state_file)
        env["NO_COLOR"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-c", _SIGNAL_SCRIPT],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert server.wait_for_requests(1, timeout=4)
        assert proc.pid is not None
        proc.send_signal(signal.SIGINT)
        _out, err = proc.communicate(timeout=3)
    finally:
        server.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=2)
    assert proc is not None
    assert proc.returncode == 130, err
    assert state_file.read_text(encoding="utf-8") == "reaped"
    assert FX1_KEY not in (err or "")


_SIGNAL_SCRIPT = r"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from fx1.interactive.headless import ExecInterrupted, ExecRequest, run_exec
from fx1.interactive.procs import REGISTRY, LiveProc

sleeper = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
Path(os.environ["CHILD_PID_FILE"]).write_text(str(sleeper.pid))


def factory(**kwargs):
    REGISTRY.add(
        LiveProc(
            pid=sleeper.pid,
            pgid=os.getpgid(sleeper.pid),
            started_at=time.monotonic(),
            argv_summary=["sleeper"],
            step="signal-test",
        )
    )
    from fx1.interactive.model_client import ModelClient, RetryPolicy

    timeout = float(kwargs.pop("timeout_s", 20))
    return ModelClient(timeout_s=timeout, retry=RetryPolicy(max_attempts=1), **kwargs)


request = ExecRequest(prompt="ping", model="fx1", timeout_s=20, session="none")
try:
    outcome = run_exec(request, backend_factory=factory, install_signals=True)
    code = outcome.exit_code
except ExecInterrupted as exc:
    code = 143 if exc.signum == signal.SIGTERM else 130
try:
    sleeper.wait(timeout=1)
    Path(os.environ["CHILD_STATE_FILE"]).write_text("reaped")
except subprocess.TimeoutExpired:
    Path(os.environ["CHILD_STATE_FILE"]).write_text("alive")
    sleeper.kill()
sys.exit(code)
"""
