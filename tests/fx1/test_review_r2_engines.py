"""Merge-defect locks for the engine, terminal, and security review."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive import model_client, procs
from fx1.interactive.approvals import ActionBounds, ApprovalDecision, ApprovalGate, ApprovalRequest
from fx1.interactive.chat_tools import ToolEnv, execute_tool_call
from fx1.interactive.console import run_console
from fx1.interactive.redact import REDACTED, SecretRegistry, redact_text
from fx1.interactive.terminal import sanitize_for_terminal
from fx1.interactive.tty_guard import _SignalGuard

_LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Linux process wrapper")
_SIGPIPE_BIT = 1 << (signal.SIGPIPE - 1)
_SIGXFSZ_BIT = 1 << (getattr(signal, "SIGXFSZ", 25) - 1)


def _sigign(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("SigIgn:"):
            return int(line.split()[1], 16)
    raise AssertionError(f"no SigIgn for {pid}")


def _comm(pid: int) -> str:
    return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()


def _script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _limits() -> procs.ProcLimits:
    return procs.ProcLimits(timeout_s=20.0, kill_grace_s=1.0, cancel_poll_s=0.05)


@_LINUX
def test_wrapped_child_does_not_ignore_sigpipe() -> None:
    """The exec wrapper must not leave Python's SIGPIPE/SIGXFSZ ignore in the child."""
    cat = "/bin/cat" if Path("/bin/cat").is_file() else "/usr/bin/cat"
    argv = procs._death_argv([cat])
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None or proc.pid is None:
                break
            try:
                if _comm(proc.pid) == "cat":
                    mask = _sigign(proc.pid)
                    assert mask & _SIGPIPE_BIT == 0
                    assert mask & _SIGXFSZ_BIT == 0
                    return
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        pytest.fail(f"cat did not stay up under the wrapper (code={proc.poll()})")
    finally:
        if proc.poll() is None and proc.pid is not None:
            os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=20)


@_LINUX
def test_wrapper_still_execs_when_prctl_cannot_load(tmp_path: Path) -> None:
    blocked = tmp_path / "ctypes.py"
    blocked.write_text(
        "class _Lib:\n"
        "    def prctl(self, *args):\n"
        "        raise OSError('blocked')\n"
        "def CDLL(*args, **kwargs):\n"
        "    return _Lib()\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    argv = procs._death_argv(["/bin/true"])
    proc = subprocess.run(argv, env=env, capture_output=True, timeout=20, check=False)
    assert proc.returncode == 0


@_LINUX
def test_wrapper_kills_when_recorded_parent_pid_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Death is 'parent pid changed', not 'parent pid is 1'. A recorded pid of 1
    with a different live parent is the reparent race, so the child must die.
    A live init parent records pid 1 and getppid()==1, so the same comparison
    leaves it alive."""
    monkeypatch.setattr(os, "getpid", lambda: 1)
    argv = procs._death_argv(["/bin/true"])
    proc = subprocess.run(argv, capture_output=True, timeout=20, check=False)
    assert proc.returncode < 0


@_LINUX
def test_relative_tool_missing_from_child_cwd_is_a_spawn_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    _script(parent / "tool", "#!/bin/sh\necho hi\n")
    monkeypatch.chdir(parent)
    result = procs.run_owned(
        ["./tool"],
        _limits(),
        cwd=child,
        registry=procs.ProcessRegistry(),
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.spawn_error is not None
    assert result.stdout == ""


def test_redact_secrets_hidden_by_marks_and_ligatures() -> None:
    key = "sk-" + ("a" * 20)
    marked = key[:6] + "\u0301" + key[6:]
    ligature = "sk-\ufb01" + ("b" * 16)
    registry = SecretRegistry()
    registry.register("supersecretvalue")
    hidden = "pre " + "super" + "\u0301" + "secretvalue post"
    assert redact_text(marked) == REDACTED
    assert redact_text("pre " + marked + " post") == f"pre {REDACTED} post"
    assert redact_text(ligature) == REDACTED
    assert redact_text(hidden, registry=registry) == f"pre {REDACTED} post"
    assert redact_text("café receipt " + ("ab" * 32)) == "café receipt " + ("ab" * 32)
    once = redact_text(marked)
    assert redact_text(once) == once


def test_honesty_rejects_glued_decimal_headlines() -> None:
    for phrase in ("sharpe3.2", "NAV1.9", "pnl4.2", "calmar2.1", "sharpe_3.2", "sortino_1.5"):
        with pytest.raises(Fx1HonestyError):
            validate_fx1_output(phrase)
    for phrase in (
        "mySharpe 2.1",
        "sharpe of two point one",
        "the sharpe ratio is useful",
        "pinball 0.21",
        "sharpe3",
        "navigation 1.9",
    ):
        assert validate_fx1_output(phrase) == phrase


def test_keep_sgr_drops_conceal_and_keeps_color() -> None:
    hidden = sanitize_for_terminal("see \x1b[8msecret\x1b[0m end", keep_sgr=True)
    assert "secret" in hidden
    assert "\x1b[8m" not in hidden
    mixed = sanitize_for_terminal("\x1b[31;8mnope\x1b[0m", keep_sgr=True)
    assert "\x1b[8m" not in mixed
    assert "nope" in mixed
    colored = sanitize_for_terminal("\x1b[1;31mred\x1b[0m", keep_sgr=True)
    assert colored == "\x1b[1;31mred\x1b[0m"
    assert sanitize_for_terminal(colored, keep_sgr=True) == colored
    plain = "keep\n\tcafé"
    assert sanitize_for_terminal(sanitize_for_terminal(plain, keep_sgr=True)) == plain


def test_proxy_and_tls_handshake_use_the_connect_budget() -> None:
    seen: dict[str, float | None] = {}

    def _watch(conn: Any, *, tls: bool) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        def create(
            _address: object,
            timeout: float | None = None,
            _source: object = None,
            *_args: object,
            **_kwargs: object,
        ) -> socket.socket:
            sock.settimeout(timeout)
            return sock

        def tunnel() -> None:
            seen["tunnel"] = sock.gettimeout()

        def wrap(raw: socket.socket, server_hostname: str | None = None) -> socket.socket:
            del server_hostname
            seen["tls"] = raw.gettimeout()
            return raw

        conn._create_connection = create
        conn._tunnel = tunnel
        conn._context.wrap_socket = wrap
        try:
            conn.connect()
            seen["after"] = conn.sock.gettimeout() if conn.sock is not None else None
        finally:
            sock.close()
        assert tls

    proxied = model_client._https_connection("example.test", 443, 90.0, ("127.0.0.1", 9))
    _watch(proxied, tls=True)
    assert seen["tunnel"] == model_client._CONNECT_TIMEOUT_S
    assert seen["tls"] == model_client._CONNECT_TIMEOUT_S
    assert seen["after"] == 90.0

    seen.clear()
    direct = model_client._https_connection("example.test", 443, 90.0, None)
    _watch(direct, tls=True)
    assert seen["tls"] == model_client._CONNECT_TIMEOUT_S
    assert seen["after"] == 90.0

    seen.clear()
    short = model_client._https_connection("example.test", 443, 2.0, ("127.0.0.1", 9))
    _watch(short, tls=True)
    assert seen["tunnel"] == 2.0
    assert seen["tls"] == 2.0
    assert seen["after"] == 2.0


def test_fatal_tty_signal_chains_the_previous_handler() -> None:
    called: dict[str, int] = {}

    def previous(signum: int, _frame: object) -> None:
        called["sig"] = signum
        raise SystemExit(143)

    read_fd, write_fd = os.pipe()
    try:
        guard = _SignalGuard(read_fd, write_fd, [False])
        guard._prev = [(signal.SIGTERM, previous)]
        with pytest.raises(SystemExit) as info:
            guard._on_fatal(signal.SIGTERM, None)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert called["sig"] == signal.SIGTERM
    assert info.value.code == 143


def test_console_arms_terminal_stop_for_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    before = signal.getsignal(signal.SIGTERM)
    seen: dict[str, object] = {}

    def read(_prompt: str) -> str:
        seen["during"] = signal.getsignal(signal.SIGTERM)
        raise EOFError

    class _Fake:
        model = "fx1"
        session_id = "s_review"

        def shutdown(self) -> None:
            return None

    run_console(input_fn=read, output_fn=lambda *_a, **_k: None, concierge_factory=lambda: _Fake())
    assert seen["during"] is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_session_grant_does_not_replay_a_different_argv() -> None:
    prompts = {"n": 0}

    def decide(_request: ApprovalRequest) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.SESSION

    gate = ApprovalGate(decide_fn=decide)
    first = ActionBounds(argv=("train", "--a"), writes=True)
    second = ActionBounds(argv=("train", "--b"), writes=True)
    assert gate.ask("run", kind="consequential", key="harness:train", bounds=first) is True
    assert gate.ask("run", kind="consequential", key="harness:train", bounds=first) is True
    assert gate.ask("run", kind="consequential", key="harness:train", bounds=second) is True
    assert prompts["n"] == 2
    assert gate.records[1].scope == "session-reuse"


def test_oversized_tool_json_does_not_run_the_named_command(tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    class _Harness:
        def run(self, name: str, argv: list[str], **_kwargs: object) -> object:
            calls.append((name, list(argv)))
            return type("Result", (), {"exit_code": 0, "stdout": "ran", "stderr": ""})()

    blob = json.dumps({"name": "train", "args": ["--danger"], "pad": "a" * 40_000})
    env = ToolEnv(
        store=None,
        researcher=None,
        harness=_Harness(),
        approvals=ApprovalGate(decide_fn=lambda _request: ApprovalDecision.SESSION),
        validator=None,
        cancel_event=None,
        session_id="s_review",
        say=lambda _text: None,
        say_error=lambda _exc: None,
        run_superpower=lambda _goal: "no",
        cwd=tmp_path,
        config_dir=tmp_path / "cfg",
    )
    text = execute_tool_call(
        {"function": {"name": "harness_run", "arguments": blob}},
        env,
    )
    assert calls == []
    assert text == "unknown_command"
