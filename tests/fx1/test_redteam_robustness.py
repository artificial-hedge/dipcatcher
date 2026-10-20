"""Hostile parser inputs and Windows/macOS branches exercised on Linux.

Windows process-tree kill, icacls, msvcrt byte locks, cp1252 console I/O, and
``os.replace`` against a file another process has open are not executed here.
Those branches are reached by monkeypatch or injection. A green result does
not mean Windows or macOS behavior was verified.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dipcatcher_cli.exec_cmd import _sigterm, _write_json, main
from fx1.flash.store import FlashStore
from fx1.interactive import profiles
from fx1.interactive.atomicstore import (
    StoreCorruptError,
    StoreLockTimeout,
    atomic_write_text,
    permission_problem,
    read_json,
)
from fx1.interactive.chat_tools import parse_tool_call
from fx1.interactive.concierge import Concierge, _family_payload_from_harness
from fx1.interactive.console import (
    _WINDOWS_NOTICE,
    _emit_engine_banner,
    _polls,
    _tty_polling_read,
    run_console,
)
from fx1.interactive.contracts import (
    CapabilityCatalog,
    CatalogUnavailable,
    ContractError,
    fetch_catalog,
)
from fx1.interactive.families import FamilyClient, FamilyClientError
from fx1.interactive.headless import (
    ExecRequest,
    _write_last_message,
    install_exec_signals,
    run_exec,
)
from fx1.interactive.jobs import SystemProcessProbe, terminate_process
from fx1.interactive.model_client import ErrorKind, ModelClient, ModelError, RetryPolicy
from fx1.interactive.planner import Exclusion, _call_model
from fx1.interactive.policy import PolicyViolation, contain_path
from fx1.interactive.procs import ProcessRegistry, ProcLimits, _apply_kill, _taskkill, run_owned
from fx1.interactive.profiles import protect_store_windows
from fx1.interactive.sessions import SessionError, SessionStore
from fx1.interactive.terminal import (
    ConsoleWriter,
    LineReader,
    TermCaps,
    _can_use_tty,
    _scoop_partial_text,
    detect_caps,
)
from fx1.serve.backends import ToolCompletion

_KEY = "robust-key-rt6"
_FUZZ = settings(
    max_examples=16,
    deadline=500,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
_DEPTH = 10_000
_PLAIN = TermCaps(False, False, False, False, True, 80, 24, False)


def _nested(depth: int) -> str:
    return '{"a":' * depth + "0" + "}" * depth


def _huge_int() -> str:
    return "1" * 5000


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def _concierge(tmp_path: Path, backend: Any = None) -> tuple[Concierge, list[str]]:
    out: list[str] = []
    factory = None if backend is None else (lambda **_kwargs: backend)
    concierge = Concierge(
        store=FlashStore(tmp_path / "flash.jsonl"),
        output_fn=out.append,
        animated=False,
        session_store=SessionStore(tmp_path / "sessions"),
        backend_factory=factory,
        input_fn=lambda _prompt: "",
    )
    return concierge, out


def _client(port: int, *, timeout_s: float = 1.0) -> ModelClient:
    return ModelClient(
        api_key=_KEY,
        api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
        timeout_s=timeout_s,
        retry=RetryPolicy(max_attempts=1),
    )


def _drain(conn: socket.socket) -> None:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = conn.recv(4096)
        if not chunk:
            return
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1].strip())
            except ValueError:
                length = 0
    missing = length - len(rest)
    while missing > 0:
        chunk = conn.recv(min(4096, missing))
        if not chunk:
            return
        missing -= len(chunk)


def _serve(send: Callable[[socket.socket], None]) -> tuple[int, threading.Thread, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    sock.settimeout(2.0)
    port = sock.getsockname()[1]

    def run() -> None:
        try:
            conn, _addr = sock.accept()
        except OSError:
            return
        try:
            conn.settimeout(1.0)
            _drain(conn)
            send(conn)
        except OSError:
            return
        finally:
            conn.close()

    thread = threading.Thread(target=run, name="rt6-http", daemon=True)
    thread.start()
    return port, thread, sock


def _finish(thread: threading.Thread, sock: socket.socket) -> None:
    sock.close()
    thread.join(timeout=2.0)


def _utf8_error() -> UnicodeDecodeError:
    return UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


class _Burst:
    def __init__(self, completion: ToolCompletion) -> None:
        self.completion = completion
        self.calls = 0

    def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> ToolCompletion:
        del messages, kwargs
        self.calls += 1
        return self.completion


class _SurrogateBackend:
    def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> ToolCompletion:
        del messages, kwargs
        return ToolCompletion(content="\ud800hi", tool_calls=None, finish_reason="stop")


class _Encoded:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding

    def isatty(self) -> bool:
        return False


# --- parsers -----------------------------------------------------------------


def test_empty_lines_do_not_end_or_crash(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = _concierge(tmp_path)
    assert concierge.handle_line("") is False
    assert concierge.handle_line("   \t") is False
    assert out == []
    assert concierge.last_status == "none"


def test_one_megabyte_line_is_bounded(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = _concierge(tmp_path)
    started = time.perf_counter()
    assert concierge.handle_line("a" * 1_000_000) is False
    assert time.perf_counter() - started < 2.0
    assert concierge.last_status == "no_endpoint"
    assert out
    assert "Traceback" not in "\n".join(out)


def test_nul_and_controls_in_slash_lines_stay_in_session(env: Path, tmp_path: Path) -> None:
    del env
    concierge, _out = _concierge(tmp_path)
    for line in ("/help\x00", "/use\tfx1", "/model\x1b[2J", "/\x00", "/sessions\r\n"):
        assert concierge.handle_line(line) is False
    assert concierge.handle_line("/model") is False
    assert concierge.active_model == "fx1"


def test_lone_surrogate_line_keeps_the_session(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = _concierge(tmp_path)
    assert concierge.handle_line("\ud800hello") is False
    assert concierge.last_status == "no_endpoint"
    assert all("\ud800" not in line for line in out)


def test_unclosed_quote_and_huge_session_limit(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del env
    concierge, _out = _concierge(tmp_path)
    assert concierge.handle_line('/use "fx1') is False
    assert concierge.handle_line("/sessions " + "9" * 5000) is False
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "usage:" in err
    assert concierge.active_model == "fx1"


def test_session_truncated_nan_duplicate_and_bad_utf8(env: Path, tmp_path: Path) -> None:
    del env
    store = SessionStore(tmp_path / "sessions")
    good = store.create(model="fx1")
    path = store.directory / f"{good['id']}.json"
    path.write_text('{"id": "aa", "id": "bb", "model": "fx1"}', encoding="utf-8")
    with pytest.raises(SessionError):
        store.load(good["id"])
    path.write_text('{"model": "fx1", "n": NaN}', encoding="utf-8")
    with pytest.raises(SessionError):
        store.load(good["id"])
    path.write_bytes(b'{"model": "fx1"')
    with pytest.raises(SessionError):
        store.load(good["id"])
    path.write_bytes(b"\xff\xfe{")
    with pytest.raises(SessionError):
        store.load(good["id"])
    other = store.create(model="fx1-lite")
    assert store.load(other["id"])["model"] == "fx1-lite"


def test_session_deep_json_and_huge_integer_are_corrupt(env: Path, tmp_path: Path) -> None:
    del env
    store = SessionStore(tmp_path / "sessions")
    record = store.create(model="fx1")
    path = store.directory / f"{record['id']}.json"
    path.write_text(_nested(_DEPTH), encoding="utf-8")
    with pytest.raises(SessionError):
        store.load(record["id"])
    path.write_text('{"n": ' + _huge_int() + "}", encoding="utf-8")
    with pytest.raises(SessionError):
        store.load(record["id"], quarantine=True)
    assert not path.exists()
    kept = store.create(model="fx1")
    assert store.load(kept["id"])["model"] == "fx1"


def test_profile_deep_json_is_quarantined(env: Path, tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(_nested(_DEPTH), encoding="utf-8")
    rows = profiles.list_models(path=path, environ={})
    assert {row["model"] for row in rows} == {"fx1", "fx1-lite"}
    assert list(path.parent.glob(path.name + ".corrupt-*"))
    health = profiles.store_health(path)
    assert any("corrupt" in item or "quarantine" in item for item in health)
    path.write_text('{"n": ' + _huge_int() + "}", encoding="utf-8")
    assert profiles.list_models(path=path, environ={})
    del env


def test_catalog_deep_json_and_huge_integer_are_unavailable() -> None:
    with pytest.raises(ContractError):
        CapabilityCatalog.from_json(_nested(_DEPTH))
    with pytest.raises(ContractError):
        CapabilityCatalog.from_json('{"n": ' + _huge_int() + "}")
    with pytest.raises(CatalogUnavailable):
        fetch_catalog(lambda _name, _args: (0, _nested(_DEPTH)))
    with pytest.raises(CatalogUnavailable):
        fetch_catalog(lambda _name, _args: (0, "{"))


def test_family_surface_hostile_json_is_a_client_error() -> None:
    client = FamilyClient(lambda _argv: (0, _nested(_DEPTH), ""))
    with pytest.raises(FamilyClientError):
        client.select("volatility")
    huge = FamilyClient(lambda _argv: (0, '{"n": ' + _huge_int() + "}", ""))
    with pytest.raises(FamilyClientError):
        huge.select("volatility")


def test_family_harness_payload_rejects_deep_json() -> None:
    class _Result:
        exit_code = 0
        stdout = _nested(_DEPTH)
        stderr = ""

    class _Harness:
        def run(self, name: str, args: list[str]) -> _Result:
            del name, args
            return _Result()

    with pytest.raises(RuntimeError, match="JSON"):
        _family_payload_from_harness(_Harness(), "volatility", 4)


def test_planner_hostile_json_falls_back() -> None:
    for payload in (_nested(_DEPTH), '{"steps": ' + _huge_int() + "}", "not-json", ""):
        found = _call_model(lambda _prompt, payload=payload: payload, "goal")
        assert isinstance(found, Exclusion)
        assert "Traceback" not in found.detail


def test_tool_arguments_type_confusion_and_hostile_json() -> None:
    samples: tuple[object, ...] = (
        {"name": "train", "args": ["--out-dir", "/tmp"]},
        ["train"],
        10**100,
        _nested(_DEPTH),
        _huge_int(),
        '{"a": NaN}',
        '{"a": Infinity}',
        None,
        "",
    )
    for raw in samples:
        name, arguments = parse_tool_call({"function": {"name": "harness_run", "arguments": raw}})
        assert name == "harness_run"
        if isinstance(raw, dict):
            assert arguments == raw
            assert arguments is not raw
        else:
            assert arguments == {}


def test_chat_refuses_a_tool_call_burst(env: Path, tmp_path: Path) -> None:
    del env
    calls = tuple(
        {"id": str(index), "type": "function", "function": {"name": "nope", "arguments": "{}"}}
        for index in range(40)
    )
    backend = _Burst(ToolCompletion(content=None, tool_calls=calls, finish_reason="tool_calls"))
    concierge, out = _concierge(tmp_path, backend)
    profiles.set_endpoint("fx1", _KEY, "http://127.0.0.1:9/v1")
    assert concierge.handle_line("run tools") is False
    assert backend.calls == 1
    assert concierge.last_status == "error"
    assert "too many tool calls" in "\n".join(out)
    assert [item for item in concierge.messages() if item.get("role") == "tool"] == []


def _model_error(
    send: Callable[[socket.socket], None], call: Callable[[ModelClient], None]
) -> ModelError:
    error: ModelError | None = None
    for _attempt in range(3):
        port, thread, sock = _serve(send)
        try:
            try:
                call(_client(port, timeout_s=2.0))
            except ModelError as exc:
                error = exc
                if exc.kind is not ErrorKind.NETWORK:
                    return exc
        finally:
            _finish(thread, sock)
    assert error is not None
    return error


def test_model_rejects_surrogate_content_and_ten_thousand_tools() -> None:
    body = json.dumps({"choices": [{"message": {"content": "\ud800"}}]}).encode()

    def send(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )

    failure = _model_error(
        send, lambda client: client.complete([{"role": "user", "content": "hi"}])
    )
    assert failure.kind is ErrorKind.MALFORMED
    assert _KEY not in str(failure)

    calls = [
        {"id": str(index), "type": "function", "function": {"name": "x", "arguments": "{}"}}
        for index in range(10_000)
    ]
    blob = json.dumps({"choices": [{"message": {"content": None, "tool_calls": calls}}]}).encode()

    def send_tools(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(blob)).encode()
            + b"\r\n\r\n"
            + blob
        )

    started = time.perf_counter()
    failure = _model_error(
        send_tools,
        lambda client: client.complete_with_tools([{"role": "user", "content": "hi"}]),
    )
    assert failure.kind is ErrorKind.MALFORMED
    assert "tool_calls" in failure.summary
    assert time.perf_counter() - started < 3.0


def test_model_duplicate_keys_nan_and_infinity() -> None:
    dup = b'{"choices":[{"message":{"content":"first","content":"second"}}]}'

    def send_dup(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(dup)).encode()
            + b"\r\n\r\n"
            + dup
        )

    port, thread, sock = _serve(send_dup)
    try:
        assert _client(port).complete([{"role": "user", "content": "hi"}]) == "second"
    finally:
        _finish(thread, sock)

    nan = b'{"choices":[{"message":{"content": NaN}}]}'

    def send_nan(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(nan)).encode()
            + b"\r\n\r\n"
            + nan
        )

    port, thread, sock = _serve(send_nan)
    try:
        with pytest.raises(ModelError) as ei:
            _client(port).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.MALFORMED
    finally:
        _finish(thread, sock)


def test_http09_and_garbage_are_malformed() -> None:
    def send_09(conn: socket.socket) -> None:
        conn.sendall(b'{"choices":[{"message":{"content":"nope"}}]}\n')

    port, thread, sock = _serve(send_09)
    try:
        with pytest.raises(ModelError) as ei:
            _client(port).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.MALFORMED
        assert "nope" not in str(ei.value)
    finally:
        _finish(thread, sock)

    def send_garbage(conn: socket.socket) -> None:
        conn.sendall(b"\x00\xff\xfe not-http\r\n\r\n")

    port, thread, sock = _serve(send_garbage)
    try:
        with pytest.raises(ModelError) as ei:
            _client(port).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.MALFORMED
    finally:
        _finish(thread, sock)


def test_content_length_short_lie_drops_the_tail() -> None:
    body = b'XXXX{"choices":[{"message":{"content":"secret-tail"}}]}'
    packet = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4\r\n\r\n" + body
    )

    def send(conn: socket.socket) -> None:
        conn.sendall(packet)

    port, thread, sock = _serve(send)
    try:
        with pytest.raises(ModelError) as ei:
            _client(port).complete([{"role": "user", "content": "hi"}])
        assert "secret-tail" not in str(ei.value)
        assert ei.value.kind is ErrorKind.MALFORMED
    finally:
        _finish(thread, sock)


def test_content_length_long_lie_closes_without_hanging() -> None:
    def send(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100000000\r\n\r\n{"
        )

    port, thread, sock = _serve(send)
    started = time.perf_counter()
    try:
        with pytest.raises(ModelError) as ei:
            _client(port, timeout_s=0.8).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind in {ErrorKind.MALFORMED, ErrorKind.NETWORK, ErrorKind.TIMEOUT}
        assert time.perf_counter() - started < 1.5
    finally:
        _finish(thread, sock)


def test_hundred_megabyte_body_is_capped() -> None:
    def send(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: 100000000\r\n\r\n"
        )
        chunk = b"x" * 65536
        for _ in range(1600):
            try:
                conn.sendall(chunk)
            except OSError:
                return

    port, thread, sock = _serve(send)
    started = time.perf_counter()
    try:
        with pytest.raises(ModelError) as ei:
            _client(port, timeout_s=2.0).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.MALFORMED
        assert "exceeded" in ei.value.summary
        assert time.perf_counter() - started < 3.0
    finally:
        _finish(thread, sock)


def test_redirect_loop_is_not_followed() -> None:
    accepted: list[int] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    sock.settimeout(0.4)
    port = sock.getsockname()[1]

    def run() -> None:
        while True:
            try:
                conn, _addr = sock.accept()
            except OSError:
                return
            accepted.append(1)
            try:
                conn.settimeout(0.4)
                _drain(conn)
                location = f"http://127.0.0.1:{port}/v1/chat/completions".encode()
                conn.sendall(
                    b"HTTP/1.1 302 Found\r\nLocation: "
                    + location
                    + b"\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                return
            finally:
                conn.close()

    thread = threading.Thread(target=run, name="rt6-redirect", daemon=True)
    thread.start()
    try:
        with pytest.raises(ModelError) as ei:
            _client(port, timeout_s=0.5).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.ROUTE_NOT_FOUND
        assert len(accepted) == 1
    finally:
        _finish(thread, sock)


def test_gzip_bomb_is_not_inflated() -> None:
    payload = gzip.compress(b'{"choices":[{"message":{"content":"inflated"}}]}' + b"\0" * 2_000_000)
    packet = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
    )

    def send(conn: socket.socket) -> None:
        conn.sendall(packet)

    port, thread, sock = _serve(send)
    started = time.perf_counter()
    try:
        with pytest.raises(ModelError) as ei:
            _client(port).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.MALFORMED
        assert "content-encoding" in ei.value.summary
        assert "inflated" not in str(ei.value)
        assert time.perf_counter() - started < 1.5
    finally:
        _finish(thread, sock)


def test_slow_chunk_times_out() -> None:
    def send(conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 80\r\n\r\n"
        )
        time.sleep(0.25)

    port, thread, sock = _serve(send)
    started = time.perf_counter()
    try:
        with pytest.raises(ModelError) as ei:
            _client(port, timeout_s=0.1).complete([{"role": "user", "content": "hi"}])
        assert ei.value.kind is ErrorKind.TIMEOUT
        assert time.perf_counter() - started < 1.2
    finally:
        _finish(thread, sock)


def test_exec_invalid_utf8_stdin_is_a_usage_error(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    del env

    class _Stdin:
        def isatty(self) -> bool:
            return False

        def read(self) -> str:
            raise _utf8_error()

    monkeypatch.setattr(sys, "stdin", _Stdin())
    with pytest.raises(SystemExit) as ei:
        main(["-"])
    assert ei.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "UTF-8" in captured.err


def test_exec_json_survives_a_lone_surrogate(capsys: pytest.CaptureFixture[str]) -> None:
    _write_json({"schema": "dipcatcher.exec.v1", "answer": "\ud800", "status": "ok"})
    out = capsys.readouterr().out
    out.encode("utf-8")
    parsed = json.loads(out)
    assert parsed["status"] == "ok"
    assert "\ud800" not in out


def test_last_message_file_rejects_a_lone_surrogate(tmp_path: Path) -> None:
    path = tmp_path / "last.txt"
    problem = _write_last_message(path, "\ud800")
    assert isinstance(problem, str)
    assert not path.exists()
    assert "\ud800" not in problem


def test_exec_surrogate_answer_does_not_crash(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_endpoint("fx1", _KEY, "http://127.0.0.1:9/v1")
    dest = tmp_path / "last.txt"
    outcome = run_exec(
        ExecRequest(prompt="hello", model="fx1", timeout_s=2.0, session="none", output_path=dest),
        backend_factory=lambda **_kwargs: _SurrogateBackend(),
        install_signals=False,
    )
    from dipcatcher_cli.exec_cmd import _emit

    _emit(outcome, json_mode=True)
    captured = capsys.readouterr()
    captured.out.encode("utf-8")
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    if dest.exists():
        dest.read_bytes().decode("utf-8")
    del env


def test_legacy_console_invalid_utf8_does_not_traceback(env: Path, tmp_path: Path) -> None:
    del env
    out: list[str] = []

    def read(_prompt: str) -> str:
        raise _utf8_error()

    run_console(input_fn=read, output_fn=out.append, engine=False, poll_timeout_s=0.05)
    text = "\n".join(out)
    assert "not valid UTF-8" in text
    assert "Traceback" not in text


def test_engine_console_invalid_utf8_ends(env: Path) -> None:
    del env
    out: list[str] = []
    state = {"n": 0}

    def read(_prompt: str) -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise _utf8_error()
        raise EOFError

    caps = TermCaps(False, False, False, False, False, 80, 24, False)
    run_console(
        input_fn=read,
        output_fn=out.append,
        engine=True,
        caps_fn=lambda: caps,
        poll_timeout_s=0.05,
    )
    assert "not valid UTF-8" in "\n".join(out)
    assert state["n"] == 1


@_FUZZ
@given(extra=st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=126), max_size=24))
def test_hypothesis_unknown_slash_does_not_crash(env: Path, tmp_path: Path, extra: str) -> None:
    del env
    concierge, _out = _concierge(tmp_path)
    assert concierge.handle_line("/zzfuzz " + extra) is False
    assert concierge.active_model == "fx1"


@_FUZZ
@given(
    raw=st.one_of(
        st.text(max_size=40),
        st.binary(max_size=40),
        st.integers(min_value=-(10**6), max_value=10**6),
        st.none(),
        st.lists(st.integers(min_value=0, max_value=9), max_size=3),
        st.dictionaries(st.text(max_size=4), st.integers(), max_size=2),
    )
)
def test_hypothesis_tool_arguments_do_not_raise(raw: object) -> None:
    name, arguments = parse_tool_call({"function": {"name": "harness_run", "arguments": raw}})
    assert name == "harness_run"
    assert isinstance(arguments, dict)


@_FUZZ
@given(data=st.binary(max_size=48))
def test_hypothesis_json_bytes_are_corrupt_or_finite(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "blob.json"
    path.write_bytes(data)
    try:
        value = read_json(path)
    except StoreCorruptError:
        return
    json.dumps(value, allow_nan=False)


@_FUZZ
@given(text=st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=160), max_size=20))
def test_hypothesis_paths_do_not_crash(tmp_path: Path, text: str) -> None:
    try:
        found = contain_path(text, [tmp_path])
    except PolicyViolation:
        return
    assert found == tmp_path or tmp_path in found.parents


# --- cross-platform branches (Linux host; not a Windows/macOS run) ----------


def test_windows_spawn_uses_new_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    captured: dict[str, object] = {}

    class _Done:
        pid = 77
        returncode = 0
        stdout = io.BytesIO(b"")
        stderr = io.BytesIO(b"")

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def popen(argv: list[str], **kwargs: object) -> _Done:
        del argv
        captured.update(kwargs)
        return _Done()

    result = run_owned(
        ["tool"],
        ProcLimits(timeout_s=1.0, kill_grace_s=0.0, cancel_poll_s=0.01),
        env={"PATH": "C:\\Windows"},
        popen=popen,
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0
    assert captured.get("creationflags") == getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
    assert "start_new_session" not in captured


def test_windows_tree_kill_is_taskkill_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(list(argv))
        assert kwargs.get("shell") is not True
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("fx1.interactive.procs.subprocess.run", fake_run)

    class _Proc:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    _apply_kill(_Proc(), None, 0.0, None, "win32")  # type: ignore[arg-type]
    assert seen == [["taskkill", "/PID", "4321", "/T", "/F"]]
    seen.clear()
    _taskkill(os.getpid())
    _taskkill(0)
    _taskkill(-3)
    assert seen == []


def test_windows_lock_records_unlock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class _Msvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 0

        def locking(self, fd: int, mode: int, nbytes: int) -> None:
            del fd, nbytes
            calls.append(mode)

    monkeypatch.setitem(sys.modules, "msvcrt", _Msvcrt())
    monkeypatch.setattr(sys, "platform", "win32")
    from fx1.interactive.atomicstore import FileLock

    with FileLock(tmp_path / "ledger.json", timeout_s=0.2, poll_s=0.01):
        assert calls[-1] == _Msvcrt.LK_NBLCK
    assert calls[-1] == _Msvcrt.LK_UNLCK

    class _Busy(_Msvcrt):
        def locking(self, fd: int, mode: int, nbytes: int) -> None:
            del fd, mode, nbytes
            raise OSError(13, "locked")

    monkeypatch.setitem(sys.modules, "msvcrt", _Busy())
    with (
        pytest.raises(StoreLockTimeout),
        FileLock(tmp_path / "other.json", timeout_s=0.05, poll_s=0.01),
    ):
        pass


def test_non_posix_skips_dir_fsync_and_permission_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``os.name == nt`` cannot run ``atomic_write_bytes`` here.

    pathlib refuses to construct ``WindowsPath`` on Linux, so the fchmod skip
    inside the write is not executed. The early returns are.
    """
    target = tmp_path / "note.txt"
    target.write_text("x", encoding="utf-8")
    opened: list[int] = []
    real_open = os.open

    def track_open(path: object, flags: int, mode: int = 0o777) -> int:
        opened.append(flags)
        return real_open(path, flags, mode)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "open", track_open)
    from fx1.interactive.atomicstore import _fsync_dir

    _fsync_dir(tmp_path)
    assert opened == []
    assert permission_problem(target) is None


def test_directory_fsync_error_keeps_the_new_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync

    def boom(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", boom)
    path = tmp_path / "kept.txt"
    atomic_write_text(path, "kept")
    assert path.read_text(encoding="utf-8") == "kept"


def test_protect_store_windows_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "credentials.json"
    path.write_text("{}\n", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    assert protect_store_windows(path, runner=runner) is False
    assert calls == []
    monkeypatch.setattr(sys, "platform", "win32")
    assert protect_store_windows(path, runner=runner, username="") is False
    assert calls == []
    assert protect_store_windows(path, runner=runner, username="ada") is True
    assert calls[0][0] == "icacls"
    assert calls[0][-1] == "ada:F"

    def broken(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del argv, kwargs
        raise OSError("icacls missing")

    assert protect_store_windows(path, runner=broken, username="ada") is False
    health = profiles.store_health(path)
    assert any("Windows ACL" in item for item in health)


def test_windows_reader_does_not_arm_readline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    caps = TermCaps(True, True, True, False, False, 80, 24, True)
    writer = ConsoleWriter(caps, out=io.StringIO(), err=io.StringIO())
    reader = LineReader(caps, writer, input_fn=lambda _prompt: "hi")
    reader._arm_readline()
    assert reader._rl_ready is False
    assert _can_use_tty(caps) is False
    assert _scoop_partial_text(io.StringIO()) == ""
    assert reader.read_line("> ") == "hi"
    reader.close()


def test_console_windows_branches_skip_posix_polling(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    seen: list[str] = []

    def read(prompt: str) -> str:
        seen.append(prompt)
        return "ok"

    assert _tty_polling_read(read, ">", 0.2) == "ok"
    assert seen == [">"]
    caps = TermCaps(True, True, True, False, False, 80, 24, True)
    assert _polls(caps, input) is False
    lines: list[str] = []

    class _Host:
        model = "fx1"

    _emit_engine_banner(_Host(), caps, lines.append)
    assert _WINDOWS_NOTICE in lines
    del env


def test_cp1252_and_pythonioencoding(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    cp1252 = _Encoded("cp1252")
    utf8 = _Encoded("utf-8")

    def size() -> os.terminal_size:
        return os.terminal_size((100, 40))

    windows = detect_caps(
        stdin=cp1252,
        stdout=cp1252,
        environ={"PYTHONUTF8": "1", "TERM": "dumb"},
        platform="win32",
        size_fn=size,
    )
    assert windows.windows is True
    assert windows.unicode is False
    forced = detect_caps(
        stdin=cp1252,
        stdout=cp1252,
        environ={"PYTHONIOENCODING": "utf-8:replace", "TERM": "dumb"},
        platform="win32",
        size_fn=size,
    )
    assert forced.unicode is True
    darwin = detect_caps(
        stdin=utf8,
        stdout=utf8,
        environ={"TERM": "xterm"},
        platform="darwin",
        size_fn=size,
    )
    assert darwin.windows is False
    assert darwin.unicode is True


def test_exec_signals_skip_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(sig: int, handler: object) -> object:
        del sig, handler
        raise ValueError("unsupported signal")

    monkeypatch.setattr(signal_module(), "signal", boom)
    assert install_exec_signals(lambda _sig, _frame: None) == []
    monkeypatch.setattr(signal_module(), "SIGTERM", None, raising=False)
    assert _sigterm() == 15
    outcome = run_exec(
        ExecRequest(prompt="/exit", model="fx1", timeout_s=1.0, session="none"),
        install_signals=True,
    )
    assert outcome.exit_code == 0
    assert outcome.status == "ok"


def test_process_probe_windows_and_darwin() -> None:
    windows_calls: list[list[str]] = []

    def tasklist(argv: list[str]) -> tuple[int, str]:
        windows_calls.append(list(argv))
        if windows_calls and "99" in " ".join(argv):
            return 0, '"python.exe","99","Console","1"\n'
        return 0, "INFO: No tasks are running matching the specified criteria.\n"

    windows = SystemProcessProbe(platform="win32", runner=tasklist)
    assert windows.is_alive(4) is False
    assert windows.is_alive(99) is True
    assert windows.token(99) is None
    assert windows_calls[0][0] == "tasklist"
    darwin_calls: list[list[str]] = []

    def ps(argv: list[str]) -> tuple[int, str]:
        darwin_calls.append(list(argv))
        if "lstart=" in argv:
            return 0, "Mon Oct 12 00:00:00 2026\n"
        return 0, "55\n"

    darwin = SystemProcessProbe(platform="darwin", runner=ps)
    assert darwin.is_alive(55) is True
    assert darwin.token(55) == "Mon Oct 12 00:00:00 2026"
    assert darwin_calls[0][0] == "ps"


def test_terminate_process_is_false_on_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        del sig
        killed.append(pid)

    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(sys, "platform", "win32")
    assert terminate_process(1234, 9) is False
    assert killed == []


def test_windows_path_syntax_is_rejected_on_linux(tmp_path: Path) -> None:
    samples = (
        r"C:foo",
        r"C:\Windows\system32",
        r"\\server\share\x",
        r"\\?\C:\Windows",
        r"\\.\pipe\evil",
        "a\x00b",
        "ok/../../etc/passwd",
    )
    for text in samples:
        with pytest.raises(PolicyViolation):
            contain_path(text, [tmp_path])
    inside = tmp_path / "sub"
    inside.mkdir()
    found = contain_path("sub\\file", [tmp_path])
    assert found.parent == inside or str(found).endswith("file")


def test_posix_replace_while_open_for_read(tmp_path: Path) -> None:
    path = tmp_path / "open.txt"
    atomic_write_text(path, "one")
    with path.open(encoding="utf-8") as handle:
        atomic_write_text(path, "two")
        assert handle.read() in {"one", "two", ""}
    assert path.read_text(encoding="utf-8") == "two"


def signal_module() -> Any:
    import signal

    return signal
