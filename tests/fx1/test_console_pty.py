"""Pseudo-terminal tests for the interactive console. POSIX only."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import pytest

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX pty")
_SRC = str(Path(__file__).resolve().parents[2] / "src")
_PROMPT = "fx1 › ".encode()

_DRIVER = r"""
from __future__ import annotations

import sys
import termios
import time
from typing import Any

import typer

from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.console import run_console
from fx1.serve.backends import ToolCompletion


def _completion(text: str) -> ToolCompletion:
    return ToolCompletion(content=text, tool_calls=None, finish_reason="stop")


class Fast:
    def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> ToolCompletion:
        del messages, kwargs
        return _completion("noted")


class Slow:
    def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> ToolCompletion:
        del messages, kwargs
        sys.stdout.write("THINKING\n")
        sys.stdout.flush()
        time.sleep(8)
        return _completion("noted")


def _factory(mode: str):
    backend = Slow if mode == "slow" else Fast

    def build(**kwargs: Any) -> Fast | Slow:
        del kwargs
        return backend()

    return build


class Scripted(Concierge):
    def __init__(self, mode: str) -> None:
        super().__init__(
            backend_factory=_factory(mode),
            animated=False,
            approvals=ApprovalGate(),
            session_store=None,
            output_fn=typer.echo,
        )
        self.mode = mode
        self._ticks = 0
        self._job_on = False

    def startup_checks(self) -> list[str]:
        return ["startup-ok"]

    @property
    def job_running(self) -> bool:
        if self.mode == "job":
            return self._job_on
        return bool(super().job_running)

    def handle_line(self, line: str) -> bool:
        if self.mode in {"round", "slow"}:
            return bool(super().handle_line(line))
        return _echo(self, line)

    def tick(self) -> None:
        if self.mode == "job":
            _tick_job(self)
            return
        super().tick()


def _echo(owner: Scripted, line: str) -> bool:
    text = line.strip()
    if text in {"/exit", "exit", "quit"}:
        owner.shutdown()
        return True
    if text == "start":
        owner._job_on = True
        owner._say("started")
        return False
    owner._say("GOT:" + line.replace("\n", "|"))
    return False


def _tick_job(owner: Scripted) -> None:
    if not owner._job_on:
        return
    owner._ticks += 1
    owner._say("[job] working: step " + str(owner._ticks))
    if owner._ticks >= 8:
        owner._job_on = False


def _prepare(mode: str) -> None:
    if mode not in {"round", "slow"}:
        return
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "sk-test-secret-value", "https://example.test/v1")


def _termios(before: list[object]) -> None:
    try:
        after = termios.tcgetattr(sys.stdin.fileno())
        mark = "OK" if list(before) == list(after) else "BAD"
    except termios.error:
        mark = "BAD"
    sys.stdout.write("TERMIOS_" + mark + "\nDONE\n")
    sys.stdout.flush()


def main() -> None:
    mode = sys.argv[1]
    before = list(termios.tcgetattr(sys.stdin.fileno()))
    code = 0
    try:
        _prepare(mode)
        run_console(concierge_factory=lambda: Scripted(mode), poll_timeout_s=0.05)
    except KeyboardInterrupt:
        code = 0
    except Exception as exc:
        sys.stdout.write("ERR:" + type(exc).__name__ + ":" + str(exc) + "\n")
        sys.stdout.flush()
        code = 1
    finally:
        _termios(before)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
"""


def _child_env(root: Path, extra: dict[str, str] | None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root / "home"),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": _SRC,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "INPUTRC": "/dev/null",
        "FX1_CONFIG_DIR": str(root / "cfg"),
        "FX1_SESSION_DIR": str(root / "sessions"),
        "DIPCATCHER_LOG": "error",
    }
    if extra:
        env.update(extra)
    return env


def _kill_child(pid: int) -> None:
    import signal

    if pid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("pty write failed")
        view = view[count:]


def _read_until(fd: int, marker: bytes, deadline: float, buf: bytearray, start: int) -> int:
    """Block until ``marker`` appears at or after ``start``. Return the end index."""
    import select

    while time.monotonic() < deadline:
        found = buf.find(marker, start)
        if found >= 0:
            return found + len(marker)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], min(0.2, remaining))
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        buf.extend(chunk)
    raise AssertionError(f"missing {marker!r} in {bytes(buf)!r}")


def _wait_status(pid: int, deadline: float) -> int:
    import select

    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return os.waitstatus_to_exitcode(status)
        select.select([], [], [], min(0.05, max(0.0, deadline - time.monotonic())))
    raise AssertionError("child still running")


def _pty(
    tmp_path: Path,
    mode: str,
    steps: list[tuple[bytes, bytes | None]],
    *,
    extra: dict[str, str] | None = None,
    history: str | None = None,
) -> tuple[str, int]:
    import pty

    root = tmp_path / mode
    (root / "cfg").mkdir(parents=True)
    (root / "home").mkdir()
    if history is not None:
        (root / "cfg" / "history").write_text(history, encoding="utf-8")
    script = root / "driver.py"
    script.write_text(_DRIVER, encoding="utf-8")
    pid, fd = pty.fork()
    if pid == 0:
        try:
            if fd > 2:
                os.close(fd)
            os.execve(
                sys.executable,
                [sys.executable, str(script), mode],
                _child_env(root, extra),
            )
        finally:
            os._exit(127)
    deadline = time.monotonic() + 2.6
    buf = bytearray()
    cursor = 0
    status: int | None = None
    try:
        for marker, send in steps:
            cursor = _read_until(fd, marker, deadline, buf, cursor)
            if send is not None:
                _write_all(fd, send)
        status = _wait_status(pid, deadline)
        return bytes(buf).decode("utf-8", errors="replace"), status
    finally:
        _kill_child(pid)
        os.close(fd)


@posix_only
def test_pty_prompt_and_round_trip(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "round",
        [(_PROMPT, b"hello\n"), (b"noted", None), (_PROMPT, b"\x04"), (b"DONE", None)],
    )
    assert "startup-ok" in text
    assert "noted" in text
    assert "sk-test-secret-value" not in text
    assert "TERMIOS_OK" in text
    assert "ERR:" not in text
    assert status == 0
    history = (tmp_path / "round" / "cfg" / "history").read_text(encoding="utf-8")
    assert "hello" in history
    assert "sk-test-secret-value" not in history


@posix_only
def test_pty_up_arrow_recalls_history(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "history",
        [(_PROMPT, b"\x1b[A\n"), (b"GOT:previous", None), (_PROMPT, b"\x04"), (b"DONE", None)],
        history='["previous"]\n',
    )
    assert "GOT:previous" in text
    assert "TERMIOS_OK" in text
    assert status == 0


@posix_only
def test_pty_multiline_continuation(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "multiline",
        [
            (_PROMPT, b"one\\\ntwo\n"),
            (b"GOT:one|two", None),
            (_PROMPT, b"/exit\n"),
            (b"DONE", None),
        ],
    )
    assert "GOT:one|two" in text
    assert "TERMIOS_OK" in text
    assert status == 0


@posix_only
def test_pty_double_ctrl_c_exits_clean(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "interrupt",
        [
            (_PROMPT, b"\x03"),
            (b"press Ctrl-C again to exit", None),
            (_PROMPT, b"\x03"),
            (b"DONE", None),
        ],
    )
    assert "press Ctrl-C again to exit" in text
    assert "ERR:" not in text
    assert "TERMIOS_OK" in text
    assert status == 0


@posix_only
def test_pty_ctrl_d_exits(tmp_path: Path) -> None:
    text, status = _pty(tmp_path, "eof", [(_PROMPT, b"\x04"), (b"DONE", None)])
    assert "DONE" in text
    assert "ERR:" not in text
    assert "TERMIOS_OK" in text
    assert status == 0


@posix_only
def test_pty_no_color_has_no_escape(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "nocolor",
        [(_PROMPT, b"\x04"), (b"DONE", None)],
        extra={"NO_COLOR": "1", "FORCE_COLOR": "1"},
    )
    # LineReader enables GNU readline bracketed paste. Those two toggles are
    # not color; NO_COLOR still removes every other escape, including SGR.
    plain = text.replace("\x1b[?2004h", "").replace("\x1b[?2004l", "")
    assert "dipcatcher · fx1 @" in plain
    assert "\x1b" not in plain
    assert "TERMIOS_OK" in text
    assert status == 0


@posix_only
def test_pty_job_progress_keeps_the_next_line(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "job",
        [
            (_PROMPT, b"start\n"),
            (b"working: step", None),
            (b"job ", b"next\n"),
            (b"GOT:next", None),
            (_PROMPT, b"/exit\n"),
            (b"DONE", None),
        ],
    )
    assert "working: step" in text
    assert "GOT:next" in text
    assert text.rfind("fx1 › ") > text.rfind("GOT:next")
    assert "ERR:" not in text
    assert status == 0


@posix_only
def test_pty_ctrl_c_during_a_turn_cancels(tmp_path: Path) -> None:
    text, status = _pty(
        tmp_path,
        "slow",
        [
            (_PROMPT, b"hello\n"),
            (b"THINKING", b"\x03"),
            (b"turn cancelled", None),
            (_PROMPT, b"/exit\n"),
            (b"DONE", None),
        ],
    )
    assert text.count("turn cancelled") == 1
    assert "sk-test-secret-value" not in text
    assert "ERR:" not in text
    assert "TERMIOS_OK" in text
    assert status == 0
