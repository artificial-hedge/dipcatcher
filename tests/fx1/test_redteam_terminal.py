"""Terminal-control injection from model text, tool output, and exports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from fx1.interactive.sessions import SessionStore
from fx1.interactive.terminal import sanitize_for_terminal
from tests.fx1.redteam_support import completion, isolate, joined, open_chat, tool
from tests.fx1.test_concierge import FakeHarnessResult

_ATTACK = (
    "SAFE\x1b]52;c;CLIPBOARD\x07"
    "\x1b]8;;http://evil.example\x07link\x1b]8;;\x07"
    "\x1b]0;owned\x07"
    "\x1b[201~\x1b[2J\x1b[H"
    "\x1bPsecret\x1b\\"
    "\x9b2J\x9d0;title\x07"
    "\u202eBIDI\u200bZW"
    "TAIL"
)


def _controls_gone(text: str) -> None:
    for marker in ("\x1b", "\x9b", "\x9d", "\u202e", "\u200b", "CLIPBOARD", "owned"):
        assert marker not in text
    assert "SAFE" in text
    assert "TAIL" in text


def test_sanitizer_strips_osc_csi_c1_bidi_and_zero_width() -> None:
    cleaned = sanitize_for_terminal(_ATTACK)
    _controls_gone(cleaned)
    assert sanitize_for_terminal(cleaned) == cleaned
    assert sanitize_for_terminal("keep\n\tme") == "keep\n\tme"
    colored = sanitize_for_terminal("\x1b[31mred\x1b[0m" + _ATTACK, keep_sgr=True)
    assert colored.startswith("\x1b[31mred\x1b[0m")
    assert "CLIPBOARD" not in colored
    assert "\u200b" not in colored


def test_model_answer_reaches_the_operator_without_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    script = [completion(_ATTACK, None)]
    concierge, out, _harness = open_chat(tmp_path, script)
    concierge.handle_line("hello")
    _controls_gone(joined(out))
    stored = joined(str(message.get("content", "")) for message in concierge.messages())
    _controls_gone(stored)


def test_harness_stdout_and_export_are_neutralized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    from fx1.interactive import profiles

    class Loud:
        def run(self, name: str, extra_args: list[str] | None = None, **_kwargs: object) -> object:
            del name, extra_args
            return FakeHarnessResult(0, _ATTACK, "err\x1b]52;c;CLIPBOARD\x07")

    profiles.set_endpoint("fx1", "test-key-value", "https://models.example.test/v1")
    store = SessionStore(tmp_path / "sessions")
    script = [
        completion(None, (tool("d", "harness_run", {"name": "doctor", "args": []}),)),
        completion("plain close", None),
    ]
    concierge, out, _harness = open_chat(tmp_path, script, harness=Loud())
    concierge._session_store = store  # noqa: SLF001
    concierge.handle_line("check")
    shown = joined(out)
    _controls_gone(shown)
    assert "\x1b" not in shown
    concierge.handle_line("/export md")
    blobs = [
        path.read_text(encoding="utf-8") for path in store.directory.rglob("*") if path.is_file()
    ]
    blobs.extend(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "fx1-config").rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    )
    blob = "\n".join(blobs)
    assert "CLIPBOARD" not in blob
    assert "\x1b]52" not in blob
    assert "\u202e" not in blob


@pytest.mark.skipif(os.name != "posix", reason="POSIX pty")
def test_pty_model_answer_does_not_emit_attack_sequences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del monkeypatch
    text = _pty_attack(tmp_path)
    plain = text.replace("\x1b[?2004h", "").replace("\x1b[?2004l", "")
    assert "SAFE" in plain
    assert "TAIL" in plain
    assert "CLIPBOARD" not in plain
    assert "\x1b]52" not in plain
    assert "\x1b[201~" not in plain
    assert "\x1b[2J" not in plain
    assert "\x9b" not in plain
    assert "\u202e" not in plain
    assert "\u200b" not in plain
    assert "ERR:" not in plain


def _pty_attack(tmp_path: Path) -> str:
    import pty

    root = tmp_path / "pty"
    (root / "cfg").mkdir(parents=True)
    (root / "home").mkdir()
    script = root / "driver.py"
    script.write_text(_DRIVER, encoding="utf-8")
    pid, fd = pty.fork()
    if pid == 0:
        try:
            if fd > 2:
                os.close(fd)
            os.chdir(root)
            os.execve(sys.executable, [sys.executable, str(script)], _child_env(root))
        finally:
            os._exit(127)
    try:
        return _read_child(pid, fd)
    finally:
        _kill_child(pid)
        os.close(fd)


def _child_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(root / "home")
    env["USERPROFILE"] = str(root / "home")
    env["FX1_CONFIG_DIR"] = str(root / "cfg")
    env["FX1_SESSION_DIR"] = str(root / "sessions")
    env["NO_COLOR"] = "1"
    env["TERM"] = "xterm"
    env.pop("MOONSHOT_API_KEY", None)
    env.pop("FX1_BASE_URL", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["RETEAM_ROOT"] = str(root)
    return env


def _read_child(pid: int, fd: int) -> str:
    import select
    import time

    deadline = time.monotonic() + 60.0
    buf = bytearray()
    prompt = "fx1 › ".encode()
    sent_hello = False
    sent_exit = False
    while time.monotonic() < deadline:
        if prompt in buf and not sent_hello:
            os.write(fd, b"hello\n")
            sent_hello = True
        tail_at = buf.find(b"TAIL")
        if tail_at >= 0 and prompt in buf[tail_at:] and not sent_exit:
            os.write(fd, b"/exit\n")
            sent_exit = True
        if b"DONE" in buf:
            return bytes(buf).decode("utf-8", errors="replace")
        remaining = deadline - time.monotonic()
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
    raise AssertionError(f"pty did not finish: {bytes(buf)!r}")


def _kill_child(pid: int) -> None:
    import contextlib
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


_DRIVER = r"""
from __future__ import annotations

import os
import sys

import typer

from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.console import run_console
from fx1.serve.backends import ToolCompletion

_ATTACK = (
    "SAFE\x1b]52;c;CLIPBOARD\x07"
    "\x1b]8;;http://evil.example\x07link\x1b]8;;\x07"
    "\x1b]0;owned\x07"
    "\x1b[201~\x1b[2J\x1b[H"
    "\x1bPsecret\x1b\\"
    "\x9b2J\x9d0;title\x07"
    "\u202eBIDI\u200bZW"
    "TAIL"
)


class Hostile:
    def complete_with_tools(self, messages: list, **kwargs: object) -> ToolCompletion:
        del messages, kwargs
        return ToolCompletion(content=_ATTACK, tool_calls=None, finish_reason="stop")


class Scripted(Concierge):
    def __init__(self) -> None:
        super().__init__(
            backend_factory=lambda **_kwargs: Hostile(),
            approvals=ApprovalGate(),
            animated=False,
            session_store=None,
            output_fn=typer.echo,
        )

    def startup_checks(self) -> list[str]:
        return ["startup-ok"]


def main() -> None:
    os.chdir(os.environ["RETEAM_ROOT"])
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "sk-" + ("b" * 24), "https://models.example.test/v1")
    run_console(concierge_factory=Scripted)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"ERR:{type(exc).__name__}\n")
        raise
    sys.stdout.write("DONE\n")
    sys.stdout.flush()
"""
