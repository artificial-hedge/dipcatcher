"""Windows and non-tty branches, exercised with fakes. This host is Linux."""

from __future__ import annotations

import io
import os
import socket
import sys
from pathlib import Path

import pytest

from fx1.interactive.console import _history_file, _prepare_stdio, _text_banner
from fx1.interactive.orb import banner, orbs_enabled
from fx1.interactive.shell import _prompt_glyphs
from fx1.interactive.terminal import (
    ConsoleWriter,
    LineReader,
    TermCaps,
    detect_caps,
)
from fx1.interactive.tty_guard import cbreak_available
from tests.fx1.ux_pty import chat_env, forbid_public_connect, hermetic_network


class _Reconfigurable:
    encoding = "cp1252"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)

    def isatty(self) -> bool:
        return True


def _caps(**overrides: object) -> TermCaps:
    values: dict[str, object] = {
        "stdin_tty": True,
        "stdout_tty": True,
        "interactive": True,
        "color": False,
        "unicode": True,
        "width": 80,
        "height": 24,
        "windows": False,
    }
    values.update(overrides)
    return TermCaps(**values)  # type: ignore[arg-type]


def test_ux_windows_reconfigures_stdio_to_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = [_Reconfigurable(), _Reconfigurable(), _Reconfigurable()]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdin", streams[0])
    monkeypatch.setattr(sys, "stdout", streams[1])
    monkeypatch.setattr(sys, "stderr", streams[2])
    _prepare_stdio()
    assert streams[0].calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert streams[1].calls == streams[0].calls
    assert streams[2].calls == streams[0].calls


def test_ux_prepare_stdio_is_a_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _Reconfigurable()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "stdout", stream)
    _prepare_stdio()
    assert stream.calls == []


def test_ux_nt_reader_uses_blocking_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "platform", "win32")
    stream = io.StringIO("hello\n")
    out = io.StringIO()
    caps = _caps(windows=True, interactive=False, stdin_tty=False, stdout_tty=False)
    writer = ConsoleWriter(caps, out=out)
    reader = LineReader(caps, writer, stdin=stream, history_path=tmp_path / "history")  # type: ignore[arg-type]
    assert cbreak_available(stream) is False  # type: ignore[arg-type]
    assert reader.read_line("> ") == "hello"
    assert "hello" in out.getvalue() or out.getvalue() == "" or "> " in out.getvalue()


def test_ux_no_termios_disables_cbreak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_import = __import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "termios":
            raise ImportError("no termios")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", blocked)
    handle = (tmp_path / "not-a-tty").open("w+")
    try:
        assert cbreak_available(handle) is False
    finally:
        handle.close()


def test_ux_history_drops_keys_lines_and_caps(tmp_path: Path) -> None:
    out = io.StringIO()
    caps = _caps(interactive=False, stdin_tty=False, stdout_tty=False)
    writer = ConsoleWriter(caps, out=out)
    path = tmp_path / "history"
    lines = iter(
        [
            "hello\n/keys set fx1 sk-history-must-not-leak\nkept",
            "second",
            "third",
        ]
    )
    reader = LineReader(
        caps,
        writer,
        stdin=io.StringIO(""),
        history_path=path,
        history_max=2,
        input_fn=lambda _prompt: next(lines),
    )
    assert "/keys set" in reader.read_line(">")
    assert reader.history() == ("hello\nkept",)
    reader.read_line(">")
    reader.read_line(">")
    assert reader.history() == ("second", "third")
    reader.save_history()
    body = path.read_text(encoding="utf-8")
    assert "/keys" not in body
    assert "sk-history-must-not-leak" not in body
    assert "third" in body
    assert stat_mode(path) == 0o600


def test_ux_history_path_uses_pathlib(tmp_path: Path) -> None:
    target = _history_file(Path("C:/fx1/cfg") / "history")
    assert target.name == "history"
    assert _history_file(tmp_path / "mine").name == "mine"


def test_ux_ascii_banner_and_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fx1.interactive.terminal.unicode_stdout",
        lambda stream=None: False,
    )
    monkeypatch.setattr("fx1.interactive.orb.unicode_stdout", lambda stream=None: False)
    text = banner("fx1", "example.test")
    assert "╭" not in text
    assert "dipcatcher |" in text
    assert _text_banner("fx1", "h", "unset", unicode=False).isascii()
    sep, mark = _prompt_glyphs()
    assert sep == " | "
    assert mark == " > "


def test_ux_status_above_keeps_the_composer() -> None:
    out = io.StringIO()
    writer = ConsoleWriter(_caps(), out=out)
    writer.set_composer_owned(True)
    writer.set_prompt_visible("fx1 > ", lambda: "QQ")
    writer.write_prompt("fx1 > ")
    writer.status("research 1.0s")
    writer.status("research 1.5s")
    text = out.getvalue()
    assert "research 1.5s" in text
    assert text.rfind("QQ") > text.find("research 1.5s")


def test_ux_orb_and_color_off_without_a_tty() -> None:
    stream = io.StringIO()
    assert orbs_enabled(stream) is False  # type: ignore[arg-type]
    caps = detect_caps(
        stdin=stream,  # type: ignore[arg-type]
        stdout=stream,  # type: ignore[arg-type]
        environ={"TERM": "dumb", "NO_COLOR": "1", "FORCE_COLOR": "1"},
        size_fn=lambda: os.terminal_size((20, 5)),
        platform="win32",
    )
    assert caps.color is False
    assert caps.interactive is False
    assert caps.windows is True
    assert caps.width == 40
    assert caps.height == 5


def stat_mode(path: Path) -> int:
    import stat

    return stat.S_IMODE(path.stat().st_mode)


def test_ux_writer_drops_hostile_controls_and_keeps_sgr() -> None:
    out = io.StringIO()
    writer = ConsoleWriter(_caps(color=True, interactive=True), out=out)
    writer.line("hi\x1b]0;t\x07\x1b[31mRED\x1b[0m\x1b[2J\x9bH")
    text = out.getvalue()
    assert "\x1b]0;" not in text
    assert "\x1b[2J" not in text
    assert "\x9b" not in text
    assert "\x1b[31mRED" in text
    writer.status("go")
    assert "\x1b[2K" in out.getvalue()


def test_ux_hermetic_env_and_socket_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = chat_env(tmp_path)
    for key, value in hermetic_network().items():
        assert env[key] == value
        assert "moonshot" not in value
    with pytest.raises(AssertionError, match="non-loopback"):
        forbid_public_connect(("api.moonshot.ai", 443))
    real = socket.socket.connect

    def wrapped(self: socket.socket, address: object) -> None:
        forbid_public_connect(address)
        real(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", wrapped)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.5)
