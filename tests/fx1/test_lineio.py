"""Offline tests for the dipcatcher terminal line reader and input history."""

from __future__ import annotations

import builtins
import io
import os
import stat
import sys
from pathlib import Path

import pytest

from fx1.interactive import lineio
from fx1.interactive.lineio import (
    History,
    InterruptAction,
    InterruptPolicy,
    LineReader,
    history_path,
    shorten,
    supports_color,
    terminal_width,
    wrap_block,
)

_CANCEL_HINT = "cancel requested — press Ctrl-C again within 2s to exit"
_CLEAR_HINT = "press Ctrl-C again within 2s to exit, or type /exit"


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TTYStream(io.StringIO):
    def __init__(self, initial: str = "", *, tty: bool = True) -> None:
        super().__init__(initial)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class BrokenStream(io.StringIO):
    def isatty(self) -> bool:
        raise OSError("not a tty")


class KeyScript:
    """Scripted Windows console: a string is a key, None is an idle poll."""

    def __init__(self, events: list[str | None]) -> None:
        self.events = list(events)

    def kbhit(self) -> bool:
        if self.events and self.events[0] is None:
            self.events.pop(0)
            return False
        return bool(self.events)

    def getwch(self) -> str:
        if not self.events or self.events[0] is None:
            raise AssertionError("getwch called without a pending key")
        key = self.events.pop(0)
        assert key is not None
        return key


def _refuse_input(prompt: str) -> str:
    raise AssertionError(f"blocking input used for {prompt!r}")


def _windows_reader(
    events: list[str | None],
    *,
    clock: ManualClock | None = None,
) -> tuple[LineReader, TTYStream, list[float]]:
    script = KeyScript(events)
    stdout = TTYStream()
    manual = clock or ManualClock()
    slept: list[float] = []

    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        manual.advance(1.0)

    reader = LineReader(
        stdin=TTYStream(),
        stdout=stdout,
        platform="win32",
        input_fn=_refuse_input,
        kbhit=script.kbhit,
        getwch=script.getwch,
        clock=manual,
        sleep=_sleep,
    )
    return reader, stdout, slept


# ---------------------------------------------------------------------------
# InterruptPolicy
# ---------------------------------------------------------------------------


def test_interrupt_idle_then_exit_within_window() -> None:
    clock = ManualClock()
    policy = InterruptPolicy(clock=clock)
    assert policy.on_interrupt(job_running=False) is InterruptAction.CLEAR_LINE
    clock.now = 1.0
    assert policy.on_interrupt(job_running=False) is InterruptAction.EXIT


def test_interrupt_running_job_cancels_then_exits() -> None:
    clock = ManualClock()
    policy = InterruptPolicy(clock=clock)
    assert policy.on_interrupt(job_running=True) is InterruptAction.CANCEL_JOB
    clock.now = 2.0
    assert policy.on_interrupt(job_running=True) is InterruptAction.EXIT


def test_interrupt_after_window_is_not_exit_and_rearms() -> None:
    clock = ManualClock()
    policy = InterruptPolicy(exit_window_s=2.0, clock=clock)
    assert policy.on_interrupt(job_running=True) is InterruptAction.CANCEL_JOB
    clock.now = 2.01
    assert policy.on_interrupt(job_running=False) is InterruptAction.CLEAR_LINE
    clock.now = 3.0
    assert policy.on_interrupt(job_running=False) is InterruptAction.EXIT


def test_interrupt_reset_clears_armed_state() -> None:
    clock = ManualClock()
    policy = InterruptPolicy(clock=clock)
    assert policy.on_interrupt(job_running=False) is InterruptAction.CLEAR_LINE
    policy.reset()
    clock.now = 0.1
    assert policy.on_interrupt(job_running=True) is InterruptAction.CANCEL_JOB


def test_interrupt_hints_mention_ctrl_c() -> None:
    policy = InterruptPolicy()
    assert policy.hint(InterruptAction.CANCEL_JOB) == _CANCEL_HINT
    assert policy.hint(InterruptAction.CLEAR_LINE) == _CLEAR_HINT
    for action in InterruptAction:
        assert "Ctrl-C" in policy.hint(action)
    custom = InterruptPolicy(exit_window_s=1.5)
    assert "1.5s" in custom.hint(InterruptAction.CLEAR_LINE)
    assert "Ctrl-C" in custom.hint(InterruptAction.EXIT)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_path_respects_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    assert history_path() == tmp_path / "history"
    assert History().path == tmp_path / "history"


def test_history_path_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FX1_CONFIG_DIR", raising=False)
    assert history_path() == Path.home() / ".fx1" / "history"


def test_history_load_missing_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "missing"
    history = History(path)
    assert history.load() == []
    assert history.entries() == []
    assert not path.exists()


def test_history_load_unreadable(tmp_path: Path) -> None:
    directory = tmp_path / "history"
    directory.mkdir()
    assert History(directory).load() == []
    garbage = tmp_path / "garbage"
    garbage.write_bytes(b"\xff\xfe not utf-8")
    assert History(garbage).load() == []


def test_history_roundtrip_filters_and_keeps_nonconsecutive(tmp_path: Path) -> None:
    path = tmp_path / "history"
    blob = "a" * 31 + "7"
    history = History(path)
    history.append("alpha")
    history.append("")
    history.append("   ")
    history.append("alpha")
    history.append("beta")
    history.append("alpha")
    history.append("/keys set fx1")
    history.append("  /keys set fx1  ")
    history.append("Authorization: Bearer supersecret")
    history.append("sk-abcdefghijklmnop")
    history.append("MOONSHOT_API_KEY=abc")
    history.append(f"token {blob}")
    history.append("alpha")
    history.append("spaced  out")
    history.append("  padded  ")
    history.append("two\nlines")
    assert history.entries() == ["alpha", "beta", "alpha", "spaced  out", "padded", "two lines"]
    history.save()
    assert history.errors == []
    text = path.read_text(encoding="utf-8")
    assert text == "alpha\nbeta\nalpha\nspaced  out\npadded\ntwo lines\n"
    assert "Bearer" not in text
    assert "sk-" not in text
    assert "MOONSHOT" not in text
    assert blob not in text
    assert "/keys" not in text
    reloaded = History(path)
    assert reloaded.load() == ["alpha", "beta", "alpha", "spaced  out", "padded", "two lines"]
    assert reloaded.entries() == ["alpha", "beta", "alpha", "spaced  out", "padded", "two lines"]


@pytest.mark.parametrize(
    ("line", "sensitive"),
    [
        ("hello world", False),
        ("pinball on the fx forecast", False),
        ("Bearer", False),
        ("Bearer ", False),
        ("Authorization: Bearer supersecret", True),
        ("bearer tokenvalue", True),
        ("sk-1234567", False),
        ("sk-12345678", True),
        ("sk_abcdefgh", True),
        ("key-12345678", True),
        ("use sk-abcdefghij please", True),
        ("MOONSHOT_API_KEY", False),
        ("MOONSHOT_API_KEY=", True),
        ("export MOONSHOT_API_KEY=secret", True),
        ("a" * 32, False),
        ("a" * 31, False),
        ("a" * 31 + "1", True),
        ("1" * 32, True),
        ("see " + "ab" * 16 + " now", False),
        ("prefix " + ("A" * 20) + ("9" * 12) + " suffix", True),
        ("/keys set fx1", False),
        ("a" * 20 + "-" + "1" * 20, False),
        ("monkey-abcdefghij", False),
        ("whisk-abcdefghij", False),
    ],
)
def test_history_is_sensitive(line: str, sensitive: bool) -> None:
    assert History.is_sensitive(line) is sensitive


def test_history_append_skips_sensitive_lines(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.append("keep")
    for line in (
        "Bearer abcdef",
        "sk-12345678",
        "sk_12345678",
        "key-abcdefgh",
        "MOONSHOT_API_KEY=x",
        "a" * 31 + "4",
    ):
        assert History.is_sensitive(line) is True
        history.append(line)
    history.append("keep")
    assert history.entries() == ["keep"]


def test_history_max_entries_keeps_newest(tmp_path: Path) -> None:
    path = tmp_path / "history"
    history = History(path, max_entries=3)
    for line in ("a", "b", "c", "d", "e"):
        history.append(line)
    assert history.entries() == ["c", "d", "e"]
    history.save()
    assert History(path, max_entries=3).load() == ["c", "d", "e"]
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert History(path, max_entries=2).load() == ["c", "d"]


def test_history_load_drops_secrets_and_duplicate_commands(tmp_path: Path) -> None:
    path = tmp_path / "history"
    path.write_text(
        "ok\nok\nMOONSHOT_API_KEY=abc\nsk-12345678\n/keys set fx1\nlater\n",
        encoding="utf-8",
    )
    assert History(path).load() == ["ok", "later"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_history_file_and_dir_modes(tmp_path: Path) -> None:
    parent = tmp_path / "cfg"
    parent.mkdir(mode=0o755)
    path = parent / "history"
    history = History(path)
    history.append("line")
    history.save()
    assert history.errors == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700


def test_history_save_swallows_replace_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    history = History(tmp_path / "history")
    history.append("kept")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(lineio.os, "replace", boom)
    history.save()
    assert history.errors == ["disk full"]
    assert history.entries() == ["kept"]
    assert not (tmp_path / "history").exists()
    leftovers = list(tmp_path.glob(".history-*"))
    assert leftovers == []


def test_history_save_swallows_mkstemp_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    history = History(tmp_path / "nested" / "history")
    history.append("kept")

    def boom(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise OSError("mkstemp failed")

    monkeypatch.setattr(lineio.tempfile, "mkstemp", boom)
    history.save()
    assert history.errors == ["mkstemp failed"]
    assert not history.path.exists()


def test_load_into_readline_false_when_module_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "readline", None)
    history = History(tmp_path / "history")
    history.append("hello")
    assert history.load_into_readline() is False


def test_load_into_readline_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delitem(sys.modules, "readline", raising=False)
    real_import = builtins.__import__

    def guarded(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "readline":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    history = History(tmp_path / "history")
    history.append("hello")
    assert history.load_into_readline() is False


def test_load_into_readline_records_add_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    added: list[str] = []

    class FakeReadline:
        def add_history(self, line: str) -> None:
            added.append(line)

    monkeypatch.setitem(sys.modules, "readline", FakeReadline())
    history = History(tmp_path / "history")
    history.append("one")
    history.append("two")
    assert history.load_into_readline() is True
    assert added == ["one", "two"]


def test_load_into_readline_false_without_add_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "readline", object())
    assert History(tmp_path / "history").load_into_readline() is False


def test_load_into_readline_swallows_add_history_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Boom:
        def add_history(self, line: str) -> None:
            raise RuntimeError(line)

    monkeypatch.setitem(sys.modules, "readline", Boom())
    history = History(tmp_path / "history")
    history.append("hello")
    assert history.load_into_readline() is False


# ---------------------------------------------------------------------------
# LineReader
# ---------------------------------------------------------------------------


def test_line_reader_blocking_uses_input_fn_and_ignores_timeout() -> None:
    calls: list[str] = []

    def input_fn(prompt: str) -> str:
        calls.append(prompt)
        return "hello"

    reader = LineReader(stdin=io.StringIO(), input_fn=input_fn, platform="linux")
    assert reader.interactive is False
    assert reader.mode == "blocking"
    assert reader.read("dip › ", 0.0) == "hello"
    assert calls == ["dip › "]


def test_line_reader_blocking_eof_propagates() -> None:
    def input_fn(prompt: str) -> str:
        raise EOFError

    reader = LineReader(stdin=io.StringIO(), input_fn=input_fn)
    with pytest.raises(EOFError):
        reader.read("dip › ", 5)


def test_line_reader_blocking_keyboard_interrupt_propagates() -> None:
    def input_fn(prompt: str) -> str:
        raise KeyboardInterrupt

    reader = LineReader(stdin=io.StringIO(), stdout=io.StringIO(), input_fn=input_fn)
    with pytest.raises(KeyboardInterrupt):
        reader.read(">", 1)


def test_line_reader_modes_and_interactive() -> None:
    posix = LineReader(stdin=TTYStream(), stdout=TTYStream(), platform="linux")
    assert posix.interactive is True
    assert posix.mode == "posix-tty"
    windows = LineReader(
        stdin=TTYStream(),
        stdout=TTYStream(),
        platform="win32",
        kbhit=lambda: False,
        getwch=lambda: "",
    )
    assert windows.interactive is True
    assert windows.mode == "windows-tty"
    headless = LineReader(
        stdin=TTYStream(tty=False),
        stdout=TTYStream(tty=False),
        platform="win32",
    )
    assert headless.interactive is False
    assert headless.mode == "blocking"
    stdout_only = LineReader(
        stdin=TTYStream(tty=True),
        stdout=TTYStream(tty=False),
        platform="linux",
    )
    assert stdout_only.interactive is False
    # Polling is decided by stdin alone: a TTY whose stdout is piped still polls.
    assert stdout_only.mode == "posix-tty"
    broken = LineReader(stdin=BrokenStream(), stdout=BrokenStream(), platform="linux")
    assert broken.interactive is False
    assert broken.mode == "blocking"


def test_posix_timeout_writes_prompt_once_and_does_not_read() -> None:
    stdin = TTYStream("should-stay\n")
    stdout = TTYStream()
    timeouts: list[float] = []

    def select_fn(
        rlist: list[TTYStream],
        _wlist: list[object],
        _xlist: list[object],
        timeout: float,
    ) -> tuple[list[TTYStream], list[object], list[object]]:
        timeouts.append(timeout)
        assert rlist == [stdin]
        return [], [], []

    reader = LineReader(
        stdin=stdin,
        stdout=stdout,
        platform="linux",
        input_fn=_refuse_input,
        select_fn=select_fn,
    )
    assert reader.mode == "posix-tty"
    assert reader.read("dip › ", 0.15) is None
    assert stdout.getvalue() == "dip › "
    assert timeouts == [0.15]
    assert stdin.tell() == 0


def test_posix_ready_returns_stripped_line() -> None:
    stdin = TTYStream("hello\r\n")
    stdout = TTYStream()

    def select_fn(
        _rlist: list[object],
        _wlist: list[object],
        _xlist: list[object],
        _timeout: float,
    ) -> tuple[list[TTYStream], list[object], list[object]]:
        return [stdin], [], []

    reader = LineReader(
        stdin=stdin,
        stdout=stdout,
        platform="darwin",
        input_fn=_refuse_input,
        select_fn=select_fn,
    )
    assert reader.read("> ", 0.2) == "hello"
    assert stdout.getvalue() == "> "


def test_posix_empty_readline_is_eof() -> None:
    stdin = TTYStream("")

    def select_fn(
        _rlist: list[object],
        _wlist: list[object],
        _xlist: list[object],
        _timeout: float,
    ) -> tuple[list[TTYStream], list[object], list[object]]:
        return [stdin], [], []

    reader = LineReader(
        stdin=stdin,
        stdout=TTYStream(),
        platform="linux",
        input_fn=_refuse_input,
        select_fn=select_fn,
    )
    with pytest.raises(EOFError):
        reader.read("> ", 0.2)


def test_windows_enter_submits_and_echoes() -> None:
    reader, stdout, slept = _windows_reader(["a", "b", "\r"])
    assert reader.mode == "windows-tty"
    assert reader.read(">", 5) == "ab"
    assert stdout.getvalue() == ">ab\n"
    assert slept == []


def test_windows_backspace_echoes_and_deletes() -> None:
    reader, stdout, _slept = _windows_reader(["a", "b", "\x7f", "c", "\r"])
    assert reader.read(">", 5) == "ac"
    assert stdout.getvalue() == ">ab\b \bc\n"


def test_windows_backspace_on_empty_buffer_does_not_echo() -> None:
    reader, stdout, _slept = _windows_reader(["\x08", "a", "\n"])
    assert reader.read(">", 5) == "a"
    assert stdout.getvalue() == ">a\n"


def test_windows_timeout_keeps_partial_buffer() -> None:
    reader, stdout, slept = _windows_reader(["a", "b", None, "c", "\r"])
    assert reader.read("dip › ", 0.05) is None
    assert stdout.getvalue() == "dip › ab"
    assert slept == [0.02]
    assert reader.read("", 0.05) == "abc"
    assert stdout.getvalue() == "dip › abc\n"


def test_windows_ctrl_c_raises_keyboard_interrupt() -> None:
    reader, _stdout, _slept = _windows_reader(["a", "\x03"])
    with pytest.raises(KeyboardInterrupt):
        reader.read(">", 5)


def test_windows_ctrl_z_raises_eof() -> None:
    reader, _stdout, _slept = _windows_reader(["\x1a"])
    with pytest.raises(EOFError):
        reader.read(">", 5)


def test_windows_empty_getwch_is_eof() -> None:
    reader, _stdout, _slept = _windows_reader([""])
    with pytest.raises(EOFError):
        reader.read(">", 5)


def test_windows_injected_readers_do_not_import_msvcrt(monkeypatch: pytest.MonkeyPatch) -> None:
    real = lineio.importlib.import_module

    def guarded(name: str, package: str | None = None) -> object:
        if name == "msvcrt":
            raise AssertionError("msvcrt imported")
        return real(name, package)

    monkeypatch.setattr(lineio.importlib, "import_module", guarded)
    reader, _stdout, _slept = _windows_reader(["z", "\r"])
    assert reader.read(">", 1) == "z"


# ---------------------------------------------------------------------------
# width, wrap, color, shorten
# ---------------------------------------------------------------------------


def test_terminal_width_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[int, int]] = []
    columns = {"value": 10}

    def fake(fallback: tuple[int, int] = (80, 24)) -> os.terminal_size:
        captured.append((fallback[0], fallback[1]))
        return os.terminal_size((columns["value"], 24))

    monkeypatch.setattr(lineio.shutil, "get_terminal_size", fake)
    assert terminal_width() == 40
    assert captured == [(80, 24)]
    columns["value"] = 39
    assert terminal_width(default=120) == 40
    assert captured[-1] == (120, 24)
    columns["value"] = 40
    assert terminal_width() == 40
    columns["value"] = 80
    assert terminal_width() == 80
    columns["value"] = 200
    assert terminal_width() == 200
    columns["value"] = 201
    assert terminal_width() == 200
    columns["value"] = 250
    assert terminal_width() == 200


def test_wrap_block_wraps_prose_and_preserves_structure() -> None:
    prose = "alpha beta gamma delta epsilon zeta"
    assert wrap_block(prose, width=12) == "alpha beta\ngamma delta\nepsilon zeta"
    assert wrap_block("short", width=40) == "short"
    assert wrap_block("12345678901", width=10) == "12345678901"
    assert wrap_block("state-of-the-art result", width=10) == "state-of-the-art\nresult"
    code = "lambda value: value + 1  # stays inside the fence"
    block = "\n".join(["intro line", "```python", code, "```", prose])
    lines = wrap_block(block, width=10).splitlines()
    assert "```python" in lines
    assert code in lines
    assert "```" in lines
    indented = "    def stay_intact(): return value"
    assert wrap_block(indented, width=8) == indented
    tabbed = "\tkeep this intact even though it is very long"
    assert wrap_block(tabbed, width=5) == tabbed
    table = "| symbol | pinball | crps |"
    assert wrap_block(table, width=4) == table
    assert wrap_block("hello\n\nworld", width=80) == "hello\n\nworld"
    assert wrap_block("hello\nworld", width=80, indent="> ") == "> hello\n> world"
    wrapped = wrap_block("one two three four", width=8, indent="|")
    assert all(line.startswith("|") for line in wrapped.splitlines())
    assert "\n" in wrapped


def test_wrap_block_defaults_to_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lineio, "terminal_width", lambda default=80: 12)
    prose = "alpha beta gamma delta epsilon zeta"
    assert wrap_block(prose) == wrap_block(prose, width=12)


@pytest.mark.parametrize(
    ("tty", "env", "expected"),
    [
        (True, {"NO_COLOR": ""}, False),
        (True, {"NO_COLOR": "1"}, False),
        (True, {"NO_COLOR": "0"}, False),
        (True, {"TERM": "dumb"}, False),
        (True, {"TERM": "dumb", "NO_COLOR": "1"}, False),
        (False, {}, False),
        (False, {"TERM": "xterm-256color"}, False),
        (True, {}, True),
        (True, {"TERM": "xterm-256color"}, True),
        (True, {"TERM": "xterm", "COLORTERM": "truecolor"}, True),
    ],
)
def test_supports_color_matrix(tty: bool, env: dict[str, str], expected: bool) -> None:
    assert supports_color(TTYStream(tty=tty), env) is expected


def test_supports_color_defaults_and_broken_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", TTYStream(tty=True))
    assert supports_color(environ={}) is True
    monkeypatch.setattr(sys, "stdout", TTYStream(tty=False))
    assert supports_color(environ={"TERM": "xterm"}) is False
    assert supports_color(BrokenStream(), {}) is False


def test_shorten_collapses_and_ellipsizes() -> None:
    assert shorten("hello", 10) == "hello"
    assert shorten("hello", 5) == "hello"
    assert shorten("hello world", 8) == "hello w…"
    assert len(shorten("hello world", 8)) == 8
    assert "..." not in shorten("hello world", 8)
    assert shorten("ab\ncd", 10) == "ab cd"
    assert shorten("ab\r\ncd", 10) == "ab cd"
    assert shorten("abcdef", 1) == "…"
    assert shorten("abcdef", 0) == ""
    assert shorten("abcdef", -3) == ""
    assert shorten("hello!", 5) == "hell…"
