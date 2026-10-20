"""Terminal capabilities, writer, line reader, and real PTY checks."""

from __future__ import annotations

import contextlib
import io
import json
import locale
import os
import re
import stat
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from fx1.interactive.terminal import (
    ConsoleWriter,
    InputCancelled,
    InterruptTracker,
    LineReader,
    Style,
    TermCaps,
    bracketed_paste_supported,
    detect_caps,
    format_count,
    format_duration,
    readline_is_libedit,
    readline_tab_binding,
    render_table,
    truncate_middle,
    wrap_text,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX tty")
_SRC = str(Path(__file__).resolve().parents[2] / "src")

_DRIVER = """\
from __future__ import annotations

import sys
import termios
from pathlib import Path

from fx1.interactive.terminal import ConsoleWriter, LineReader, Style, TermCaps, detect_caps


def _token(before: list[object], after: list[object]) -> str:
    if before == after:
        return "OK"
    changed = [str(index) for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
    return "BAD:" + ",".join(changed)


def _run(mode: str, reader: LineReader, writer: ConsoleWriter, caps: TermCaps) -> None:
    if mode == "round":
        writer.line("GOT:" + reader.read_line("dip> "))
        return
    if mode == "history":
        reader.load_history()
        writer.line("GOT:" + reader.read_line("dip> "))
        return
    if mode == "interrupt":
        try:
            reader.read_line("dip> ")
            writer.line("NOINT")
        except KeyboardInterrupt:
            writer.line("INT")
        return
    if mode == "eof":
        try:
            reader.read_line("dip> ")
            writer.line("NOEOF")
        except EOFError:
            writer.line("EOF")
        return
    if mode == "poll":
        state = {"n": 0}

        def idle() -> None:
            state["n"] += 1
            if state["n"] <= 2:
                writer.line("IDLE" + str(state["n"]))

        line = reader.read_line("dip> ", idle=idle, poll_s=0.05, job_active=lambda: True)
        writer.line("GOT:" + line)
        writer.line("IDLES:" + str(state["n"]))
        return
    if mode == "pollint":
        try:
            reader.read_line("dip> ", idle=lambda: None, poll_s=0.05, job_active=lambda: True)
            writer.line("NOINT")
        except KeyboardInterrupt:
            writer.line("INT")
        return
    if mode == "nocolor":
        style = Style(caps)
        writer.line("COLOR_" + ("ON" if caps.color else "OFF"))
        writer.line(style.red("danger") + style.bold("X"))
        return
    if mode == "secret":
        secret = reader.read_secret("secret> ")
        writer.line("GOTLEN:" + str(len(secret)))
        return
    writer.line("BADMODE")


def main() -> None:
    mode = sys.argv[1]
    history = Path(sys.argv[2])
    caps = detect_caps()
    writer = ConsoleWriter(caps, caps_fn=lambda: caps)
    reader = LineReader(caps, writer, history_path=history)
    before = termios.tcgetattr(sys.stdin.fileno())
    try:
        _run(mode, reader, writer, caps)
    finally:
        reader.close()
        after = termios.tcgetattr(sys.stdin.fileno())
        writer.line("TERMIOS_" + _token(before, after))
        writer.line("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stdout.write("ERR:" + type(exc).__name__ + ":" + str(exc) + "\\n")
        sys.stdout.flush()
"""


class _Stream:
    def __init__(self, *, tty: bool, encoding: str | None) -> None:
        self._tty = tty
        self.encoding = encoding

    def isatty(self) -> bool:
        return self._tty


class _Boom:
    encoding = None

    def isatty(self) -> bool:
        raise OSError("closed")


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Chunked(io.TextIOBase):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def write(self, data: str) -> int:
        self.parts.extend(data)
        return len(data)

    def flush(self) -> None:
        return None


def _caps(
    *,
    stdin_tty: bool = False,
    stdout_tty: bool = False,
    interactive: bool = False,
    color: bool = False,
    unicode: bool = True,
    width: int = 80,
    height: int = 24,
    windows: bool = False,
) -> TermCaps:
    return TermCaps(
        stdin_tty=stdin_tty,
        stdout_tty=stdout_tty,
        interactive=interactive,
        color=color,
        unicode=unicode,
        width=width,
        height=height,
        windows=windows,
    )


def _size(columns: int, lines: int) -> Callable[[], os.terminal_size]:
    def measure() -> os.terminal_size:
        return os.terminal_size((columns, lines))

    return measure


def _detect(
    *,
    tty: bool = True,
    encoding: str | None = "utf-8",
    environ: Mapping[str, str] | None = None,
    columns: int = 80,
    lines: int = 24,
    platform: str = "linux",
    stdout: object | None = None,
    stdin: object | None = None,
) -> TermCaps:
    out = stdout if stdout is not None else _Stream(tty=tty, encoding=encoding)
    inn = stdin if stdin is not None else _Stream(tty=tty, encoding=encoding)
    env = {"TERM": "xterm"} if environ is None else environ
    return detect_caps(
        stdin=inn,  # type: ignore[arg-type]
        stdout=out,  # type: ignore[arg-type]
        environ=env,
        size_fn=_size(columns, lines),
        platform=platform,
    )


def _writer(caps: TermCaps | None = None, **kwargs: object) -> tuple[ConsoleWriter, io.StringIO]:
    buf = io.StringIO()
    extra = dict(kwargs)
    extra.setdefault("out", buf)
    return ConsoleWriter(caps or _caps(), **extra), buf  # type: ignore[arg-type]


def _scripted(lines: list[str]) -> Callable[[str], str]:
    pending = list(lines)

    def read(prompt: str) -> str:
        del prompt
        if not pending:
            raise EOFError
        return pending.pop(0)

    return read


def _reader(
    lines: list[str] | None = None,
    *,
    caps: TermCaps | None = None,
    **kwargs: object,
) -> tuple[LineReader, io.StringIO]:
    chosen = caps or _caps()
    writer, buf = _writer(chosen)
    reader = LineReader(
        chosen,
        writer,
        input_fn=_scripted(lines or []),
        **kwargs,  # type: ignore[arg-type]
    )
    return reader, buf


def test_driver_source_compiles() -> None:
    compile(_DRIVER, "terminal_driver.py", "exec")


@pytest.mark.parametrize(
    ("env", "tty", "expected"),
    [
        ({"TERM": "xterm"}, True, True),
        ({"TERM": "xterm", "NO_COLOR": ""}, True, False),
        ({"TERM": "xterm", "NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),
        ({"TERM": "dumb"}, True, False),
        ({"TERM": "dumb", "FORCE_COLOR": "1"}, True, True),
        ({"TERM": "dumb", "FORCE_COLOR": " 1 "}, True, True),
        ({"TERM": "dumb", "FORCE_COLOR": "0"}, True, False),
        ({"TERM": "dumb", "FORCE_COLOR": "false"}, True, False),
        ({"TERM": "dumb", "FORCE_COLOR": "no"}, True, False),
        ({"TERM": "dumb", "FORCE_COLOR": "off"}, True, False),
        ({"TERM": "dumb", "FORCE_COLOR": ""}, True, False),
        ({"TERM": "dumb", "FORCE_COLOR": " 0 "}, True, False),
        ({"TERM": "xterm"}, False, False),
        ({"TERM": "xterm", "FORCE_COLOR": "2"}, False, True),
        ({"TERM": "xterm", "FORCE_COLOR": "true"}, True, True),
    ],
)
def test_color_matrix(env: dict[str, str], tty: bool, expected: bool) -> None:
    caps = _detect(tty=tty, environ=env)
    assert caps.color is expected


@pytest.mark.parametrize(
    ("stdin_tty", "stdout_tty", "term", "expected"),
    [
        (True, True, "xterm", True),
        (True, True, "dumb", False),
        (True, True, "", True),
        (False, True, "xterm", False),
        (True, False, "xterm", False),
    ],
)
def test_interactive_matrix(stdin_tty: bool, stdout_tty: bool, term: str, expected: bool) -> None:
    caps = detect_caps(
        stdin=_Stream(tty=stdin_tty, encoding="utf-8"),  # type: ignore[arg-type]
        stdout=_Stream(tty=stdout_tty, encoding="utf-8"),  # type: ignore[arg-type]
        environ={"TERM": term},
        size_fn=_size(80, 24),
        platform="linux",
    )
    assert caps.interactive is expected
    assert caps.stdin_tty is stdin_tty
    assert caps.stdout_tty is stdout_tty


@pytest.mark.parametrize(
    ("columns", "lines", "width", "height"),
    [
        (10, 5, 40, 5),
        (40, 10, 40, 10),
        (80, 24, 80, 24),
        (200, 50, 200, 50),
        (201, 3, 200, 3),
        (500, 500, 200, 500),
    ],
)
def test_width_clamps_and_height_passes_through(
    columns: int, lines: int, width: int, height: int
) -> None:
    caps = _detect(columns=columns, lines=lines)
    assert caps.width == width
    assert caps.height == height


def test_width_falls_back_when_size_fails() -> None:
    def boom() -> os.terminal_size:
        raise OSError("no tty")

    caps = detect_caps(
        stdin=_Stream(tty=False, encoding="utf-8"),  # type: ignore[arg-type]
        stdout=_Stream(tty=False, encoding="utf-8"),  # type: ignore[arg-type]
        environ={"TERM": "dumb"},
        size_fn=boom,
        platform="linux",
    )
    assert (caps.width, caps.height) == (80, 24)


def test_nonpositive_size_falls_back() -> None:
    caps = _detect(columns=0, lines=24)
    assert (caps.width, caps.height) == (80, 24)
    caps = _detect(columns=-5, lines=-1)
    assert (caps.width, caps.height) == (80, 24)


def test_isatty_oserror_is_not_a_tty() -> None:
    caps = detect_caps(
        stdin=_Boom(),  # type: ignore[arg-type]
        stdout=_Boom(),  # type: ignore[arg-type]
        environ={"TERM": "xterm"},
        size_fn=_size(80, 24),
        platform="linux",
    )
    assert caps.stdin_tty is False
    assert caps.stdout_tty is False
    assert caps.interactive is False
    assert caps.color is False


def test_windows_flag_follows_platform() -> None:
    assert _detect(platform="win32").windows is True
    assert _detect(platform="linux").windows is False


def test_default_size_fn_stays_in_range() -> None:
    caps = detect_caps(
        stdin=_Stream(tty=False, encoding="utf-8"),  # type: ignore[arg-type]
        stdout=_Stream(tty=False, encoding="utf-8"),  # type: ignore[arg-type]
        environ={"TERM": "dumb"},
        platform="linux",
    )
    assert 40 <= caps.width <= 200
    assert caps.height >= 1


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        ("utf-8", True),
        ("UTF-8", True),
        ("utf8", True),
        ("cp1252", False),
        ("ascii", False),
    ],
)
def test_unicode_from_stdout_encoding(encoding: str, expected: bool) -> None:
    assert _detect(encoding=encoding, environ={"TERM": "xterm"}).unicode is expected


def test_pythonioencoding_beats_stdout() -> None:
    caps = _detect(encoding="utf-8", environ={"TERM": "xterm", "PYTHONIOENCODING": "ascii"})
    assert caps.unicode is False
    caps = _detect(encoding="ascii", environ={"TERM": "xterm", "PYTHONIOENCODING": "utf-8:replace"})
    assert caps.unicode is True


def test_empty_pythonioencoding_falls_through() -> None:
    caps = _detect(encoding="utf-8", environ={"TERM": "xterm", "PYTHONIOENCODING": ""})
    assert caps.unicode is True


def test_unicode_from_locale_when_encoding_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fx1.interactive.terminal.locale.getpreferredencoding",
        lambda _do_setlocale=False: "UTF-8",
    )
    caps = _detect(encoding=None, environ={"TERM": "xterm"})
    assert caps.unicode is True
    monkeypatch.setattr(
        "fx1.interactive.terminal.locale.getpreferredencoding",
        lambda _do_setlocale=False: "ANSI_X3.4-1968",
    )
    caps = _detect(encoding=None, environ={"TERM": "xterm"})
    assert caps.unicode is False


def test_unicode_false_when_locale_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_do_setlocale: bool = False) -> str:
        raise LookupError("no locale")

    monkeypatch.setattr("fx1.interactive.terminal.locale.getpreferredencoding", boom)
    assert _detect(encoding=None, environ={"TERM": "xterm"}).unicode is False
    assert locale.getpreferredencoding is not None


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("bold", "1"),
        ("dim", "2"),
        ("red", "31"),
        ("green", "32"),
        ("yellow", "33"),
        ("magenta", "35"),
        ("cyan", "36"),
    ],
)
def test_style_codes(name: str, code: str) -> None:
    style = Style(_caps(color=True))
    assert getattr(style, name)("x") == f"\x1b[{code}mx\x1b[0m"


def test_style_without_color_is_plain() -> None:
    style = Style(_caps(color=False))
    for name in ("bold", "dim", "red", "green", "yellow", "magenta", "cyan"):
        assert getattr(style, name)("x") == "x"


def test_style_empty_text_has_no_codes() -> None:
    assert Style(_caps(color=True)).red("") == ""


def test_strip_ansi_removes_sgr_csi_and_osc() -> None:
    raw = "\x1b[31mred\x1b[0m \x1b[2K \x1b]0;title\x07"
    assert Style.strip_ansi(raw) == "red  "
    assert Style.strip_ansi("plain") == "plain"


def test_wrap_preserves_blank_lines_and_does_not_join() -> None:
    assert wrap_text("a\n\n\nb", 20) == "a\n\n\nb"
    assert wrap_text("aa\nbb", 20) == "aa\nbb"
    assert wrap_text("hello", 10) == "hello"
    assert wrap_text("hi yo", 10) == "hi yo"


def test_wrap_splits_only_overlong_tokens() -> None:
    assert wrap_text("abcdefghijk", 5) == "abcde\nfghij\nk"
    assert wrap_text("ab", 0) == "a\nb"


def test_wrap_indent() -> None:
    assert (
        wrap_text("hello world", 10, indent="..", subsequent_indent="....") == "..hello\n....world"
    )


def test_wrap_preserves_fences_and_long_urls() -> None:
    src = "```python\nhttps://example.com/very-long-path\n```\nhello world"
    out = wrap_text(src, 8)
    assert "https://example.com/very-long-path" in out.splitlines()
    assert "hello world" not in out
    assert "hello" in out.splitlines()
    assert "world" in out.splitlines()
    fenced = '"""\nnot wrapped even if very long line here\n"""'
    assert wrap_text(fenced, 10) == fenced


def test_truncate_middle() -> None:
    assert truncate_middle("abcdef", 6) == "abcdef"
    assert truncate_middle("abcdefghij", 7) == "abc…hij"
    assert truncate_middle("abcdef", 4) == "a…ef"
    assert truncate_middle("abcdef", 1) == "…"
    assert truncate_middle("abcdef", 0) == ""
    assert truncate_middle("ab", 4) == "ab"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0ms"),
        (0.4, "400ms"),
        (1, "1.0s"),
        (59.9, "59.9s"),
        (60, "1m00s"),
        (90, "1m30s"),
        (3600, "1h00m"),
        (3661, "1h01m"),
        (-5, "0ms"),
        (float("nan"), "0ms"),
        (float("inf"), "0ms"),
    ],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


@pytest.mark.parametrize(
    ("count", "text"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1k"),
        (1500, "1.5k"),
        (1_000_000, "1M"),
        (2_500_000, "2.5M"),
        (3_000_000_000, "3B"),
        (-1500, "-1.5k"),
    ],
)
def test_format_count(count: int, text: str) -> None:
    assert format_count(count) == text


def test_render_table_alignment() -> None:
    text = render_table(
        [{"name": "fx1", "kind": "model"}, {"name": "fx1-lite", "kind": "model"}],
        width=40,
    )
    assert text == ("name      kind \n--------  -----\nfx1       model\nfx1-lite  model")
    assert "\x1b" not in text


def test_render_table_wraps_and_ellipsizes() -> None:
    wrapped = render_table([{"note": "hello world"}], width=8, columns=["note"])
    assert wrapped == "note    \n--------\nhello   \nworld   "
    assert render_table([], width=6, columns=["abcdefghij"]) == "abcde…\n------"
    assert render_table([], width=40) == ""
    assert render_table([{"a": "1", "b": "2"}], width=20, columns=["b"]) == "b\n-\n2"


def test_render_table_lines_fit_width() -> None:
    text = render_table([{"u": "http://example.com/abcdef"}], width=12, columns=["u"])
    assert "\x1b" not in text
    assert text.splitlines()
    assert all(len(line) <= 12 for line in text.splitlines())


def test_interrupt_hint_then_exit() -> None:
    clock = _Clock()
    tracker = InterruptTracker(window_s=2.0, monotonic=clock)
    assert InterruptTracker.HINT == "press Ctrl-C again to exit, or /exit"
    assert tracker.on_interrupt() == "hint"
    clock.now = 2.0
    assert tracker.on_interrupt() == "exit"
    assert tracker.on_interrupt() == "hint"


def test_interrupt_outside_window_and_reset() -> None:
    clock = _Clock()
    tracker = InterruptTracker(window_s=2.0, monotonic=clock)
    assert tracker.on_interrupt() == "hint"
    clock.now = 2.1
    assert tracker.on_interrupt() == "hint"
    tracker.reset()
    clock.now = 2.2
    assert tracker.on_interrupt() == "hint"


def test_interrupt_clock_going_backwards_is_a_hint() -> None:
    clock = _Clock(5.0)
    tracker = InterruptTracker(window_s=2.0, monotonic=clock)
    assert tracker.on_interrupt() == "hint"
    clock.now = 4.0
    assert tracker.on_interrupt() == "hint"


def test_interrupt_rejects_negative_window() -> None:
    with pytest.raises(ValueError, match="window_s"):
        InterruptTracker(window_s=-0.1)


def test_writer_line_out_and_err() -> None:
    err = io.StringIO()
    writer, out = _writer(_caps(), err=err)
    writer.line("ok")
    writer.line("bad", err=True)
    writer.line("")
    assert out.getvalue() == "ok\n\n"
    assert err.getvalue() == "bad\n"


def test_writer_strips_ansi_when_color_off_and_keeps_it_when_on() -> None:
    plain, out = _writer(_caps(color=False))
    plain.line("\x1b[31mred\x1b[0m")
    assert out.getvalue() == "red\n"
    colored, cout = _writer(_caps(color=True))
    colored.line("\x1b[31mred\x1b[0m")
    assert "\x1b[31mred\x1b[0m" in cout.getvalue()


def test_above_prompt_redraws_on_interactive_tty() -> None:
    caps = _caps(interactive=True, stdin_tty=True, stdout_tty=True)
    writer, out = _writer(caps)
    writer.set_prompt_visible("dip> ", lambda: "ab")
    writer.above_prompt("hello")
    assert out.getvalue() == "\r\x1b[2Khello\ndip> ab"


def test_above_prompt_without_prompt_is_a_plain_line() -> None:
    writer, out = _writer(_caps(interactive=True))
    writer.above_prompt("hello")
    assert out.getvalue() == "hello\n"


def test_above_prompt_noninteractive_has_no_escape() -> None:
    writer, out = _writer(_caps())
    writer.set_prompt_visible("dip> ", lambda: "ab")
    writer.above_prompt("hello")
    assert "\x1b" not in out.getvalue()
    assert out.getvalue() == "\nhello\ndip> ab"


def test_above_prompt_dumb_tty_clears_with_spaces() -> None:
    writer, out = _writer(_caps(stdout_tty=True, width=4))
    writer.set_prompt_visible(">", lambda: "")
    writer.above_prompt("hi")
    text = out.getvalue()
    assert "\x1b" not in text
    assert text == "\r    \rhi\n>"


def test_status_in_place_and_clear() -> None:
    caps = _caps(interactive=True, stdout_tty=True, stdin_tty=True, width=10)
    writer, out = _writer(caps)
    writer.status("hello")
    writer.status("abcdefghijklmnop")
    writer.line("next")
    writer.clear_status()
    assert out.getvalue() == "\r\x1b[2Khello\r\x1b[2Kabcdefghi…\r\x1b[2Knext\n"


def test_plain_status_is_throttled() -> None:
    clock = _Clock(10.0)
    writer, out = _writer(_caps(), clock=clock, status_interval_s=1.0)
    writer.status("one")
    writer.status("one")
    writer.status("two")
    assert out.getvalue() == "one\n"
    clock.now = 12.0
    writer.status("two")
    writer.status("three")
    assert out.getvalue() == "one\ntwo\n"
    clock.now = 14.0
    writer.status("three")
    writer.clear_status()
    writer.status("three")
    assert out.getvalue() == "one\ntwo\nthree\nthree\n"


def test_writer_lines_do_not_interleave() -> None:
    stream = _Chunked()
    writer = ConsoleWriter(_caps(), out=stream)
    threads: list[threading.Thread] = []

    def work(index: int) -> None:
        for seq in range(20):
            writer.line(f"T{index}-{seq}-END")

    for index in range(8):
        thread = threading.Thread(target=work, args=(index,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()
    lines = "".join(stream.parts).splitlines()
    expected = {f"T{index}-{seq}-END" for index in range(8) for seq in range(20)}
    assert len(lines) == len(expected)
    assert set(lines) == expected
    assert all(re.fullmatch(r"T\d+-\d+-END", line) for line in lines)


def test_refresh_caps_and_read_line_refreshes() -> None:
    calls = {"n": 0}
    base = _caps(width=80)

    def fresh() -> TermCaps:
        calls["n"] += 1
        return _caps(width=64, height=12)

    writer, _buf = _writer(base, caps_fn=fresh)
    assert writer.refresh_caps().width == 64
    assert writer.caps.height == 12
    reader = LineReader(base, writer, input_fn=lambda _prompt: "z")
    assert reader.read_line(">") == "z"
    assert calls["n"] == 2


def test_nontty_stringio_never_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("select")

    monkeypatch.setattr("select.select", explode)
    stdin = io.StringIO("piped\n")
    caps = _caps()
    out = io.StringIO()
    err = io.StringIO()
    writer = ConsoleWriter(caps, out=out, err=err)

    def job() -> bool:
        raise AssertionError("job")

    reader = LineReader(caps, writer, stdin=stdin)
    assert reader.read_line("ask> ", job_active=job) == "piped"
    assert out.getvalue().startswith("ask> ")
    assert "\x1b" not in out.getvalue()
    assert err.getvalue() == ""
    empty = LineReader(caps, writer, stdin=io.StringIO(""))
    with pytest.raises(EOFError):
        empty.read_line(">")


def test_windows_uses_plain_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "got:" + prompt)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("select")

    monkeypatch.setattr("select.select", explode)
    caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True, windows=True)
    reader = LineReader(caps, ConsoleWriter(caps, out=io.StringIO()))

    def job() -> bool:
        raise AssertionError("job")

    assert reader.read_line("p> ", job_active=job) == "got:p> "


def test_input_fn_does_not_touch_builtin_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(prompt: str) -> str:
        raise AssertionError(prompt)

    monkeypatch.setattr("builtins.input", explode)
    caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True)
    reader = LineReader(caps, ConsoleWriter(caps, out=io.StringIO()), input_fn=lambda _p: "safe")
    assert reader.read_line(">") == "safe"


def test_keyboard_interrupt_clears_interactive_line() -> None:
    def boom(_prompt: str) -> str:
        raise KeyboardInterrupt

    caps = _caps(interactive=True, stdin_tty=True, stdout_tty=True)
    writer, out = _writer(caps)
    reader = LineReader(caps, writer, input_fn=boom)
    with pytest.raises(KeyboardInterrupt):
        reader.read_line("p> ")
    assert "\x1b[2K" in out.getvalue()
    out.seek(0)
    out.truncate(0)
    writer.above_prompt("later")
    assert out.getvalue() == "later\n"


def test_keyboard_interrupt_on_plain_stream_has_no_escape() -> None:
    def boom(_prompt: str) -> str:
        raise KeyboardInterrupt

    writer, out = _writer(_caps())
    reader = LineReader(_caps(), writer, input_fn=boom)
    with pytest.raises(KeyboardInterrupt):
        reader.read_line(">")
    assert "\x1b" not in out.getvalue()
    assert out.getvalue().endswith("\n")


def test_read_line_keeps_internal_newlines() -> None:
    reader, _buf = _reader(["a\nb\n"])
    assert reader.read_line(">") == "a\nb"
    assert reader.history() == ("a\nb",)


def test_multiline_continuation_and_backslash_parity() -> None:
    reader, _buf = _reader(["foo\\", "bar"])
    prompts: list[str] = []
    pending = _scripted(["a\\", "b\\", "c"])

    def tracked(prompt: str) -> str:
        prompts.append(prompt)
        return pending(prompt)

    assert reader.read_message(">") == "foo\nbar"
    other = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()), input_fn=tracked)
    assert other.read_message(">") == "a\nb\nc"
    assert prompts == [">", "… ", "… "]
    literal = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        input_fn=_scripted(["foo\\\\"]),
    )
    assert literal.read_message(">") == "foo\\\\"
    continued = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        input_fn=_scripted(["foo\\\\\\", "z"]),
    )
    assert continued.read_message(">") == "foo\\\\\nz"
    blank = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        input_fn=_scripted(["\\", "x"]),
    )
    assert blank.read_message(">") == "\nx"


def test_fences_and_paste() -> None:
    quotes, _buf = _reader(['"""', "alpha", "beta", '"""'])
    assert quotes.read_message(">") == "alpha\nbeta"
    ticks, _buf = _reader(["```", "alpha", "```"])
    assert ticks.read_message(">") == "alpha"
    not_fence, _buf = _reader(["```python"])
    assert not_fence.read_message(">") == "```python"
    paste, _buf = _reader(["one\ntwo\nthree"])
    assert paste.read_message(">") == "one\ntwo\nthree"
    marked, _buf = _reader(["\x1b[200~one\ntwo\x1b[201~"])
    assert marked.read_message(">") == "one\ntwo"


def test_ctrl_d_cancels_open_block_and_eof_exits() -> None:
    empty, _buf = _reader([])
    with pytest.raises(EOFError):
        empty.read_message(">")
    cancelled, _buf = _reader(['"""', "hidden"])
    with pytest.raises(InputCancelled):
        cancelled.read_message(">")
    assert cancelled.history() == ('"""', "hidden")
    half, _buf = _reader(["foo\\"])
    with pytest.raises(InputCancelled):
        half.read_message(">")


def test_custom_continuation_prompt() -> None:
    seen: list[str] = []

    def read(prompt: str) -> str:
        seen.append(prompt)
        if len(seen) == 1:
            return "a\\"
        return "b"

    reader = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()), input_fn=read)
    assert reader.read_message(">", continuation_prompt=".. ") == "a\nb"
    assert seen == [">", ".. "]


def test_secret_is_not_stored(tmp_path: Path) -> None:
    seen: list[str] = []

    def secret(prompt: str) -> str:
        seen.append(prompt)
        return "s3cret"

    path = tmp_path / "h.json"
    reader = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        secret_fn=secret,
        input_fn=_scripted(["visible"]),
    )
    assert reader.read_secret("key> ") == "s3cret"
    assert seen == ["key> "]
    assert reader.read_line(">") == "visible"
    reader.save_history()
    body = path.read_text(encoding="utf-8")
    assert "s3cret" not in body
    assert "visible" in body
    assert reader.history() == ("visible",)


def test_history_roundtrip_dedup_max_and_skip(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "h.json"
    reader = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        history_max=3,
        input_fn=_scripted(["a", "a", "b", "/keys set sk-test", "c", "café"]),
    )
    for _ in range(6):
        reader.read_line(">")
    assert reader.history() == ("b", "c", "café")
    reader.save_history()
    body = path.read_text(encoding="utf-8")
    assert "sk-test" not in body
    assert "/keys" not in body
    assert list(path.parent.glob("*.tmp")) == []
    again = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        history_max=3,
        input_fn=_scripted([]),
    )
    again.load_history()
    assert again.history() == ("b", "c", "café")


@posix_only
def test_history_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "h.json"
    reader = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        input_fn=_scripted(["ok"]),
    )
    reader.read_line(">")
    reader.save_history()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_corrupt_and_foreign_history_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    reader = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        input_fn=_scripted([]),
    )
    reader.load_history()
    assert reader.history() == ()
    path.write_bytes(b"\xff\xfe")
    reader.load_history()
    assert reader.history() == ()
    path.write_text("{}\n", encoding="utf-8")
    reader.load_history()
    assert reader.history() == ()
    path.write_text('[1, "ok", null, "ok"]\n', encoding="utf-8")
    reader.load_history()
    assert reader.history() == ("ok",)


def test_custom_skip_and_zero_history(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    reader = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        history_max=0,
        skip_history=lambda line: line.startswith("nope"),
        input_fn=_scripted(["nope x", "yes"]),
    )
    reader.read_line(">")
    reader.read_line(">")
    assert reader.history() == ()
    reader.save_history()
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_context_manager_saves_history(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    with LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=path,
        input_fn=_scripted(["hi"]),
    ) as reader:
        assert reader.read_line(">") == "hi"
    assert json.loads(path.read_text(encoding="utf-8")) == ["hi"]


def test_nonconsecutive_duplicates_are_kept() -> None:
    reader, _buf = _reader(["a", "b", "a"])
    for _ in range(3):
        reader.read_line(">")
    assert reader.history() == ("a", "b", "a")


def test_completer_matches_slash_tokens_in_order() -> None:
    def words() -> list[str]:
        return ["/help", "/health", "/exit", "fx1", "/help"]

    reader = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()), words=words)
    assert reader.complete("/he", 0) == "/help"
    assert reader.complete("/he", 1) == "/health"
    assert reader.complete("/he", 2) is None
    assert reader.complete("/he", -1) is None
    assert reader.complete("/", 0) == "/help"
    assert reader.complete("fx", 0) is None
    assert reader.complete("hello", 0) is None


def test_completer_survives_words_errors_and_missing_words() -> None:
    def boom() -> list[str]:
        raise RuntimeError("boom")

    broken = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()), words=boom)
    assert broken.complete("/h", 0) is None
    bare = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()))
    assert bare.complete("/h", 0) is None


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ("libedit 3.0", True),
        ("LIBEDIT", True),
        ("GNU readline", False),
        (None, False),
        ("", False),
    ],
)
def test_readline_libedit_detection(doc: str | None, expected: bool) -> None:
    assert readline_is_libedit(doc) is expected


def test_readline_tab_binding_syntax() -> None:
    assert readline_tab_binding(libedit=False) == "tab: complete"
    assert readline_tab_binding(libedit=True) == "bind ^I rl_complete"


@pytest.mark.parametrize(
    ("version", "libedit", "expected"),
    [
        (2050, False, True),
        (0x0800, False, True),
        (0x0700, False, False),
        (None, False, False),
        (2050, True, False),
    ],
)
def test_bracketed_paste_support(version: int | None, libedit: bool, expected: bool) -> None:
    assert bracketed_paste_supported(version, libedit=libedit) is expected


def _start_reader(target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@posix_only
def test_poll_mode_idle_returns_line_and_records_history(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    handles = {"write": write_fd}
    stream = os.fdopen(read_fd, "r", encoding="utf-8")
    thread: threading.Thread | None = None
    try:
        caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True)
        writer, _out = _writer(caps)
        state = {"n": 0}

        def idle() -> None:
            state["n"] += 1
            if state["n"] == 2:
                os.write(handles["write"], b"ping\n")

        reader = LineReader(caps, writer, stdin=stream, history_path=tmp_path / "h.json")
        result: dict[str, object] = {}

        def run() -> None:
            try:
                result["line"] = reader.read_line(
                    "p> ", idle=idle, poll_s=0.02, job_active=lambda: True
                )
            except Exception as exc:
                result["error"] = exc

        thread = _start_reader(run)
        thread.join(1.5)
        assert not thread.is_alive()
        assert result.get("error") is None
        assert result.get("line") == "ping"
        assert state["n"] >= 2
        reader.close()
        assert "ping" in (tmp_path / "h.json").read_text(encoding="utf-8")
    finally:
        if handles["write"] >= 0:
            os.close(handles["write"])
        stream.close()
        if thread is not None:
            thread.join(0.5)


@posix_only
def test_poll_idle_redraws_above_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(prompt: str) -> str:
        raise AssertionError(prompt)

    monkeypatch.setattr("builtins.input", explode)
    read_fd, write_fd = os.pipe()
    handles = {"write": write_fd}
    stream = os.fdopen(read_fd, "r", encoding="utf-8")
    thread: threading.Thread | None = None
    try:
        caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True, width=40)
        writer, out = _writer(caps)
        state = {"n": 0}

        def idle() -> None:
            state["n"] += 1
            if state["n"] == 1:
                writer.above_prompt("tick")
            if state["n"] == 2:
                os.write(handles["write"], b"ok\n")

        reader = LineReader(caps, writer, stdin=stream)
        result: dict[str, object] = {}

        def run() -> None:
            try:
                result["line"] = reader.read_line(
                    "p> ", idle=idle, poll_s=0.02, job_active=lambda: True
                )
            except Exception as exc:
                result["error"] = exc

        thread = _start_reader(run)
        thread.join(1.5)
        assert not thread.is_alive()
        assert result.get("line") == "ok"
        text = out.getvalue()
        assert "\x1b[2Ktick\n" in text
        assert "p> " in text
    finally:
        if handles["write"] >= 0:
            os.close(handles["write"])
        stream.close()
        if thread is not None:
            thread.join(0.5)


@posix_only
def test_poll_mode_eof(tmp_path: Path) -> None:
    del tmp_path
    read_fd, write_fd = os.pipe()
    handles = {"write": write_fd}
    stream = os.fdopen(read_fd, "r", encoding="utf-8")
    thread: threading.Thread | None = None
    try:
        caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True)
        writer, _out = _writer(caps)
        state = {"n": 0}

        def idle() -> None:
            state["n"] += 1
            if state["n"] == 1 and handles["write"] >= 0:
                os.close(handles["write"])
                handles["write"] = -1

        reader = LineReader(caps, writer, stdin=stream)
        result: dict[str, object] = {}

        def run() -> None:
            try:
                result["line"] = reader.read_line(
                    "p> ", idle=idle, poll_s=0.02, job_active=lambda: True
                )
            except Exception as exc:
                result["error"] = exc

        thread = _start_reader(run)
        thread.join(1.5)
        assert not thread.is_alive()
        assert isinstance(result.get("error"), EOFError)
        assert state["n"] >= 1
    finally:
        if handles["write"] >= 0:
            os.close(handles["write"])
        stream.close()
        if thread is not None:
            thread.join(0.5)


def _child_env(home: Path, extra: dict[str, str] | None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": _SRC,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "INPUTRC": "/dev/null",
    }
    if extra:
        env.update(extra)
    return env


def _kill_child(pid: int) -> None:
    import signal

    if pid <= 0:
        return
    with contextlib.suppress(ProcessLookupError):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def _read_until(fd: int, marker: bytes, deadline: float) -> bytes:
    import select

    buf = bytearray()
    while time.monotonic() < deadline:
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
        if marker in buf:
            return bytes(buf)
    raise AssertionError(f"missing {marker!r} in {bytes(buf)!r}")


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("pty write failed")
        view = view[count:]


def _pty_run(
    tmp_path: Path,
    mode: str,
    *,
    ready: bytes,
    send: bytes | None,
    done: bytes,
    extra_env: dict[str, str] | None = None,
    history_text: str | None = None,
) -> str:
    import pty

    script = tmp_path / "terminal_driver.py"
    script.write_text(_DRIVER, encoding="utf-8")
    history = tmp_path / "history.json"
    if history_text is not None:
        history.write_text(history_text, encoding="utf-8")
    pid, fd = pty.fork()
    if pid == 0:
        try:
            if fd > 2:
                os.close(fd)
            os.execve(
                sys.executable,
                [sys.executable, str(script), mode, str(history)],
                _child_env(tmp_path, extra_env),
            )
        finally:
            os._exit(127)
    try:
        deadline = time.monotonic() + 1.8
        buf = _read_until(fd, ready, deadline)
        if send is not None:
            _write_all(fd, send)
        if done not in buf:
            buf += _read_until(fd, done, deadline)
        return buf.decode("utf-8", errors="replace")
    finally:
        _kill_child(pid)
        os.close(fd)


@posix_only
def test_pty_round_trip_and_termios(tmp_path: Path) -> None:
    text = _pty_run(tmp_path, "round", ready=b"dip> ", send=b"hello\n", done=b"DONE")
    assert "GOT:hello" in text
    assert "TERMIOS_OK" in text
    assert "hello" in (tmp_path / "history.json").read_text(encoding="utf-8")


@posix_only
def test_pty_up_arrow_recalls_history(tmp_path: Path) -> None:
    text = _pty_run(
        tmp_path,
        "history",
        ready=b"dip> ",
        send=b"\x1b[A\n",
        done=b"DONE",
        history_text='["previous"]\n',
    )
    assert "GOT:previous" in text
    assert "TERMIOS_OK" in text


@posix_only
def test_pty_ctrl_c_raises_and_restores_termios(tmp_path: Path) -> None:
    text = _pty_run(tmp_path, "interrupt", ready=b"dip> ", send=b"\x03", done=b"DONE")
    assert "INT" in text
    assert "NOINT" not in text
    assert "TERMIOS_OK" in text


@posix_only
def test_pty_poll_idle_and_history(tmp_path: Path) -> None:
    text = _pty_run(tmp_path, "poll", ready=b"IDLE2", send=b"/status\n", done=b"DONE")
    assert "GOT:/status" in text
    assert "TERMIOS_OK" in text
    match = re.search(r"IDLES:(\d+)", text)
    assert match is not None and int(match.group(1)) >= 2
    assert "/status" in (tmp_path / "history.json").read_text(encoding="utf-8")


@posix_only
def test_pty_poll_ctrl_c(tmp_path: Path) -> None:
    text = _pty_run(tmp_path, "pollint", ready=b"dip> ", send=b"\x03", done=b"DONE")
    assert "INT" in text
    assert "NOINT" not in text
    assert "TERMIOS_OK" in text


@posix_only
def test_pty_ctrl_d_is_eof(tmp_path: Path) -> None:
    text = _pty_run(tmp_path, "eof", ready=b"dip> ", send=b"\x04", done=b"DONE")
    assert "EOF" in text
    assert "NOEOF" not in text
    assert "TERMIOS_OK" in text


@posix_only
def test_pty_no_color_has_no_escape(tmp_path: Path) -> None:
    text = _pty_run(
        tmp_path,
        "nocolor",
        ready=b"DONE",
        send=None,
        done=b"DONE",
        extra_env={"NO_COLOR": "1", "FORCE_COLOR": "1"},
    )
    assert "COLOR_OFF" in text
    assert "danger" in text
    assert "\x1b" not in text
    assert "TERMIOS_OK" in text


@posix_only
def test_pty_secret_is_not_echoed_or_saved(tmp_path: Path) -> None:
    text = _pty_run(tmp_path, "secret", ready=b"secret> ", send=b"s3cret\n", done=b"DONE")
    assert "GOTLEN:6" in text
    assert "s3cret" not in text
    assert "TERMIOS_OK" in text
    assert "s3cret" not in (tmp_path / "history.json").read_text(encoding="utf-8")
