"""Terminal capabilities, console writing, and line editing.

The reader is hybrid:

* No background job, on a POSIX TTY, with ``readline`` available: ``input()``
  provides history, arrows, Ctrl-A/E, reverse-search, and Tab. GNU readline
  binds Tab with ``tab: complete``; libedit (macOS) uses
  ``bind ^I rl_complete``. Bracketed paste is turned on for GNU readline 8+
  (``_READLINE_VERSION >= 0x0800``). libedit has no equivalent setting; lines
  that arrive within ``paste_window_s`` are joined instead.
* A background job is active: cooked-mode ``select`` on stdin so ``idle`` can
  paint progress while the user types. Arrow editing is unavailable in this
  mode. Finished lines are still appended to history.
* Non-TTY stdin: one blocking read and no poll, matching piped ``input()``.
* Windows: plain ``input()``. ``readline``, ``termios``, and ``select`` on
  stdin are not used. Those imports stay inside POSIX-only branches so this
  module imports on Windows.

Prompts go to stdout. SGR color is emitted only when ``TermCaps.color`` is
true (``NO_COLOR``, any value, wins over ``FORCE_COLOR``). Cursor controls are
emitted only when ``TermCaps.interactive`` is true.

``LineReader`` belongs to the thread that owns the terminal. ``ConsoleWriter``
is safe to call from worker threads. History is a JSON list, written
atomically mode ``0600`` through ``fx1.interactive.atomicstore``.
"""

from __future__ import annotations

import contextlib
import locale
import math
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from fx1.interactive.atomicstore import (
    FileLock,
    StoreCorruptError,
    StoreError,
    atomic_write_json,
    read_json,
)
from fx1.interactive.redact import contains_secret

_WIDTH_MIN = 40
_WIDTH_MAX = 200
_WIDTH_FALLBACK = 80
_HEIGHT_FALLBACK = 24
_FENCE_MARKS = frozenset({"```", '"""'})
_PASTE_MARKERS = ("\x1b[200~", "\x1b[201~")
_PASTE_LIMIT = 200
# Longer sequences first. Lone ESC is last so it only catches what the others miss.
_TERMINAL_UNSAFE_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\[[0-?]*[ -/]*$"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x9d[^\x07\x9c]*(?:\x07|\x9c)?"
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"
    r"|[\x90\x98\x9e\x9f][^\x9c]*(?:\x9c)?"
    r"|\x1b[@-_]"
    r"|\x1b"
    r"|[\x00-\x08\x0b-\x1f\x7f]"
    r"|[\u0080-\u009f]"
    r"|[\u200e\u200f\u202a-\u202e\u2066-\u2069]"
    r"|[\u00ad\u061c\u180e\u200b-\u200d\u2028\u2029\u2060\u206a-\u206f\ufeff\ufe0e\ufe0f]"
)
_SGR_RE = re.compile(r"\x1b\[[0-9;:]*m")
_SGR_SLOT_RE = re.compile("\ue000(\\d+)\ue001")
_FENCE_OPEN_RE = re.compile(r"^`{3}[A-Za-z0-9_+#.-]*$")
_ZERO_WIDTH_CHARS = frozenset({0x200B, 0x200C, 0x200D, 0x2060, 0xFE0E, 0xFE0F, 0xFEFF})
_STREAM_LOCKS: dict[int, threading.Lock] = {}
_STREAM_LOCK_GUARD = threading.Lock()

Completer = Callable[[str, int], str | None]


@dataclass(frozen=True, slots=True)
class TermCaps:
    """Detected terminal features. ``width`` is clamped to 40..200."""

    stdin_tty: bool
    stdout_tty: bool
    interactive: bool
    color: bool
    unicode: bool
    width: int
    height: int
    windows: bool


class InputCancelled(Exception):
    """Ctrl-D discarded an in-progress multiline block.

    A Ctrl-D before any line was accepted is ``EOFError`` instead, so the
    console can exit.
    """


def detect_caps(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    size_fn: Callable[[], os.terminal_size] | None = None,
    platform: str | None = None,
) -> TermCaps:
    """Detect TTY, color, Unicode, and size.

    ``NO_COLOR`` set to any value disables color even when ``FORCE_COLOR`` is
    also set. Without ``NO_COLOR``, ``FORCE_COLOR`` (not ``0``/``false``/
    ``no``/``off``/empty) enables color on a dumb terminal and on non-TTYs.
    Otherwise color is on only for a non-dumb TTY stdout.

    ``interactive`` is both streams being TTYs and ``TERM`` not exactly
    ``dumb`` (comparison is case-sensitive). An empty ``TERM`` is not dumb.

    Unicode is decided from ``PYTHONIOENCODING``, then ``stdout.encoding``,
    then ``locale.getpreferredencoding(False)``. An unknown encoding is false.
    ``width`` falls back to 80 when the size cannot be read or is non-positive,
    then clamps to 40..200. ``height`` falls back to 24 and is not upper-clamped.
    """
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    env = os.environ if environ is None else environ
    system = sys.platform if platform is None else platform
    measure = shutil.get_terminal_size if size_fn is None else size_fn
    stdin_tty = _stream_isatty(in_stream)
    stdout_tty = _stream_isatty(out_stream)
    term = env.get("TERM", "")
    width, height = _terminal_size(measure)
    plain = plain_requested(env)
    interactive = stdin_tty and stdout_tty and term != "dumb"
    color = _color_enabled(env, stdout_tty=stdout_tty, term=term)
    if plain:
        interactive = False
        color = False
    return TermCaps(
        stdin_tty=stdin_tty,
        stdout_tty=stdout_tty,
        interactive=interactive,
        color=color,
        unicode=_unicode_enabled(env, out_stream),
        width=width,
        height=height,
        windows=system == "win32",
    )


def readline_is_libedit(doc: str | None) -> bool:
    """True when a ``readline`` module docstring says it is libedit."""
    return doc is not None and "libedit" in doc.lower()


def plain_requested(env: Mapping[str, str] | None = None) -> bool:
    """True when ``DIPCATCHER_PLAIN`` asks for a text-only terminal."""
    source = os.environ if env is None else env
    raw = source.get("DIPCATCHER_PLAIN")
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def stream_lock(stream: TextIO) -> threading.Lock:
    """Lock shared by :class:`ConsoleWriter` and the orb animator for ``stream``."""
    key = id(stream)
    with _STREAM_LOCK_GUARD:
        lock = _STREAM_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _STREAM_LOCKS[key] = lock
        return lock


_SGR_PLAIN = frozenset(
    {
        0,
        1,
        2,
        22,
        *range(30, 38),
        39,
        *range(40, 48),
        49,
        *range(90, 98),
        *range(100, 108),
    }
)


def _safe_sgr(sequence: str) -> bool:
    """True for reset, intensity, and color. Conceal, blink, and reverse are not."""
    body = sequence[2:-1]
    if body == "":
        return True
    return _sgr_body_safe(body)


def _sgr_body_safe(body: str) -> bool:
    try:
        codes = [int(part) if part else 0 for part in re.split(r"[;:]", body)]
    except ValueError:
        return False
    return _sgr_codes_safe(codes)


def _sgr_codes_safe(codes: list[int]) -> bool:
    index = 0
    while index < len(codes):
        step = _sgr_step(codes, index)
        if step is None:
            return False
        index = step
    return True


def _sgr_step(codes: list[int], index: int) -> int | None:
    code = codes[index]
    if code in _SGR_PLAIN:
        return index + 1
    if code in {38, 48}:
        return _sgr_color(codes, index)
    return None


def _sgr_color(codes: list[int], index: int) -> int | None:
    if index + 1 >= len(codes):
        return None
    mode = codes[index + 1]
    if mode == 5:
        return _sgr_indexed(codes, index)
    if mode == 2:
        return _sgr_rgb(codes, index)
    return None


def _sgr_indexed(codes: list[int], index: int) -> int | None:
    if index + 2 >= len(codes) or not _byte_ok(codes[index + 2]):
        return None
    return index + 3


def _sgr_rgb(codes: list[int], index: int) -> int | None:
    if index + 4 >= len(codes):
        return None
    if all(_byte_ok(codes[index + offset]) for offset in (2, 3, 4)):
        return index + 5
    return None


def _byte_ok(value: int) -> bool:
    return 0 <= value <= 255


def sanitize_for_terminal(text: str, *, keep_sgr: bool = False) -> str:
    """Drop terminal controls. Printable text, newlines, and tabs remain.

    OSC (including clipboard and title writes), CSI other than SGR, DCS, APC,
    PM, SOS, C1 controls, bidi overrides, and zero-width format characters
    are removed. The result is idempotent. ``keep_sgr`` retains color and
    intensity SGR and nothing else.
    """
    if text == "":
        return ""
    if not keep_sgr:
        return _TERMINAL_UNSAFE_RE.sub("", text)
    return _sanitize_keeping_sgr(text)


def _sanitize_keeping_sgr(text: str) -> str:
    saved: list[str] = []

    def grab(match: re.Match[str]) -> str:
        if not _safe_sgr(match.group(0)):
            return ""
        saved.append(match.group(0))
        return f"\ue000{len(saved) - 1}\ue001"

    def put(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 0 <= index < len(saved):
            return saved[index]
        return ""

    protected = _SGR_RE.sub(grab, text)
    return _SGR_SLOT_RE.sub(put, _TERMINAL_UNSAFE_RE.sub("", protected))


def readline_tab_binding(*, libedit: bool) -> str:
    """Tab-completion binding for GNU readline or libedit.

    GNU readline: ``tab: complete``. libedit: ``bind ^I rl_complete``.
    """
    if libedit:
        return "bind ^I rl_complete"
    return "tab: complete"


def bracketed_paste_supported(version: int | None, *, libedit: bool) -> bool:
    """GNU readline 8+ reports paste brackets. libedit does not."""
    if libedit or version is None:
        return False
    return version >= 0x0800


class Style:
    """SGR wrappers. With color disabled, every method returns the text unchanged."""

    def __init__(self, caps: TermCaps) -> None:
        self._color = caps.color

    def bold(self, text: str) -> str:
        return self._paint("1", text)

    def dim(self, text: str) -> str:
        return self._paint("2", text)

    def red(self, text: str) -> str:
        return self._paint("31", text)

    def green(self, text: str) -> str:
        return self._paint("32", text)

    def yellow(self, text: str) -> str:
        return self._paint("33", text)

    def cyan(self, text: str) -> str:
        return self._paint("36", text)

    def magenta(self, text: str) -> str:
        return self._paint("35", text)

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove terminal controls. Alias of :func:`sanitize_for_terminal`."""
        return sanitize_for_terminal(text)

    def _paint(self, code: str, text: str) -> str:
        if not self._color or text == "":
            return text
        return f"\x1b[{code}m{text}\x1b[0m"


def wrap_text(
    text: str,
    width: int,
    *,
    indent: str = "",
    subsequent_indent: str = "",
) -> str:
    """Wrap plain text to ``width`` without joining original lines.

    Blank lines stay blank. A fenced region (three backticks, optionally with a
    language tag, or three double-quotes, through the matching closer) is
    copied unchanged. Tokens and URLs split only when longer than the content
    width. Display width follows Unicode East Asian Width; combining marks
    and zero-width joiners contribute nothing.
    """
    safe = width if width > 0 else 1
    rendered: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        fence, pieces = _consume_line(line, fence, safe, indent, subsequent_indent)
        rendered.extend(pieces)
    return "\n".join(rendered)


def display_width(text: str) -> int:
    """Terminal columns occupied by ``text`` (East Asian width, not ``len``)."""
    return sum(_char_width(ch) for ch in text)


def truncate_middle(text: str, width: int) -> str:
    """Shorten ``text`` to ``width`` with a single-character ellipsis in the middle."""
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    if width == 1:
        return "…"
    keep = width - 1
    left = keep // 2
    right = keep - left
    if right == 0:
        return _take_prefix(text, left) + "…"
    return _take_prefix(text, left) + "…" + _take_suffix(text, right)


def format_duration(seconds: float) -> str:
    """Compact duration: ``400ms``, ``1.5s``, ``1m05s``, ``2h03m``.

    Non-finite and negative values render as ``0ms``. Sub-minute values keep
    one decimal. Longer values drop the fractional second.
    """
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    if seconds < 1:
        return f"{round(seconds * 1000):.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(seconds)
    if whole < 3600:
        minutes, secs = divmod(whole, 60)
        return f"{minutes}m{secs:02d}s"
    hours, rem = divmod(whole, 3600)
    return f"{hours}h{rem // 60:02d}m"


def format_count(n: int) -> str:
    """Compact integer: ``999``, ``1.5k``, ``2M``, ``3B``. Negatives keep the sign."""
    sign = "-" if n < 0 else ""
    value = abs(n)
    for suffix, scale in (("B", 1_000_000_000), ("M", 1_000_000), ("k", 1_000)):
        if value >= scale:
            return sign + _scaled(value, scale) + suffix
    return f"{sign}{value}"


def render_table(
    rows: Sequence[Mapping[str, str]],
    *,
    width: int,
    columns: Sequence[str] | None = None,
) -> str:
    """Plain (no ANSI) width-aware table.

    Columns default to the first row's keys. Cells wrap on spaces; a token
    longer than its column is split, and a header that does not fit is
    ellipsized. When ``width`` cannot hold one character per column plus the
    two-space gutters, the rendered line may exceed ``width``.
    """
    cols = list(columns) if columns is not None else _columns_of(rows)
    if not cols:
        return ""
    widths = _column_widths(cols, rows, width if width > 0 else 1)
    shown = [_fit(sanitize_for_terminal(col), w) for col, w in zip(cols, widths, strict=True)]
    lines = [_join_cells(shown)]
    lines.append(_join_cells("-" * w for w in widths))
    for row in rows:
        lines.extend(_body_lines(cols, row, widths))
    return "\n".join(lines)


class ConsoleWriter:
    """Locked stdout/stderr writer. Worker threads may call every method."""

    def __init__(
        self,
        caps: TermCaps,
        *,
        out: TextIO | None = None,
        err: TextIO | None = None,
        caps_fn: Callable[[], TermCaps] | None = None,
        clock: Callable[[], float] | None = None,
        status_interval_s: float = 0.2,
    ) -> None:
        self._caps = caps
        self._out = sys.stdout if out is None else out
        self._err = sys.stderr if err is None else err
        frozen = caps
        self._caps_fn = caps_fn if caps_fn is not None else (lambda: frozen)
        self._clock = time.monotonic if clock is None else clock
        self._status_interval_s = status_interval_s if status_interval_s > 0 else 0.0
        self._lock = stream_lock(self._out)
        self._prompt: str | None = None
        self._buffer_fn: Callable[[], str] | None = None
        self._status_active = False
        self._status_seen = False
        self._owns_composer = False
        self._above_text = ""
        self._last_status = ""
        self._last_status_at = 0.0

    @property
    def caps(self) -> TermCaps:
        with self._lock:
            return self._caps

    @property
    def out(self) -> TextIO:
        return self._out

    def refresh_caps(self) -> TermCaps:
        """Replace capabilities from ``caps_fn``.

        The default function returns the caps passed to the constructor, so
        tests stay stable. Pass ``caps_fn=detect_caps`` to track resize and
        ``NO_COLOR`` for the life of the process. ``LineReader.read_line``
        calls this before every read.
        """
        fresh = self._caps_fn()
        with self._lock:
            self._caps = fresh
            return fresh

    def emit(self, text: str = "") -> None:
        """Thread-safe transcript line. Stays above a visible composer."""
        self.line(text)

    def line(self, text: str = "", *, err: bool = False) -> None:
        stream = self._err if err else self._out
        with self._lock:
            if stream is self._out and self._prompt is not None:
                self._write_above_locked(text)
                return
            self._end_status_locked(stream)
            self._emit_line_locked(stream, text)

    def above_prompt(self, text: str) -> None:
        """Print ``text`` above the live prompt, then redraw that prompt.

        When no prompt is visible this is a plain ``line``. ``buffer_fn`` must
        not call back into the writer (it runs while the writer lock is held).
        """
        with self._lock:
            if self._prompt is None:
                self._end_status_locked(self._out)
                self._emit_line_locked(self._out, text)
                return
            self._write_above_locked(text)

    def status(self, text: str) -> None:
        """In-place status on an interactive TTY; a throttled plain line otherwise.

        Interactive updates are not throttled. Plain updates skip an unchanged
        line and also skip a changed line that arrives before
        ``status_interval_s`` (default 0.2s). ``clear_status`` resets that.
        """
        clean = " ".join(text.split())
        with self._lock:
            self._write_status_locked(clean)

    def clear_status(self) -> None:
        with self._lock:
            if self._status_active and self._caps.interactive and not self._owns_composer:
                self._out.write("\r\x1b[2K")
                self._out.flush()
            self._status_active = False
            self._status_seen = False
            self._above_text = ""
            self._last_status = ""

    def set_prompt_visible(
        self,
        prompt: str | None,
        buffer_fn: Callable[[], str] | None = None,
    ) -> None:
        """Record the prompt ``LineReader`` is showing. ``None`` hides it.

        This does not write the prompt. The reader writes it (via ``input`` or
        ``write_prompt``) so readline and cooked mode stay in charge of echo.
        """
        with self._lock:
            self._prompt = prompt
            self._buffer_fn = buffer_fn

    def write_prompt(self, prompt: str) -> None:
        """Write ``prompt`` with no newline, under the writer lock."""
        with self._lock:
            shown = sanitize_for_terminal(prompt, keep_sgr=self._caps.color)
            self._out.write(shown)
            self._out.flush()

    def set_composer_owned(self, owned: bool) -> None:
        """When true, status is drawn above the composer instead of over it."""
        with self._lock:
            self._owns_composer = owned

    def release_composer(self) -> None:
        """Drop composer ownership without erasing the line the user submitted."""
        with self._lock:
            self._owns_composer = False
            self._status_active = False
            self._above_text = ""
            self._prompt = None
            self._buffer_fn = None

    def redraw_composer(self) -> None:
        with self._lock:
            if self._prompt is None:
                return
            self._paint_composer_locked()

    def repaint(self) -> None:
        """Redraw status and composer after SIGWINCH. No-op without a prompt."""
        with self._lock:
            if self._prompt is None:
                return
            if self._status_active and self._above_text:
                self._paint_status_above_locked(self._above_text)
                return
            self._paint_composer_locked()

    def discard_line(self) -> None:
        """Erase the physical line and forget prompt plus status.

        ``LineReader`` calls this while handling ``KeyboardInterrupt``, after
        termios has been restored.
        """
        with self._lock:
            with contextlib.suppress(OSError):
                self._out.write(self._erase_sequence())
                self._out.flush()
            self._prompt = None
            self._buffer_fn = None
            self._status_active = False

    def _write_status_locked(self, clean: str) -> None:
        if self._prompt is not None and self._owns_composer and self._caps.interactive:
            self._paint_status_above_locked(clean)
            return
        if self._prompt is not None and self._caps.interactive:
            return
        if self._caps.interactive:
            self._paint_status_inline_locked(clean)
            return
        self._status_plain_locked(clean)

    def _paint_status_inline_locked(self, text: str) -> None:
        shown = _clip_end(self._render_user(text), self._caps.width)
        self._out.write("\r\x1b[2K" + shown)
        self._out.flush()
        self._status_active = True

    def _paint_status_above_locked(self, text: str) -> None:
        shown = _clip_end(self._render_user(text), max(1, self._caps.width))
        self._above_text = text
        if self._status_active:
            self._out.write("\x1b[1A")
        self._out.write("\r\x1b[2K" + shown + "\n")
        self._status_active = True
        self._paint_composer_locked()

    def _paint_composer_locked(self) -> None:
        prompt = sanitize_for_terminal(self._prompt or "", keep_sgr=self._caps.color)
        buffer = sanitize_for_terminal(self._buffer_text(), keep_sgr=self._caps.color)
        shown = _fit_composer(prompt, buffer, self._caps.width)
        if self._caps.interactive:
            self._out.write("\r\x1b[2K")
        self._out.write(shown)
        self._out.flush()

    def _end_status_locked(self, stream: TextIO) -> None:
        if stream is self._out and self._status_active and self._caps.interactive:
            stream.write("\r\x1b[2K")
        self._status_active = False
        self._status_seen = False

    def _emit_line_locked(self, stream: TextIO, text: str) -> None:
        shown = self._render_user(text)
        stream.write(shown)
        if not shown.endswith("\n"):
            stream.write("\n")
        stream.flush()

    def _write_above_locked(self, text: str) -> None:
        self._out.write(self._erase_sequence())
        self._status_active = False
        self._status_seen = False
        self._emit_line_locked(self._out, text)
        self._redraw_locked()

    def _redraw_locked(self) -> None:
        prompt = sanitize_for_terminal(self._prompt or "", keep_sgr=self._caps.color)
        buffer = sanitize_for_terminal(self._buffer_text(), keep_sgr=self._caps.color)
        self._out.write(prompt)
        self._out.write(buffer)
        self._out.flush()

    def _buffer_text(self) -> str:
        fn = self._buffer_fn
        if fn is None:
            return ""
        try:
            value = fn()
        except Exception:
            return ""
        return value or ""

    def _status_plain_locked(self, text: str) -> None:
        shown = self._render_user(text)
        now = self._clock()
        if self._status_seen and shown == self._last_status:
            return
        if self._status_seen and now - self._last_status_at < self._status_interval_s:
            return
        self._last_status = shown
        self._last_status_at = now
        self._status_seen = True
        self._emit_line_locked(self._out, shown)

    def _render_user(self, text: str) -> str:
        return sanitize_for_terminal(text, keep_sgr=self._caps.color)

    def _erase_sequence(self) -> str:
        if self._caps.interactive:
            return "\r\x1b[2K"
        if self._caps.stdout_tty:
            return "\r" + (" " * max(1, self._caps.width)) + "\r"
        return "\n"


class InterruptTracker:
    """Double Ctrl-C: the first is a hint, a second inside the window exits.

    ``LineReader`` does not consult this. It always re-raises
    ``KeyboardInterrupt`` after clearing the pending line. The console calls
    ``on_interrupt`` and prints ``HINT`` when the result is ``hint``.
    """

    HINT: str = "press Ctrl-C again to exit, or /exit"

    def __init__(
        self,
        *,
        window_s: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_s < 0:
            raise ValueError("window_s must be >= 0")
        self._window_s = window_s
        self._monotonic = monotonic
        self._last: float | None = None

    def on_interrupt(self) -> Literal["hint", "exit"]:
        now = self._monotonic()
        previous = self._last
        if previous is not None and _inside_window(now, previous, self._window_s):
            self._last = None
            return "exit"
        self._last = now
        return "hint"

    def reset(self) -> None:
        self._last = None


def _skip_key_lines(line: str) -> bool:
    """Default history filter: ``/keys`` lines and credential-shaped text."""
    return any(_private_history_line(part) for part in line.split("\n"))


def _private_history_line(part: str) -> bool:
    if part.lstrip().casefold().startswith("/keys"):
        return True
    return contains_secret(part)


def _history_keep(line: str, skip: Callable[[str], bool]) -> str:
    """Drop blank lines and lines ``skip`` rejects. Other lines stay, in order."""
    kept = [part for part in line.split("\n") if part != "" and not skip(part)]
    return "\n".join(kept)


def _readline_flat(line: str) -> str:
    """Readline history entries cannot carry a raw newline."""
    return line.replace("\n", " ")


class LineReader:
    """Hybrid line editor. Not safe to call ``read_*`` from two threads.

    Injected ``stdin`` that is not ``sys.stdin`` is read directly and never
    via ``input()``, so tests and pipes cannot block on the process terminal.
    ``input_fn``, when set, replaces every read (including secrets' sibling
    line reads — secrets use ``secret_fn`` or ``getpass``).
    """

    def __init__(
        self,
        caps: TermCaps,
        writer: ConsoleWriter,
        *,
        history_path: Path | None = None,
        history_max: int = 1000,
        words: Callable[[], Iterable[str]] | None = None,
        input_fn: Callable[[str], str] | None = None,
        skip_history: Callable[[str], bool] = _skip_key_lines,
        stdin: TextIO | None = None,
        secret_fn: Callable[[str], str] | None = None,
        paste_window_s: float = 0.01,
    ) -> None:
        self._writer = writer
        self._history_path = history_path
        self._history_max = history_max if history_max > 0 else 0
        self._words = words
        self._input_fn = input_fn
        self._skip_history = skip_history
        self._stdin = sys.stdin if stdin is None else stdin
        self._secret_fn = secret_fn
        self._paste_window_s = paste_window_s if paste_window_s > 0 else 0.0
        self._history: list[str] = []
        self._history_base = 0
        self._queued_lines: list[str] = []
        self._stdin_prefix = ""
        self._io_thread = threading.get_ident()
        self._buffer_snapshot = ""
        self._matches: list[str] = []
        self._rl_ready = False
        self._completer_set = False
        self._saved_completer: Completer | None = None
        self._add_history: Callable[[str], None] | None = None
        self._clear_history: Callable[[], None] | None = None
        self._get_buffer: Callable[[], str] | None = None

    def __enter__(self) -> LineReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def history(self) -> tuple[str, ...]:
        """In-memory history after consecutive de-dup and ``skip_history``."""
        return tuple(self._history)

    def read_line(
        self,
        prompt: str,
        *,
        idle: Callable[[], None] | None = None,
        poll_s: float = 0.15,
        job_active: Callable[[], bool] | None = None,
    ) -> str:
        """Read one submission. Strips one trailing newline.

        Raises ``EOFError`` on Ctrl-D at the start of a read and
        ``KeyboardInterrupt`` on Ctrl-C after clearing the pending line.
        Embedded newlines (bracketed paste) are preserved. ``job_active`` is
        consulted at most once.
        """
        self._writer.refresh_caps()
        caps = self._writer.caps
        polling = self._should_poll(caps, job_active)
        from fx1.interactive.tty_guard import guard_tty

        try:
            with guard_tty(self._stdin, self._writer.out):
                return _read_guarded(self, caps, prompt, idle, poll_s, polling=polling)
        except KeyboardInterrupt:
            self._writer.discard_line()
            raise

    def read_message(
        self,
        prompt: str,
        *,
        continuation_prompt: str = "… ",
        idle: Callable[[], None] | None = None,
        poll_s: float = 0.15,
        job_active: Callable[[], bool] | None = None,
    ) -> str:
        """Read a possibly multiline message.

        A line whose trailing backslash count is odd continues (the final
        backslash is removed). A line that is exactly three backticks or three
        double-quotes opens a fence closed by that same line. A value that already contains
        newlines (bracketed paste, or ``input_fn``) is one paste. On a TTY,
        further lines readable within ``paste_window_s`` are joined the same
        way. Ctrl-D after the block has started raises ``InputCancelled``.
        """
        builder = _MessageBuilder()
        prompt_now = prompt
        while True:
            try:
                line = self._message_source(
                    prompt_now, idle=idle, poll_s=poll_s, job_active=job_active
                )
            except EOFError:
                if builder.started:
                    raise InputCancelled from None
                raise
            prompt_now = continuation_prompt
            if builder.feed(line):
                self._queued_lines.extend(builder.pending)
                return builder.text()

    def read_secret(self, prompt: str) -> str:
        """Read a secret with echo off. Never stored in history.

        ``secret_fn`` replaces ``getpass`` in tests. The prompt is written to
        the writer stream (stdout by default).
        """
        if self._secret_fn is not None:
            return self._secret_fn(prompt)
        import getpass

        with _saved_termios(self._writer.out):
            return getpass.getpass(prompt, stream=self._writer.out)

    def load_history(self) -> None:
        """Replace memory from disk. Missing, corrupt, or foreign JSON is ignored.

        Corrupt files are left in place (they are not evidence). The next
        successful ``save_history`` replaces them.
        """
        self._history = _load_history_file(
            self._history_path, limit=self._history_max, skip=self._skip_history
        )
        self._history_base = len(self._history)
        if self._rl_ready:
            self._replay_history()

    def save_history(self) -> None:
        """Merge this session's new lines into the on-disk history and write it.

        The file lock covers the read-modify-write, so two live sessions append
        instead of dropping each other's lines. Mode is ``0600``. No-op without
        a path. ``history_max`` of zero clears the file. Secrets and ``/keys``
        lines are omitted. A read-only directory raises; :meth:`close` does not.
        """
        path = self._history_path
        if path is None:
            return
        if self._history_max <= 0:
            self._store_history(path, [])
            return
        fresh = self._history[self._history_base :]
        with FileLock(path):
            disk = _load_history_file(path, limit=self._history_max, skip=self._skip_history)
            merged = _clip_history(
                [*disk, *fresh], limit=self._history_max, skip=self._skip_history
            )
            atomic_write_json(path, merged)
        self._history = merged
        self._history_base = len(self._history)
        if self._rl_ready:
            self._replay_history()

    def close(self) -> None:
        try:
            self.save_history()
        except (OSError, StoreError):
            pass
        finally:
            self._detach_readline()

    def complete(self, text: str, state: int) -> str | None:
        """Readline completer. Matches ``words()`` only when the buffer starts with ``/``.

        ``words()`` should yield full tokens (``/help``, ``fx1``). Order is
        preserved and duplicates are dropped. A raising ``words()`` yields no
        matches so Tab cannot take down the prompt.
        """
        if state == 0:
            self._matches = self._collect(text)
        if state < 0 or state >= len(self._matches):
            return None
        return self._matches[state]

    def _should_poll(self, caps: TermCaps, job_active: Callable[[], bool] | None) -> bool:
        if self._input_fn is not None or not _can_use_tty(caps):
            return False
        if _stream_fd(self._stdin) is None:
            return False
        return _call_job(job_active)

    def _should_edit(self, caps: TermCaps, polling: bool) -> bool:
        if polling or self._input_fn is not None:
            return False
        return _can_use_tty(caps) and self._stdin is sys.stdin

    def _read_recorded(
        self,
        caps: TermCaps,
        prompt: str,
        idle: Callable[[], None] | None,
        poll_s: float,
        *,
        polling: bool,
    ) -> str:
        editing = self._should_edit(caps, polling)
        if editing:
            self._arm_readline()
        self._io_thread = threading.get_ident()
        buffer_fn = self._snapshot_buffer if editing else None
        self._writer.set_prompt_visible(prompt, buffer_fn)
        try:
            raw = self._dispatch(prompt, idle, poll_s, polling=polling, editing=editing)
        finally:
            self._writer.set_prompt_visible(None)
        line = _clean_line(raw)
        self._remember(line)
        return line

    def _dispatch(
        self,
        prompt: str,
        idle: Callable[[], None] | None,
        poll_s: float,
        *,
        polling: bool,
        editing: bool,
    ) -> str:
        if self._input_fn is not None:
            return self._input_fn(prompt)
        if polling and _cbreak_ready(self._stdin):
            from fx1.interactive.tty_guard import read_cbreak

            return read_cbreak(self._stdin, self._writer, prompt, idle, poll_s)
        if polling:
            return self._read_polled(prompt, idle, poll_s)
        if editing:
            return input(prompt)
        return self._read_blocking(prompt)

    def _read_blocking(self, prompt: str) -> str:
        if self._stdin is sys.stdin:
            return input(prompt)
        self._writer.write_prompt(prompt)
        return self._prefixed(_read_stream_line(self._stdin))

    def _read_polled(self, prompt: str, idle: Callable[[], None] | None, poll_s: float) -> str:
        self._writer.write_prompt(prompt)
        timeout = poll_s if poll_s > 0 else 0.0
        while True:
            if _stdin_ready(self._stdin, timeout):
                return self._prefixed(_read_stream_line(self._stdin))
            if idle is not None:
                idle()

    def _next_message_line(
        self,
        prompt: str,
        *,
        idle: Callable[[], None] | None,
        poll_s: float,
        job_active: Callable[[], bool] | None,
    ) -> str:
        line = self.read_line(prompt, idle=idle, poll_s=poll_s, job_active=job_active)
        extra = self._paste_burst(line)
        if not extra:
            return line
        return "\n".join([line, *extra])

    def _paste_burst(self, line: str) -> list[str]:
        if "\n" in line or self._input_fn is not None:
            return []
        if not _can_use_tty(self._writer.caps) or _stream_fd(self._stdin) is None:
            return []
        return self._drain_ready(self._paste_window_s)

    def _message_source(
        self,
        prompt: str,
        *,
        idle: Callable[[], None] | None,
        poll_s: float,
        job_active: Callable[[], bool] | None,
    ) -> str:
        if self._queued_lines:
            text = "\n".join(self._queued_lines)
            self._queued_lines = []
            return text
        return self._next_message_line(prompt, idle=idle, poll_s=poll_s, job_active=job_active)

    def _snapshot_buffer(self) -> str:
        if threading.get_ident() != self._io_thread:
            return self._buffer_snapshot
        get = self._get_buffer
        if get is None:
            return self._buffer_snapshot
        try:
            value = get() or ""
        except (OSError, RuntimeError):
            return self._buffer_snapshot
        self._buffer_snapshot = value
        return value

    def _store_history(self, path: Path, payload: list[str]) -> None:
        with FileLock(path):
            atomic_write_json(path, payload)
        self._history = payload
        self._history_base = len(payload)

    def _drain_ready(self, first_timeout: float) -> list[str]:
        found: list[str] = []
        timeout = first_timeout
        while len(found) < _PASTE_LIMIT and _stdin_ready(self._stdin, timeout):
            extra = self._stdin.readline()
            if extra == "":
                break
            self._accept_paste(found, extra)
            timeout = 0.0
        if self._stdin is not sys.stdin:
            self._take_partial(found)
        return found

    def _prefixed(self, raw: str) -> str:
        prefix = self._stdin_prefix
        self._stdin_prefix = ""
        if prefix == "":
            return raw
        return prefix + raw

    def _take_partial(self, found: list[str]) -> None:
        raw = _scoop_partial_text(self._stdin)
        if raw == "":
            return
        if "\x1b[201~" in raw:
            self._accept_paste(found, raw)
            return
        self._stdin_prefix += raw

    def _accept_paste(self, found: list[str], raw: str) -> None:
        if raw == "" or len(found) >= _PASTE_LIMIT:
            return
        cleaned = _clean_line(raw)
        if cleaned == "" and "\n" not in raw and "\r" not in raw:
            return
        found.append(cleaned)
        self._remember(cleaned)

    def _remember(self, line: str) -> None:
        kept = _history_keep(line, self._skip_history)
        if self._history_max <= 0 or kept == "":
            return
        if self._history and self._history[-1] == kept:
            return
        self._history.append(kept)
        overflow = len(self._history) - self._history_max
        if overflow > 0:
            del self._history[:overflow]
            self._history_base = max(0, self._history_base - overflow)
        self._readline_add(kept)

    def _collect(self, text: str) -> list[str]:
        if self._words is None:
            return []
        if not self._completion_context(text).lstrip().startswith("/"):
            return []
        try:
            options = self._words()
        except Exception:
            return []
        found: list[str] = []
        seen: set[str] = set()
        for word in options:
            if not isinstance(word, str) or word in seen or not word.startswith(text):
                continue
            if contains_secret(word):
                continue
            seen.add(word)
            found.append(word)
        return found

    def _completion_context(self, text: str) -> str:
        get = self._get_buffer
        if get is None:
            return text
        try:
            live = get()
        except (OSError, RuntimeError):
            return text
        if live:
            return live
        return text

    def _arm_readline(self) -> None:
        if self._rl_ready or os.name != "posix":
            return
        try:
            self._install_readline()
        except ImportError:
            return
        self._replay_history()
        self._rl_ready = True

    def _install_readline(self) -> None:
        import readline

        _disable_auto_history()
        self._saved_completer = readline.get_completer()
        readline.set_completer(self.complete)
        self._completer_set = True
        libedit = readline_is_libedit(readline.__doc__)
        _bind(readline_tab_binding(libedit=libedit))
        with contextlib.suppress(AttributeError, OSError, ValueError):
            readline.set_completer_delims(" \t\n")
        with contextlib.suppress(AttributeError, OSError, ValueError):
            readline.set_history_length(max(self._history_max, 0))
        version = getattr(readline, "_READLINE_VERSION", None)
        number = version if isinstance(version, int) else None
        if bracketed_paste_supported(number, libedit=libedit):
            _bind("set enable-bracketed-paste on")
        self._add_history = readline.add_history
        self._clear_history = readline.clear_history
        self._get_buffer = readline.get_line_buffer

    def _replay_history(self) -> None:
        clear = self._clear_history
        add = self._add_history
        if clear is None or add is None:
            return
        clear()
        for line in self._history:
            add(_readline_flat(line))

    def _readline_add(self, line: str) -> None:
        if not self._rl_ready:
            return
        add = self._add_history
        if add is None:
            return
        with contextlib.suppress(ValueError, OSError):
            add(_readline_flat(line))

    def _detach_readline(self) -> None:
        if self._completer_set and os.name == "posix":
            _restore_completer(self._saved_completer)
        self._completer_set = False
        self._rl_ready = False
        self._add_history = None
        self._clear_history = None
        self._get_buffer = None


class _MessageBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.fence: str | None = None
        self.started = False
        self.pending: list[str] = []

    def feed(self, text: str) -> bool:
        self.started = True
        self.pending = []
        pieces = text.split("\n")
        pasted = len(pieces) > 1
        finished = False
        for index, piece in enumerate(pieces):
            last = index + 1 == len(pieces)
            finished = self._take(piece, may_finish=last or not pasted)
            if finished and not last:
                self.pending = pieces[index + 1 :]
                return True
        return finished

    def text(self) -> str:
        return "\n".join(self.parts)

    def _take(self, line: str, *, may_finish: bool) -> bool:
        if self.fence is not None:
            return self._take_fenced(line)
        if line in _FENCE_MARKS:
            self.fence = line
            return False
        continued = _continued_body(line)
        if continued is not None:
            self.parts.append(continued)
            return False
        self.parts.append(line)
        return may_finish

    def _take_fenced(self, line: str) -> bool:
        if line == self.fence:
            self.fence = None
            return True
        self.parts.append(line)
        return False


def unicode_stdout(stream: TextIO | None = None) -> bool:
    """True when ``stream`` (default stdout) can encode a non-ASCII glyph."""
    return detect_caps(stdout=stream).unicode


def _fit_composer(prompt: str, buffer: str, width: int) -> str:
    if width <= 0:
        return ""
    if display_width(prompt) >= width:
        return _take_prefix(prompt, width)
    room = width - display_width(prompt)
    if display_width(buffer) <= room:
        return prompt + buffer
    return prompt + _take_suffix(buffer, room)


def _cbreak_ready(stream: TextIO) -> bool:
    from fx1.interactive.tty_guard import cbreak_available

    return cbreak_available(stream)


def _read_guarded(
    reader: LineReader,
    caps: TermCaps,
    prompt: str,
    idle: Callable[[], None] | None,
    poll_s: float,
    *,
    polling: bool,
) -> str:
    if reader._should_edit(caps, polling):  # noqa: SLF001
        with _saved_termios(reader._stdin):  # noqa: SLF001
            return reader._read_recorded(  # noqa: SLF001
                caps, prompt, idle, poll_s, polling=polling
            )
    return reader._read_recorded(caps, prompt, idle, poll_s, polling=polling)  # noqa: SLF001


def _call_job(job_active: Callable[[], bool] | None) -> bool:
    if job_active is None:
        return False
    return bool(job_active())


def _can_use_tty(caps: TermCaps) -> bool:
    return os.name == "posix" and not caps.windows and caps.stdin_tty


def _inside_window(now: float, previous: float, window_s: float) -> bool:
    delta = now - previous
    return 0 <= delta <= window_s


def _continued_body(line: str) -> str | None:
    marks = len(line) - len(line.rstrip("\\"))
    if marks == 0 or marks % 2 == 0:
        return None
    return line[:-1]


def _clean_line(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    for marker in _PASTE_MARKERS:
        if marker in text:
            text = text.replace(marker, "")
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _read_stream_line(stream: TextIO) -> str:
    line = stream.readline()
    if line == "":
        raise EOFError
    return line


def _stream_isatty(stream: object) -> bool:
    fn = getattr(stream, "isatty", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except (OSError, ValueError):
        return False


def _stream_fd(stream: TextIO) -> int | None:
    fileno = getattr(stream, "fileno", None)
    if not callable(fileno):
        return None
    try:
        fd = fileno()
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(fd, int) and fd >= 0:
        return fd
    return None


def _color_enabled(env: Mapping[str, str], *, stdout_tty: bool, term: str) -> bool:
    if "NO_COLOR" in env:
        return False
    if _force_color(env.get("FORCE_COLOR")):
        return True
    return stdout_tty and term != "dumb"


def _force_color(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _unicode_enabled(env: Mapping[str, str], stdout: TextIO) -> bool:
    forced = _pythonio_encoding(env)
    if forced is not None:
        return _is_utf(forced)
    encoding = getattr(stdout, "encoding", None)
    if isinstance(encoding, str) and encoding != "":
        return _is_utf(encoding)
    return _locale_is_utf()


def _pythonio_encoding(env: Mapping[str, str]) -> str | None:
    raw = env.get("PYTHONIOENCODING")
    if raw is None:
        return None
    token = raw.split(":", 1)[0].strip()
    if token == "":
        return None
    return token


def _is_utf(encoding: str) -> bool:
    compact = encoding.lower().replace("-", "").replace("_", "")
    return compact.startswith("utf")


def _locale_is_utf() -> bool:
    try:
        encoding = locale.getpreferredencoding(False)
    except (LookupError, ValueError):
        return False
    return _is_utf(encoding)


def _terminal_size(size_fn: Callable[[], os.terminal_size]) -> tuple[int, int]:
    try:
        size = size_fn()
    except (OSError, ValueError, AttributeError, TypeError):
        return _WIDTH_FALLBACK, _HEIGHT_FALLBACK
    pair = _size_pair(size)
    if pair is None or pair[0] <= 0 or pair[1] <= 0:
        return _WIDTH_FALLBACK, _HEIGHT_FALLBACK
    return _clamp(pair[0], _WIDTH_MIN, _WIDTH_MAX), max(1, pair[1])


def _size_pair(size: object) -> tuple[int, int] | None:
    columns = getattr(size, "columns", None)
    lines = getattr(size, "lines", None)
    if isinstance(columns, int) and isinstance(lines, int):
        return columns, lines
    if isinstance(size, tuple) and len(size) >= 2:
        raw_columns, raw_lines = size[0], size[1]
        if isinstance(raw_columns, int) and isinstance(raw_lines, int):
            return raw_columns, raw_lines
    return None


def _clamp(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def _consume_line(
    line: str,
    fence: str | None,
    width: int,
    indent: str,
    subsequent: str,
) -> tuple[str | None, list[str]]:
    if fence is None:
        opener = _opens_fence(line)
        if opener is not None:
            return opener, [line]
    if fence is not None:
        if line.strip() == fence:
            return None, [line]
        return fence, [line]
    if line.strip() == "":
        return None, [""]
    return None, _wrap_line(line, width, indent, subsequent)


def _opens_fence(line: str) -> str | None:
    stripped = line.strip()
    if stripped == '"""':
        return '"""'
    if _FENCE_OPEN_RE.fullmatch(stripped):
        return "```"
    return None


def _wrap_line(line: str, width: int, indent: str, subsequent: str) -> list[str]:
    words = line.split()
    if not words:
        return [""]
    lines: list[str] = []
    pending = words
    first = True
    while pending:
        prefix = indent if first else subsequent
        taken, pending = _take_row(pending, max(1, width - display_width(prefix)))
        if not taken:
            break
        lines.append(prefix + " ".join(taken))
        first = False
    return lines


def _take_row(words: list[str], budget: int) -> tuple[list[str], list[str]]:
    if budget < 1:
        budget = 1
    row: list[str] = []
    used = 0
    index = 0
    while index < len(words):
        word = words[index]
        word_w = display_width(word)
        needed = word_w if not row else word_w + 1
        if needed <= budget - used:
            row.append(word)
            used += needed
            index += 1
            continue
        if row:
            return row, words[index:]
        return _split_overflow(word, budget, words[index + 1 :])
    return row, []


def _split_overflow(word: str, budget: int, tail: list[str]) -> tuple[list[str], list[str]]:
    piece = _take_prefix(word, budget)
    if piece == "":
        piece = word[:1]
    rest = word[len(piece) :]
    if rest:
        return [piece], [rest, *tail]
    return [piece], tail


def _scaled(value: int, scale: int) -> str:
    text = f"{value / scale:.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def _columns_of(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    return [str(key) for key in rows[0]]


def _column_widths(cols: Sequence[str], rows: Sequence[Mapping[str, str]], width: int) -> list[int]:
    natural = [_natural_width(col, rows) for col in cols]
    gaps = 2 * (len(cols) - 1)
    usable = width - gaps
    if usable < len(cols):
        usable = len(cols)
    if sum(natural) <= usable:
        return natural
    return _shrink_widths(natural, usable)


def _cell_text(row: Mapping[str, str], col: str) -> str:
    value: object = row.get(col, "")
    if not isinstance(value, str):
        return ""
    return sanitize_for_terminal(value)


def _natural_width(col: str, rows: Sequence[Mapping[str, str]]) -> int:
    longest = display_width(sanitize_for_terminal(col))
    for row in rows:
        for part in _cell_text(row, col).split("\n"):
            longest = max(longest, display_width(part))
    return max(longest, 1)


def _shrink_widths(natural: list[int], usable: int) -> list[int]:
    count = len(natural)
    floor = 3 if usable >= count * 3 else 1
    extra = usable - floor * count
    if extra < 0:
        floor = 1
        extra = max(0, usable - count)
    weights = [max(1, width) for width in natural]
    return _spread(count, floor, extra, weights)


def _spread(count: int, floor: int, extra: int, weights: list[int]) -> list[int]:
    widths = [floor] * count
    if extra <= 0:
        return widths
    total = sum(weights) or 1
    raw = [extra * weight / total for weight in weights]
    whole = [int(part) for part in raw]
    for index, part in enumerate(whole):
        widths[index] += part
    _give_remainder(widths, extra - sum(whole), raw, weights)
    return widths


def _give_remainder(widths: list[int], left: int, raw: list[float], weights: list[int]) -> None:
    if left <= 0:
        return
    order = sorted(
        range(len(widths)),
        key=lambda index: (raw[index] - int(raw[index]), weights[index]),
        reverse=True,
    )
    for index in order[:left]:
        widths[index] += 1


def _body_lines(cols: Sequence[str], row: Mapping[str, str], widths: Sequence[int]) -> list[str]:
    blocks = [
        _cell_lines(_cell_text(row, col), width) for col, width in zip(cols, widths, strict=True)
    ]
    height = max(len(block) for block in blocks)
    lines: list[str] = []
    for index in range(height):
        cells = [
            _fit(block[index] if index < len(block) else "", width)
            for block, width in zip(blocks, widths, strict=True)
        ]
        lines.append(_join_cells(cells))
    return lines


def _cell_lines(text: str, width: int) -> list[str]:
    if text == "":
        return [""]
    lines: list[str] = []
    for part in text.split("\n"):
        if part.strip() == "":
            lines.append("")
            continue
        lines.extend(_wrap_line(part, width, "", ""))
    return lines or [""]


def _char_width(ch: str) -> int:
    if ch == "\t":
        return 1
    if unicodedata.combining(ch):
        return 0
    code = ord(ch)
    if code < 32 or 0x7F <= code <= 0x9F:
        return 0
    if code in _ZERO_WIDTH_CHARS:
        return 0
    if unicodedata.east_asian_width(ch) in {"W", "F"}:
        return 2
    return 1


def _take_prefix(text: str, width: int) -> str:
    if width <= 0:
        return ""
    used = 0
    out: list[str] = []
    for ch in text:
        needed = _char_width(ch)
        if used + needed > width:
            break
        out.append(ch)
        used += needed
    return "".join(out)


def _take_suffix(text: str, width: int) -> str:
    if width <= 0:
        return ""
    used = 0
    chars: list[str] = []
    for ch in reversed(text):
        needed = _char_width(ch)
        if used + needed > width:
            break
        chars.append(ch)
        used += needed
    chars.reverse()
    return "".join(chars)


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    current = display_width(text)
    if current <= width:
        return text + (" " * (width - current))
    return _clip_end(text, width)


def _clip_end(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    if width == 1:
        return "…"
    return _take_prefix(text, width - 1) + "…"


def _join_cells(cells: Iterable[str]) -> str:
    return "  ".join(cells)


def _clip_history(
    items: Iterable[object],
    *,
    limit: int,
    skip: Callable[[str], bool],
) -> list[str]:
    kept: list[str] = []
    if limit <= 0:
        return kept
    for item in items:
        if not isinstance(item, str) or item == "" or skip(item):
            continue
        if kept and kept[-1] == item:
            continue
        kept.append(item)
    if len(kept) > limit:
        return kept[-limit:]
    return kept


def _load_history_file(
    path: Path | None,
    *,
    limit: int,
    skip: Callable[[str], bool],
) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        payload = read_json(path)
    except (OSError, StoreCorruptError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return _clip_history(payload, limit=limit, skip=skip)


def _scoop_partial_text(stream: TextIO) -> str:
    """Read a partial tty line that cooked ``readline`` will not deliver.

    Bracketed paste often ends with ``ESC[201~`` and no trailing newline.
    ``BufferedReader.peek`` is not used here: on a canonical tty it blocks
    until a full line exists.
    """
    if os.name != "posix":
        return ""
    data = _scoop_partial_bytes(stream)
    if data == b"":
        return ""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return data.decode(encoding, errors="replace")


def _scoop_partial_bytes(stream: TextIO) -> bytes:
    fd = _stream_fd(stream)
    if fd is None:
        return b""
    try:
        import termios
    except ImportError:
        return b""
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return b""
    return _read_noncanonical(fd, saved)


def _read_noncanonical(fd: int, saved: Any) -> bytes:
    import termios

    updated = termios.tcgetattr(fd)
    updated[3] = int(updated[3]) & ~termios.ICANON
    cc = list(updated[6])
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    updated[6] = cc
    try:
        termios.tcsetattr(fd, termios.TCSANOW, updated)
        return _nonblocking_read(fd)
    finally:
        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSANOW, saved)


def _nonblocking_read(fd: int) -> bytes:
    import fcntl

    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        return os.read(fd, 65536)
    except BlockingIOError:
        return b""
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)


def _stdin_ready(stream: TextIO, timeout: float) -> bool:
    if os.name != "posix":
        return False
    fd = _stream_fd(stream)
    if fd is None:
        return False
    import select

    ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
    return bool(ready)


@contextlib.contextmanager
def _saved_termios(stream: TextIO) -> Iterator[None]:
    """Restore termios on the way out, including exceptions and signals that raise."""
    if os.name != "posix":
        yield
        return
    fd = _stream_fd(stream)
    if fd is None:
        yield
        return
    try:
        import termios
    except ImportError:
        yield
        return
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _disable_auto_history() -> None:
    import readline

    try:
        readline.set_auto_history(False)
    except AttributeError:
        return


def _bind(command: str) -> None:
    import readline

    try:
        readline.parse_and_bind(command)
    except (AttributeError, OSError, ValueError):
        return


def _restore_completer(previous: Completer | None) -> None:
    try:
        import readline
    except ImportError:
        return
    readline.set_completer(previous)


__all__ = [
    "ConsoleWriter",
    "InputCancelled",
    "InterruptTracker",
    "LineReader",
    "Style",
    "TermCaps",
    "bracketed_paste_supported",
    "detect_caps",
    "display_width",
    "format_count",
    "format_duration",
    "plain_requested",
    "unicode_stdout",
    "readline_is_libedit",
    "readline_tab_binding",
    "render_table",
    "sanitize_for_terminal",
    "stream_lock",
    "truncate_middle",
    "wrap_text",
]
