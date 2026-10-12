"""Terminal line I/O for the dipcatcher console.

POSIX TTYs are polled with ``select`` so a background job can paint progress
while a line is still being typed. Windows consoles are polled with
``msvcrt``. Pipes and files use a blocking read. Input history lives under
the fx-1 config directory and never retains credentials.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import re
import select
import shutil
import stat
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO, cast

from fx1.interactive.profiles import config_dir

_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
_DIR_MODE = stat.S_IRWXU
_WINDOWS_POLL_S = 0.02
_HISTORY_NAME = "history"
_MOONSHOT_MARK = "MOONSHOT_API_KEY="
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_PREFIX_RE = re.compile(r"\b(?:sk-|sk_|key-)\S{8,}")
_BLOB_RE = re.compile(r"[A-Za-z0-9]{32,}")

__all__ = [
    "History",
    "InterruptAction",
    "InterruptPolicy",
    "LineReader",
    "history_path",
    "shorten",
    "supports_color",
    "terminal_width",
    "wrap_block",
]


class InterruptAction(StrEnum):
    """What one Ctrl-C means under :class:`InterruptPolicy`."""

    CANCEL_JOB = "cancel_job"
    CLEAR_LINE = "clear_line"
    EXIT = "exit"


class InterruptPolicy:
    """Codex-style Ctrl-C.

    The first press cancels the running background job, or just clears the
    line when idle. A second press within ``exit_window_s`` exits. Submitting
    a normal line resets the window (call :meth:`reset`).
    """

    def __init__(
        self,
        *,
        exit_window_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._exit_window_s = exit_window_s
        self._clock = clock
        self._armed_at: float | None = None

    def on_interrupt(self, *, job_running: bool) -> InterruptAction:
        now = self._clock()
        if _within_window(self._armed_at, now, self._exit_window_s):
            self._armed_at = None
            return InterruptAction.EXIT
        self._armed_at = now
        if job_running:
            return InterruptAction.CANCEL_JOB
        return InterruptAction.CLEAR_LINE

    def reset(self) -> None:
        self._armed_at = None

    def hint(self, action: InterruptAction) -> str:
        window = _format_window(self._exit_window_s)
        if action is InterruptAction.CANCEL_JOB:
            return f"cancel requested — press Ctrl-C again within {window} to exit"
        if action is InterruptAction.CLEAR_LINE:
            return f"press Ctrl-C again within {window} to exit, or type /exit"
        return f"press Ctrl-C again within {window} to exit"


def history_path() -> Path:
    """``$FX1_CONFIG_DIR/history``, or ``~/.fx1/history`` when unset."""
    return config_dir() / _HISTORY_NAME


class History:
    """Persistent input history.

    At most ``max_entries`` lines, mode 0600, rewritten atomically on
    :meth:`save`. Never stores empty lines, consecutive duplicates, lines
    starting with ``/keys``, or lines matching secret patterns (``Bearer``
    plus a token, ``sk-`` / ``sk_`` / ``key-`` plus 8 characters,
    ``MOONSHOT_API_KEY=``, or a 32+ character alphanumeric blob that contains
    a digit). ``errors`` records :meth:`save` failures; save never raises.
    """

    def __init__(self, path: Path | None = None, *, max_entries: int = 1000) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        chosen = history_path() if path is None else path
        self._path = chosen.expanduser()
        self._max_entries = max_entries
        self._entries: list[str] = []
        self.errors: list[str] = []

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[str]:
        """Read the file if present. Missing or unreadable files yield ``[]``.

        The same filters as :meth:`append` apply, then the last
        ``max_entries`` lines are kept.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            self._entries = []
            return []
        self._entries = _trim(_collect(raw), self._max_entries)
        return self.entries()

    def load_into_readline(self) -> bool:
        """Push :meth:`entries` into ``readline``. Never raises.

        True when ``readline`` imports and every entry was handed to
        ``add_history``. False when ``readline`` is missing (Windows) or the
        import fails.
        """
        add_history = _readline_add_history()
        if add_history is None:
            return False
        try:
            for entry in self._entries:
                add_history(entry)
        except Exception:
            return False
        return True

    def append(self, line: str) -> None:
        previous = self._entries[-1] if self._entries else None
        accepted = _accept(line, previous)
        if accepted is None:
            return
        self._entries.append(accepted)
        self._entries = _trim(self._entries, self._max_entries)

    def save(self) -> None:
        """Atomically rewrite the history file. Failures land in ``errors``."""
        temporary: str | None = None
        try:
            temporary = self._write_temp()
            os.replace(temporary, self._path)
            temporary = None
            _chmod_quiet(self._path, _FILE_MODE)
        except OSError as exc:
            self.errors.append(_os_message(exc))
        finally:
            _unlink_quiet(temporary)

    def entries(self) -> list[str]:
        return list(self._entries)

    @staticmethod
    def is_sensitive(line: str) -> bool:
        return _line_is_sensitive(line)

    def _write_temp(self) -> str:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        _chmod_quiet(parent, _DIR_MODE)
        fd, temporary = tempfile.mkstemp(dir=parent, prefix=".history-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                _write_entries(handle, self._entries)
        except OSError:
            _unlink_quiet(temporary)
            raise
        _chmod_quiet(temporary, _FILE_MODE)
        return temporary


class LineReader:
    """Prompt-and-poll line reader.

    ``read(prompt, timeout_s)`` returns the submitted line without its
    trailing newline, ``None`` when the poll times out (render progress and
    call ``read`` again with an empty prompt), and raises ``EOFError`` at
    end of input. ``KeyboardInterrupt`` propagates.

    POSIX TTY: ``select`` on stdin, then ``readline``. Windows TTY:
    ``msvcrt.kbhit`` / ``getwch``, assembling a line and keeping a partial
    buffer across timeouts. Anything else: blocking ``input_fn(prompt)``
    (the timeout is ignored).
    """

    def __init__(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        platform: str | None = None,
        input_fn: Callable[[str], str] = input,
        select_fn: Callable[..., Any] | None = None,
        kbhit: Callable[[], bool] | None = None,
        getwch: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._stdin: TextIO = sys.stdin if stdin is None else stdin
        self._stdout: TextIO = sys.stdout if stdout is None else stdout
        self._platform = sys.platform if platform is None else platform
        self._input_fn = input_fn
        self._select_fn = select_fn
        self._kbhit = kbhit
        self._getwch = getwch
        self._clock = clock
        self._sleep = sleep
        self._pending = ""

    @property
    def interactive(self) -> bool:
        return _stream_isatty(self._stdin) and _stream_isatty(self._stdout)

    @property
    def mode(self) -> str:
        # Polling needs a terminal on stdin only: a TTY whose output is piped
        # (``dipcatcher | tee``) still renders background progress between
        # keystrokes, matching the pre-LineReader console behaviour.
        if not _stream_isatty(self._stdin):
            return "blocking"
        if self._platform == "win32":
            return "windows-tty"
        return "posix-tty"

    def read(self, prompt: str, timeout_s: float) -> str | None:
        kind = self.mode
        if kind == "blocking":
            return self._input_fn(prompt)
        if kind == "windows-tty":
            return self._read_windows(prompt, timeout_s)
        return self._read_posix(prompt, timeout_s)

    def _read_posix(self, prompt: str, timeout_s: float) -> str | None:
        self._stdout.write(prompt)
        self._stdout.flush()
        if not _select_ready(self._selector(), self._stdin, timeout_s):
            return None
        line = self._stdin.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")

    def _read_windows(self, prompt: str, timeout_s: float) -> str | None:
        _write_prompt(self._stdout, prompt)
        kbhit, getwch = self._windows_getters()
        deadline = self._clock() + timeout_s
        while self._clock() <= deadline:
            if kbhit():
                if _apply_key(self, getwch()):
                    return _take_pending(self)
            else:
                self._sleep(_WINDOWS_POLL_S)
        return None

    def _selector(self) -> Callable[..., Any]:
        if self._select_fn is not None:
            return self._select_fn
        return select.select

    def _windows_getters(self) -> tuple[Callable[[], bool], Callable[[], str]]:
        kbhit = self._kbhit
        getwch = self._getwch
        if kbhit is not None and getwch is not None:
            return kbhit, getwch
        loaded_kbhit, loaded_getwch = _load_msvcrt()
        if kbhit is None:
            kbhit = loaded_kbhit
            self._kbhit = kbhit
        if getwch is None:
            getwch = loaded_getwch
            self._getwch = getwch
        return kbhit, getwch


def terminal_width(default: int = 80) -> int:
    """Columns from the terminal, clamped to ``[40, 200]``."""
    columns = shutil.get_terminal_size((default, 24)).columns
    return _clamp(columns, 40, 200)


def wrap_block(text: str, *, width: int | None = None, indent: str = "") -> str:
    """Soft-wrap prose to ``width`` (default :func:`terminal_width`).

    Blank lines, lines that start with four spaces or a tab, fenced code
    regions (triple backticks), table rows that start with ``|``, and lines
    that already fit are not reflowed. Every output line is prefixed with
    ``indent``. Long words stay intact (``break_long_words`` and
    ``break_on_hyphens`` are off).
    """
    columns = terminal_width() if width is None else max(width, 1)
    return "\n".join(_wrap_lines(text, columns, indent))


def supports_color(
    stream: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """True when ``stream`` is a TTY and the environment allows color.

    False when ``NO_COLOR`` is set to any value, ``TERM`` is ``dumb``, or
    the stream is not a TTY.
    """
    env = os.environ if environ is None else environ
    if "NO_COLOR" in env or env.get("TERM") == "dumb":
        return False
    target = sys.stdout if stream is None else stream
    return _stream_isatty(target)


def shorten(text: str, limit: int) -> str:
    """Collapse to one line and ellipsize to ``limit`` characters.

    The ellipsis is the single character ``…`` and counts toward ``limit``.
    """
    flat = _single_line(text)
    if limit <= 0:
        return ""
    if len(flat) <= limit:
        return flat
    if limit == 1:
        return "…"
    return flat[: limit - 1] + "…"


def _within_window(armed_at: float | None, now: float, window_s: float) -> bool:
    if armed_at is None:
        return False
    return (now - armed_at) <= window_s


def _format_window(seconds: float) -> str:
    if seconds.is_integer():
        return f"{int(seconds)}s"
    rendered = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return f"{rendered}s"


def _line_is_sensitive(line: str) -> bool:
    if _BEARER_RE.search(line) or _PREFIX_RE.search(line):
        return True
    if _MOONSHOT_MARK in line:
        return True
    return _has_secret_blob(line)


def _has_secret_blob(line: str) -> bool:
    return any(any(char.isdigit() for char in match.group()) for match in _BLOB_RE.finditer(line))


def _accept(line: str, previous: str | None) -> str | None:
    text = _normalize(line)
    if _rejected(text, previous):
        return None
    return text


def _normalize(line: str) -> str:
    flat = line.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return flat.strip()


def _rejected(text: str, previous: str | None) -> bool:
    if not text or text.startswith("/keys") or _line_is_sensitive(text):
        return True
    return previous is not None and text == previous


def _collect(raw: str) -> list[str]:
    kept: list[str] = []
    for line in raw.splitlines():
        previous = kept[-1] if kept else None
        accepted = _accept(line, previous)
        if accepted is not None:
            kept.append(accepted)
    return kept


def _trim(lines: list[str], max_entries: int) -> list[str]:
    if max_entries < 1:
        return []
    if len(lines) <= max_entries:
        return lines
    return lines[-max_entries:]


def _write_entries(handle: TextIO, entries: list[str]) -> None:
    if not entries:
        return
    handle.write("\n".join(entries))
    handle.write("\n")


def _chmod_quiet(path: Path | str, mode: int) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, mode)


def _unlink_quiet(path: str | None) -> None:
    if not path:
        return
    with contextlib.suppress(OSError):
        os.unlink(path)


def _os_message(exc: OSError) -> str:
    message = str(exc)
    if message:
        return message
    return exc.__class__.__name__


def _readline_add_history() -> Callable[[str], object] | None:
    module = _import_readline()
    if module is None:
        return None
    add_history = getattr(module, "add_history", None)
    if not callable(add_history):
        return None
    return cast(Callable[[str], object], add_history)


def _import_readline() -> object | None:
    if _readline_blocked():
        return None
    try:
        return _load_readline()
    except Exception:
        return None


def _load_readline() -> object:
    import readline

    return readline


def _readline_blocked() -> bool:
    if "readline" not in sys.modules:
        return False
    cached: object = sys.modules["readline"]
    return cached is None


def _select_ready(selector: Callable[..., Any], stdin: TextIO, timeout_s: float) -> bool:
    ready, _, _ = selector([stdin], [], [], timeout_s)
    return bool(ready)


def _write_prompt(stdout: TextIO, prompt: str) -> None:
    if not prompt:
        return
    stdout.write(prompt)
    stdout.flush()


def _apply_key(reader: LineReader, ch: str) -> bool:
    """Apply one Windows console key. True means Enter submitted the line."""
    if ch in {"\r", "\n"}:
        reader._stdout.write("\n")
        reader._stdout.flush()
        return True
    if ch in {"\x08", "\x7f"}:
        _backspace(reader)
        return False
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in {"\x1a", ""}:
        raise EOFError
    reader._pending += ch
    reader._stdout.write(ch)
    reader._stdout.flush()
    return False


def _backspace(reader: LineReader) -> None:
    if not reader._pending:
        return
    reader._pending = reader._pending[:-1]
    reader._stdout.write("\b \b")
    reader._stdout.flush()


def _take_pending(reader: LineReader) -> str:
    line = reader._pending
    reader._pending = ""
    return line


def _load_msvcrt() -> tuple[Callable[[], bool], Callable[[], str]]:
    try:
        module = importlib.import_module("msvcrt")
    except ImportError as exc:
        raise RuntimeError("windows console input requires msvcrt") from exc
    kbhit_raw = getattr(module, "kbhit", None)
    getwch_raw = getattr(module, "getwch", None)
    if not callable(kbhit_raw) or not callable(getwch_raw):
        raise RuntimeError("msvcrt console input is unavailable")
    return _adapt_msvcrt(
        cast(Callable[..., object], kbhit_raw),
        cast(Callable[..., object], getwch_raw),
    )


def _adapt_msvcrt(
    kbhit_raw: Callable[..., object],
    getwch_raw: Callable[..., object],
) -> tuple[Callable[[], bool], Callable[[], str]]:
    def kbhit() -> bool:
        return bool(kbhit_raw())

    def getwch() -> str:
        value = getwch_raw()
        if isinstance(value, str):
            return value
        return str(value)

    return kbhit, getwch


def _stream_isatty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (OSError, ValueError):
        return False


def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _single_line(text: str) -> str:
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _line_is_literal(line: str) -> bool:
    return line == "" or line.startswith(("\t", "    ", "|"))


def _wrap_lines(text: str, columns: int, indent: str) -> list[str]:
    output: list[str] = []
    fenced = False
    for line in text.splitlines():
        fenced = _emit_line(output, line, columns, indent, fenced=fenced)
    return output


def _emit_line(
    output: list[str],
    line: str,
    columns: int,
    indent: str,
    *,
    fenced: bool,
) -> bool:
    opener = line.startswith("```")
    literal = fenced or opener or _line_is_literal(line) or len(line) <= columns
    if literal:
        output.append(indent + line)
    else:
        output.extend(_wrap_prose(line, columns, indent))
    if opener:
        return not fenced
    return fenced


def _wrap_prose(line: str, columns: int, indent: str) -> list[str]:
    parts = textwrap.wrap(
        line,
        width=columns,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not parts:
        return [indent + line]
    return [indent + part for part in parts]
