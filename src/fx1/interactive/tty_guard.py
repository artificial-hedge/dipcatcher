"""TTY ownership while a line is being read.

The signal handler only flips a flag and restores termios. It must not take
the writer lock: the handler runs on the main thread, and that lock may
already be held. SIGTSTP is left alone so Ctrl-Z still suspends.
"""

from __future__ import annotations

import contextlib
import contextvars
import errno
import os
import signal
from collections.abc import Callable, Iterator
from typing import Any, TextIO

_PASTE_ON = b"\x1b[200~"
_PASTE_OFF = b"\x1b[201~"
_BRACKET_OFF = b"\x1b[?2004l"
_PASTE_CAP = 200_000
_RESIZE: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "dipcatcher_tty_resize", default=None
)


def cbreak_available(stream: TextIO) -> bool:
    """True when ``stream`` is a POSIX tty we can put into cbreak."""
    if os.name != "posix":
        return False
    fd = _fd(stream)
    if fd is None:
        return False
    try:
        import termios
    except ImportError:
        return False
    try:
        termios.tcgetattr(fd)
    except termios.error:
        return False
    return True


def disable_bracketed_paste(stream: TextIO) -> None:
    """Turn bracketed paste off. No-op when ``stream`` is not a POSIX tty."""
    fd = _tty_fd(stream)
    if fd is None:
        return
    with contextlib.suppress(OSError):
        os.write(fd, _BRACKET_OFF)


def resize_requested() -> bool:
    """True once after SIGWINCH, then clear the flag."""
    flag = _RESIZE.get()
    if flag is None or not flag[0]:
        return False
    flag[0] = False
    return True


@contextlib.contextmanager
def guard_tty(stdin: TextIO, stdout: TextIO) -> Iterator[None]:
    """Restore termios and paste mode on SIGTERM and SIGQUIT."""
    fd = _tty_fd(stdin)
    out_fd = _tty_fd(stdout)
    if fd is None:
        yield
        return
    flag = [False]
    token = _RESIZE.set(flag)
    guard = _SignalGuard(fd, out_fd, flag)
    try:
        guard.install()
        yield
    finally:
        guard.restore_handlers()
        _RESIZE.reset(token)


def read_cbreak(
    stdin: TextIO,
    writer: Any,
    prompt: str,
    idle: Callable[[], None] | None,
    poll_s: float,
) -> str:
    """Read one line with echo off so a status redraw can keep the buffer.

    Ctrl-C still raises ``KeyboardInterrupt`` (ISIG stays on). Ctrl-D on an
    empty buffer raises ``EOFError``.
    """
    fd = _fd(stdin)
    if fd is None:
        raise OSError("cbreak requires a tty")
    editor = _Editor()
    writer.set_composer_owned(True)
    writer.set_prompt_visible(prompt, lambda: editor.text)
    try:
        with _cbreak(fd):
            writer.write_prompt(prompt)
            _pump(fd, writer, editor, idle, poll_s)
    finally:
        writer.release_composer()
    if editor.eof:
        raise EOFError
    _commit_newline(writer)
    return editor.text


class _SignalGuard:
    def __init__(self, fd: int, out_fd: int | None, flag: list[bool]) -> None:
        self._fd = fd
        self._out_fd = out_fd
        self._flag = flag
        self._saved: list[Any] | None = None
        self._prev: list[tuple[int, Any]] = []

    def install(self) -> None:
        import termios

        with contextlib.suppress(termios.error):
            self._saved = termios.tcgetattr(self._fd)
        for name in ("SIGTERM", "SIGQUIT", "SIGHUP", "SIGWINCH"):
            self._hook(name)

    def restore_handlers(self) -> None:
        for sig, previous in self._prev:
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, previous)
        self._prev.clear()

    def _hook(self, name: str) -> None:
        sig = getattr(signal, name, None)
        if not isinstance(sig, int):
            return
        handler = self._on_winch if name == "SIGWINCH" else self._on_fatal
        try:
            previous = signal.signal(sig, handler)
        except (OSError, ValueError):
            return
        self._prev.append((sig, previous))

    def _on_winch(self, signum: int, frame: object) -> None:
        self._flag[0] = True
        _chain(self._previous(signum), signum, frame)

    def _on_fatal(self, signum: int, frame: object) -> None:
        self._restore_termios()
        self._paste_off()
        _chain(self._previous(signum), signum, frame)
        raise SystemExit(128 + signum)

    def _previous(self, signum: int) -> Any:
        for sig, previous in self._prev:
            if sig == signum:
                return previous
        return None

    def _restore_termios(self) -> None:
        saved = self._saved
        if saved is None:
            return
        import termios

        with contextlib.suppress(termios.error):
            termios.tcsetattr(self._fd, termios.TCSANOW, saved)

    def _paste_off(self) -> None:
        out = self._out_fd if self._out_fd is not None else self._fd
        with contextlib.suppress(OSError):
            os.write(out, _BRACKET_OFF)


class _Editor:
    def __init__(self) -> None:
        self.text = ""
        self.eof = False
        self.done = False
        self._raw = bytearray()
        self._paste = False

    def feed(self, data: bytes) -> None:
        self._raw.extend(data)
        self._drain()

    def _drain(self) -> None:
        while self._raw and not self.done:
            if not self._step():
                return

    def _step(self) -> bool:
        if self._paste:
            return self._take_paste()
        if self._raw[0] == 0x1B:
            return self._take_escape()
        text, used = _one_char(self._raw)
        if used == 0:
            return False
        del self._raw[:used]
        self._take_char(text)
        return True

    def _take_char(self, text: str) -> None:
        if text in {"\n", "\r"}:
            self.done = True
            return
        if text in {"\x7f", "\x08"}:
            self.text = _drop_last(self.text)
            return
        if text == "\x04":
            self._ctrl_d()
            return
        if text == "\x15":
            self.text = ""
            return
        if text == "\x17":
            self.text = _drop_word(self.text)
            return
        if text < " " or text == "\x7f":
            return
        self.text += text

    def _ctrl_d(self) -> None:
        if self.text == "":
            self.eof = True
            self.done = True

    def _take_escape(self) -> bool:
        if self._raw.startswith(_PASTE_ON):
            del self._raw[: len(_PASTE_ON)]
            self._paste = True
            return True
        if self._raw.startswith(_PASTE_OFF):
            del self._raw[: len(_PASTE_OFF)]
            return True
        if len(self._raw) >= 2 and self._raw[1] == ord("["):
            return self._take_csi()
        if len(self._raw) < 2:
            return False
        del self._raw[:2]
        return True

    def _take_csi(self) -> bool:
        end = _csi_end(self._raw)
        if end is None:
            return False
        del self._raw[:end]
        return True

    def _take_paste(self) -> bool:
        end = self._raw.find(_PASTE_OFF)
        if end < 0:
            return self._paste_overflow()
        chunk = bytes(self._raw[:end])
        del self._raw[: end + len(_PASTE_OFF)]
        self._paste = False
        self._insert_paste(chunk)
        return True

    def _paste_overflow(self) -> bool:
        if len(self._raw) < _PASTE_CAP:
            return False
        chunk = bytes(self._raw)
        self._raw.clear()
        self._paste = False
        self._insert_paste(chunk)
        return True

    def _insert_paste(self, data: bytes) -> None:
        text = data.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        head, sep, _tail = text.partition("\n")
        self.text += head
        if sep:
            self.done = True


def _pump(
    fd: int,
    writer: Any,
    editor: _Editor,
    idle: Callable[[], None] | None,
    poll_s: float,
) -> None:
    timeout = poll_s if poll_s > 0 else 0.0
    while not editor.done:
        if resize_requested():
            _repaint(writer)
        if not _ready(fd, timeout):
            _idle(idle)
            continue
        chunk = _read_some(fd)
        if chunk is None:
            continue
        if chunk == b"":
            editor.eof = True
            editor.done = True
            return
        editor.feed(chunk)
        if not editor.done:
            writer.redraw_composer()


def _repaint(writer: Any) -> None:
    refresh = getattr(writer, "refresh_caps", None)
    if callable(refresh):
        refresh()
    repaint = getattr(writer, "repaint", None)
    if callable(repaint):
        repaint()


def _idle(idle: Callable[[], None] | None) -> None:
    if idle is not None:
        idle()


def _read_some(fd: int) -> bytes | None:
    try:
        return os.read(fd, 4096)
    except OSError as exc:
        if exc.errno == errno.EINTR:
            return None
        raise


def _commit_newline(writer: Any) -> None:
    stream = getattr(writer, "out", None)
    fd = _fd(stream) if stream is not None else None
    if fd is None:
        return
    with contextlib.suppress(OSError):
        os.write(fd, b"\r\n")


@contextlib.contextmanager
def _cbreak(fd: int) -> Iterator[None]:
    import termios

    saved = termios.tcgetattr(fd)
    updated = termios.tcgetattr(fd)
    updated[3] = int(updated[3]) & ~(termios.ICANON | termios.ECHO)
    cc = list(updated[6])
    cc[termios.VMIN] = 1
    cc[termios.VTIME] = 0
    updated[6] = cc
    termios.tcsetattr(fd, termios.TCSANOW, updated)
    try:
        yield
    finally:
        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _chain(previous: Any, signum: int, frame: object) -> None:
    if previous in (None, signal.SIG_DFL, signal.SIG_IGN):
        return
    if callable(previous):
        previous(signum, frame)


def _csi_end(raw: bytearray) -> int | None:
    for index in range(2, len(raw)):
        if 0x40 <= raw[index] <= 0x7E:
            return index + 1
    if len(raw) > 32:
        return 1
    return None


def _one_char(raw: bytearray) -> tuple[str, int]:
    limit = min(4, len(raw))
    for size in range(1, limit + 1):
        try:
            text = bytes(raw[:size]).decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text:
            return text, size
    if len(raw) >= 4:
        return bytes(raw[:1]).decode("utf-8", errors="replace"), 1
    return "", 0


def _drop_last(text: str) -> str:
    if text == "":
        return ""
    return text[:-1]


def _drop_word(text: str) -> str:
    trimmed = text.rstrip()
    if trimmed == "":
        return ""
    pieces = trimmed.split(" ")
    pieces.pop()
    return " ".join(pieces)


def _ready(fd: int, timeout: float) -> bool:
    import select

    ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
    return bool(ready)


def _tty_fd(stream: TextIO) -> int | None:
    if not _isatty(stream):
        return None
    return _fd(stream)


def _isatty(stream: object) -> bool:
    fn = getattr(stream, "isatty", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except (OSError, ValueError):
        return False


def _fd(stream: object) -> int | None:
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
