"""Live pty and adversarial checks for the terminal and orb engines.

POSIX pty cases skip cleanly elsewhere. Every test pins ``FX1_CONFIG_DIR``
and ``FX1_SESSION_DIR`` to a temp directory.
"""

from __future__ import annotations

import contextlib
import io
import os
import select
import stat
import subprocess
import sys
import threading
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from fx1.interactive import orb
from fx1.interactive.terminal import (
    ConsoleWriter,
    InputCancelled,
    LineReader,
    TermCaps,
    detect_caps,
    render_table,
    sanitize_for_terminal,
    wrap_text,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX pty required")
_SRC = str(Path(__file__).resolve().parents[2] / "src")
_ATTACK = "A\x1bP+qSECRET\x1b\\B\x9b2JC\u202eD\x1b_E\x1b\\F\x1b]52;c;CLIPBOARD\x07Z"

_DRIVER = r"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from fx1.interactive.terminal import (
    ConsoleWriter,
    InputCancelled,
    InterruptTracker,
    LineReader,
    detect_caps,
)


def _words():
    return [
        "/help",
        "/health",
        "/exit",
        "/use",
        "fx1",
        "fx1-lite",
        "/harness",
        "run",
        "volatility",
    ]


def _history(reader, writer):
    reader.load_history()
    writer.line("GOT:" + reader.read_line("dip> "))


def _paste(reader, writer):
    text = reader.read_message("dip> ")
    writer.line("MSG:" + text.replace("\n", "|"))


def _async(reader, writer):
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    def emit(index: int) -> None:
        for seq in range(80):
            writer.emit(f"M{index}-{seq}-END")

    threads = [threading.Thread(target=emit, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    line = reader.read_line("dip> ", poll_s=0.02, job_active=lambda: True)
    for thread in threads:
        thread.join(2)
    writer.line("GOT:" + line)


def _double_int(reader, writer):
    tracker = InterruptTracker(window_s=2.0)
    try:
        reader.read_line("dip> ")
        writer.line("NOINT")
    except KeyboardInterrupt:
        writer.line("FIRST:" + tracker.on_interrupt())
    try:
        reader.read_line("dip> ")
        writer.line("NOINT")
    except KeyboardInterrupt:
        writer.line("SECOND:" + tracker.on_interrupt())


def _eof_block(reader, writer):
    try:
        reader.read_message("dip> ")
        writer.line("NOCANCEL")
    except InputCancelled:
        writer.line("CANCEL")
    except EOFError:
        writer.line("EOF")


def _winch(reader, writer):
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    writer.line("GOT:" + reader.read_line("dip> "))


def _cpu(reader, writer):
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    started = time.process_time()
    line = reader.read_line("dip> ")
    spent = time.process_time() - started
    writer.line(f"CPU:{spent:.4f}")
    writer.line("GOT:" + line)


def _plain(reader, writer):
    del reader
    writer.line(_ATTACK)
    writer.line("VISIBLE")


def _tab(reader, writer):
    writer.line("GOT:" + reader.read_line("dip> "))


_ATTACK = "A\x1bP+qSECRET\x1b\\B\x9b2JC\u202eD\x1b_CLIP\x07Z"


def main() -> None:
    mode = sys.argv[1]
    history = Path(sys.argv[2])
    caps = detect_caps()
    writer = ConsoleWriter(caps, caps_fn=lambda: caps)
    reader = LineReader(caps, writer, history_path=history, words=_words)
    modes = {
        "history": _history,
        "paste": _paste,
        "async": _async,
        "double": _double_int,
        "eofblock": _eof_block,
        "winch": _winch,
        "cpu": _cpu,
        "plain": _plain,
        "tab": _tab,
    }
    try:
        modes[mode](reader, writer)
    finally:
        reader.close()
        writer.line("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stdout.write("ERR:" + type(exc).__name__ + ":" + str(exc) + "\n")
        sys.stdout.flush()
"""


@pytest.fixture(autouse=True)
def _fx1_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sess"))


def _caps(**overrides: object) -> TermCaps:
    values: dict[str, object] = {
        "stdin_tty": False,
        "stdout_tty": False,
        "interactive": False,
        "color": False,
        "unicode": True,
        "width": 80,
        "height": 24,
        "windows": False,
    }
    values.update(overrides)
    return TermCaps(**values)  # type: ignore[arg-type]


def _reader(lines: list[str], **kwargs: object) -> LineReader:
    pending = list(lines)

    def read(prompt: str) -> str:
        del prompt
        if not pending:
            raise EOFError
        return pending.pop(0)

    caps = _caps()
    return LineReader(
        caps,
        ConsoleWriter(caps, out=io.StringIO()),
        input_fn=read,
        **kwargs,  # type: ignore[arg-type]
    )


def _columns(text: str) -> int:
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return width


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _child_env(home: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
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
        "FX1_CONFIG_DIR": str(home / "cfg"),
        "FX1_SESSION_DIR": str(home / "sess"),
    }
    if extra:
        env.update(extra)
    return env


def _kill_child(pid: int) -> None:
    import signal

    if pid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, OSError):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def _pty_run(
    tmp_path: Path,
    mode: str,
    *,
    ready: bytes,
    send: bytes | Callable[[int], None] | None,
    done: bytes,
    extra_env: dict[str, str] | None = None,
    history_text: str | None = None,
    timeout: float = 4.0,
) -> bytes:
    import pty

    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "terminal_live_driver.py"
    script.write_text(_DRIVER, encoding="utf-8")
    history = tmp_path / "history.json"
    if history_text is not None:
        history.write_text(history_text, encoding="utf-8")
        os.chmod(history, 0o600)
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
    buf = bytearray()
    stop = threading.Event()

    def drain() -> None:
        while not stop.is_set():
            ready_fds, _, _ = select.select([fd], [], [], 0.05)
            if not ready_fds:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            buf.extend(chunk)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + timeout
        while ready not in buf and time.monotonic() < deadline:
            time.sleep(0.02)
        if ready not in buf:
            raise AssertionError(f"missing {ready!r} in {bytes(buf)!r}")
        if isinstance(send, bytes):
            os.write(fd, send)
        elif send is not None:
            send(fd)
        while done not in buf and time.monotonic() < deadline:
            time.sleep(0.02)
        if done not in buf:
            raise AssertionError(f"missing {done!r} in {bytes(buf[-500:])!r}")
        return bytes(buf)
    finally:
        stop.set()
        thread.join(0.5)
        _kill_child(pid)
        os.close(fd)


def test_sanitize_corpus_is_printable_and_idempotent() -> None:
    samples = [
        "\x1b]0;owned\x07",
        "\x1b]2;owned\x07",
        "\x1b]52;c;Q0xJUFJEX1NFQ1JFVA==\x07",
        "\x1b]52;c;SECRET\x1b\\",
        "\x1b[2J\x1b[H\x1b[3J",
        "\x1b[10S\x1b[5A\x1b[1;1H",
        "\x1bP+qSECRET\x1b\\",
        "\x1b_APCSECRET\x1b\\",
        "\x1b^PMSECRET\x1b\\",
        "\x1bXSOSECRET\x1b\\",
        "\x9b2J",
        "\x9d0;title\x07",
        "\x90DCS\x9c",
        "\u202eABC\u202c",
        "\u202a\u202b\u202c\u202d\u202e",
        "\u2066\u2067\u2068\u2069",
        "\u200e\u200f",
        "keep\n\tme café 你好",
        "\x1b",
        "\x1b[",
    ]
    for sample in samples:
        cleaned = sanitize_for_terminal(sample)
        assert "\x1b" not in cleaned
        assert "\u202e" not in cleaned
        assert "\u009b" not in cleaned
        assert "SECRET" not in cleaned
        assert sanitize_for_terminal(cleaned) == cleaned
    assert sanitize_for_terminal("keep\n\tme") == "keep\n\tme"
    colored = sanitize_for_terminal("\x1b[31mred\x1b[0m\x1b]0;t\x07", keep_sgr=True)
    assert colored == "\x1b[31mred\x1b[0m"
    assert sanitize_for_terminal(colored, keep_sgr=True) == colored


def test_color_on_writer_strips_clipboard_and_title_writes() -> None:
    buf = io.StringIO()
    caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True, color=True)
    writer = ConsoleWriter(caps, out=buf)
    writer.line("x\x1b]52;c;CLIPBOARD\x07y")
    writer.status("\x1b]0;title\x07ok")
    text = buf.getvalue()
    assert "CLIPBOARD" not in text
    assert "\x1b]" not in text
    assert "xy" in text.replace("\n", "")


def test_interactive_prompt_does_not_emit_osc_when_color_is_off() -> None:
    buf = io.StringIO()
    caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True, color=False)
    writer = ConsoleWriter(caps, out=buf)
    writer.write_prompt("\x1b]0;title\x07>")
    assert "\x1b" not in buf.getvalue()
    assert buf.getvalue().endswith(">")


def test_plain_mode_disables_color_and_cursor_controls() -> None:
    stream = _Tty()
    caps = detect_caps(
        stdin=stream,
        stdout=stream,
        environ={"TERM": "xterm", "FORCE_COLOR": "1", "DIPCATCHER_PLAIN": "1"},
        size_fn=lambda: os.terminal_size((80, 24)),
        platform="linux",
    )
    assert caps.color is False
    assert caps.interactive is False
    buf = io.StringIO()
    writer = ConsoleWriter(caps, out=buf)
    writer.line(_ATTACK)
    assert "\x1b" not in buf.getvalue()
    assert "SECRET" not in buf.getvalue()
    assert "CLIPBOARD" not in buf.getvalue()


def test_non_tty_subprocess_has_no_ansi_bytes(tmp_path: Path) -> None:
    code = (
        "from fx1.interactive.terminal import ConsoleWriter, detect_caps\n"
        "caps = detect_caps()\n"
        "ConsoleWriter(caps).line(" + repr(_ATTACK) + ")\n"
        "ConsoleWriter(caps).line('VISIBLE')\n"
    )
    env = _child_env(tmp_path, {"TERM": "dumb", "DIPCATCHER_PLAIN": "1", "FORCE_COLOR": "1"})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 0
    assert b"\x1b" not in proc.stdout
    assert b"SECRET" not in proc.stdout
    assert b"CLIPBOARD" not in proc.stdout
    assert b"VISIBLE" in proc.stdout


def test_line_redraws_the_composer_above_worker_output() -> None:
    buf = io.StringIO()
    caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True, color=True)
    writer = ConsoleWriter(caps, out=buf)
    writer.set_prompt_visible("> ", lambda: "ab")
    writer.emit("hello")
    assert buf.getvalue() == "\r\x1b[2Khello\n> ab"


def test_history_skips_key_commands_and_credential_shapes(tmp_path: Path) -> None:
    lines = [
        "  /keys set sk-abcdefghijklmnopqrstuvwxyz",
        "/KEYS add tok",
        "note api_key=sk-abcdefghijklmnopqrstuvwxyz",
        "hello",
    ]
    reader = _reader(lines, history_path=tmp_path / "h.json")
    for _ in range(len(lines)):
        reader.read_line(">")
    reader.save_history()
    body = (tmp_path / "h.json").read_text(encoding="utf-8")
    assert reader.history() == ("hello",)
    assert "sk-" not in body
    assert "/keys" not in body.lower()
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "h.json").stat().st_mode) == 0o600


def test_concurrent_sessions_append_history(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    barrier = threading.Barrier(2)
    caps = _caps()

    def session(name: str) -> None:
        reader = LineReader(
            caps,
            ConsoleWriter(caps, out=io.StringIO()),
            history_path=path,
            input_fn=lambda _prompt: name,
        )
        reader.read_line(">")
        barrier.wait(2)
        reader.save_history()

    threads = [threading.Thread(target=session, args=(name,)) for name in ("alpha", "beta")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    saved = path.read_text(encoding="utf-8")
    assert "alpha" in saved
    assert "beta" in saved
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_history_max_corrupt_unreadable_and_readonly_close(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    reader = _reader(["a", "b", "c", "d"], history_path=path, history_max=2)
    for _ in range(4):
        reader.read_line(">")
    reader.save_history()
    assert reader.history() == ("c", "d")
    path.write_bytes(b"\xff" * 4096)
    again = _reader([], history_path=path)
    again.load_history()
    assert again.history() == ()
    if os.name == "posix":
        path.chmod(0)
        try:
            quiet = _reader([], history_path=path)
            quiet.load_history()
            assert quiet.history() == ()
        finally:
            path.chmod(0o600)
        locked = tmp_path / "ro"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            blocked = _reader(["kept"], history_path=locked / "h.json")
            assert blocked.read_line(">") == "kept"
            blocked.close()
            assert blocked.history() == ("kept",)
        finally:
            locked.chmod(0o700)


def test_completion_subcommands_and_secret_words_are_filtered() -> None:
    def words() -> list[str]:
        return [
            "/help",
            "/health",
            "/use",
            "fx1",
            "fx1-lite",
            "/harness",
            "run",
            "volatility",
            "/keys set sk-abcdefghijklmnopqrstuvwxyz",
            "sk-abcdefghijklmnopqrstuvwxyz",
        ]

    reader = _reader([], words=words)
    assert [reader.complete("/h", index) for index in range(4)] == [
        "/help",
        "/health",
        "/harness",
        None,
    ]
    reader._get_buffer = lambda: "/use fx"  # noqa: SLF001 — completer context
    assert reader.complete("fx", 0) == "fx1"
    assert reader.complete("fx", 1) == "fx1-lite"
    reader._matches = []
    reader._get_buffer = lambda: "/harness run "
    assert reader.complete("run", 0) == "run"
    assert reader.complete("vol", 0) == "volatility"
    reader._matches = []
    reader._get_buffer = lambda: "/use "
    offered = [reader.complete("", index) for index in range(12)]
    blob = " ".join(item or "" for item in offered)
    assert "fx1" in offered
    assert "sk-" not in blob
    assert "/keys set" not in blob


def test_fence_close_inside_a_paste_keeps_the_tail() -> None:
    reader = _reader(['"""\nsecret\n"""\nplease run'])
    assert reader.read_message(">") == "secret"
    assert reader.read_message(">") == "please run"
    with pytest.raises(EOFError):
        reader.read_message(">")


def test_backslash_fence_and_eof_mid_block() -> None:
    continued = _reader(["alpha\\\nbeta"])
    assert continued.read_message(">") == "alpha\nbeta"
    fenced = _reader(['"""', "hidden"])
    with pytest.raises(InputCancelled):
        fenced.read_message(">")


@posix_only
def test_partial_bracketed_paste_is_one_message() -> None:
    import pty

    master, slave = pty.openpty()
    try:
        os.write(master, b"\x1b[200~one\ntwo\nthree\x1b[201~")
        time.sleep(0.02)
        stream = os.fdopen(slave, "r", encoding="utf-8", newline="")
        caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True)
        reader = LineReader(
            caps,
            ConsoleWriter(caps, out=io.StringIO()),
            stdin=stream,
            paste_window_s=0.05,
        )
        box: dict[str, object] = {}

        def run() -> None:
            try:
                box["msg"] = reader.read_message(">")
            except Exception as exc:
                box["err"] = exc

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(1.5)
        assert not thread.is_alive()
        assert box.get("err") is None
        assert box.get("msg") == "one\ntwo\nthree"
    finally:
        os.close(master)


def test_tables_honor_width_unicode_and_missing_cells() -> None:
    for width in (20, 40, 59, 60, 120):
        text = render_table(
            [{"name": "fx1-lite", "note": "你好世界 " + ("x" * 40)}],
            width=width,
            columns=["name", "note"],
        )
        assert text
        for line in text.splitlines():
            assert _columns(line) <= width
        wrapped = wrap_text("你" * 30, width)
        for line in wrapped.splitlines():
            assert _columns(line) <= width
    missing = render_table(
        [{"name": "fx1", "note": "alpha\nbravo", "gone": None}],
        width=24,
        columns=["name", "note", "gone", "missing"],
    )
    assert "None" not in missing
    assert "alpha" in missing and "bravo" in missing
    hostile = render_table([{"a": "\x1b[2Jboom\x1b]52;c;SECRET\x07"}], width=20, columns=["a"])
    assert "\x1b" not in hostile
    assert "SECRET" not in hostile
    assert "boom" in hostile
    token = render_table([{"u": "a" * 80}], width=20, columns=["u"])
    assert all(len(line) <= 20 for line in token.splitlines())
    marked = render_table([{"a": "e\u0301"}], width=8, columns=["a"])
    assert marked.splitlines()[1] == "-"


@posix_only
def test_pty_history_navigation_and_tab(tmp_path: Path) -> None:
    text = _pty_run(
        tmp_path,
        "history",
        ready=b"dip> ",
        send=b"\x1b[A\x1b[A\x1b[B\r",
        done=b"DONE",
        history_text='["older", "newer"]\n',
    )
    assert b"GOT:newer" in text
    tab = _pty_run(
        tmp_path / "tab",
        "tab",
        ready=b"dip> ",
        send=b"/ex\t\r",
        done=b"DONE",
    )
    assert b"GOT:/exit" in tab


@posix_only
def test_pty_bracketed_paste_variants(tmp_path: Path) -> None:
    one = _pty_run(
        tmp_path / "one",
        "paste",
        ready=b"dip> ",
        send=b"\x1b[200~only\x1b[201~\r",
        done=b"DONE",
    )
    assert b"MSG:only" in one
    many = "\n".join(f"L{index}" for index in range(200))
    pasted = _pty_run(
        tmp_path / "many",
        "paste",
        ready=b"dip> ",
        send=b"\x1b[200~" + many.encode() + b"\x1b[201~\r",
        done=b"DONE",
        timeout=6,
    )
    assert b"MSG:" in pasted
    body = pasted.split(b"MSG:", 1)[1].split(b"\r", 1)[0]
    parts = body.split(b"|")
    assert parts[0] == b"L0"
    assert parts[-1] == b"L199"
    assert len(parts) == 200
    wide = _pty_run(
        tmp_path / "wide",
        "paste",
        ready=b"dip> ",
        send=b"\x1b[200~" + (b"Z" * 10000) + b"\x1b[201~\r",
        done=b"DONE",
        timeout=6,
    )
    assert b"MSG:" + b"Z" * 10000 in wide
    mixed = _pty_run(
        tmp_path / "mix",
        "paste",
        ready=b"dip> ",
        send=b"\x1b[200~hello\tworld\x1b[31mX\x1b[201~\r",
        done=b"DONE",
    )
    assert b"MSG:hello\tworld\x1b[31mX" in mixed


@posix_only
def test_pty_async_emit_preserves_the_typed_line(tmp_path: Path) -> None:
    text = _pty_run(
        tmp_path,
        "async",
        ready=b"READY",
        send=b"hello\n",
        done=b"DONE",
        timeout=8,
    )
    assert b"GOT:hello" in text
    assert b"ERR:" not in text
    found = text.count(b"-END")
    assert found == 8 * 80


@posix_only
def test_pty_double_ctrl_c_and_eof(tmp_path: Path) -> None:
    def send_twice(fd: int) -> None:
        os.write(fd, b"\x03")
        time.sleep(0.3)
        os.write(fd, b"\x03")

    both = _pty_run(
        tmp_path / "both",
        "double",
        ready=b"dip> ",
        send=send_twice,
        done=b"SECOND:exit",
        timeout=3,
    )
    assert b"FIRST:hint" in both
    assert b"SECOND:exit" in both

    def send_eof(fd: int) -> None:
        os.write(fd, b'"""\n')
        time.sleep(0.2)
        os.write(fd, b"hello\n")
        time.sleep(0.2)
        os.write(fd, b"\x04")

    eof = _pty_run(
        tmp_path / "eof",
        "eofblock",
        ready=b"dip> ",
        send=send_eof,
        done=b"CANCEL",
        timeout=3,
    )
    assert b"CANCEL" in eof or b"EOF" in eof
    assert b"NOCANCEL" not in eof


@posix_only
def test_pty_sigwinch_during_read(tmp_path: Path) -> None:
    import pty
    import signal

    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "terminal_live_driver.py"
    script.write_text(_DRIVER, encoding="utf-8")
    history = tmp_path / "history.json"
    pid, fd = pty.fork()
    if pid == 0:
        try:
            if fd > 2:
                os.close(fd)
            os.execve(
                sys.executable,
                [sys.executable, str(script), "winch", str(history)],
                _child_env(tmp_path),
            )
        finally:
            os._exit(127)
    buf = bytearray()
    try:
        deadline = time.monotonic() + 3
        while b"READY" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                buf.extend(os.read(fd, 65536))
        for _ in range(40):
            os.kill(pid, signal.SIGWINCH)
        os.write(fd, b"ok\n")
        while b"DONE" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf.extend(chunk)
        assert b"GOT:ok" in buf
        assert b"ERR:" not in buf
    finally:
        _kill_child(pid)
        os.close(fd)


@posix_only
def test_idle_read_process_time(tmp_path: Path) -> None:
    import pty

    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "terminal_live_driver.py"
    script.write_text(_DRIVER, encoding="utf-8")
    history = tmp_path / "history.json"
    pid, fd = pty.fork()
    if pid == 0:
        try:
            if fd > 2:
                os.close(fd)
            os.execve(
                sys.executable,
                [sys.executable, str(script), "cpu", str(history)],
                _child_env(tmp_path),
            )
        finally:
            os._exit(127)
    buf = bytearray()
    try:
        deadline = time.monotonic() + 1.5
        while b"READY" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                buf.extend(os.read(fd, 4096))
        assert b"READY" in buf
        time.sleep(2.0)
        os.write(fd, b"ok\n")
        deadline = time.monotonic() + 2
        while b"DONE" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf.extend(chunk)
        assert b"GOT:ok" in buf
        match = b"CPU:"
        assert match in buf
        start = buf.index(match) + len(match)
        spent = float(buf[start:].split()[0])
        assert spent < 0.10
    finally:
        _kill_child(pid)
        os.close(fd)


@posix_only
def test_pty_plain_and_dumb_emit_no_ansi(tmp_path: Path) -> None:
    plain = _pty_run(
        tmp_path / "plain",
        "plain",
        ready=b"VISIBLE",
        send=None,
        done=b"DONE",
        extra_env={"DIPCATCHER_PLAIN": "1", "FORCE_COLOR": "1", "TERM": "xterm-256color"},
    )
    assert b"\x1b" not in plain
    assert b"SECRET" not in plain
    assert b"VISIBLE" in plain
    dumb = _pty_run(
        tmp_path / "dumb",
        "plain",
        ready=b"VISIBLE",
        send=None,
        done=b"DONE",
        extra_env={"TERM": "dumb", "FORCE_COLOR": "1"},
    )
    assert b"\x1b" not in dumb
    assert b"VISIBLE" in dumb


def test_orb_motion_stops_and_respects_plain_and_dumb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FXI_ORBS", raising=False)
    monkeypatch.delenv("DIPCATCHER_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    stream = _Tty()
    started = time.monotonic()
    with orb.OrbAnimator(orb.OrbState.WORKING, fps=1, stream=stream) as anim:
        time.sleep(0.05)
    elapsed = time.monotonic() - started
    text = stream.getvalue()
    assert elapsed < 0.4
    assert "\x1b7" in text
    assert "\x1b[J" not in text
    assert anim._thread is not None  # noqa: SLF001
    assert not anim._thread.is_alive()  # noqa: SLF001
    monkeypatch.setenv("TERM", "dumb")
    quiet = _Tty()
    with orb.OrbAnimator(orb.OrbState.SEARCHING, fps=30, stream=quiet):
        time.sleep(0.08)
    assert quiet.getvalue() == ""
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("DIPCATCHER_PLAIN", "1")
    assert orb.orbs_enabled(_Tty()) is False
    plain = _Tty()
    with orb.OrbAnimator(orb.OrbState.SOLVING, fps=30, stream=plain):
        time.sleep(0.05)
    assert plain.getvalue() == ""


def test_orb_banner_rows_share_one_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    plain = orb.banner("fx1", "api.moonshot.ai", color=False)
    widths = {len(line) for line in plain.splitlines()}
    assert len(widths) == 1
    assert "fx1" in plain and "api.moonshot.ai" in plain
    colored = orb.banner("fx1", "api.moonshot.ai", color=True)
    stripped = [sanitize_for_terminal(line) for line in colored.splitlines()]
    assert len({len(line) for line in stripped}) == 1
    monkeypatch.setenv("TERM", "dumb")
    dumb = orb.banner("fx1", "api.moonshot.ai", color=True)
    assert "\x1b" not in dumb


def test_orb_does_not_write_while_the_console_lock_is_held(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FXI_ORBS", raising=False)
    monkeypatch.delenv("DIPCATCHER_PLAIN", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    stream = _Tty()
    caps = _caps(stdin_tty=True, stdout_tty=True, interactive=True, color=True)
    writer = ConsoleWriter(caps, out=stream)
    held = threading.Event()
    release = threading.Event()
    snap = {"n": 0}

    def buffer() -> str:
        held.set()
        snap["n"] = len(stream.getvalue())
        release.wait(1.0)
        return "ab"

    def blocker() -> None:
        writer.set_prompt_visible("> ", buffer)
        writer.above_prompt("ping")

    with orb.OrbAnimator(orb.OrbState.WORKING, fps=40, stream=stream):
        time.sleep(0.05)
        assert len(stream.getvalue()) > 0
        thread = threading.Thread(target=blocker)
        thread.start()
        assert held.wait(1.0)
        time.sleep(0.15)
        during = len(stream.getvalue())
        release.set()
        thread.join(1.0)
        assert not thread.is_alive()
    assert during == snap["n"]


def test_non_tty_orb_is_silent() -> None:
    stream = io.StringIO()
    with orb.OrbAnimator(orb.OrbState.LISTENING, stream=stream):
        time.sleep(0.05)
    assert stream.getvalue() == ""
