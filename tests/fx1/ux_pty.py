"""Drive the real ``python -m dipcatcher_cli`` on a pty. No fixed sleeps."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

HANG_S = 60.0
LOOPBACK_HTTP = "http://127.0.0.1:9"
LOOPBACK_BASE = "http://127.0.0.1:9/v1"
_CHILD_RE = re.compile(rb"CHILD (\d+)")


def forbid_public_connect(address: object) -> None:
    """Raise if ``address`` is not a loopback socket target."""
    host = address[0] if isinstance(address, tuple) and address else address
    text = str(host)
    if text in {"127.0.0.1", "::1", "localhost"} or text.startswith("127."):
        return
    raise AssertionError(f"non-loopback connect to {text}")


def hermetic_network() -> dict[str, str]:
    """Force model and proxy traffic onto an unroutable loopback discard port.

    ``NO_PROXY`` still exempts 127.0.0.1 so a ``StubModelServer`` is reached
    directly. Public names never get a route.
    """
    return {
        "FX1_BASE_URL": LOOPBACK_BASE,
        "HTTP_PROXY": LOOPBACK_HTTP,
        "HTTPS_PROXY": LOOPBACK_HTTP,
        "http_proxy": LOOPBACK_HTTP,
        "https_proxy": LOOPBACK_HTTP,
        "ALL_PROXY": LOOPBACK_HTTP,
        "all_proxy": LOOPBACK_HTTP,
    }


_WRAPPER = r"""
import fcntl
import os
import signal
import struct
import subprocess
import sys
import termios

rows = int(os.environ.get("UX_ROWS", "24"))
cols = int(os.environ.get("UX_COLS", "80"))
fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
before = termios.tcgetattr(0)
proc = subprocess.Popen([sys.executable, "-m", "dipcatcher_cli", *sys.argv[1:]])
# The pty delivers SIGINT/SIGQUIT/SIGTSTP to the whole foreground group.
# Ignore them here so only dipcatcher (spawned with the default disposition)
# handles the key.
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGQUIT, signal.SIG_IGN)
signal.signal(signal.SIGTSTP, signal.SIG_IGN)
sys.stdout.write("CHILD %s\n" % proc.pid)
sys.stdout.flush()
code = proc.wait()
after = termios.tcgetattr(0)
mask_o = termios.OPOST
mask_l = termios.ECHO | termios.ICANON | termios.ISIG
same = (before[1] & mask_o, before[3] & mask_l) == (after[1] & mask_o, after[3] & mask_l)
sys.stdout.write("TERMIOS_%s\n" % ("OK" if same else "BAD"))
sys.stdout.flush()
raise SystemExit(code)
"""


@dataclass
class PtyResult:
    text: str
    raw: bytes
    status: int
    child: int = 0
    root: Path = field(default_factory=Path)
    events: list[tuple[float, str, bytes]] = field(default_factory=list)
    orphans: list[int] = field(default_factory=list)


def chat_env(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal env. Does not copy the parent environment or any secret."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root / "home"),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "INPUTRC": "/dev/null",
        "FX1_CONFIG_DIR": str(root / "cfg"),
        "FX1_SESSION_DIR": str(root / "sessions"),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        **hermetic_network(),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "DIPCATCHER_LOG": "error",
    }
    if extra:
        for key, value in extra.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return env


def prepare(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    (root / "cfg").mkdir(parents=True)
    (root / "sessions").mkdir()
    (root / "home").mkdir()
    return root


def run_pty(
    root: Path,
    actions: list[tuple[object, ...]],
    *,
    args: tuple[str, ...] = ("chat",),
    extra: dict[str, str] | None = None,
    cols: int = 80,
    rows: int = 24,
    hang_s: float = HANG_S,
    wrap: bool = True,
    command: list[str] | None = None,
    leave_running: bool = False,
) -> PtyResult:
    """Run dipcatcher on a new pty. ``actions`` are wait/send/signal/resize steps."""
    import pty

    env = chat_env(root, extra)
    env["UX_ROWS"] = str(rows)
    env["UX_COLS"] = str(cols)
    script = root / "wrap.py"
    if command is not None:
        argv = list(command)
    elif wrap:
        script.write_text(_WRAPPER, encoding="utf-8")
        argv = [os.environ.get("UX_PYTHON", "") or _python(), str(script), *args]
    else:
        argv = [_python(), "-m", "dipcatcher_cli", *args]
    pid, fd = pty.fork()
    if pid == 0:
        try:
            if fd > 2:
                os.close(fd)
            _set_size(0, rows, cols)
            os.execve(argv[0], argv, env)
        finally:
            os._exit(127)
    os.set_blocking(fd, False)
    deadline = time.monotonic() + hang_s
    buf = bytearray()
    events: list[tuple[float, str, bytes]] = []
    orphans: list[int] = []
    child = 0
    status: int | None = None
    try:
        status = _drive(
            pid, fd, actions, buf, deadline, events, orphans, leave_running=leave_running
        )
        child = _child_pid(buf)
        text = bytes(buf).decode("utf-8", "replace")
        return PtyResult(text, bytes(buf), status, child, root, events, orphans)
    finally:
        _reap(pid)
        os.close(fd)


def _drive(
    pid: int,
    fd: int,
    actions: list[tuple[object, ...]],
    buf: bytearray,
    deadline: float,
    events: list[tuple[float, str, bytes]],
    orphans: list[int],
    leave_running: bool = False,
) -> int:
    import select

    index = 0
    origin = 0
    while time.monotonic() < deadline:
        if index >= len(actions):
            if leave_running:
                return 0
            return _wait_exit(pid, fd, buf, deadline)
        kind = actions[index][0]
        if kind == "wait":
            marker = _as_bytes(actions[index][1])
            found = bytes(buf).find(marker, origin)
            if found >= 0:
                origin = found + len(marker)
                events.append((time.monotonic(), "wait", marker))
                index += 1
                continue
        elif kind == "send":
            payload = actions[index][1]
            _write_all(fd, payload, buf, deadline)
            events.append((time.monotonic(), "send", _as_bytes(payload)))
            index += 1
            continue
        elif kind == "signal":
            child = _child_pid(buf)
            if child > 0:
                orphans.extend(_children(child))
                os.kill(child, int(actions[index][1]))
                index += 1
                continue
        elif kind == "resize":
            _resize(fd, buf, int(actions[index][1]), int(actions[index][2]))
            index += 1
            continue
        elif kind == "stopped":
            if _child_stopped(buf):
                index += 1
                continue
        else:
            raise AssertionError(f"unknown pty action {kind!r}")
        if not _pump(fd, buf, deadline):
            wpid, raw = os.waitpid(pid, os.WNOHANG)
            if wpid == pid:
                _drain(fd, buf)
                return os.waitstatus_to_exitcode(raw)
        select.select([], [], [], 0.02)
    tail = bytes(buf[-180:]).decode("utf-8", "replace")
    step = actions[index] if index < len(actions) else ("done",)
    raise AssertionError(
        f"pty deadline step={index} {step!r} origin={origin} "
        f"z={bytes(buf).count(b'Z')} state={_child_state(buf)} len={len(buf)} tail={tail!r}"
    )


def _as_bytes(marker: object) -> bytes:
    if isinstance(marker, bytes):
        return marker
    if isinstance(marker, str):
        return marker.encode()
    return b""


def _pump(fd: int, buf: bytearray, deadline: float) -> bool:
    import select

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    ready, _, _ = select.select([fd], [], [], min(0.2, remaining))
    if not ready:
        return True
    return _drain(fd, buf)


def _drain(fd: int, buf: bytearray) -> bool:
    import errno

    try:
        chunk = os.read(fd, 65536)
    except OSError as exc:
        return exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
    if not chunk:
        return False
    buf.extend(chunk)
    return True


def _wait_exit(pid: int, fd: int, buf: bytearray, deadline: float) -> int:
    import select

    while time.monotonic() < deadline:
        _drain(fd, buf)
        wpid, raw = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            _drain(fd, buf)
            return os.waitstatus_to_exitcode(raw)
        select.select([fd], [], [], 0.05)
    raise AssertionError("child still running at hang guard")


def _resize(fd: int, buf: bytearray, rows: int, cols: int) -> None:
    _set_size(fd, rows, cols)
    child = _child_pid(buf)
    if child > 0:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGWINCH)


def _child_state(buf: bytearray) -> str:
    child = _child_pid(buf)
    if child <= 0:
        return "?"
    try:
        text = Path(f"/proc/{child}/stat").read_text(encoding="utf-8")
    except OSError:
        return "gone"
    return text.rsplit(")", 1)[-1].split()[0]


def _child_stopped(buf: bytearray) -> bool:
    return _child_state(buf) in {"T", "t"}


def _children(pid: int) -> list[int]:
    found: list[int] = []
    task = Path(f"/proc/{pid}/task")
    if not task.is_dir():
        return found
    for item in task.iterdir():
        raw = item / "children"
        try:
            text = raw.read_text(encoding="utf-8")
        except OSError:
            continue
        found.extend(int(part) for part in text.split() if part.isdigit())
    return found


def _child_pid(buf: bytes | bytearray) -> int:
    match = _CHILD_RE.search(bytes(buf))
    if match is None:
        return 0
    return int(match.group(1))


def _write_all(fd: int, data: object, buf: bytearray, deadline: float) -> None:
    import errno

    if isinstance(data, str):
        payload = data.encode()
    elif isinstance(data, bytes):
        payload = data
    else:
        raise TypeError(type(data))
    view = memoryview(payload)
    while view:
        if time.monotonic() >= deadline:
            raise AssertionError("pty write blocked until the hang guard")
        try:
            count = os.write(fd, view)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            _drain(fd, buf)
            continue
        if count <= 0:
            raise OSError("pty write failed")
        view = view[count:]
        _drain(fd, buf)


def _set_size(fd: int, rows: int, cols: int) -> None:
    import fcntl
    import struct
    import termios

    packed = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


def _reap(pid: int) -> None:
    if pid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


def _python() -> str:
    import sys

    return sys.executable
