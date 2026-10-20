"""Real-process helpers for dipcatcher crash, signal, and fault tests.

Synchronization waits on pty output, marker files, or ``/proc`` state.
Hang guards are at least 60 seconds. Every pid this module starts is
recorded so the caller can kill that exact pid.
"""

from __future__ import annotations

import contextlib
import json
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

HANG_S = 60.0
_PROMPT = "› "
_METRIC = Path(os.environ.get("CHAOS_METRICS", "/tmp/dipcatcher_wave4/D_metrics.jsonl"))

_SITECUSTOMIZE = """\
import os
import sys

def _mark() -> None:
    marker = os.environ.get("DIPCATCHER_CHAOS_MARKER", "")
    if not marker or not sys.argv or sys.argv[0] == "-c":
        return
    if "lane-power" not in sys.argv:
        return
    fd = os.open(marker, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, (str(os.getpid()) + "\\n").encode())
    finally:
        os.close(fd)

try:
    _mark()
except Exception:
    pass
"""


@dataclass
class Lab:
    """Isolated config, sessions, and work directory for one chaos test."""

    root: Path
    config: Path
    sessions: Path
    work: Path
    marker: Path
    secret: str

    def token(self) -> str:
        """Unique string carried by this lab's child environment."""
        return str(self.config)


@dataclass
class PtyProc:
    """One chat (or other) process on a pseudo-terminal."""

    proc: subprocess.Popen[bytes]
    master: int
    buf: bytearray = field(default_factory=bytearray)
    closed: bool = False

    @property
    def pid(self) -> int:
        value = self.proc.pid
        return 0 if value is None else value

    def write_line(self, line: str) -> None:
        _write_all(self.master, (line + "\n").encode())

    def mark(self) -> int:
        """Buffer length. Pass it to :meth:`read_until` to ignore earlier text."""
        return len(self.buf)

    def read_until(self, token: str, timeout: float = HANG_S, *, start: int = 0) -> str:
        """Block until ``token`` appears at or after ``start``."""
        needle = token.encode()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.buf.find(needle, start) >= 0:
                return self.text()
            if self.proc.poll() is not None and not _pending(self.master):
                break
            _read_ready(self.master, self.buf, deadline)
        raise AssertionError(f"missing {token!r} rc={self.proc.poll()} tail={self.text()[-900:]}")

    def text(self) -> str:
        return self.buf.decode("utf-8", errors="replace")

    def signal(self, sig: int) -> None:
        if self.pid > 0 and self.proc.poll() is None:
            os.kill(self.pid, sig)

    def wait(self, timeout: float = HANG_S) -> int:
        """Wait for exit. Returns the wait status converted to an exit code."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                return code
            _slice(deadline)
        raise AssertionError(f"pid {self.pid} still running")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.proc.poll() is None and self.pid > 0:
            os.kill(self.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        os.close(self.master)


@dataclass
class Owned:
    """Pids and pty sessions this test must reap."""

    pids: list[int] = field(default_factory=list)
    ptys: list[PtyProc] = field(default_factory=list)

    def add_pid(self, pid: int) -> None:
        if pid > 0:
            self.pids.append(pid)

    def add_pty(self, chat: PtyProc) -> PtyProc:
        self.ptys.append(chat)
        self.add_pid(chat.pid)
        return chat

    def close(self) -> None:
        for chat in self.ptys:
            chat.close()
        for pid in self.pids:
            _kill(pid)


def make_lab(root: Path, secret: str) -> Lab:
    """Create an isolated lab and a sitecustomize that counts lane-power starts."""
    config = root / "config"
    sessions = root / "sessions"
    work = root / "work"
    home = root / "home"
    py = root / "py"
    for path in (config, sessions, work, home, py, config / "runs"):
        path.mkdir(parents=True, exist_ok=True)
    (py / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    marker = work / "lane-power-starts"
    return Lab(root, config, sessions, work, marker, secret)


def lab_env(lab: Lab) -> dict[str, str]:
    """Child environment. The API key is not placed here."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(lab.root / "home"),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
        "FORCE_COLOR": "0",
        "FX1_CONFIG_DIR": str(lab.config),
        "FX1_SESSION_DIR": str(lab.sessions),
        "DIPCATCHER_CHAOS_MARKER": str(lab.marker),
        "DIPCATCHER_LOG": "error",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(lab.root / "py"),
    }


def configure_endpoint(lab: Lab, model: str, url: str) -> None:
    """Store the stub endpoint in the lab config. Restores the parent environment."""
    names = ("FX1_CONFIG_DIR", "FX1_SESSION_DIR", "HOME")
    saved = {name: os.environ.get(name) for name in names}
    os.environ["FX1_CONFIG_DIR"] = str(lab.config)
    os.environ["FX1_SESSION_DIR"] = str(lab.sessions)
    os.environ["HOME"] = str(lab.root / "home")
    try:
        from fx1.interactive.profiles import set_endpoint

        set_endpoint(model, lab.secret, url)
    finally:
        _restore_env(saved)


def chat_argv(*, resume: bool = False) -> list[str]:
    argv = [sys.executable, "-m", "dipcatcher_cli", "chat", "--model", "fx1"]
    if resume:
        argv.append("--resume")
    return argv


def spawn_pty(argv: list[str], env: Mapping[str, str], cwd: Path) -> PtyProc:
    """Start ``argv`` on a new pty. The caller must :meth:`PtyProc.close` it."""
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=dict(env),
            cwd=cwd,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        os.close(master)
        os.close(slave)
        raise
    os.close(slave)
    return PtyProc(proc, master)


def spawn_exec(argv: list[str], env: Mapping[str, str], cwd: Path) -> subprocess.Popen[str]:
    """Start a non-pty process. stdout and stderr are captured as text."""
    return subprocess.Popen(
        argv,
        env=dict(env),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def lane_plan(out_dir: Path) -> str:
    """Model plan for the catalog-minimum SYNTHETIC lane-power step."""
    payload = {
        "steps": [
            {
                "capability": "lane-power",
                "args": {
                    "--defects": "0.0",
                    "--n-steps": 16,
                    "--n-seeds": 1,
                    "--lanes": "serial_watch",
                    "--out-dir": str(out_dir),
                },
                "why": "minimum SYNTHETIC lane power bench",
            }
        ],
        "rationale": "SYNTHETIC lane-power only",
    }
    return json.dumps(payload)


def empty_plan() -> str:
    """Model plan that selects no harness step."""
    return json.dumps({"steps": [], "rationale": "no harness step"})


def marker_count(path: Path) -> int:
    if not path.is_file():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines)


def marker_pids(path: Path) -> list[int]:
    if not path.is_file():
        return []
    found: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().isdigit():
            found.append(int(line.strip()))
    return found


def wait_until(pred: Callable[[], bool], timeout: float, label: str) -> float:
    """Return elapsed seconds once ``pred`` is true. ``timeout`` is the hang guard."""
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        if pred():
            return time.monotonic() - start
        _slice(deadline)
    raise AssertionError(label)


def pid_alive(pid: int) -> bool:
    """True when Linux ``/proc`` shows the pid and it is not a zombie."""
    if pid <= 0:
        return False
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    parts = raw.rpartition(")")[2].split()
    return bool(parts) and parts[0] not in {"Z", "X"}


def descendants(root: int) -> list[int]:
    """Every live process whose ancestor chain reaches ``root`` (Linux ``PPid``)."""
    if root <= 0 or not Path("/proc").is_dir():
        return []
    children = _children_by_ppid()
    found: list[int] = []
    stack = list(children.get(root, ()))
    seen: set[int] = set()
    while stack and len(found) < 512:
        pid = stack.pop()
        if pid in seen or pid <= 0 or pid == os.getpid():
            continue
        seen.add(pid)
        if pid_alive(pid):
            found.append(pid)
        stack.extend(children.get(pid, ()))
    return found


def marked_pids(token: str) -> list[int]:
    """Pids, other than this one, whose cmdline or environ contains ``token``."""
    needle = token.encode()
    me = os.getpid()
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir() or not needle:
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        carries = _file_has(entry / "cmdline", needle) or _file_has(entry / "environ", needle)
        if carries and pid_alive(pid):
            found.append(pid)
    return found


def fd_count(pid: int) -> int:
    root = Path(f"/proc/{pid}/fd")
    try:
        return sum(1 for _entry in root.iterdir())
    except OSError:
        return -1


def thread_count(pid: int) -> int:
    root = Path(f"/proc/{pid}/task")
    try:
        return sum(1 for _entry in root.iterdir())
    except OSError:
        return -1


def load_sessions(directory: Path) -> list[dict[str, Any]]:
    """Schema-valid sessions. A symlink or corrupt file raises."""
    from fx1.interactive.sessions import SessionStore

    store = SessionStore(directory)
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith(".") or ".corrupt-" in path.name:
            continue
        loaded = store.load(path.stem)
        if not isinstance(loaded, dict) or loaded.get("schema") != 1:
            raise AssertionError(f"session {path.name} is not schema 1")
        records.append(loaded)
    return records


def job_payloads(config: Path) -> list[dict[str, Any]]:
    root = config / "jobs"
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith(".") or ".corrupt-" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            found.append(payload)
    return found


def nonterminal_jobs(config: Path) -> list[dict[str, Any]]:
    terminal = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}
    return [job for job in job_payloads(config) if str(job.get("state")) not in terminal]


def stray_temps(*roots: Path) -> list[str]:
    """``*.tmp`` names left behind. Lock files are not temps."""
    found: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.endswith(".tmp"):
                found.append(str(path.relative_to(root)))
    return found


def broken_json(*roots: Path) -> list[str]:
    """JSON files that do not parse. A half-written receipt shows up here."""
    found: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if ".corrupt-" in path.name or path.name.startswith("."):
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                found.append(path.name)
    return found


def usage_problems(record: Mapping[str, Any]) -> list[str]:
    """Negative or non-integer token counts on stored turns."""
    turns = record.get("turns")
    if not isinstance(turns, list):
        return ["turns missing"]
    problems: list[str] = []
    for turn in turns:
        problems.extend(_usage_row(turn))
    return problems


def assert_clean(text: str, secret: str) -> None:
    """Fail without echoing ``secret`` when it leaked into ``text``."""
    if secret and secret in text:
        raise AssertionError("secret appeared in captured text")


def record(scenario: str, **fields: object) -> None:
    """Append one measured row. Values must not include secrets."""
    _METRIC.parent.mkdir(parents=True, exist_ok=True)
    row = {"scenario": scenario, **fields}
    with _METRIC.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def enable_mid_reset(server: Any) -> None:
    """Teach a stub to reset the socket after a partial JSON body.

    Queue ``Reply`` with header ``X-Chaos: rst``. The stub module is patched
    only for the duration of that request.
    """
    import tests.fx1.stub_model_server as stub

    original = server._handle

    def _handle(handler: BaseHTTPRequestHandler) -> None:
        real = stub._send
        stub._send = lambda item, reply: _send_or_reset(real, item, reply)
        try:
            original(handler)
        finally:
            stub._send = real

    server._handle = _handle


def kill_pids(pids: list[int]) -> None:
    for pid in pids:
        _kill(pid)


def _usage_row(turn: object) -> list[str]:
    if not isinstance(turn, dict):
        return ["turn not object"]
    usage = turn.get("usage")
    if usage is None:
        return []
    if not isinstance(usage, dict):
        return ["usage not object"]
    problems: list[str] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            problems.append(f"bad {key}")
    return problems


def _send_or_reset(real: Callable[..., None], handler: BaseHTTPRequestHandler, reply: Any) -> None:
    headers = getattr(reply, "headers", {})
    if isinstance(headers, dict) and headers.get("X-Chaos") == "rst":
        _reset_mid_body(handler)
        return
    real(handler, reply)


def _reset_mid_body(handler: BaseHTTPRequestHandler) -> None:
    handler.close_connection = True
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", "80")
    handler.end_headers()
    handler.wfile.write(b'{"cho')
    handler.wfile.flush()
    with contextlib.suppress(OSError):
        handler.connection.shutdown(socket.SHUT_RDWR)


def _children_by_ppid() -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return grouped
    for entry in entries:
        if not entry.name.isdigit():
            continue
        parent = _ppid(entry / "status")
        if parent is None:
            continue
        grouped.setdefault(parent, []).append(int(entry.name))
    return grouped


def _ppid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("PPid:"):
            raw = line.split(":", 1)[1].strip()
            return int(raw) if raw.isdigit() else None
    return None


def _file_has(path: Path, needle: bytes) -> bool:
    try:
        return needle in path.read_bytes()
    except OSError:
        return False


def _restore_env(saved: Mapping[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("pty write failed")
        view = view[count:]


def _read_ready(fd: int, buf: bytearray, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    ready, _, _ = select.select([fd], [], [], min(0.2, remaining))
    if not ready:
        return
    try:
        chunk = os.read(fd, 4096)
    except OSError:
        return
    if chunk:
        buf.extend(chunk)


def _pending(fd: int) -> bool:
    ready, _, _ = select.select([fd], [], [], 0)
    return bool(ready)


def _slice(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(0.05, remaining))


def _kill(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)
