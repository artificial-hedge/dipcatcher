"""Real ``dipcatcher chat`` under a POSIX pty. Hang guards stay at or above 60s."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from tests.fx1.journeys_pty.checks import final_problem, job_state_of, newest_session
from tests.fx1.journeys_pty.match import STARTUP_HANG_S, hang_seconds, match_output, wait_requests
from tests.fx1.journeys_pty.textutil import (
    KEY_PROMPT_RE,
    PROMPT_RE,
    URL_PROMPT_RE,
    snippet,
    strip_terminal,
    substitute,
)

_CAPTURE_CAP = 2_000_000
_HINT = "press Ctrl-C again to exit"


class PtyProc:
    """One chat process. The parent only writes user bytes and signals."""

    def __init__(self) -> None:
        self.master: int | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.text = ""
        self.overflow = False
        self._decoder: Any = None

    def start(self, argv: list[str], env: dict[str, str], work: Path) -> str | None:
        if os.name != "posix":
            return "pty journeys require a POSIX terminal"
        import pty

        master, slave = pty.openpty()
        _set_winsize(master)
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                cwd=str(work),
                env=env,
                close_fds=True,
            )
        except OSError as exc:
            os.close(master)
            os.close(slave)
            return f"failed to start chat: {exc}"
        os.close(slave)
        self.master = master
        import codecs

        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        return None

    def region(self, start: int) -> str:
        return self.text[start:]

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send_line(self, text: str) -> None:
        self._write((text + "\n").encode("utf-8"))

    def interrupt(self) -> None:
        self._signal(signal.SIGINT)

    def kill(self) -> None:
        self._signal(signal.SIGKILL)
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return

    def pump(self, timeout: float) -> bool:
        """Read whatever is ready. True when the caller should keep waiting."""
        if self.master is None:
            return False
        if not _ready(self.master, timeout):
            return self.running()
        got = self._consume()
        return self.running() or got

    def wait_until(self, predicate: Any, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if predicate():
                return True
            if not self.pump(0.2) and not predicate():
                return False
        return bool(predicate())

    def wait_exit(self, deadline: float) -> int | None:
        while time.monotonic() < deadline:
            self.pump(0.2)
            code = self._code()
            if code is not None:
                self.pump(0.0)
                return code
        return None

    def close(self) -> int | None:
        if self.running():
            self.kill()
        code = self._code()
        self._close_master()
        return code

    def _code(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()

    def _write(self, data: bytes) -> None:
        if self.master is None:
            raise OSError("pty is closed")
        view = memoryview(data)
        while view:
            count = os.write(self.master, view)
            if count <= 0:
                raise OSError("pty write failed")
            view = view[count:]

    def _signal(self, sig: int) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return

    def _consume(self) -> bool:
        if self.master is None:
            return False
        try:
            chunk = os.read(self.master, 8192)
        except OSError:
            return False
        if not chunk:
            return False
        if self.overflow or self._decoder is None:
            return True
        self.text += strip_terminal(self._decoder.decode(chunk))
        if len(self.text) > _CAPTURE_CAP:
            self.overflow = True
        return True

    def _close_master(self) -> None:
        if self.master is None:
            return
        os.close(self.master)
        self.master = None


class _Drive:
    def __init__(
        self,
        paths: dict[str, Path],
        stub: Any,
        stub_url: str,
        model: str,
    ) -> None:
        self.paths = paths
        self.stub = stub
        self.stub_url = stub_url
        self.model = model
        self.proc: PtyProc | None = None
        self.parts: list[str] = []
        self.last_code: int | None = None
        self.mark = 0

    def transcript(self) -> str:
        live = "" if self.proc is None else self.proc.text
        return "\n".join([*self.parts, live])

    def retire(self) -> None:
        proc = self.proc
        if proc is None:
            return
        self.parts.append(proc.text)
        self.last_code = proc.close()
        self.proc = None

    def apply(self, step: dict[str, Any]) -> str | None:
        if step.get("signal") == "KILL":
            return self._kill(step)
        problem = self._begin(step)
        if problem:
            return problem
        return _body(self, step)

    def _kill(self, step: dict[str, Any]) -> str | None:
        proc = self.proc
        if proc is None or not proc.running():
            return "no live process to kill"
        if not self._wait_running(time.monotonic() + hang_seconds(step)):
            return "session never recorded a running job before kill"
        text = proc.text
        proc.kill()
        self.parts.append(text)
        self.last_code = proc.close()
        self.proc = None
        return None

    def _wait_running(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if _job_running(self.paths["sessions"]):
                return True
            proc = self.proc
            if proc is None or not proc.pump(0.1):
                return _job_running(self.paths["sessions"])
        return _job_running(self.paths["sessions"])

    def _begin(self, step: dict[str, Any]) -> str | None:
        if self._continue(step):
            return None
        self.retire()
        proc = PtyProc()
        problem = proc.start(_argv(step, self.model), _env(self.paths), self.paths["work"])
        if problem:
            return problem
        self.proc = proc
        if not _needs_prompt_first(step):
            self.mark = 0
            return None
        deadline = time.monotonic() + STARTUP_HANG_S
        if not proc.wait_until(lambda: PROMPT_RE.search(proc.text), deadline):
            return f"prompt did not appear: {snippet(proc.text)}"
        self.mark = len(proc.text)
        return None

    def _continue(self, step: dict[str, Any]) -> bool:
        if self.proc is None or not self.proc.running() or step.get("resume"):
            return False
        self.mark = len(self.proc.text)
        return True


def _job_running(sessions: Path) -> bool:
    return job_state_of(newest_session(sessions)) == "running"


def _needs_prompt_first(step: dict[str, Any]) -> bool:
    return bool(step.get("say") or step.get("signal") in {"INT", "INT2"})


def _argv(step: dict[str, Any], model: str) -> list[str]:
    return _chat_argv(bool(step.get("resume")), model)


def _env(paths: dict[str, Path]) -> dict[str, str]:
    return child_env(paths["config"], paths["sessions"], paths["home"])


def run_pty(
    journey: dict[str, Any],
    paths: dict[str, Path],
    *,
    stub: Any,
    stub_url: str,
) -> tuple[str | None, str, int | None, bool]:
    """Return ``(problem, transcript, exit_code, ended)``."""
    drive = _Drive(paths, stub, stub_url, "fx1")
    problem: str | None = None
    try:
        problem = _steps(drive, journey["steps"])
    except OSError as exc:
        problem = f"pty failed: {exc}"
    finally:
        drive.retire()
    transcript = "\n".join(drive.parts)
    if problem:
        return problem, transcript, drive.last_code, False
    final = final_problem(
        journey.get("final", {}),
        paths=paths,
        transcript=transcript,
        exit_code=drive.last_code,
        ended=False,
        stub_url=stub_url,
    )
    return final, transcript, drive.last_code, False


def _steps(drive: _Drive, steps: list[dict[str, Any]]) -> str | None:
    for step in steps:
        problem = drive.apply(step)
        if problem:
            return problem
    return None


def _body(drive: _Drive, step: dict[str, Any]) -> str | None:
    proc = drive.proc
    if proc is None:
        return "pty process is not running"
    deadline = time.monotonic() + hang_seconds(step)
    problem = _do_signal(proc, step, drive.mark, deadline)
    if problem:
        return problem
    problem = _send_user(proc, step, drive.mark, drive.stub_url, drive.paths["work"], deadline)
    if problem:
        return problem
    seen = 0 if drive.stub is None else len(drive.stub.requests)
    return _finish(drive, step, seen)


# Linux syscall numbers for a tty read blocked in the kernel. A SIGINT that
# arrives while readline is between ``input()`` calls is dropped; the second
# Ctrl-C has to land in one of these, and the product window is only 2s.
_BLOCKING_READS = {
    "x86_64": frozenset({"0", "7", "23", "270", "271"}),
    "amd64": frozenset({"0", "7", "23", "270", "271"}),
    "aarch64": frozenset({"63", "72", "73"}),
    "arm64": frozenset({"63", "72", "73"}),
}


def _do_signal(proc: PtyProc, step: dict[str, Any], start: int, deadline: float) -> str | None:
    kind = step.get("signal")
    if kind not in {"INT", "INT2"}:
        return None
    proc.interrupt()
    if kind != "INT2":
        return None
    seen = proc.wait_until(lambda: _HINT in proc.region(start), deadline)
    if not seen:
        return f"Ctrl-C hint did not appear: {snippet(proc.region(start))}"
    return _second_interrupt(proc, start, deadline)


def _second_interrupt(proc: PtyProc, start: int, deadline: float) -> str | None:
    """Send the second Ctrl-C once the prompt is back and the read is blocked."""
    ready = proc.wait_until(lambda: _second_ctrl_c_ready(proc, start), deadline)
    if ready:
        proc.interrupt()
        return None
    if not proc.running():
        return None
    return f"prompt did not return after Ctrl-C: {snippet(proc.region(start))}"


def _second_ctrl_c_ready(proc: PtyProc, start: int) -> bool:
    if not _prompt_after_hint(proc, start) or not proc.running():
        return False
    if _syscall_visible(proc):
        return _blocking_read(proc)
    return _output_settled(proc)


def _prompt_after_hint(proc: PtyProc, start: int) -> bool:
    region = proc.region(start)
    hint_at = region.rfind(_HINT)
    if hint_at < 0:
        return False
    return PROMPT_RE.search(region[hint_at + len(_HINT) :]) is not None


def _syscall_visible(proc: PtyProc) -> bool:
    child = proc.proc
    if child is None or os.name != "posix":
        return False
    if os.uname().machine not in _BLOCKING_READS:
        return False
    return Path(f"/proc/{child.pid}/syscall").exists()


def _blocking_read(proc: PtyProc) -> bool:
    """True when Linux shows the child blocked in a tty read."""
    child = proc.proc
    if child is None:
        return False
    numbers = _BLOCKING_READS[os.uname().machine]
    try:
        status = Path(f"/proc/{child.pid}/status").read_text(encoding="utf-8")
        syscall = Path(f"/proc/{child.pid}/syscall").read_text(encoding="utf-8")
    except OSError:
        return False
    if not _sleeping(status):
        return False
    return syscall.split(maxsplit=1)[0] in numbers


def _sleeping(status: str) -> bool:
    for line in status.splitlines():
        if line.startswith("State:"):
            parts = line.split()
            return len(parts) > 1 and parts[1] == "S"
    return False


def _output_settled(proc: PtyProc) -> bool:
    """Non-Linux fallback: the prompt is up and a short read added no bytes."""
    before = len(proc.text)
    proc.pump(0.05)
    return len(proc.text) == before and proc.running()


def _send_user(
    proc: PtyProc,
    step: dict[str, Any],
    start: int,
    stub_url: str,
    work: Path,
    deadline: float,
) -> str | None:
    say = _opt(step.get("say"), stub_url, work)
    if say is not None:
        proc.send_line(say)
    secret = _opt(step.get("secret"), stub_url, work)
    if secret is not None and not proc.wait_until(
        lambda: KEY_PROMPT_RE.search(proc.region(start)), deadline
    ):
        return f"API key prompt did not appear: {snippet(proc.region(start))}"
    if secret is not None:
        proc.send_line(secret)
    url = _opt(step.get("url"), stub_url, work)
    if url is not None and not proc.wait_until(
        lambda: URL_PROMPT_RE.search(proc.region(start)), deadline
    ):
        return f"endpoint URL prompt did not appear: {snippet(proc.region(start))}"
    if url is not None:
        proc.send_line(url)
    return None


def _finish(drive: _Drive, step: dict[str, Any], seen: int) -> str | None:
    proc = drive.proc
    if proc is None:
        return "pty process is not running"
    wait = step.get("wait", "idle")
    hang = hang_seconds(step)
    deadline = time.monotonic() + hang
    if wait == "request":
        return _finish_request(drive, step, seen, hang)
    if wait == "exit":
        return _finish_exit(drive, step, deadline)
    ready = proc.wait_until(lambda: _idle_ready(drive, step), deadline)
    if ready:
        return None
    return _idle_failure(drive, step)


def _finish_request(drive: _Drive, step: dict[str, Any], seen: int, hang: float) -> str | None:
    proc = drive.proc
    if not wait_requests(proc, drive.stub, seen, hang):
        return "model request was not observed"
    return _matched(drive, step)


def _finish_exit(drive: _Drive, step: dict[str, Any], deadline: float) -> str | None:
    proc = drive.proc
    if proc is None:
        return "pty process is not running"
    code = proc.wait_exit(deadline)
    if code is None:
        return f"process did not exit: {snippet(proc.region(drive.mark))}"
    drive.last_code = code
    return _matched(drive, step)


def _idle_ready(drive: _Drive, step: dict[str, Any]) -> bool:
    proc = drive.proc
    if proc is None:
        return False
    if _matched(drive, step) is not None:
        return False
    return PROMPT_RE.search(proc.region(drive.mark)) is not None


def _idle_failure(drive: _Drive, step: dict[str, Any]) -> str:
    problem = _matched(drive, step)
    if problem:
        return problem
    proc = drive.proc
    if proc is not None and proc.overflow:
        return "pty output exceeded the capture cap before the step finished"
    shown = "" if proc is None else proc.region(drive.mark)
    return f"prompt did not return: {snippet(shown)}"


def _matched(drive: _Drive, step: dict[str, Any]) -> str | None:
    proc = drive.proc
    region = "" if proc is None else proc.region(drive.mark)
    return match_output(step, region, stub_url=drive.stub_url, work=str(drive.paths["work"]))


def _opt(value: object, stub_url: str, work: Path) -> str | None:
    if not isinstance(value, str):
        return None
    return substitute(value, stub_url=stub_url, work=str(work))


def child_env(config: Path, sessions: Path, home: Path) -> dict[str, str]:
    """Clean environment. Inherited credentials are not forwarded."""
    src = str(Path(__file__).resolve().parents[3] / "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": src,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
        "FX1_CONFIG_DIR": str(config),
        "FX1_SESSION_DIR": str(sessions),
        "DIPCATCHER_LOG": "error",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "INPUTRC": "/dev/null",
    }
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _ready(fd: int, timeout: float) -> bool:
    import select

    try:
        ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
    except (ValueError, OSError):
        return False
    return bool(ready)


def _set_winsize(fd: int) -> None:
    import struct
    import termios

    try:
        import fcntl

        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    except OSError:
        return


def _chat_argv(resume: bool, model: str) -> list[str]:
    import sys

    argv = [sys.executable, "-m", "dipcatcher_cli", "chat", "--model", model]
    if resume:
        argv.append("--resume")
    return argv
