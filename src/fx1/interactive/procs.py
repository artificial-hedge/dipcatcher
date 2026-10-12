"""Owned subprocesses: scrubbed env, deadlines, cancellation, group kill.

Children are started in a new session (POSIX) or a new process group
(Windows) so a timeout or cancel can signal the whole tree, not only the
direct child. Spawn failures, deadlines, and cancellation return a
:class:`ProcResult` and do not raise. Output capture is per-stream and
bounded; excess bytes are dropped and the child keeps running.

``DEFAULT_ENV_ALLOW`` and the ``ENV_DENY_*`` tuples are the audit surface
for credential isolation. A trailing ``*`` on an allow pattern is a
case-insensitive prefix match.

Known limits: POSIX teardown signals the process group. A descendant that
calls ``setsid`` leaves that group and is not tracked. Zombie detection
reads Linux ``/proc``; other POSIX hosts wait out ``kill_grace_s``.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any

from fx1.interactive.redact import is_sensitive_key, redact_argv, redact_text

EXIT_TIMEOUT = 124
EXIT_CANCEL = 130
EXIT_SPAWN = 127
TRUNCATION_MARKER = "\n[output truncated]\n"

# Case-insensitive. A trailing "*" is a prefix. Original key spelling is kept.
DEFAULT_ENV_ALLOW: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "USERNAME",
    "LOGNAME",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LANGUAGE",
    "TERM",
    "TZ",
    "VIRTUAL_ENV",
    "FX1_CONFIG_DIR",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "NO_COLOR",
    "PYTHONUNBUFFERED",
    "DIPCATCHER_CHILD",
    "LC_*",
    "PYTHON*",
    "QUANT_*",
    "DIPCATCHER_*",
    "MLFLOW_*",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "FTP_PROXY",
    "SOCKS_PROXY",
)

ENV_DENY_EXACT: tuple[str, ...] = (
    "FX1_BASE_URL",
    "GITHUB_TOKEN",
)
ENV_DENY_PREFIXES: tuple[str, ...] = (
    "MOONSHOT_",
    "AWS_",
    "GATE_SIGNING_KEY",
    "WITNESS_SIGNING_KEY",
    "PROOFCORE_SIGNING_KEY",
    "FX1_BASE_URL",
)
ENV_DENY_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_TOKEN", "_SECRET")

_DENY_EXACT = frozenset(item.casefold() for item in ENV_DENY_EXACT)
_DENY_PREFIXES = tuple(item.casefold() for item in ENV_DENY_PREFIXES)
_DENY_SUFFIXES = tuple(item.casefold() for item in ENV_DENY_SUFFIXES)
_FORCED_ENV: dict[str, str] = {"PYTHONUNBUFFERED": "1", "DIPCATCHER_CHILD": "1"}
_READ_CHUNK = 65536

Emit = Callable[[str, Mapping[str, Any]], None]
Killer = Callable[[int], None]
PopenFactory = Callable[..., subprocess.Popen[bytes]]
Clock = Callable[[], float]


@dataclass(frozen=True)
class ProcLimits:
    """Bounds for one owned child. ``cancel_poll_s`` is the wake interval."""

    timeout_s: float
    kill_grace_s: float = 2.0
    max_output_bytes: int = 2 * 1024 * 1024
    cancel_poll_s: float = 0.05


@dataclass(frozen=True)
class ProcResult:
    """Outcome of :func:`run_owned`. Timeout, cancel, and spawn do not raise."""

    argv_summary: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    cancelled: bool
    truncated: bool
    spawn_error: str | None
    duration_s: float
    pid: int | None
    pgid: int | None


@dataclass(frozen=True)
class LiveProc:
    """One process the registry can still stop."""

    pid: int
    pgid: int | None
    started_at: float
    argv_summary: list[str]
    step: str | None


@dataclass
class _Capture:
    data: bytes = b""
    truncated: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def store(self, data: bytes, truncated: bool) -> None:
        with self.lock:
            self.data = data
            self.truncated = truncated

    def snapshot(self) -> tuple[bytes, bool]:
        with self.lock:
            return self.data, self.truncated


class ProcessRegistry:
    """Thread-safe set of live owned processes. ``REGISTRY`` is the default."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[int, LiveProc] = {}
        self._atexit_installed = False

    def add(self, proc: LiveProc) -> None:
        """Track *proc* until :meth:`discard` or :meth:`terminate_all`."""
        with self._lock:
            self._live[proc.pid] = proc

    def discard(self, pid: int) -> None:
        """Drop *pid*. Unknown pids are ignored."""
        with self._lock:
            self._live.pop(pid, None)

    def live(self) -> list[LiveProc]:
        """Snapshot of tracked processes, oldest first. The list is a copy."""
        with self._lock:
            found = [
                replace(proc, argv_summary=list(proc.argv_summary)) for proc in self._live.values()
            ]
        found.sort(key=lambda proc: (proc.started_at, proc.pid))
        return found

    def terminate_all(self, grace_s: float = 2.0) -> int:
        """Signal every tracked process group. Returns how many were signaled."""
        with self._lock:
            snapshot = list(self._live.values())
        for proc in snapshot:
            self._stop_one(proc, grace_s)
        return len(snapshot)

    def install_atexit(self) -> None:
        """Register :meth:`terminate_all` once, for interpreter shutdown."""
        with self._lock:
            if self._atexit_installed:
                return
            self._atexit_installed = True
        atexit.register(self.terminate_all)

    def _stop_one(self, proc: LiveProc, grace_s: float) -> None:
        try:
            _tree_kill(proc.pid, proc.pgid, grace_s)
        finally:
            self.discard(proc.pid)


REGISTRY = ProcessRegistry()


def scrub_env(
    base: Mapping[str, str],
    *,
    extra: Mapping[str, str] | None = None,
    allow: Iterable[str] = DEFAULT_ENV_ALLOW,
    allow_secrets: Iterable[str] = (),
) -> dict[str, str]:
    """Copy *base* through the allow-list, then drop sensitive names.

    *extra* is applied after that filter, so correlation ids and other
    non-secret knobs pass even when they are not on the allow-list. A
    sensitive *extra* name is still refused unless it is listed in
    *allow_secrets* (those names are never copied from *base*).
    ``DIPCATCHER_CHILD=1`` and ``PYTHONUNBUFFERED=1`` are always set last.
    """
    patterns = tuple(allow)
    secret_ok = {item.casefold() for item in allow_secrets}
    cleaned: dict[str, str] = {}
    _take_base(cleaned, base, patterns)
    _take_extra(cleaned, extra, secret_ok)
    cleaned.update(_FORCED_ENV)
    return cleaned


def run_owned(
    argv: Sequence[str],
    limits: ProcLimits,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    cancel_event: threading.Event | None = None,
    registry: ProcessRegistry | None = None,
    emit: Emit | None = None,
    popen: PopenFactory | None = None,
    killer: Killer | None = None,
    step: str | None = None,
    clock: Clock | None = None,
) -> ProcResult:
    """Run *argv* as an owned process. Never raises for spawn, timeout, or cancel.

    *env* ``None`` scrubs ``os.environ``. A mapping is used verbatim — callers
    that need isolation pass :func:`scrub_env` themselves. *killer*, when
    set, replaces platform tree-kill and is invoked with the child pid.
    *clock* defaults to ``time.monotonic`` and must advance while waiting.
    """
    summary = redact_argv([str(part) for part in argv])
    now = _clock_of(clock)
    started = now()
    step_safe = _clean_step(step)
    if _cancelled(cancel_event):
        return _finished_early(
            summary,
            EXIT_CANCEL,
            "\n[cancel] process group terminated\n",
            cancelled=True,
            started=started,
            clock=now,
            emit=emit,
            step=step_safe,
        )
    child_env = _child_env(env)
    factory = _popen if popen is None else popen
    try:
        proc = _spawn(factory, [str(part) for part in argv], child_env, cwd, sys.platform)
    except (OSError, ValueError) as exc:
        return _spawn_failed(summary, exc, started, now, emit, step_safe)
    return _owned_loop(
        proc,
        summary,
        limits,
        cancel_event,
        _registry_of(registry),
        emit,
        killer,
        step_safe,
        now,
        sys.platform,
        started,
    )


def _popen(argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(list(argv), **kwargs)  # noqa: S603 - argv is a sequence, no shell


def _clock_of(clock: Clock | None) -> Clock:
    if clock is None:
        return time.monotonic
    return clock


def _registry_of(registry: ProcessRegistry | None) -> ProcessRegistry:
    if registry is None:
        return REGISTRY
    return registry


def _child_env(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is None:
        return scrub_env(os.environ)
    return {str(key): str(value) for key, value in env.items()}


def _clean_step(step: str | None) -> str | None:
    if step is None:
        return None
    return redact_text(step)


def _cancelled(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _take_base(cleaned: dict[str, str], base: Mapping[str, str], patterns: tuple[str, ...]) -> None:
    for key, value in base.items():
        if _keep_base(key, patterns):
            cleaned[key] = value


def _take_extra(
    cleaned: dict[str, str], extra: Mapping[str, str] | None, secret_ok: set[str]
) -> None:
    if extra is None:
        return
    for key, value in extra.items():
        if _keep_extra(key, secret_ok):
            cleaned[key] = value


def _keep_base(key: str, patterns: tuple[str, ...]) -> bool:
    return _allowed(key, patterns) and not _denied(key)


def _keep_extra(key: str, secret_ok: set[str]) -> bool:
    if key.casefold() in secret_ok:
        return True
    return not _denied(key)


def _allowed(key: str, patterns: tuple[str, ...]) -> bool:
    folded = key.casefold()
    return any(_matches(folded, pattern) for pattern in patterns)


def _matches(folded: str, pattern: str) -> bool:
    pat = pattern.casefold()
    if pat.endswith("*"):
        return folded.startswith(pat[:-1])
    return folded == pat


def _denied(name: str) -> bool:
    folded = name.casefold()
    if folded in _DENY_EXACT or _prefix_denied(folded):
        return True
    if _suffix_denied(folded) or "password" in folded:
        return True
    return is_sensitive_key(name)


def _prefix_denied(folded: str) -> bool:
    return any(folded.startswith(prefix) for prefix in _DENY_PREFIXES)


def _suffix_denied(folded: str) -> bool:
    return any(folded.endswith(suffix) for suffix in _DENY_SUFFIXES)


def _spawn(
    popen: PopenFactory,
    argv: Sequence[str],
    env: Mapping[str, str],
    cwd: str | Path | None,
    platform: str,
) -> subprocess.Popen[bytes]:
    kwargs = _spawn_kwargs(cwd, platform)
    kwargs["env"] = dict(env)
    return popen(argv, **kwargs)


def _spawn_kwargs(cwd: str | Path | None, platform: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if cwd is not None:
        kwargs["cwd"] = os.fspath(cwd)
    kwargs.update(_platform_kwargs(platform))
    return kwargs


def _platform_kwargs(platform: str) -> dict[str, Any]:
    if platform == "win32":
        flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": flag}
    return {"start_new_session": True}


def _spawn_failed(
    summary: list[str],
    exc: Exception,
    started: float,
    clock: Clock,
    emit: Emit | None,
    step: str | None,
) -> ProcResult:
    message = redact_text(f"{type(exc).__name__}: {exc}")
    result = _result(
        summary,
        exit_code=EXIT_SPAWN,
        stdout="",
        stderr="",
        timed_out=False,
        cancelled=False,
        truncated=False,
        spawn_error=message,
        duration_s=_elapsed(started, clock),
        pid=None,
        pgid=None,
    )
    _emit(emit, "proc.exit", _exit_fields(result, step))
    return result


def _finished_early(
    summary: list[str],
    exit_code: int,
    stderr: str,
    *,
    cancelled: bool,
    started: float,
    clock: Clock,
    emit: Emit | None,
    step: str | None,
) -> ProcResult:
    result = _result(
        summary,
        exit_code=exit_code,
        stdout="",
        stderr=stderr,
        timed_out=False,
        cancelled=cancelled,
        truncated=False,
        spawn_error=None,
        duration_s=_elapsed(started, clock),
        pid=None,
        pgid=None,
    )
    _emit(emit, "proc.exit", _exit_fields(result, step))
    return result


def _owned_loop(
    proc: subprocess.Popen[bytes],
    summary: list[str],
    limits: ProcLimits,
    cancel_event: threading.Event | None,
    registry: ProcessRegistry,
    emit: Emit | None,
    killer: Killer | None,
    step: str | None,
    clock: Clock,
    platform: str,
    started: float,
) -> ProcResult:
    pid = proc.pid
    pgid = _lookup_pgid(pid, platform)
    threads: list[threading.Thread] = []
    boxes = (_Capture(), _Capture())
    try:
        return _supervise(
            proc,
            pid,
            pgid,
            summary,
            limits,
            cancel_event,
            registry,
            emit,
            killer,
            step,
            clock,
            platform,
            started,
            threads,
            boxes,
        )
    finally:
        _cleanup(proc, pid, pgid, threads, limits, killer, platform, registry)


def _supervise(
    proc: subprocess.Popen[bytes],
    pid: int,
    pgid: int | None,
    summary: list[str],
    limits: ProcLimits,
    cancel_event: threading.Event | None,
    registry: ProcessRegistry,
    emit: Emit | None,
    killer: Killer | None,
    step: str | None,
    clock: Clock,
    platform: str,
    started: float,
    threads: list[threading.Thread],
    boxes: tuple[_Capture, _Capture],
) -> ProcResult:
    threads.extend(_start_readers(proc, _output_limit(limits), boxes[0], boxes[1]))
    _remember(registry, pid, pgid, summary, step, clock)
    _emit(emit, "proc.spawn", {"pid": pid, "pgid": pgid, "argv": list(summary), "step": step})
    reason = _poll_until_done(proc, limits, cancel_event, clock)
    _ensure_stopped(proc, reason, pgid, limits, killer, platform, emit, summary, step, pid)
    _join_readers(threads, _reader_join_timeout(limits))
    return _finish(proc, reason, boxes, summary, limits, started, clock, pid, pgid, emit, step)


def _cleanup(
    proc: subprocess.Popen[bytes],
    pid: int,
    pgid: int | None,
    threads: list[threading.Thread],
    limits: ProcLimits,
    killer: Killer | None,
    platform: str,
    registry: ProcessRegistry,
) -> None:
    if proc.poll() is None:
        _apply_kill(proc, pgid, limits.kill_grace_s, killer, platform)
    _join_readers(threads, 0.5)
    registry.discard(pid)


def _finish(
    proc: subprocess.Popen[bytes],
    reason: str | None,
    boxes: tuple[_Capture, _Capture],
    summary: list[str],
    limits: ProcLimits,
    started: float,
    clock: Clock,
    pid: int,
    pgid: int | None,
    emit: Emit | None,
    step: str | None,
) -> ProcResult:
    stdout, out_flag = _render(boxes[0])
    stderr, err_flag = _render(boxes[1])
    result = _result(
        summary,
        exit_code=_exit_code(proc, reason),
        stdout=stdout,
        stderr=_annotate(stderr, reason, limits),
        timed_out=reason == "timeout",
        cancelled=reason == "cancel",
        truncated=out_flag or err_flag,
        spawn_error=None,
        duration_s=_elapsed(started, clock),
        pid=pid,
        pgid=pgid,
    )
    _emit(emit, "proc.exit", _exit_fields(result, step))
    return result


def _result(
    summary: list[str],
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    cancelled: bool,
    truncated: bool,
    spawn_error: str | None,
    duration_s: float,
    pid: int | None,
    pgid: int | None,
) -> ProcResult:
    return ProcResult(
        argv_summary=list(summary),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        cancelled=cancelled,
        truncated=truncated,
        spawn_error=spawn_error,
        duration_s=duration_s,
        pid=pid,
        pgid=pgid,
    )


def _exit_fields(result: ProcResult, step: str | None) -> dict[str, Any]:
    return {
        "pid": result.pid,
        "pgid": result.pgid,
        "argv": list(result.argv_summary),
        "step": step,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "truncated": result.truncated,
        "duration_s": result.duration_s,
        "spawn_error": result.spawn_error,
    }


def _elapsed(started: float, clock: Clock) -> float:
    return max(0.0, clock() - started)


def _output_limit(limits: ProcLimits) -> int:
    if limits.max_output_bytes < 0:
        return 0
    return limits.max_output_bytes


def _reader_join_timeout(limits: ProcLimits) -> float:
    grace = limits.kill_grace_s if limits.kill_grace_s > 0 else 0.0
    return max(1.0, grace)


def _exit_code(proc: subprocess.Popen[bytes], reason: str | None) -> int:
    if reason == "timeout":
        return EXIT_TIMEOUT
    if reason == "cancel":
        return EXIT_CANCEL
    if proc.returncode is None:
        return 1
    return proc.returncode


def _annotate(stderr: str, reason: str | None, limits: ProcLimits) -> str:
    if reason == "timeout":
        return f"{stderr}\n[timeout] exceeded {limits.timeout_s:g}s; process group terminated\n"
    if reason == "cancel":
        return f"{stderr}\n[cancel] process group terminated\n"
    return stderr


def _render(box: _Capture) -> tuple[str, bool]:
    data, truncated = box.snapshot()
    text = data.decode("utf-8", errors="replace")
    if truncated:
        return text + TRUNCATION_MARKER, True
    return text, False


def _remember(
    registry: ProcessRegistry,
    pid: int,
    pgid: int | None,
    summary: list[str],
    step: str | None,
    clock: Clock,
) -> None:
    registry.add(
        LiveProc(
            pid=pid,
            pgid=pgid,
            started_at=clock(),
            argv_summary=list(summary),
            step=step,
        )
    )


def _lookup_pgid(pid: int, platform: str) -> int | None:
    if platform == "win32" or not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        return pid


def _poll_until_done(
    proc: subprocess.Popen[bytes],
    limits: ProcLimits,
    cancel_event: threading.Event | None,
    clock: Clock,
) -> str | None:
    deadline = clock() + limits.timeout_s
    while True:
        if proc.poll() is not None:
            return None
        if _cancelled(cancel_event):
            return "cancel"
        now = clock()
        if now >= deadline:
            return "timeout"
        _wait_slice(proc, _slice_for(limits, deadline, now))


def _slice_for(limits: ProcLimits, deadline: float, now: float) -> float:
    remaining = deadline - now
    if remaining <= 0:
        return 0.0
    poll = limits.cancel_poll_s if limits.cancel_poll_s > 0 else 0.001
    return min(poll, remaining)


def _wait_slice(proc: subprocess.Popen[bytes], slice_s: float) -> None:
    if slice_s <= 0:
        return
    try:
        proc.wait(timeout=slice_s)
    except subprocess.TimeoutExpired:
        return


def _ensure_stopped(
    proc: subprocess.Popen[bytes],
    reason: str | None,
    pgid: int | None,
    limits: ProcLimits,
    killer: Killer | None,
    platform: str,
    emit: Emit | None,
    summary: list[str],
    step: str | None,
    pid: int,
) -> None:
    if reason is None and not _group_has_members(pgid):
        return
    why = reason if reason is not None else "orphan"
    _emit(
        emit,
        "proc.kill",
        {"pid": pid, "pgid": pgid, "argv": list(summary), "step": step, "reason": why},
    )
    _apply_kill(proc, pgid, limits.kill_grace_s, killer, platform)


def _apply_kill(
    proc: subprocess.Popen[bytes],
    pgid: int | None,
    grace_s: float,
    killer: Killer | None,
    platform: str,
) -> None:
    if killer is not None:
        killer(proc.pid)
        _wait_proc(proc, grace_s)
        return
    if platform == "win32":
        _taskkill(proc.pid)
        _wait_proc(proc, grace_s)
        return
    _posix_escalate(proc, pgid, grace_s)


def _wait_proc(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    timeout = grace_s if grace_s > 0 else 0.0
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return


def _tree_kill(pid: int, pgid: int | None, grace_s: float) -> None:
    if sys.platform == "win32":
        _taskkill(pid)
        return
    _posix_kill_ids(pid, pgid, grace_s)


def _taskkill(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    try:
        subprocess.run(  # noqa: S603 - fixed taskkill argv, no shell
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return


def _posix_escalate(proc: subprocess.Popen[bytes], pgid: int | None, grace_s: float) -> None:
    _signal_group(proc.pid, pgid, signal.SIGTERM)
    if _wait_proc_group(proc, pgid, grace_s):
        return
    _signal_group(proc.pid, pgid, _sigkill())
    _wait_proc(proc, 1.0)


def _posix_kill_ids(pid: int, pgid: int | None, grace_s: float) -> None:
    _signal_group(pid, pgid, signal.SIGTERM)
    if _wait_ids_gone(pid, pgid, grace_s):
        return
    _signal_group(pid, pgid, _sigkill())


def _wait_proc_group(proc: subprocess.Popen[bytes], pgid: int | None, grace_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, grace_s)
    while True:
        if proc.poll() is not None and not _group_has_members(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        _pause_briefly(deadline)


def _wait_ids_gone(pid: int, pgid: int | None, grace_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, grace_s)
    while True:
        if _group_quiet(pid, pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        _pause_briefly(deadline)


def _group_quiet(pid: int, pgid: int | None) -> bool:
    if _pid_alive(pid):
        return False
    return not _group_has_members(pgid)


def _pause_briefly(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    time.sleep(min(0.05, remaining))


def _sigkill() -> int:
    sig = getattr(signal, "SIGKILL", None)
    if isinstance(sig, int):
        return sig
    return signal.SIGTERM


def _signal_group(pid: int, pgid: int | None, sig: int) -> None:
    target = _safe_pgid(pgid)
    if target is not None and _killpg(target, sig):
        return
    _signal_pid(pid, sig)


def _killpg(pgid: int, sig: int) -> bool:
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, sig)
    except OSError:
        return False
    return True


def _signal_pid(pid: int, sig: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    try:
        os.kill(pid, sig)
    except OSError:
        return


def _safe_pgid(pgid: int | None) -> int | None:
    if pgid is None or pgid <= 0:
        return None
    if pgid == _own_pgrp():
        return None
    return pgid


def _own_pgrp() -> int | None:
    if os.name != "posix":
        return None
    try:
        return os.getpgrp()
    except OSError:
        return None


def _group_has_members(pgid: int | None) -> bool:
    target = _safe_pgid(pgid)
    if target is None:
        return False
    return _killpg(target, 0)


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or _zombie(pid):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _zombie(pid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    parts = text.rpartition(")")[2].split()
    return bool(parts) and parts[0] == "Z"


def _start_readers(
    proc: subprocess.Popen[bytes],
    limit: int,
    stdout_box: _Capture,
    stderr_box: _Capture,
) -> list[threading.Thread]:
    threads = [
        threading.Thread(
            target=_read_pipe,
            args=(proc.stdout, limit, stdout_box),
            name="proc-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_read_pipe,
            args=(proc.stderr, limit, stderr_box),
            name="proc-stderr",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    return threads


def _join_readers(threads: list[threading.Thread], timeout: float) -> None:
    for thread in threads:
        thread.join(timeout=timeout)


def _read_pipe(stream: IO[bytes] | None, limit: int, box: _Capture) -> None:
    if stream is None:
        return
    kept = bytearray()
    truncated = False
    try:
        truncated = _consume_into(stream, limit, kept)
    finally:
        box.store(bytes(kept), truncated)
        _close_quietly(stream)


def _consume_into(stream: IO[bytes], limit: int, kept: bytearray) -> bool:
    truncated = False
    while True:
        chunk = _read_chunk(stream)
        if not chunk:
            break
        truncated = _absorb(kept, chunk, limit, truncated)
    return truncated


def _read_chunk(stream: IO[bytes]) -> bytes:
    try:
        return stream.read(_READ_CHUNK)
    except (OSError, ValueError):
        return b""


def _absorb(kept: bytearray, chunk: bytes, limit: int, truncated: bool) -> bool:
    cap = limit if limit > 0 else 0
    if len(kept) >= cap:
        return True
    room = cap - len(kept)
    if len(chunk) <= room:
        kept.extend(chunk)
        return truncated
    kept.extend(chunk[:room])
    return True


def _close_quietly(stream: IO[bytes]) -> None:
    with contextlib.suppress(OSError):
        stream.close()


def _emit(emit: Emit | None, kind: str, fields: Mapping[str, Any]) -> None:
    if emit is None:
        return
    emit(kind, fields)
