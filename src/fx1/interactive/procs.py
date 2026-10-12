"""Owned subprocesses: scrubbed env, deadlines, cancellation, group kill.

Children are started in a new session (POSIX) or a new process group
(Windows) so a timeout or cancel can signal the whole tree, not only the
direct child. Spawn failures, deadlines, and cancellation return a
:class:`ProcResult` and do not raise. Output capture is per-stream and
bounded; excess bytes are dropped and the child keeps running.

``DEFAULT_ENV_ALLOW`` and the ``ENV_DENY_*`` tuples are the audit surface
for credential isolation. A trailing ``*`` on an allow pattern is a
case-insensitive prefix match.

Known limits: teardown signals the process group and, on Linux, descendants
still listed as children of the owned pid (``setsid`` does not change the
parent). A descendant that has already double-forked into a new session and
been reparented is not visible, and the result does not claim that pid was
killed. On Linux the direct child is started through a one-shot exec wrapper
that arms ``PR_SET_PDEATHSIG`` (``SIGKILL``) so the direct child dies if this
process is killed, including by ``SIGKILL``. That signal is not inherited by
grandchildren, so a double-fork plus ``setsid`` escapee still survives both
parent death and the descendant sweep. Reader threads still stop when that
descendant keeps the pipe open. Zombie detection reads Linux ``/proc``. A
direct child that is still alive after the kill attempt stays in the registry
and the result says the cancellation or timeout is pending.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any

from fx1.interactive.redact import is_sensitive_key, redact_argv, redact_text, register_secret

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
    # Loader and startup hooks are not secrets, but extra= bypasses the
    # allow-list. Reintroducing them would execute code in the child.
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "PYTHONSTARTUP",
    "BASH_ENV",
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
_THREAD_DEFAULTS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_READ_CHUNK = 65536
_READ_POLL_S = 0.2
_MAX_DESCENDANTS = 512

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
    *allow_secrets* (those names are never copied from *base*). When the
    parent did not set them, thread-pin variables default to one thread;
    *extra* may override those defaults. ``DIPCATCHER_CHILD=1`` and
    ``PYTHONUNBUFFERED=1`` are always set last. Copied values are redacted.
    Values of credential-named variables are registered first so a later
    argv or allowed variable cannot carry the same secret, except names
    passed in *allow_secrets*.
    """
    patterns = tuple(allow)
    secret_ok = {item.casefold() for item in allow_secrets}
    _register_credential_values(base, secret_ok)
    _register_credential_values(extra, secret_ok)
    cleaned: dict[str, str] = {}
    _take_base(cleaned, base, patterns)
    _fill_thread_defaults(cleaned)
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
    that need isolation pass :func:`scrub_env` themselves. The child is
    started with the redacted argv, which is also the recorded summary.
    *killer*, when
    set, replaces platform tree-kill and is invoked with the child pid.
    *clock* defaults to ``time.monotonic`` and must advance while waiting.
    """
    child_env = _child_env(env)
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
    factory = _popen if popen is None else popen
    try:
        proc = _spawn(factory, summary, child_env, cwd, sys.platform)
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


# Linux-only. Runs in the child, then execs the real argv. prctl persists
# across exec. It is best-effort: a missing libc symbol must not swallow the
# command. SIGPIPE and SIGXFSZ are restored so Python's ignore mask does not
# leak across exec. The child dies only when its parent pid is no longer the
# pid recorded at spawn (a live pid-1 parent is not "already reparented").
_DEATH_WRAPPER = (
    "import os,signal,sys\n"
    "try:\n"
    " import ctypes\n"
    " ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))\n"
    "except Exception:\n"
    " pass\n"
    "for _name in ('SIGPIPE','SIGXFSZ'):\n"
    " _sig=getattr(signal,_name,None)\n"
    " signal.signal(_sig, signal.SIG_DFL) if isinstance(_sig, int) else None\n"
    "os.kill(os.getpid(), signal.SIGKILL) if os.getppid()!=int(sys.argv[1]) else None\n"
    "try:\n"
    " os.execvp(sys.argv[2], sys.argv[2:])\n"
    "except OSError:\n"
    " os._exit(127)\n"
)


def _popen(argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
    command = _death_argv(
        [str(part) for part in argv],
        path=_child_path(kwargs.get("env")),
        cwd=_cwd_text(kwargs.get("cwd")),
    )
    return subprocess.Popen(command, **kwargs)  # noqa: S603 - argv is a sequence, no shell


def _death_argv(argv: list[str], *, path: str | None = None, cwd: str | None = None) -> list[str]:
    """Prefix a Linux parent-death wrapper. Missing executables stay unwrapped.

    ``Popen`` must still raise for a missing binary so the caller reports
    spawn exit 127. Executability is judged from the child's cwd and PATH.
    Other platforms spawn *argv* unchanged.
    """
    if sys.platform != "linux" or not argv or not _can_exec(argv[0], path_env=path, cwd=cwd):
        return argv
    return [sys.executable, "-c", _DEATH_WRAPPER, str(os.getpid()), *argv]


def _child_path(env: object) -> str | None:
    """PATH the child will see. ``None`` means the parent environment."""
    if not isinstance(env, Mapping):
        return None
    raw = env.get("PATH")
    if raw is None:
        return ""
    return str(raw)


def _cwd_text(cwd: object) -> str | None:
    if isinstance(cwd, (str, Path)):
        return os.fspath(cwd)
    return None


def _can_exec(path: str, *, path_env: str | None = None, cwd: str | None = None) -> bool:
    if _has_dir_sep(path):
        return _is_exec(_rooted(path, cwd))
    return _on_path(path, path_env)


def _has_dir_sep(path: str) -> bool:
    if os.sep in path:
        return True
    return os.altsep is not None and os.altsep in path


def _rooted(path: str, cwd: str | None) -> str:
    if os.path.isabs(path):
        return path
    base = os.getcwd() if cwd is None else cwd
    return os.path.join(base, path)


def _is_exec(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _on_path(name: str, path_env: str | None = None) -> bool:
    text = os.environ.get("PATH", "") if path_env is None else path_env
    for directory in text.split(os.pathsep):
        candidate = os.path.join(directory, name) if directory else name
        if _is_exec(candidate):
            return True
    return False


@contextlib.contextmanager
def terminal_stop_scope(
    registry: ProcessRegistry | None = None, *, grace_s: float = 2.0
) -> Iterator[None]:
    """Arm SIGTERM/SIGHUP for this session, then put the previous handlers back."""
    saved = _saved_stop_handlers()
    with contextlib.suppress(OSError, ValueError):
        install_terminal_stop(registry, grace_s=grace_s)
    try:
        yield
    finally:
        _restore_stop_handlers(saved)


def _saved_stop_handlers() -> list[tuple[int, Any]]:
    saved: list[tuple[int, Any]] = []
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if isinstance(sig, int):
            saved.append((sig, signal.getsignal(sig)))
    return saved


def _restore_stop_handlers(saved: list[tuple[int, Any]]) -> None:
    for sig, handler in saved:
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sig, handler)


def install_terminal_stop(registry: ProcessRegistry | None = None, *, grace_s: float = 2.0) -> bool:
    """Arm SIGTERM (exit 143) and SIGHUP (exit 129) when they still have the default disposition.

    An existing Python handler, including exec's, is left in place. The handler
    stops *registry* and raises ``SystemExit`` so a ``finally`` can persist state.
    """
    target = _registry_of(registry)
    armed = _arm_stop("SIGTERM", 143, target, grace_s)
    return _arm_stop("SIGHUP", 129, target, grace_s) or armed


def _arm_stop(name: str, code: int, registry: ProcessRegistry, grace_s: float) -> bool:
    sig = getattr(signal, name, None)
    if not isinstance(sig, int) or signal.getsignal(sig) is not signal.SIG_DFL:
        return False
    signal.signal(sig, _stop_handler(code, registry, grace_s))
    return True


def _stop_handler(
    code: int, registry: ProcessRegistry, grace_s: float
) -> Callable[[int, object], None]:
    def handler(_signum: int, _frame: object) -> None:
        registry.terminate_all(grace_s)
        raise SystemExit(code)

    return handler


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


_PROXY_ENV = frozenset({"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "socks_proxy"})


def _credential_name(name: str) -> bool:
    folded = name.casefold()
    if _suffix_denied(folded):
        return True
    return is_sensitive_key(name)


def _register_credential_values(source: Mapping[str, str] | None, skip: set[str]) -> None:
    """Remember credential-named values so later copies and argv can be scrubbed."""
    if source is None:
        return
    for key, value in source.items():
        if key.casefold() in skip:
            continue
        if _credential_name(key):
            register_secret(value)


def _take_base(cleaned: dict[str, str], base: Mapping[str, str], patterns: tuple[str, ...]) -> None:
    for key, value in base.items():
        if _keep_base(key, patterns):
            _put_env(cleaned, key, value)


def _fill_thread_defaults(cleaned: dict[str, str]) -> None:
    present = {key.casefold() for key in cleaned}
    for key, value in _THREAD_DEFAULTS.items():
        if key.casefold() not in present:
            cleaned[key] = value


def _take_extra(
    cleaned: dict[str, str], extra: Mapping[str, str] | None, secret_ok: set[str]
) -> None:
    if extra is None:
        return
    for key, value in extra.items():
        if not _keep_extra(key, secret_ok):
            continue
        if key.casefold() in secret_ok:
            cleaned[key] = value
            continue
        _put_env(cleaned, key, value)


def _put_env(cleaned: dict[str, str], key: str, value: str) -> None:
    stored = _scrubbed_env_value(key, value)
    if stored is None:
        return
    cleaned[key] = redact_text(stored)


def _scrubbed_env_value(key: str, value: str) -> str | None:
    if key.casefold() not in _PROXY_ENV:
        return value
    return _proxy_authority(value)


def _proxy_authority(value: str) -> str | None:
    """Scheme, host, and port. Userinfo, path, and query are dropped."""
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not host:
        return None
    return _proxy_url(parsed.scheme, host, port)


def _proxy_url(scheme: str, host: str, port: int | None) -> str:
    name = f"[{host}]" if ":" in host else host
    if port is None:
        return f"{scheme}://{name}"
    return f"{scheme}://{name}:{port}"


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
    stop = threading.Event()
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
            stop,
        )
    finally:
        _cleanup(proc, pid, pgid, threads, limits, killer, platform, registry, stop)


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
    stop: threading.Event,
) -> ProcResult:
    threads.extend(_start_readers(proc, _output_limit(limits), boxes[0], boxes[1], stop))
    _remember(registry, pid, pgid, summary, step, clock)
    _emit(emit, "proc.spawn", {"pid": pid, "pgid": pgid, "argv": list(summary), "step": step})
    reason = _poll_until_done(proc, limits, cancel_event, clock)
    _ensure_stopped(proc, reason, pgid, limits, killer, platform, emit, summary, step, pid)
    stop.set()
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
    stop: threading.Event,
) -> None:
    if proc.poll() is None:
        _apply_kill(proc, pgid, limits.kill_grace_s, killer, platform)
    # Unblock readers before joining. Closing the pipe from this thread
    # deadlocks a BufferedReader stuck in read(); the stop flag does not.
    stop.set()
    _join_readers(threads, 0.5)
    if proc.poll() is not None:
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
        stderr=_annotate(stderr, reason, limits, proc.poll() is None),
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


def _annotate(stderr: str, reason: str | None, limits: ProcLimits, still_alive: bool) -> str:
    if reason == "timeout":
        return _timeout_note(stderr, limits, still_alive)
    if reason == "cancel":
        return _cancel_note(stderr, still_alive)
    return stderr


def _timeout_note(stderr: str, limits: ProcLimits, still_alive: bool) -> str:
    if still_alive:
        return f"{stderr}\n[timeout] pending; process still alive after {limits.timeout_s:g}s\n"
    return f"{stderr}\n[timeout] exceeded {limits.timeout_s:g}s; process group terminated\n"


def _cancel_note(stderr: str, still_alive: bool) -> str:
    if still_alive:
        return f"{stderr}\n[cancel] pending; process still alive\n"
    return f"{stderr}\n[cancel] process group terminated\n"


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
    extras = _descendant_pids(proc.pid)
    _signal_group(proc.pid, pgid, signal.SIGTERM)
    _signal_pids(extras, signal.SIGTERM)
    if _tree_quiet(proc, pgid, extras, grace_s):
        return
    extras = _merge_pids(extras, _descendant_pids(proc.pid))
    _signal_group(proc.pid, pgid, _sigkill())
    _signal_pids(extras, _sigkill())
    _wait_proc(proc, 1.0)
    _wait_pids(extras, 1.0)


def _posix_kill_ids(pid: int, pgid: int | None, grace_s: float) -> None:
    extras = _descendant_pids(pid)
    _signal_group(pid, pgid, signal.SIGTERM)
    _signal_pids(extras, signal.SIGTERM)
    if _ids_quiet(pid, pgid, extras, grace_s):
        return
    extras = _merge_pids(extras, _descendant_pids(pid))
    _signal_group(pid, pgid, _sigkill())
    _signal_pids(extras, _sigkill())


def _tree_quiet(
    proc: subprocess.Popen[bytes], pgid: int | None, extras: list[int], grace_s: float
) -> bool:
    deadline = time.monotonic() + max(0.0, grace_s)
    while True:
        direct_done = proc.poll() is not None and not _group_has_members(pgid)
        if direct_done and not _any_alive(extras):
            return True
        if time.monotonic() >= deadline:
            return False
        _pause_briefly(deadline)


def _ids_quiet(pid: int, pgid: int | None, extras: list[int], grace_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, grace_s)
    while True:
        if _group_quiet(pid, pgid) and not _any_alive(extras):
            return True
        if time.monotonic() >= deadline:
            return False
        _pause_briefly(deadline)


def _merge_pids(left: list[int], right: list[int]) -> list[int]:
    return list(dict.fromkeys([*left, *right]))


def _signal_pids(pids: list[int], sig: int) -> None:
    for pid in pids:
        _signal_pid(pid, sig)


def _wait_pids(pids: list[int], grace_s: float) -> None:
    deadline = time.monotonic() + max(0.0, grace_s)
    while True:
        if not _any_alive(pids):
            return
        if time.monotonic() >= deadline:
            return
        _pause_briefly(deadline)


def _any_alive(pids: list[int]) -> bool:
    return any(_pid_alive(pid) for pid in pids)


def _descendant_pids(root: int) -> list[int]:
    """Linux children of *root*, including setsid grandchildren still parented here.

    A double-fork that has already been reparented is absent from this list.
    """
    if root <= 0 or not Path("/proc").is_dir():
        return []
    found: list[int] = []
    seen: set[int] = set()
    stack = _direct_children(root)
    while stack and len(found) < _MAX_DESCENDANTS:
        pid = stack.pop()
        if pid in seen or pid <= 0 or pid == os.getpid():
            continue
        seen.add(pid)
        found.append(pid)
        stack.extend(_direct_children(pid))
    return found


def _direct_children(pid: int) -> list[int]:
    task_dir = Path(f"/proc/{pid}/task")
    try:
        entries = list(task_dir.iterdir())
    except OSError:
        return []
    found: list[int] = []
    for entry in entries:
        found.extend(_read_children(entry / "children"))
    return found


def _read_children(path: Path) -> list[int]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [int(part) for part in text.split() if part.isdigit()]


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
    stop: threading.Event,
) -> list[threading.Thread]:
    threads = [
        threading.Thread(
            target=_read_pipe,
            args=(proc.stdout, limit, stdout_box, stop),
            name="proc-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_read_pipe,
            args=(proc.stderr, limit, stderr_box, stop),
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


def _read_pipe(stream: IO[bytes] | None, limit: int, box: _Capture, stop: threading.Event) -> None:
    if stream is None:
        return
    kept = bytearray()
    truncated = False
    try:
        truncated = _consume_into(stream, limit, kept, stop)
    finally:
        box.store(bytes(kept), truncated)
        _close_quietly(stream)


def _consume_into(stream: IO[bytes], limit: int, kept: bytearray, stop: threading.Event) -> bool:
    if os.name != "posix" or not _selectable(stream):
        return _consume_blocking(stream, limit, kept)
    return _consume_polled(stream, limit, kept, stop)


def _consume_blocking(stream: IO[bytes], limit: int, kept: bytearray) -> bool:
    truncated = False
    while True:
        chunk = _read_chunk(stream)
        if not chunk:
            break
        truncated = _absorb(kept, chunk, limit, truncated)
    return truncated


def _consume_polled(stream: IO[bytes], limit: int, kept: bytearray, stop: threading.Event) -> bool:
    truncated = False
    while True:
        chunk = _poll_read(stream, stop)
        if not chunk:
            break
        truncated = _absorb(kept, chunk, limit, truncated)
    return truncated


def _selectable(stream: IO[bytes]) -> bool:
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    return isinstance(fd, int) and fd >= 0


def _poll_read(stream: IO[bytes], stop: threading.Event) -> bytes | None:
    fd = stream.fileno()
    while True:
        if _readable(fd, 0.0):
            return _read_chunk(stream)
        if stop.is_set():
            return None
        _readable(fd, _READ_POLL_S)


def _readable(fd: int, timeout: float) -> bool:
    try:
        ready, _, _ = select.select([fd], [], [], timeout)
    except (OSError, ValueError):
        return False
    return bool(ready)


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
