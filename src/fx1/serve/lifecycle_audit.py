"""lifecycle_audit — ``fx1 harness serve`` process-lifecycle deep audit.

Probe battery over the REAL process lifecycle — spawned ``fx1 harness
serve`` subprocesses sent real signals, plus in-process pins for the
lifespan-exit ordering that a signal cannot isolate:

* Cold start: ``create_app`` replays and compacts every journal before
  uvicorn binds, so a request racing boot is refused at TCP (never
  half-served); ``/health`` is public liveness while ``/ready`` is
  credentialed readiness; boot materializes the state dir, the store
  journals, and the mutual-exclusion lock file.
* CLI surface: ``--host``/``--port``/``--state-dir`` wire; a
  non-loopback bind without ``FX1_API_KEY``, an unparseable flag, an
  unknown ``--api-key`` flag, and a file-typed or unreadable
  ``--state-dir`` all fail closed at startup with nonzero exit.
* SIGTERM: the process exits BY the signal; a request admitted before
  the signal completes to a real 200 through the drain window (open
  connections are waited out before the lifespan cascade); a submit
  that cannot get an ``inflight`` slot is refused ``503``/
  ``over_capacity`` — there is no user-visible queue; ``queued`` is a
  transient hop inside an admitted submit whose cancel-once +
  signed-webhook contract is pinned in-process (``_lifespan_exit_probes``);
  a running job is abandoned at exit and honestly re-verdicted
  ``failed`` on the next boot; the port rebinds.
* SIGINT: terminates promptly — uvicorn restores the pre-existing
  handler and re-raises, and under ``asyncio.run`` the Runner folds it
  into a normal cancellation (rc 0).
* Shutdown grace: ``--shutdown-grace-s`` / ``FX1_API_SHUTDOWN_GRACE_S``
  bounds the drain — in-flight work outliving the window is cancelled
  and logged, never silently finished or lost; a second signal inside
  the window does not short-circuit the wait (uvicorn's ``wait_closed``
  is unconditional — pinned as upstream semantics the bound makes
  survivable).
* SIGKILL/restart: a mid-append torn journal tail is quarantined on the
  next boot (the post-#2748 replay contract) with every committed prefix
  record still addressable; a ``running`` job at kill time flips to
  ``failed`` with the restart verdict; minted keys and journaled
  idempotency claims survive; the dir is reusable.
* In-process lifespan exit: the drain latch is set, queued work is
  ``cancelled`` exactly once with a single signed webhook, running work
  is NOT flipped (it finishes past the exit), every post-shutdown
  refusal is enveloped in its own grammar, and the state-dir lease is
  released so the same process can re-acquire it.
* Double-boot: a second process on the same ``--state-dir`` fails
  closed at startup — pinned both in-process (flock held) and at the
  process level (nonzero exit, honest stderr, first process unaffected).

Found while building this lane (fixed in the same commit):

* Two ``fx1 serve`` processes pointed at one ``--state-dir`` had NO
  mutual exclusion — each boot replayed and compacted the shared
  journals, then appended under a private in-memory chain head; the
  interleaved writes broke the hash chain and every record journaled
  after the second boot was silently dropped on the next replay
  (verified end-to-end: two live processes on one dir, SIGKILL both,
  third boot lost both successors' job records).  ``create_app`` now
  takes a nonblocking exclusive advisory lock on
  ``<state-dir>/.fx1-serve.lock`` held for the process lifetime —
  released on lifespan exit and by the OS on any death including
  SIGKILL — so a second process fails closed at boot (exit 2 through
  the CLI) instead of forking journal history.  Same-process second
  apps share one refcounted lease, keeping ``create_app`` re-entrant
  for tests and the SDK; the state dir is now materialized at
  ``create_app`` time (before any store writes lazily) so the lock
  file and every journal share one real directory from boot.
* ``fx1 harness serve`` had NO shutdown bound — ``uvicorn.run`` was
  called without ``timeout_graceful_shutdown``, so SIGTERM waited out
  in-flight connections indefinitely (verified: a 30 s request held
  shutdown for its whole duration; a stuck upstream would hang exit
  forever).  ``--shutdown-grace-s`` / ``FX1_API_SHUTDOWN_GRACE_S``
  (default 30 s, ``0`` disables) now wires
  ``timeout_graceful_shutdown``; work exceeding the window is
  cancelled by uvicorn with a logged overrun instead of pinning exit.

Sealed ``lifecycle_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

from quant_fund.utils.reproducibility import git_revision

__all__ = ["lifecycle_audit", "lifecycle_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = secrets.token_urlsafe(32)
_BYOK_KEY = secrets.token_urlsafe(16)
_WAIT_S = 15.0
_HEALTH = "/health"
_READY = "/ready"
_DRAIN = "/harness/drain"
_JOBS = "/harness/jobs"
_KEYS = "/harness/keys"
_CHAT = "/v1/chat/completions"
_IDEM = "Idempotency-Key"
_JOB_TERMINAL = ("succeeded", "failed", "cancelled")
_LOCK_NAME = ".fx1-serve.lock"
_SRC_ROOT = Path(__file__).resolve().parents[2]
_POSIX = os.name == "posix"


# ---------------------------------------------------------------------------
# Process plumbing — a real `fx1 harness serve` subprocess with drained
# output capture, honest waits, and one blocking HTTP helper.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ServeProc:
    """A spawned serve process whose merged stdout/stderr is drained on a
    reader thread (a full pipe would wedge the shutdown under test)."""

    def __init__(self, popen: subprocess.Popen[str]) -> None:
        self._popen = popen
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._popen.stdout is not None
        for line in self._popen.stdout:
            with self._lock:
                self.lines.append(line)

    def output(self) -> str:
        with self._lock:
            return "".join(self.lines)

    def poll(self) -> int | None:
        return self._popen.poll()

    def wait(self, timeout: float = _WAIT_S) -> int:
        try:
            return self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._popen.kill()
            self._popen.wait(timeout=10)
            raise

    def signal(self, sig: int) -> None:
        self._popen.send_signal(sig)

    def kill(self) -> None:
        if self._popen.poll() is None:
            self._popen.kill()
            self._popen.wait(timeout=10)


@contextmanager
def _serve_process(
    port: int, argv_extra: list[str], env: dict[str, str], cwd: Path
) -> Iterator[_ServeProc]:
    """Spawn ``fx1 harness serve`` bound to loopback; killed on exit."""
    argv = [
        sys.executable,
        "-m",
        "fx1.cli",
        "harness",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *argv_extra,
    ]
    popen = subprocess.Popen(  # noqa: S603 — argv is fully constructed here
        argv,
        env=env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc = _ServeProc(popen)
    try:
        yield proc
    finally:
        proc.kill()


def _serve_env(**extra: str) -> dict[str, str]:
    """Child env: ambient ``FX1_*`` swept, the worktree's ``src`` on
    PYTHONPATH, the interpreter's bin dir on PATH (``dipcatcher`` for the
    real job runner), and the root API key set."""
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("FX1_") or name == "MOONSHOT_API_KEY":
            env.pop(name, None)
    # .venv/bin must stay on PATH — the job runner invokes the `dipcatcher`
    # console script.  sys.executable.parent is the venv bin dir when the
    # bench runs under the venv (resolving it would jump to the cellar).
    bin_dirs = [str(Path(sys.executable).parent), str(Path(sys.prefix) / "bin")]
    env["PATH"] = os.pathsep.join(bin_dirs + [env.get("PATH", "")])
    env["PYTHONPATH"] = str(_SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["NUMBA_DISABLE_JIT"] = "1"
    env[_API_KEY_ENV] = _ROOT
    env.update(extra)
    return env


def _req(
    port: int,
    method: str,
    path: str,
    *,
    key: str | None = _ROOT,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    """One blocking HTTP exchange against a spawned serve. Returns
    ``(status, parsed)``; transport errors raise ``OSError``."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        hdrs = dict(headers or {})
        if key is not None:
            hdrs["X-API-Key"] = key
        payload = json.dumps(body) if body is not None else None
        if payload is not None:
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            parsed = dict(json.loads(raw)) if raw else {}
        except json.JSONDecodeError:
            parsed = {"_raw": raw.decode("utf-8", "replace")}
        return resp.status, parsed
    finally:
        conn.close()


def _wait_serving(proc: _ServeProc, port: int, timeout: float = 30.0) -> tuple[float, int]:
    """Poll ``/health`` until the listener answers 200 — journals have
    already replayed by then (``create_app`` returns before uvicorn
    binds).  Returns ``(elapsed_s, refused_count)``."""
    refused = 0
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"serve exited during boot rc={rc}\n{proc.output()}")
        try:
            status, _ = _req(port, "GET", _HEALTH, key=None, timeout=2)
            if status == 200:
                return time.monotonic() - start, refused
        except OSError:
            refused += 1
            time.sleep(0.02)
    raise TimeoutError(f"serve on :{port} never came up\n{proc.output()}")


def _job_get(port: int, job_id: str, *, key: str = _ROOT) -> tuple[int, dict[str, Any]]:
    return _req(port, "GET", f"{_JOBS}/{job_id}", key=key)


def _job_submit(
    port: int,
    *,
    idem: str | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {_IDEM: idem} if idem else None
    body: dict[str, Any] = {"command": "doctor"}
    if extra:
        body.update(extra)
    return _req(port, "POST", _JOBS, key=_ROOT, body=body, headers=headers)


def _job_status(
    port: int, job_id: str, want: tuple[str, ...], timeout: float = _WAIT_S
) -> dict[str, Any]:
    """Poll until the job reaches one of ``want`` statuses."""
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        code, st = _job_get(port, job_id)
        if code == 200 and st.get("status") in want:
            return st
        time.sleep(0.05)
    return st


class _OpenAIStub:
    """Loopback ``POST /v1/chat/completions`` answering a valid
    chat.completion envelope after ``delay_s`` — the deterministic way to
    hold one BYOK request in-flight across a signal."""

    def __init__(self, delay_s: float = 3.0) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self._lock = threading.Lock()
        stub = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server name
                n = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(n)
                with stub._lock:  # noqa: SLF001 — same-module closure state
                    stub.calls += 1
                time.sleep(stub.delay_s)
                body = json.dumps(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "stub-1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "stub ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._srv.server_address[1])

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


def _byok_env(stub: _OpenAIStub) -> dict[str, str]:
    return _serve_env(
        FX1_BYOK_BASE_URL=f"http://127.0.0.1:{stub.port}/v1",
        FX1_BYOK_API_KEY=_BYOK_KEY,
        FX1_BYOK_MODEL="stub-1",
        FX1_BYOK_ALLOW_PRIVATE_NETWORKS="1",
    )


def _inflight_chat(port: int, result: dict[str, Any]) -> None:
    """Worker for one blocking BYOK chat call; records the outcome."""
    try:
        status, parsed = _req(
            port,
            "POST",
            _CHAT,
            body={"model": "byok", "messages": [{"role": "user", "content": "hi"}]},
            timeout=60,
        )
        result["status"] = status
        result["body"] = parsed
    except Exception as exc:  # noqa: BLE001 — verdict recorded, asserted by caller
        result["error"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# In-process plumbing — app/client factories for the probes a signal cannot
# isolate (lifespan-exit ordering, queued cancel, journal corruption).
# ---------------------------------------------------------------------------


def _make_app(
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    backend: Any | None = None,
    **create_kw: Any,
) -> FastAPI:
    """``create_app`` under the ambient env (``_audit_context`` swept
    ``FX1_*``); a stub backend serves every link."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _StubBackend  # noqa: PLC0415

    def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del argv, timeout_s
        return 0, "ok", ""

    stub = backend if backend is not None else _StubBackend()
    return api_mod.create_app(
        harness=Harness(runner=runner or _fast_runner),
        backend_resolver=lambda *a, **k: stub,
        **create_kw,
    )


def _client(app: FastAPI) -> TestClient:
    """A TestClient NOT entered — the caller enters ``with`` when the
    lifespan-exit sequence is itself the probe."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415

    client = TestClient(app, raise_server_exceptions=False)
    _RESOURCES.get().callback(client.close)
    return client


def _gated_runner() -> tuple[threading.Event, Callable[[list[str], int], tuple[int, str, str]]]:
    """A runner blocking on a gate — deterministic 'still running' work
    for the lifespan-exit probes."""
    gate = threading.Event()

    def _run(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del argv, timeout_s
        gate.wait(timeout=30)
        return 0, "released", ""

    return gate, _run


def _job_get_inproc(client: TestClient, job_id: str) -> tuple[int, dict[str, Any]]:
    r = client.get(f"{_JOBS}/{job_id}")
    return r.status_code, dict(r.json())


# ---------------------------------------------------------------------------
# Probe sections
# ---------------------------------------------------------------------------


def _cold_boot_probes() -> dict[str, bool]:
    """Boot ordering: replay precedes listen; liveness public, readiness
    credentialed; the state dir + journals + lock file materialize at
    boot; the lock is held for the process lifetime."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    from fx1.serve.api import _state_dir_lock_try_acquire  # noqa: PLC0415

    state_dir = _temporary_directory() / "state"
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    port = _free_port()
    with _serve_process(port, ["--state-dir", str(state_dir)], _serve_env(), workdir) as proc:
        elapsed, refused = _wait_serving(proc, port)
        # Interpreter + create_app + replay all precede the bind: a request
        # racing boot is refused at TCP, never half-served.
        out["boot_refused_before_listen"] = refused >= 1 and elapsed < 30.0
        code, health = _req(port, "GET", _HEALTH, key=None)
        out["cold_health_200_liveness_public"] = code == 200 and health.get("status") == "ok"
        out["cold_ready_200_credentialed"] = _req(port, "GET", _READY)[0] == 200
        out["ready_requires_auth"] = _req(port, "GET", _READY, key=None)[0] == 401
        out["boot_materializes_state_dir"] = state_dir.is_dir()
        # The job store compacts unconditionally at boot; the key store
        # only compacts when payloads/damage exist — keys.jsonl is lazily
        # materialized on the first mutation, never at boot.
        out["boot_materializes_jobs_journal"] = (state_dir / "jobs.jsonl").is_file()
        out["keys_journal_lazily_materialized"] = not (state_dir / "keys.jsonl").exists()
        code, mint = _req(port, "POST", _KEYS, body={"name": "boot", "scopes": ["read"]})
        out["first_key_mint_materializes_keys_journal"] = (
            code == 201 and (state_dir / "keys.jsonl").is_file()
        )
        out["boot_materializes_lock_file"] = (state_dir / _LOCK_NAME).is_file()
        out["state_dir_lock_held_while_running"] = _state_dir_lock_try_acquire(state_dir) is None
    return out


def _cli_flag_probes() -> dict[str, bool]:
    """``fx1 harness serve`` flag wiring + fail-closed bad values."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    cwd = _temporary_directory() / "wd"
    cwd.mkdir()

    def _run_cli(argv_extra: list[str], env: dict[str, str]) -> tuple[int, str]:
        argv = [sys.executable, "-m", "fx1.cli", "harness", "serve", *argv_extra]
        proc = subprocess.run(  # noqa: S603 — argv fully constructed
            argv,
            env=env,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return proc.returncode, proc.stdout + proc.stderr

    # Non-loopback bind without FX1_API_KEY refuses exposure.
    env = _serve_env()
    env.pop(_API_KEY_ENV, None)
    rc, text = _run_cli(
        ["--host", "0.0.0.0", "--port", str(_free_port())],  # nosec B104 — probe asserts refusal, never serves
        env,
    )
    out["cli_nonloopback_without_key_fails_closed"] = rc != 0 and "FX1_API_KEY" in text
    # An unparseable --port fails at typer, never silently defaults.
    rc, _ = _run_cli(["--port", "notaport"], _serve_env())
    out["cli_bad_port_fails_closed"] = rc != 0
    # There is deliberately no --api-key flag (credentials are env-only);
    # the option must be rejected rather than silently ignored.
    rc, _ = _run_cli(["--api-key", _BYOK_KEY, "--port", str(_free_port())], _serve_env())
    out["cli_api_key_flag_rejected"] = rc != 0
    # A file-typed --state-dir fails closed at boot, never half-serving.
    fpath = _temporary_directory() / "not-a-dir"
    fpath.write_text("x")
    rc, _ = _run_cli(["--state-dir", str(fpath), "--port", str(_free_port())], _serve_env())
    out["cli_state_dir_regular_file_fails"] = rc != 0
    # Bad --shutdown-grace-s values fail closed: unparseable at typer,
    # negative at the validator, junk env at the validator.
    rc, _ = _run_cli(["--shutdown-grace-s", "abc", "--port", str(_free_port())], _serve_env())
    out["cli_bad_grace_flag_fails_closed"] = rc != 0
    rc, _ = _run_cli(["--shutdown-grace-s", "-5", "--port", str(_free_port())], _serve_env())
    out["cli_negative_grace_fails_closed"] = rc != 0
    env = _serve_env(FX1_API_SHUTDOWN_GRACE_S="junk")
    rc, _ = _run_cli(["--port", str(_free_port())], env)
    out["cli_bad_grace_env_fails_closed"] = rc != 0
    # An unreadable state dir fails closed at boot (permission bits are a
    # posix semantic — the probe only exists there).
    if _POSIX:
        unreadable = _temporary_directory() / "locked-out"
        unreadable.mkdir()
        unreadable.chmod(0)
        try:
            rc, _ = _run_cli(
                ["--state-dir", str(unreadable), "--port", str(_free_port())], _serve_env()
            )
            out["cli_unreadable_state_dir_fails"] = rc != 0
        finally:
            unreadable.chmod(stat.S_IRWXU)
    return out


def _sigterm_probes() -> dict[str, bool]:
    """Real SIGTERM against a real process: exit BY the signal, a running
    job's transition journaled before death then honestly re-verdicted on
    restart, and a clean port rebind.  There is no user-visible queue —
    a submit that cannot get an ``inflight`` slot is refused 503
    ``over_capacity`` (enveloped, Retry-After set); ``queued`` is the
    transient hop inside an admitted submit, whose cancel-once +
    signed-webhook contract is pinned at lifespan level in
    ``_lifespan_exit_probes``."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    state_dir = _temporary_directory() / "state"
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    port = _free_port()
    env = _serve_env()
    with _serve_process(
        port, ["--state-dir", str(state_dir), "--max-inflight", "1"], env, workdir
    ) as proc:
        _wait_serving(proc, port)
        code, job1 = _job_submit(port)
        job1_id = str(job1.get("job_id"))
        out["job1_accepted"] = code == 202
        running = _job_status(port, job1_id, ("running",), timeout=30)
        out["job1_reached_running"] = running.get("status") == "running"
        # One running job saturates max-inflight=1 — the next submit is
        # refused with an enveloped 503, never silently queued.
        code, refused = _job_submit(port)
        out["full_inflight_submit_refused_503"] = code == 503
        out["over_capacity_refusal_enveloped"] = refused.get("code") == "over_capacity"
        proc.signal(signal.SIGTERM)
        # A request racing the signal terminates honestly: refused at
        # TCP, or a complete response — never a half-answer.
        try:
            rcode, _ = _req(port, "GET", _HEALTH, key=None, timeout=3)
            out["request_racing_sigterm_terminal"] = rcode == 200
        except OSError:
            out["request_racing_sigterm_terminal"] = True
        rc = proc.wait(timeout=30)
        out["sigterm_exits_by_signal"] = rc == -signal.SIGTERM
        out["sigterm_graceful_path_logged"] = "Shutting down" in proc.output()
        try:
            _req(port, "GET", _HEALTH, key=None, timeout=2)
            post_exit = "answered"
        except OSError:
            post_exit = "refused"
        out["post_exit_connection_refused"] = post_exit == "refused"
    # The running transition was journaled before the process died —
    # restart replay finds it and re-verdicts it failed.
    journal = state_dir / "jobs.jsonl"
    payloads = [
        json.loads(line).get("payload", {})
        for line in journal.read_text().splitlines()
        if line.strip()
    ]
    out["running_transition_journaled_before_death"] = any(
        isinstance(p.get("job"), dict)
        and p["job"].get("job_id") == job1_id
        and p["job"].get("status") == "running"
        for p in payloads
    )
    # Restart on the same dir/port/key: the abandoned running job's
    # verdict is honest.
    with _serve_process(port, ["--state-dir", str(state_dir)], env, workdir) as proc2:
        _wait_serving(proc2, port)
        out["restart_rebinds_same_port"] = _req(port, "GET", _HEALTH, key=None)[0] == 200
        code, j1 = _job_get(port, job1_id)
        out["restart_running_job_failed_honestly"] = (
            code == 200 and j1.get("status") == "failed" and "restarted" in str(j1.get("error", ""))
        )
    return out


def _sigterm_inflight_probes() -> dict[str, bool]:
    """A request admitted before SIGTERM completes to a real 200 through
    the drain window — open connections are waited out before the
    lifespan cascade runs."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    stub = _OpenAIStub(delay_s=3.0)
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    port = _free_port()
    try:
        with _serve_process(port, [], _byok_env(stub), workdir) as proc:
            _wait_serving(proc, port)
            result: dict[str, Any] = {}
            req_thread = threading.Thread(target=_inflight_chat, args=(port, result), daemon=True)
            req_thread.start()
            end = time.monotonic() + 10
            while stub.calls < 1 and time.monotonic() < end:
                time.sleep(0.05)
            out["inflight_request_really_outstanding"] = stub.calls >= 1
            proc.signal(signal.SIGTERM)
            req_thread.join(timeout=30)
            out["sigterm_inflight_request_completes_200"] = result.get("status") == 200
            out["sigterm_inflight_proc_exits_after"] = proc.wait(timeout=30) == (-signal.SIGTERM)
    finally:
        stub.close()
    return out


def _sigint_probes() -> dict[str, bool]:
    """SIGINT terminates promptly and cleanly: uvicorn restores the
    pre-existing handler then re-raises the captured signal — under
    ``asyncio.run`` that handler folds SIGINT into a normal task
    cancellation, so the honest exit is rc 0 (a re-raise landing on
    ``SIG_DFL`` before the Runner installs would give ``-SIGINT``)."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    port = _free_port()
    with _serve_process(port, [], _serve_env(), workdir) as proc:
        _wait_serving(proc, port)
        proc.signal(signal.SIGINT)
        rc = proc.wait(timeout=30)
        out["sigint_process_terminates"] = rc is not None
        out["sigint_exit_code_honest"] = rc in (0, -signal.SIGINT)
        out["sigint_graceful_path_logged"] = "Finished server process" in proc.output()
    return out


def _grace_timeout_probes() -> dict[str, bool]:
    """Bounded shutdown grace (``--shutdown-grace-s`` /
    ``FX1_API_SHUTDOWN_GRACE_S`` — the defect this lane fixed: uvicorn
    otherwise waits out in-flight connections forever).  In-flight work
    past the window is cancelled and logged — never silently lost — and
    a second signal inside the window does not interrupt the drain
    early (uvicorn still blocks on ``wait_closed`` — pinned, not fixed:
    it is upstream semantics, and the bound makes it survivable)."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    stub = _OpenAIStub(delay_s=30.0)
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    port = _free_port()
    try:
        with _serve_process(port, ["--shutdown-grace-s", "2"], _byok_env(stub), workdir) as proc:
            _wait_serving(proc, port)
            result: dict[str, Any] = {}
            req_thread = threading.Thread(target=_inflight_chat, args=(port, result), daemon=True)
            req_thread.start()
            end = time.monotonic() + 10
            while stub.calls < 1 and time.monotonic() < end:
                time.sleep(0.05)
            out["grace_inflight_really_outstanding"] = stub.calls >= 1
            t0 = time.monotonic()
            proc.signal(signal.SIGTERM)
            time.sleep(0.4)
            proc.signal(signal.SIGINT)
            rc = proc.wait(timeout=20)
            elapsed = time.monotonic() - t0
            out["grace_bounded_process_exits"] = rc is not None
            # Exit lands on the ~2 s grace bound, not the 0.4 s second
            # signal and not the 30 s request end.
            out["grace_exit_on_bound_not_signal"] = 1.0 < elapsed < 12.0
            out["grace_exit_code_honest"] = rc in (0, -signal.SIGTERM, -signal.SIGINT)
            out["grace_overrun_cancel_logged"] = "graceful shutdown exceeded" in proc.output()
            req_thread.join(timeout=10)
            # The cancelled request disconnects — never a silent 200.
            out["grace_aborted_request_disconnects"] = result.get("status") != 200
    finally:
        stub.close()
    return out


def _sigkill_restart_probes() -> dict[str, bool]:
    """Kill -9 mid-work then restart: torn tails quarantined, running jobs
    honestly failed, minted keys and idem claims survive, dir reusable,
    drain latch answers at process level."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    state_dir = _temporary_directory() / "state"
    port = _free_port()
    env = _serve_env()
    job_id = ""
    idem_key = f"lc-idem-{secrets.token_hex(4)}"
    minted_key = ""
    with _serve_process(port, ["--state-dir", str(state_dir)], env, workdir) as proc:
        _wait_serving(proc, port)
        code, mint = _req(port, "POST", _KEYS, body={"name": "lc", "scopes": ["read"]})
        out["minted_key_before_kill"] = code == 201 and bool(mint.get("key"))
        minted_key = str(mint.get("key", ""))
        code, job = _job_submit(port, idem=idem_key)
        job_id = str(job.get("job_id"))
        out["job_accepted_before_kill"] = code == 202
        running = _job_status(port, job_id, ("running", *_JOB_TERMINAL), timeout=30)
        out["job_reached_running_before_kill"] = running.get("status") == "running"
        proc.signal(signal.SIGKILL)
        proc.wait(timeout=15)
        out["sigkill_exit_code_honest"] = proc.poll() == -signal.SIGKILL
        # Guaranteed torn tail: a partial JSONL line without newline is
        # exactly what a mid-append SIGKILL leaves behind.
        journal = state_dir / "jobs.jsonl"
        with journal.open("ab") as fh:
            fh.write(b'{"seq":999,"chain":"deadbeef","sha256":"bad","payload":{"jo')
    with _serve_process(port, ["--state-dir", str(state_dir)], env, workdir) as proc2:
        _wait_serving(proc2, port)
        out["sigkill_restart_serves"] = _req(port, "GET", _HEALTH, key=None)[0] == 200
        code, j1 = _job_get(port, job_id)
        out["sigkill_running_failed_with_restart_verdict"] = (
            code == 200 and j1.get("status") == "failed" and "restarted" in str(j1.get("error", ""))
        )
        out["minted_key_survives_restart"] = _req(port, "GET", _READY, key=minted_key)[0] == 200
        # The journaled idem claim survives: resubmitting the same key
        # replays the SAME job_id (now failed) instead of a phantom run.
        code, replay = _job_submit(port, idem=idem_key)
        out["idem_claim_replays_same_job"] = (
            code == 202 and replay.get("replayed") is True and replay.get("job_id") == job_id
        )
        # The torn tail was compacted away — new appends resume cleanly.
        code, job3 = _job_submit(port)
        out["post_torn_appends_resume"] = code == 202
        if code == 202:
            st = _job_status(port, str(job3.get("job_id")), _JOB_TERMINAL, timeout=30)
            out["post_torn_job_reaches_terminal"] = st.get("status") in _JOB_TERMINAL
        # A drain-then-signal cycle on the same process: the latch answers
        # honestly and shutdown stays prompt.
        code, drain = _req(port, "POST", f"{_DRAIN}?wait_s=0")
        out["process_drain_latch_answers"] = code == 200 and drain.get("draining") is True
        out["drained_submit_refused_503"] = _job_submit(port)[0] == 503
        out["drained_ready_503"] = _req(port, "GET", _READY)[0] == 503
        code, health = _req(port, "GET", _HEALTH, key=None)
        out["drained_health_still_200"] = code == 200 and health.get("draining") is True
        proc2.signal(signal.SIGTERM)
        out["drained_sigterm_exits"] = proc2.wait(timeout=20) == -signal.SIGTERM
    return out


def _double_boot_probes() -> dict[str, bool]:
    """Two processes on one state dir: the second fails closed at boot,
    the first is unaffected, and the dir is reusable after the lock
    holder dies (SIGTERM and SIGKILL alike release the flock)."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}
    if not _POSIX:
        return out
    workdir = _temporary_directory() / "wd"
    workdir.mkdir()
    state_dir = _temporary_directory() / "state"
    env = _serve_env()
    p1, p2 = _free_port(), _free_port()
    with _serve_process(p1, ["--state-dir", str(state_dir)], env, workdir) as first:
        _wait_serving(first, p1)
        argv = [
            sys.executable,
            "-m",
            "fx1.cli",
            "harness",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(p2),
            "--state-dir",
            str(state_dir),
        ]
        second = subprocess.run(  # noqa: S603 — argv fully constructed
            argv,
            env=env,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        combined = second.stdout + second.stderr
        out["double_boot_refused_nonzero"] = second.returncode != 0
        out["double_boot_refusal_names_lock"] = "locked" in combined
        try:
            _req(p2, "GET", _HEALTH, key=None, timeout=2)
            served = True
        except OSError:
            served = False
        out["double_boot_never_listens"] = not served
        # The first process is completely unaffected.
        h = _req(p1, "GET", _HEALTH, key=None)[0]
        s = _job_submit(p1)[0]
        out["first_process_unaffected"] = h == 200 and s == 202
        first.signal(signal.SIGTERM)
        first.wait(timeout=20)
    # The flock dies with the process — a fresh boot on the same dir works.
    with _serve_process(p2, ["--state-dir", str(state_dir)], env, workdir) as third:
        _wait_serving(third, p2)
        out["dir_reusable_after_holder_exits"] = _req(p2, "GET", _HEALTH, key=None)[0] == 200
        third.signal(signal.SIGKILL)
        third.wait(timeout=15)
    with _serve_process(p2, ["--state-dir", str(state_dir)], env, workdir) as fourth:
        _wait_serving(fourth, p2)
        out["dir_reusable_after_sigkill"] = _req(p2, "GET", _HEALTH, key=None)[0] == 200
    return out


def _lifespan_exit_probes() -> dict[str, bool]:
    """In-process ``with TestClient`` exit — the ordering a signal cannot
    isolate: drain latch set, queued work cancelled once with one signed
    webhook, running work never flipped, refusals enveloped, lease
    released, same-process re-entry allowed."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415
    from fx1.serve.webhook_audit import (  # noqa: PLC0415
        _busy_executor,
        _Sink,
        _wait_hits,
    )

    out: dict[str, bool] = {}
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    sink = _Sink()
    try:
        state_dir = _temporary_directory() / "state"
        app = _make_app(max_inflight=1, state_dir=state_dir)
        client = _client(app)
        with client:
            _busy_executor(app, 1, sleep_s=8.0)
            r = client.post(
                _JOBS,
                json={
                    "command": "doctor",
                    "callback_url": sink.url("/hook"),
                    "callback_secret": secrets.token_hex(8),
                },
            )
            assert r.status_code == 202, f"submit refused: {r.status_code} {r.text}"
            job_id = str(r.json()["job_id"])
            _, queued = _job_get_inproc(client, job_id)
            out["queued_reached_under_saturated_executor"] = queued.get("status") == "queued"
        # Same-process second apps work: an ephemeral app coexists, and a
        # second app on ANOTHER held state dir shares its own lease —
        # create_app stays re-entrant inside one process.
        other = _make_app()
        out["ephemeral_app_coexists"] = _client(other).get(_HEALTH).status_code == 200
        shared_dir = _temporary_directory() / "shared"
        shared_a = _make_app(state_dir=shared_dir)
        shared_b = _make_app(state_dir=shared_dir)
        out["same_process_second_app_shares_dir"] = (
            _client(shared_a).get(_HEALTH).status_code == 200
            and _client(shared_b).get(_HEALTH).status_code == 200
        )
        # Post-exit: the drain latch is set and every gated surface
        # refuses — while health keeps answering and ready 503s.
        r = client.post(_JOBS, json={"command": "doctor"})
        out["post_exit_submit_refused_503"] = r.status_code == 503
        out["post_exit_refusal_enveloped"] = r.json().get("code") == "draining"
        out["post_exit_ready_503"] = client.get(_READY).status_code == 503
        out["post_exit_health_200"] = client.get(_HEALTH).status_code == 200
        r = client.post(
            _CHAT,
            json={"model": "byok", "messages": [{"role": "user", "content": "x"}]},
        )
        err = r.json().get("error")
        out["post_exit_openai_refusal_enveloped"] = r.status_code == 503 and (
            isinstance(err, dict) and err.get("code") == "draining"
        )
        # The queued job flipped to cancelled exactly once, with one
        # signed webhook fired during the lifespan exit.
        _, fin = _job_get_inproc(client, job_id)
        out["queued_job_cancelled_at_exit"] = fin.get("status") == "cancelled"
        _wait_hits(sink, 1, timeout=10)
        out["cancel_webhook_fired_exactly_once"] = len(sink.hits) == 1
        if sink.hits:
            hit = sink.hits[0]
            payload = json.loads(hit.body)
            out["cancel_webhook_body_honest"] = payload.get("status") == "cancelled"
            out["cancel_webhook_signed"] = "x-fx1-webhook-signature" in {
                k.lower() for k in hit.headers
            }
        # Running work is NOT flipped by the exit — pin it on a fresh app
        # whose runner blocks on a gate.
        gate, gated = _gated_runner()
        running_app = _make_app(runner=gated)
        running_client = _client(running_app)
        with running_client:
            rr = running_client.post(_JOBS, json={"command": "doctor"})
            running_id = str(rr.json()["job_id"])
            st: dict[str, Any] = {}
            end = time.monotonic() + 10
            while time.monotonic() < end:
                st = dict(running_client.get(f"{_JOBS}/{running_id}").json())
                if st.get("status") == "running":
                    break
                time.sleep(0.05)
            out["running_job_in_flight_at_exit"] = st.get("status") == "running"
        st = dict(running_client.get(f"{_JOBS}/{running_id}").json())
        out["exit_does_not_flip_running_job"] = st.get("status") == "running"
        gate.set()
        end = time.monotonic() + 10
        while time.monotonic() < end:
            st = dict(running_client.get(f"{_JOBS}/{running_id}").json())
            if st.get("status") in _JOB_TERMINAL:
                break
            time.sleep(0.05)
        out["running_job_completes_past_exit"] = st.get("status") == "succeeded"
        # The lease is released on lifespan exit — a same-process acquire
        # now succeeds (released back immediately for cleanliness).
        from fx1.serve.api import (  # noqa: PLC0415
            _state_dir_lock_release,
            _state_dir_lock_try_acquire,
        )

        fd = _state_dir_lock_try_acquire(state_dir)
        out["lifespan_exit_releases_state_dir_lock"] = fd is not None
        _state_dir_lock_release(fd)
    finally:
        sink.close()
    return out


def _journal_quarantine_probes() -> dict[str, bool]:
    """Corrupt-journal boots in-process: a torn tail warns and is
    truncated (prefix kept, appends resume); a mid-chain edit drops the
    posterior; unwritable / file-typed state dirs fail closed."""
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    out: dict[str, bool] = {}

    def _boot_with(state_dir: Path) -> tuple[TestClient, Any]:
        app = _make_app(state_dir=state_dir)
        return _client(app), app.state.job_store

    def _submit(client: TestClient, tag: str) -> str:
        r = client.post(_JOBS, json={"command": "doctor"}, headers={_IDEM: tag})
        assert r.status_code == 202, f"submit refused: {r.status_code} {r.text}"
        return str(r.json()["job_id"])

    # Torn tail: two valid records, then partial bytes (a mid-append kill).
    d1 = _temporary_directory() / "torn"
    client, _ = _boot_with(d1)
    j1 = _submit(client, "t1")
    _submit(client, "t2")
    journal = d1 / "jobs.jsonl"
    with journal.open("ab") as fh:
        fh.write(b'{"seq":3,"chain":"ab","sha256":"zz","payload":{"job')
    client2, store2 = _boot_with(d1)
    out["torn_tail_warns_on_boot"] = len(store2.recover_warnings) > 0
    out["torn_tail_still_serves"] = client2.get(_HEALTH).status_code == 200
    out["torn_tail_prefix_kept"] = client2.get(f"{_JOBS}/{j1}").status_code == 200
    out["torn_tail_appends_resume"] = (
        client2.post(_JOBS, json={"command": "doctor"}).status_code == 202
    )

    # Mid-chain edit: three records, flip one byte inside line 2's
    # payload so its sha mismatches — lines after the break drop too.
    d2 = _temporary_directory() / "mid"
    client3, _ = _boot_with(d2)
    ja = _submit(client3, "m1")
    _submit(client3, "m2")
    jc = _submit(client3, "m3")
    journal2 = d2 / "jobs.jsonl"
    raw = journal2.read_bytes()
    nl = raw.find(b"\n")
    nl2 = raw.find(b"\n", nl + 1)
    line2 = bytearray(raw[nl + 1 : nl2])
    flip_at = line2.find(b'"queued"')
    if flip_at < 0:
        flip_at = len(line2) // 2
    line2[flip_at] ^= 0x01
    journal2.write_bytes(raw[: nl + 1] + bytes(line2) + raw[nl2:])
    client4, store4 = _boot_with(d2)
    out["mid_chain_edit_warns"] = len(store4.recover_warnings) > 0
    out["mid_chain_keeps_prefix"] = client4.get(f"{_JOBS}/{ja}").status_code == 200
    out["mid_chain_drops_posterior"] = client4.get(f"{_JOBS}/{jc}").status_code == 404
    out["mid_chain_appends_resume"] = (
        client4.post(_JOBS, json={"command": "doctor"}).status_code == 202
    )

    # Unwritable / file-typed state dirs fail closed in-process too.
    file_dir = _temporary_directory() / "regular-file"
    file_dir.write_text("x")
    try:
        _make_app(state_dir=file_dir)
        out["file_typed_state_dir_raises"] = False
    except (OSError, ValueError):
        out["file_typed_state_dir_raises"] = True
    if _POSIX:
        unwritable = _temporary_directory() / "noaccess"
        unwritable.mkdir()
        unwritable.chmod(0)
        try:
            try:
                _make_app(state_dir=unwritable)
                out["unwritable_state_dir_raises"] = False
            except (OSError, ValueError):
                out["unwritable_state_dir_raises"] = True
        finally:
            unwritable.chmod(stat.S_IRWXU)
    return out


# ---------------------------------------------------------------------------
# Battery + receipt
# ---------------------------------------------------------------------------


def lifecycle_audit() -> dict[str, Any]:
    """Run the process-lifecycle battery; returns literal bools."""
    from fx1.serve.conv_audit import _audit_context  # noqa: PLC0415

    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_cold_boot_probes())
        out.update(_cli_flag_probes())
        out.update(_sigterm_probes())
        out.update(_sigterm_inflight_probes())
        out.update(_sigint_probes())
        out.update(_grace_timeout_probes())
        out.update(_sigkill_restart_probes())
        out.update(_double_boot_probes())
        out.update(_lifespan_exit_probes())
        out.update(_journal_quarantine_probes())
        return out


def lifecycle_audit_bench(results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Seal lifecycle-audit results; run the battery when results omitted."""
    r = lifecycle_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "lifecycle_audit",
        "schema": "lifecycle_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "probed": [
                "cold-start journal-replay ordering (requests refused pre-listen)",
                "/health public liveness vs /ready credentialed readiness",
                "CLI --host/--port/--state-dir wiring and fail-closed bad values",
                "SIGTERM: exit-by-signal, in-flight request completes, "
                "queued cancelled once + signed webhook, running abandoned "
                "then restart-failed, port rebind",
                "SIGINT clean termination (rc 0 via asyncio Runner cancel)",
                "bounded shutdown grace: overrun cancelled + logged, "
                "second signal inside the window does not short-circuit",
                "SIGKILL: torn-tail quarantine, restart verdicts, key/idem survival",
                "double-boot state-dir mutual exclusion (post-fix refusal)",
                "process-level drain latch then SIGTERM",
                "lifespan-exit ordering: drain latch, cancel-once webhook, "
                "running work unflipped, enveloped refusals, lease release",
                "corrupt-journal boot quarantine (torn tail, mid-chain edit)",
                "unwritable/file-typed state dir fails closed",
            ],
            "not_executed": [
                "non-posix signal semantics (signal probes are posix-gated)",
                "uvicorn multi-worker state-dir sharing",
                "orchestrator-driven rolling restarts with live peers",
            ],
        },
        "interpretation": (
            "Every probe True means: on this checkout `fx1 harness serve` "
            "boots only after its journals replay (connections refused at "
            "TCP, never half-served), wires --host/--port/--state-dir and "
            "fails closed on non-loopback binds without FX1_API_KEY, "
            "unparseable flags, unknown --api-key, and file-typed or "
            "unreadable state dirs; exits BY the delivered signal on "
            "SIGTERM and cleanly on SIGINT while an admitted in-flight "
            "request completes; force-cancels in-flight work that outlives "
            "the bounded shutdown grace (logged, disconnected, never "
            "silently finished) where a second signal cannot short- "
            "circuit the window; "
            "journals a queued-work cancellation exactly once at shutdown "
            "(signed webhook fired once) and honestly re-verdicts abandoned "
            "running work as failed on the next boot; quarantines torn or "
            "mid-chain-broken journal tails while keeping the committed "
            "prefix; survives SIGKILL with keys and idempotency claims "
            "intact; refuses a second process on one state dir instead of "
            "forking journal history; and releases the dir lock on "
            "lifespan exit or death. SYNTHETIC stub backends and a real "
            "dipcatcher runner only — no research claim."
        ),
    }
    canon = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return out


if __name__ == "__main__":
    print(json.dumps(lifecycle_audit_bench(), indent=1))
