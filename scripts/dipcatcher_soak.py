"""Bounded soak and leak check for the dipcatcher concierge stack.

The harness drives ``Concierge`` the way ``headless`` builds it: non-interactive
approvals, a single-attempt model client, no animation, and a session store.
Work runs against the loopback stub and the real harness. The evidence record
is a short run on this machine. It is not the 24-hour mixed-workload soak the
product contract (P6) requires.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "dipcatcher.soak.v1"
SCOPE = (
    "Short bounded soak on this machine. This is NOT the 24-hour "
    "mixed-workload soak the product contract (P6) requires. "
    "Crash-restart cycles here are a bounded sample, not that soak."
)
WORKLOADS: tuple[str, ...] = (
    "chat_fx1",
    "chat_fx1_lite",
    "tool_loop",
    "superpower",
    "cancel_turn",
    "cancel_job",
    "fault_401",
    "fault_429",
    "fault_500",
    "fault_timeout",
    "fault_malformed",
    "session_resume",
    "approval_denial",
    "crash_restart",
)
_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out"})
_SECRET_NAME = "SOAK_PLANTED_API_KEY"
_FX1_KEY = "sk-soak-fx1-test-key-aaa111"
_LITE_KEY = "sk-soak-lite-test-key-bbb222"
_DAEMON_PREFIXES = ("fx1-stub-",)
_DAEMON_NAMES = frozenset({"proc-stdout", "proc-stderr"})
_EXPECTED: dict[str, frozenset[str]] = {
    "chat_fx1": frozenset({"ok"}),
    "chat_fx1_lite": frozenset({"ok"}),
    "tool_loop": frozenset({"ok"}),
    "superpower": frozenset({"ok"}),
    "cancel_turn": frozenset({"cancelled"}),
    "cancel_job": frozenset({"ok", "cancelled"}),
    "fault_401": frozenset({"error"}),
    "fault_429": frozenset({"error"}),
    "fault_500": frozenset({"error"}),
    "fault_timeout": frozenset({"budget", "error", "timeout"}),
    "fault_malformed": frozenset({"error"}),
    "session_resume": frozenset({"ok"}),
    "approval_denial": frozenset({"ok", "error"}),
    "crash_restart": frozenset({"ok"}),
}
_BENCH_NOTE = (
    "Qualified SYNTHETIC lane-power at the catalog minimum "
    "(--n-steps 16, --n-seeds 1, one lane, one defect). "
    "Catalog defaults are larger; this record is not a full bench."
)


class SoakUsageError(Exception):
    """CLI usage error. Process exit code is 2."""


@dataclass(frozen=True)
class Bounds:
    """Resource ceilings checked after warm-up. Exceeding one is a failure."""

    fd_growth: int = 32
    nondaemon_growth: int = 0
    daemon_cap: int = 12
    tracemalloc_growth_bytes: int = 96 * 1024 * 1024
    tracemalloc_steady_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ResourceSample:
    """One observation of process resources. ``None`` means inspection failed."""

    fd_count: int | None
    nondaemon_threads: int
    daemon_threads: tuple[str, ...]
    traced_bytes: int
    child_pids: tuple[int, ...] | None
    registry_pids: tuple[int, ...]


@dataclass(frozen=True)
class CycleResult:
    """One workload attempt. ``recovered`` is false when the outcome was not the expected one."""

    kind: str
    status: str
    recovered: bool
    detail: str


def planted_secret(seed: int) -> str:
    """Secret planted in the environment. It must not appear in scratch files."""
    return f"sk-soak-planted-{seed & 0xFFFFFFFF:08x}"


def schedule(seed: int) -> list[str]:
    """Deterministic workload order for ``seed``. Every kind appears once per pass."""
    import random

    order = list(WORKLOADS)
    random.Random(seed).shuffle(order)
    return order


def drive_until(
    duration_s: float,
    kinds: Sequence[str],
    clock: Callable[[], float],
    run_one: Callable[[str], CycleResult],
    *,
    max_cycles: int = 10000,
) -> list[CycleResult]:
    """Run one full pass, then keep going until ``duration_s`` has elapsed."""
    if not kinds:
        return []
    start = clock()
    out: list[CycleResult] = []
    index = 0
    while index < max_cycles:
        if index >= len(kinds) and _elapsed(clock, start) >= duration_s:
            break
        out.append(run_one(kinds[index % len(kinds)]))
        index += 1
    return out


def child_pids(
    pid: int,
    *,
    proc_root: Path | None = None,
    ps_runner: Callable[[], str | None] | None = None,
) -> tuple[int, ...] | None:
    """Direct children of ``pid``. ``None`` means the OS inspection failed."""
    root = Path("/proc") if proc_root is None else proc_root
    found = _proc_children(pid, root)
    if found is not None:
        return found
    runner = _default_ps if ps_runner is None else ps_runner
    return _ps_children(pid, runner)


def count_fds(fd_root: Path | None = None) -> int | None:
    """Open descriptor count from ``/proc/self/fd``. ``None`` when unavailable."""
    root = Path("/proc/self/fd") if fd_root is None else fd_root
    if not root.is_dir():
        return None
    try:
        return sum(1 for _entry in root.iterdir())
    except OSError:
        return None


def thread_census() -> tuple[int, tuple[str, ...]]:
    """``(non-daemon count, sorted daemon names)`` for threads that are alive."""
    nondaemon = 0
    daemons: list[str] = []
    for thread in threading.enumerate():
        if not thread.is_alive():
            continue
        if thread.daemon:
            daemons.append(thread.name)
        else:
            nondaemon += 1
    daemons.sort()
    return nondaemon, tuple(daemons)


def registry_pids(registry: Any) -> tuple[int, ...]:
    """Pids currently in a process registry. A missing ``live`` is a failure sentinel."""
    live = getattr(registry, "live", None)
    if not callable(live):
        return (-1,)
    try:
        rows = live()
    except Exception:
        return (-1,)
    found: list[int] = []
    for row in rows:
        pid = getattr(row, "pid", None)
        if isinstance(pid, int) and not isinstance(pid, bool):
            found.append(pid)
    return tuple(sorted(found))


def leak_problems(
    children: tuple[int, ...] | None,
    owned: Sequence[int],
) -> tuple[str, ...]:
    """Failures for orphan OS children or a non-empty owned-process registry."""
    problems: list[str] = []
    if children is None:
        problems.append("child inspection unavailable")
    else:
        problems.extend(f"orphan child pid {pid}" for pid in children)
    problems.extend(f"registry live pid {pid}" for pid in owned)
    return tuple(problems)


def diff_resources(
    baseline: ResourceSample,
    current: ResourceSample,
    bounds: Bounds,
    *,
    check_alloc: bool,
) -> tuple[str, ...]:
    """Compare a sample with the post-warm-up baseline."""
    problems = list(leak_problems(current.child_pids, current.registry_pids))
    problems.extend(_fd_problems(baseline.fd_count, current.fd_count, bounds.fd_growth))
    problems.extend(_thread_problems(baseline, current, bounds))
    if check_alloc:
        growth = current.traced_bytes - baseline.traced_bytes
        if growth > bounds.tracemalloc_growth_bytes:
            problems.append(f"tracemalloc grew by {growth} bytes")
    return tuple(problems)


def audit_job_views(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Non-terminal jobs fail. Interrupted jobs need a non-empty reason."""
    problems: list[str] = []
    for record in records:
        problem = _job_problem(record)
        if problem:
            problems.append(problem)
    return tuple(problems)


def audit_sessions(root: Path) -> tuple[str, ...]:
    """Every session file must be a JSON object and must not still be running."""
    if not root.is_dir():
        return ("session dir missing",)
    problems: list[str] = []
    for path in sorted(root.glob("*.json")):
        problems.extend(_session_file(path))
    return tuple(problems)


def secret_hits(roots: Sequence[Path], secret: str) -> tuple[str, ...]:
    """Paths under ``roots`` whose bytes contain ``secret``."""
    if not secret:
        return ("empty planted secret",)
    hits: list[str] = []
    token = secret.encode("utf-8")
    for root in roots:
        hits.extend(_scan_tree(root, token))
    return tuple(hits)


def steady_growth_problem(samples: Sequence[int], pass_length: int, bound: int) -> str | None:
    """Growth after the first full pass. The pass itself may load the catalog once."""
    if pass_length < 1 or len(samples) <= pass_length:
        return None
    growth = samples[-1] - samples[pass_length - 1]
    if growth > bound:
        return f"tracemalloc grew by {growth} bytes after the first pass"
    return None


def trim_request_log(server: Any, *, keep: int = 4) -> None:
    """Drop old stub request bodies so the steady check is not the request archive.

    The loopback stub keeps every parsed chat body. Each body includes the
    growing transcript, so the archive is quadratic. The soak only needs the
    queue to drain; a short tail is enough for ``wait_for_requests``.
    """
    if keep < 1:
        return
    requests = getattr(server, "_requests", None)
    if isinstance(requests, list) and len(requests) > keep:
        del requests[:-keep]


def fault_recovered(kind: str, status: str, detail: str = "") -> bool:
    """True when a workload ended in its expected handled state."""
    allowed = _EXPECTED.get(kind)
    if allowed is None or status not in allowed:
        return False
    if kind == "approval_denial":
        return "not approved" in detail
    if kind == "superpower":
        return "lane-power ran" in detail and "SYNTHETIC" in detail
    if kind == "cancel_job":
        return "stopped" in detail and "child observed" in detail
    if kind == "crash_restart":
        return _crash_detail_ok(detail)
    return True


def _crash_detail_ok(detail: str) -> bool:
    """A crash cycle restarted nothing and left no live child."""
    needed = ("interrupted=1", "duplicates=0", "orphans=0")
    return all(token in detail for token in needed)


def lane_power_plan(out_dir: Path) -> str:
    """Model plan selecting the minimum qualified SYNTHETIC lane-power step."""
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


def build_record(
    *,
    seed: int,
    duration_s: float,
    elapsed_s: float,
    warmup_s: float,
    cycles: Sequence[CycleResult],
    failures: Sequence[str],
    resources: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    session_problems: Sequence[str],
    secret_found: Sequence[str],
    bounds: Bounds,
) -> dict[str, Any]:
    """Assemble the ``dipcatcher.soak.v1`` document. ``ok`` is false when anything failed."""
    counts = {kind: 0 for kind in WORKLOADS}
    for cycle in cycles:
        counts[cycle.kind] = counts.get(cycle.kind, 0) + 1
    listed = list(failures)
    orphans = resources.get("orphan_children", [])
    owned = resources.get("registry_live", [])
    failed = bool(listed or session_problems or secret_found or orphans or owned)
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "reference_24h": False,
        "seed": seed,
        "duration_s": duration_s,
        "elapsed_s": round(elapsed_s, 6),
        "warmup_s": round(warmup_s, 6),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "bounds": _bounds_dict(bounds),
        "counts": counts,
        "cycles": [_cycle_dict(cycle) for cycle in cycles],
        "failures": listed,
        "resources": dict(resources),
        "jobs": [dict(job) for job in jobs],
        "session_problems": list(session_problems),
        "secret_hits": list(secret_found),
        "orphan_children": list(orphans),
        "registry_live": list(owned),
        "bench": {
            "command": "lane-power",
            "evidence_class": "SYNTHETIC",
            "parameters": {
                "--defects": "0.0",
                "--lanes": "serial_watch",
                "--n-seeds": 1,
                "--n-steps": 16,
            },
            "note": _BENCH_NOTE,
        },
        "ok": not failed,
    }


def write_record(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically write the evidence document with mode 0600."""
    from fx1.interactive.atomicstore import atomic_write_json

    atomic_write_json(path, dict(record), mode=0o600)


def run_soak(
    *,
    duration_s: float,
    seed: int,
    config_dir: Path,
    session_dir: Path,
    work_dir: Path,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run the mixed workload and return the evidence document."""
    options = _SoakOptions(duration_s, seed, config_dir, session_dir, work_dir, clock)
    driver = _Driver(options)
    try:
        return driver.run()
    except Exception as exc:
        return _crashed_record(options, exc)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Exit 0 when the record is clean, 1 on failures, 2 on usage errors."""
    try:
        options = _parse(argv)
    except SoakUsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    record = run_soak(
        duration_s=options.duration_s,
        seed=options.seed,
        config_dir=options.config_dir,
        session_dir=options.session_dir,
        work_dir=options.work_dir,
    )
    _emit(options.out, record)
    return 0 if record.get("ok") is True else 1


@dataclass(frozen=True)
class _SoakOptions:
    duration_s: float
    seed: int
    config_dir: Path
    session_dir: Path
    work_dir: Path
    clock: Callable[[], float] | None = None
    out: Path | None = None


class _Timeout:
    def __init__(self, value: float) -> None:
        self.value = value


class _Driver:
    """One soak process. Methods stay small so each step can fail on its own."""

    def __init__(self, options: _SoakOptions) -> None:
        self.options = options
        self.bounds = Bounds()
        self.secret = planted_secret(options.seed)
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.server: Any = None
        self.concierge: Any = None
        self.store: Any = None
        self.harness: Any = None
        self.timeout = _Timeout(8.0)
        self.bench_dir = options.config_dir / "runs" / "bench-out"
        self._cwd = Path.cwd()
        self._traced = False
        self._baseline: ResourceSample | None = None
        self._traced_samples: list[int] = []
        self._warmup_s = 0.0
        self._elapsed_s = 0.0

    def run(self) -> dict[str, Any]:
        try:
            self._prepare()
            cycles = self._exercise()
            return self._finish(cycles)
        finally:
            self._cleanup()

    def _prepare(self) -> None:
        _prepare_dirs(self.options)
        _install_env(self.options, self.secret)
        os.chdir(self.options.work_dir)
        self.bench_dir.mkdir(parents=True, exist_ok=True)
        self._start_stub()
        _store_endpoints(self.server)
        self._open_stack()
        self._traced = _start_trace()
        started = time.monotonic()
        status, detail = self._chat("fx1", "warmup pong")
        self._warmup_s = time.monotonic() - started
        if status != "ok":
            self.failures.append(f"warmup: {status}: {detail}")
        self._baseline = self._sample()

    def _exercise(self) -> list[CycleResult]:
        clock = time.monotonic if self.options.clock is None else self.options.clock
        started = time.monotonic()
        cycles = drive_until(
            self.options.duration_s,
            schedule(self.options.seed),
            clock,
            self._run_kind,
        )
        self._elapsed_s = time.monotonic() - started
        return cycles

    def _finish(self, cycles: Sequence[CycleResult]) -> dict[str, Any]:
        self._reconcile()
        sample = self._sample()
        baseline = self._baseline if self._baseline is not None else sample
        self.failures.extend(diff_resources(baseline, sample, self.bounds, check_alloc=True))
        steady = steady_growth_problem(
            self._traced_samples, len(WORKLOADS), self.bounds.tracemalloc_steady_bytes
        )
        if steady:
            self.failures.append(steady)
        jobs = _job_views(self.options.config_dir / "jobs")
        self.failures.extend(audit_job_views(jobs))
        self.failures.extend(_corrupt_jobs(self.options.config_dir / "jobs"))
        sessions = audit_sessions(self.options.session_dir)
        self.failures.extend(sessions)
        hits = secret_hits(
            (self.options.config_dir, self.options.session_dir, self.options.work_dir),
            self.secret,
        )
        self.failures.extend(f"secret in {hit}" for hit in hits)
        resources = _resource_view(baseline, sample)
        return build_record(
            seed=self.options.seed,
            duration_s=self.options.duration_s,
            elapsed_s=self._elapsed_s,
            warmup_s=self._warmup_s,
            cycles=cycles,
            failures=self.failures,
            resources=resources,
            jobs=jobs,
            session_problems=sessions,
            secret_found=hits,
            bounds=self.bounds,
        )

    def _cleanup(self) -> None:
        _shutdown(self.concierge)
        _close_server(self.server)
        _stop_trace(self._traced)
        os.environ.pop(_SECRET_NAME, None)
        os.chdir(self._cwd)

    def _run_kind(self, kind: str) -> CycleResult:
        self.lines.clear()
        try:
            result = _ACTIONS[kind](self)
        except Exception as exc:
            result = CycleResult(kind, "error", False, _safe(f"{type(exc).__name__}: {exc}"))
        self._note_queue(result.kind)
        self._after_cycle(result)
        return result

    def _after_cycle(self, result: CycleResult) -> None:
        if not result.recovered:
            self.failures.append(f"{result.kind}: {result.status}: {result.detail}")
        trim_request_log(self.server)
        baseline = self._baseline
        if baseline is None:
            return
        self.server.clear_requests()
        sample = self._sample()
        self._traced_samples.append(sample.traced_bytes)
        problems = diff_resources(baseline, sample, self.bounds, check_alloc=True)
        self.failures.extend(f"{result.kind}: {problem}" for problem in problems)
        self._reconcile()
        self.failures.extend(
            f"{result.kind}: {problem}" for problem in audit_job_views(_job_views(self._jobs()))
        )

    def _start_stub(self) -> None:
        reply_cls, server_cls = _load_stub()
        self._reply = reply_cls
        server = server_cls({"fx1": _FX1_KEY, "fx1-lite": _LITE_KEY})
        server.start()
        self.server = server

    def _open_stack(self) -> None:
        from fx1.harness import Harness
        from fx1.interactive.concierge import Concierge
        from fx1.interactive.headless import exec_approvals
        from fx1.interactive.sessions import SessionStore

        self.store = SessionStore()
        self.harness = Harness()
        self._approvals = exec_approvals(())
        self.concierge = Concierge(
            "fx1",
            approvals=self._approvals,
            backend_factory=_factory(self.timeout),
            input_fn=_block_input,
            output_fn=self._capture,
            animated=False,
            session_store=self.store,
            harness=self.harness,
        )

    def _capture(self, *args: object, **_kwargs: object) -> None:
        self.lines.append("" if not args else str(args[0]))

    def _sample(self) -> ResourceSample:
        from fx1.interactive.procs import REGISTRY

        gc.collect()
        current, _peak = tracemalloc.get_traced_memory()
        nondaemon, daemons = thread_census()
        return ResourceSample(
            fd_count=count_fds(),
            nondaemon_threads=nondaemon,
            daemon_threads=daemons,
            traced_bytes=current,
            child_pids=child_pids(os.getpid()),
            registry_pids=registry_pids(REGISTRY),
        )

    def _chat(self, model: str, prompt: str) -> tuple[str, str]:
        self._use(model)
        self.concierge.handle_line(prompt)
        return str(self.concierge.last_status), _safe(str(self.concierge.last_answer))

    def _use(self, model: str) -> None:
        if str(self.concierge.active_model) == model:
            return
        self.concierge.handle_line(f"/use {model}")

    def _queue(self, reply: Any) -> None:
        self.server.queue(reply)

    def _ok(self, model: str, text: str) -> Any:
        return self._reply.ok(text, served_model=model, usage=_usage())

    def _run_line(self, line: str, timeout_s: float) -> str:
        box: dict[str, str] = {"status": "error", "detail": ""}

        def target() -> None:
            try:
                self.concierge.handle_line(line)
            except Exception as exc:
                box["status"] = "error"
                box["detail"] = f"{type(exc).__name__}: {exc}"
                return
            box["status"] = str(self.concierge.last_status)
            box["detail"] = str(self.concierge.last_answer)

        thread = threading.Thread(target=target, name="soak-turn")
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            _cancel_running(self.concierge)
            thread.join(5.0)
            return "timeout"
        if box["detail"]:
            self.lines.append(box["detail"])
        return box["status"]

    def _jobs(self) -> Path:
        return self.options.config_dir / "jobs"

    def _reconcile(self) -> None:
        from fx1.interactive.jobs import JobLedger

        root = self._jobs()
        if root.is_dir():
            JobLedger(root).reconcile()

    def action_chat_fx1(self) -> CycleResult:
        return self._named_chat("chat_fx1", "fx1", "say pong")

    def action_chat_lite(self) -> CycleResult:
        return self._named_chat("chat_fx1_lite", "fx1-lite", "say pong lite")

    def _named_chat(self, kind: str, model: str, prompt: str) -> CycleResult:
        status, detail = self._chat(model, prompt)
        return _cycle(kind, status, detail)

    def _note_queue(self, kind: str) -> None:
        queued = getattr(self.server, "_queue", None)
        if isinstance(queued, list) and queued:
            self.failures.append(f"{kind}: stub reply queue was not drained")

    def action_tool_loop(self) -> CycleResult:
        self._use("fx1")
        call = {
            "id": "call_soak_1",
            "type": "function",
            "function": {
                "name": "flash_retrieve",
                "arguments": json.dumps({"query": "qlike"}),
            },
        }
        self._queue(self._reply.ok(None, tool_calls=[call], served_model="fx1", usage=_usage()))
        self._queue(self._ok("fx1", "pong from tools"))
        self.concierge.handle_line("look up the note")
        status = str(self.concierge.last_status)
        detail = _safe("\n".join(self.lines[-6:]))
        return _cycle("tool_loop", status, detail)

    def action_superpower(self) -> CycleResult:
        self._use("fx1")
        self._queue(self._ok("fx1", lane_power_plan(self.bench_dir)))
        self._queue(self._ok("fx1", "SYNTHETIC lane-power bench completed."))
        status = self._run_line("/superpower lane power alarm rate --sync", 25.0)
        detail = _superpower_detail(self.concierge, self.lines)
        return _cycle("superpower", status, detail)

    def action_cancel_turn(self) -> CycleResult:
        self._use("fx1")
        self._queue(self._reply.delay(4.0, self._ok("fx1", "late pong")))
        target = len(self.server.requests) + 1
        status = self._cancel_line("cancel this ping", target)
        detail = _safe(status)
        return _cycle("cancel_turn", status, detail)

    def action_cancel_job(self) -> CycleResult:
        self._use("fx1")
        self._queue(self._ok("fx1", lane_power_plan(self.bench_dir)))
        self.concierge.handle_line("/superpower lane power alarm rate")
        saw_child = _wait_owned(self.concierge, 8.0)
        stopped = self.concierge.cancel_job()
        self.concierge.join_job(5.0)
        if self.concierge.job_running:
            _cancel_running(self.concierge)
            self.concierge.join_job(3.0)
        self.concierge.handle_line("/status")
        status = str(self.concierge.last_status)
        detail = "stopped" if not self.concierge.job_running else "still running"
        if saw_child:
            detail = f"{detail}; child observed"
        if not stopped and "stopped" in detail:
            detail = f"{detail}; cancel found the job already finished"
        return _cycle("cancel_job", status, detail)

    def action_fault(self, kind: str, reply: Any) -> CycleResult:
        self._use("fx1")
        self._queue(reply)
        self.concierge.handle_line(f"fault {kind}")
        status = str(self.concierge.last_status)
        return _cycle(kind, status, _safe("\n".join(self.lines[-4:])))

    def action_401(self) -> CycleResult:
        return self.action_fault("fault_401", self._reply.error(401))

    def action_429(self) -> CycleResult:
        return self.action_fault("fault_429", self._reply.error(429))

    def action_500(self) -> CycleResult:
        return self.action_fault("fault_500", self._reply.error(500))

    def action_malformed(self) -> CycleResult:
        return self.action_fault("fault_malformed", self._reply.malformed())

    def action_timeout(self) -> CycleResult:
        self.timeout.value = 0.35
        try:
            reply = self._reply.delay(3.0, self._ok("fx1", "too late"))
            return self.action_fault("fault_timeout", reply)
        finally:
            self.timeout.value = 8.0

    def action_session_resume(self) -> CycleResult:
        session_id = _session_id(self.concierge)
        if session_id is None:
            self._chat("fx1", "remember pong")
            session_id = _session_id(self.concierge)
        if session_id is None:
            return CycleResult("session_resume", "error", False, "no session id")
        _shutdown(self.concierge)
        self._open_stack()
        self.concierge.handle_line(f"/resume {session_id}")
        resumed = "resumed session" in "\n".join(self.lines)
        status, _detail = self._chat("fx1", "resume pong")
        detail = "resumed" if resumed else "resume text missing"
        recovered = resumed and fault_recovered("session_resume", status, detail)
        return CycleResult("session_resume", status, recovered, detail)

    def action_denial(self) -> CycleResult:
        self.concierge.handle_line("/harness run train")
        detail = _safe("\n".join(self.lines[-6:]))
        status = str(self.concierge.last_status)
        return _cycle("approval_denial", status, detail)

    def action_crash_restart(self) -> CycleResult:
        detail = run_crash_restart(self.options.config_dir, self.options.work_dir)
        status = "ok" if _crash_detail_ok(detail) else "error"
        return _cycle("crash_restart", status, detail)

    def _cancel_line(self, prompt: str, target: int) -> str:
        box: dict[str, str] = {"status": "error"}

        def target_fn() -> None:
            try:
                self.concierge.handle_line(prompt)
            except Exception as exc:
                box["status"] = "error"
                box["detail"] = str(exc)
                return
            box["status"] = str(self.concierge.last_status)

        thread = threading.Thread(target=target_fn, name="soak-turn")
        thread.start()
        self.server.wait_for_requests(target, timeout=3.0)
        self.concierge.cancel_turn()
        thread.join(5.0)
        if thread.is_alive():
            return "timeout"
        return box["status"]


_ACTIONS: dict[str, Callable[[_Driver], CycleResult]] = {}


def _bind_actions() -> None:
    _ACTIONS.update(
        {
            "chat_fx1": _Driver.action_chat_fx1,
            "chat_fx1_lite": _Driver.action_chat_lite,
            "tool_loop": _Driver.action_tool_loop,
            "superpower": _Driver.action_superpower,
            "cancel_turn": _Driver.action_cancel_turn,
            "cancel_job": _Driver.action_cancel_job,
            "fault_401": _Driver.action_401,
            "fault_429": _Driver.action_429,
            "fault_500": _Driver.action_500,
            "fault_timeout": _Driver.action_timeout,
            "fault_malformed": _Driver.action_malformed,
            "session_resume": _Driver.action_session_resume,
            "approval_denial": _Driver.action_denial,
            "crash_restart": _Driver.action_crash_restart,
        }
    )


_bind_actions()


_CRASH_PARENT = r"""
import sys
from pathlib import Path

from fx1.interactive.jobs import JobLedger
from fx1.interactive.procs import ProcLimits, ProcessRegistry, run_owned

config, marker, ready = (Path(arg) for arg in sys.argv[1:4])
ledger = JobLedger(config / "jobs")
job = ledger.create(kind="superpower", goal="bounded crash cycle")
step = ledger.begin_step(
    job.job_id,
    name="sleeper",
    argv=["sleeper"],
    idempotent=True,
    idempotency_key="crash-cycle",
)


def _emit(kind: str, fields: object) -> None:
    if kind != "proc.spawn" or not isinstance(fields, dict):
        return
    pid = fields.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return
    pgid = fields.get("pgid")
    gid = pgid if isinstance(pgid, int) and not isinstance(pgid, bool) else None
    ledger.attach_process(job.job_id, step.step_id, pid=pid, pgid=gid)


_SLEEPER = (
    "import os,sys,time,pathlib\n"
    "p=pathlib.Path(sys.argv[1])\n"
    "p.parent.mkdir(parents=True, exist_ok=True)\n"
    "fd=os.open(p, os.O_APPEND|os.O_CREAT|os.O_WRONLY, 0o600)\n"
    "os.write(fd, (str(os.getpid())+'\\n').encode())\n"
    "os.close(fd)\n"
    "pathlib.Path(sys.argv[2]).write_text('ready', encoding='utf-8')\n"
    "time.sleep(180)\n"
)
run_owned(
    [sys.executable, "-c", _SLEEPER, str(marker), str(ready)],
    ProcLimits(timeout_s=180, kill_grace_s=1.0),
    registry=ProcessRegistry(),
    emit=_emit,
)
"""

_CRASH_RESUME = r"""
import os
import sys
from pathlib import Path

def _live(pid: int) -> bool:
    path = Path(f"/proc/{pid}/stat")
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    parts = raw.rpartition(")")[2].split()
    return bool(parts) and parts[0] not in {"Z", "X"}

config = Path(sys.argv[1])
marker = Path(sys.argv[2])
from fx1.interactive.jobs import JobLedger

report = JobLedger(config / "jobs").reconcile()
text = marker.read_text(encoding="utf-8") if marker.is_file() else ""
lines = [line for line in text.splitlines() if line.strip().isdigit()]
live = [int(line) for line in lines if _live(int(line))]
for pid in live:
    try:
        os.kill(pid, 9)
    except OSError:
        pass
dup = 0 if len(lines) == 1 else max(len(lines), 1)
print(
    f"interrupted={len(report.interrupted)} duplicates={dup} "
    f"orphans={len(live)} reported={len(report.orphan_running)}"
)
"""


def run_crash_restart(config: Path, work: Path) -> str:
    """SIGKILL a child mid-step, then reconcile. One execution, no live orphan."""
    token = f"{os.getpid()}-{time.monotonic_ns()}"
    marker = work / f"crash-marker-{token}"
    ready = work / f"crash-ready-{token}"
    err_path = work / f"crash-err-{token}"
    parent = _spawn_crash(config, marker, ready, err_path)
    try:
        return _finish_crash(parent, config, marker, ready, err_path)
    finally:
        _reap_crash(parent, marker)


def _spawn_crash(config: Path, marker: Path, ready: Path, err_path: Path) -> subprocess.Popen[str]:
    err = err_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [sys.executable, "-c", _CRASH_PARENT, str(config), str(marker), str(ready)],
            env=_crash_env(config),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err,
            text=True,
            start_new_session=True,
        )
    except Exception:
        err.close()
        raise


def _crash_env(config: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(config),
        "FX1_CONFIG_DIR": str(config),
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _finish_crash(
    parent: subprocess.Popen[str],
    config: Path,
    marker: Path,
    ready: Path,
    err_path: Path,
) -> str:
    if not _wait_flag(ready, parent):
        return _crash_failure(parent, err_path, "ready timeout")
    if not _wait_running_job(config / "jobs", parent):
        return _crash_failure(parent, err_path, "job was not running")
    _signal_pid(parent.pid, signal.SIGKILL)
    _wait_exit(parent)
    return _resume_crash(config, marker, err_path)


def _wait_flag(flag: Path, parent: subprocess.Popen[str]) -> bool:
    return _wait_pred(lambda: flag.is_file(), parent)


def _wait_running_job(root: Path, parent: subprocess.Popen[str]) -> bool:
    return _wait_pred(lambda: _job_has_pid(root), parent)


def _wait_pred(pred: Callable[[], bool], parent: subprocess.Popen[str]) -> bool:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if pred():
            return True
        if parent.poll() is not None:
            return False
        _pause(deadline)
    return False


def _job_has_pid(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(_running_with_pid(path) for path in root.glob("j*.json"))


def _running_with_pid(path: Path) -> bool:
    payload = _read_obj(path)
    if payload.get("state") != "running":
        return False
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return False
    return any(_step_pid(step) for step in steps)


def _step_pid(step: object) -> bool:
    if not isinstance(step, dict):
        return False
    pid = step.get("pid")
    return isinstance(pid, int) and not isinstance(pid, bool) and pid > 0


def _read_obj(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _resume_crash(config: Path, marker: Path, err_path: Path) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CRASH_RESUME, str(config), str(marker)],
            env=_crash_env(config),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "resume timeout"
    if proc.returncode != 0:
        tail = _clip_text(proc.stderr or _read_text(err_path))
        return f"resume failed {proc.returncode} {tail}"
    summary = proc.stdout.strip()
    return summary or "resume produced no summary"


def _crash_failure(parent: subprocess.Popen[str], err_path: Path, reason: str) -> str:
    _signal_pid(parent.pid, signal.SIGKILL)
    _wait_exit(parent)
    return f"{reason} parent={parent.poll()} {_clip_text(_read_text(err_path))}"


def _reap_crash(parent: subprocess.Popen[str], marker: Path) -> None:
    if parent.poll() is None:
        _signal_pid(parent.pid, signal.SIGKILL)
    _wait_exit(parent)
    _kill_marker(marker)
    err = parent.stderr
    if err is not None:
        err.close()


def _kill_marker(marker: Path) -> None:
    for line in _read_text(marker).splitlines():
        if line.strip().isdigit():
            _signal_pid(int(line), signal.SIGKILL)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _clip_text(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > 240:
        return flat[:237] + "..."
    return flat


def _signal_pid(pid: int | None, sig: int) -> None:
    if pid is None or pid <= 0:
        return
    try:
        os.kill(pid, sig)
    except OSError:
        return


def _wait_exit(parent: subprocess.Popen[str]) -> None:
    try:
        parent.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return


def _pause(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(0.05, remaining))


def _elapsed(clock: Callable[[], float], start: float) -> float:
    value = clock() - start
    if value < 0:
        return 0.0
    return value


def _proc_children(pid: int, root: Path) -> tuple[int, ...] | None:
    if not root.is_dir():
        return None
    found: list[int] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return None
    for entry in entries:
        child = _child_pid(entry, pid)
        if child is not None:
            found.append(child)
    return tuple(sorted(found))


def _child_pid(entry: Path, parent: int) -> int | None:
    if not entry.name.isdigit():
        return None
    ppid = _read_ppid(entry / "status")
    if ppid != parent:
        return None
    return int(entry.name)


def _read_ppid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("PPid:"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _default_ps() -> str | None:
    try:
        proc = subprocess.run(
            ["ps", "-o", "pid=,ppid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _ps_children(pid: int, runner: Callable[[], str | None]) -> tuple[int, ...] | None:
    try:
        text = runner()
    except Exception:
        return None
    if text is None:
        return None
    return _parse_ps(text, pid)


def _parse_ps(text: str, parent: int) -> tuple[int, ...]:
    found: list[int] = []
    for line in text.splitlines():
        pid = _ps_child(line, parent)
        if pid is not None:
            found.append(pid)
    return tuple(sorted(found))


def _ps_child(line: str, parent: int) -> int | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        pid = int(parts[0])
        ppid = int(parts[1])
    except ValueError:
        return None
    if ppid != parent:
        return None
    return pid


def _fd_problems(before: int | None, after: int | None, bound: int) -> tuple[str, ...]:
    if before is None or after is None:
        return ("fd inspection unavailable",)
    growth = after - before
    if growth > bound:
        return (f"fd count grew by {growth}",)
    return ()


def _thread_problems(
    baseline: ResourceSample, current: ResourceSample, bounds: Bounds
) -> tuple[str, ...]:
    problems: list[str] = []
    growth = current.nondaemon_threads - baseline.nondaemon_threads
    if growth > bounds.nondaemon_growth:
        problems.append(f"non-daemon threads grew by {growth}")
    problems.extend(_daemon_problems(current.daemon_threads, bounds.daemon_cap))
    return tuple(problems)


def _daemon_problems(names: Sequence[str], cap: int) -> tuple[str, ...]:
    problems: list[str] = []
    if len(names) > cap:
        problems.append(f"daemon threads {len(names)} exceed cap {cap}")
    for name in names:
        if _daemon_allowed(name):
            continue
        if name.startswith("dip-"):
            problems.append(f"background job thread still alive: {name}")
        else:
            problems.append(f"unexpected daemon thread: {name}")
    return tuple(problems)


def _daemon_allowed(name: str) -> bool:
    if name in _DAEMON_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in _DAEMON_PREFIXES)


def _job_problem(record: Mapping[str, Any]) -> str | None:
    job_id = str(record.get("job_id", "?"))
    state = str(record.get("state", ""))
    reason = str(record.get("error") or record.get("reason") or "").strip()
    problem = _state_problem(job_id, state, reason)
    if problem:
        return problem
    for step in _steps(record):
        step_problem = _step_problem(job_id, step, reason)
        if step_problem:
            return step_problem
    return None


def _state_problem(job_id: str, state: str, reason: str) -> str | None:
    if state == "interrupted":
        if reason:
            return None
        return f"{job_id}: interrupted without a reason"
    if state not in _TERMINAL:
        return f"{job_id}: non-terminal state {state or 'missing'}"
    return None


def _steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = record.get("steps") or []
    if not isinstance(raw, list):
        return []
    return [step for step in raw if isinstance(step, Mapping)]


def _step_problem(job_id: str, step: Mapping[str, Any], reason: str) -> str | None:
    state = str(step.get("state", ""))
    detail = str(step.get("detail") or "").strip()
    if state == "interrupted":
        if detail or reason:
            return None
        return f"{job_id}: step interrupted without a reason"
    if state not in _TERMINAL:
        return f"{job_id}: step non-terminal state {state or 'missing'}"
    return None


def _session_file(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (f"{path.name}: invalid JSON ({type(exc).__name__})",)
    if not isinstance(payload, dict):
        return (f"{path.name}: not an object",)
    return _session_job(path.name, payload.get("job"))


def _session_job(name: str, job: Any) -> tuple[str, ...]:
    if not isinstance(job, dict):
        return ()
    state = job.get("state")
    if state == "running":
        return (f"{name}: job still running",)
    if state == "interrupted" and not str(job.get("resume") or "").strip():
        return (f"{name}: interrupted without a reason",)
    return ()


def _scan_tree(root: Path, token: bytes) -> list[str]:
    if not root.exists():
        return []
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and _contains(path, token):
            hits.append(str(path))
    return hits


def _contains(path: Path, token: bytes) -> bool:
    try:
        return token in path.read_bytes()
    except OSError:
        return False


def _bounds_dict(bounds: Bounds) -> dict[str, int]:
    return {
        "daemon_cap": bounds.daemon_cap,
        "fd_growth": bounds.fd_growth,
        "nondaemon_growth": bounds.nondaemon_growth,
        "tracemalloc_growth_bytes": bounds.tracemalloc_growth_bytes,
        "tracemalloc_steady_bytes": bounds.tracemalloc_steady_bytes,
    }


def _cycle_dict(cycle: CycleResult) -> dict[str, Any]:
    return {
        "detail": cycle.detail,
        "kind": cycle.kind,
        "recovered": cycle.recovered,
        "status": cycle.status,
    }


def _cycle(kind: str, status: str, detail: str) -> CycleResult:
    safe = _safe(detail)
    return CycleResult(kind, status, fault_recovered(kind, status, safe), safe)


def _safe(text: str) -> str:
    from fx1.interactive.redact import redact_text

    cleaned = str(redact_text(text)).replace("\n", " ")
    if len(cleaned) > 400:
        return cleaned[:397] + "..."
    return cleaned


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


def _prepare_dirs(options: _SoakOptions) -> None:
    options.config_dir.mkdir(parents=True, exist_ok=True)
    options.session_dir.mkdir(parents=True, exist_ok=True)
    options.work_dir.mkdir(parents=True, exist_ok=True)


def _install_env(options: _SoakOptions, secret: str) -> None:
    os.environ["FX1_CONFIG_DIR"] = str(options.config_dir)
    os.environ["FX1_SESSION_DIR"] = str(options.session_dir)
    os.environ["NO_COLOR"] = "1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.pop("MOONSHOT_API_KEY", None)
    os.environ.pop("FX1_BASE_URL", None)
    os.environ[_SECRET_NAME] = secret


def _store_endpoints(server: Any) -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint("fx1", _FX1_KEY, server.url)
    set_endpoint("fx1-lite", _LITE_KEY, server.url)


def _factory(box: _Timeout) -> Callable[..., Any]:
    def factory(**kwargs: Any) -> Any:
        from fx1.interactive.model_client import ModelClient, RetryPolicy

        return ModelClient(timeout_s=box.value, retry=RetryPolicy(max_attempts=1), **kwargs)

    return factory


def _block_input(_prompt: str = "", *_args: object, **_kwargs: object) -> str:
    raise RuntimeError("soak does not read interactive prompts")


def _load_stub() -> tuple[Any, Any]:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.fx1.stub_model_server import Reply, StubModelServer

    return Reply, StubModelServer


def _start_trace() -> bool:
    if tracemalloc.is_tracing():
        return False
    tracemalloc.start()
    return True


def _stop_trace(started: bool) -> None:
    if started and tracemalloc.is_tracing():
        tracemalloc.stop()


def _shutdown(concierge: Any) -> None:
    if concierge is None:
        return
    try:
        concierge.shutdown(timeout_s=2.0)
    except Exception:
        return


def _close_server(server: Any) -> None:
    if server is None:
        return
    try:
        server.close()
    except Exception:
        return


def _cancel_running(concierge: Any) -> None:
    from fx1.interactive.procs import REGISTRY

    cancel_turn = getattr(concierge, "cancel_turn", None)
    if callable(cancel_turn):
        cancel_turn()
    cancel_job = getattr(concierge, "cancel_job", None)
    if callable(cancel_job):
        cancel_job()
    holder = getattr(concierge, "_superpower_cancel", None)
    if isinstance(holder, dict):
        event = holder.get("cancel")
        if isinstance(event, threading.Event):
            event.set()
    try:
        REGISTRY.terminate_all(0.2)
    except Exception:
        return


def _wait_owned(concierge: Any, timeout_s: float) -> bool:
    from fx1.interactive.procs import REGISTRY

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if registry_pids(REGISTRY) or child_pids(os.getpid()):
            return True
        if not bool(getattr(concierge, "job_running", False)):
            return False
        time.sleep(0.05)
    return bool(registry_pids(REGISTRY) or child_pids(os.getpid()))


def _session_id(concierge: Any) -> str | None:
    record = getattr(concierge, "_session", None)
    if not isinstance(record, dict):
        return None
    ident = record.get("id")
    if isinstance(ident, str) and ident:
        return ident
    return None


def _superpower_detail(concierge: Any, lines: Sequence[str]) -> str:
    parts = [_run_blob(run) for run in _runs(concierge)]
    parts.extend(lines[-8:])
    return " ".join(part for part in parts if part)


def _runs(concierge: Any) -> list[Any]:
    result = getattr(concierge, "last_result", None)
    runs = getattr(result, "runs", None)
    if isinstance(runs, list):
        return list(runs)
    return []


def _run_blob(run: Any) -> str:
    ident = str(getattr(run, "id", ""))
    status = str(getattr(run, "status", ""))
    detail = str(getattr(run, "detail", ""))
    return f"{ident} {status} {detail}"


def _job_views(root: Path) -> list[dict[str, Any]]:
    from fx1.interactive.jobs import JobLedger

    if not root.is_dir():
        return []
    views: list[dict[str, Any]] = []
    for record in JobLedger(root).list(limit=500):
        views.append(_view(record))
    return views


def _view(record: Any) -> dict[str, Any]:
    return {
        "error": record.error or "",
        "job_id": record.job_id,
        "state": str(record.state),
        "steps": [{"detail": step.detail, "state": str(step.state)} for step in record.steps],
    }


def _corrupt_jobs(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    names = [path.name for path in root.iterdir() if ".corrupt-" in path.name]
    return tuple(f"corrupt job file {name}" for name in sorted(names))


def _resource_view(baseline: ResourceSample, current: ResourceSample) -> dict[str, Any]:
    children = [] if current.child_pids is None else list(current.child_pids)
    return {
        "baseline_fd": baseline.fd_count,
        "baseline_traced_bytes": baseline.traced_bytes,
        "child_inspection": current.child_pids is not None,
        "fd": current.fd_count,
        "fd_delta": _delta(baseline.fd_count, current.fd_count),
        "nondaemon_threads": current.nondaemon_threads,
        "daemon_threads": list(current.daemon_threads),
        "orphan_children": children,
        "registry_live": list(current.registry_pids),
        "traced_bytes": current.traced_bytes,
        "traced_delta": current.traced_bytes - baseline.traced_bytes,
    }


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _crashed_record(options: _SoakOptions, exc: BaseException) -> dict[str, Any]:
    failure = _safe(f"{type(exc).__name__}: {exc}")
    return build_record(
        seed=options.seed,
        duration_s=options.duration_s,
        elapsed_s=0.0,
        warmup_s=0.0,
        cycles=(),
        failures=(failure,),
        resources={"orphan_children": [], "registry_live": []},
        jobs=(),
        session_problems=(),
        secret_found=(),
        bounds=Bounds(),
    )


def _parse(argv: Sequence[str] | None) -> _SoakOptions:
    parser = argparse.ArgumentParser(prog="dipcatcher_soak")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _require_duration(float(args.duration_s))
    config, sessions, work = _dirs(args.config_dir, args.session_dir, args.work_dir)
    return _SoakOptions(
        duration_s=float(args.duration_s),
        seed=int(args.seed),
        config_dir=config,
        session_dir=sessions,
        work_dir=work,
        out=args.out,
    )


def _require_duration(value: float) -> None:
    if value <= 0 or value > 86400:
        raise SoakUsageError("duration-s must be greater than 0 and at most 86400")


def _dirs(config: Path | None, sessions: Path | None, work: Path | None) -> tuple[Path, Path, Path]:
    import tempfile

    if config is None or sessions is None or work is None:
        root = Path(tempfile.mkdtemp(prefix="dipcatcher-soak-"))
        config = root / "config" if config is None else config
        sessions = root / "sessions" if sessions is None else sessions
        work = root / "work" if work is None else work
    return config, sessions, work


def _emit(path: Path | None, record: Mapping[str, Any]) -> None:
    if path is None:
        json.dump(record, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    write_record(path, record)


if __name__ == "__main__":
    sys.exit(main())
