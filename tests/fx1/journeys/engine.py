"""Run declarative chat journeys through the real concierge.

The suite is scripted and hermetic: a loopback stub stands in for fx1 and
fx1-lite. It is not the moderated five-user trial. See README.md for the DSL.

``python -m tests.fx1.journeys.engine --freeze`` writes ``journeys.sha256``.
Do that only when the suite has 100 journeys; until the file exists the gate
treats the suite as unfrozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.approvals import ApprovalDecision, ApprovalGate, ApprovalRequest
from fx1.interactive.concierge import Concierge
from fx1.interactive.jobs import JobLedger, JobState
from fx1.interactive.model_client import ModelClient, RetryPolicy
from fx1.interactive.redact import redact_text
from fx1.interactive.sessions import SessionStore
from fx1.webresearch.engine import ResearchProgress, ResearchReport, SourceDoc
from tests.fx1.capabilities_fixture import sample_catalog_document
from tests.fx1.journeys.schema import (
    CATEGORIES,
    OUT_OF_CHAT_OPS,
    ApprovalExpect,
    ApprovalScript,
    CommandScript,
    EvidenceExpect,
    Expectation,
    FileExpect,
    FinalState,
    Journey,
    ReplySpec,
    SessionExpect,
    Setup,
    Step,
    load_journeys,
)
from tests.fx1.stub_model_server import Reply, StubModelServer

SCOPE = (
    "Scripted automated chat-journey suite. This measures scripted completion "
    "through the concierge against a loopback model stub. It is not the moderated "
    "five-user trial and it does not retain terminal recordings."
)
_FREEZE_NAME = "journeys.sha256"
_MODELS = ("fx1", "fx1-lite")
_SLOW_S = 0.2
_CANCEL_DELAY_FLOOR_S = 0.4
_TURN_JOIN_S = 2.0
_JOB_JOIN_S = 2.0
_TINY_RETRY = RetryPolicy(
    max_attempts=2,
    base_delay_s=0.0,
    max_delay_s=0.0,
    jitter=0.0,
    honor_retry_after_up_to_s=0.0,
)
_USAGE = {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}
_VIOLATION_TEXT = {
    "forbidden_headline": "The strategy Sharpe is 2.4 and that is the headline result.",
    "live_claim": "This book shows live trading profits.",
    "unlabeled_synthetic": "The synthetic result shows accuracy 0.91 on the shard.",
}
_ENDPOINT_ENV = (
    "MOONSHOT_API_KEY",
    "FX1_BASE_URL",
    "DIPCATCHER_FX1_API_KEY",
    "DIPCATCHER_FX1_BASE_URL",
    "DIPCATCHER_FX1_LITE_API_KEY",
    "DIPCATCHER_FX1_LITE_BASE_URL",
)
_SECRET_ENV = re.compile(r"(?i)(^|_)(api[_-]?key|token|secret|password|credential)($|_)")
_CONTROLS = re.compile(r"\x1b|\x9b|\x9d|\x9e|\x9f|[\x00-\x08\x0b\x0c\x0e-\x1a]")
_SURFACE = "dipcatcher.family_surface.v1"


@dataclass(frozen=True)
class JourneyResult:
    """One journey's outcome. ``failures`` is empty when ``passed`` is true."""

    id: str
    category: str
    title: str
    critical: bool
    in_chat: bool
    passed: bool
    failures: tuple[str, ...]
    elapsed_s: float


@dataclass(frozen=True)
class SuiteReport:
    """Aggregate result. ``scope`` states this is not the five-user trial."""

    results: tuple[JourneyResult, ...]
    digest: str
    frozen: bool
    scope: str
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready report: counts, failures, digest, and the scope sentence."""

        failing = [item for item in self.results if not item.passed]
        return {
            "categories": category_counts(self.results),
            "critical_passing": sum(1 for item in self.results if item.critical and item.passed),
            "critical_total": sum(1 for item in self.results if item.critical),
            "digest": self.digest,
            "elapsed_s": round(self.elapsed_s, 3),
            "failures": [_failure_row(item) for item in failing],
            "frozen": self.frozen,
            "in_chat_passing": sum(1 for item in self.results if item.in_chat and item.passed),
            "in_chat_total": sum(1 for item in self.results if item.in_chat),
            "passed": sum(1 for item in self.results if item.passed),
            "results": [_result_row(item) for item in self.results],
            "scope": self.scope,
            "total": len(self.results),
        }


@dataclass(frozen=True)
class InvariantView:
    """Inputs for the checks that run on every journey."""

    chunks: tuple[str, ...]
    logs: str
    blobs: tuple[tuple[str, str], ...]
    json_errors: tuple[str, ...]
    secrets: tuple[str, ...]
    children: tuple[int, ...]
    child_envs: tuple[str, ...]


@dataclass(frozen=True)
class _Freeze:
    digest: str
    count: int
    files: tuple[tuple[str, str], ...]


class _Result:
    """Object the harness and catalog loaders both understand."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = False
        self.cancelled = False
        self.truncated = False


class _ScriptedHarness:
    """Fake lab: catalog fixture, family surface, and per-command scripts."""

    def __init__(self, commands: Mapping[str, CommandScript], cwd: Path) -> None:
        self._commands = dict(commands)
        self._cwd = cwd

    def run(self, name: str, extra_args: list[str] | None = None, **_kwargs: Any) -> _Result:
        args = list(extra_args or [])
        script = self._commands.get(name)
        if script is not None:
            return self._scripted(name, script)
        if name == "capabilities":
            return _Result(0, json.dumps(sample_catalog_document()), "")
        if name == "families":
            return _Result(0, _families_stdout(args), "")
        return _Result(0, _default_stdout(name), "")

    def _scripted(self, name: str, script: CommandScript) -> _Result:
        _write_receipt(self._cwd, script)
        stdout = _default_stdout(name) if script.stdout is None else script.stdout
        return _Result(script.exit_code, stdout, script.stderr)


class _RecordingHarness:
    """Records child env mappings passed into ``run``."""

    def __init__(self, inner: Any, sink: list[str]) -> None:
        self.inner = inner
        self._sink = sink

    def run(self, name: str, extra_args: list[str] | None = None, **kwargs: Any) -> Any:
        env = kwargs.get("env_extra")
        if isinstance(env, Mapping):
            self._sink.append(_mapping_text(env))
        return self.inner.run(name, extra_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _Researcher:
    """Returns scripted pages. ``hold`` waits until cancel, bounded by 2s."""

    def __init__(self, sources: Sequence[SourceDoc], hold: bool) -> None:
        self._sources = list(sources)
        self._hold = hold
        self.goals: list[str] = []

    def redirect(self, new_goal: str) -> None:
        del new_goal

    def run(
        self,
        goal: str,
        *,
        budget: Any = None,
        queries: Any = None,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[ResearchProgress], None] | None = None,
    ) -> ResearchReport:
        del budget, queries
        self.goals.append(goal)
        if on_progress is not None:
            on_progress(ResearchProgress(event="started", detail=goal))
        cancelled = _wait_cancel(self._hold, cancel_event)
        if cancelled and on_progress is not None:
            on_progress(ResearchProgress(event="cancelled", detail="research cancelled"))
        return ResearchReport(goal=goal, cancelled=cancelled, sources=list(self._sources))


class _LogSink(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.append(record.getMessage())
        except Exception:
            return


class _Env:
    def __init__(self, config: Path, session: Path, cwd: Path) -> None:
        self._config = config
        self._session = session
        self._cwd = cwd
        self._old_env: dict[str, str] = {}
        self._old_cwd = Path.cwd()
        self._installed = False

    def install(self) -> None:
        self._old_env = dict(os.environ)
        self._old_cwd = Path.cwd()
        self._installed = True
        os.chdir(self._cwd)
        os.environ["FX1_CONFIG_DIR"] = str(self._config)
        os.environ["FX1_SESSION_DIR"] = str(self._session)
        os.environ["NO_COLOR"] = "1"
        for name in list(os.environ):
            if name in _ENDPOINT_ENV or _is_secret_env(name):
                os.environ.pop(name, None)

    def restore(self) -> None:
        if not self._installed:
            return
        os.chdir(self._old_cwd)
        os.environ.clear()
        os.environ.update(self._old_env)
        self._installed = False


def package_dir() -> Path:
    """Directory that holds the journey YAML files."""

    return Path(__file__).resolve().parent


def yaml_paths(directory: Path | None = None) -> tuple[Path, ...]:
    """Sorted ``*.yaml`` journey files."""

    root = directory or package_dir()
    return tuple(sorted(path for path in root.glob("*.yaml") if path.is_file()))


def fingerprint_tree(directory: Path | None = None) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Overall digest plus ``(filename, sha256)`` in filename order."""

    outer = hashlib.sha256()
    rows: list[tuple[str, str]] = []
    for path in yaml_paths(directory):
        data = path.read_bytes()
        file_hex = hashlib.sha256(data).hexdigest()
        rows.append((path.name, file_hex))
        outer.update(path.name.encode("utf-8"))
        outer.update(b"\0")
        outer.update(data)
        outer.update(b"\0")
    return outer.hexdigest(), tuple(rows)


def journey_digest(directory: Path | None = None) -> str:
    """SHA-256 of the sorted journey YAML bytes."""

    digest, _rows = fingerprint_tree(directory)
    return digest


def write_freeze(directory: Path | None = None, dest: Path | None = None) -> Path:
    """Write the freeze file. Creating it turns on the 100-journey gate."""

    root = directory or package_dir()
    target = dest or (root / _FREEZE_NAME)
    digest, rows = fingerprint_tree(root)
    lines = [
        "# dipcatcher chat-journey freeze",
        "# Scripted automated suite only. Not the moderated five-user trial.",
        "algorithm sha256",
        f"digest {digest}",
        f"count {len(rows)}",
    ]
    lines.extend(f"{name} {file_hex}" for name, file_hex in rows)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def child_pids() -> set[int]:
    """Direct child pids of this process. Empty when ``/proc`` is unavailable."""

    me = os.getpid()
    found: set[int] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        pid = _pid_name(entry)
        if pid is not None and _ppid(entry) == me:
            found.add(pid)
    return found


def unexpected_children(before: set[int]) -> list[int]:
    """Child pids that appeared after ``before`` and are still alive."""

    return sorted(pid for pid in child_pids() if pid not in before)


def invariant_problems(view: InvariantView) -> list[str]:
    """Global checks: display, secrets, JSON, and orphan children."""

    problems: list[str] = []
    _scan_display(view.chunks, problems)
    _scan_secrets(view, problems)
    problems.extend(view.json_errors)
    for pid in view.children:
        problems.append(f"orphan child process {pid}")
    return problems


def make_harness(setup: Setup, cwd: Path, child_envs: list[str]) -> _RecordingHarness:
    """Fake catalog harness, or the real ``Harness`` when ``setup`` says so."""

    if setup.harness_mode == "real":
        from fx1.harness import Harness

        inner: Any = Harness()
    else:
        inner = _ScriptedHarness(dict(setup.commands), cwd)
    return _RecordingHarness(inner, child_envs)


def run_journey(journey: Journey, *, root: Path | None = None) -> JourneyResult:
    """Execute one journey under ``root`` (a private temp dir when omitted)."""

    if root is None:
        with tempfile.TemporaryDirectory(prefix="dip-journey-") as raw:
            return _execute(journey, Path(raw))
    return _execute(journey, root)


def run_suite(directory: Path | None = None, *, report_path: Path | None = None) -> SuiteReport:
    """Run every YAML journey once and optionally write the JSON report."""

    root = directory or package_dir()
    journeys = load_journeys(root)
    started = time.monotonic()
    results = tuple(run_journey(journey) for journey in journeys)
    digest, _rows = fingerprint_tree(root)
    report = SuiteReport(
        results=results,
        digest=digest,
        frozen=(root / _FREEZE_NAME).is_file(),
        scope=SCOPE,
        elapsed_s=time.monotonic() - started,
    )
    _write_report(report, report_path)
    return report


def evaluate(report: SuiteReport, *, frozen: bool) -> list[str]:
    """Gate rules. Unfrozen suites must be entirely green and at most 100 long."""

    problems: list[str] = []
    _check_count(problems, len(report.results), frozen=frozen)
    _check_passes(problems, report, frozen=frozen)
    _check_critical(problems, report)
    if frozen:
        _check_categories(problems, report)
    return problems


def gate_problems(report: SuiteReport, directory: Path | None = None) -> list[str]:
    """Freeze digest plus :func:`evaluate`. A present freeze file means frozen."""

    root = directory or package_dir()
    digest, rows = fingerprint_tree(root)
    freeze = _read_freeze(root / _FREEZE_NAME)
    problems = _freeze_problems(freeze, digest, rows)
    problems.extend(evaluate(report, frozen=freeze is not None))
    return problems


def category_counts(results: Sequence[JourneyResult]) -> dict[str, dict[str, int]]:
    """Per-category totals. Missing P1 categories are present with zeros."""

    counts = {name: _blank_count() for name in CATEGORIES}
    for name, bucket in _tally(results).items():
        counts[name] = bucket
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``--list``, ``--freeze``, or run the suite and print gate problems."""

    parser = argparse.ArgumentParser(prog="python -m tests.fx1.journeys.engine")
    parser.add_argument("--freeze", action="store_true", help="write journeys.sha256")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--list", action="store_true", help="print journey ids")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.list:
        return _print_list()
    if args.freeze:
        return _print_freeze()
    report = run_suite(report_path=args.report)
    problems = gate_problems(report)
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


class _Runner:
    def __init__(self, journey: Journey, root: Path) -> None:
        self.journey = journey
        self.root = root
        self.config = root / "config"
        self.session_dir = root / "sessions"
        self.cwd = root / "cwd"
        self.out: list[str] = []
        self.logs: list[str] = []
        self.child_envs: list[str] = []
        self.failures: list[str] = []
        self.planted: list[str] = []
        self.stored: dict[str, str] = {}
        self.stub_keys: dict[str, str] = {}
        self.table: dict[str, str] = {}
        self._secret_queue: list[str] = []
        self._input_queue: list[str] = []
        self._before: set[int] = set()
        self.elapsed = 0.0
        self.concierge: Any = None
        self.stub: StubModelServer | None = None
        self.sessions: SessionStore | None = None
        self.flash: Any = None
        self.harness: Any = None
        self.researcher: _Researcher | None = None
        self._env = _Env(self.config, self.session_dir, self.cwd)
        self._closed = False
        self._getpass: Any = None
        self._secho: Any = None
        self._popen: Any = None
        self._log_handler: logging.Handler | None = None
        self._log_level = logging.WARNING

    def go(self) -> None:
        self._prepare_dirs()
        self._env.install()
        self._assign_keys()
        self._open_stub()
        self._fill_table()
        self._collect_planted()
        self._install_taps()
        self._before = child_pids()
        self._plant()
        self._open_concierge()
        self._startup()
        self._guard_in_chat()
        self._steps()
        self._final()

    def invariants(self) -> None:
        view = InvariantView(
            chunks=tuple(self.out),
            logs="\n".join(self.logs),
            blobs=tuple(_secret_blobs(self)),
            json_errors=tuple(_json_errors(self.config, self.session_dir)),
            secrets=tuple(self.planted),
            children=tuple(unexpected_children(self._before)),
            child_envs=tuple(self.child_envs),
        )
        for problem in invariant_problems(view):
            self.fail(problem)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _quiet(self._shutdown)
        _quiet(self._close_stub)
        _quiet(self._restore_taps)
        _quiet(self._kill_new_children)
        _quiet(self._env.restore)

    def fail(self, message: str) -> None:
        if message not in self.failures:
            self.failures.append(message)

    def safe(self, exc: BaseException) -> str:
        text = str(redact_text(str(exc)))
        for secret in self.planted:
            text = text.replace(secret, "[REDACTED]")
        return text[:200]

    def to_result(self) -> JourneyResult:
        journey = self.journey
        return JourneyResult(
            id=journey.id,
            category=journey.category,
            title=journey.title,
            critical=journey.critical,
            in_chat=journey.in_chat,
            passed=not self.failures,
            failures=tuple(self.failures),
            elapsed_s=self.elapsed,
        )

    def _prepare_dirs(self) -> None:
        self.config.mkdir(parents=True)
        self.session_dir.mkdir(parents=True)
        self.cwd.mkdir(parents=True)

    def _assign_keys(self) -> None:
        accept = dict(self.journey.setup.accept_keys)
        for model in _MODELS:
            good = accept.get(model) or f"jrnk-{_slug(model)}-{os.urandom(8).hex()}"
            if self._mode(model) == "bad_key":
                self.stored[model] = f"jrnk-bad-{_slug(model)}-{os.urandom(8).hex()}"
                self.stub_keys[model] = good
            else:
                self.stored[model] = good
                if self._mode(model) != "unknown_model":
                    self.stub_keys[model] = good

    def _open_stub(self) -> None:
        self.stub = StubModelServer(self.stub_keys)
        self.stub.start()

    def _fill_table(self) -> None:
        stub = self._require_stub()
        self.table = {
            "stub_url": stub.url,
            "stub_base": stub.base_url,
            "key_fx1": self.stored["fx1"],
            "key_lite": self.stored["fx1-lite"],
        }

    def _collect_planted(self) -> None:
        values = list(self.stored.values()) + list(self.stub_keys.values())
        for step in self.journey.steps:
            values.extend(self._sub(item) for item in step.secrets)
        self.planted = _dedupe(values)

    def _plant(self) -> None:
        self._plant_endpoints()
        self._queue_replies()
        self._install_default()
        self._plant_flash()
        self._plant_sessions()
        self._plant_jobs()
        self.harness = make_harness(self.journey.setup, self.cwd, self.child_envs)
        self.researcher = _Researcher(
            _web_docs(self.journey.setup, self._sub), self.journey.setup.web_hold
        )

    def _open_concierge(self) -> None:
        from fx1.flash.store import FlashStore

        if self.flash is None:
            self.flash = FlashStore(self.config / "flash" / "entries.jsonl")
        if self.sessions is None:
            self.sessions = SessionStore(self.session_dir)
        self.concierge = Concierge(
            model="fx1",
            store=self.flash,
            researcher=self.researcher,
            harness=self.harness,
            approvals=_gate(self.journey.setup.approvals),
            backend_factory=self._factory,
            input_fn=self._read_input,
            output_fn=self.out.append,
            animated=False,
            session_store=self.sessions,
        )

    def _startup(self) -> None:
        if self.concierge is None:
            return
        for line in self.concierge.startup_checks():
            self.out.append(str(line))

    def _steps(self) -> None:
        index = 0
        steps = self.journey.steps
        while index < len(steps):
            nxt = steps[index + 1] if index + 1 < len(steps) else None
            if nxt is not None and _is_cancel_pair(steps[index], nxt):
                self._say_and_cancel(steps[index], nxt)
                index += 2
                continue
            self._run_step(steps[index])
            index += 1

    def _run_step(self, step: Step) -> None:
        start = len(self.out)
        try:
            _dispatch(self, step)
        except Exception as exc:
            self.fail(f"{step.op} raised {type(exc).__name__}: {self.safe(exc)}")
        self._assert_step(step, start)

    def _say_and_cancel(self, say: Step, cancel: Step) -> None:
        start = len(self.out)
        before = len(self._require_stub().requests)
        self._arm_prompts(say)
        box: dict[str, BaseException | None] = {"exc": None}
        thread = threading.Thread(
            target=lambda: _turn(self, say, box), name="journey-turn", daemon=True
        )
        thread.start()
        self._cancel_when_inflight(before)
        _join_turn(thread, self)
        exc = box["exc"]
        if exc is not None:
            self.fail(f"say raised {type(exc).__name__}: {self.safe(exc)}")
        self._assert_step(say, start)
        self._assert_step(cancel, start)

    def _final(self) -> None:
        _assert_final(self, self.journey.final, "\n".join(self.out))

    def _guard_in_chat(self) -> None:
        if not self.journey.in_chat:
            return
        used = [step.op for step in self.journey.steps if step.op in OUT_OF_CHAT_OPS]
        if used:
            self.fail("in_chat journey used " + ",".join(used))

    def _mode(self, model: str) -> str:
        for name, mode in self.journey.setup.endpoints:
            if name == model:
                return mode
        return "missing"

    def _sub(self, text: str) -> str:
        return _substitute(text, self.table)

    def _factory(self, **kwargs: Any) -> ModelClient:
        return ModelClient(
            retry=_TINY_RETRY,
            sleep=_no_sleep,
            timeout_s=2.0,
            proxies=_no_proxies,
            bypass=_bypass_all,
            **kwargs,
        )

    def _read_input(self, prompt: str = "") -> str:
        del prompt
        if not self._input_queue:
            return ""
        return self._input_queue.pop(0)

    def _arm_prompts(self, step: Step) -> None:
        self._secret_queue = [self._sub(item) for item in step.secrets]
        self._input_queue = [self._sub(item) for item in step.inputs]

    def _handle(self, text: str) -> None:
        self.concierge.handle_line(text)
        _pump(self.concierge)

    def _assert_step(self, step: Step, start: int) -> None:
        _assert_view(self, step, "\n".join(self.out[start:]), "step")

    def _shutdown(self) -> None:
        if self.concierge is not None:
            self.concierge.shutdown(timeout_s=1.0)

    def _close_stub(self) -> None:
        if self.stub is not None:
            self.stub.close()

    def _require_stub(self) -> StubModelServer:
        if self.stub is None:
            raise RuntimeError("stub server is not running")
        return self.stub

    def _cancel_when_inflight(self, before: int) -> None:
        arrived = self._require_stub().wait_for_requests(before + 1, timeout=1.5)
        if not arrived:
            self.fail("cancel turn: model request was not in flight")
            return
        self.concierge.cancel_turn()

    def _plant_endpoints(self) -> None:
        from fx1.interactive import profiles

        stub = self._require_stub()
        for model in _MODELS:
            if self._mode(model) == "missing":
                continue
            profiles.set_endpoint(model, self.stored[model], stub.url)

    def _queue_replies(self) -> None:
        stub = self._require_stub()
        for index, spec in enumerate(self.journey.setup.replies):
            stub.queue(_reply(spec, index, _cancel_boost(self.journey.steps, index)))

    def _install_default(self) -> None:
        if not any(self._mode(model) in {"rate_limited", "down", "slow"} for model in _MODELS):
            return
        stub = self._require_stub()

        def handler(record: Any) -> Reply:
            return _mode_reply(stub, self._mode(str(record.model or "")), record)

        stub.set_default(handler)

    def _plant_flash(self) -> None:
        from fx1.flash.store import FlashStore

        self.flash = FlashStore(self.config / "flash" / "entries.jsonl")
        for item in self.journey.setup.flash:
            self.flash.add(
                text=self._sub(item.text),
                task=self._sub(item.task),
                private=item.private,
                tags=[self._sub(tag) for tag in item.tags],
            )

    def _plant_sessions(self) -> None:
        self.sessions = SessionStore(self.session_dir)
        for seed in self.journey.setup.sessions:
            messages = [
                {"role": role, "content": self._sub(content)} for role, content in seed.messages
            ]
            record = self.sessions.create(model=seed.model, messages=messages)
            _stamp_job(self.sessions, record, seed.job_state)

    def _plant_jobs(self) -> None:
        if not self.journey.setup.jobs:
            return
        ledger = JobLedger(self.config / "jobs")
        for seed in self.journey.setup.jobs:
            record = ledger.create(kind=seed.kind, goal=self._sub(seed.goal))
            if seed.state == "running":
                ledger.transition(record.job_id, JobState.RUNNING)

    def _install_taps(self) -> None:
        self._install_getpass()
        self._install_secho()
        self._install_popen()
        self._install_log()

    def _install_getpass(self) -> None:
        import getpass

        original = getpass.getpass

        def patched(prompt: str = "", stream: object = None) -> str:
            del prompt, stream
            if not self._secret_queue:
                self.fail("secret prompt was not scripted")
                return ""
            return self._secret_queue.pop(0)

        getpass.getpass = patched
        self._getpass = original

    def _install_secho(self) -> None:
        import typer

        from fx1.interactive import concierge as concierge_module

        original: Any = typer.secho

        def wrapped(message: object = "", *args: object, **kwargs: object) -> None:
            self.out.append(str(message))
            original(message, *args, **kwargs)

        typer.secho = wrapped
        concierge_module.typer.secho = wrapped
        self._secho = original

    def _install_popen(self) -> None:
        original = subprocess.Popen
        sink = self.child_envs
        secrets = self.planted

        class _Tap(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                env = kwargs.get("env")
                blob = _env_text(env if isinstance(env, Mapping) else None)
                if _hit(blob, secrets):
                    sink.append(blob)
                super().__init__(*args, **kwargs)

        subprocess.Popen = _Tap  # type: ignore[misc]
        self._popen = original

    def _install_log(self) -> None:
        root = logging.getLogger()
        self._log_level = root.level
        handler = _LogSink(self.logs)
        handler.setLevel(logging.INFO)
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
        self._log_handler = handler

    def _restore_taps(self) -> None:
        import getpass

        import typer

        from fx1.interactive import concierge as concierge_module

        if self._getpass is not None:
            getpass.getpass = self._getpass
        if self._secho is not None:
            typer.secho = self._secho
            concierge_module.typer.secho = self._secho
        if self._popen is not None:
            subprocess.Popen = self._popen  # type: ignore[misc]
        _detach_log(self._log_handler, self._log_level)

    def _kill_new_children(self) -> None:
        for pid in unexpected_children(self._before):
            _stop_pid(pid)


def _execute(journey: Journey, root: Path) -> JourneyResult:
    started = time.monotonic()
    runner = _Runner(journey, root)
    try:
        runner.go()
    except Exception as exc:
        runner.fail(f"runner raised {type(exc).__name__}: {runner.safe(exc)}")
    finally:
        _finish(runner, started)
    return runner.to_result()


def _finish(runner: _Runner, started: float) -> None:
    try:
        runner.invariants()
    except Exception as exc:
        runner.fail(f"invariants raised {type(exc).__name__}: {runner.safe(exc)}")
    runner.elapsed = time.monotonic() - started
    limit = 25.0 if runner.journey.setup.harness_mode == "real" else 3.0
    if runner.elapsed > limit:
        runner.fail(f"elapsed {runner.elapsed:.2f}s exceeds {limit:.0f}s")
    runner.close()


def _dispatch(runner: _Runner, step: Step) -> None:
    if step.op == "say":
        _say(runner, step)
        return
    if step.op == "cancel":
        _cancel_job(runner, step)
        return
    if step.op == "restart":
        _restart(runner, step)
        return
    if step.op == "wait_job":
        runner.concierge.join_job(timeout_s=_JOB_JOIN_S)
        return
    if step.op == "shell":
        _shell(runner, step)
        return
    if step.op == "edit_config":
        _edit(runner, step)
        return
    _expert(runner, step)


def _say(runner: _Runner, step: Step) -> None:
    runner._arm_prompts(step)
    runner._handle(runner._sub(step.text))


def _cancel_job(runner: _Runner, step: Step) -> None:
    if step.cancel == "turn":
        runner.fail("cancel turn must be the step after the say it interrupts")
        return
    runner._handle("/cancel")


def _restart(runner: _Runner, step: Step) -> None:
    runner._shutdown()
    runner._open_concierge()
    _maybe_resume(runner, step.resume)
    runner._startup()


def _maybe_resume(runner: _Runner, resume: bool) -> None:
    if not resume or runner.concierge is None:
        return
    runner.concierge.restore_latest()
    notice = str(getattr(runner.concierge, "resume_notice", "") or "")
    if notice:
        runner.out.append(notice)


def _shell(runner: _Runner, step: Step) -> None:
    try:
        completed = subprocess.run(
            list(step.shell),
            cwd=runner.cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        runner.fail("shell timed out")
        return
    _append_streams(runner, completed.stdout, completed.stderr)


def _edit(runner: _Runner, step: Step) -> None:
    target = (runner.config / step.edit_path).resolve()
    if not _inside(target, runner.config):
        runner.fail("edit_config path escapes the config directory")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(runner._sub(step.edit_text), encoding="utf-8")


def _expert(runner: _Runner, step: Step) -> None:
    result = runner.harness.run(step.expert, [])
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    _append_streams(runner, stdout, stderr)


def _append_streams(runner: _Runner, stdout: str, stderr: str) -> None:
    if stdout:
        runner.out.append(stdout)
    if stderr:
        runner.out.append(stderr)


def _turn(runner: _Runner, step: Step, box: dict[str, BaseException | None]) -> None:
    try:
        runner._handle(runner._sub(step.text))
    except Exception as exc:
        box["exc"] = exc


def _join_turn(thread: threading.Thread, runner: _Runner) -> None:
    thread.join(_TURN_JOIN_S)
    if not thread.is_alive() or runner.concierge is None:
        return
    runner.concierge.cancel_turn()
    thread.join(1.0)
    if thread.is_alive():
        runner.fail("cancel turn: chat turn did not stop")


def _assert_view(runner: _Runner, step: Step, window: str, where: str) -> None:
    _expect(runner, window, step.expect, step.forbid, where)
    _status(runner, step.status)
    _model(runner, step.model)
    _approvals(runner, step.approvals)
    _files(runner, step.files)
    _session(runner, step.session)
    _evidence(runner, step.evidence)
    _sent(runner, step.sent)


def _assert_final(runner: _Runner, final: FinalState, text: str) -> None:
    _expect(runner, text, final.expect, final.forbid, "final")
    _status(runner, final.status)
    _model(runner, final.model)
    _approvals(runner, final.approvals)
    _files(runner, final.files)
    _session(runner, final.session)
    _evidence(runner, final.evidence)
    _sent(runner, final.sent)
    if final.job:
        _job(runner, final.job)
    if final.restarted is not None:
        _restarted(runner, final.restarted)


def _expect(
    runner: _Runner,
    text: str,
    expect: Sequence[Expectation],
    forbid: Sequence[Expectation],
    where: str,
) -> None:
    for item in expect:
        if not _contains(text, item, runner.table):
            runner.fail(f"{where} missing {_shown(item.text, runner.planted, runner.table)}")
    for item in forbid:
        if _contains(text, item, runner.table):
            runner.fail(f"{where} forbade {_shown(item.text, runner.planted, runner.table)}")


def _status(runner: _Runner, expected: str) -> None:
    if not expected or runner.concierge is None:
        return
    found = str(runner.concierge.last_status)
    if found != expected:
        runner.fail(f"status {found!r} != {expected!r}")


def _model(runner: _Runner, expected: str) -> None:
    if not expected or runner.concierge is None:
        return
    found = str(runner.concierge.active_model)
    if found != expected:
        runner.fail(f"model {found!r} != {expected!r}")


def _approvals(runner: _Runner, expected: tuple[ApprovalExpect, ...] | None) -> None:
    if expected is None or runner.concierge is None:
        return
    rows = _snapshot(runner.concierge)
    if not expected and rows:
        runner.fail(f"expected no approvals, found {len(rows)}")
        return
    for item in expected:
        if not any(_row_matches(row, item) for row in rows):
            runner.fail(f"missing approval key={item.key or '*'} scope={item.scope or '*'}")


def _files(runner: _Runner, specs: Sequence[FileExpect]) -> None:
    for spec in specs:
        _one_file(runner, spec)


def _one_file(runner: _Runner, spec: FileExpect) -> None:
    matches = _file_matches(runner, spec)
    present = any(path.exists() for path in matches)
    if spec.exists and not present:
        runner.fail(f"missing file {spec.path}")
        return
    if not spec.exists and present:
        runner.fail(f"unexpected file {spec.path}")
        return
    if not spec.exists:
        return
    for path in matches:
        _file_body(runner, path, spec)


def _file_matches(runner: _Runner, spec: FileExpect) -> list[Path]:
    base = _anchored(runner, spec.path)
    if not spec.glob:
        return [base]
    parent = base if base.is_dir() or not base.exists() else base.parent
    return sorted(parent.glob(spec.glob))


def _file_body(runner: _Runner, path: Path, spec: FileExpect) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for needle in spec.contains:
        if runner._sub(needle) not in text:
            runner.fail(f"{path.name} missing {_shown(needle, runner.planted, runner.table)}")
    for needle in spec.forbid:
        if runner._sub(needle) in text:
            runner.fail(f"{path.name} forbade {_shown(needle, runner.planted, runner.table)}")
    _file_mode(runner, path, spec.mode)


def _file_mode(runner: _Runner, path: Path, mode: str) -> None:
    if not mode or os.name != "posix":
        return
    found = oct(path.stat().st_mode & 0o777)
    expected = oct(int(mode, 8))
    if found != expected:
        runner.fail(f"{path.name} mode {found} != {expected}")


def _session(runner: _Runner, spec: SessionExpect | None) -> None:
    if spec is None:
        return
    blob = _session_blob(runner)
    record = _latest(runner)
    _turns(runner, spec, record)
    _session_model(runner, spec.model)
    _session_text(runner, blob, spec)
    if spec.job:
        _job(runner, spec.job)


def _turns(runner: _Runner, spec: SessionExpect, record: Mapping[str, Any] | None) -> None:
    if spec.turns_at_least is None:
        return
    turns = 0 if record is None else len(list(record.get("turns") or []))
    if turns < spec.turns_at_least:
        runner.fail(f"session turns {turns} < {spec.turns_at_least}")


def _session_model(runner: _Runner, expected: str) -> None:
    if not expected or runner.concierge is None:
        return
    if str(runner.concierge.active_model) != expected:
        runner.fail(f"session model {runner.concierge.active_model!r} != {expected!r}")


def _session_text(runner: _Runner, blob: str, spec: SessionExpect) -> None:
    for text in spec.messages_contain:
        if runner._sub(text) not in blob:
            runner.fail(f"session missing {_shown(text, runner.planted, runner.table)}")
    for text in spec.messages_forbid:
        if runner._sub(text) in blob:
            runner.fail(f"session forbade {_shown(text, runner.planted, runner.table)}")


def _evidence(runner: _Runner, spec: EvidenceExpect | None) -> None:
    if spec is None:
        return
    blob = _evidence_blob(runner)
    if spec.status and spec.status not in blob:
        runner.fail(f"evidence status missing {spec.status!r}")
    for text in spec.contains:
        if runner._sub(text) not in blob:
            runner.fail(f"evidence missing {_shown(text, runner.planted, runner.table)}")


def _sent(runner: _Runner, items: Sequence[Expectation]) -> None:
    if not items:
        return
    _expect(runner, _last_request_text(runner), items, (), "sent")


def _job(runner: _Runner, state: str) -> None:
    record = _latest(runner)
    if record is None:
        runner.fail(f"job state {state!r} but no session was persisted")
        return
    found = str((record.get("job") or {}).get("state") or "")
    if found != state:
        runner.fail(f"job state {found!r} != {state!r}")


def _restarted(runner: _Runner, restarted: bool) -> None:
    running = bool(runner.concierge is not None and runner.concierge.job_running)
    if running is restarted:
        return
    runner.fail("background job running" if running else "background job was not running")


def _scan_display(chunks: Sequence[str], problems: list[str]) -> None:
    for chunk in chunks:
        _one_chunk(chunk, problems)


def _one_chunk(chunk: str, problems: list[str]) -> None:
    if "Traceback" in chunk:
        problems.append("output contains a traceback")
    if _CONTROLS.search(chunk):
        problems.append("output contains a terminal control sequence")
    try:
        validate_fx1_output(chunk)
    except Fx1HonestyError as exc:
        problems.append(
            f"displayed text failed the honesty gate ({exc.code or type(exc).__name__})"
        )


def _scan_secrets(view: InvariantView, problems: list[str]) -> None:
    blobs = [("output", "\n".join(view.chunks)), ("log", view.logs), *view.blobs]
    blobs.extend(("child environment", item) for item in view.child_envs)
    for label, text in blobs:
        if _hit(text, view.secrets):
            problems.append(f"planted secret appeared in {label}")


def _secret_blobs(runner: _Runner) -> list[tuple[str, str]]:
    blobs: list[tuple[str, str]] = []
    for path in _text_files(runner.root):
        if _is_credential(path, runner.config):
            continue
        label = str(path.relative_to(runner.root))
        blobs.append((label, path.read_text(encoding="utf-8", errors="replace")))
    return blobs


def _json_errors(config: Path, session_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in _json_targets(config, session_dir):
        _check_json(path, errors)
    return errors


def _json_targets(config: Path, session_dir: Path) -> list[Path]:
    targets: list[Path] = []
    targets.extend(_match_paths(session_dir, "*.json"))
    targets.extend(_match_paths(config / "jobs", "*.json"))
    credentials = config / "credentials.json"
    if credentials.is_file():
        targets.append(credentials)
    targets.extend(_match_paths(config / "flash", "*.jsonl"))
    return targets


def _check_json(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    pieces = text.splitlines() if path.suffix == ".jsonl" else [text]
    for index, piece in enumerate(pieces, start=1):
        if piece.strip():
            _load_piece(path, index, piece, errors)


def _load_piece(path: Path, index: int, piece: str, errors: list[str]) -> None:
    try:
        json.loads(piece)
    except json.JSONDecodeError:
        errors.append(f"invalid JSON in {path.name} line {index}")


def _text_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.stat().st_size <= 1_000_000]


def _match_paths(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob(pattern) if path.is_file()]


def _is_credential(path: Path, config: Path) -> bool:
    try:
        path.resolve().relative_to(config.resolve())
    except ValueError:
        return False
    return path.name.startswith("credentials")


def _latest(runner: _Runner) -> dict[str, Any] | None:
    if runner.sessions is None:
        return None
    loaded = runner.sessions.latest()
    if isinstance(loaded, dict):
        return loaded
    return None


def _session_blob(runner: _Runner) -> str:
    parts: list[str] = []
    if runner.concierge is not None:
        for message in runner.concierge.messages():
            parts.append(str(message.get("content") or ""))
    record = _latest(runner)
    if record is not None:
        parts.append(json.dumps(record))
    return "\n".join(parts)


def _evidence_blob(runner: _Runner) -> str:
    parts = ["\n".join(runner.out)]
    result = None if runner.concierge is None else runner.concierge.last_result
    if result is not None:
        parts.append(str(getattr(result, "status", "")))
        parts.append(str(getattr(result, "summary", "") or ""))
    return "\n".join(parts)


def _last_request_text(runner: _Runner) -> str:
    if runner.stub is None or not runner.stub.requests:
        return ""
    body = runner.stub.requests[-1].body or {}
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return ""
    return "\n".join(_message_text(item) for item in messages)


def _message_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return ""


def _snapshot(concierge: Any) -> list[dict[str, Any]]:
    gate = getattr(concierge, "_approvals", None)
    snap = getattr(gate, "snapshot", None)
    if not callable(snap):
        return []
    rows = snap()
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _row_matches(row: Mapping[str, Any], item: ApprovalExpect) -> bool:
    if item.key and str(row.get("key") or "") != item.key:
        return False
    if item.scope and str(row.get("scope") or "") != item.scope:
        return False
    if item.kind and str(row.get("kind") or "") != item.kind:
        return False
    return item.approved is None or bool(row.get("approved")) is item.approved


def _anchored(runner: _Runner, raw: str) -> Path:
    text = runner._sub(raw)
    if text.startswith("config:"):
        return runner.config / text.removeprefix("config:")
    if text.startswith("session:"):
        return runner.session_dir / text.removeprefix("session:")
    if text.startswith("cwd:"):
        return runner.cwd / text.removeprefix("cwd:")
    return runner.cwd / text


def _contains(text: str, item: Expectation, table: Mapping[str, str]) -> bool:
    needle = _substitute(item.text, table)
    if item.case_insensitive:
        return needle.casefold() in text.casefold()
    return needle in text


def _shown(text: str, secrets: Sequence[str], table: Mapping[str, str]) -> str:
    rendered = _substitute(text, table)
    if _hit(rendered, secrets):
        return "<planted-secret>"
    if len(rendered) > 80:
        return repr(rendered[:80] + "...")
    return repr(rendered)


def _substitute(text: str, table: Mapping[str, str]) -> str:
    rendered = text
    for key, value in table.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _hit(text: str, secrets: Sequence[str]) -> bool:
    return any(bool(secret) and secret in text for secret in secrets)


def _gate(scripts: tuple[ApprovalScript, ...]) -> ApprovalGate:
    if not scripts:
        return ApprovalGate()
    pending = list(scripts)

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        return _as_decision(_take_decision(pending, request.key))

    return ApprovalGate(decide_fn=decide)


def _take_decision(pending: list[ApprovalScript], key: str | None) -> str:
    wanted = key or ""
    for index, item in enumerate(pending):
        if not item.key or item.key == wanted:
            pending.pop(index)
            return item.decision
    return "deny"


def _as_decision(name: str) -> ApprovalDecision:
    if name == "once":
        return ApprovalDecision.ONCE
    if name == "session":
        return ApprovalDecision.SESSION
    return ApprovalDecision.DENY


def _reply(spec: ReplySpec, index: int, boost: float) -> Reply:
    delay = max(spec.delay_s, boost)
    reply = _bare_reply(spec, index)
    if delay > 0:
        return Reply.delay(delay, reply)
    return reply


def _bare_reply(spec: ReplySpec, index: int) -> Reply:
    if spec.kind == "violation":
        return Reply.ok(_VIOLATION_TEXT[spec.violation], usage=_USAGE)
    if spec.kind == "error":
        return _error_reply(spec)
    if spec.kind == "tool":
        return Reply.ok(spec.text or None, tool_calls=[_tool_call(spec, index)], usage=_USAGE)
    return Reply.ok(spec.text, usage=_USAGE)


def _error_reply(spec: ReplySpec) -> Reply:
    headers = {"Retry-After": "0"} if spec.status_code == 429 else None
    body = {
        "error": {
            "message": spec.error_message,
            "type": spec.error_type,
            "code": spec.error_code,
        }
    }
    return Reply.error(spec.status_code, headers=headers, body=body)


def _tool_call(spec: ReplySpec, index: int) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": spec.tool_name, "arguments": spec.tool_args_json},
    }


def _cancel_boost(steps: Sequence[Step], index: int) -> float:
    """Hold the reply that a following ``cancel: turn`` must interrupt."""

    say_index = _say_index(steps, index)
    if say_index is None or say_index + 1 >= len(steps):
        return 0.0
    nxt = steps[say_index + 1]
    if nxt.op == "cancel" and nxt.cancel == "turn":
        return _CANCEL_DELAY_FLOOR_S
    return 0.0


def _say_index(steps: Sequence[Step], reply_index: int) -> int | None:
    """Map queued reply K to the Kth plain-text say (slash lines do not call the model)."""

    seen = 0
    for index, step in enumerate(steps):
        if not _model_say(step):
            continue
        if seen == reply_index:
            return index
        seen += 1
    return None


def _model_say(step: Step) -> bool:
    return step.op == "say" and not step.text.lstrip().startswith("/")


def _mode_reply(stub: StubModelServer, mode: str, record: Any) -> Reply:
    if mode == "rate_limited":
        return _error_reply(
            ReplySpec(
                kind="error",
                status_code=429,
                error_code="rate_limit_exceeded",
                error_message="slow down",
                error_type="rate_limit_error",
            )
        )
    if mode == "down":
        return _error_reply(
            ReplySpec(
                kind="error",
                status_code=503,
                error_code="server_error",
                error_message="unavailable",
                error_type="server_error",
            )
        )
    if mode == "slow":
        return Reply.delay(_SLOW_S, stub.builtin_reply(record))
    return stub.builtin_reply(record)


def _web_docs(setup: Setup, sub: Callable[[str], str]) -> list[SourceDoc]:
    return [
        SourceDoc(url=item.url, title=item.title, status=item.status, text=sub(item.text))
        for item in setup.web
    ]


def _stamp_job(store: SessionStore, record: dict[str, Any], state: str) -> None:
    if state == "idle":
        return
    job = dict(record.get("job") or {})
    job["state"] = state
    record["job"] = job
    store.save(record)


def _families_stdout(args: Sequence[str]) -> str:
    goal = _arg_value(args, "--select").casefold()
    selected: list[dict[str, Any]] = []
    if "volatility" in goal or "qlike" in goal:
        selected.append(_family("volatility"))
    if "conformal" in goal:
        selected.append(_family("conformal"))
    exclusions: list[dict[str, str]] = []
    if not selected:
        exclusions.append({"id": "volatility", "reason": "no suitable qualified family"})
    return json.dumps({"schema": _SURFACE, "selected": selected, "exclusions": exclusions})


def _family(family_id: str) -> dict[str, Any]:
    return {"id": family_id, "executable": True, "qualification": "required"}


def _arg_value(args: Sequence[str], flag: str) -> str:
    if flag not in args:
        return ""
    index = list(args).index(flag)
    if index + 1 >= len(args):
        return ""
    return str(args[index + 1])


def _default_stdout(name: str) -> str:
    return f"DATA_LABEL=SYNTHETIC\n{name} finished\n"


def _write_receipt(cwd: Path, script: CommandScript) -> None:
    if not script.receipt_path:
        return
    target = (cwd / script.receipt_path).resolve()
    if not _inside(target, cwd):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script.receipt_body, encoding="utf-8")


def _wait_cancel(hold: bool, cancel_event: threading.Event | None) -> bool:
    if hold and cancel_event is not None:
        cancel_event.wait(2.0)
    return bool(cancel_event is not None and cancel_event.is_set())


def _is_cancel_pair(step: Step, nxt: Step | None) -> bool:
    return bool(
        nxt is not None and step.op == "say" and nxt.op == "cancel" and nxt.cancel == "turn"
    )


def _pump(concierge: Any) -> None:
    gate = getattr(concierge, "_approvals", None)
    pump = getattr(gate, "pump", None)
    if callable(pump):
        pump()


def _no_sleep(delay: float) -> None:
    del delay


def _no_proxies() -> dict[str, str]:
    return {}


def _bypass_all(host: str) -> bool:
    del host
    return True


def _slug(model: str) -> str:
    if model == "fx1-lite":
        return "lite"
    return "fx1"


def _is_secret_env(name: str) -> bool:
    if name in {"FX1_CONFIG_DIR", "FX1_SESSION_DIR", "NO_COLOR"}:
        return False
    return _SECRET_ENV.search(name) is not None


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    chosen: list[str] = []
    for item in values:
        if item and item not in seen:
            seen.add(item)
            chosen.append(item)
    return chosen


def _mapping_text(env: Mapping[Any, Any]) -> str:
    return "\n".join(f"{key}={value}" for key, value in env.items())


def _env_text(env: Mapping[Any, Any] | None) -> str:
    if env is None:
        return _mapping_text(os.environ)
    return _mapping_text(env)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _pid_name(entry: Path) -> int | None:
    if not entry.name.isdigit():
        return None
    return int(entry.name)


def _ppid(entry: Path) -> int | None:
    try:
        raw = (entry / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    mark = raw.rfind(")")
    if mark < 0:
        return None
    parts = raw[mark + 2 :].split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _stop_pid(pid: int) -> None:
    if os.name != "posix" or pid <= 1:
        return
    _signal(pid, signal.SIGTERM)
    if _alive(pid):
        _signal(pid, signal.SIGKILL)


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        return


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _detach_log(handler: logging.Handler | None, level: int) -> None:
    root = logging.getLogger()
    if handler is not None:
        root.removeHandler(handler)
    root.setLevel(level)


def _quiet(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        return


def _blank_count() -> dict[str, int]:
    return {"critical": 0, "in_chat": 0, "in_chat_passing": 0, "passed": 0, "total": 0}


def _tally(results: Sequence[JourneyResult]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in results:
        _add_result(counts.setdefault(item.category, _blank_count()), item)
    return counts


def _add_result(bucket: dict[str, int], item: JourneyResult) -> None:
    bucket["total"] += 1
    if item.passed:
        bucket["passed"] += 1
    if item.critical:
        bucket["critical"] += 1
    if item.in_chat:
        bucket["in_chat"] += 1
    if item.in_chat and item.passed:
        bucket["in_chat_passing"] += 1


def _failure_row(item: JourneyResult) -> dict[str, Any]:
    return {
        "category": item.category,
        "critical": item.critical,
        "failures": list(item.failures),
        "id": item.id,
        "in_chat": item.in_chat,
    }


def _result_row(item: JourneyResult) -> dict[str, Any]:
    return {
        "category": item.category,
        "critical": item.critical,
        "elapsed_s": round(item.elapsed_s, 3),
        "failures": list(item.failures),
        "id": item.id,
        "in_chat": item.in_chat,
        "passed": item.passed,
        "title": item.title,
    }


def _check_count(problems: list[str], total: int, *, frozen: bool) -> None:
    if frozen and total != 100:
        problems.append(f"frozen suite has {total} journeys; expected 100")
    if not frozen and total > 100:
        problems.append(f"unfrozen suite has {total} journeys; maximum is 100")


def _check_passes(problems: list[str], report: SuiteReport, *, frozen: bool) -> None:
    passing = sum(1 for item in report.results if item.in_chat and item.passed)
    if frozen and passing < 99:
        problems.append(f"in-chat passes {passing}; frozen suite needs >= 99")
    if frozen:
        return
    failed = [item.id for item in report.results if not item.passed]
    if failed:
        problems.append("unfrozen suite has failures: " + ", ".join(failed))


def _check_critical(problems: list[str], report: SuiteReport) -> None:
    failed = [item.id for item in report.results if item.critical and not item.passed]
    if failed:
        problems.append("critical journey failed: " + ", ".join(failed))


def _check_categories(problems: list[str], report: SuiteReport) -> None:
    counts = category_counts(report.results)
    short = [f"{name}={counts[name]['total']}" for name in CATEGORIES if counts[name]["total"] < 8]
    if short:
        problems.append("category floor 8: " + ", ".join(short))


def _read_freeze(path: Path) -> _Freeze | None:
    if not path.is_file():
        return None
    try:
        return _parse_freeze(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _Freeze("", -1, ())


def _parse_freeze(text: str) -> _Freeze:
    digest = ""
    count = -1
    files: list[tuple[str, str]] = []
    for raw in text.splitlines():
        digest, count = _apply_freeze_line(raw.strip(), digest, count, files)
    return _Freeze(digest, count, tuple(files))


def _apply_freeze_line(
    line: str, digest: str, count: int, files: list[tuple[str, str]]
) -> tuple[str, int]:
    if not line or line.startswith("#") or line.startswith("algorithm "):
        return digest, count
    if line.startswith("digest "):
        return line.split(" ", 1)[1].strip(), count
    if line.startswith("count "):
        return digest, int(line.split(" ", 1)[1])
    name, file_hex = line.split(" ", 1)
    files.append((name, file_hex.strip()))
    return digest, count


def _freeze_problems(
    freeze: _Freeze | None, digest: str, rows: tuple[tuple[str, str], ...]
) -> list[str]:
    if freeze is None:
        return []
    problems: list[str] = []
    if freeze.digest != digest:
        problems.append("journeys.sha256 digest does not match the yaml files")
    if freeze.count != len(rows):
        problems.append(f"journeys.sha256 count {freeze.count} != {len(rows)} files")
    if freeze.files != rows:
        problems.append("journeys.sha256 file list does not match the yaml files")
    return problems


def _write_report(report: SuiteReport, report_path: Path | None) -> None:
    target = report_path or _env_report()
    if target is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    target.write_text(payload + "\n", encoding="utf-8")


def _env_report() -> Path | None:
    raw = os.environ.get("DIPCATCHER_JOURNEY_REPORT", "").strip()
    if not raw:
        return None
    return Path(raw)


def _print_list() -> int:
    for journey in load_journeys():
        flag = "critical" if journey.critical else "normal"
        print(f"{journey.id}\t{journey.category}\t{flag}\t{journey.title}")
    return 0


def _print_freeze() -> int:
    path = write_freeze()
    count = len(yaml_paths())
    print(path)
    if count != 100:
        print(
            f"warning: froze {count} journeys; the gate expects 100 when this file exists",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
