"""One non-interactive turn through the same concierge path as chat.

The driver never prompts. Consequential actions are denied unless the caller
lists their approval keys. A missing endpoint is a status, not a fabricated
answer. SIGINT and SIGTERM cancel the turn and owned subprocesses.

``Concierge`` may expose ``last_answer``, ``last_status``, ``last_usage``,
``last_identity``, ``last_latency_s``, ``last_result``, ``messages``, and
``active_model``. When a member is absent the driver uses the honesty-gated
assistant text already stored on the turn, and leaves served-model identity
and model latency unset rather than inventing them.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import os
import re
import signal
import stat
import threading
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import FrameType
from typing import Any

EXEC_SCHEMA = "dipcatcher.exec.v1"
EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_WITHHELD = 3
EXIT_BUDGET = 4
EXIT_NO_ENDPOINT = 5
EXIT_INTERRUPTED = 130
EXIT_TERMINATED = 143

_STATUSES = frozenset({"ok", "error", "withheld", "cancelled", "timeout", "budget", "no_endpoint"})
_EXIT = {
    "ok": EXIT_OK,
    "error": EXIT_RUNTIME,
    "withheld": EXIT_WITHHELD,
    "timeout": EXIT_BUDGET,
    "budget": EXIT_BUDGET,
    "no_endpoint": EXIT_NO_ENDPOINT,
    "cancelled": EXIT_INTERRUPTED,
}
_SESSION_MODES = frozenset({"none", "new", "last"})
_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_KEYS_SET_REASON = (
    "exec refuses /keys set because it would prompt for a secret; "
    "configure the endpoint outside exec"
)
_MAX_TIMEOUT_S = 86400.0
_MIN_SOCKET_S = 0.05
_STOP_GRACE_S = 0.2

SignalHandler = Callable[[int, FrameType | None], Any]


class ExecUsageError(Exception):
    """The exec request is not runnable. Process exit code is 2."""


class ExecInterrupted(BaseException):
    """SIGINT or SIGTERM arrived. ``BaseException`` so chat's ``except Exception`` cannot swallow it."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"interrupted by signal {signum}")


@dataclass(frozen=True)
class ExecRequest:
    """One scripted turn. ``session`` is ``none``, ``new``, ``last``, or a session id."""

    prompt: str
    model: str = "fx1"
    timeout_s: float = 300.0
    approve_keys: tuple[str, ...] = ()
    session: str = "none"
    output_path: Any = None


@dataclass(frozen=True)
class ExecOutcome:
    """Result of one exec turn. ``as_json`` is the ``dipcatcher.exec.v1`` document."""

    status: str
    model: str
    served_model: str | None
    answer: str
    usage: dict[str, int] | None
    latency_s: float | None
    evidence: dict[str, Any] | None
    session_id: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    duration_s: float
    exit_code: int

    def as_json(self) -> dict[str, Any]:
        usage = None if self.usage is None else dict(self.usage)
        evidence = None if self.evidence is None else dict(self.evidence)
        return {
            "schema": EXEC_SCHEMA,
            "status": self.status,
            "model": self.model,
            "served_model": self.served_model,
            "answer": self.answer,
            "usage": usage,
            "latency_s": self.latency_s,
            "evidence": evidence,
            "session_id": self.session_id,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class _Judgement:
    status: str
    answer: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass
class _RunState:
    request: ExecRequest
    deadline: float
    monotonic: Callable[[], float]
    registry: Any
    install_signals: bool
    concierge_factory: Callable[..., Any] | None
    backend_factory: Callable[..., Any] | None
    session_store: Any
    harness: Any = None
    concierge: Any = None
    gate: Any = None
    backend: Any = None
    lines: list[str] = field(default_factory=list)
    stderr: io.StringIO = field(default_factory=io.StringIO)
    denied_before: int = 0
    messages_before: int = 0
    timed_out: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    stop_lock: threading.Lock = field(default_factory=threading.Lock)
    signum: int | None = None
    timer: threading.Timer | None = None
    previous_signals: list[tuple[int, Any]] = field(default_factory=list)
    endpoint_url: str | None = None
    resume_warning: str = ""


def exec_approvals(keys: tuple[str, ...]) -> Any:
    """Fail-closed gate. Only the listed keys are approved; nothing is prompted."""
    from fx1.interactive.approvals import ApprovalGate, ApprovalPolicy

    return ApprovalGate.non_interactive(ApprovalPolicy(allow_keys=keys))


def install_exec_signals(handler: SignalHandler) -> list[tuple[int, Any]]:
    """Install SIGINT and SIGTERM. Unsupported signals are skipped (Windows)."""
    installed: list[tuple[int, Any]] = []
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if not isinstance(sig, int):
            continue
        try:
            previous = signal.signal(sig, handler)
        except (OSError, ValueError):
            continue
        installed.append((sig, previous))
    return installed


def cleanup_owned(
    concierge: Any | None = None,
    registry: Any | None = None,
    grace_s: float = _STOP_GRACE_S,
) -> None:
    """Cancel the turn, terminate owned subprocesses, and flush the session."""
    _cancel_only(concierge)
    _terminate(registry, grace_s)
    _flush(concierge, grace_s)


def exit_code_for(status: str, signum: int | None = None) -> int:
    """Map a turn status and an optional caught signal onto a process exit code."""
    if signum is not None and signum == _sigterm_value():
        return EXIT_TERMINATED
    if signum is not None:
        return EXIT_INTERRUPTED
    return _EXIT.get(status, EXIT_RUNTIME)


def run_exec(
    request: ExecRequest,
    *,
    monotonic: Callable[[], float] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    session_store: Any = None,
    registry: Any = None,
    install_signals: bool = True,
    concierge_factory: Callable[..., Any] | None = None,
    harness: Any = None,
) -> ExecOutcome:
    """Run ``request.prompt`` once. Does not read stdin and does not prompt."""
    clock = time.monotonic if monotonic is None else monotonic
    started = clock()
    normalized = _validated(request)
    shortcut = _shortcut(normalized, started, clock)
    if shortcut is not None:
        return _attach_file(shortcut, normalized.output_path)
    state = _open_state(
        normalized,
        clock,
        started,
        backend_factory,
        session_store,
        registry,
        install_signals,
        concierge_factory,
        harness,
    )
    try:
        try:
            _execute(state)
        except ExecInterrupted as exc:
            state.signum = exc.signum
            _reap(state)
        return _finish(state, started, clock)
    finally:
        _reap(state)
        _disarm(state)


def _validated(request: ExecRequest) -> ExecRequest:
    if not request.prompt.strip():
        raise ExecUsageError("exec needs a prompt; pass text or '-' to read stdin")
    return ExecRequest(
        prompt=request.prompt,
        model=_require_model(request.model),
        timeout_s=_require_timeout(request.timeout_s),
        approve_keys=tuple(request.approve_keys),
        session=_require_session(request.session),
        output_path=request.output_path,
    )


def _shortcut(
    request: ExecRequest, started: float, clock: Callable[[], float]
) -> ExecOutcome | None:
    if _is_exit(request.prompt):
        return _bare(request.model, "ok", EXIT_OK, started, clock)
    if _is_keys_set(request.prompt):
        return _bare(
            request.model,
            "error",
            EXIT_USAGE,
            started,
            clock,
            errors=(_KEYS_SET_REASON,),
        )
    if request.timeout_s == 0:
        return _bare(request.model, "timeout", EXIT_BUDGET, started, clock, errors=("timed out",))
    return None


def _open_state(
    request: ExecRequest,
    clock: Callable[[], float],
    started: float,
    backend_factory: Callable[..., Any] | None,
    session_store: Any,
    registry: Any,
    install_signals: bool,
    concierge_factory: Callable[..., Any] | None,
    harness: Any,
) -> _RunState:
    del started
    state = _RunState(
        request=request,
        deadline=clock() + request.timeout_s,
        monotonic=clock,
        registry=_registry(registry),
        install_signals=install_signals,
        concierge_factory=concierge_factory,
        backend_factory=backend_factory,
        session_store=_store_for(request.session, session_store),
        harness=harness,
    )
    state.gate = exec_approvals(request.approve_keys)
    state.denied_before = int(state.gate.denied())
    state.concierge = _make_concierge(state)
    return state


def _execute(state: _RunState) -> None:
    if state.install_signals:
        state.previous_signals = install_exec_signals(_signal_handler(state))
    _arm_timer(state)
    try:
        _resume(state)
        _dispatch(state)
        _join_background(state)
    finally:
        _mark_done(state)
        _cancel_timer(state)


def _finish(state: _RunState, started: float, clock: Callable[[], float]) -> ExecOutcome:
    judgement = _clean(_interpret(state), state.endpoint_url)
    _flush(state.concierge, _STOP_GRACE_S)
    outcome = _outcome_from(state, judgement, started, clock)
    return _attach_file(outcome, state.request.output_path)


def _outcome_from(
    state: _RunState,
    judgement: _Judgement,
    started: float,
    clock: Callable[[], float],
) -> ExecOutcome:
    served, latency = _served_and_latency(state)
    session_id = None
    if state.request.session != "none":
        session_id = _session_id(state.concierge)
    return ExecOutcome(
        status=judgement.status,
        model=_active_model(state.concierge, state.request.model),
        served_model=served,
        answer=judgement.answer,
        usage=_usage(state),
        latency_s=latency,
        evidence=_evidence(state),
        session_id=session_id,
        warnings=judgement.warnings,
        errors=judgement.errors,
        duration_s=_elapsed(started, clock),
        exit_code=exit_code_for(judgement.status, state.signum),
    )


def _bare(
    model: str,
    status: str,
    code: int,
    started: float,
    clock: Callable[[], float],
    errors: tuple[str, ...] = (),
) -> ExecOutcome:
    return ExecOutcome(
        status=status,
        model=model,
        served_model=None,
        answer="",
        usage=None,
        latency_s=None,
        evidence=None,
        session_id=None,
        warnings=(),
        errors=errors,
        duration_s=_elapsed(started, clock),
        exit_code=code,
    )


def _attach_file(outcome: ExecOutcome, path: Any) -> ExecOutcome:
    problem = _write_last_message(path, outcome.answer)
    if problem is None:
        return outcome
    code = EXIT_RUNTIME if outcome.exit_code == EXIT_OK else outcome.exit_code
    status = "error" if outcome.exit_code == EXIT_OK else outcome.status
    return replace(outcome, status=status, errors=(*outcome.errors, problem), exit_code=code)


def _require_model(model: str) -> str:
    from fx1.interactive.profiles import check_model

    try:
        return check_model(model)
    except KeyError as exc:
        raise ExecUsageError(_scrub(str(exc))) from exc


def _require_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecUsageError("timeout-s must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > _MAX_TIMEOUT_S:
        raise ExecUsageError("timeout-s must be a non-negative number <= 86400")
    return number


def _require_session(mode: str) -> str:
    if mode in _SESSION_MODES or _SESSION_ID.fullmatch(mode):
        return mode
    raise ExecUsageError("session must be none, new, last, or a 32-character session id")


def _command_token(prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped:
        return ""
    return stripped.split(" ", 1)[0].casefold()


def _is_exit(prompt: str) -> bool:
    return _command_token(prompt) == "/exit"


def _is_keys_set(prompt: str) -> bool:
    parts = prompt.strip().split()
    if len(parts) < 2:
        return False
    return parts[0].casefold() == "/keys" and parts[1].casefold() == "set"


def _store_for(mode: str, injected: Any) -> Any:
    if mode == "none":
        return None
    if injected is not None:
        return injected
    from fx1.interactive.sessions import SessionStore

    return SessionStore()


def _registry(given: Any) -> Any:
    if given is not None:
        return given
    from fx1.interactive.procs import REGISTRY

    return REGISTRY


def _make_concierge(state: _RunState) -> Any:
    factory = _wrapping_factory(state)
    builder = state.concierge_factory
    if builder is None:
        from fx1.interactive.concierge import Concierge

        builder = Concierge
    kwargs: dict[str, Any] = {
        "approvals": state.gate,
        "backend_factory": factory,
        "input_fn": _block_input,
        "output_fn": _capture(state),
        "animated": False,
        "session_store": state.session_store,
    }
    if state.harness is not None:
        kwargs["harness"] = state.harness
    return builder(state.request.model, **kwargs)


def _wrapping_factory(state: _RunState) -> Callable[..., Any]:
    inner = state.backend_factory
    if inner is None:
        from fx1.interactive.model_client import ModelClient

        inner = ModelClient

    def factory(**kwargs: Any) -> Any:
        url = kwargs.get("api_url")
        if isinstance(url, str):
            state.endpoint_url = url
        budget = _socket_budget(state)
        built = _call_backend_factory(inner, budget, kwargs)
        state.backend = built
        return built

    return factory


def _socket_budget(state: _RunState) -> float:
    remaining = state.deadline - state.monotonic()
    if remaining < _MIN_SOCKET_S:
        return _MIN_SOCKET_S
    return remaining


def _call_backend_factory(
    factory: Callable[..., Any], timeout_s: float, kwargs: dict[str, Any]
) -> Any:
    extra: dict[str, Any] = {}
    if _accepts_keyword(factory, "timeout_s"):
        extra["timeout_s"] = timeout_s
    if _explicit_param(factory, "retry"):
        extra["retry"] = _single_attempt()
    return factory(**extra, **kwargs)


def _single_attempt() -> Any:
    from fx1.interactive.model_client import RetryPolicy

    return RetryPolicy(max_attempts=1)


def _explicit_param(factory: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    param = signature.parameters.get(name)
    if param is None:
        return False
    return param.kind is not inspect.Parameter.VAR_KEYWORD


def _accepts_keyword(factory: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name:
            return True
    return False


def _capture(state: _RunState) -> Callable[..., None]:
    def output(*args: object, **_kwargs: object) -> None:
        if not args:
            state.lines.append("")
            return
        state.lines.append(str(args[0]))

    return output


def _block_input(_prompt: str = "", *_args: object, **_kwargs: object) -> str:
    raise RuntimeError("exec mode does not read interactive prompts")


def _signal_handler(state: _RunState) -> SignalHandler:
    def handler(signum: int, _frame: FrameType | None) -> None:
        # Assignment and raise only. Locks and process kills run after unwind.
        state.signum = signum
        raise ExecInterrupted(signum)

    return handler


def _arm_timer(state: _RunState) -> None:
    delay = state.deadline - state.monotonic()
    if delay < 0:
        delay = 0.0

    def fire() -> None:
        if not _claim_stop(state):
            return
        _cancel_only(state.concierge)
        _terminate(state.registry, _STOP_GRACE_S)

    timer = threading.Timer(delay, fire)
    timer.name = "fx1-exec-deadline"
    timer.daemon = True
    timer.start()
    state.timer = timer


def _claim_stop(state: _RunState) -> bool:
    """Record a deadline only when the turn has not already finished."""
    with state.stop_lock:
        if state.done.is_set():
            return False
        state.timed_out.set()
        return True


def _mark_done(state: _RunState) -> None:
    with state.stop_lock:
        state.done.set()


def _cancel_timer(state: _RunState) -> None:
    timer = state.timer
    if timer is None:
        return
    state.timer = None
    timer.cancel()
    if timer.is_alive() and timer is not threading.current_thread():
        timer.join(timeout=1.0)


def _disarm(state: _RunState) -> None:
    _mark_done(state)
    _cancel_timer(state)
    previous = state.previous_signals
    state.previous_signals = []
    for sig, handler in reversed(previous):
        _restore_signal(sig, handler)


def _restore_signal(sig: int, handler: Any) -> None:
    try:
        signal.signal(sig, handler)
    except (OSError, ValueError):
        return


def _resume(state: _RunState) -> None:
    mode = state.request.session
    if mode in {"none", "new"}:
        return
    if mode == "last":
        _resume_last(state)
        return
    _resume_id(state, mode)


def _resume_last(state: _RunState) -> None:
    restore = getattr(state.concierge, "restore_latest", None)
    if not callable(restore):
        raise ExecUsageError("session resume is not available")
    restore(preferred_model=state.request.model)
    notice = str(getattr(state.concierge, "resume_notice", "") or "")
    if "resumed session" not in notice:
        raise ExecUsageError(notice or "no saved session to resume")
    state.resume_warning = notice


def _resume_id(state: _RunState, session_id: str) -> None:
    restore = getattr(state.concierge, "restore", None)
    if callable(restore):
        restore(session_id)
        return
    cmd = getattr(state.concierge, "_cmd_resume", None)
    if not callable(cmd):
        raise ExecUsageError("session resume is not available")
    before = len(state.lines)
    try:
        cmd(session_id)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise ExecUsageError(_scrub(str(exc))) from exc
    _take_resume_notice(state, before)


def _take_resume_notice(state: _RunState, before: int) -> None:
    noticed = [line.strip() for line in state.lines[before:] if line.strip()]
    del state.lines[before:]
    if noticed:
        state.resume_warning = noticed[-1]


def _dispatch(state: _RunState) -> None:
    state.messages_before = len(_message_list(state.concierge))
    with contextlib.redirect_stderr(state.stderr):
        state.concierge.handle_line(state.request.prompt)


def _join_background(state: _RunState) -> None:
    if not _job_running(state.concierge):
        return
    remaining = state.deadline - state.monotonic()
    if remaining <= 0:
        state.timed_out.set()
        _cancel_only(state.concierge)
        _terminate(state.registry, _STOP_GRACE_S)
        return
    join = getattr(state.concierge, "join_job", None)
    if callable(join):
        join(remaining)
    if _job_running(state.concierge):
        state.timed_out.set()
        _cancel_only(state.concierge)
        _terminate(state.registry, _STOP_GRACE_S)


def _job_running(concierge: Any) -> bool:
    job = getattr(concierge, "_job", None)
    if job is None:
        return False
    return bool(getattr(job, "running", False))


def _interpret(state: _RunState) -> _Judgement:
    if state.signum is not None:
        detail = f"interrupted by signal {state.signum}"
        return _Judgement("cancelled", "", (), (detail,))
    if state.timed_out.is_set() or _stderr_is_timeout(state):
        return _timeout_judgement(state)
    public = _public_judgement(state)
    base = _text_judgement(state) if public is None else public
    base = _normalize_special(base, state)
    base = _with_denials(state, base)
    return _with_resume(base, state.resume_warning)


def _timeout_judgement(state: _RunState) -> _Judgement:
    extra = _stderr_errors(state.stderr.getvalue())
    return _Judgement("timeout", "", (), ("timed out", *extra))


def _public_judgement(state: _RunState) -> _Judgement | None:
    status = getattr(state.concierge, "last_status", None)
    if not isinstance(status, str) or status not in _STATUSES:
        return None
    answer = _public_answer(state, status)
    if answer is None:
        return None
    errors = _stderr_errors(state.stderr.getvalue()) if status == "error" else ()
    return _Judgement(status, answer, _progress(state.lines, answer), errors)


def _public_answer(state: _RunState, status: str) -> str | None:
    raw = getattr(state.concierge, "last_answer", None)
    answer = raw if isinstance(raw, str) else ""
    if status == "ok" and answer == "" and _has_visible_text(state):
        return None
    if status == "withheld":
        return _withheld_answer(state, answer)
    if status == "no_endpoint":
        return answer or _no_endpoint_line(state.lines) or ""
    return answer


def _withheld_answer(state: _RunState, answer: str) -> str:
    if answer:
        return answer
    return _withheld_line(state.lines) or "answer withheld by the fx-1 honesty gate"


def _stderr_is_timeout(state: _RunState) -> bool:
    return _is_timeout_text(_plain(state.stderr.getvalue()))


def _has_visible_text(state: _RunState) -> bool:
    if any(line.strip() for line in state.lines):
        return True
    return _assistant_answer(state) is not None


def _text_judgement(state: _RunState) -> _Judgement:
    kind = _text_kind(state)
    if kind == "no_endpoint":
        line = _no_endpoint_line(state.lines) or "no model endpoint configured"
        return _Judgement("no_endpoint", "", (), (line,))
    if kind == "withheld":
        line = _withheld_line(state.lines) or "answer withheld by the fx-1 honesty gate"
        return _Judgement("withheld", line, (), ())
    if kind == "budget":
        return _Judgement("budget", "", (), ("budget exceeded",))
    if kind == "timeout":
        return _timeout_judgement(state)
    if kind == "error":
        return _Judgement("error", "", (), _stderr_errors(state.stderr.getvalue()))
    return _ok_judgement(state)


def _text_kind(state: _RunState) -> str | None:
    err = _plain(state.stderr.getvalue())
    if _no_endpoint_line(state.lines) is not None:
        return "no_endpoint"
    if _withheld_line(state.lines) is not None:
        return "withheld"
    if _contains_budget(state.lines, err):
        return "budget"
    if _is_timeout_text(err):
        return "timeout"
    if _is_error_text(err):
        return "error"
    return None


def _ok_judgement(state: _RunState) -> _Judgement:
    summary = _job_summary(state.concierge)
    if summary:
        return _Judgement("ok", summary, _progress(state.lines, summary), ())
    assistant = _assistant_answer(state)
    if assistant is not None:
        return _Judgement("ok", assistant, _progress(state.lines, assistant), ())
    blob = "\n".join(line for line in state.lines if line).strip()
    if blob:
        return _Judgement("ok", blob, (), ())
    err = _stderr_errors(state.stderr.getvalue())
    if err:
        return _Judgement("error", "", (), err)
    return _Judgement("ok", "", (), ())


def _normalize_special(judgement: _Judgement, state: _RunState) -> _Judgement:
    if judgement.status == "no_endpoint":
        return _as_no_endpoint(judgement, state)
    if judgement.status == "error" and judgement.answer:
        errors = (judgement.answer, *judgement.errors)
        return _Judgement("error", "", judgement.warnings, errors)
    return judgement


def _as_no_endpoint(judgement: _Judgement, state: _RunState) -> _Judgement:
    errors = judgement.errors
    if judgement.answer:
        errors = (judgement.answer, *errors)
    if not errors:
        line = _no_endpoint_line(state.lines) or "no model endpoint configured"
        errors = (line,)
    return _Judgement("no_endpoint", "", judgement.warnings, errors)


def _with_denials(state: _RunState, judgement: _Judgement) -> _Judgement:
    denials = _denial_lines(state)
    if not denials:
        return judgement
    errors = judgement.errors + denials
    if judgement.status != "ok":
        return _Judgement(judgement.status, judgement.answer, judgement.warnings, errors)
    return _Judgement("error", "", judgement.warnings, errors)


def _with_resume(judgement: _Judgement, notice: str) -> _Judgement:
    if not notice:
        return judgement
    return _Judgement(
        judgement.status, judgement.answer, (notice, *judgement.warnings), judgement.errors
    )


def _denial_lines(state: _RunState) -> tuple[str, ...]:
    gate = state.gate
    if gate is None:
        return ()
    denied = [row for row in gate.snapshot() if not row.get("approved")]
    fresh = denied[state.denied_before :]
    lines: list[str] = []
    for row in fresh:
        question = str(row.get("question") or "consequential action")
        key = row.get("key")
        label = key if isinstance(key, str) and key else str(row.get("kind") or "action")
        lines.append(f"denied {label}: {question}")
    return tuple(lines)


def _no_endpoint_line(lines: Sequence[str]) -> str | None:
    for line in lines:
        if "no model endpoint configured" in line.casefold():
            return line.strip()
    return None


def _withheld_line(lines: Sequence[str]) -> str | None:
    for line in lines:
        if "answer withheld by the fx-1 honesty gate" in line.casefold():
            return line.strip()
    return None


def _contains_budget(lines: Sequence[str], err: str) -> bool:
    blob = "\n".join(lines) + "\n" + err
    return "budget exceeded" in blob.casefold()


def _is_timeout_text(text: str) -> bool:
    lowered = text.casefold()
    return "timeouterror" in lowered or "timed out" in lowered


def _is_error_text(text: str) -> bool:
    return any(line.strip().casefold().startswith("error:") for line in text.splitlines())


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _stderr_errors(text: str) -> tuple[str, ...]:
    cleaned = _plain(text)
    return tuple(line.strip() for line in cleaned.splitlines() if line.strip())


def _progress(lines: Sequence[str], answer: str) -> tuple[str, ...]:
    kept: list[str] = []
    target = answer.strip()
    for line in lines:
        text = line.strip()
        if not text or text == target:
            continue
        kept.append(text)
    return tuple(kept)


def _message_list(concierge: Any) -> list[Any]:
    reader = getattr(concierge, "messages", None)
    if callable(reader):
        got = _safe_messages(reader)
        if got is not None:
            return got
    raw = getattr(concierge, "_messages", None)
    if isinstance(raw, list):
        return list(raw)
    return []


def _safe_messages(reader: Callable[[], Any]) -> list[Any] | None:
    try:
        got = reader()
    except Exception:
        return None
    if isinstance(got, list):
        return list(got)
    return None


def _assistant_answer(state: _RunState) -> str | None:
    fresh = _message_list(state.concierge)[state.messages_before :]
    for message in reversed(fresh):
        content = _assistant_content(message)
        if content is not None:
            return content
    return None


def _assistant_content(message: Any) -> str | None:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def _job_summary(concierge: Any) -> str | None:
    result = _result_of(concierge)
    summary = getattr(result, "summary", None) if result is not None else None
    if isinstance(summary, str) and summary.strip():
        return summary
    return None


def _result_of(concierge: Any) -> Any:
    result = getattr(concierge, "last_result", None)
    if result is not None:
        return result
    job = getattr(concierge, "_job", None)
    if job is None:
        return None
    return getattr(job, "result", None)


def _served_and_latency(state: _RunState) -> tuple[str | None, float | None]:
    identity = getattr(state.concierge, "last_identity", None)
    if identity is None:
        identity = getattr(state.backend, "last_identity", None)
    served = getattr(identity, "served", None) if identity is not None else None
    served_text = served if isinstance(served, str) and served else None
    latency = getattr(state.concierge, "last_latency_s", None)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return served_text, None
    if not math.isfinite(float(latency)):
        return served_text, None
    return served_text, float(latency)


def _usage(state: _RunState) -> dict[str, int] | None:
    for owner in (state.concierge, state.backend):
        parsed = _usage_dict(getattr(owner, "last_usage", None))
        if parsed is not None:
            return parsed
    return None


def _usage_dict(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    cleaned: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
            cleaned[key] = value
    if not cleaned:
        return None
    return cleaned


def _evidence(state: _RunState) -> dict[str, Any] | None:
    result = _result_of(state.concierge)
    if result is None:
        return None
    summary = getattr(result, "summary", None)
    artifacts = _artifacts(result)
    if not isinstance(summary, str) and not artifacts:
        return None
    text = summary if isinstance(summary, str) else ""
    return {"summary": _scrub(_hide_url(text, state.endpoint_url)), "artifacts": artifacts}


def _artifacts(result: Any) -> list[str]:
    runs = getattr(result, "runs", None)
    if not isinstance(runs, list):
        return []
    found: list[str] = []
    for run in runs:
        found.extend(_run_artifacts(run))
    return found[:32]


def _run_artifacts(run: Any) -> list[str]:
    items = getattr(run, "artifacts", None)
    if not isinstance(items, list):
        return []
    return [_scrub(item) for item in items if isinstance(item, str) and item]


def _session_id(concierge: Any) -> str | None:
    for name in ("session_id", "last_session_id"):
        value = getattr(concierge, name, None)
        if isinstance(value, str) and value:
            return value
    record = getattr(concierge, "_session", None)
    if not isinstance(record, dict):
        return None
    ident = record.get("id")
    if isinstance(ident, str) and ident:
        return ident
    return None


def _active_model(concierge: Any, fallback: str) -> str:
    if concierge is None:
        return fallback
    active = getattr(concierge, "active_model", None)
    if isinstance(active, str) and active:
        return active
    model = getattr(concierge, "model", None)
    if isinstance(model, str) and model:
        return model
    return fallback


def _clean(judgement: _Judgement, url: str | None) -> _Judgement:
    return _Judgement(
        judgement.status,
        _scrub(_hide_url(judgement.answer, url)),
        tuple(_scrub(_hide_url(item, url)) for item in judgement.warnings),
        tuple(_scrub(_hide_url(item, url)) for item in judgement.errors),
    )


def _hide_url(text: str, url: str | None) -> str:
    if not url or url not in text:
        return text
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return text.replace(url, host)


def _scrub(text: str) -> str:
    from fx1.interactive.redact import redact_text

    return redact_text(text)


def _write_last_message(path: Any, answer: str) -> str | None:
    if path is None:
        return None
    from fx1.interactive.atomicstore import StoreError, atomic_write_text

    try:
        target = Path(path)
    except (TypeError, ValueError):
        return "could not write the last message (invalid path)"
    problem = _output_problem(target)
    if problem is not None:
        return problem
    try:
        atomic_write_text(target, answer, mode=0o600)
    except (OSError, StoreError, ValueError) as exc:
        reason = getattr(exc, "strerror", None) or "could not encode text"
        return _scrub(f"could not write the last message ({reason})")
    return None


def _output_problem(path: Path) -> str | None:
    if _unsafe_output_text(os.fspath(path)):
        return "could not write the last message (unsafe path)"
    try:
        return _special_output(path)
    except (OSError, ValueError):
        return "could not write the last message (unsafe path)"


def _special_output(path: Path) -> str | None:
    if path.is_symlink():
        return "could not write the last message (unsafe path)"
    if not os.path.lexists(path):
        return None
    if stat.S_ISREG(path.lstat().st_mode):
        return None
    return "could not write the last message (unsafe path)"


def _unsafe_output_text(text: str) -> bool:
    if "\x00" in text or "\n" in text or "\r" in text:
        return True
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    return len(text) >= 2 and text[0].isalpha() and text[1] == ":"


def _elapsed(started: float, clock: Callable[[], float]) -> float:
    value = clock() - started
    if value < 0:
        return 0.0
    return round(float(value), 6)


def _cancel_only(concierge: Any) -> None:
    if concierge is None:
        return
    for name in ("cancel_turn", "cancel_job"):
        method = getattr(concierge, name, None)
        if callable(method):
            _call_quiet(method)


def _reap(state: _RunState) -> None:
    if state.signum is None and not state.timed_out.is_set():
        return
    _cancel_only(state.concierge)
    _terminate(state.registry, _STOP_GRACE_S)


def _flush(concierge: Any, grace_s: float) -> None:
    if concierge is None:
        return
    shutdown = getattr(concierge, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown(timeout_s=grace_s)
    except Exception:
        return


def _terminate(registry: Any, grace_s: float) -> None:
    target = _registry(registry)
    terminate = getattr(target, "terminate_all", None)
    if not callable(terminate):
        return
    try:
        terminate(grace_s)
    except Exception:
        return


def _call_quiet(fn: Callable[..., Any]) -> None:
    try:
        fn()
    except Exception:
        return


def _sigterm_value() -> int:
    sig = getattr(signal, "SIGTERM", None)
    if isinstance(sig, int):
        return sig
    return 15
