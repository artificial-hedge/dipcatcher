"""Approval gates for consequential actions and private-data sharing.

The front door keeps the lab's fail-closed posture: approvals default to
*deny* whenever no interactive operator can answer (piped stdin, tests,
scripts). An interactive session asks on the console. Two reasons are
first-class:

- ``consequential`` — running a harness command that trains/optimizes or
  otherwise mutates lab state beyond cheap diagnostics;
- ``private_data`` — moving private flash context (or other machine-local
  material) into a model prompt or a web-search query.

A session grant covers one ``(kind, key)`` pair and the bounds recorded with
it. A later ask reuses that grant only when those bounds are unchanged.
Non-interactive callers pass an :class:`ApprovalPolicy` via
:meth:`ApprovalGate.non_interactive` and are never blocked on input. The gate
records every decision (who, and why) so a session can explain what was
authorized. Worker threads never read stdin; the main thread drains queued
prompts with :meth:`ApprovalGate.pump`. A worker that is not answered in time
is denied and recorded.
"""

from __future__ import annotations

import re
import threading
import weakref
from collections import deque
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fx1.interactive.redact import redact_text

AskFn = Callable[[str], bool]  # question -> approved


class ApprovalDecision(StrEnum):
    """Operator or policy choice for one approval request."""

    DENY = "deny"
    ONCE = "once"
    SESSION = "session"


@dataclass(frozen=True)
class ApprovalRequest:
    """Stable identity of one approval prompt.

    ``key`` is an action id such as ``harness:research``. An empty key cannot
    be reused as a session grant.
    """

    key: str
    kind: str
    question: str


DecideFn = Callable[[ApprovalRequest], ApprovalDecision]

_STDIN_PROMPTERS: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()
_PROMPT_CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_PROMPT_FORMAT = re.compile("[\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


@dataclass(frozen=True)
class ActionBounds:
    """Recorded limits of one approval. A session grant covers only equal bounds."""

    argv: tuple[str, ...] = ()
    runtime_s: int | None = None
    writes: bool = False
    shares_data: bool = False

    def __post_init__(self) -> None:
        _validate_bounds(self.argv, self.runtime_s, self.writes, self.shares_data)


class ApprovalPolicy:
    """Explicit non-interactive allow-list. Everything else is denied.

    ``allow_kinds`` may approve a whole kind (for example ``read``). The kind
    ``consequential`` is never approved by kind alone; a consequential action
    needs its own key in ``allow_keys``. A bare string is not a key list.
    ``max_runtime_s``, ``allow_writes``, and ``allow_data_sharing`` are caps:
    a request that does not prove it is inside them is denied.
    """

    def __init__(
        self,
        *,
        allow_keys: Collection[str] = (),
        allow_kinds: Collection[str] = (),
        max_runtime_s: int | None = None,
        allow_writes: bool = True,
        allow_data_sharing: bool = True,
    ) -> None:
        self._allow_keys = _string_set(allow_keys, what="allow_keys")
        kinds = _string_set(allow_kinds, what="allow_kinds")
        self._allow_kinds = frozenset(kind for kind in kinds if kind.casefold() != "consequential")
        self._max_runtime_s = _runtime_cap(max_runtime_s)
        self._allow_writes = _bool_flag(allow_writes, "allow_writes")
        self._allow_data_sharing = _bool_flag(allow_data_sharing, "allow_data_sharing")

    def decide(
        self,
        request: ApprovalRequest,
        bounds: ActionBounds | None = None,
    ) -> ApprovalDecision:
        """``ONCE`` when the key (or a non-consequential kind) is listed, else ``DENY``."""
        if not _within_caps(self, bounds):
            return ApprovalDecision.DENY
        if request.key and request.key in self._allow_keys:
            return ApprovalDecision.ONCE
        if request.kind.casefold() == "consequential":
            return ApprovalDecision.DENY
        if request.kind in self._allow_kinds:
            return ApprovalDecision.ONCE
        return ApprovalDecision.DENY


@dataclass
class ApprovalRecord:
    kind: str
    question: str
    approved: bool
    timestamp: str
    key: str | None = None
    scope: str = "once"
    actor: str = "gate"

    def __repr__(self) -> str:
        question = _visible_question(self.question)
        key = _visible_question(self.key) if self.key else None
        return (
            "ApprovalRecord("
            f"kind={self.kind!r}, question={question!r}, approved={self.approved!r}, "
            f"timestamp={self.timestamp!r}, key={key!r}, scope={self.scope!r}, "
            f"actor={self.actor!r})"
        )


@dataclass
class _Waiter:
    question: str
    kind: str
    key: str | None
    bounds: ActionBounds | None
    event: threading.Event = field(default_factory=threading.Event)
    settled: bool = False
    expired: bool = False
    approved: bool = False


@dataclass
class ApprovalGate:
    """Fail-closed approval: ``ask`` returns False unless a prompter or policy approves.

    Existing callers that pass only ``ask_fn`` keep yes/no behavior. Pass
    ``key`` to reuse a session grant for the same :class:`ActionBounds`.
    ``interactive=False`` never calls a prompter. ``ApprovalGate()`` with
    neither a prompter nor a policy denies and records scope ``no-prompter``.
    A console prompter invoked off the main thread is queued for :meth:`pump`
    and never reads stdin itself.
    """

    ask_fn: AskFn | None = None
    records: list[ApprovalRecord] = field(default_factory=list, repr=False)
    decide_fn: DecideFn | None = None
    policy: ApprovalPolicy | None = None
    interactive: bool = True
    clock: Callable[[], datetime] | None = None
    prompt_timeout_s: float = 30.0
    _session: dict[tuple[str, str], ActionBounds | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _pending: deque[_Waiter] = field(default_factory=deque, init=False, repr=False, compare=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    @classmethod
    def non_interactive(cls, policy: ApprovalPolicy) -> ApprovalGate:
        """A gate that applies *policy* and never reads stdin."""
        return cls(policy=policy, interactive=False)

    def ask(
        self,
        question: str,
        *,
        kind: str = "action",
        key: str | None = None,
        bounds: ActionBounds | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Approve *question*. Equal ``(kind, key, bounds)`` reuses a session grant."""
        if bounds is not None and not isinstance(bounds, ActionBounds):
            raise ValueError("bounds must be ActionBounds")
        outcome = self._begin(question, kind=kind, key=key, bounds=bounds)
        if isinstance(outcome, _Waiter):
            return self._wait_waiter(outcome, timeout_s)
        return outcome

    def pump(self, *, limit: int = 1) -> int:
        """Resolve up to *limit* queued worker prompts on the main thread."""
        if threading.current_thread() is not threading.main_thread() or limit < 1:
            return 0
        done = 0
        while done < limit:
            waiter = self._take_pending()
            if waiter is None:
                break
            self._finish_waiter(waiter)
            done += 1
        return done

    def cancel_all(self) -> int:
        """Deny every queued request. Returns how many were still waiting."""
        with self._lock:
            waiting = [item for item in self._pending if not item.settled]
            self._pending.clear()
        return _settle_cancelled(self, waiting)

    def pending_count(self) -> int:
        """How many worker requests are still waiting for :meth:`pump`."""
        with self._lock:
            return sum(1 for item in self._pending if not item.settled)

    def revoke(self, key: str | None = None) -> None:
        """Drop session grants. ``key=None`` clears every grant."""
        with self._lock:
            if key is None:
                self._session.clear()
                return
            self._session = {pair: grant for pair, grant in self._session.items() if pair[1] != key}

    def snapshot(self) -> list[dict[str, Any]]:
        """Redacted decision log for session persistence.

        ``reusable`` is true while that record's ``(kind, key)`` is still a
        live session grant. Questions and keys are passed through secret redaction.
        """
        with self._lock:
            return [self._public_record(record) for record in self.records]

    def denied(self) -> int:
        with self._lock:
            return sum(1 for record in self.records if not record.approved)

    def granted(self) -> int:
        with self._lock:
            return sum(1 for record in self.records if record.approved)

    def _resolve(
        self,
        request: ApprovalRequest,
        bounds: ActionBounds | None,
    ) -> tuple[ApprovalDecision, str, str]:
        if not self.interactive:
            return _from_policy(self.policy, request, bounds)
        if self.decide_fn is not None:
            return _from_decide(self.decide_fn, request)
        if self.ask_fn is not None:
            return _from_ask(self.ask_fn, request.question)
        if self.policy is not None:
            return _from_policy(self.policy, request, bounds)
        return ApprovalDecision.DENY, "no-prompter", "none"

    def _append(
        self,
        kind: str,
        question: str,
        approved: bool,
        key: str | None,
        scope: str,
        actor: str,
    ) -> None:
        self.records.append(
            ApprovalRecord(
                kind=kind,
                question=question,
                approved=approved,
                timestamp=self._now(),
                key=key,
                scope=scope,
                actor=actor,
            )
        )

    def _now(self) -> str:
        if self.clock is not None:
            value = self.clock()
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).isoformat(timespec="seconds")
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _public_record(self, record: ApprovalRecord) -> dict[str, Any]:
        reusable = (
            record.key is not None
            and record.approved
            and (record.kind, record.key) in self._session
        )
        return {
            "kind": record.kind,
            "key": _visible_question(record.key) if record.key else None,
            "question": _visible_question(record.question),
            "approved": record.approved,
            "scope": record.scope,
            "actor": record.actor,
            "timestamp": record.timestamp,
            "reusable": reusable,
        }

    def _begin(
        self,
        question: str,
        *,
        kind: str,
        key: str | None,
        bounds: ActionBounds | None,
    ) -> bool | _Waiter:
        with self._lock:
            reused = self._reuse_locked(question, kind, key, bounds)
            if reused is not None:
                return reused
            if self._defer_to_main():
                return self._enqueue_locked(question, kind, key, bounds)
            return self._decide_locked(question, kind, key, bounds)

    def _reuse_locked(
        self,
        question: str,
        kind: str,
        key: str | None,
        bounds: ActionBounds | None,
    ) -> bool | None:
        stable = key or None
        if stable is None or (kind, stable) not in self._session:
            return None
        if self._session[(kind, stable)] != bounds:
            return None
        self._append(kind, question, True, stable, "session-reuse", "session")
        return True

    def _decide_locked(
        self,
        question: str,
        kind: str,
        key: str | None,
        bounds: ActionBounds | None,
    ) -> bool:
        stable = key or None
        decision, scope, actor = self._resolve(
            ApprovalRequest(stable or "", kind, question), bounds
        )
        decision, scope = _bind_session(decision, scope, stable)
        if decision is ApprovalDecision.SESSION and stable is not None:
            self._session[(kind, stable)] = bounds
        approved = decision is not ApprovalDecision.DENY
        self._append(kind, question, approved, stable, scope, actor)
        return approved

    def _defer_to_main(self) -> bool:
        if threading.current_thread() is threading.main_thread():
            return False
        return _reads_stdin(self._active_prompter())

    def _active_prompter(self) -> Callable[..., Any] | None:
        if not self.interactive:
            return None
        if self.decide_fn is not None:
            return self.decide_fn
        return self.ask_fn

    def _enqueue_locked(
        self,
        question: str,
        kind: str,
        key: str | None,
        bounds: ActionBounds | None,
    ) -> _Waiter:
        waiter = _Waiter(question=question, kind=kind, key=key or None, bounds=bounds)
        self._pending.append(waiter)
        return waiter

    def _wait_waiter(self, waiter: _Waiter, timeout_s: float | None) -> bool:
        limit = self.prompt_timeout_s if timeout_s is None else timeout_s
        if isinstance(limit, bool) or limit < 0:
            raise ValueError("timeout_s must be a non-negative number")
        if waiter.event.wait(limit):
            return waiter.approved
        self._settle(waiter, approved=False, scope="timeout", actor="timeout", force=True)
        return bool(waiter.approved)

    def _take_pending(self) -> _Waiter | None:
        with self._lock:
            while self._pending:
                waiter = self._pending.popleft()
                if not waiter.settled:
                    return waiter
            return None

    def _finish_waiter(self, waiter: _Waiter) -> None:
        prompted = self._prompt_waiter(waiter)
        if prompted is None:
            return
        approved, scope, actor = prompted
        self._settle(waiter, approved=approved, scope=scope, actor=actor)

    def _prompt_waiter(self, waiter: _Waiter) -> tuple[bool, str, str] | None:
        with self._lock:
            if waiter.settled or waiter.expired:
                return None
        decision, scope, actor = self._resolve(
            ApprovalRequest(waiter.key or "", waiter.kind, waiter.question),
            waiter.bounds,
        )
        decision, scope = _bind_session(decision, scope, waiter.key)
        return (decision is not ApprovalDecision.DENY, scope, actor)

    def _settle(
        self,
        waiter: _Waiter,
        *,
        approved: bool,
        scope: str,
        actor: str,
        force: bool = False,
    ) -> bool:
        with self._lock:
            if not _can_settle(waiter, force=force):
                return False
            _mark_settled(waiter, approved=approved, scope=scope)
            self._drop_pending(waiter)
            if approved and scope == "session" and waiter.key is not None:
                self._session[(waiter.kind, waiter.key)] = waiter.bounds
            self._append(waiter.kind, waiter.question, approved, waiter.key, scope, actor)
            waiter.event.set()
            return True

    def _drop_pending(self, waiter: _Waiter) -> None:
        self._pending = deque(item for item in self._pending if item is not waiter)


def _bind_session(
    decision: ApprovalDecision,
    scope: str,
    key: str | None,
) -> tuple[ApprovalDecision, str]:
    """A SESSION choice with no key cannot be reused; record it as once."""
    if decision is ApprovalDecision.SESSION and key is None:
        return ApprovalDecision.ONCE, "once"
    if decision is ApprovalDecision.SESSION:
        return decision, "session"
    return decision, scope


def _from_decide(
    decide_fn: DecideFn,
    request: ApprovalRequest,
) -> tuple[ApprovalDecision, str, str]:
    try:
        decision = decide_fn(request)
    except (Exception, KeyboardInterrupt):
        return ApprovalDecision.DENY, "denied", "operator"
    if decision is ApprovalDecision.ONCE:
        return ApprovalDecision.ONCE, "once", "operator"
    if decision is ApprovalDecision.SESSION:
        return ApprovalDecision.SESSION, "session", "operator"
    return ApprovalDecision.DENY, "denied", "operator"


def _from_ask(ask_fn: AskFn, question: str) -> tuple[ApprovalDecision, str, str]:
    try:
        approved = bool(ask_fn(question))
    except (Exception, KeyboardInterrupt):
        return ApprovalDecision.DENY, "denied", "operator"
    if approved:
        return ApprovalDecision.ONCE, "once", "operator"
    return ApprovalDecision.DENY, "denied", "operator"


def _from_policy(
    policy: ApprovalPolicy | None,
    request: ApprovalRequest,
    bounds: ActionBounds | None,
) -> tuple[ApprovalDecision, str, str]:
    if policy is None:
        return ApprovalDecision.DENY, "no-prompter", "none"
    decision = policy.decide(request, bounds)
    if decision is ApprovalDecision.ONCE:
        return ApprovalDecision.ONCE, "policy", "policy"
    return ApprovalDecision.DENY, "denied", "policy"


def console_ask_fn(input_fn: Callable[[str], str] = input) -> AskFn:
    """Interactive y/N prompter; anything but an explicit yes is a deny."""

    def ask(question: str) -> bool:
        if threading.current_thread() is not threading.main_thread():
            return False
        shown = _visible_question(question)
        try:
            answer = input_fn(f"approval required — {shown} [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

    _STDIN_PROMPTERS.add(ask)
    return ask


def console_decide_fn(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> DecideFn:
    """``[y]es`` once, ``[a]lways`` this session, ``[N]o``. Anything else denies.

    EOF and ``KeyboardInterrupt`` deny. When *output_fn* is the builtin
    ``print``, the prompt is handed to *input_fn* (same shape as
    :func:`console_ask_fn`). Any other *output_fn* receives the prompt and
    *input_fn* is called with an empty string.
    """

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        if threading.current_thread() is not threading.main_thread():
            return ApprovalDecision.DENY
        shown = _visible_question(request.question)
        prompt = f"approval required — {shown} [y]es once / [a]lways this session / [N]o: "
        try:
            answer = _read_decision(prompt, input_fn, output_fn)
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY
        return _parse_decision(answer)

    _STDIN_PROMPTERS.add(decide)
    return decide


def _read_decision(
    prompt: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    if output_fn is print:
        return input_fn(prompt).strip().lower()
    output_fn(prompt)
    return input_fn("").strip().lower()


def _parse_decision(answer: str) -> ApprovalDecision:
    if answer in {"y", "yes"}:
        return ApprovalDecision.ONCE
    if answer in {"a", "always"}:
        return ApprovalDecision.SESSION
    return ApprovalDecision.DENY


def _visible_question(question: str) -> str:
    cleaned = _PROMPT_FORMAT.sub("", _PROMPT_CONTROLS.sub("", question))
    cleaned = " ".join(redact_text(cleaned).split())
    if len(cleaned) > 180:
        return cleaned[:180]
    return cleaned


def _reads_stdin(fn: Callable[..., Any] | None) -> bool:
    return fn is not None and fn in _STDIN_PROMPTERS


def _can_settle(waiter: _Waiter, *, force: bool) -> bool:
    if waiter.settled:
        return False
    return not (waiter.expired and not force)


def _mark_settled(waiter: _Waiter, *, approved: bool, scope: str) -> None:
    if scope == "timeout":
        waiter.expired = True
    waiter.settled = True
    waiter.approved = approved


def _settle_cancelled(gate: ApprovalGate, waiting: list[_Waiter]) -> int:
    count = 0
    for waiter in waiting:
        settled = gate._settle(
            waiter,
            approved=False,
            scope="cancelled",
            actor="cancel",
            force=True,
        )
        if settled:
            count += 1
    return count


def _string_set(values: Collection[str], *, what: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise ValueError(f"{what} must be a collection of strings")
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{what} must be non-empty strings")
        cleaned.append(item)
    return frozenset(cleaned)


def _runtime_cap(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_runtime_s must be a non-negative int")
    return value


def _bool_flag(value: bool, what: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be a bool")
    return value


def _caps_restricted(policy: ApprovalPolicy) -> bool:
    if policy._max_runtime_s is not None:
        return True
    return not policy._allow_writes or not policy._allow_data_sharing


def _runtime_inside(cap: int | None, runtime: int | None) -> bool:
    if cap is None:
        return True
    return not (runtime is None or runtime > cap)


def _bounds_inside(policy: ApprovalPolicy, bounds: ActionBounds) -> bool:
    if bounds.writes and not policy._allow_writes:
        return False
    if bounds.shares_data and not policy._allow_data_sharing:
        return False
    return _runtime_inside(policy._max_runtime_s, bounds.runtime_s)


def _within_caps(policy: ApprovalPolicy, bounds: ActionBounds | None) -> bool:
    if not _caps_restricted(policy):
        return True
    if bounds is None:
        return False
    return _bounds_inside(policy, bounds)


def _bad_argv(argv: tuple[str, ...]) -> bool:
    return any(not isinstance(part, str) for part in argv)


def _check_runtime(runtime: int | None) -> None:
    if runtime is None:
        return
    if isinstance(runtime, bool) or not isinstance(runtime, int) or runtime < 0:
        raise ValueError("runtime_s must be a non-negative int")


def _validate_bounds(
    argv: tuple[str, ...],
    runtime: int | None,
    writes: bool,
    shares: bool,
) -> None:
    if not isinstance(argv, tuple) or _bad_argv(argv):
        raise ValueError("argv must be a tuple of strings")
    _check_runtime(runtime)
    if not isinstance(writes, bool) or not isinstance(shares, bool):
        raise ValueError("writes and shares_data must be bool")


__all__ = [
    "ActionBounds",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ApprovalRequest",
    "AskFn",
    "DecideFn",
    "console_ask_fn",
    "console_decide_fn",
]
