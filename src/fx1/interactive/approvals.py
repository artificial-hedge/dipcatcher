"""Approval gates for consequential actions and private-data sharing.

The front door keeps the lab's fail-closed posture: approvals default to
*deny* whenever no interactive operator can answer (piped stdin, tests,
scripts). An interactive session asks on the console. Two reasons are
first-class:

- ``consequential`` — running a harness command that trains/optimizes or
  otherwise mutates lab state beyond cheap diagnostics;
- ``private_data`` — moving private flash context (or other machine-local
  material) into a model prompt or a web-search query.

A session grant covers exactly one ``(kind, key)`` pair and is reused without
a second prompt. Non-interactive callers pass an :class:`ApprovalPolicy` via
:meth:`ApprovalGate.non_interactive` and are never blocked on input. The gate
records every decision so a session can explain what was (and was not)
authorized. Prompts are serialized with a lock so worker threads share one
console question at a time.
"""

from __future__ import annotations

import threading
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


class ApprovalPolicy:
    """Explicit non-interactive allow-list. Everything else is denied.

    ``allow_kinds`` may approve a whole kind (for example ``read``). The kind
    ``consequential`` is never approved by kind alone; a consequential action
    needs its own key in ``allow_keys``.
    """

    def __init__(
        self,
        *,
        allow_keys: Collection[str] = (),
        allow_kinds: Collection[str] = (),
    ) -> None:
        self._allow_keys = frozenset(allow_keys)
        self._allow_kinds = frozenset(
            kind for kind in allow_kinds if kind.casefold() != "consequential"
        )

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        """``ONCE`` when the key (or a non-consequential kind) is listed, else ``DENY``."""
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

    def __repr__(self) -> str:
        question = redact_text(self.question)
        key = redact_text(self.key) if self.key else None
        return (
            "ApprovalRecord("
            f"kind={self.kind!r}, question={question!r}, approved={self.approved!r}, "
            f"timestamp={self.timestamp!r}, key={key!r}, scope={self.scope!r})"
        )


@dataclass
class ApprovalGate:
    """Fail-closed approval: ``ask`` returns False unless a prompter or policy approves.

    Existing callers that pass only ``ask_fn`` keep yes/no behavior. Pass
    ``key`` to reuse a session grant. ``interactive=False`` never calls a
    prompter. ``ApprovalGate()`` with neither a prompter nor a policy denies
    and records scope ``no-prompter``.
    """

    ask_fn: AskFn | None = None
    records: list[ApprovalRecord] = field(default_factory=list, repr=False)
    decide_fn: DecideFn | None = None
    policy: ApprovalPolicy | None = None
    interactive: bool = True
    clock: Callable[[], datetime] | None = None
    _session: dict[tuple[str, str], bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @classmethod
    def non_interactive(cls, policy: ApprovalPolicy) -> ApprovalGate:
        """A gate that applies *policy* and never reads stdin."""
        return cls(policy=policy, interactive=False)

    def ask(self, question: str, *, kind: str = "action", key: str | None = None) -> bool:
        """Approve *question*. A session grant for ``(kind, key)`` skips the prompt."""
        with self._lock:
            return self._ask_locked(question, kind=kind, key=key)

    def revoke(self, key: str | None = None) -> None:
        """Drop session grants. ``key=None`` clears every grant."""
        with self._lock:
            if key is None:
                self._session.clear()
                return
            self._session = {pair: True for pair in self._session if pair[1] != key}

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

    def _ask_locked(self, question: str, *, kind: str, key: str | None) -> bool:
        stable = key or None
        if stable is not None and (kind, stable) in self._session:
            self._append(kind, question, True, stable, "session-reuse")
            return True
        decision, scope = self._resolve(ApprovalRequest(stable or "", kind, question))
        decision, scope = _bind_session(decision, scope, stable)
        if decision is ApprovalDecision.SESSION and stable is not None:
            self._session[(kind, stable)] = True
        approved = decision is not ApprovalDecision.DENY
        self._append(kind, question, approved, stable, scope)
        return approved

    def _resolve(self, request: ApprovalRequest) -> tuple[ApprovalDecision, str]:
        if not self.interactive:
            return _from_policy(self.policy, request)
        if self.decide_fn is not None:
            return _from_decide(self.decide_fn, request)
        if self.ask_fn is not None:
            return _from_ask(self.ask_fn, request.question)
        if self.policy is not None:
            return _from_policy(self.policy, request)
        return ApprovalDecision.DENY, "no-prompter"

    def _append(
        self,
        kind: str,
        question: str,
        approved: bool,
        key: str | None,
        scope: str,
    ) -> None:
        self.records.append(
            ApprovalRecord(
                kind=kind,
                question=question,
                approved=approved,
                timestamp=self._now(),
                key=key,
                scope=scope,
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
            "key": redact_text(record.key) if record.key else None,
            "question": redact_text(record.question),
            "approved": record.approved,
            "scope": record.scope,
            "timestamp": record.timestamp,
            "reusable": reusable,
        }


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
) -> tuple[ApprovalDecision, str]:
    try:
        decision = decide_fn(request)
    except (Exception, KeyboardInterrupt):
        return ApprovalDecision.DENY, "denied"
    if decision is ApprovalDecision.ONCE:
        return ApprovalDecision.ONCE, "once"
    if decision is ApprovalDecision.SESSION:
        return ApprovalDecision.SESSION, "session"
    return ApprovalDecision.DENY, "denied"


def _from_ask(ask_fn: AskFn, question: str) -> tuple[ApprovalDecision, str]:
    try:
        approved = bool(ask_fn(question))
    except (Exception, KeyboardInterrupt):
        return ApprovalDecision.DENY, "denied"
    if approved:
        return ApprovalDecision.ONCE, "once"
    return ApprovalDecision.DENY, "denied"


def _from_policy(
    policy: ApprovalPolicy | None,
    request: ApprovalRequest,
) -> tuple[ApprovalDecision, str]:
    if policy is None:
        return ApprovalDecision.DENY, "no-prompter"
    decision = policy.decide(request)
    if decision is ApprovalDecision.ONCE:
        return ApprovalDecision.ONCE, "policy"
    return ApprovalDecision.DENY, "denied"


def console_ask_fn(input_fn: Callable[[str], str] = input) -> AskFn:
    """Interactive y/N prompter; anything but an explicit yes is a deny."""

    def ask(question: str) -> bool:
        try:
            answer = input_fn(f"approval required — {question} [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

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
        prompt = (
            f"approval required — {request.question} [y]es once / [a]lways this session / [N]o: "
        )
        try:
            answer = _read_decision(prompt, input_fn, output_fn)
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY
        return _parse_decision(answer)

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


__all__ = [
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
