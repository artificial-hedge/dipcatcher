"""Audit a declared lease/fencing model without authorizing external access.

Every task starts unleased at first_sequence. Per-task sequence numbers must be
unique and contiguous; event clocks are nondecreasing and publication cannot
precede an event. Each view uses only events published by that view's clock.
Grant requires no active lease and an epoch strictly greater than every prior
legal grant. Renew/revoke require the exact active worker and epoch. Renewal
strictly extends expiry. Leases are active on [grant, expiry); an event at expiry
cannot renew the expired lease, but may grant a higher epoch to any worker.

Invalid state transitions do not mutate the last legal lease, but permanently
taint that view's history: no later access is labeled consistent from it. A
sequence or causal-clock fault stops replay. Same-clock log events follow
sequence order; an access at that clock observes all published events at that
clock. There is no implicit takeover, lease resurrection, clock skew tolerance,
authority election, or acceptance of a merely larger undeclared fencing epoch.

decision_time controls the audit view. A visible access claim is evaluated using
the separate event view available at its access_time, not metadata learned later.
Thus future renewals, revocations, duplicate sequences and bad events cannot
change an earlier access outcome. Claim publication must follow access. Findings
cover the decision-visible log/claims; supplied future rows are explicitly counted
but unaudited until visible. A passed audit is consistency of supplied declarations,
not proof that a storage system enforces fencing or that a worker was authenticated.

Bounds: 10000 events, 500 task IDs across events/claims, 1000 claims, 200 diagnostics,
200 task/claim results per page. Global replay plus cached distinct task/access
views is conservatively charged against at most 500000 event-row visits, including
future rows and future claims. Sorting is O(W log N), state transitions O(W).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Sequence = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("clock must be an explicit timezone-aware ISO datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class Event(InputModel):
    task_id: Name
    sequence: Sequence
    event_time: AwareDatetime
    available_time: AwareDatetime
    action: Literal["grant", "renew", "revoke"]
    worker_id: Name
    epoch: Sequence
    expires_time: AwareDatetime | None = None

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("expires_time", mode="before")
    @classmethod
    def optional_clock(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)

    @model_validator(mode="after")
    def expiry_field(self) -> Self:
        if (self.action == "revoke") != (self.expires_time is None):
            raise ValueError("grant/renew require expiry; revoke cannot carry expiry")
        return self


class AccessClaim(InputModel):
    task_id: Name
    worker_id: Name
    epoch: Sequence
    access_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("access_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    events: list[Event] = Field(max_length=10_000)
    claims: list[AccessClaim] = Field(default_factory=list, max_length=1000)
    decision_time: AwareDatetime
    first_sequence: Sequence = 0
    max_replay_rows: int = Field(default=100_000, strict=True, ge=0, le=500_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    task_offset: int = Field(default=0, strict=True, ge=0, le=500)
    task_limit: int = Field(default=100, strict=True, ge=1, le=200)
    claim_offset: int = Field(default=0, strict=True, ge=0, le=1000)
    claim_limit: int = Field(default=100, strict=True, ge=1, le=200)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        counts = Counter(row.task_id for row in self.events)
        if len(set(counts) | {row.task_id for row in self.claims}) > 500:
            raise ValueError("at most 500 task IDs are supported")
        views = {(row.task_id, row.access_time) for row in self.claims}
        if len(self.events) + sum(counts[task] for task, _ in views) > self.max_replay_rows:
            raise ValueError("global plus claim-view replay work exceeds max_replay_rows")
        return self


class Finding(OutputModel):
    code: str
    task_id: str
    event_row_index: int | None = None
    other_event_row_index: int | None = None
    claim_row_index: int | None = None


class TaskResult(OutputModel):
    task_id: str
    history_consistent: bool
    projected_state: Literal["unleased", "active", "expired", "revoked"]
    active_worker_id: str | None
    active_epoch: int | None
    highest_legal_grant_epoch: int | None
    expires_time: datetime | None
    last_legal_event_row_index: int | None
    grant_event_row_index: int | None
    first_fault_event_row_index: int | None
    visible_event_count: int
    applied_event_count: int
    maximum_applied_available_time: datetime | None


class ClaimResult(OutputModel):
    claim_row_index: int
    status: Literal[
        "not_available",
        "invalid_claim_clock",
        "untrusted_history",
        "no_active_lease",
        "stale_fencing_epoch",
        "undeclared_fencing_epoch",
        "worker_mismatch",
        "consistent",
    ]
    active_worker_id: str | None
    active_epoch: int | None
    grant_event_row_index: int | None
    last_legal_event_row_index: int | None
    first_fault_event_row_index: int | None
    maximum_lease_available_time: datetime | None


class Output(OutputModel):
    passed: bool
    task_count: int
    event_count: int
    visible_event_count: int
    future_event_count: int
    claim_count: int
    future_claim_count: int
    declared_replay_rows: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    tasks: list[TaskResult]
    task_offset: int
    tasks_have_more: bool
    claim_status_counts: dict[str, int]
    claims: list[ClaimResult]
    claim_offset: int
    claims_have_more: bool
    external_fencing_enforcement_verified: Literal[False] = False
    worker_authentication_verified: Literal[False] = False


@dataclass
class _Lease:
    worker: str
    epoch: int
    expires: datetime
    grant: int
    last: int
    revoked: bool = False


@dataclass
class _View:
    lease: _Lease | None
    highest_epoch: int | None
    tainted: bool
    fault_row: int | None
    visible: int
    applied: int
    maximum: datetime | None


_Reporter = Callable[[str, str, int | None, int | None, int | None], None]


def _replay(
    request: Input, task: str, rows: list[int], clock: datetime, report: _Reporter | None
) -> _View:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in rows:
        event = request.events[index]
        if event.available_time <= clock:
            groups[event.sequence].append(index)
    view = _View(None, None, False, None, sum(map(len, groups.values())), 0, None)
    expected = request.first_sequence
    previous_time: datetime | None = None

    def fault(code: str, index: int, other: int | None = None) -> None:
        if not view.tainted:
            view.fault_row = index
        view.tainted = True
        if report is not None:
            report(code, task, index, other, None)

    for sequence in sorted(groups):
        indexes = groups[sequence]
        index = indexes[0]
        if sequence != expected:
            fault("sequence_gap_or_before_start", index)
            break
        if len(indexes) != 1:
            fault("ambiguous_sequence", index, indexes[1])
            break
        event = request.events[index]
        if event.event_time > event.available_time:
            fault("publication_precedes_event", index)
            break
        if previous_time is not None and event.event_time < previous_time:
            fault("event_clock_regression", index)
            break
        previous_time = event.event_time
        expected += 1
        lease = view.lease
        active = lease is not None and not lease.revoked and event.event_time < lease.expires
        if event.action == "grant":
            if active:
                fault("grant_while_lease_active", index)
                continue
            if view.highest_epoch is not None and event.epoch <= view.highest_epoch:
                fault("grant_epoch_not_increasing", index)
                continue
            assert event.expires_time is not None
            if event.expires_time <= event.event_time:
                fault("grant_expiry_not_after_event", index)
                continue
            view.lease = _Lease(event.worker_id, event.epoch, event.expires_time, index, index)
            view.highest_epoch = event.epoch
        elif lease is None or not active:
            fault("transition_without_active_lease", index)
            continue
        elif (event.worker_id, event.epoch) != (lease.worker, lease.epoch):
            fault("transition_owner_or_epoch_mismatch", index)
            continue
        elif event.action == "renew":
            assert event.expires_time is not None
            if event.expires_time <= lease.expires:
                fault("renewal_does_not_extend_expiry", index)
                continue
            lease.expires = event.expires_time
            lease.last = index
        else:
            lease.revoked = True
            lease.last = index
        view.applied += 1
        view.maximum = (
            event.available_time
            if view.maximum is None
            else max(view.maximum, event.available_time)
        )
    return view


def _active(view: _View, clock: datetime) -> _Lease | None:
    lease = view.lease
    return lease if lease is not None and not lease.revoked and clock < lease.expires else None


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []

    def report(
        code: str, task: str, event: int | None, other: int | None, claim: int | None
    ) -> None:
        issues[code] += 1
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    task_id=task,
                    event_row_index=event,
                    other_event_row_index=other,
                    claim_row_index=claim,
                )
            )

    rows: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(request.events):
        rows[event.task_id].append(index)
    tasks = sorted(
        {event.task_id for event in request.events if event.available_time <= request.decision_time}
        | {
            claim.task_id
            for claim in request.claims
            if claim.available_time <= request.decision_time
        }
    )
    views: dict[tuple[str, datetime], _View] = {}
    task_results: list[TaskResult] = []
    visible_events = 0
    for position, task in enumerate(tasks):
        view = _replay(request, task, rows[task], request.decision_time, report)
        views[task, request.decision_time] = view
        visible_events += view.visible
        if request.task_offset <= position < request.task_offset + request.task_limit:
            active = _active(view, request.decision_time)
            lease = view.lease
            task_results.append(
                TaskResult(
                    task_id=task,
                    history_consistent=not view.tainted,
                    projected_state="unleased"
                    if lease is None
                    else "revoked"
                    if lease.revoked
                    else "active"
                    if active is not None
                    else "expired",
                    active_worker_id=None if active is None else active.worker,
                    active_epoch=None if active is None else active.epoch,
                    highest_legal_grant_epoch=view.highest_epoch,
                    expires_time=None if lease is None else lease.expires,
                    last_legal_event_row_index=None if lease is None else lease.last,
                    grant_event_row_index=None if lease is None else lease.grant,
                    first_fault_event_row_index=view.fault_row,
                    visible_event_count=view.visible,
                    applied_event_count=view.applied,
                    maximum_applied_available_time=view.maximum,
                )
            )
    claim_results: list[ClaimResult] = []
    claim_counts: Counter[str] = Counter()
    for index, claim in enumerate(request.claims):
        active = None
        claim_view: _View | None = None
        status: Literal[
            "not_available",
            "invalid_claim_clock",
            "untrusted_history",
            "no_active_lease",
            "stale_fencing_epoch",
            "undeclared_fencing_epoch",
            "worker_mismatch",
            "consistent",
        ]
        if claim.available_time > request.decision_time:
            status = "not_available"
        elif claim.access_time > claim.available_time:
            status = "invalid_claim_clock"
        else:
            key = (claim.task_id, claim.access_time)
            if key not in views:
                views[key] = _replay(
                    request, claim.task_id, rows[claim.task_id], claim.access_time, None
                )
            claim_view = views[key]
            active = _active(claim_view, claim.access_time)
            if claim_view.tainted:
                status = "untrusted_history"
            elif active is None:
                status = "no_active_lease"
            elif claim.epoch < active.epoch:
                status = "stale_fencing_epoch"
            elif claim.epoch > active.epoch:
                status = "undeclared_fencing_epoch"
            elif claim.worker_id != active.worker:
                status = "worker_mismatch"
            else:
                status = "consistent"
        claim_counts[status] += 1
        if status not in {"not_available", "consistent"}:
            report(
                "access_" + status,
                claim.task_id,
                None if claim_view is None else claim_view.fault_row,
                None,
                index,
            )
        if request.claim_offset <= index < request.claim_offset + request.claim_limit:
            claim_results.append(
                ClaimResult(
                    claim_row_index=index,
                    status=status,
                    active_worker_id=None if active is None else active.worker,
                    active_epoch=None if active is None else active.epoch,
                    grant_event_row_index=None
                    if claim_view is None or claim_view.lease is None
                    else claim_view.lease.grant,
                    last_legal_event_row_index=None
                    if claim_view is None or claim_view.lease is None
                    else claim_view.lease.last,
                    first_fault_event_row_index=None
                    if claim_view is None
                    else claim_view.fault_row,
                    maximum_lease_available_time=None if claim_view is None else claim_view.maximum,
                )
            )
    charged_views = {(claim.task_id, claim.access_time) for claim in request.claims}
    return Output(
        passed=not issues,
        task_count=len(tasks),
        event_count=len(request.events),
        visible_event_count=visible_events,
        future_event_count=len(request.events) - visible_events,
        claim_count=len(request.claims),
        future_claim_count=claim_counts["not_available"],
        declared_replay_rows=len(request.events)
        + sum(len(rows[task]) for task, _ in charged_views),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        tasks=task_results,
        task_offset=request.task_offset,
        tasks_have_more=request.task_offset + len(task_results) < len(tasks),
        claim_status_counts=dict(sorted(claim_counts.items())),
        claims=claim_results,
        claim_offset=request.claim_offset,
        claims_have_more=request.claim_offset + len(claim_results) < len(request.claims),
    )


OPERATION = Operation(
    id="skills.audit_task_leases",
    kind="skill",
    description="Audit declared per-task lease transitions, strict fencing epochs and historical worker access claims without asserting external enforcement.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
