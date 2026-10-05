"""Audit a declared batch protocol on one ordered log, without asserting durability.

Input row order is log order. Sequences must form the contiguous prefix beginning
at first_sequence, and recorded_at clocks must be nondecreasing. Every batch
starts unbegun; this is not a replay continuation with hidden initial state.
Legal transitions are begin -> zero or more writes -> one commit/abort. Invalid
transitions are diagnosed without changing the last legal state. Commit requires
all declared members exactly once, with matching hash labels and optional order;
abort does not require complete membership. Hashes are opaque supplied labels.

A checkpoint declares exactly the batch IDs whose first legal terminal transition
is commit at or before through_sequence. This definition includes commits with
membership errors: a logged commit does not prove a valid transaction. Checkpoint
comparison is withheld if sequence ordering is invalid. Checkpoints neither
flush data nor prove that an external store committed anything.

External durability claims reference a logged commit sequence and an evidence ID;
their observed_at clock must not precede that commit. Only this reference/clock
consistency is checked. Evidence bytes, signatures, storage durability, atomicity,
isolation and rollback effects are outside this audit's authority.

Bounds: 2000 batch declarations, 20000 events and expected members, 200 checkpoints
with 20000 total claimed IDs, 2000 durability claims, and 200 diagnostics. Explicit
checkpoint comparison work is bounded by batch_count * checkpoint_count (maximum
400000). Batch and checkpoint result pages have independent offsets and limits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Digest = Annotated[str, Field(strict=True, pattern=r"\A[0-9a-f]{64}\z")]
Sequence = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
State = Literal["not_started", "open", "committed", "aborted"]


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


class Member(InputModel):
    member_id: Name
    payload_hash: Digest


class Batch(InputModel):
    batch_id: Name
    expected_members: list[Member] = Field(max_length=20_000)
    write_order: Literal["declared", "any"] = "declared"
    required_terminal: Literal["commit", "abort", "either", "optional"] = "commit"


class Event(InputModel):
    sequence: Sequence
    recorded_at: AwareDatetime
    kind: Literal["begin", "write", "commit", "abort", "checkpoint"]
    batch_id: Name | None = None
    member_id: Name | None = None
    payload_hash: Digest | None = None
    through_sequence: int | None = Field(default=None, strict=True, ge=-1, le=2**63 - 1)
    committed_batch_ids: list[Name] | None = Field(default=None, max_length=2000)

    @field_validator("recorded_at", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def event_shape(self) -> Self:
        if self.kind == "checkpoint":
            if (
                self.batch_id is not None
                or self.member_id is not None
                or self.payload_hash is not None
            ):
                raise ValueError("checkpoints cannot carry batch/write fields")
            if self.through_sequence is None or self.committed_batch_ids is None:
                raise ValueError("checkpoints require through_sequence and committed_batch_ids")
        else:
            if (
                self.batch_id is None
                or self.through_sequence is not None
                or self.committed_batch_ids is not None
            ):
                raise ValueError("batch events require batch_id and cannot carry checkpoint fields")
            if self.kind == "write":
                if self.member_id is None or self.payload_hash is None:
                    raise ValueError("writes require member_id and payload_hash")
            elif self.member_id is not None or self.payload_hash is not None:
                raise ValueError("only writes may carry member/hash fields")
        return self


class DurabilityClaim(InputModel):
    evidence_id: Name
    batch_id: Name
    commit_sequence: Sequence
    observed_at: AwareDatetime

    @field_validator("observed_at", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    batches: list[Batch] = Field(max_length=2000)
    events: list[Event] = Field(max_length=20_000)
    durability_claims: list[DurabilityClaim] = Field(default_factory=list, max_length=2000)
    first_sequence: Sequence = 0
    allow_open_batches: bool = Field(default=False, strict=True)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=2000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)
    checkpoint_offset: int = Field(default=0, strict=True, ge=0, le=200)
    checkpoint_limit: int = Field(default=50, strict=True, ge=1, le=100)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if sum(len(batch.expected_members) for batch in self.batches) > 20_000:
            raise ValueError("at most 20000 expected-member declarations are supported")
        checkpoints = [row for row in self.events if row.kind == "checkpoint"]
        if (
            len(checkpoints) > 200
            or sum(len(row.committed_batch_ids or []) for row in checkpoints) > 20_000
        ):
            raise ValueError(
                "at most 200 checkpoints and 20000 claimed checkpoint IDs are supported"
            )
        if self.events and self.first_sequence + len(self.events) - 1 > 2**63 - 1:
            raise ValueError("the declared sequence prefix exceeds signed 64-bit range")
        return self


class Finding(OutputModel):
    code: str
    batch_row_index: int | None = None
    event_row_index: int | None = None
    other_event_row_index: int | None = None
    claim_row_index: int | None = None
    member_id: str | None = None


class BatchResult(OutputModel):
    batch_row_index: int
    batch_id: str
    state: State
    begin_event_row_index: int | None
    terminal_event_row_index: int | None
    logged_commit_sequence: int | None
    expected_member_count: int
    observed_write_count: int
    distinct_written_members: int
    matching_checkpoint_count: int
    consistent_durability_claim_count: int
    passed: bool


class CheckpointResult(OutputModel):
    event_row_index: int
    through_sequence: int
    claimed_commit_count: int
    expected_logged_commit_count: int | None
    missing_logged_commits: int | None
    unexpected_claimed_commits: int | None
    matches_logged_prefix: bool | None


class Output(OutputModel):
    passed: bool
    sequence_order_valid: bool
    recorded_clock_order_valid: bool
    batch_count: int
    event_count: int
    checkpoint_count: int
    durability_claim_count: int
    legal_logged_commit_count: int
    consistent_durability_claim_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    batches: list[BatchResult]
    offset: int
    has_more: bool
    checkpoints: list[CheckpointResult]
    checkpoint_offset: int
    checkpoints_have_more: bool
    payload_bytes_verified: Literal[False] = False
    external_durability_verified: Literal[False] = False
    atomicity_or_isolation_verified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid: set[int] = set()

    def report(
        code: str,
        batch: int | None = None,
        event: int | None = None,
        other: int | None = None,
        claim: int | None = None,
        member: str | None = None,
    ) -> None:
        issues[code] += 1
        if batch is not None:
            invalid.add(batch)
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    batch_row_index=batch,
                    event_row_index=event,
                    other_event_row_index=other,
                    claim_row_index=claim,
                    member_id=member,
                )
            )

    declarations: dict[str, list[int]] = defaultdict(list)
    for index, batch in enumerate(request.batches):
        declarations[batch.batch_id].append(index)
    unique = {name: rows[0] for name, rows in declarations.items() if len(rows) == 1}
    expected: list[dict[str, str]] = []
    for index, batch in enumerate(request.batches):
        if len(declarations[batch.batch_id]) != 1:
            report("ambiguous_batch_id", batch=index)
        counts = Counter(member.member_id for member in batch.expected_members)
        for member, count in counts.items():
            if count > 1:
                report("ambiguous_expected_member", batch=index, member=member)
        expected.append(
            {
                member.member_id: member.payload_hash
                for member in batch.expected_members
                if counts[member.member_id] == 1
            }
        )

    sequence_valid = clock_valid = True
    for index, event in enumerate(request.events):
        if event.sequence != request.first_sequence + index:
            sequence_valid = False
            report("noncontiguous_or_out_of_order_sequence", event=index)
        if index and event.recorded_at < request.events[index - 1].recorded_at:
            clock_valid = False
            report("recorded_clock_reversal", event=index, other=index - 1)

    states: list[State] = ["not_started"] * len(request.batches)
    begins: list[int | None] = [None] * len(request.batches)
    terminals: list[int | None] = [None] * len(request.batches)
    writes: list[dict[str, int]] = [{} for _ in request.batches]
    write_counts = [0] * len(request.batches)
    checkpoint_rows: list[int] = []
    for index, event in enumerate(request.events):
        if event.kind == "checkpoint":
            checkpoint_rows.append(index)
            continue
        position = unique.get(event.batch_id or "")
        if position is None:
            report("unresolved_event_batch", event=index)
            continue
        state = states[position]
        if event.kind == "begin":
            if state != "not_started":
                report(
                    "begin_after_batch_started", batch=position, event=index, other=begins[position]
                )
            else:
                states[position], begins[position] = "open", index
            continue
        if state != "open":
            report(
                "event_outside_open_batch", batch=position, event=index, other=terminals[position]
            )
            continue
        if event.kind == "write":
            member = event.member_id or ""
            ordinal = write_counts[position]
            write_counts[position] += 1
            if member in writes[position]:
                report(
                    "duplicate_member_write",
                    batch=position,
                    event=index,
                    other=writes[position][member],
                    member=member,
                )
            else:
                writes[position][member] = index
            if member not in expected[position]:
                report("unexpected_or_ambiguous_member", batch=position, event=index, member=member)
            elif event.payload_hash != expected[position][member]:
                report("member_hash_label_mismatch", batch=position, event=index, member=member)
            specification = request.batches[position]
            if specification.write_order == "declared" and (
                ordinal >= len(specification.expected_members)
                or specification.expected_members[ordinal].member_id != member
            ):
                report("member_write_order_mismatch", batch=position, event=index, member=member)
        else:
            states[position] = "committed" if event.kind == "commit" else "aborted"
            terminals[position] = index
            if event.kind == "commit":
                for member in sorted(set(expected[position]) - set(writes[position])):
                    report("commit_missing_member", batch=position, event=index, member=member)
    for index, batch in enumerate(request.batches):
        state = states[index]
        if state == "open" and not request.allow_open_batches:
            report("unterminated_batch", batch=index)
        required = batch.required_terminal
        if required != "optional" and (
            state not in ("committed", "aborted")
            or (required == "commit" and state != "committed")
            or (required == "abort" and state != "aborted")
        ):
            report("required_terminal_not_observed", batch=index)

    committed = [
        (index, terminal)
        for index, terminal in enumerate(terminals)
        if states[index] == "committed" and terminal is not None
    ]
    checkpoint_matches = [0] * len(request.batches)
    checkpoint_results: list[CheckpointResult] = []
    for checkpoint_index, event_index in enumerate(checkpoint_rows):
        event = request.events[event_index]
        through = event.through_sequence
        if through is None:
            raise ValueError("checkpoint through_sequence is missing")
        claimed = event.committed_batch_ids or []
        claimed_set = set(claimed)
        duplicate = len(claimed) != len(claimed_set)
        if duplicate:
            report("duplicate_checkpoint_batch", event=event_index)
        in_range = request.first_sequence - 1 <= through < event.sequence
        expected_count = missing_count = unexpected_count = None
        matches: bool | None = None
        if not sequence_valid or not in_range:
            report("unassessable_checkpoint_prefix", event=event_index)
        else:
            covered = [
                position
                for position, terminal in committed
                if request.events[terminal].sequence <= through
            ]
            expected_ids = {request.batches[position].batch_id for position in covered}
            expected_count = len(expected_ids)
            missing_count, unexpected_count = (
                len(expected_ids - claimed_set),
                len(claimed_set - expected_ids),
            )
            matches = not duplicate and not missing_count and not unexpected_count
            if not matches:
                report("checkpoint_commit_set_mismatch", event=event_index)
            else:
                for position in covered:
                    checkpoint_matches[position] += 1
        if (
            request.checkpoint_offset
            <= checkpoint_index
            < request.checkpoint_offset + request.checkpoint_limit
        ):
            checkpoint_results.append(
                CheckpointResult(
                    event_row_index=event_index,
                    through_sequence=through,
                    claimed_commit_count=len(claimed),
                    expected_logged_commit_count=expected_count,
                    missing_logged_commits=missing_count,
                    unexpected_claimed_commits=unexpected_count,
                    matches_logged_prefix=matches,
                )
            )

    evidence_counts = Counter(claim.evidence_id for claim in request.durability_claims)
    consistent_claims = [0] * len(request.batches)
    for index, claim in enumerate(request.durability_claims):
        position = unique.get(claim.batch_id)
        if evidence_counts[claim.evidence_id] != 1:
            report("ambiguous_evidence_id", batch=position, claim=index)
            continue
        terminal = None if position is None else terminals[position]
        if position is None or states[position] != "committed" or terminal is None:
            report("claim_without_logged_commit", batch=position, claim=index)
            continue
        event = request.events[terminal]
        if not sequence_valid or claim.commit_sequence != event.sequence:
            report("claim_commit_sequence_mismatch", batch=position, event=terminal, claim=index)
        elif claim.observed_at < event.recorded_at:
            report("claim_precedes_logged_commit", batch=position, event=terminal, claim=index)
        else:
            consistent_claims[position] += 1
    results: list[BatchResult] = []
    for index in range(request.offset, min(len(request.batches), request.offset + request.limit)):
        terminal = terminals[index]
        results.append(
            BatchResult(
                batch_row_index=index,
                batch_id=request.batches[index].batch_id,
                state=states[index],
                begin_event_row_index=begins[index],
                terminal_event_row_index=terminal,
                logged_commit_sequence=request.events[terminal].sequence
                if states[index] == "committed" and terminal is not None
                else None,
                expected_member_count=len(request.batches[index].expected_members),
                observed_write_count=write_counts[index],
                distinct_written_members=len(writes[index]),
                matching_checkpoint_count=checkpoint_matches[index],
                consistent_durability_claim_count=consistent_claims[index],
                passed=index not in invalid and sequence_valid and clock_valid,
            )
        )
    return Output(
        passed=not issues,
        sequence_order_valid=sequence_valid,
        recorded_clock_order_valid=clock_valid,
        batch_count=len(request.batches),
        event_count=len(request.events),
        checkpoint_count=len(checkpoint_rows),
        durability_claim_count=len(request.durability_claims),
        legal_logged_commit_count=len(committed),
        consistent_durability_claim_count=sum(consistent_claims),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        batches=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.batches),
        checkpoints=checkpoint_results,
        checkpoint_offset=request.checkpoint_offset,
        checkpoints_have_more=request.checkpoint_offset + len(checkpoint_results)
        < len(checkpoint_rows),
    )


OPERATION = Operation(
    id="skills.audit_transaction_batches",
    kind="skill",
    description="Audit declared batch state transitions, member hash labels and logged checkpoint prefixes while separating external durability assertions from verified storage evidence.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
