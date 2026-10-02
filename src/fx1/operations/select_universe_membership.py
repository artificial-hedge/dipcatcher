"""Select membership at a historical decision using observable interval revisions.

Each membership_id versions one inclusion/exclusion interval. Unbounded intervals
also represent state-transition events. Among intervals active at the decision,
the latest effective_from wins; an expired temporary exclusion can reveal an
older still-active inclusion. Conflicting states at the same effective clock
are ambiguous. No current-universe filter is applied to historical decisions.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]


def _clock(value: object) -> datetime:
    if isinstance(value, str):
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


class Membership(InputModel):
    membership_id: Name
    security_id: Name
    state: Literal["included", "excluded"]
    effective_from: AwareDatetime
    effective_to: AwareDatetime | None = None
    available_time: AwareDatetime
    revision_id: Name
    source: Name

    @field_validator("effective_from", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("effective_to", mode="before")
    @classmethod
    def optional_clock(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)

    @model_validator(mode="after")
    def nonempty_interval(self) -> Self:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be strictly after effective_from")
        return self


class Input(InputModel):
    memberships: list[Membership] = Field(max_length=10_000)
    decision_time: AwareDatetime
    security_ids: list[Name] | None = Field(default=None, min_length=1, max_length=10_000)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=200, strict=True, ge=1, le=500)
    max_lineage_per_security: int = Field(default=10, strict=True, ge=2, le=20)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("security_ids")
    @classmethod
    def unique_securities(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("security_ids must be distinct")
        return value


class Selection(OutputModel):
    security_id: str
    status: Literal["included", "excluded", "missing", "ambiguous"]
    effective_from: datetime | None
    selected_row_index: int | None
    supporting_row_indices: list[int]
    omitted_supporting_rows: int
    conflicting_membership_ids: int


class MembershipLineage(OutputModel):
    membership_row_index: int
    membership: Membership


class Output(OutputModel):
    decision_time: datetime
    security_count: int
    included_count: int
    excluded_count: int
    missing_count: int
    ambiguous_count: int
    unavailable_input_rows: int
    revision_policy: str = "latest_availability_per_membership_id_before_effective_filtering"
    selection_policy: str = "latest_active_effective_from_with_conflicting_states_ambiguous"
    interval_policy: str = "effective_from_inclusive_effective_to_exclusive"
    offset: int
    next_offset: int | None
    selections: list[Selection]
    membership_lineage: list[MembershipLineage]


def _signature(row: Membership) -> tuple[object, ...]:
    return row.security_id, row.state, row.effective_from, row.effective_to, row.source


def execute(request: Input, context: OperationContext) -> Output:
    known: dict[str, list[int]] = {}
    unavailable = 0
    for index, row in enumerate(request.memberships):
        if row.available_time > request.decision_time:
            unavailable += 1
            continue
        prior = known.get(row.membership_id)
        if prior is None or row.available_time > request.memberships[prior[0]].available_time:
            known[row.membership_id] = [index]
        elif row.available_time == request.memberships[prior[0]].available_time:
            prior.append(index)

    active: dict[str, list[int]] = defaultdict(list)
    conflicted: set[str] = set()
    for membership_id, indices in known.items():
        signatures = {_signature(request.memberships[index]) for index in indices}
        if len(signatures) > 1:
            conflicted.add(membership_id)
            representatives = indices
        else:
            representatives = [
                max(indices, key=lambda index: (request.memberships[index].revision_id, -index))
            ]
        for index in representatives:
            row = request.memberships[index]
            if row.effective_from <= request.decision_time and (
                row.effective_to is None or request.decision_time < row.effective_to
            ):
                active[row.security_id].append(index)

    securities = sorted(
        request.security_ids
        if request.security_ids is not None
        else {row.security_id for row in request.memberships}
    )
    counts = {"included": 0, "excluded": 0, "missing": 0, "ambiguous": 0}
    selections: list[Selection] = []
    lineage: set[int] = set()
    for position, security in enumerate(securities):
        indices = active.get(security, [])
        effective: datetime | None = None
        selected: int | None = None
        winners: list[int] = []
        conflict_ids: set[str] = set()
        status: Literal["included", "excluded", "missing", "ambiguous"] = "missing"
        if indices:
            effective = max(request.memberships[index].effective_from for index in indices)
            winners = [index for index in indices if request.memberships[index].effective_from == effective]
            states = {request.memberships[index].state for index in winners}
            conflict_ids = {
                request.memberships[index].membership_id
                for index in winners
                if request.memberships[index].membership_id in conflicted
            }
            if len(states) > 1 or conflict_ids:
                status = "ambiguous"
            else:
                status = next(iter(states))
                selected = max(
                    winners,
                    key=lambda index: (
                        request.memberships[index].available_time,
                        request.memberships[index].revision_id,
                        -index,
                    ),
                )
        counts[status] += 1
        if request.offset <= position < request.offset + request.limit:
            priority = ([selected] if selected is not None else []) + sorted(winners)
            examples = list(dict.fromkeys(priority))[: request.max_lineage_per_security]
            lineage.update(examples)
            selections.append(
                Selection(
                    security_id=security,
                    status=status,
                    effective_from=effective,
                    selected_row_index=selected,
                    supporting_row_indices=examples,
                    omitted_supporting_rows=len(winners) - len(examples),
                    conflicting_membership_ids=len(conflict_ids),
                )
            )
    stop = request.offset + len(selections)
    return Output(
        decision_time=request.decision_time,
        security_count=len(securities),
        included_count=counts["included"],
        excluded_count=counts["excluded"],
        missing_count=counts["missing"],
        ambiguous_count=counts["ambiguous"],
        unavailable_input_rows=unavailable,
        offset=request.offset,
        next_offset=stop if stop < len(securities) else None,
        selections=selections,
        membership_lineage=[
            MembershipLineage(membership_row_index=index, membership=request.memberships[index])
            for index in sorted(lineage)
        ],
    )


OPERATION = Operation(
    id="skills.select_universe_membership",
    kind="skill",
    description=(
        "Select historical security membership from observable inclusion/exclusion interval "
        "revisions, preserving prior membership before delists. Report missing/ambiguous "
        "states, use explicit half-open intervals, and paginate securities with row lineage."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
