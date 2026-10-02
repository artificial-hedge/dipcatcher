"""Resolve bitemporal ticker/exchange mappings without guessing through ambiguity.

A mapping_id identifies one effective interval across corrections. Its latest
available version replaces older versions before effective-date filtering.
Conflicting records at that same availability clock remain ambiguous. Matching
is case-sensitive and does not normalize exchange or ticker spellings.
"""

from __future__ import annotations

from bisect import bisect_right
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


class Mapping(InputModel):
    mapping_id: Name
    security_id: Name
    ticker: Name
    exchange: Name
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None = None
    available_time: AwareDatetime
    revision_id: Name
    source: Name

    @field_validator("valid_from", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("valid_to", mode="before")
    @classmethod
    def optional_clock(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)

    @model_validator(mode="after")
    def nonempty_interval(self) -> Self:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be strictly after valid_from")
        return self


class Query(InputModel):
    ticker: Name
    exchange: Name
    decision_time: AwareDatetime

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    mappings: list[Mapping] = Field(max_length=10_000)
    queries: list[Query] = Field(min_length=1, max_length=10_000)
    work_budget: int = Field(default=250_000, strict=True, ge=1, le=1_000_000)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)
    max_candidates_per_query: int = Field(default=10, strict=True, ge=2, le=20)


class Resolution(OutputModel):
    query_row_index: int
    ticker: str
    exchange: str
    decision_time: datetime
    status: Literal["resolved", "missing", "ambiguous"]
    security_id: str | None
    selected_mapping_row_index: int | None
    candidate_mapping_rows: int
    candidate_security_count: int
    conflicting_mapping_ids: int
    candidate_row_indices: list[int]
    omitted_candidate_rows: int


class MappingLineage(OutputModel):
    mapping_row_index: int
    mapping: Mapping


class Output(OutputModel):
    query_count: int
    resolved_count: int
    missing_count: int
    ambiguous_count: int
    work_units: int
    revision_policy: str = "latest_availability_then_compatible_greatest_revision_id_earliest_row"
    interval_policy: str = "valid_from_inclusive_valid_to_exclusive"
    offset: int
    next_offset: int | None
    resolutions: list[Resolution]
    mapping_lineage: list[MappingLineage]


def _signature(row: Mapping) -> tuple[object, ...]:
    return row.security_id, row.ticker, row.exchange, row.valid_from, row.valid_to, row.source


def execute(request: Input, context: OperationContext) -> Output:
    versions: dict[str, dict[datetime, list[int]]] = defaultdict(lambda: defaultdict(list))
    lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, row in enumerate(request.mappings):
        versions[row.mapping_id][row.available_time].append(index)
        lookup[(row.ticker, row.exchange)].add(row.mapping_id)
    clocks = {mapping_id: sorted(vintages) for mapping_id, vintages in versions.items()}
    minimum_work = sum(len(lookup.get((query.ticker, query.exchange), ())) for query in request.queries)
    if minimum_work > request.work_budget:
        raise ValueError("candidate mapping/query work exceeds work_budget; split the query batch")

    work = resolved = missing = ambiguous = 0
    output: list[Resolution] = []
    lineage_indices: set[int] = set()
    for query_index, query in enumerate(request.queries):
        candidates: list[int] = []
        conflicting_ids = 0
        for mapping_id in sorted(lookup.get((query.ticker, query.exchange), ())):
            work += 1
            if work > request.work_budget:
                raise ValueError("candidate mapping work exceeds work_budget; split the query batch")
            position = bisect_right(clocks[mapping_id], query.decision_time) - 1
            if position < 0:
                continue
            indices = versions[mapping_id][clocks[mapping_id][position]]
            work += len(indices)
            if work > request.work_budget:
                raise ValueError("revision inspection work exceeds work_budget; split the query batch")
            signatures = {_signature(request.mappings[index]) for index in indices}
            applicable = [
                index
                for index in indices
                if (request.mappings[index].ticker, request.mappings[index].exchange)
                == (query.ticker, query.exchange)
                and request.mappings[index].valid_from <= query.decision_time
                and (
                    request.mappings[index].valid_to is None
                    or query.decision_time < request.mappings[index].valid_to
                )
            ]
            if not applicable:
                continue
            if len(signatures) > 1:
                conflicting_ids += 1
                candidates.extend(applicable)
            else:
                candidates.append(
                    max(applicable, key=lambda index: (request.mappings[index].revision_id, -index))
                )
        identities = {request.mappings[index].security_id for index in candidates}
        selected_index: int | None = None
        security: str | None = None
        status: Literal["resolved", "missing", "ambiguous"]
        if not candidates:
            status = "missing"
            missing += 1
        elif conflicting_ids or len(identities) != 1:
            status = "ambiguous"
            ambiguous += 1
        else:
            status = "resolved"
            resolved += 1
            security = next(iter(identities))
            selected_index = max(
                candidates,
                key=lambda index: (
                    request.mappings[index].available_time,
                    request.mappings[index].revision_id,
                    -index,
                ),
            )
        if request.offset <= query_index < request.offset + request.limit:
            # Include the chosen row even if other compatible mappings precede it.
            priority = ([selected_index] if selected_index is not None else []) + sorted(candidates)
            examples = list(dict.fromkeys(priority))[: request.max_candidates_per_query]
            lineage_indices.update(examples)
            output.append(
                Resolution(
                    query_row_index=query_index,
                    ticker=query.ticker,
                    exchange=query.exchange,
                    decision_time=query.decision_time,
                    status=status,
                    security_id=security,
                    selected_mapping_row_index=selected_index,
                    candidate_mapping_rows=len(candidates),
                    candidate_security_count=len(identities),
                    conflicting_mapping_ids=conflicting_ids,
                    candidate_row_indices=examples,
                    omitted_candidate_rows=len(candidates) - len(examples),
                )
            )

    stop = request.offset + len(output)
    return Output(
        query_count=len(request.queries),
        resolved_count=resolved,
        missing_count=missing,
        ambiguous_count=ambiguous,
        work_units=work,
        offset=request.offset,
        next_offset=stop if stop < len(request.queries) else None,
        resolutions=output,
        mapping_lineage=[
            MappingLineage(mapping_row_index=index, mapping=request.mappings[index])
            for index in sorted(lineage_indices)
        ],
    )


OPERATION = Operation(
    id="skills.resolve_security_identity",
    kind="skill",
    description=(
        "Resolve exact ticker/exchange queries through available security-master revisions "
        "and half-open effective intervals; report missing/ambiguous identities, retain row "
        "lineage, and enforce an explicit candidate-inspection budget."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
