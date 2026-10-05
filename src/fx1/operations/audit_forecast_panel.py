"""Audit an explicit forecast panel against completed-outcome metadata.

Forecast keys add model_version to (security_id, target, horizon, decision_time).
Realization keys omit model_version, allowing several models to share a label.
Horizon is an opaque label: the supplied target_time defines when its outcome
completes. This contract requires target_time > decision_time and realization
availability >= target_time. It does not apply to advance announcements.

Duplicate keys, including exact copies, block eligibility; no vintage is chosen
implicitly. A row is eligible only when its unique forecast and realization are
temporally valid, have equal target_time, and are known at evaluation_asof.
Values are inspected only for duplicate equality; no score or evidence is made.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, StrictFloat, StrictInt, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel, canonical_json

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Cell = (
    Annotated[str, Field(strict=True, max_length=512)] | StrictBool | StrictInt | StrictFloat | None
)
OutcomeKey = tuple[str, str, str, datetime]
ForecastKey = tuple[str, str, str, datetime, str]
Status = Literal[
    "invalid_forecast",
    "duplicate_forecast",
    "ambiguous_realization",
    "invalid_realization",
    "target_time_mismatch",
    "not_due",
    "missing_realization",
    "realization_not_available",
    "eligible",
]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError(
            "clock must be an explicit timezone-aware ISO datetime of at most 64 chars"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class Forecast(InputModel):
    security_id: Name
    target: Name
    horizon: Name
    model_version: Name
    decision_time: AwareDatetime
    target_time: AwareDatetime
    max_source_available_time: AwareDatetime
    forecast_available_time: AwareDatetime
    values: dict[Name, Cell] = Field(min_length=1, max_length=16)

    @field_validator(
        "decision_time",
        "target_time",
        "max_source_available_time",
        "forecast_available_time",
        mode="before",
    )
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class Realization(InputModel):
    security_id: Name
    target: Name
    horizon: Name
    decision_time: AwareDatetime
    target_time: AwareDatetime
    available_time: AwareDatetime
    revision_id: Name
    source: Name
    value: Cell

    @field_validator("decision_time", "target_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    forecasts: list[Forecast] = Field(max_length=10_000)
    realizations: list[Realization] = Field(max_length=10_000)
    evaluation_asof: AwareDatetime
    require_due_coverage: bool = Field(default=True, strict=True)
    require_no_orphan_realizations: bool = Field(default=False, strict=True)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indices: int = Field(default=5, strict=True, ge=1, le=10)

    @field_validator("evaluation_asof", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Finding(OutputModel):
    table: Literal["forecasts", "realizations"]
    code: str
    row_indices: list[int]
    omitted_row_indices: int = 0
    counterpart_row_index: int | None = None
    conflicts: bool = False


class Eligibility(OutputModel):
    forecast_row_index: int
    status: Status
    realization_row_index: int | None


class Output(OutputModel):
    forecast_rows: int
    realization_rows: int
    distinct_forecast_keys: int
    distinct_outcome_forecast_keys: int
    distinct_realization_keys: int
    duplicate_forecast_groups: int
    conflicting_forecast_groups: int
    duplicate_realization_groups: int
    conflicting_realization_groups: int
    invalid_forecast_rows: int
    invalid_realization_rows: int
    orphan_realization_keys: int
    orphan_realization_rows: int
    eligible_forecast_rows: int
    due_forecast_rows_without_eligible_realization: int
    status_counts: dict[str, int]
    evaluation_asof: datetime
    require_due_coverage: bool
    require_no_orphan_realizations: bool
    passed: bool
    eligibility: list[Eligibility]
    next_offset: int | None
    diagnostic_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def _key(row: Forecast | Realization) -> OutcomeKey:
    return row.security_id, row.target, row.horizon, row.decision_time


def execute(request: Input, context: OperationContext) -> Output:
    forecast_groups: dict[ForecastKey, list[int]] = defaultdict(list)
    realization_groups: dict[OutcomeKey, list[int]] = defaultdict(list)
    outcome_forecast_keys: set[OutcomeKey] = set()
    bad_forecasts: set[int] = set()
    bad_realizations: set[int] = set()
    diagnostics: list[Finding] = []
    diagnostic_count = 0

    def report(
        table: Literal["forecasts", "realizations"],
        code: str,
        indices: list[int],
        *,
        conflicts: bool = False,
        counterpart: int | None = None,
    ) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Finding(
                    table=table,
                    code=code,
                    row_indices=indices[: request.max_row_indices],
                    omitted_row_indices=max(0, len(indices) - request.max_row_indices),
                    conflicts=conflicts,
                    counterpart_row_index=counterpart,
                )
            )

    for index, forecast in enumerate(request.forecasts):
        key = _key(forecast)
        forecast_groups[(*key, forecast.model_version)].append(index)
        outcome_forecast_keys.add(key)
        issues = []
        if forecast.target_time <= forecast.decision_time:
            issues.append("target_not_after_decision")
        if forecast.max_source_available_time > forecast.decision_time:
            issues.append("source_after_decision")
        if forecast.forecast_available_time > forecast.decision_time:
            issues.append("forecast_after_decision")
        if forecast.max_source_available_time > forecast.forecast_available_time:
            issues.append("source_after_forecast")
        if forecast.decision_time > request.evaluation_asof:
            issues.append("decision_after_evaluation")
        if issues:
            bad_forecasts.add(index)
        for issue in issues:
            report("forecasts", issue, [index])

    for index, realization in enumerate(request.realizations):
        realization_groups[_key(realization)].append(index)
        issues = []
        if realization.target_time <= realization.decision_time:
            issues.append("target_not_after_decision")
        if realization.available_time < realization.target_time:
            issues.append("outcome_available_before_completion")
        if issues:
            bad_realizations.add(index)
        for issue in issues:
            report("realizations", issue, [index])

    duplicate_forecasts = conflicting_forecasts = 0
    duplicate_realizations = conflicting_realizations = 0
    for indices in forecast_groups.values():
        if len(indices) > 1:
            duplicate_forecasts += 1
            first = canonical_json(request.forecasts[indices[0]].model_dump(mode="json"))
            conflict = any(
                canonical_json(request.forecasts[index].model_dump(mode="json")) != first
                for index in indices[1:]
            )
            conflicting_forecasts += int(conflict)
            report("forecasts", "duplicate_key", indices, conflicts=conflict)
    orphan_keys = orphan_rows = 0
    for key, indices in realization_groups.items():
        if len(indices) > 1:
            duplicate_realizations += 1
            first = canonical_json(request.realizations[indices[0]].model_dump(mode="json"))
            conflict = any(
                canonical_json(request.realizations[index].model_dump(mode="json")) != first
                for index in indices[1:]
            )
            conflicting_realizations += int(conflict)
            report("realizations", "duplicate_key", indices, conflicts=conflict)
        if key not in outcome_forecast_keys:
            orphan_keys += 1
            orphan_rows += len(indices)
            report("realizations", "orphan_key", indices)

    counts: Counter[str] = Counter()
    eligibility: list[Eligibility] = []
    due_without_eligible = 0
    mismatches = 0
    for index, forecast in enumerate(request.forecasts):
        key = _key(forecast)
        candidates = realization_groups.get(key, [])
        realization_index = candidates[0] if len(candidates) == 1 else None
        matched_realization = (
            request.realizations[realization_index] if realization_index is not None else None
        )
        status: Status
        if index in bad_forecasts:
            status = "invalid_forecast"
        elif len(forecast_groups[(*key, forecast.model_version)]) > 1:
            status = "duplicate_forecast"
        elif len(candidates) > 1:
            status = "ambiguous_realization"
        elif realization_index in bad_realizations:
            status = "invalid_realization"
        elif (
            matched_realization is not None
            and matched_realization.target_time != forecast.target_time
        ):
            status = "target_time_mismatch"
            mismatches += 1
            report("forecasts", status, [index], counterpart=realization_index)
        elif forecast.target_time > request.evaluation_asof:
            status = "not_due"
        elif matched_realization is None:
            status = "missing_realization"
            report("forecasts", status, [index])
        elif matched_realization.available_time > request.evaluation_asof:
            status = "realization_not_available"
        else:
            status = "eligible"
        counts[status] += 1
        if forecast.target_time <= request.evaluation_asof and status != "eligible":
            due_without_eligible += 1
        if request.offset <= index < request.offset + request.limit:
            eligibility.append(
                Eligibility(
                    forecast_row_index=index,
                    status=status,
                    realization_row_index=realization_index,
                )
            )

    passed = not (
        bad_forecasts
        or bad_realizations
        or duplicate_forecasts
        or duplicate_realizations
        or mismatches
        or (request.require_due_coverage and due_without_eligible)
        or (request.require_no_orphan_realizations and orphan_keys)
    )
    next_offset = request.offset + len(eligibility)
    return Output(
        forecast_rows=len(request.forecasts),
        realization_rows=len(request.realizations),
        distinct_forecast_keys=len(forecast_groups),
        distinct_outcome_forecast_keys=len(outcome_forecast_keys),
        distinct_realization_keys=len(realization_groups),
        duplicate_forecast_groups=duplicate_forecasts,
        conflicting_forecast_groups=conflicting_forecasts,
        duplicate_realization_groups=duplicate_realizations,
        conflicting_realization_groups=conflicting_realizations,
        invalid_forecast_rows=len(bad_forecasts),
        invalid_realization_rows=len(bad_realizations),
        orphan_realization_keys=orphan_keys,
        orphan_realization_rows=orphan_rows,
        eligible_forecast_rows=counts["eligible"],
        due_forecast_rows_without_eligible_realization=due_without_eligible,
        status_counts=dict(sorted(counts.items())),
        evaluation_asof=request.evaluation_asof,
        require_due_coverage=request.require_due_coverage,
        require_no_orphan_realizations=request.require_no_orphan_realizations,
        passed=passed,
        eligibility=eligibility,
        next_offset=next_offset if next_offset < len(request.forecasts) else None,
        diagnostic_count=diagnostic_count,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_forecast_panel",
    kind="skill",
    description=(
        "Audit model-specific forecast keys, shared completed-outcome coverage, duplicate conflicts, "
        "and temporal eligibility at an explicit evaluation clock; horizon labels infer no calendar."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
