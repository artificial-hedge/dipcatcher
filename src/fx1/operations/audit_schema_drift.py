"""Compare observed scalar schemas of two supplied tables, without type coercion.

Optional means absent from at least one supplied row; nullable means an
explicit null was observed. These describe the samples, not the full source
schema. Null-only fields carry an empty non-null type set. Comparisons require
at least one row in each table; an empty table cannot establish schema drift.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Text = Annotated[str, Field(strict=True, max_length=4096)]
Scalar = Text | StrictBool | StrictInt | StrictFloat | None
ChangeCode = Literal[
    "added_field",
    "removed_field",
    "non_null_types_changed",
    "optionality_changed",
    "nullability_changed",
]


class Input(InputModel):
    reference_rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    candidate_rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    numeric_policy: Literal["distinct", "compatible"] = "distinct"
    max_changes: int = Field(default=100, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        names: set[str] = set()
        cells = 0
        for table in (self.reference_rows, self.candidate_rows):
            for row in table:
                if len(row) > 64:
                    raise ValueError("each row may contain at most 64 fields")
                cells += len(row)
                names.update(row)
        if cells > 250_000:
            raise ValueError("both tables combined may contain at most 250000 fields")
        if len(names) > 256:
            raise ValueError("both tables combined may contain at most 256 distinct fields")
        return self


class FieldProfile(OutputModel):
    name: str
    present_rows: int
    absent_rows: int
    null_rows: int
    non_null_types: list[str]
    optional: bool
    nullable: bool


class FieldChange(OutputModel):
    name: str
    changes: list[ChangeCode]
    reference: FieldProfile | None
    candidate: FieldProfile | None


class Output(OutputModel):
    reference_row_count: int
    candidate_row_count: int
    numeric_policy: str
    comparison_basis: Literal["observed_rows_only"] = "observed_rows_only"
    assessment: Literal["compared", "insufficient_observations"]
    schema_equal: bool | None
    reference_fields: list[FieldProfile]
    candidate_fields: list[FieldProfile]
    changed_field_count: int
    change_counts: dict[str, int]
    changes: list[FieldChange]
    omitted_changed_fields: int


def _profile(rows: list[dict[str, Scalar]], *, compatible_numbers: bool) -> dict[str, FieldProfile]:
    present: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    types: dict[str, set[str]] = {}
    for row in rows:
        for name, value in row.items():
            present[name] += 1
            kinds = types.setdefault(name, set())
            if value is None:
                nulls[name] += 1
            elif isinstance(value, bool):
                kinds.add("boolean")
            elif isinstance(value, int):
                kinds.add("number" if compatible_numbers else "integer")
            elif isinstance(value, float):
                kinds.add("number" if compatible_numbers else "float")
            else:
                kinds.add("string")
    return {
        name: FieldProfile(
            name=name,
            present_rows=present[name],
            absent_rows=len(rows) - present[name],
            null_rows=nulls[name],
            non_null_types=sorted(types[name]),
            optional=present[name] < len(rows),
            nullable=nulls[name] > 0,
        )
        for name in sorted(present)
    }


def execute(request: Input, context: OperationContext) -> Output:
    compatible = request.numeric_policy == "compatible"
    reference = _profile(request.reference_rows, compatible_numbers=compatible)
    candidate = _profile(request.candidate_rows, compatible_numbers=compatible)
    comparable = bool(request.reference_rows) and bool(request.candidate_rows)
    changes: list[FieldChange] = []
    changed_count = 0
    counts: Counter[str] = Counter()
    if comparable:
        for name in sorted(reference.keys() | candidate.keys()):
            before, after = reference.get(name), candidate.get(name)
            codes: list[ChangeCode] = []
            if before is None:
                codes.append("added_field")
            elif after is None:
                codes.append("removed_field")
            else:
                if before.non_null_types != after.non_null_types:
                    codes.append("non_null_types_changed")
                if before.optional != after.optional:
                    codes.append("optionality_changed")
                if before.nullable != after.nullable:
                    codes.append("nullability_changed")
            if codes:
                changed_count += 1
                counts.update(codes)
                if len(changes) < request.max_changes:
                    changes.append(
                        FieldChange(name=name, changes=codes, reference=before, candidate=after)
                    )
    return Output(
        reference_row_count=len(request.reference_rows),
        candidate_row_count=len(request.candidate_rows),
        numeric_policy=request.numeric_policy,
        assessment="compared" if comparable else "insufficient_observations",
        schema_equal=changed_count == 0 if comparable else None,
        reference_fields=list(reference.values()),
        candidate_fields=list(candidate.values()),
        changed_field_count=changed_count,
        change_counts=dict(sorted(counts.items())),
        changes=changes,
        omitted_changed_fields=changed_count - len(changes),
    )


OPERATION = Operation(
    id="skills.audit_schema_drift",
    kind="skill",
    description=(
        "Compare observed scalar field types, optionality and nullability between two tables. "
        "Reports added/removed fields and bounded change details; bool stays distinct from "
        "numbers, with explicit distinct/compatible integer-float policy and empty-data status."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
