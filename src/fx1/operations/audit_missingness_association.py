"""Describe pairwise association between supplied missingness indicators.

Missing is absent, null, or optionally the exact empty string. Contingency
counts and Jaccard similarity are descriptive. Phi is Pearson correlation of
the binary missingness indicators, not evidence of causality or market signal.
Zero-denominator statistics are null, including phi for constant indicators.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Text = Annotated[str, Field(strict=True, max_length=1024)]
Scalar = Text | StrictBool | StrictInt | StrictFloat | None


class Input(InputModel):
    rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    columns: list[Name] = Field(min_length=2, max_length=23)
    empty_string_is_missing: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("columns must contain distinct names")
        if any(len(row) > 64 for row in self.rows):
            raise ValueError("each row may contain at most 64 fields")
        if sum(map(len, self.rows)) > 250_000:
            raise ValueError("rows may contain at most 250000 fields")
        return self


class ColumnIndicator(OutputModel):
    column: str
    absent_count: int
    null_count: int
    empty_string_count: int
    missing_count: int
    missing_rate: float | None


class Association(OutputModel):
    left_column: str
    right_column: str
    both_missing: int
    left_missing_only: int
    right_missing_only: int
    neither_missing: int
    joint_missing_rate: float | None
    jaccard: float | None
    phi: float | None
    left_missing_given_right_missing: float | None
    right_missing_given_left_missing: float | None


class Output(OutputModel):
    row_count: int
    column_count: int
    pair_count: int
    assessment: Literal["observed", "no_observations"]
    interpretation: Literal["descriptive_association_only"] = "descriptive_association_only"
    empty_string_is_missing: bool
    columns: list[ColumnIndicator]
    associations: list[Association]


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.rows)
    masks: dict[str, int] = {}
    profiles: list[ColumnIndicator] = []
    for name in request.columns:
        absent = nulls = empty_strings = mask = 0
        for index, row in enumerate(request.rows):
            absent_here = name not in row
            value = row.get(name)
            null_here = not absent_here and value is None
            empty_here = isinstance(value, str) and value == ""
            absent += int(absent_here)
            nulls += int(null_here)
            empty_strings += int(empty_here)
            if absent_here or null_here or (request.empty_string_is_missing and empty_here):
                mask |= 1 << index
        masks[name] = mask
        missing = mask.bit_count()
        profiles.append(
            ColumnIndicator(
                column=name,
                absent_count=absent,
                null_count=nulls,
                empty_string_count=empty_strings,
                missing_count=missing,
                missing_rate=missing / count if count else None,
            )
        )
    associations: list[Association] = []
    for left, right in combinations(request.columns, 2):
        left_count = masks[left].bit_count()
        right_count = masks[right].bit_count()
        both = (masks[left] & masks[right]).bit_count()
        left_only = left_count - both
        right_only = right_count - both
        neither = count - both - left_only - right_only
        union = both + left_only + right_only
        denominator_squared = (
            left_count * (count - left_count) * right_count * (count - right_count)
        )
        phi = (
            (both * neither - left_only * right_only) / sqrt(denominator_squared)
            if denominator_squared
            else None
        )
        associations.append(
            Association(
                left_column=left,
                right_column=right,
                both_missing=both,
                left_missing_only=left_only,
                right_missing_only=right_only,
                neither_missing=neither,
                joint_missing_rate=both / count if count else None,
                jaccard=both / union if union else None,
                phi=min(1.0, max(-1.0, phi)) if phi is not None else None,
                left_missing_given_right_missing=both / right_count if right_count else None,
                right_missing_given_left_missing=both / left_count if left_count else None,
            )
        )
    return Output(
        row_count=count,
        column_count=len(request.columns),
        pair_count=len(associations),
        assessment="observed" if count else "no_observations",
        empty_string_is_missing=request.empty_string_is_missing,
        columns=profiles,
        associations=associations,
    )


OPERATION = Operation(
    id="skills.audit_missingness_association",
    kind="skill",
    description=(
        "Compute exact pairwise missingness contingency counts, Jaccard similarity, phi, "
        "and conditional missingness rates across up to 23 selected columns. Distinguishes "
        "absent/null/optional empty strings and reports undefined statistics as null."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
