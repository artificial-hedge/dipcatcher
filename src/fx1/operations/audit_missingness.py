"""Measure absent fields, nulls, and exact empty strings without imputing data.

Runs follow supplied row order, not inferred time order. Completeness covers
only selected columns, or the union of observed columns if none are supplied.
Whitespace-only strings are values; only the exact empty string is optional
missing data. Empty inputs and tables without columns have no passing result.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Text = Annotated[str, Field(strict=True, max_length=4096)]
Scalar = Text | StrictBool | StrictInt | StrictFloat | None


class Input(InputModel):
    rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    columns: list[Name] | None = Field(default=None, min_length=1, max_length=256)
    empty_string_is_missing: bool = Field(default=False, strict=True)
    max_incomplete_examples: int = Field(default=25, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if any(len(row) > 64 for row in self.rows):
            raise ValueError("each row may contain at most 64 fields")
        if sum(map(len, self.rows)) > 250_000:
            raise ValueError("rows may contain at most 250000 total fields")
        names = {name for row in self.rows for name in row}
        if len(names) > 256:
            raise ValueError("rows may contain at most 256 distinct fields")
        if self.columns is not None and len(self.columns) != len(set(self.columns)):
            raise ValueError("columns must contain distinct field names")
        return self


class ColumnMissingness(OutputModel):
    column: str
    absent_count: int
    null_count: int
    empty_string_count: int
    effective_missing_count: int
    observed_value_count: int
    absent_rate: float | None
    null_rate: float | None
    empty_string_rate: float | None
    effective_missing_rate: float | None
    longest_missing_run: int
    leading_missing_run: int
    trailing_missing_run: int


class Output(OutputModel):
    row_count: int
    column_count: int
    empty_string_is_missing: bool
    assessment: Literal["observed", "no_observations", "no_columns"]
    no_missing_values: bool | None
    complete_row_count: int | None
    incomplete_row_count: int | None
    complete_row_rate: float | None
    effective_missing_cells: int
    effective_missing_cell_rate: float | None
    columns: list[ColumnMissingness]
    incomplete_row_indices: list[int]
    omitted_incomplete_examples: int


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.rows)
    names = (
        request.columns
        if request.columns is not None
        else sorted({name for row in request.rows for name in row})
    )
    # One marker per row avoids keeping a row-by-column boolean matrix.
    incomplete = [False] * count
    profiles: list[ColumnMissingness] = []
    missing_cells = 0
    for name in names:
        absent = nulls = empty_strings = missing = run = longest = leading = 0
        leading_open = True
        for index, row in enumerate(request.rows):
            absent_here = name not in row
            value = row.get(name)
            null_here = not absent_here and value is None
            empty_here = isinstance(value, str) and value == ""
            absent += int(absent_here)
            nulls += int(null_here)
            empty_strings += int(empty_here)
            missing_here = (
                absent_here or null_here or (request.empty_string_is_missing and empty_here)
            )
            if missing_here:
                incomplete[index] = True
                missing += 1
                run += 1
                longest = max(longest, run)
                if leading_open:
                    leading += 1
            else:
                run = 0
                leading_open = False
        missing_cells += missing
        profiles.append(
            ColumnMissingness(
                column=name,
                absent_count=absent,
                null_count=nulls,
                empty_string_count=empty_strings,
                effective_missing_count=missing,
                observed_value_count=count - missing,
                absent_rate=absent / count if count else None,
                null_rate=nulls / count if count else None,
                empty_string_rate=empty_strings / count if count else None,
                effective_missing_rate=missing / count if count else None,
                longest_missing_run=longest,
                leading_missing_run=leading,
                trailing_missing_run=run,
            )
        )
    incomplete_count = sum(incomplete)
    examples = [index for index, marked in enumerate(incomplete) if marked][
        : request.max_incomplete_examples
    ]
    observed = count > 0 and bool(names)
    return Output(
        row_count=count,
        column_count=len(names),
        empty_string_is_missing=request.empty_string_is_missing,
        assessment="observed" if observed else ("no_observations" if not count else "no_columns"),
        no_missing_values=missing_cells == 0 if observed else None,
        complete_row_count=count - incomplete_count if names else None,
        incomplete_row_count=incomplete_count if names else None,
        complete_row_rate=(count - incomplete_count) / count if observed else None,
        effective_missing_cells=missing_cells,
        effective_missing_cell_rate=missing_cells / (count * len(names)) if observed else None,
        columns=profiles,
        incomplete_row_indices=examples,
        omitted_incomplete_examples=incomplete_count - len(examples),
    )


OPERATION = Operation(
    id="skills.audit_missingness",
    kind="skill",
    description=(
        "Measure per-column absent fields, explicit nulls, optional exact empty strings, "
        "missing runs in input order, and row completeness. Empty observations or absent "
        "column scope produce an unassessed result, with bounded incomplete-row examples."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
