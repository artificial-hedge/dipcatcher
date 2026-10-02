"""Audit supplied OHLCV bars without repairing or discarding bad observations."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

# CSV null/nonfinite/nonnumeric cells can be submitted as raw strings. JSON
# numeric values themselves must be finite so fingerprints use valid JSON.
AuditCell = (
    Annotated[float, Field(strict=True)] | Annotated[str, Field(strict=True, max_length=256)] | None
)
FailureCode = Literal[
    "missing_value",
    "nonfinite_value",
    "nonnumeric_value",
    "nonpositive_price",
    "negative_volume",
    "inverted_range",
    "outside_envelope",
]


class Bar(InputModel):
    open: AuditCell
    high: AuditCell
    low: AuditCell
    close: AuditCell
    volume: AuditCell


class Input(InputModel):
    bars: list[Bar] = Field(min_length=1, max_length=10_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)


class Finding(OutputModel):
    row_index: int
    field: str
    code: FailureCode


class Output(OutputModel):
    row_count: int
    valid_rows: int
    invalid_rows: int
    passed: bool
    violation_count: int
    violation_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    counts: Counter[str] = Counter()
    diagnostics: list[Finding] = []
    invalid_rows = 0

    for row_index, bar in enumerate(request.bars):
        failures: list[Finding] = []
        usable: dict[str, float] = {}
        for field in ("open", "high", "low", "close", "volume"):
            value = getattr(bar, field)
            code: FailureCode | None = None
            if value is None:
                code = "missing_value"
            elif isinstance(value, str):
                code = (
                    "nonfinite_value"
                    if value.strip().lower()
                    in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
                    else "nonnumeric_value"
                )
            elif not isfinite(value):
                code = "nonfinite_value"
            elif field == "volume" and value < 0:
                code = "negative_volume"
            elif field != "volume" and value <= 0:
                code = "nonpositive_price"
            else:
                usable[field] = value
            if code is not None:
                failures.append(Finding(row_index=row_index, field=field, code=code))

        if "low" in usable and "high" in usable:
            low, high = usable["low"], usable["high"]
            if low > high:
                failures.append(Finding(row_index=row_index, field="high", code="inverted_range"))
            else:
                for field in ("open", "close"):
                    if field in usable and not low <= usable[field] <= high:
                        failures.append(
                            Finding(row_index=row_index, field=field, code="outside_envelope")
                        )

        if failures:
            invalid_rows += 1
        for failure in failures:
            counts[failure.code] += 1
            if len(diagnostics) < request.max_diagnostics:
                diagnostics.append(failure)

    violation_count = sum(counts.values())
    return Output(
        row_count=len(request.bars),
        valid_rows=len(request.bars) - invalid_rows,
        invalid_rows=invalid_rows,
        passed=invalid_rows == 0,
        violation_count=violation_count,
        violation_counts=dict(sorted(counts.items())),
        diagnostics=diagnostics,
        omitted_diagnostics=violation_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_bar_integrity",
    kind="skill",
    description=(
        "Audit supplied OHLCV rows for missing/nonfinite/nonpositive prices, negative "
        "volume, inverted ranges, and candle-envelope violations; never repair data."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
