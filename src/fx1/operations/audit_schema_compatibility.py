"""Directional compatibility of two caller-declared flat logical schemas.

Compatibility means every row admitted by the writer declaration is accepted
by the reader declaration under the supported value-domain promotions below.
It does not inspect rows or perform conversions. Field names are exact and
case-sensitive. Required means present; nullable concerns present null values.
Defaults, aliases, nested fields, unions and inferred conversions are unsupported.
The writer schema is closed: undeclared fields cannot occur. Reader treatment
of fields declared only by the writer follows extra_writer_field_policy.

Integer promotions require full range containment. Integers may promote to
finite binary32/binary64 only when their entire range fits 24/53 significant
binary digits. Binary32 may promote to binary64. Decimal(p,s) denotes signed
base-ten values with at most p digits and s fractional digits, 0<=s<=p<=76;
decimal promotions preserve both fractional and integral capacity. Integral
decimal/integer conversions are allowed only with full domain containment.

Timestamps are signed 64-bit ticks: both tick unit and UTC/naive semantics must
match exactly. Finer units can overflow and coarser units can truncate, so no
timestamp rescaling is assumed. Physical value-unit labels must match exactly,
including absence; even familiar unit conversions are not inferred.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Kind = Literal[
    "boolean",
    "utf8",
    "binary",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "decimal",
    "date",
    "timestamp",
]
_INTEGER_RANGES = {
    **{f"int{bits}": (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) for bits in (8, 16, 32, 64)},
    **{f"uint{bits}": (0, 2**bits - 1) for bits in (8, 16, 32, 64)},
}
_FLOAT_DIGITS = {"float32": 24, "float64": 53}


class LogicalType(InputModel):
    kind: Kind
    precision: int | None = Field(default=None, strict=True, ge=1, le=76)
    scale: int | None = Field(default=None, strict=True, ge=0, le=76)
    time_unit: Literal["s", "ms", "us", "ns"] | None = None
    timezone: Literal["UTC", "naive"] | None = None

    @model_validator(mode="after")
    def applicable_parameters(self) -> Self:
        if self.kind == "decimal":
            if self.precision is None or self.scale is None or self.scale > self.precision:
                raise ValueError("decimal requires 0 <= scale <= precision <= 76")
        elif self.precision is not None or self.scale is not None:
            raise ValueError("precision and scale apply only to decimal")
        if self.kind == "timestamp":
            if self.time_unit is None or self.timezone is None:
                raise ValueError("timestamp requires explicit time_unit and timezone semantics")
        elif self.time_unit is not None or self.timezone is not None:
            raise ValueError("time_unit and timezone apply only to timestamp")
        return self


class SchemaField(InputModel):
    name: Name
    type: LogicalType
    required: bool = Field(strict=True)
    nullable: bool = Field(strict=True)
    value_unit: Name | None = None


class Input(InputModel):
    writer_fields: list[SchemaField] = Field(max_length=512)
    reader_fields: list[SchemaField] = Field(max_length=512)
    extra_writer_field_policy: Literal["ignore", "reject"] = "reject"
    offset: int = Field(default=0, strict=True, ge=0, le=1_024)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def distinct_names(self) -> Self:
        for fields in (self.writer_fields, self.reader_fields):
            if len({field.name for field in fields}) != len(fields):
                raise ValueError("field names must be unique within each schema")
        return self


class Diagnostic(OutputModel):
    field_name: str
    writer_field_index: int | None
    reader_field_index: int | None
    code: str


class FieldResult(OutputModel):
    field_name: str
    writer_field_index: int | None
    reader_field_index: int | None
    compatible: bool
    action: (
        Literal["identity", "lossless_promotion", "optional_reader_absent", "ignore_writer_field"]
        | None
    )
    incompatibility_codes: list[str]


class Output(OutputModel):
    compatible: bool
    direction: Literal["all_writer_rows_must_be_accepted_by_reader"] = (
        "all_writer_rows_must_be_accepted_by_reader"
    )
    data_values_inspected: Literal[False] = False
    conversions_performed: Literal[False] = False
    writer_field_count: int
    reader_field_count: int
    field_result_count: int
    incompatible_field_count: int
    action_counts: dict[str, int]
    extra_writer_field_policy: str
    issue_counts: dict[str, int]
    diagnostic_count: int
    omitted_diagnostics: int
    offset: int
    next_offset: int | None
    fields: list[FieldResult]
    diagnostics: list[Diagnostic]


def _type_compatibility(writer: LogicalType, reader: LogicalType) -> tuple[bool, str]:
    first, second = writer.kind, reader.kind
    if first == second:
        if first == "decimal":
            assert writer.precision is not None and writer.scale is not None
            assert reader.precision is not None and reader.scale is not None
            compatible = (
                reader.scale >= writer.scale
                and reader.precision - reader.scale >= writer.precision - writer.scale
            )
            return compatible, "decimal_capacity_loss"
        if first == "timestamp":
            return (
                writer.time_unit == reader.time_unit and writer.timezone == reader.timezone,
                "timestamp_unit_or_timezone_changed",
            )
        return True, ""
    if first in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[first]
        if second in _INTEGER_RANGES:
            target_low, target_high = _INTEGER_RANGES[second]
            return target_low <= low and high <= target_high, "integer_range_loss"
        if second in _FLOAT_DIGITS:
            return max(abs(low), abs(high)) <= 2 ** _FLOAT_DIGITS[
                second
            ], "integer_float_precision_loss"
        if second == "decimal":
            assert reader.precision is not None and reader.scale is not None
            maximum = 10 ** (reader.precision - reader.scale) - 1
            return max(abs(low), abs(high)) <= maximum, "integer_decimal_capacity_loss"
    if first == "float32" and second == "float64":
        return True, ""
    if first == "decimal" and writer.scale == 0:
        assert writer.precision is not None
        maximum = 10**writer.precision - 1
        if second in _INTEGER_RANGES:
            low, high = _INTEGER_RANGES[second]
            return low <= -maximum and maximum <= high, "decimal_integer_range_loss"
        if second in _FLOAT_DIGITS:
            return maximum <= 2 ** _FLOAT_DIGITS[second], "decimal_float_precision_loss"
    return False, "unsupported_type_change"


def execute(request: Input, context: OperationContext) -> Output:
    writer = {field.name: (index, field) for index, field in enumerate(request.writer_fields)}
    reader = {field.name: (index, field) for index, field in enumerate(request.reader_fields)}
    names = sorted(set(writer) | set(reader))
    results: list[FieldResult] = []
    diagnostics: list[Diagnostic] = []
    issues: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    incompatible_count = 0
    for position, name in enumerate(names):
        source, target = writer.get(name), reader.get(name)
        source_index = source[0] if source else None
        target_index = target[0] if target else None
        codes: list[str] = []
        action: (
            Literal[
                "identity", "lossless_promotion", "optional_reader_absent", "ignore_writer_field"
            ]
            | None
        ) = None
        if source is None and target is not None:
            if target[1].required:
                codes.append("required_reader_field_absent")
            else:
                action = "optional_reader_absent"
        elif target is None:
            if request.extra_writer_field_policy == "reject":
                codes.append("extra_writer_field_rejected")
            else:
                action = "ignore_writer_field"
        elif source is not None:
            first, second = source[1], target[1]
            if second.required and not first.required:
                codes.append("writer_does_not_guarantee_presence")
            if first.nullable and not second.nullable:
                codes.append("reader_rejects_writer_nulls")
            if first.value_unit != second.value_unit:
                codes.append("physical_value_unit_changed")
            valid_type, code = _type_compatibility(first.type, second.type)
            if not valid_type:
                codes.append(code)
            if not codes:
                action = "identity" if first.type == second.type else "lossless_promotion"
        incompatible_count += bool(codes)
        if action is not None:
            actions[action] += 1
        for code in codes:
            issues[code] += 1
            if len(diagnostics) < request.max_diagnostics:
                diagnostics.append(
                    Diagnostic(
                        field_name=name,
                        writer_field_index=source_index,
                        reader_field_index=target_index,
                        code=code,
                    )
                )
        if request.offset <= position < request.offset + request.limit:
            results.append(
                FieldResult(
                    field_name=name,
                    writer_field_index=source_index,
                    reader_field_index=target_index,
                    compatible=not codes,
                    action=action,
                    incompatibility_codes=codes,
                )
            )
    diagnostic_count = sum(issues.values())
    end = min(len(names), request.offset + request.limit)
    return Output(
        compatible=incompatible_count == 0,
        writer_field_count=len(writer),
        reader_field_count=len(reader),
        field_result_count=len(names),
        incompatible_field_count=incompatible_count,
        action_counts=dict(sorted(actions.items())),
        extra_writer_field_policy=request.extra_writer_field_policy,
        issue_counts=dict(sorted(issues.items())),
        diagnostic_count=diagnostic_count,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
        offset=request.offset,
        next_offset=end if end < len(names) else None,
        fields=results,
        diagnostics=diagnostics,
    )


OPERATION = Operation(
    id="skills.audit_schema_compatibility",
    kind="skill",
    description=(
        "Audit directional flat writer-to-reader schema compatibility with presence/null "
        "rules, full-domain lossless promotions, and explicit decimal/timestamp/unit semantics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
