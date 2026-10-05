"""Audit declared price bases, factor provenance and exact transformation values.

Prices distinguish raw, split_adjusted, total_return and undeclared bases, plus
an explicit basis_version identifying the caller's normalization convention.
A derivation states target = source_record * factor.multiplier. Factors explicitly
declare both endpoint bases/versions, security, currency, availability and a
half-open event-validity interval. Source/target event, security, field and currency
must agree. Currency conversions and vendor adjustment formulas are not inferred.

Vendor records without a derivation can pass as declared_only; this never verifies
their economic adjustment basis. require_derived_adjusted can require a declared
derivation for adjusted rows. A numerically matching equation independently checks
only the supplied positive values; provenance/availability failures still make the
record invalid. Even a reconciled declaration does not verify vendor factors or
the truth/completeness of a corporate-action history.

Duplicate IDs are ambiguous, including identical rows. A topological traversal
propagates known dependency availability and invalid dependencies; unresolved
references and cycles cannot certify a complete availability chain. Every supplied
record and factor is audited, including unused factors and rows outside the page.
Clocks require explicit timezone-aware ISO strings/datetimes, normalized to UTC.
maximum_source_available_time includes the record itself and all declared ancestor
records/factors; it is null when that full dependency maximum cannot be resolved.

Bounds: 5000 records and 5000 factors; one parent/factor per derivation, hence linear
dependency work. Exact decimal inputs have 38 integral and 18 fractional digits.
Comparison uses absolute + relative*abs(source*factor) tolerance. At most 200
diagnostics and 200 record summaries are returned. No price value is rewritten.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Currency = Annotated[str, Field(strict=True, min_length=3, max_length=8, pattern=r"\A[A-Z]{3,8}\z")]
DecimalText = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=58,
        pattern=r"\A-?(?:0|[1-9][0-9]{0,37})(?:\.[0-9]{1,18})?\z",
    ),
]
KnownBasis = Literal["raw", "split_adjusted", "total_return"]
Basis = Literal["raw", "split_adjusted", "total_return", "undeclared"]


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


class Derivation(InputModel):
    source_record_id: Name
    factor_id: Name
    rule: Literal["multiply"] = "multiply"


class PriceRecord(InputModel):
    record_id: Name
    security_id: Name
    field: Name
    currency: Currency
    basis: Basis
    basis_version: Name
    value: DecimalText
    event_time: AwareDatetime
    available_time: AwareDatetime
    source: Name
    revision_id: Name
    derivation: Derivation | None = None

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class FactorRecord(InputModel):
    factor_id: Name
    security_id: Name
    currency: Currency
    from_basis: KnownBasis
    from_basis_version: Name
    to_basis: KnownBasis
    to_basis_version: Name
    multiplier: DecimalText
    specification_id: Name
    available_time: AwareDatetime
    valid_from: AwareDatetime | None
    valid_to: AwareDatetime | None
    source: Name
    revision_id: Name

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("valid_from", "valid_to", mode="before")
    @classmethod
    def optional_clock(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)


class Input(InputModel):
    records: list[PriceRecord] = Field(min_length=1, max_length=5000)
    factors: list[FactorRecord] = Field(max_length=5000)
    decision_time: AwareDatetime
    require_derived_adjusted: bool = Field(default=False, strict=True)
    absolute_tolerance: DecimalText = "0"
    relative_tolerance: DecimalText = "0"
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=5000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def tolerances(self) -> Self:
        if Fraction(self.absolute_tolerance) < 0 or Fraction(self.relative_tolerance) < 0:
            raise ValueError("price tolerances must be nonnegative")
        return self


class ExactValue(OutputModel):
    numerator: str
    denominator: str


class Finding(OutputModel):
    code: str
    record_row_index: int | None = None
    source_record_row_index: int | None = None
    factor_row_index: int | None = None


class RecordResult(OutputModel):
    record_row_index: int
    record_id: str
    basis: Basis
    basis_version: str
    currency: str
    status: Literal["declared_only", "reconciled_declaration", "invalid"]
    source_record_row_index: int | None
    factor_row_index: int | None
    value_equation_matches: bool | None
    expected_value: ExactValue | None
    absolute_difference: ExactValue | None
    allowed_difference: ExactValue | None
    availability_chain_complete: bool
    maximum_source_available_time: datetime | None
    passed: bool


class Output(OutputModel):
    passed: bool
    decision_time: datetime
    require_derived_adjusted: bool
    record_count: int
    factor_count: int
    referenced_factor_count: int
    declared_only_count: int
    reconciled_declaration_count: int
    invalid_record_count: int
    invalid_factor_count: int
    checked_equation_count: int
    matching_equation_count: int
    complete_availability_chain_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    records: list[RecordResult]
    offset: int
    has_more: bool
    vendor_price_basis_verified: Literal[False] = False
    factor_economic_truth_verified: Literal[False] = False
    corporate_action_completeness_verified: Literal[False] = False


def _exact(value: Fraction | None) -> ExactValue | None:
    if value is None:
        return None
    return ExactValue(numerator=str(value.numerator), denominator=str(value.denominator))


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid_records: set[int] = set()
    invalid_factors: set[int] = set()

    def report(
        code: str, record: int | None = None, source: int | None = None, factor: int | None = None
    ) -> None:
        issues[code] += 1
        if record is not None:
            invalid_records.add(record)
        elif factor is not None:
            invalid_factors.add(factor)
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    record_row_index=record,
                    source_record_row_index=source,
                    factor_row_index=factor,
                )
            )

    record_rows: dict[str, list[int]] = defaultdict(list)
    factor_rows: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(request.records):
        record_rows[record.record_id].append(index)
    for index, factor in enumerate(request.factors):
        factor_rows[factor.factor_id].append(index)
    unique_records = {name: rows[0] for name, rows in record_rows.items() if len(rows) == 1}
    unique_factors = {name: rows[0] for name, rows in factor_rows.items() if len(rows) == 1}
    prices = [Fraction(record.value) for record in request.records]
    multipliers = [Fraction(factor.multiplier) for factor in request.factors]
    for index, factor in enumerate(request.factors):
        if len(factor_rows[factor.factor_id]) != 1:
            report("ambiguous_factor_id", factor=index)
        if multipliers[index] <= 0:
            report("nonpositive_factor", factor=index)
        if (
            factor.valid_from is not None
            and factor.valid_to is not None
            and factor.valid_from >= factor.valid_to
        ):
            report("empty_or_inverted_factor_validity", factor=index)
        if factor.available_time > request.decision_time:
            report("factor_after_decision", factor=index)

    sources: list[int | None] = [None] * len(request.records)
    factors: list[int | None] = [None] * len(request.records)
    children: list[list[int]] = [[] for _ in request.records]
    indegree = [0] * len(request.records)
    complete = [True] * len(request.records)
    maximum = [record.available_time for record in request.records]
    referenced_factors: set[int] = set()
    for index, record in enumerate(request.records):
        if len(record_rows[record.record_id]) != 1:
            report("ambiguous_record_id", record=index)
        if prices[index] <= 0:
            report("nonpositive_price", record=index)
        if record.basis == "undeclared":
            report("undeclared_price_basis", record=index)
        if record.available_time > request.decision_time:
            report("record_after_decision", record=index)
        derivation = record.derivation
        if derivation is None:
            if request.require_derived_adjusted and record.basis in (
                "split_adjusted",
                "total_return",
            ):
                report("adjusted_derivation_required", record=index)
            continue
        source_position = unique_records.get(derivation.source_record_id)
        factor_position = unique_factors.get(derivation.factor_id)
        sources[index], factors[index] = source_position, factor_position
        if source_position is None:
            complete[index] = False
            report("unresolved_source_record", record=index)
        else:
            children[source_position].append(index)
            indegree[index] = 1
        if factor_position is None:
            complete[index] = False
            report("unresolved_factor", record=index, source=source_position)
        else:
            referenced_factors.add(factor_position)
            maximum[index] = max(maximum[index], request.factors[factor_position].available_time)

    expected: list[Fraction | None] = [None] * len(request.records)
    differences: list[Fraction | None] = [None] * len(request.records)
    allowed: list[Fraction | None] = [None] * len(request.records)
    matches: list[bool | None] = [None] * len(request.records)
    absolute, relative = Fraction(request.absolute_tolerance), Fraction(request.relative_tolerance)
    pending = deque(index for index, degree in enumerate(indegree) if degree == 0)
    processed: set[int] = set()
    while pending:
        index = pending.popleft()
        processed.add(index)
        record = request.records[index]
        source_index, factor_index = sources[index], factors[index]
        if source_index is not None:
            source = request.records[source_index]
            maximum[index] = max(maximum[index], maximum[source_index])
            complete[index] = complete[index] and complete[source_index]
            if source_index in invalid_records:
                report("invalid_source_dependency", record=index, source=source_index)
            if (record.security_id, record.event_time, record.field, record.currency) != (
                source.security_id,
                source.event_time,
                source.field,
                source.currency,
            ):
                report("source_price_identity_mismatch", record=index, source=source_index)
            if source.available_time > record.available_time:
                report("record_precedes_source_availability", record=index, source=source_index)
        if factor_index is not None:
            factor = request.factors[factor_index]
            if factor_index in invalid_factors:
                report("invalid_factor_dependency", record=index, factor=factor_index)
            if (factor.security_id, factor.currency, factor.to_basis, factor.to_basis_version) != (
                record.security_id,
                record.currency,
                record.basis,
                record.basis_version,
            ):
                report("factor_target_metadata_mismatch", record=index, factor=factor_index)
            if (factor.valid_from is not None and record.event_time < factor.valid_from) or (
                factor.valid_to is not None and record.event_time >= factor.valid_to
            ):
                report("factor_outside_event_validity", record=index, factor=factor_index)
            if factor.available_time > record.available_time:
                report("record_precedes_factor_availability", record=index, factor=factor_index)
            if source_index is not None:
                source = request.records[source_index]
                if (factor.from_basis, factor.from_basis_version) != (
                    source.basis,
                    source.basis_version,
                ):
                    report(
                        "factor_source_basis_mismatch",
                        record=index,
                        source=source_index,
                        factor=factor_index,
                    )
                if prices[index] > 0 and prices[source_index] > 0 and multipliers[factor_index] > 0:
                    computed = prices[source_index] * multipliers[factor_index]
                    difference = abs(prices[index] - computed)
                    tolerance = absolute + relative * abs(computed)
                    expected[index], differences[index], allowed[index] = (
                        computed,
                        difference,
                        tolerance,
                    )
                    matches[index] = difference <= tolerance
                    if not matches[index]:
                        report(
                            "price_equation_mismatch",
                            record=index,
                            source=source_index,
                            factor=factor_index,
                        )
        if record.derivation is not None and maximum[index] > request.decision_time:
            report(
                "dependency_chain_after_decision",
                record=index,
                source=source_index,
                factor=factor_index,
            )
        for child in children[index]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    for index in range(len(request.records)):
        if index not in processed:
            complete[index] = False
            report(
                "cyclic_or_cycle_dependent_derivation",
                record=index,
                source=sources[index],
                factor=factors[index],
            )

    statuses: list[Literal["declared_only", "reconciled_declaration", "invalid"]] = []
    for index, record in enumerate(request.records):
        statuses.append(
            "invalid"
            if index in invalid_records
            else "declared_only"
            if record.derivation is None
            else "reconciled_declaration"
        )
    results = [
        RecordResult(
            record_row_index=index,
            record_id=request.records[index].record_id,
            basis=request.records[index].basis,
            basis_version=request.records[index].basis_version,
            currency=request.records[index].currency,
            status=statuses[index],
            source_record_row_index=sources[index],
            factor_row_index=factors[index],
            value_equation_matches=matches[index],
            expected_value=_exact(expected[index]),
            absolute_difference=_exact(differences[index]),
            allowed_difference=_exact(allowed[index]),
            availability_chain_complete=complete[index],
            maximum_source_available_time=maximum[index] if complete[index] else None,
            passed=index not in invalid_records,
        )
        for index in range(
            request.offset, min(len(request.records), request.offset + request.limit)
        )
    ]
    return Output(
        passed=not issues,
        decision_time=request.decision_time,
        require_derived_adjusted=request.require_derived_adjusted,
        record_count=len(request.records),
        factor_count=len(request.factors),
        referenced_factor_count=len(referenced_factors),
        declared_only_count=statuses.count("declared_only"),
        reconciled_declaration_count=statuses.count("reconciled_declaration"),
        invalid_record_count=len(invalid_records),
        invalid_factor_count=len(invalid_factors),
        checked_equation_count=sum(match is not None for match in matches),
        matching_equation_count=sum(match is True for match in matches),
        complete_availability_chain_count=sum(complete),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        records=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.records),
    )


OPERATION = Operation(
    id="skills.audit_price_basis",
    kind="skill",
    description="Audit declared price bases/currencies and factor lineage, exact price transformations and complete dependency availability without inferring vendor adjustments.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
