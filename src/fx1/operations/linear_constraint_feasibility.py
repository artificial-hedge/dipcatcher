"""Exact bounded Fourier-Motzkin feasibility with independently checked evidence.

Supplied rows mean A x <= b over unrestricted real variables, with coefficients
equal to the exact binary64 inputs. Strict inequalities and integer variables
are unsupported. Encode equality as two opposite nonstrict rows. Variables are
eliminated last to first; each upper/lower pair yields its nonnegative linear
combination. Positive row normalization and identical-left-side deduplication
retain the tightest right side; equal rows retain the first encountered lineage.

Every derived row carries nonnegative multipliers of original source rows.
A negative constant row yields a checked Farkas certificate lambda A = 0,
lambda b = -1, lambda >= 0. Otherwise reverse reconstruction chooses each
coordinate nearest zero in its closed feasible interval; all original slacks
are checked exactly. This deterministic witness need not minimize a norm.
Exact rational strings are authoritative. Binary64 previews can overflow or
underflow and rounded coordinates can violate an exact boundary; those facts
are reported rather than changing the exact feasibility result.

Bounds: six variables, 32 original rows, coefficient/bound magnitude <=1e6,
128 retained projected rows, at most 4096 generated rows including initial and
carried rows, and 4096 bits per reduced rational intermediate. Growth/work
limits raise an error, never an uncertified feasible/infeasible outcome.
Reference: Rekha Thomas, Chapter 1 (elimination and Farkas alternatives),
https://sites.math.washington.edu/~thomas/teaching/m583_s2008_web/main.pdf
"""

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
Scalar = Annotated[float, Field(strict=True, ge=-1e6, le=1e6)]


class Constraint(InputModel):
    constraint_id: Name
    coefficients: list[Scalar] = Field(min_length=1, max_length=6)
    upper_bound: Scalar


class Input(InputModel):
    variables: list[Name] = Field(min_length=1, max_length=6)
    constraints: list[Constraint] = Field(max_length=32)
    max_generated_rows: int = Field(default=2048, strict=True, ge=1, le=4096)

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be unique")
        if len({row.constraint_id for row in self.constraints}) != len(self.constraints):
            raise ValueError("constraint IDs must be unique")
        if any(len(row.coefficients) != len(self.variables) for row in self.constraints):
            raise ValueError("each coefficient vector must match the variables")
        if len(self.constraints) > self.max_generated_rows:
            raise ValueError("initial rows exceed the generated-row budget")
        return self


class ExactNumber(OutputModel):
    numerator: str
    denominator: str
    value: float | None
    preview_status: Literal["finite", "nonzero_underflow", "overflow"]


class Coordinate(OutputModel):
    variable_index: int
    variable: str
    coordinate: ExactNumber


class Slack(OutputModel):
    constraint_row_index: int
    constraint_id: str
    slack: ExactNumber


class Multiplier(OutputModel):
    constraint_row_index: int
    constraint_id: str
    multiplier: ExactNumber


class FarkasCertificate(OutputModel):
    source_multipliers: list[Multiplier]
    combined_coefficients: list[ExactNumber]
    combined_upper_bound: ExactNumber
    nonnegative_multipliers_verified: Literal[True] = True
    zero_left_side_verified: Literal[True] = True
    negative_unit_right_side_verified: Literal[True] = True


class EliminationStage(OutputModel):
    variable_index: int
    input_rows: int
    positive_rows: int
    negative_rows: int
    zero_rows: int
    candidate_pair_count: int
    generated_rows: int
    retained_rows: int
    contradiction_found: bool


class Output(OutputModel):
    status: Literal["feasible", "infeasible"]
    variables: list[str]
    original_constraint_count: int
    exact_witness: list[Coordinate] | None
    original_constraint_slacks: list[Slack] | None
    rounded_witness_satisfies_original: bool | None
    farkas_certificate: FarkasCertificate | None
    elimination_stages: list[EliminationStage]
    total_generated_rows: int
    maximum_retained_rows: int
    exact_certificate_verified: Literal[True] = True
    integer_feasibility_verified: Literal[False] = False
    external_constraints_verified: Literal[False] = False


@dataclass
class _Row:
    coefficients: list[Fraction]
    bound: Fraction
    multipliers: list[Fraction]


@dataclass
class _Budget:
    limit: int
    generated: int = 0
    maximum_retained: int = 0


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 4096:
        raise ValueError("linear feasibility exceeds the 4096-bit rational budget")
    return value


def _dot(first: list[Fraction], second: list[Fraction]) -> Fraction:
    value = Fraction()
    for left, right in zip(first, second, strict=True):
        value = _bounded(value + _bounded(left * right))
    return value


def _number(value: Fraction) -> ExactNumber:
    _bounded(value)
    try:
        preview = float(value)
    except OverflowError:
        preview = None
    status: Literal["finite", "nonzero_underflow", "overflow"] = "finite"
    if preview is None or not isfinite(preview):
        preview = None
        status = "overflow"
    elif value and preview == 0:
        status = "nonzero_underflow"
    return ExactNumber(
        numerator=str(value.numerator),
        denominator=str(value.denominator),
        value=preview,
        preview_status=status,
    )


def _normalized(row: _Row) -> _Row:
    scale = next((abs(value) for value in row.coefficients if value), abs(row.bound))
    if not scale:
        scale = Fraction(1)
    return _Row(
        [_bounded(value / scale) for value in row.coefficients],
        _bounded(row.bound / scale),
        [_bounded(value / scale) for value in row.multipliers],
    )


def _collect(candidates: Iterable[_Row], budget: _Budget) -> tuple[list[_Row], _Row | None]:
    retained: dict[tuple[Fraction, ...], _Row] = {}
    for candidate in candidates:
        budget.generated += 1
        if budget.generated > budget.limit:
            raise ValueError("linear feasibility exceeds the generated-row budget")
        row = _normalized(candidate)
        if not any(row.coefficients):
            if row.bound < 0:
                return list(retained.values()), row
            continue
        key = tuple(row.coefficients)
        current = retained.get(key)
        if current is None or row.bound < current.bound:
            retained[key] = row
        budget.maximum_retained = max(budget.maximum_retained, len(retained))
        if len(retained) > 128:
            raise ValueError("linear feasibility exceeds 128 retained projected rows")
    return list(retained.values()), None


def _project(positive: list[_Row], negative: list[_Row], zero: list[_Row]) -> Iterable[_Row]:
    for row in zero:
        yield _Row(row.coefficients[:-1], row.bound, row.multipliers)
    for upper in positive:
        first = upper.coefficients[-1]
        for lower in negative:
            second = lower.coefficients[-1]
            coefficients = [
                _bounded(_bounded(left / first) - _bounded(right / second))
                for left, right in zip(
                    upper.coefficients[:-1], lower.coefficients[:-1], strict=True
                )
            ]
            bound = _bounded(_bounded(upper.bound / first) - _bounded(lower.bound / second))
            multipliers = [
                _bounded(_bounded(left / first) - _bounded(right / second))
                for left, right in zip(upper.multipliers, lower.multipliers, strict=True)
            ]
            yield _Row(coefficients, bound, multipliers)


def _certificate(
    request: Input, contradiction: _Row, coefficients: list[list[Fraction]], bounds: list[Fraction]
) -> FarkasCertificate:
    multipliers = [_bounded(value / -contradiction.bound) for value in contradiction.multipliers]
    combined = [
        _dot(multipliers, [row[column] for row in coefficients])
        for column in range(len(request.variables))
    ]
    combined_bound = _dot(multipliers, bounds)
    if any(value < 0 for value in multipliers) or any(combined) or combined_bound != -1:
        raise ValueError("exact Farkas certificate verification failed")
    return FarkasCertificate(
        source_multipliers=[
            Multiplier(
                constraint_row_index=index,
                constraint_id=row.constraint_id,
                multiplier=_number(multipliers[index]),
            )
            for index, row in enumerate(request.constraints)
        ],
        combined_coefficients=[_number(value) for value in combined],
        combined_upper_bound=_number(combined_bound),
    )


def execute(request: Input, context: OperationContext) -> Output:
    dimension = len(request.variables)
    count = len(request.constraints)
    coefficients = [[Fraction(value) for value in row.coefficients] for row in request.constraints]
    bounds = [Fraction(row.upper_bound) for row in request.constraints]
    budget = _Budget(request.max_generated_rows)
    initial = (
        _Row(
            coefficients[index],
            bounds[index],
            [Fraction(int(index == other)) for other in range(count)],
        )
        for index in range(count)
    )
    rows, contradiction = _collect(initial, budget)
    history: dict[int, list[_Row]] = {}
    stages: list[EliminationStage] = []
    for variable in reversed(range(dimension)):
        if contradiction is not None:
            break
        history[variable] = rows
        positive = [row for row in rows if row.coefficients[-1] > 0]
        negative = [row for row in rows if row.coefficients[-1] < 0]
        zero = [row for row in rows if not row.coefficients[-1]]
        pairs = len(positive) * len(negative)
        if budget.generated + len(zero) + pairs > budget.limit:
            raise ValueError("next elimination stage exceeds the generated-row budget")
        previous_count, previous_generated = len(rows), budget.generated
        rows, contradiction = _collect(_project(positive, negative, zero), budget)
        stages.append(
            EliminationStage(
                variable_index=variable,
                input_rows=previous_count,
                positive_rows=len(positive),
                negative_rows=len(negative),
                zero_rows=len(zero),
                candidate_pair_count=pairs,
                generated_rows=budget.generated - previous_generated,
                retained_rows=len(rows),
                contradiction_found=contradiction is not None,
            )
        )
    if contradiction is not None:
        return Output(
            status="infeasible",
            variables=request.variables,
            original_constraint_count=count,
            exact_witness=None,
            original_constraint_slacks=None,
            rounded_witness_satisfies_original=None,
            farkas_certificate=_certificate(request, contradiction, coefficients, bounds),
            elimination_stages=stages,
            total_generated_rows=budget.generated,
            maximum_retained_rows=budget.maximum_retained,
        )
    witness: list[Fraction] = []
    for variable in range(dimension):
        lower_bound: Fraction | None = None
        upper_bound: Fraction | None = None
        for row in history[variable]:
            remaining = _bounded(row.bound - _dot(row.coefficients[:-1], witness))
            coefficient = row.coefficients[-1]
            if not coefficient:
                if remaining < 0:
                    raise ValueError("projected zero-coefficient feasibility failed")
                continue
            value = _bounded(remaining / coefficient)
            if coefficient > 0:
                upper_bound = value if upper_bound is None else min(upper_bound, value)
            else:
                lower_bound = value if lower_bound is None else max(lower_bound, value)
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            raise ValueError("reconstructed feasible interval is empty")
        coordinate = Fraction()
        if lower_bound is not None and lower_bound > 0:
            coordinate = lower_bound
        elif upper_bound is not None and upper_bound < 0:
            coordinate = upper_bound
        witness.append(coordinate)
    slacks = [
        _bounded(bound - _dot(row, witness))
        for row, bound in zip(coefficients, bounds, strict=True)
    ]
    if any(value < 0 for value in slacks):
        raise ValueError("original-row exact witness verification failed")
    numbers = [_number(value) for value in witness]
    previews = [number.value for number in numbers]
    rounded_valid: bool | None = None
    if all(value is not None for value in previews):
        rounded = [Fraction(value) for value in previews if value is not None]
        rounded_valid = all(
            _dot(row, rounded) <= bound for row, bound in zip(coefficients, bounds, strict=True)
        )
    return Output(
        status="feasible",
        variables=request.variables,
        original_constraint_count=count,
        exact_witness=[
            Coordinate(variable_index=index, variable=name, coordinate=numbers[index])
            for index, name in enumerate(request.variables)
        ],
        original_constraint_slacks=[
            Slack(
                constraint_row_index=index,
                constraint_id=row.constraint_id,
                slack=_number(slacks[index]),
            )
            for index, row in enumerate(request.constraints)
        ],
        rounded_witness_satisfies_original=rounded_valid,
        farkas_certificate=None,
        elimination_stages=stages,
        total_generated_rows=budget.generated,
        maximum_retained_rows=budget.maximum_retained,
    )


OPERATION = Operation(
    id="features.linear_constraint_feasibility",
    kind="feature",
    description="Solve bounded exact real linear inequalities by Fourier-Motzkin elimination, returning an independently checked witness or normalized Farkas certificate.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
