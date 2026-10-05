"""Reconcile explicitly supplied parent/detail aggregates without float rounding.

Decimal strings are exact, with at most 38 integral and 18 fractional digits.
Each parent declares an expected set of detail IDs and one weighted sum, weighted
mean or count. Count counts resolved, non-null membership rows and requires unit
weights. Sums/means allow nonnegative weights; zero weights are retained. Null
detail values either invalidate reconciliation or are explicitly skipped; skipped
rows contribute neither value, count nor weight. Empty sums/counts are zero;
zero-total-weight means are undefined, never silently zero.

Repeated parent/detail memberships either all invalidate and are excluded, or
all contribute under include_repeated. Repeated parent or detail IDs are always
ambiguous, even with identical contents. Expected membership omissions, unresolved
IDs, invalid weights and rejected unexpected memberships prevent reconciliation.
observed_result describes only accepted resolved contributions and may therefore
be partial; comparison is withheld unless membership resolution is complete.

Comparison uses |supplied-computed| <= absolute_tolerance + relative_tolerance *
|computed|, exactly. This audits arithmetic and supplied membership declarations;
it cannot establish whether those declarations cover an external population.
Bounds: 1000 parents, 10000 details, 20000 memberships and 20000 total expected-ID
declarations, 200 diagnostics with up to 10 source indexes, and 200 parent results
per page. Arithmetic size is bounded by the decimal input width and row limits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
DecimalText = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=58,
        pattern=r"\A-?(?:0|[1-9][0-9]{0,37})(?:\.[0-9]{1,18})?\z",
    ),
]


class Detail(InputModel):
    detail_id: Name
    value: DecimalText | None


class Parent(InputModel):
    parent_id: Name
    aggregation: Literal["weighted_sum", "weighted_mean", "count"]
    supplied_value: DecimalText | None
    expected_detail_ids: list[Name] = Field(max_length=10_000)


class Membership(InputModel):
    parent_id: Name
    detail_id: Name
    weight: DecimalText = "1"


class Input(InputModel):
    parents: list[Parent] = Field(min_length=1, max_length=1000)
    details: list[Detail] = Field(max_length=10_000)
    memberships: list[Membership] = Field(max_length=20_000)
    duplicate_membership_policy: Literal["reject", "include_repeated"] = "reject"
    unexpected_membership_policy: Literal["reject", "include"] = "reject"
    null_value_policy: Literal["reject", "skip"] = "reject"
    absolute_tolerance: DecimalText = "0"
    relative_tolerance: DecimalText = "0"
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    max_row_indexes: int = Field(default=5, strict=True, ge=1, le=10)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def bounds_and_tolerances(self) -> Self:
        if sum(len(row.expected_detail_ids) for row in self.parents) > 20_000:
            raise ValueError("at most 20000 expected-detail declarations are supported")
        if Fraction(self.absolute_tolerance) < 0 or Fraction(self.relative_tolerance) < 0:
            raise ValueError("aggregate tolerances must be nonnegative")
        return self


class ExactValue(OutputModel):
    numerator: str
    denominator: str


class Finding(OutputModel):
    code: str
    parent_row_index: int | None = None
    detail_row_index: int | None = None
    detail_id: str | None = None
    membership_row_indexes: list[int] = Field(default_factory=list)
    omitted_membership_row_indexes: int = 0


class ParentResult(OutputModel):
    parent_row_index: int
    parent_id: str
    aggregation: Literal["weighted_sum", "weighted_mean", "count"]
    expected_distinct_details: int
    declared_membership_count: int
    contributing_membership_count: int
    skipped_null_membership_count: int
    observed_weight: ExactValue
    observed_weighted_sum: ExactValue
    observed_result: ExactValue | None
    fully_resolved: bool = Field(
        description="All membership/weight requirements pass and the observed result is defined; independent of supplied-value presence or agreement."
    )
    supplied_value: ExactValue | None
    absolute_difference: ExactValue | None
    allowed_difference: ExactValue | None
    matches_within_tolerance: bool | None
    passed: bool


class Output(OutputModel):
    passed: bool
    parent_count: int
    detail_count: int
    membership_count: int
    duplicate_membership_policy: Literal["reject", "include_repeated"]
    unexpected_membership_policy: Literal["reject", "include"]
    null_value_policy: Literal["reject", "skip"]
    duplicate_membership_group_count: int
    extra_membership_declarations: int
    unreferenced_detail_row_count: int
    complete_parent_count: int
    matching_parent_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    parents: list[ParentResult]
    offset: int
    has_more: bool
    external_population_completeness_verified: Literal[False] = False


def _exact(value: Fraction) -> ExactValue:
    return ExactValue(numerator=str(value.numerator), denominator=str(value.denominator))


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid_parents: set[int] = set()

    def report(
        code: str,
        parent: int | None = None,
        detail: int | None = None,
        detail_id: str | None = None,
        memberships: list[int] | None = None,
    ) -> None:
        issues[code] += 1
        if parent is not None:
            invalid_parents.add(parent)
        if len(findings) < request.max_diagnostics:
            rows = memberships or []
            findings.append(
                Finding(
                    code=code,
                    parent_row_index=parent,
                    detail_row_index=detail,
                    detail_id=detail_id,
                    membership_row_indexes=rows[: request.max_row_indexes],
                    omitted_membership_row_indexes=max(0, len(rows) - request.max_row_indexes),
                )
            )

    parent_rows: dict[str, list[int]] = defaultdict(list)
    detail_rows: dict[str, list[int]] = defaultdict(list)
    for index, parent in enumerate(request.parents):
        parent_rows[parent.parent_id].append(index)
    for index, detail in enumerate(request.details):
        detail_rows[detail.detail_id].append(index)
    parent_index = {name: rows[0] for name, rows in parent_rows.items() if len(rows) == 1}
    detail_index = {name: rows[0] for name, rows in detail_rows.items() if len(rows) == 1}
    for index, parent in enumerate(request.parents):
        if len(parent_rows[parent.parent_id]) != 1:
            report("ambiguous_parent_id", parent=index)
    for index, detail in enumerate(request.details):
        if len(detail_rows[detail.detail_id]) != 1:
            report("ambiguous_detail_id", detail=index, detail_id=detail.detail_id)

    pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_parent: dict[str, list[int]] = defaultdict(list)
    referenced_details: set[str] = set()
    for index, member in enumerate(request.memberships):
        pairs[member.parent_id, member.detail_id].append(index)
        by_parent[member.parent_id].append(index)
        referenced_details.add(member.detail_id)
    duplicate_groups = extras = 0
    for (parent_id, detail_id), rows in pairs.items():
        if len(rows) > 1:
            duplicate_groups += 1
            extras += len(rows) - 1
            if request.duplicate_membership_policy == "reject":
                report(
                    "duplicate_membership",
                    parent=parent_index.get(parent_id),
                    detail_id=detail_id,
                    memberships=rows,
                )

    expected = [set(parent.expected_detail_ids) for parent in request.parents]
    for index, parent in enumerate(request.parents):
        for detail_id, count in Counter(parent.expected_detail_ids).items():
            if count > 1:
                report("duplicate_expected_detail", parent=index, detail_id=detail_id)
        if len(parent_rows[parent.parent_id]) != 1:
            continue
        actual = {request.memberships[row].detail_id for row in by_parent[parent.parent_id]}
        for detail_id in sorted(expected[index] - actual):
            report("missing_expected_membership", parent=index, detail_id=detail_id)
        for detail_id in sorted(expected[index]):
            if detail_id not in detail_index:
                report("unresolved_expected_detail", parent=index, detail_id=detail_id)
        if request.unexpected_membership_policy == "reject":
            for detail_id in sorted(actual - expected[index]):
                report(
                    "unexpected_membership",
                    parent=index,
                    detail_id=detail_id,
                    memberships=pairs[parent.parent_id, detail_id],
                )

    values = [
        None if detail.value is None else Fraction(detail.value) for detail in request.details
    ]
    sums = [Fraction(0) for _ in request.parents]
    weights = [Fraction(0) for _ in request.parents]
    contributions = [0] * len(request.parents)
    skipped = [0] * len(request.parents)
    for index, member in enumerate(request.memberships):
        parent_position = parent_index.get(member.parent_id)
        if parent_position is None:
            report("unresolved_membership_parent", detail_id=member.detail_id, memberships=[index])
            continue
        detail_position = detail_index.get(member.detail_id)
        if detail_position is None:
            report(
                "unresolved_membership_detail",
                parent=parent_position,
                detail_id=member.detail_id,
                memberships=[index],
            )
            continue
        weight = Fraction(member.weight)
        if weight < 0:
            report(
                "negative_weight",
                parent=parent_position,
                detail=detail_position,
                memberships=[index],
            )
            continue
        if request.parents[parent_position].aggregation == "count" and weight != 1:
            report(
                "count_requires_unit_weight",
                parent=parent_position,
                detail=detail_position,
                memberships=[index],
            )
            continue
        if (
            request.duplicate_membership_policy == "reject"
            and len(pairs[member.parent_id, member.detail_id]) > 1
        ):
            continue
        if (
            request.unexpected_membership_policy == "reject"
            and member.detail_id not in expected[parent_position]
        ):
            continue
        value = values[detail_position]
        if value is None:
            if request.null_value_policy == "reject":
                report(
                    "null_detail_value",
                    parent=parent_position,
                    detail=detail_position,
                    memberships=[index],
                )
            else:
                skipped[parent_position] += 1
            continue
        sums[parent_position] += weight * value
        weights[parent_position] += weight
        contributions[parent_position] += 1

    absolute, relative = Fraction(request.absolute_tolerance), Fraction(request.relative_tolerance)
    results: list[ParentResult] = []
    complete_count = matching_count = 0
    for index, parent in enumerate(request.parents):
        ambiguous = len(parent_rows[parent.parent_id]) != 1
        observed: Fraction | None
        if ambiguous:
            observed = None
        elif parent.aggregation == "count":
            observed = Fraction(contributions[index])
        elif parent.aggregation == "weighted_sum":
            observed = sums[index]
        elif weights[index] == 0:
            observed = None
            report("undefined_weighted_mean", parent=index)
        else:
            observed = sums[index] / weights[index]
        complete = index not in invalid_parents
        supplied = None if parent.supplied_value is None else Fraction(parent.supplied_value)
        difference = allowed = None
        matches = None
        if complete and observed is not None:
            complete_count += 1
            if supplied is None:
                report("missing_supplied_aggregate", parent=index)
            else:
                difference = abs(supplied - observed)
                allowed = absolute + relative * abs(observed)
                matches = difference <= allowed
                if matches:
                    matching_count += 1
                else:
                    report("aggregate_mismatch", parent=index)
        if request.offset <= index < request.offset + request.limit:
            results.append(
                ParentResult(
                    parent_row_index=index,
                    parent_id=parent.parent_id,
                    aggregation=parent.aggregation,
                    expected_distinct_details=len(expected[index]),
                    declared_membership_count=len(by_parent[parent.parent_id]),
                    contributing_membership_count=contributions[index],
                    skipped_null_membership_count=skipped[index],
                    observed_weight=_exact(weights[index]),
                    observed_weighted_sum=_exact(sums[index]),
                    observed_result=None if observed is None else _exact(observed),
                    fully_resolved=complete,
                    supplied_value=None if supplied is None else _exact(supplied),
                    absolute_difference=None if difference is None else _exact(difference),
                    allowed_difference=None if allowed is None else _exact(allowed),
                    matches_within_tolerance=matches,
                    passed=complete and matches is True,
                )
            )
    return Output(
        passed=not issues,
        parent_count=len(request.parents),
        detail_count=len(request.details),
        membership_count=len(request.memberships),
        duplicate_membership_policy=request.duplicate_membership_policy,
        unexpected_membership_policy=request.unexpected_membership_policy,
        null_value_policy=request.null_value_policy,
        duplicate_membership_group_count=duplicate_groups,
        extra_membership_declarations=extras,
        unreferenced_detail_row_count=sum(
            detail.detail_id not in referenced_details for detail in request.details
        ),
        complete_parent_count=complete_count,
        matching_parent_count=matching_count,
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        parents=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.parents),
    )


OPERATION = Operation(
    id="skills.reconcile_aggregates",
    kind="skill",
    description="Reconcile declared detail memberships with exact weighted sums, means and counts, explicit completeness policies and rational tolerance comparisons.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
