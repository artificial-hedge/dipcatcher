"""Score supplied Markov forecasts through exact evidence-conditioned filtering.

Initial and transition inputs are nonnegative masses, normalized exactly per
row. Each path position is a nonempty set of possible state names or null for
unobserved. Forward prediction followed by restriction and renormalization sums
over all compatible hidden paths without enumerating them. The score is minus
the sum of log evidence masses. Null/full-state observations add zero loss.

Zero evidence mass makes the whole path loss infinite; subsequent conditional
distributions are undefined, never restarted. This is a time-homogeneous model
over caller-defined equally spaced steps, not an estimated model. A joint path
score is not divided by path length. Positive path weights are normalized only
for the aggregate. Proper interpretation of coarsened observations requires a
prespecified observation partition or noninformative observation mechanism;
selection, lengths, weights, Markov assumptions and timing are not verified.

Bounds: 2..16 states, 64 paths, 256 positions/path, 1024 total positions and
131072 state-pair work cells. Exact normalized fractions have a 16384-bit cap.
At most 200 trace positions return per page; all paths are scored before paging.
Numerical probability previews may underflow, with counts; a nonzero loss outside
binary64 range fails. Logarithms remain approximate. Path-law reference:
https://data140.org/textbook/content/chapter-10/transitions/
"""

from __future__ import annotations

from fractions import Fraction
from math import isfinite, log, log1p
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
MassValue = Annotated[float, Field(strict=True, ge=0, le=1e100)]
MassRow = Annotated[list[MassValue], Field(min_length=2, max_length=16)]
Evidence = Annotated[list[Name], Field(min_length=1, max_length=16)]


class Path(InputModel):
    observations: list[Evidence | None] = Field(min_length=1, max_length=256)
    weight: float = Field(default=1, strict=True, ge=1e-100, le=1e100)


class Input(InputModel):
    states: list[Name] = Field(min_length=2, max_length=16)
    initial_masses: MassRow
    transition_masses: list[MassRow] = Field(min_length=2, max_length=16)
    paths: list[Path] = Field(min_length=1, max_length=64)
    trace_offset: int = Field(default=0, strict=True, ge=0, le=1024)
    trace_limit: int = Field(default=50, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def finite_model_and_evidence(self) -> Self:
        size = len(self.states)
        universe = set(self.states)
        if len(universe) != size or len(self.transition_masses) != size:
            raise ValueError("states must be unique and transition rows align with them")
        for row in (self.initial_masses, *self.transition_masses):
            if len(row) != size or not any(row):
                raise ValueError("every probability mass row must align and have positive total")
            if any(0 < value < 1e-100 for value in row):
                raise ValueError("positive probability masses must be at least 1e-100")
        positions = sum(len(path.observations) for path in self.paths)
        if positions > 1024 or positions * size * size > 131_072:
            raise ValueError("Markov scoring exceeds 1024 positions or 131072 state-pair cells")
        for path in self.paths:
            for evidence in path.observations:
                if evidence is not None and (
                    len(set(evidence)) != len(evidence) or not set(evidence) <= universe
                ):
                    raise ValueError("each evidence set must contain unique declared states")
        return self


class Trace(OutputModel):
    trace_index: int
    path_index: int
    position: int
    allowed_states: list[str] | None
    status: Literal["conditioned", "zero_evidence_mass", "undefined_after_zero_prefix"]
    predictive_probabilities: list[float] | None
    posterior_probabilities: list[float] | None
    predictive_positive_underflows: int
    posterior_positive_underflows: int
    conditional_evidence_probability: float | None
    evidence_probability_underflow: bool
    negative_log_evidence: float | None


class PathScore(OutputModel):
    path_index: int
    position_count: int
    informative_position_count: int
    weight: float
    negative_log_path_probability: float | None
    first_impossible_position: int | None
    status: Literal["finite", "infinite_zero_likelihood"]


class Output(OutputModel):
    states: list[str]
    path_count: int
    total_position_count: int
    uninformative_path_count: int
    infinite_path_count: int
    weighted_mean_negative_log_path_probability: float | None
    aggregate_status: Literal["finite", "infinite_zero_likelihood"]
    paths: list[PathScore]
    trace: list[Trace]
    trace_offset: int
    trace_has_more: bool
    path_length_normalization_applied: Literal[False] = False
    probability_masses_normalized: Literal[True] = True
    time_homogeneous_transition_model: Literal[True] = True
    observation_mechanism_verified: Literal[False] = False
    outcome_independent_lengths_and_weights_verified: Literal[False] = False
    markov_and_spacing_assumptions_verified: Literal[False] = False
    forecast_timing_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 16_384:
        raise ValueError("Markov filter fraction exceeds 16384-bit arithmetic budget")
    return value


def _normalize(values: list[float]) -> list[Fraction]:
    exact = [Fraction(value) for value in values]
    total = sum(exact, Fraction())
    return [_bounded(value / total) for value in exact]


def _number(value: Fraction) -> float:
    result = float(value)
    if not isfinite(result) or (value and result == 0):
        raise ValueError("nonzero Markov score is outside finite binary64 range")
    return result


def _loss(probability: Fraction) -> float:
    if probability > Fraction(1, 2):
        return -log1p(_number(probability - 1))
    exponent = probability.numerator.bit_length() - probability.denominator.bit_length()
    scaled = Fraction(probability.numerator << -exponent, probability.denominator)
    result = -log(float(scaled)) - exponent * log(2)
    if not isfinite(result) or result <= 0:
        raise ValueError("positive evidence log loss is outside supported numerical range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    size = len(request.states)
    positions = {name: index for index, name in enumerate(request.states)}
    initial = _normalize(request.initial_masses)
    transitions = [_normalize(row) for row in request.transition_masses]
    rows: list[PathScore] = []
    trace: list[Trace] = []
    trace_index = infinite_count = uninformative = 0
    weighted_loss = total_weight = Fraction()
    for path_index, path in enumerate(request.paths):
        posterior = initial
        loss = Fraction()
        first_impossible: int | None = None
        informative = 0
        for position, evidence in enumerate(path.observations):
            informative += evidence is not None and len(evidence) < size
            predictive: list[Fraction] | None = None
            probability: Fraction | None = None
            increment: float | None = None
            status: Literal["conditioned", "zero_evidence_mass", "undefined_after_zero_prefix"]
            if first_impossible is not None:
                status = "undefined_after_zero_prefix"
            else:
                if position == 0:
                    predictive = initial
                else:
                    predictive = []
                    for target in range(size):
                        mass = Fraction()
                        for source in range(size):
                            mass = _bounded(
                                mass + _bounded(posterior[source] * transitions[source][target])
                            )
                        predictive.append(mass)
                allowed = set(range(size)) if evidence is None else {positions[s] for s in evidence}
                probability = _bounded(sum((predictive[i] for i in sorted(allowed)), Fraction()))
                if not probability:
                    first_impossible = position
                    status = "zero_evidence_mass"
                else:
                    increment = _loss(probability)
                    loss = _bounded(loss + Fraction(increment))
                    posterior = [
                        _bounded(value / probability) if i in allowed else Fraction()
                        for i, value in enumerate(predictive)
                    ]
                    status = "conditioned"
            if request.trace_offset <= trace_index < request.trace_offset + request.trace_limit:
                trace.append(
                    Trace(
                        trace_index=trace_index,
                        path_index=path_index,
                        position=position,
                        allowed_states=evidence,
                        status=status,
                        predictive_probabilities=None
                        if predictive is None
                        else [float(p) for p in predictive],
                        posterior_probabilities=[float(p) for p in posterior]
                        if status == "conditioned"
                        else None,
                        predictive_positive_underflows=0
                        if predictive is None
                        else sum(bool(p and float(p) == 0) for p in predictive),
                        posterior_positive_underflows=sum(
                            bool(p and float(p) == 0) for p in posterior
                        )
                        if status == "conditioned"
                        else 0,
                        conditional_evidence_probability=None
                        if probability is None
                        else float(probability),
                        evidence_probability_underflow=bool(
                            probability and float(probability) == 0
                        ),
                        negative_log_evidence=increment,
                    )
                )
            trace_index += 1
        weight = Fraction(path.weight)
        total_weight += weight
        infinite_count += first_impossible is not None
        uninformative += informative == 0
        if first_impossible is None:
            weighted_loss = _bounded(weighted_loss + weight * loss)
        rows.append(
            PathScore(
                path_index=path_index,
                position_count=len(path.observations),
                informative_position_count=informative,
                weight=path.weight,
                negative_log_path_probability=_number(loss) if first_impossible is None else None,
                first_impossible_position=first_impossible,
                status="finite" if first_impossible is None else "infinite_zero_likelihood",
            )
        )
    return Output(
        states=request.states,
        path_count=len(rows),
        total_position_count=trace_index,
        uninformative_path_count=uninformative,
        infinite_path_count=infinite_count,
        weighted_mean_negative_log_path_probability=None
        if infinite_count
        else _number(weighted_loss / total_weight),
        aggregate_status="infinite_zero_likelihood" if infinite_count else "finite",
        paths=rows,
        trace=trace,
        trace_offset=request.trace_offset,
        trace_has_more=request.trace_offset + len(trace) < trace_index,
    )


OPERATION = Operation(
    id="skills.score_markov_paths",
    kind="skill",
    description="Score fully or partially observed Markov paths using exact forward evidence conditioning, explicit zero support and paginated predictive/posterior traces.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
