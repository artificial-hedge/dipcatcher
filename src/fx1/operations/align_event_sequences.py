"""Align two ordered event streams under explicitly supplied eligibility and costs.

This is global sequence alignment: every row is consumed by one match or one
gap. Matches require equal string keys and an inclusive clock tolerance, with
optional clock direction. Costs are exact nonnegative integers, including an
optional cost per microsecond of clock separation. No match asserts identity or
economic truth. Inputs must already be in nondecreasing event-time order.

Equal-cost predecessors are retained as a bit mask. Traceback prefers match,
then left-only, then right-only, working backward from the complete prefixes.
Optimal-path counts saturate at two: ambiguity includes different orderings of
gap operations even if the matched row pairs are identical. Diagnostics locate
equal-cost predecessor choices on the selected optimal path.

The prefix dynamic-programming recurrence follows global alignment as presented
in MIT 6.047, chapter 2; this implementation supplies its own event eligibility:
https://ocw.mit.edu/courses/6-047-computational-biology-fall-2015/pages/open-textbook/
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Cost = Annotated[int, Field(strict=True, ge=0, le=10**9)]
Move = Literal["match", "left_only", "right_only"]
Direction = Literal["either", "right_not_before_left", "right_not_after_left"]
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MATCH, _LEFT, _RIGHT = 1, 2, 4


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


def _epoch_microseconds(clock: datetime) -> int:
    delta = clock - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class Event(InputModel):
    match_key: Name
    event_time: AwareDatetime

    @field_validator("event_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    left: list[Event] = Field(max_length=2_000)
    right: list[Event] = Field(max_length=2_000)
    max_time_difference_microseconds: int = Field(default=0, strict=True, ge=0, le=10**18)
    clock_direction: Direction = "either"
    left_gap_cost: Cost = 1
    right_gap_cost: Cost = 1
    match_cost: Cost = 0
    time_difference_cost_per_microsecond: Cost = 0
    max_dp_cells: int = Field(default=250_000, strict=True, ge=1, le=1_000_000)
    offset: int = Field(default=0, strict=True, ge=0, le=4_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_ambiguity_diagnostics: int = Field(default=20, strict=True, ge=0, le=100)

    @model_validator(mode="after")
    def ordered_and_bounded(self) -> Self:
        if (len(self.left) + 1) * (len(self.right) + 1) > self.max_dp_cells:
            raise ValueError("(left_count + 1) * (right_count + 1) exceeds max_dp_cells")
        for rows in (self.left, self.right):
            if any(
                rows[index].event_time < rows[index - 1].event_time for index in range(1, len(rows))
            ):
                raise ValueError("both streams must be supplied in nondecreasing event-time order")
        return self


class Step(OutputModel):
    alignment_index: int
    operation: Move
    left_row_index: int | None
    right_row_index: int | None
    right_minus_left_microseconds: int | None
    cost: int


class Ambiguity(OutputModel):
    left_prefix_length: int
    right_prefix_length: int
    chosen_predecessor: Move
    equally_optimal_predecessors: list[Move]


class Output(OutputModel):
    left_count: int
    right_count: int
    dp_cells: int
    eligible_pair_count: int
    total_cost: int
    matched_pair_count: int
    unmatched_left_count: int
    unmatched_right_count: int
    alignment_step_count: int
    optimal_path_count_capped_at_two: int
    optimal_alignment_ambiguous: bool
    ambiguity_definition: Literal["distinct_operation_paths_including_gap_order"] = (
        "distinct_operation_paths_including_gap_order"
    )
    alignment_semantics: Literal["cost_optimum_not_verified_event_identity"] = (
        "cost_optimum_not_verified_event_identity"
    )
    tie_policy: Literal["backward_match_then_left_only_then_right_only"] = (
        "backward_match_then_left_only_then_right_only"
    )
    clock_direction: Direction
    max_time_difference_microseconds: int
    left_gap_cost: int
    right_gap_cost: int
    match_cost: int
    time_difference_cost_per_microsecond: int
    selected_path_ambiguity_count: int
    omitted_ambiguity_diagnostics: int
    ambiguity_diagnostics: list[Ambiguity]
    offset: int
    next_offset: int | None
    steps: list[Step]


def execute(request: Input, context: OperationContext) -> Output:
    left_count, right_count = len(request.left), len(request.right)
    width = right_count + 1
    masks = bytearray((left_count + 1) * width)
    left_times = [_epoch_microseconds(row.event_time) for row in request.left]
    right_times = [_epoch_microseconds(row.event_time) for row in request.right]
    previous_costs = [index * request.right_gap_cost for index in range(width)]
    previous_counts = [1] * width
    for column in range(1, width):
        masks[column] = _RIGHT
    eligible_count = 0
    for row in range(1, left_count + 1):
        current_costs = [row * request.left_gap_cost] + [0] * right_count
        current_counts = [1] + [0] * right_count
        masks[row * width] = _LEFT
        for column in range(1, width):
            delta = right_times[column - 1] - left_times[row - 1]
            eligible = (
                request.left[row - 1].match_key == request.right[column - 1].match_key
                and abs(delta) <= request.max_time_difference_microseconds
                and (request.clock_direction != "right_not_before_left" or delta >= 0)
                and (request.clock_direction != "right_not_after_left" or delta <= 0)
            )
            alternatives = [
                (previous_costs[column] + request.left_gap_cost, _LEFT, previous_counts[column]),
                (
                    current_costs[column - 1] + request.right_gap_cost,
                    _RIGHT,
                    current_counts[column - 1],
                ),
            ]
            if eligible:
                eligible_count += 1
                match_cost = (
                    request.match_cost + abs(delta) * request.time_difference_cost_per_microsecond
                )
                alternatives.append(
                    (previous_costs[column - 1] + match_cost, _MATCH, previous_counts[column - 1])
                )
            minimum = min(cost for cost, _, _ in alternatives)
            mask = 0
            paths = 0
            for cost, move, predecessor_paths in alternatives:
                if cost == minimum:
                    mask |= move
                    paths = min(2, paths + predecessor_paths)
            current_costs[column] = minimum
            current_counts[column] = paths
            masks[row * width + column] = mask
        previous_costs, previous_counts = current_costs, current_counts

    row, column = left_count, right_count
    reversed_steps: list[tuple[Move, int | None, int | None, int | None, int]] = []
    ambiguities: list[Ambiguity] = []
    ambiguity_count = 0
    matched_count = 0
    while row or column:
        mask = masks[row * width + column]
        moves: list[Move] = []
        if mask & _MATCH:
            moves.append("match")
        if mask & _LEFT:
            moves.append("left_only")
        if mask & _RIGHT:
            moves.append("right_only")
        chosen = moves[0]
        if len(moves) > 1:
            ambiguity_count += 1
            if len(ambiguities) < request.max_ambiguity_diagnostics:
                ambiguities.append(
                    Ambiguity(
                        left_prefix_length=row,
                        right_prefix_length=column,
                        chosen_predecessor=chosen,
                        equally_optimal_predecessors=moves,
                    )
                )
        if chosen == "match":
            delta = right_times[column - 1] - left_times[row - 1]
            cost = request.match_cost + abs(delta) * request.time_difference_cost_per_microsecond
            reversed_steps.append((chosen, row - 1, column - 1, delta, cost))
            matched_count += 1
            row -= 1
            column -= 1
        elif chosen == "left_only":
            reversed_steps.append((chosen, row - 1, None, None, request.left_gap_cost))
            row -= 1
        else:
            reversed_steps.append((chosen, None, column - 1, None, request.right_gap_cost))
            column -= 1
    ordered_steps = list(reversed(reversed_steps))
    page_end = min(len(ordered_steps), request.offset + request.limit)
    steps = [
        Step(
            alignment_index=index,
            operation=step[0],
            left_row_index=step[1],
            right_row_index=step[2],
            right_minus_left_microseconds=step[3],
            cost=step[4],
        )
        for index in range(request.offset, page_end)
        for step in [ordered_steps[index]]
    ]
    return Output(
        left_count=left_count,
        right_count=right_count,
        dp_cells=len(masks),
        eligible_pair_count=eligible_count,
        total_cost=previous_costs[-1],
        matched_pair_count=matched_count,
        unmatched_left_count=left_count - matched_count,
        unmatched_right_count=right_count - matched_count,
        alignment_step_count=len(ordered_steps),
        optimal_path_count_capped_at_two=previous_counts[-1],
        optimal_alignment_ambiguous=previous_counts[-1] == 2,
        clock_direction=request.clock_direction,
        max_time_difference_microseconds=request.max_time_difference_microseconds,
        left_gap_cost=request.left_gap_cost,
        right_gap_cost=request.right_gap_cost,
        match_cost=request.match_cost,
        time_difference_cost_per_microsecond=request.time_difference_cost_per_microsecond,
        selected_path_ambiguity_count=ambiguity_count,
        omitted_ambiguity_diagnostics=ambiguity_count - len(ambiguities),
        ambiguity_diagnostics=ambiguities,
        offset=request.offset,
        next_offset=page_end if page_end < len(ordered_steps) else None,
        steps=steps,
    )


OPERATION = Operation(
    id="skills.align_event_sequences",
    kind="skill",
    description=(
        "Globally align ordered event streams with explicit key/time eligibility, integer "
        "costs, source-row lineage, deterministic ties, and optimal-path ambiguity."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
