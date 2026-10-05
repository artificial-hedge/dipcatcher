"""Sample entropy using equal populations of consecutive templates.

For N supplied values and embedding length m, both lengths m and m+1 use
starting indices 0,...,N-m-1. Compare each unordered distinct pair once; self
matches are excluded, but overlapping templates are included. Chebyshev
distance is the maximum absolute coordinate difference. The supplied absolute
tolerance is not scaled by sample standard deviation. The caller explicitly
chooses distance<=tolerance (default) or distance<tolerance.

Let B count matching length-m pairs and A count matching length-(m+1) pairs.
Return -log(A/B). When B=0 the ratio is undefined; when B>0 and A=0 the
entropy is positive infinity, represented by a null value and explicit status.
Fewer than two eligible templates have their own status. No pseudocount,
clipping, temporal exclusion window or independence assumption is added.

All differences and threshold comparisons use exact integer units of the
binary64 values and tolerance, avoiding cancellation and overflow in matching.
The logarithm is a floating approximation and uses log1p when A is near B.
The caller must order and select observations available for its intended use.
This statistic does not establish deterministic dynamics or market evidence.
Reference: Richman, Lake and Moorman (2004), Sample Entropy, pp174-175:
https://www.wavemetrics.com/sites/default/files/2018-10/04RichmanMoorman_SampleEntropyChapter.pdf
"""

from math import log1p
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Rule = Literal["inclusive", "strict"]


class Input(InputModel):
    values: list[Value] = Field(min_length=2, max_length=2_048)
    embedding_length: int = Field(default=2, strict=True, ge=1, le=32)
    tolerance: float = Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)
    threshold_rule: Rule = "inclusive"

    @model_validator(mode="after")
    def bound_comparisons(self) -> Self:
        templates = max(0, len(self.values) - self.embedding_length)
        if templates * (templates - 1) // 2 * (self.embedding_length + 1) > 2_000_000:
            raise ValueError("at most 2000000 candidate coordinate comparisons are supported")
        return self


class Output(OutputModel):
    observation_count: int
    embedding_length: int
    template_population: int
    candidate_pairs: int
    matching_m_pairs: int
    matching_m_plus_one_pairs: int
    coordinate_comparisons: int
    tolerance: float
    threshold_rule: Rule
    conditional_match_probability: float | None
    sample_entropy: float | None
    status: Literal[
        "finite", "no_template_matches", "infinite_no_forward_matches", "insufficient_templates"
    ]
    distance: Literal["chebyshev"] = "chebyshev"
    log_base: Literal["natural"] = "natural"
    self_matches_included: Literal[False] = False
    overlapping_templates_included: Literal[True] = True
    tolerance_scaled_automatically: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    """Count nested matches on one bounded template population using exact thresholds."""
    ratios = [value.as_integer_ratio() for value in (*request.values, request.tolerance)]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    units = [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ]
    threshold = units.pop()
    length = request.embedding_length
    templates = max(0, len(request.values) - length)
    inclusive = request.threshold_rule == "inclusive"
    matches = forwards = comparisons = 0
    for left in range(templates):
        for right in range(left + 1, templates):
            matched = True
            for offset in range(length):
                difference = abs(units[left + offset] - units[right + offset])
                comparisons += 1
                if difference > threshold or (not inclusive and difference == threshold):
                    matched = False
                    break
            if not matched:
                continue
            matches += 1
            difference = abs(units[left + length] - units[right + length])
            comparisons += 1
            forwards += difference <= threshold if inclusive else difference < threshold
    probability = entropy = None
    status: Literal[
        "finite", "no_template_matches", "infinite_no_forward_matches", "insufficient_templates"
    ]
    if templates < 2:
        status = "insufficient_templates"
    elif not matches:
        status = "no_template_matches"
    elif not forwards:
        status = "infinite_no_forward_matches"
        probability = 0.0
    else:
        status = "finite"
        probability = forwards / matches
        entropy = log1p((matches - forwards) / forwards)
    return Output(
        observation_count=len(request.values),
        embedding_length=length,
        template_population=templates,
        candidate_pairs=templates * (templates - 1) // 2,
        matching_m_pairs=matches,
        matching_m_plus_one_pairs=forwards,
        coordinate_comparisons=comparisons,
        tolerance=request.tolerance,
        threshold_rule=request.threshold_rule,
        conditional_match_probability=probability,
        sample_entropy=entropy,
        status=status,
    )


OPERATION = Operation(
    id="features.sample_entropy",
    kind="feature",
    description=(
        "Count distinct matching template pairs at consecutive embedding lengths using exact "
        "Chebyshev tolerance comparisons and return sample entropy with explicit zero-match statuses."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
