"""Bounded basic SSA: Hankel embedding, eigentriple grouping, diagonal averaging.

For N observations and window L, H[i,j]=x[i+j], K=N-L+1. No centering,
detrending, padding, forecasting or missing-value filling occurs. Normalize by
an exact power of two and use only NumPy's thin float64 SVD as a decomposition
primitive. Caller groups contain disjoint zero-based singular-component
indices in descending singular-value order. Groups need not cover every
component. The numerical-rank threshold is diagnostic, not an automatic
component filter. Even components below that threshold are used if selected.

Using the rounded SVD factors as exact rationals, form each rank-one matrix
and average entries with i+j=t. Exact addition lets group summation commute
with diagonal averaging. Round each restored group series, then sum those
returned group series to obtain selected_reconstruction. Residuals and their
diagnostics use this returned reconstruction. all_component_reconstruction
instead uses all SVD components before final rounding; its errors expose
decomposition/rounding effects. Singular-energy fractions describe the SVD
trajectory matrix, not unweighted time-series variance explained.

Power-of-two normalization must preserve every input exactly; unsupported
dynamic range fails. Nonzero reconstruction conversion underflow fails.
Diagnostic positive square-root underflow is explicitly flagged. Fractions
are bounded to 16384 bits. Limits: N<=512, L<=64, L*K<=16384,
L*K*min(L,K)<=524288; <=16 groups and <=8192 returned group samples.
SVD remains approximate. Repeated singular subspaces can rotate, so splitting
them among groups is not uniquely defined or platform reproducible. This
whole-block fit is not causal; caller verifies spacing, ordering and source
availability. Reconstruction does not identify signal, noise or market evidence.

Basic SSA reference (embedding, grouping, diagonal averaging):
https://ssa.cf.ac.uk/zhigljavsky/changepoint/Methodology/node2.html
SVD primitive and shape convention:
https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
"""

from fractions import Fraction
from math import frexp, isfinite
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
ComponentIndex = Annotated[int, Field(strict=True, ge=0, le=63)]
GroupIndices = Annotated[list[ComponentIndex], Field(min_length=1, max_length=64)]


class Input(InputModel):
    values: list[Value] = Field(min_length=3, max_length=512)
    window: int = Field(strict=True, ge=2, le=64)
    groups: list[GroupIndices] = Field(min_length=1, max_length=16)
    relative_rank_tolerance: float = Field(default=1e-12, strict=True, ge=1e-15, le=1e-3)

    @model_validator(mode="after")
    def bounded_embedding(self) -> Self:
        count = len(self.values)
        if self.window >= count:
            raise ValueError("window must be smaller than the observation count")
        columns = count - self.window + 1
        components = min(self.window, columns)
        if self.window * columns > 16_384 or self.window * columns * components > 524_288:
            raise ValueError("SSA exceeds 16384 matrix cells or 524288 component-matrix cells")
        indices = [index for group in self.groups for index in group]
        if len(set(indices)) != len(indices) or any(index >= components for index in indices):
            raise ValueError("groups must be disjoint and contain valid singular-component indices")
        if count * len(self.groups) > 8_192:
            raise ValueError("SSA exceeds 8192 returned group samples")
        return self


class Magnitude(OutputModel):
    value: float
    positive_underflow: bool


class Group(OutputModel):
    component_indices: list[int]
    below_rank_threshold_indices: list[int]
    reconstructed_values: list[float]
    singular_squared_mass_fraction: float | None


class Output(OutputModel):
    observation_count: int
    window: int
    trajectory_columns: int
    component_count: int
    normalization_power_of_two: int
    normalized_singular_values: list[float]
    normalized_rank_threshold: float
    numerical_rank: int
    groups: list[Group]
    unselected_component_indices: list[int]
    all_components_selected: bool
    selected_reconstruction: list[float]
    residuals: list[float]
    all_component_reconstruction: list[float]
    anti_diagonal_counts: list[int]
    residual_root_mean_square: Magnitude
    maximum_absolute_residual: float
    all_component_reconstruction_root_mean_square_error: Magnitude
    all_component_maximum_absolute_error: float
    svd_relative_frobenius_reconstruction_error: Magnitude | None
    close_adjacent_singular_index_pairs: list[list[int]]
    close_pairs_split_between_selection_groups: list[list[int]]
    status: Literal["reconstructed", "zero_input"]
    centering: Literal["none"] = "none"
    rank_threshold_filters_components: Literal[False] = False
    singular_mass_is_series_variance_explained: Literal[False] = False
    repeated_subspace_basis_unique: Literal[False] = False
    evenly_spaced_input_verified: Literal[False] = False
    timing_verified: Literal[False] = False
    numpy_version: str


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 16_384:
        raise ValueError("SSA rational arithmetic exceeds the 16384-bit budget")
    return value


def _number(value: Fraction, label: str) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite binary64 range") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _root(value: Fraction) -> Magnitude:
    _bounded(value)
    result = correctly_rounded_sqrt(value.numerator, value.denominator)
    if not isfinite(result):
        raise ValueError("SSA square-root diagnostic exceeds finite binary64 range")
    return Magnitude(value=result, positive_underflow=bool(value and result == 0))


def execute(request: Input, context: OperationContext) -> Output:
    count, rows = len(request.values), request.window
    columns = count - rows + 1
    components = min(rows, columns)
    maximum = max(abs(value) for value in request.values)
    exponent = frexp(maximum)[1] if maximum else 0
    scale = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
    original = [Fraction(value) for value in request.values]
    normalized_exact = [value / scale for value in original]
    normalized = [_number(value, "normalized input") for value in normalized_exact]
    if any(
        Fraction(rounded) != exact
        for rounded, exact in zip(normalized, normalized_exact, strict=True)
    ):
        raise ValueError("power-of-two normalization cannot preserve the supplied dynamic range")
    trajectory = np.array(
        [[normalized[i + j] for j in range(columns)] for i in range(rows)], dtype=np.float64
    )
    try:
        left, singular, right = np.linalg.svd(trajectory, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("SSA singular-value decomposition did not converge") from error
    if not all(np.isfinite(array).all() for array in (left, singular, right)):
        raise ValueError("SSA singular-value decomposition returned nonfinite factors")
    singular_values = [float(value) for value in singular]
    if any(value < 0 for value in singular_values) or any(
        singular_values[index] < singular_values[index + 1] for index in range(components - 1)
    ):
        raise ValueError("SSA singular values must be nonnegative and descending")
    exact_singular = [Fraction(value) for value in singular_values]
    threshold = exact_singular[0] * Fraction(request.relative_rank_tolerance)
    retained = [value > threshold for value in exact_singular]
    mass = sum((value * value for value in exact_singular), Fraction())
    if maximum and not mass:
        raise ValueError("SVD returned zero singular energy for a nonzero input")

    # Own rank-one reconstruction, keeping products/sums exact after the SVD.
    # Accumulate along each anti-diagonal rather than using a Hankelization API.
    full_matrix = [[Fraction() for _ in range(columns)] for _ in range(rows)]
    counts = [0] * count
    for i in range(rows):
        for j in range(columns):
            counts[i + j] += 1
    elementary: list[list[Fraction]] = []
    for component in range(components):
        diagonal_sums = [Fraction() for _ in range(count)]
        weighted_left = [
            Fraction(float(left[i, component])) * exact_singular[component] for i in range(rows)
        ]
        right_values = [Fraction(float(right[component, j])) for j in range(columns)]
        for i in range(rows):
            for j in range(columns):
                contribution = weighted_left[i] * right_values[j]
                full_matrix[i][j] += contribution
                diagonal_sums[i + j] += contribution
        elementary.append(
            [
                _bounded(total / multiplicity)
                for total, multiplicity in zip(diagonal_sums, counts, strict=True)
            ]
        )

    groups: list[Group] = []
    for indices in request.groups:
        reconstruction = [
            _number(
                sum((elementary[index][t] for index in indices), Fraction()) * scale,
                "group reconstruction",
            )
            for t in range(count)
        ]
        group_mass = sum((exact_singular[index] ** 2 for index in indices), Fraction())
        groups.append(
            Group(
                component_indices=indices,
                below_rank_threshold_indices=[index for index in indices if not retained[index]],
                reconstructed_values=reconstruction,
                singular_squared_mass_fraction=_number(group_mass / mass, "singular mass fraction")
                if mass
                else None,
            )
        )
    selected = [
        _number(
            sum((Fraction(group.reconstructed_values[t]) for group in groups), Fraction()),
            "selected reconstruction",
        )
        for t in range(count)
    ]
    all_reconstructed = [
        _number(
            sum((component[t] for component in elementary), Fraction()) * scale,
            "all-component reconstruction",
        )
        for t in range(count)
    ]
    residuals = [
        value - Fraction(reconstruction)
        for value, reconstruction in zip(original, selected, strict=True)
    ]
    all_errors = [
        value - Fraction(reconstruction)
        for value, reconstruction in zip(original, all_reconstructed, strict=True)
    ]
    matrix_energy, matrix_error = Fraction(), Fraction()
    for i in range(rows):
        for j in range(columns):
            value = normalized_exact[i + j]
            matrix_energy += value * value
            matrix_error += (value - full_matrix[i][j]) ** 2
    membership = {
        component: group_index
        for group_index, group in enumerate(request.groups)
        for component in group
    }
    close_pairs = [
        [index, index + 1]
        for index in range(components - 1)
        if exact_singular[index] - exact_singular[index + 1] <= threshold
    ]
    return Output(
        observation_count=count,
        window=rows,
        trajectory_columns=columns,
        component_count=components,
        normalization_power_of_two=exponent,
        normalized_singular_values=singular_values,
        normalized_rank_threshold=_number(threshold, "rank threshold"),
        numerical_rank=sum(retained),
        groups=groups,
        unselected_component_indices=[
            index for index in range(components) if index not in membership
        ],
        all_components_selected=len(membership) == components,
        selected_reconstruction=selected,
        residuals=[_number(value, "residual") for value in residuals],
        all_component_reconstruction=all_reconstructed,
        anti_diagonal_counts=counts,
        residual_root_mean_square=_root(
            sum((value * value for value in residuals), Fraction()) / count
        ),
        maximum_absolute_residual=_number(
            max(abs(value) for value in residuals), "maximum residual"
        ),
        all_component_reconstruction_root_mean_square_error=_root(
            sum((value * value for value in all_errors), Fraction()) / count
        ),
        all_component_maximum_absolute_error=_number(
            max(abs(value) for value in all_errors), "maximum all-component error"
        ),
        svd_relative_frobenius_reconstruction_error=_root(matrix_error / matrix_energy)
        if matrix_energy
        else None,
        close_adjacent_singular_index_pairs=close_pairs,
        close_pairs_split_between_selection_groups=[
            pair for pair in close_pairs if membership.get(pair[0]) != membership.get(pair[1])
        ],
        status="reconstructed" if maximum else "zero_input",
        numpy_version=np.__version__,
    )


OPERATION = Operation(
    id="features.singular_spectrum_analysis",
    kind="feature",
    description="Decompose bounded Hankel trajectories and reconstruct caller-selected SSA component groups.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
