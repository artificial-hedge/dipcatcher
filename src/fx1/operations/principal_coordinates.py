"""Classical principal coordinates from supplied symmetric dissimilarities.

Normalize by the largest distance, form B=-J D^2 J/2 with exact rational
double centering, then use NumPy's symmetric float64 eigensolver. Return the
largest positive axes exceeding relative_eigenvalue_tolerance*max(abs(lambda)).
Negative eigenvalues are disclosed, not repaired by changing distances. This
is an independently implemented centering/selection/reconstruction pipeline,
not a call to a PCoA implementation. Reference: Gower (1966), as described at
https://scikit.bio/docs/latest/generated/skbio.stats.ordination.pcoa.html

Eigenvalues use normalized squared-distance units; coordinates restore original
distance units. Fix each axis sign using its largest-magnitude component, with
first-row tie breaking. Repeated eigenspaces may still rotate across platforms.
Reconstruction diagnostics use the returned rounded coordinates. No triangle
inequality, latent dimension, statistical significance, availability or Euclidean
validity is certified. Matrix decomposition remains approximate, even though
double centering is exact before conversion. Nonzero conversion underflow fails.
"""

from __future__ import annotations

from fractions import Fraction
from math import fsum, isfinite, sqrt
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Distance = Annotated[float, Field(strict=True, ge=0, le=1e100)]
DistanceRow = Annotated[list[Distance], Field(min_length=2, max_length=128)]


class Input(InputModel):
    distances: list[DistanceRow] = Field(min_length=2, max_length=128)
    dimensions: int = Field(default=2, strict=True, ge=1, le=32)
    labels: list[str] | None = Field(default=None, min_length=2, max_length=128)
    relative_eigenvalue_tolerance: float = Field(default=1e-12, strict=True, ge=1e-15, le=1e-3)

    @model_validator(mode="after")
    def square_symmetric(self) -> Self:
        size = len(self.distances)
        if any(len(row) != size for row in self.distances):
            raise ValueError("dissimilarities must form a square matrix")
        if any(self.distances[i][i] != 0 for i in range(size)):
            raise ValueError("dissimilarity diagonal must be exactly zero")
        if any(self.distances[i][j] != self.distances[j][i] for i in range(size) for j in range(i)):
            raise ValueError("dissimilarities must be exactly symmetric")
        if self.labels is not None and (
            len(self.labels) != size
            or len(set(self.labels)) != size
            or any(not 1 <= len(label) <= 64 for label in self.labels)
        ):
            raise ValueError(
                "labels must be unique, align with rows and contain 1 through 64 characters"
            )
        return self


class Output(OutputModel):
    point_count: int
    requested_dimensions: int
    retained_dimensions: int
    labels: list[str] | None
    coordinates: list[list[float]]
    distance_scale: float
    normalized_eigenvalues: list[float]
    retained_normalized_eigenvalues: list[float]
    eigenvalue_threshold: float
    positive_axis_count: int
    negative_axis_count: int
    near_zero_axis_count: int
    retained_positive_eigenvalue_mass_fraction: float | None
    significant_negative_eigenvalue_mass_fraction: float | None
    reconstruction_stress: float | None
    maximum_absolute_distance_error: float
    eigensolver_maximum_residual: float
    status: Literal["embedded", "zero_distance_matrix"]
    eigenvalue_units: Literal["distance_squared_divided_by_distance_scale_squared"] = (
        "distance_squared_divided_by_distance_scale_squared"
    )
    reconstruction_uses_returned_coordinates: Literal[True] = True
    negative_eigenvalue_correction_applied: Literal[False] = False
    repeated_eigenspace_orientation_unique: Literal[False] = False
    numpy_version: str


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} is outside representable finite range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    size = len(request.distances)
    scale = max(max(row) for row in request.distances)
    if scale == 0:
        return Output(
            point_count=size,
            requested_dimensions=request.dimensions,
            retained_dimensions=0,
            labels=request.labels,
            coordinates=[[] for _ in range(size)],
            distance_scale=0,
            normalized_eigenvalues=[0.0] * size,
            retained_normalized_eigenvalues=[],
            eigenvalue_threshold=0,
            positive_axis_count=0,
            negative_axis_count=0,
            near_zero_axis_count=size,
            retained_positive_eigenvalue_mass_fraction=None,
            significant_negative_eigenvalue_mass_fraction=None,
            reconstruction_stress=None,
            maximum_absolute_distance_error=0,
            eigensolver_maximum_residual=0,
            status="zero_distance_matrix",
            numpy_version=np.__version__,
        )
    exact_scale = Fraction(scale)
    squared = [[(Fraction(value) / exact_scale) ** 2 for value in row] for row in request.distances]
    row_means = [sum(row, Fraction()) / size for row in squared]
    grand_mean = sum(row_means, Fraction()) / size
    gram = np.array(
        [
            [
                _number(
                    -(squared[i][j] - row_means[i] - row_means[j] + grand_mean) / 2,
                    "centered Gram entry",
                )
                for j in range(size)
            ]
            for i in range(size)
        ],
        dtype=np.float64,
    )
    try:
        values, vectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError as exc:
        raise ValueError("symmetric eigensolver did not converge") from exc
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError("symmetric eigensolver returned a nonfinite result")
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    threshold = request.relative_eigenvalue_tolerance * float(np.max(np.abs(values)))
    positives = [index for index, value in enumerate(values) if value > threshold]
    negatives = [index for index, value in enumerate(values) if value < -threshold]
    selected = positives[: request.dimensions]
    normalized = np.empty((size, len(selected)), dtype=np.float64)
    for axis, index in enumerate(selected):
        direction = vectors[:, index].copy()
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            direction *= -1
        normalized[:, axis] = direction * sqrt(float(values[index]))
    coordinates = [
        [_number(Fraction(float(value)) * exact_scale, "coordinate") for value in row]
        for row in normalized
    ]
    error_sum = Fraction()
    original_sum = Fraction()
    maximum_error = Fraction()
    for i in range(size):
        for j in range(i):
            original = Fraction(request.distances[i][j]) / exact_scale
            reconstructed_squared = sum(
                (
                    ((Fraction(a) - Fraction(b)) / exact_scale) ** 2
                    for a, b in zip(coordinates[i], coordinates[j], strict=True)
                ),
                Fraction(),
            )
            reconstructed_float = correctly_rounded_sqrt(
                reconstructed_squared.numerator, reconstructed_squared.denominator
            )
            if not isfinite(reconstructed_float) or (
                reconstructed_squared and reconstructed_float == 0
            ):
                raise ValueError("reconstructed normalized distance is outside finite range")
            reconstructed = Fraction(reconstructed_float)
            error = reconstructed - original
            error_sum += error * error
            original_sum += original * original
            maximum_error = max(maximum_error, abs(error) * exact_scale)
    ratio = error_sum / original_sum
    stress = correctly_rounded_sqrt(ratio.numerator, ratio.denominator)
    if not isfinite(stress) or (ratio and stress == 0):
        raise ValueError("nonzero reconstruction stress is outside finite range")
    positive_mass = fsum(float(values[index]) for index in positives)
    negative_mass = fsum(-float(values[index]) for index in negatives)
    residual = float(np.max(np.abs(gram @ vectors - vectors * values)))
    return Output(
        point_count=size,
        requested_dimensions=request.dimensions,
        retained_dimensions=len(selected),
        labels=request.labels,
        coordinates=coordinates,
        distance_scale=scale,
        normalized_eigenvalues=[float(value) for value in values],
        retained_normalized_eigenvalues=[float(values[index]) for index in selected],
        eigenvalue_threshold=threshold,
        positive_axis_count=len(positives),
        negative_axis_count=len(negatives),
        near_zero_axis_count=size - len(positives) - len(negatives),
        retained_positive_eigenvalue_mass_fraction=(
            fsum(float(values[index]) for index in selected) / positive_mass
            if positive_mass
            else None
        ),
        significant_negative_eigenvalue_mass_fraction=(
            negative_mass / (positive_mass + negative_mass)
            if positive_mass + negative_mass
            else None
        ),
        reconstruction_stress=stress,
        maximum_absolute_distance_error=_number(maximum_error, "maximum distance error"),
        eigensolver_maximum_residual=residual,
        status="embedded",
        numpy_version=np.__version__,
    )


OPERATION = Operation(
    id="features.principal_coordinates",
    kind="feature",
    description=(
        "Embed symmetric dissimilarities using exact double centering and a numerical symmetric "
        "eigensolver, retaining positive axes, negative-eigenvalue diagnostics and reconstruction "
        "errors from returned coordinates. No distance repair or latent-dimension claim."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
