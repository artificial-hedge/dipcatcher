"""Exact planar hull predicates, lineage and query classification.

The Graham-Andrew scan discards collinear interior boundary points and returns
extreme vertices counterclockwise from the lexicographically smallest point.
All predicates and area/centroid calculations use exact supplied binary floats.
Distances are rounded square roots; their sum is an approximate perimeter.
Degenerate segment perimeter means twice its length; point perimeter is zero.
No coordinate system, units, uncertainty region or source timing is inferred.
Algorithm reference: https://doc.cgal.org/latest/Convex_hull_2/index.html
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Coordinate = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Point = tuple[Coordinate, Coordinate]
ExactPoint = tuple[Fraction, Fraction]


class Input(InputModel):
    points: list[Point] = Field(min_length=1, max_length=1024)
    queries: list[Point] = Field(default_factory=list, max_length=256)


class Number(OutputModel):
    numerator: str
    denominator: str
    value: float
    preview_underflow: bool


class Vertex(OutputModel):
    coordinates: tuple[float, float]
    source_rows: list[int]


class Query(OutputModel):
    query_row: int
    location: Literal["interior", "boundary", "exterior"]
    boundary_edges: list[int]
    vertex_index: int | None


class Output(OutputModel):
    input_count: int
    distinct_point_count: int
    affine_dimension: Literal[0, 1, 2]
    vertices: list[Vertex]
    edges: list[tuple[int, int]]
    area: Number
    area_centroid: tuple[Number, Number] | None
    perimeter: float
    perimeter_approximation: Literal["sum_of_rounded_exact_squared_distance_roots"] = (
        "sum_of_rounded_exact_squared_distance_roots"
    )
    queries: list[Query]
    predicates: Literal["exact_supplied_binary_float_coordinates"] = (
        "exact_supplied_binary_float_coordinates"
    )
    coordinate_system_verified: Literal[False] = False
    availability_validated: Literal[False] = False


def _number(value: Fraction) -> Number:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("exact result exceeds the 12000-bit serialization budget")
    preview = float(value)
    if not math.isfinite(preview):
        raise ValueError("result exceeds finite binary64 range")
    return Number(
        numerator=str(value.numerator),
        denominator=str(value.denominator),
        value=preview,
        preview_underflow=bool(value) and preview == 0,
    )


def _cross(origin: ExactPoint, first: ExactPoint, second: ExactPoint) -> Fraction:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def _chain(points: list[ExactPoint]) -> list[ExactPoint]:
    chain: list[ExactPoint] = []
    for point in points:
        while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0:
            chain.pop()
        chain.append(point)
    return chain


def _on_segment(point: ExactPoint, first: ExactPoint, second: ExactPoint) -> bool:
    return (
        _cross(first, second, point) == 0
        and min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
        and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
    )


def execute(request: Input, context: OperationContext) -> Output:
    source_rows: dict[ExactPoint, list[int]] = {}
    for row, point in enumerate(request.points):
        exact = (Fraction(point[0]), Fraction(point[1]))
        source_rows.setdefault(exact, []).append(row)
    ordered = sorted(source_rows)
    hull = (
        ordered
        if len(ordered) == 1
        else _chain(ordered)[:-1] + _chain(list(reversed(ordered)))[:-1]
    )
    dimension: Literal[0, 1, 2] = 0 if len(hull) == 1 else 1 if len(hull) == 2 else 2
    edges = (
        []
        if dimension == 0
        else [(0, 1)]
        if dimension == 1
        else [(index, (index + 1) % len(hull)) for index in range(len(hull))]
    )
    double_area = Fraction()
    centroid_x = Fraction()
    centroid_y = Fraction()
    lengths: list[Fraction] = []
    for first_index, second_index in edges:
        first, second = hull[first_index], hull[second_index]
        determinant = first[0] * second[1] - second[0] * first[1]
        double_area += determinant
        centroid_x += (first[0] + second[0]) * determinant
        centroid_y += (first[1] + second[1]) * determinant
        square = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
        length = correctly_rounded_sqrt(square.numerator, square.denominator)
        if not math.isfinite(length) or (square and length == 0):
            raise ValueError("edge length cannot be represented as a nonzero finite float")
        lengths.append(Fraction(length))
    centroid = None
    if dimension == 2:
        if double_area <= 0:
            raise ArithmeticError("counterclockwise hull must have positive area")
        centroid = (
            _number(centroid_x / (3 * double_area)),
            _number(centroid_y / (3 * double_area)),
        )
    else:
        double_area = Fraction()
    results: list[Query] = []
    vertex_lookup = {point: index for index, point in enumerate(hull)}
    for row, point in enumerate(request.queries):
        query = (Fraction(point[0]), Fraction(point[1]))
        boundary = [
            index
            for index, (first, second) in enumerate(edges)
            if _on_segment(query, hull[first], hull[second])
        ]
        vertex_index = vertex_lookup.get(query)
        location: Literal["interior", "boundary", "exterior"] = "exterior"
        if vertex_index is not None or boundary:
            location = "boundary"
        elif dimension == 2 and all(
            _cross(hull[first], hull[second], query) > 0 for first, second in edges
        ):
            location = "interior"
        results.append(
            Query(
                query_row=row, location=location, boundary_edges=boundary, vertex_index=vertex_index
            )
        )
    return Output(
        input_count=len(request.points),
        distinct_point_count=len(ordered),
        affine_dimension=dimension,
        vertices=[
            Vertex(coordinates=(float(point[0]), float(point[1])), source_rows=source_rows[point])
            for point in hull
        ],
        edges=edges,
        area=_number(double_area / 2),
        area_centroid=centroid,
        perimeter=float(sum(lengths, Fraction()) * (2 if dimension == 1 else 1)),
        queries=results,
    )


OPERATION = Operation(
    id="features.convex_hull_2d",
    kind="feature",
    description="Construct an exact-predicate planar convex hull with source-row lineage, area/centroid, perimeter and boundary-aware point classification, including collinear and coincident inputs.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
