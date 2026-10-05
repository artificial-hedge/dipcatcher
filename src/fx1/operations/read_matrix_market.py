"""Read bounded Matrix Market coordinate matrices without constructing dense arrays.

Format: https://math.nist.gov/MatrixMarket/formats.html. Only uncompressed ASCII
.mtx coordinate files are accepted. Real, integer, complex and pattern fields
are supported; array storage and nonstandard numeric tokens are rejected.
Symmetric, skew-symmetric and Hermitian sources must be square and supply only
the lower triangle. Skew diagonals must be omitted; Hermitian diagonals must
have exactly zero imaginary component. Pattern supports general/symmetric only,
and Hermitian requires complex values. Banner identifiers are case insensitive.

Input indices are one based. Output entries are zero based, ordered by row and
column, and include the symmetry-implied triangle. Duplicate coordinates either
raise or sum exactly before expansion; pattern duplicates always raise. Explicit
zeros and cancellations are retained. Pattern values are null (structural
presence), while numerical components are exact decimal strings, never binary
floating-point approximations. Integer tokens have at most 128 digits; real
tokens have at most 50 coefficient digits, decimal exponent -350 through 300,
and nonzero decimal order -300 through 300. Exact sums use 700-digit arithmetic.

Limits: 8 MB source, 200005 physical lines, 1024 bytes per line, dimension 1e9,
100000 declared coordinate records, 200000 symmetry-expanded entries and 200
preview entries. Empty dimensions require zero records. No matrix properties,
research claims or values beyond the supplied file are inferred.
"""

from __future__ import annotations

import hashlib
import re
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    localcontext,
)
from typing import Literal, cast

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_NATURAL = re.compile(r"[0-9]+\Z")
MatrixField = Literal["real", "integer", "complex", "pattern"]
MatrixSymmetry = Literal["general", "symmetric", "skew-symmetric", "hermitian"]


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    duplicate_policy: Literal["reject", "sum"] = "reject"
    offset: int = Field(default=0, strict=True, ge=0, le=200_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Entry(OutputModel):
    row: int
    column: int
    real: str | None
    imaginary: str | None


class Output(OutputModel):
    rows: int
    columns: int
    field: MatrixField
    symmetry: MatrixSymmetry
    duplicate_policy: Literal["reject", "sum"]
    input_records: int
    stored_unique_entries: int
    duplicate_records_merged: int
    expanded_entry_count: int
    explicit_input_zero_records: int | None
    reduced_zero_entries: int | None
    expanded_nonzero_entries: int | None
    entries: list[Entry]
    offset: int
    has_more: bool
    output_index_base: Literal[0] = 0
    numerical_encoding: Literal["exact_decimal_strings_pattern_values_null"] = (
        "exact_decimal_strings_pattern_values_null"
    )
    symmetry_expanded: Literal[True] = True
    dense_array_allocated: Literal[False] = False
    source_bytes: int
    source_sha256: str


def _natural(token: str, maximum: int) -> int:
    if len(token) > 10 or _NATURAL.fullmatch(token) is None:
        raise ValueError("dimensions, record counts and indices require unsigned decimal integers")
    value = int(token)
    if value > maximum:
        raise ValueError("Matrix Market dimension, count or index exceeds its bound")
    return value


def _number(token: str, integer: bool) -> Decimal:
    if integer:
        if len(token.lstrip("+-")) > 128 or _INTEGER.fullmatch(token) is None:
            raise ValueError("integer matrix values require at most 128 signed decimal digits")
        return Decimal(token)
    match = _REAL.fullmatch(token) if len(token) <= 128 else None
    if match is None:
        raise ValueError(
            "real matrix values require finite decimal notation with optional e exponent"
        )
    exponent = match.group(1)
    if exponent is not None and len(exponent.lstrip("+-")) > 3:
        raise ValueError("real matrix exponent spelling exceeds three decimal digits")
    value = Decimal(token)
    components = value.as_tuple()
    # The lexical guard admits only finite Decimals, whose exponent is an int.
    decimal_exponent = int(components.exponent)
    if len(components.digits) > 50 or not -350 <= decimal_exponent <= 300:
        raise ValueError(
            "real matrix coefficient or decimal exponent exceeds its exact-arithmetic bound"
        )
    if value and not -300 <= value.adjusted() <= 300:
        raise ValueError("nonzero real matrix values require decimal order -300 through 300")
    return value


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".mtx",), max_bytes=8_000_000)
    if content.count(b"\n") > 200_005:
        raise ValueError("Matrix Market source exceeds 200005 physical lines")
    try:
        text = content.decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Matrix Market source must be ASCII") from exc
    raw_lines = text.split("\n")
    if raw_lines[-1:] == [""]:
        raw_lines.pop()
    if not raw_lines or len(raw_lines) > 200_005:
        raise ValueError("Matrix Market source needs a banner within 200005 physical lines")
    lines = [line.removesuffix("\r") for line in raw_lines]
    if any(len(line) > 1024 or "\r" in line or "\x00" in line for line in lines):
        raise ValueError(
            "Matrix Market lines exceed 1024 bytes or contain unsupported CR/NUL characters"
        )
    banner = lines[0].lower().split()
    if len(banner) != 5 or banner[:3] != ["%%matrixmarket", "matrix", "coordinate"]:
        raise ValueError("expected a Matrix Market matrix coordinate banner")
    field, symmetry = banner[3:]
    if field not in ("real", "integer", "complex", "pattern"):
        raise ValueError("unsupported Matrix Market field")
    if symmetry not in ("general", "symmetric", "skew-symmetric", "hermitian"):
        raise ValueError("unsupported Matrix Market symmetry")
    if field == "pattern" and (
        symmetry not in ("general", "symmetric") or request.duplicate_policy == "sum"
    ):
        raise ValueError(
            "pattern matrices support general/symmetric storage and duplicate rejection only"
        )
    if symmetry == "hermitian" and field != "complex":
        raise ValueError("Hermitian storage requires a complex field")
    records = [
        line.split() for line in lines[1:] if line.strip() and not line.lstrip().startswith("%")
    ]
    if not records or len(records[0]) != 3:
        raise ValueError("Matrix Market source requires one rows/columns/records dimension line")
    rows = _natural(records[0][0], 1_000_000_000)
    columns = _natural(records[0][1], 1_000_000_000)
    declared = _natural(records[0][2], 100_000)
    if len(records) - 1 != declared:
        raise ValueError("Matrix Market coordinate record count differs from its declaration")
    if (rows == 0 or columns == 0) and declared:
        raise ValueError("empty Matrix Market dimensions cannot contain coordinates")
    if symmetry != "general" and rows != columns:
        raise ValueError("symmetric Matrix Market storage requires a square matrix")
    stored: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    duplicates = input_zeros = 0
    expected_tokens = {"pattern": 2, "integer": 3, "real": 3, "complex": 4}[field]
    zero = Decimal(0)
    # Maximum accepted component spans powers -350 through 300. Summing at
    # most 100000 such values needs at most 656 significant decimal digits.
    exact_context = Context(
        prec=700,
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow, Inexact, Rounded],
    )
    with localcontext(exact_context):
        for tokens in records[1:]:
            if len(tokens) != expected_tokens:
                raise ValueError("Matrix Market coordinate has the wrong number of fields")
            row, column = _natural(tokens[0], rows), _natural(tokens[1], columns)
            if row == 0 or column == 0:
                raise ValueError("Matrix Market source coordinates are one based")
            if symmetry != "general" and row < column:
                raise ValueError(
                    "symmetric Matrix Market input must contain only the lower triangle"
                )
            if symmetry == "skew-symmetric" and row == column:
                raise ValueError("skew-symmetric Matrix Market diagonals must be omitted")
            real = Decimal(1) if field == "pattern" else _number(tokens[2], field == "integer")
            imaginary = _number(tokens[3], False) if field == "complex" else zero
            if symmetry == "hermitian" and row == column and imaginary:
                raise ValueError(
                    "Hermitian diagonal coordinates must have zero imaginary component"
                )
            input_zeros += int(not real and not imaginary)
            key = row - 1, column - 1
            if key in stored:
                if request.duplicate_policy == "reject":
                    raise ValueError("Matrix Market source contains a duplicate coordinate")
                duplicates += 1
                previous_real, previous_imaginary = stored[key]
                real += previous_real
                imaginary += previous_imaginary
            stored[key] = real, imaginary
    expanded = dict(stored)
    for (row, column), (real, imaginary) in stored.items():
        if row == column or symmetry == "general":
            continue
        if symmetry == "skew-symmetric":
            expanded[column, row] = real.copy_negate(), imaginary.copy_negate()
        elif symmetry == "hermitian":
            expanded[column, row] = real, imaginary.copy_negate()
        else:
            expanded[column, row] = real, imaginary
    if len(expanded) > 200_000:
        raise ValueError("Matrix Market symmetry expansion exceeds 200000 entries")
    ordered = sorted(expanded)
    selected = ordered[request.offset : request.offset + request.limit]
    entries = [
        Entry(
            row=row,
            column=column,
            real=None if field == "pattern" else format(expanded[row, column][0], "f"),
            imaginary=format(expanded[row, column][1], "f") if field == "complex" else None,
        )
        for row, column in selected
    ]
    return Output(
        rows=rows,
        columns=columns,
        field=cast(MatrixField, field),
        symmetry=cast(MatrixSymmetry, symmetry),
        duplicate_policy=request.duplicate_policy,
        input_records=declared,
        stored_unique_entries=len(stored),
        duplicate_records_merged=duplicates,
        expanded_entry_count=len(expanded),
        explicit_input_zero_records=None if field == "pattern" else input_zeros,
        reduced_zero_entries=None
        if field == "pattern"
        else sum(not real and not imaginary for real, imaginary in stored.values()),
        expanded_nonzero_entries=None
        if field == "pattern"
        else sum(bool(real or imaginary) for real, imaginary in expanded.values()),
        entries=entries,
        offset=request.offset,
        has_more=request.offset + len(entries) < len(expanded),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_matrix_market",
    kind="plugin",
    description="Parse bounded Matrix Market coordinate files with exact decimal duplicate reduction, strict triangle/index/count checks and paginated symmetry-expanded sparse entries; no dense tensor allocation.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
