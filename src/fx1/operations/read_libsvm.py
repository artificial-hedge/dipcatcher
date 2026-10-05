"""Read a strict sparse LIBSVM/SVMLight text subset without dense allocation.

Each ASCII record contains one numeric label followed by zero or more positive
index:value pairs in strictly increasing index order. Indices use unsigned
decimal spelling without signs or leading zeros. Index zero/precomputed
kernels, duplicate or reordered indices, multiple labels, qid/cost extensions
and missing values are unsupported. Explicit zero coordinates remain stored;
absent coordinates are not emitted. feature_count is an optional caller-declared
dimension; otherwise only the maximum observed index is reported.

Numeric tokens use decimal notation with optional e/E exponent, at most 128
characters and three exponent digits. Their binary64 conversion must be finite
and must not underflow a nonzero token to zero. Original tokens are returned as
exact decimal strings, retaining signs, precision and exponent spelling without
claiming exact binary64 representability. Labels carry no inferred task meaning.

LF/CRLF are supported; bare CR, NUL and non-ASCII bytes fail even in comments.
Empty/whitespace-only lines and hash comments have explicit policies. An enabled
comment starts at column zero or after space/tab; qid remains unsupported. Only
space and tab separate tokens. No header is inferred. The full source is parsed
before returning independently paged row-label summaries and sparse coordinates.

Limits: 8 MB, 100000 physical lines, 65536 bytes/line, 50000 rows, 10000 stored
features/row, 250000 total coordinates, feature indices through 1e9, 200 row
summaries and 2000 coordinates/page. Reference: cjlin1/libsvm README data format.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?\Z")
_INDEX = re.compile(r"[1-9][0-9]{0,9}\Z")


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    feature_count: int | None = Field(default=None, strict=True, ge=1, le=1_000_000_000)
    comments: Literal["reject", "hash_after_whitespace"] = "reject"
    blank_lines: Literal["reject", "skip"] = "reject"
    row_offset: int = Field(default=0, strict=True, ge=0, le=50_000)
    row_limit: int = Field(default=50, strict=True, ge=1, le=200)
    coordinate_offset: int = Field(default=0, strict=True, ge=0, le=250_000)
    coordinate_limit: int = Field(default=500, strict=True, ge=1, le=2_000)


class Row(OutputModel):
    row_index: int
    source_line: int
    source_byte_start: int
    label: str
    stored_features: int
    explicit_zero_features: int
    coordinate_start: int


class Coordinate(OutputModel):
    coordinate_index: int
    row_index: int
    source_line: int
    feature_index: int
    value: str


class Output(OutputModel):
    row_count: int
    has_rows: bool
    coordinate_count: int
    explicit_zero_coordinates: int
    nonzero_coordinates: int
    declared_feature_count: int | None
    maximum_observed_feature_index: int
    physical_line_count: int
    comment_lines: int
    inline_comments: int
    skipped_blank_lines: int
    rows: list[Row]
    row_offset: int
    has_more_rows: bool
    coordinates: list[Coordinate]
    coordinate_offset: int
    has_more_coordinates: bool
    feature_index_base: Literal[1] = 1
    numeric_encoding: Literal["original_exact_decimal_tokens"] = "original_exact_decimal_tokens"
    label_semantics: Literal["not_inferred"] = "not_inferred"
    dense_array_allocated: Literal[False] = False
    full_source_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


def _number(token: str) -> bool:
    match = _NUMBER.fullmatch(token) if len(token) <= 128 else None
    if match is None:
        raise ValueError("LIBSVM values require bounded decimal numeric tokens")
    exponent = match.group(1)
    if exponent is not None and len(exponent.lstrip("+-")) > 3:
        raise ValueError("LIBSVM numeric exponent spelling exceeds three digits")
    mantissa = re.split("[eE]", token, maxsplit=1)[0]
    is_zero = not any(character in "123456789" for character in mantissa)
    converted = float(token)
    if not math.isfinite(converted) or (converted == 0 and not is_zero):
        raise ValueError("LIBSVM numeric token overflows or nonzero token underflows binary64")
    return is_zero


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(
        request.path, suffixes=(".libsvm", ".svm", ".svmlight", ".txt"), max_bytes=8_000_000
    )
    if b"\x00" in content or content.count(b"\n") > 100_000:
        raise ValueError("LIBSVM source contains NUL or exceeds 100000 physical lines")
    lines = content.split(b"\n") if content else []
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) > 100_000:
        raise ValueError("LIBSVM source exceeds 100000 physical lines")
    rows: list[Row] = []
    coordinates: list[Coordinate] = []
    row_count = coordinate_count = zero_count = maximum_index = 0
    byte_offset = comments = inline_comments = blanks = 0
    for line_index, raw in enumerate(lines):
        has_lf = line_index + 1 < len(lines) or content.endswith(b"\n")
        record = raw[:-1] if has_lf and raw.endswith(b"\r") else raw
        start = byte_offset
        byte_offset += len(raw) + int(has_lf)
        if len(record) > 65_536 or b"\r" in record:
            raise ValueError("LIBSVM line exceeds 65536 bytes or contains a bare CR")
        try:
            line = record.decode("ascii")
        except UnicodeError as exc:
            raise ValueError("LIBSVM source must be ASCII, including comments") from exc
        if any(ord(character) < 32 and character != "\t" for character in line):
            raise ValueError("LIBSVM records support only space/tab token separation")
        hash_position = line.find("#")
        if hash_position >= 0:
            if request.comments == "reject" or (
                hash_position and line[hash_position - 1] not in " \t"
            ):
                raise ValueError("LIBSVM hash comment is disabled or not preceded by whitespace")
            prefix = line[:hash_position]
            if not prefix.strip(" \t"):
                comments += 1
                continue
            inline_comments += 1
            line = prefix
        if not line.strip(" \t"):
            if request.blank_lines == "reject":
                raise ValueError("empty LIBSVM records are not allowed")
            blanks += 1
            continue
        if row_count >= 50_000:
            raise ValueError("LIBSVM source exceeds 50000 rows")
        tokens = re.split(r"[ \t]+", line.strip(" \t"))
        if len(tokens) - 1 > 10_000:
            raise ValueError("LIBSVM row exceeds 10000 stored features")
        label = tokens[0]
        _number(label)
        previous_index = row_zeros = 0
        coordinate_start = coordinate_count
        for token in tokens[1:]:
            pieces = token.split(":")
            if len(pieces) != 2 or _INDEX.fullmatch(pieces[0]) is None:
                raise ValueError("LIBSVM features require a positive decimal index:value pair")
            index = int(pieces[0])
            if index <= previous_index or index > (request.feature_count or 1_000_000_000):
                raise ValueError(
                    "LIBSVM indices must increase strictly within the declared/maximum dimension"
                )
            if coordinate_count >= 250_000:
                raise ValueError("LIBSVM source exceeds 250000 coordinates")
            is_zero = _number(pieces[1])
            row_zeros += is_zero
            zero_count += is_zero
            if (
                request.coordinate_offset
                <= coordinate_count
                < request.coordinate_offset + request.coordinate_limit
            ):
                coordinates.append(
                    Coordinate(
                        coordinate_index=coordinate_count,
                        row_index=row_count,
                        source_line=line_index + 1,
                        feature_index=index,
                        value=pieces[1],
                    )
                )
            coordinate_count += 1
            previous_index = index
        maximum_index = max(maximum_index, previous_index)
        if request.row_offset <= row_count < request.row_offset + request.row_limit:
            rows.append(
                Row(
                    row_index=row_count,
                    source_line=line_index + 1,
                    source_byte_start=start,
                    label=label,
                    stored_features=len(tokens) - 1,
                    explicit_zero_features=row_zeros,
                    coordinate_start=coordinate_start,
                )
            )
        row_count += 1
    return Output(
        row_count=row_count,
        has_rows=row_count > 0,
        coordinate_count=coordinate_count,
        explicit_zero_coordinates=zero_count,
        nonzero_coordinates=coordinate_count - zero_count,
        declared_feature_count=request.feature_count,
        maximum_observed_feature_index=maximum_index,
        physical_line_count=len(lines),
        comment_lines=comments,
        inline_comments=inline_comments,
        skipped_blank_lines=blanks,
        rows=rows,
        row_offset=request.row_offset,
        has_more_rows=request.row_offset + len(rows) < row_count,
        coordinates=coordinates,
        coordinate_offset=request.coordinate_offset,
        has_more_coordinates=request.coordinate_offset + len(coordinates) < coordinate_count,
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_libsvm",
    kind="plugin",
    description="Parse a bounded strict LIBSVM sparse subset with exact numeric-token encodings, increasing positive indices, explicit zeros and independent row/coordinate pages; validate the full source without dense allocation.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
