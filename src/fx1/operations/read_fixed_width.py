"""Parse exact-width UTF-8 records using Unicode code-point column offsets.

Spans are zero-based [start, end), count code points rather than bytes, graphemes
or screen columns, and must not overlap. Every data/header record has exactly
record_width code points. Cells remain exact strings: no trimming, normalization,
padding, inferred types or inferred names. Gaps between spans are ignored.

LF and CRLF may coexist; bare CR, NUL, an initial UTF-8 BOM and invalid UTF-8
anywhere are rejected. All physical lines receive encoding/length checks,
including skipped/comment lines. Classification order is physical skip_lines,
an exact prefix at column zero, then empty-line policy. Only an empty string is
a blank line; spaces remain record content. first_record consumes the first
remaining record as an observed header using the supplied spans, without deriving
or matching column names. The entire source is validated before a page returns.

Limits: 4 MB source, 100000 physical lines, 16384 code points per line, 50000 data
records, 64 columns, 2M total parsed data cells, 200 returned rows/10000 page
cells and 512 KB encoded row/header output. Source lineage uses one-based
physical lines and exact byte offsets; the hash covers original line endings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Column(InputModel):
    name: str = Field(strict=True, min_length=1, max_length=64)
    start: int = Field(strict=True, ge=0, le=16_383)
    end: int = Field(strict=True, ge=1, le=16_384)

    @field_validator("name")
    @classmethod
    def unicode_name(cls, value: str) -> str:
        value.encode("utf-8")
        if "\x00" in value:
            raise ValueError("column names cannot contain NUL")
        return value


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    columns: list[Column] = Field(min_length=1, max_length=64)
    record_width: int = Field(strict=True, ge=1, le=16_384)
    skip_lines: int = Field(default=0, strict=True, ge=0, le=1_000)
    comment_prefix: str | None = Field(default=None, min_length=1, max_length=32)
    blank_lines: Literal["reject", "skip"] = "reject"
    header: Literal["none", "first_record"] = "none"
    offset: int = Field(default=0, strict=True, ge=0, le=50_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def spans_and_work(self) -> Self:
        if len({column.name for column in self.columns}) != len(self.columns):
            raise ValueError("fixed-width column names must be unique")
        previous_end = 0
        for column in sorted(self.columns, key=lambda column: column.start):
            if (
                column.end <= column.start
                or column.end > self.record_width
                or column.start < previous_end
            ):
                raise ValueError(
                    "column spans must be nonempty, nonoverlapping and within record_width"
                )
            previous_end = column.end
        if len(self.columns) * self.limit > 10_000:
            raise ValueError("fixed-width output page exceeds 10000 cells")
        if self.comment_prefix is not None:
            self.comment_prefix.encode("utf-8")
            if any(character in self.comment_prefix for character in "\r\n\x00"):
                raise ValueError("comment_prefix cannot contain line endings or NUL")
        return self


class Row(OutputModel):
    row_index: int
    source_line: int
    source_byte_start: int
    source_record_bytes: int
    cells: list[str]


class Output(OutputModel):
    columns: list[str]
    column_spans: list[list[int]]
    position_convention: Literal["zero_based_half_open_unicode_code_points"] = (
        "zero_based_half_open_unicode_code_points"
    )
    record_width: int
    physical_line_count: int
    skipped_lines: int
    comment_lines: int
    skipped_blank_lines: int
    header_source_line: int | None
    header_cells: list[str] | None
    data_row_count: int
    has_data_rows: bool
    parsed_data_cells: int
    rows: list[Row]
    offset: int
    has_more: bool
    full_source_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(
        request.path, suffixes=(".txt", ".fwf", ".dat"), max_bytes=4_000_000
    )
    if content.startswith(b"\xef\xbb\xbf") or b"\x00" in content:
        raise ValueError("fixed-width source cannot contain an initial UTF-8 BOM or NUL")
    if content.count(b"\n") > 100_000:
        raise ValueError("fixed-width source exceeds 100000 physical lines")
    lines = content.split(b"\n") if content else []
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) > 100_000 or request.skip_lines > len(lines):
        raise ValueError(
            "source exceeds its physical-line bound or skip_lines exceeds source length"
        )
    byte_offset = row_count = comment_count = blank_count = page_bytes = 0
    header_line: int | None = None
    header_cells: list[str] | None = None
    rows: list[Row] = []
    for index, raw in enumerate(lines):
        has_lf = index + 1 < len(lines) or content.endswith(b"\n")
        record = raw[:-1] if has_lf and raw.endswith(b"\r") else raw
        start = byte_offset
        byte_offset += len(raw) + int(has_lf)
        if b"\r" in record:
            raise ValueError("fixed-width source contains a bare CR")
        try:
            line = record.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"invalid UTF-8 on physical line {index + 1}") from exc
        if len(line) > 16_384:
            raise ValueError("fixed-width physical line exceeds 16384 code points")
        if index < request.skip_lines:
            continue
        if request.comment_prefix is not None and line.startswith(request.comment_prefix):
            comment_count += 1
            continue
        if not line:
            if request.blank_lines == "reject":
                raise ValueError(f"empty fixed-width line {index + 1} is not allowed")
            blank_count += 1
            continue
        if len(line) != request.record_width:
            raise ValueError(
                f"fixed-width line {index + 1} has {len(line)} code points; expected {request.record_width}"
            )
        if request.header == "first_record" and header_line is None:
            header_line = index + 1
            header_cells = [line[column.start : column.end] for column in request.columns]
            page_bytes += len(json.dumps(header_cells, ensure_ascii=False).encode("utf-8"))
        else:
            if row_count >= 50_000 or (row_count + 1) * len(request.columns) > 2_000_000:
                raise ValueError("fixed-width source exceeds 50000 records or 2M parsed cells")
            if request.offset <= row_count < request.offset + request.limit:
                row = Row(
                    row_index=row_count,
                    source_line=index + 1,
                    source_byte_start=start,
                    source_record_bytes=len(record),
                    cells=[line[column.start : column.end] for column in request.columns],
                )
                page_bytes += len(json.dumps(row.model_dump(), ensure_ascii=False).encode("utf-8"))
                rows.append(row)
            row_count += 1
        if page_bytes > 512_000:
            raise ValueError("fixed-width encoded page/header output exceeds 512 KB")
    if request.header == "first_record" and header_line is None:
        raise ValueError("fixed-width source has no eligible header record")
    return Output(
        columns=[column.name for column in request.columns],
        column_spans=[[column.start, column.end] for column in request.columns],
        record_width=request.record_width,
        physical_line_count=len(lines),
        skipped_lines=request.skip_lines,
        comment_lines=comment_count,
        skipped_blank_lines=blank_count,
        header_source_line=header_line,
        header_cells=header_cells,
        data_row_count=row_count,
        has_data_rows=row_count > 0,
        parsed_data_cells=row_count * len(request.columns),
        rows=rows,
        offset=request.offset,
        has_more=request.offset + len(rows) < row_count,
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_fixed_width",
    kind="plugin",
    description="Read exact-width UTF-8 records through explicit nonoverlapping Unicode code-point spans, retaining exact cells and physical-line/byte lineage while validating the full bounded source.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
