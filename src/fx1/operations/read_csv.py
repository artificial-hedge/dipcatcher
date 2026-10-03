"""Stream a bounded workspace CSV as exact strings with explicit dialect settings.

The complete file is parsed before a page is returned, so malformed rows
outside the requested page cannot be mistaken for a valid dataset. This reader
does not infer numeric types, timestamps, availability, or a price basis.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator

from pydantic import Field, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @field_validator("delimiter")
    @classmethod
    def valid_delimiter(cls, value: str) -> str:
        if value in ("\r", "\n", "\0", '"'):
            raise ValueError("delimiter cannot be a newline, NUL, or quote")
        return value


class Output(OutputModel):
    columns: list[str]
    rows: list[dict[str, str]]
    total_rows: int
    offset: int
    returned_rows: int
    has_more: bool
    blank_records_skipped: int
    source_sha256: str
    source_bytes: int


def execute(request: Input, context: OperationContext) -> Output:
    """Decode UTF-8, validate headers and record widths, then return a page."""
    rows: list[dict[str, str]] = []
    total_rows = 0
    blank_records = 0
    page_cell_characters = 0

    def bounded_lines(text: io.TextIOWrapper) -> Iterator[str]:
        while line := text.readline(1_000_001):
            if len(line) > 1_000_000:
                raise ValueError("CSV physical lines cannot exceed 1000000 characters")
            if "\0" in line:
                raise ValueError("CSV input cannot contain NUL characters")
            yield line

    with context.open_binary(
        request.path, suffixes=(".csv", ".tsv"), max_bytes=64_000_000
    ) as source:
        buffered = io.BufferedReader(source)
        with io.TextIOWrapper(buffered, encoding="utf-8-sig", newline="") as text:
            reader = csv.reader(bounded_lines(text), delimiter=request.delimiter, strict=True)
            try:
                columns = next(reader, None)
                if not columns or any(not name.strip() for name in columns):
                    raise ValueError("CSV must begin with a nonempty header with no blank names")
                if len(columns) > 256 or any(len(name) > 256 for name in columns):
                    raise ValueError("CSV header exceeds 256 columns or a 256-character name")
                if len(set(columns)) != len(columns):
                    raise ValueError("CSV header contains duplicate column names")
                for record in reader:
                    if not record:
                        blank_records += 1
                        continue
                    if len(record) != len(columns):
                        raise ValueError(
                            f"CSV record ending on line {reader.line_num} has {len(record)} cells; "
                            f"expected {len(columns)}"
                        )
                    if total_rows >= 100_000:
                        raise ValueError("CSV input exceeds 100000 data records")
                    if request.offset <= total_rows < request.offset + request.limit:
                        page_cell_characters += sum(map(len, record)) + sum(map(len, columns))
                        if page_cell_characters > 1_000_000:
                            raise ValueError(
                                "CSV page exceeds 1000000 cell/key characters; reduce limit"
                            )
                        rows.append(dict(zip(columns, record, strict=True)))
                    total_rows += 1
            except UnicodeDecodeError as exc:
                raise ValueError("CSV input must be UTF-8 encoded") from exc
            except csv.Error as exc:
                raise ValueError(
                    f"invalid CSV near physical line {reader.line_num}: {exc}"
                ) from exc
            source_sha256, source_bytes = source.source_sha256, source.bytes_read
    return Output(
        columns=columns,
        rows=rows,
        total_rows=total_rows,
        offset=request.offset,
        returned_rows=len(rows),
        has_more=request.offset + len(rows) < total_rows,
        blank_records_skipped=blank_records,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
    )


OPERATION = Operation(
    id="plugins.read_csv",
    kind="plugin",
    description=(
        "Read a UTF-8 workspace CSV/TSV as strings with strict header and row-width checks; "
        "explicit delimiter, streaming validation, pagination, 64 MB and 100000-record limits, "
        "and a hash of every source byte."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
    version="1.1.0",
)
