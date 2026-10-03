"""Inspect bounded workspace Parquet metadata without materializing data columns."""

from __future__ import annotations

import hashlib

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    row_group_offset: int = Field(default=0, strict=True, ge=0)
    row_group_limit: int = Field(default=20, strict=True, ge=1, le=100)


class Column(OutputModel):
    name: str
    logical_type: str
    nullable: bool


class RowGroup(OutputModel):
    index: int
    rows: int
    compressed_bytes: int
    uncompressed_bytes: int
    compression_codecs: list[str]


class Output(OutputModel):
    columns: list[Column]
    leaf_column_count: int
    total_rows: int
    row_group_count: int
    row_groups: list[RowGroup]
    row_group_offset: int
    has_more_row_groups: bool
    format_version: str
    source_sha256: str
    source_bytes: int


def execute(request: Input, context: OperationContext) -> Output:
    """Parse footer and Arrow schema from the exact bytes fingerprinted in output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    content = context.read_bytes(request.path, suffixes=(".parquet",), max_bytes=8_000_000)
    try:
        parquet = pq.ParquetFile(
            pa.BufferReader(content),
            thrift_string_size_limit=1_000_000,
            thrift_container_size_limit=100_000,
        )
        metadata = parquet.metadata
        schema = parquet.schema_arrow
        if len(schema) > 256 or metadata.num_columns > 1024:
            raise ValueError("Parquet schema exceeds 256 top-level or 1024 leaf columns")
        if metadata.num_row_groups > 10_000:
            raise ValueError("Parquet metadata exceeds 10000 row groups")
        columns = [
            Column(name=field.name, logical_type=str(field.type), nullable=field.nullable)
            for field in schema
        ]
        row_groups: list[RowGroup] = []
        stop = min(metadata.num_row_groups, request.row_group_offset + request.row_group_limit)
        for index in range(request.row_group_offset, stop):
            group = metadata.row_group(index)
            compressed = 0
            uncompressed = 0
            codecs: set[str] = set()
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                compressed += column.total_compressed_size
                uncompressed += column.total_uncompressed_size
                codecs.add(column.compression)
            row_groups.append(
                RowGroup(
                    index=index,
                    rows=group.num_rows,
                    compressed_bytes=compressed,
                    uncompressed_bytes=uncompressed,
                    compression_codecs=sorted(codecs),
                )
            )
        return Output(
            columns=columns,
            leaf_column_count=metadata.num_columns,
            total_rows=metadata.num_rows,
            row_group_count=metadata.num_row_groups,
            row_groups=row_groups,
            row_group_offset=request.row_group_offset,
            has_more_row_groups=stop < metadata.num_row_groups,
            format_version=str(metadata.format_version),
            source_sha256=hashlib.sha256(content).hexdigest(),
            source_bytes=len(content),
        )
    except (pa.ArrowException, OSError) as exc:
        raise ValueError(f"invalid or unsupported Parquet metadata: {exc}") from exc


OPERATION = Operation(
    id="plugins.inspect_parquet",
    kind="plugin",
    description=(
        "Inspect a workspace Parquet schema, row counts, compression and paginated row-group "
        "sizes without decoding data columns; 8 MB source limit and source byte hash."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
