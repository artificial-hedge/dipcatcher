"""Inspect one bounded SQLite database image using a private in-memory connection.

The contained source is read once, hashed, then passed to sqlite3.deserialize;
SQLite never receives its filesystem path. Only a fixed schema query, one
table_xinfo pragma and a quoted ordinary-table projection are permitted.
Views, virtual/shadow tables, internal tables, generated columns, attached
databases, extensions and arbitrary SQL cannot supply preview rows. Schema
objects of other kinds may be described without executing them. Table/column
selection uses exact stored names. No type converters or user functions are
registered. A table scan is explicitly unordered and is not a portable row ID.

Sources must have rollback-format headers: both journal version bytes equal 1,
valid page size, exact current page count and matching header change counters.
Standalone status is a caller-supplied assumption. These checks do not establish
that no hot rollback journal exists or that the image is committed or recovered;
a main-file image may contain changes that normal recovery would undo. Sidecars
are never checked, read or modified. WAL databases are unsupported. Reading a
live file does not establish an atomic snapshot; schema/preview success does not
certify overall database integrity. SQLite 3.37+ and Python's deserialize and
setlimit APIs are required. Local availability is checked before file parsing.

Limits: 8 MB source, 16384 pages, 128 schema objects/128 KB encoded descriptions,
64 columns/128 KB encoded column descriptions, 200 output rows, offset 100000,
10000 output cells, 4096 UTF-8/blob
bytes per value and 512 KB encoded cell data. The native row/value limit is
262144 bytes, SQL length 65536, expression depth 32 and VDBE program size 25000.
A progress handler interrupts after at most 1000 callbacks scheduled approximately
every 1000 VM instructions. Per-statement scheduling is not an exact aggregate
instruction, process-memory or wall-clock cap. Cache/temp storage stays
in memory. Integers are decimal strings, blobs hex, reals finite JSON numbers,
and invalid UTF-8 or oversized values fail rather than being truncated.

References: Python sqlite3 deserialize/setlimit/set_authorizer documentation;
sqlite.org/security.html, fileformat2.html, pragma.html and c3ref/deserialize.html.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from typing import Annotated, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    table: Name | None = None
    columns: list[Name] | None = Field(default=None, min_length=1, max_length=64)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)
    max_progress_callbacks: int = Field(default=500, strict=True, ge=1, le=1_000)

    @field_validator("table")
    @classmethod
    def table_name(cls, value: str | None) -> str | None:
        if value is not None:
            _text(value, "table name", 256)
        return value

    @model_validator(mode="after")
    def column_names(self) -> Self:
        if self.columns is not None:
            if self.table is None:
                raise ValueError("column selection requires a selected table")
            if len(set(self.columns)) != len(self.columns):
                raise ValueError("column selection must be distinct")
            if len(self.columns) * self.limit > 10_000:
                raise ValueError("SQLite preview exceeds 10000 requested output cells")
            for name in self.columns:
                _text(name, "column name", 256)
        return self


class SchemaObject(OutputModel):
    type: Literal["table", "index", "view", "trigger"]
    name: str
    table_name: str
    root_page: int
    declared_sql: str | None
    ordinary_table_candidate: bool


class Column(OutputModel):
    index: int
    name: str
    declared_type: str
    not_null: bool
    default_expression: str | None
    primary_key_position: int


class NullCell(OutputModel):
    storage_class: Literal["null"] = "null"
    value: None = None


class IntegerCell(OutputModel):
    storage_class: Literal["integer"] = "integer"
    value: str
    encoding: Literal["decimal_string"] = "decimal_string"


class RealCell(OutputModel):
    storage_class: Literal["real"] = "real"
    value: float = Field(strict=True, allow_inf_nan=False)


class TextCell(OutputModel):
    storage_class: Literal["text"] = "text"
    value: str


class BlobCell(OutputModel):
    storage_class: Literal["blob"] = "blob"
    value: str
    encoding: Literal["hex"] = "hex"


Cell = Annotated[
    NullCell | IntegerCell | RealCell | TextCell | BlobCell, Field(discriminator="storage_class")
]


class Output(OutputModel):
    schema_objects: list[SchemaObject]
    schema_object_count: int
    ordinary_table_candidates: int
    selected_table: str | None
    table_columns: list[Column]
    selected_columns: list[str]
    rows: list[list[Cell]]
    returned_rows: int
    offset: int
    has_more: bool | None
    total_table_rows: None = None
    row_order: Literal["unordered_sqlite_table_scan"] = "unordered_sqlite_table_scan"
    page_size: int
    page_count: int
    text_encoding: Literal["UTF-8", "UTF-16le", "UTF-16be"]
    query_only: Literal[True] = True
    trusted_schema: Literal[False] = False
    authorizer_enforced: Literal[True] = True
    defensive_mode: bool
    progress_callbacks: int
    progress_callback_limit: int
    progress_instruction_interval: Literal[1000] = 1000
    sqlite_version: str
    sidecars_checked: Literal[False] = False
    rollback_journal_included: Literal[False] = False
    committed_state_verified: Literal[False] = False
    wal_sidecars_included: Literal[False] = False
    source_snapshot_guaranteed: Literal[False] = False
    full_integrity_verified: Literal[False] = False
    source_bytes: int
    source_sha256: str


def _text(value: object, label: str, maximum: int, *, nul_allowed: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"SQLite {label} must be text")
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError(f"SQLite {label} contains invalid Unicode") from exc
    if length > maximum or (not nul_allowed and "\x00" in value):
        raise ValueError(f"SQLite {label} exceeds its byte limit or contains NUL")
    return value


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _ordinary_declaration(sql: str | None) -> bool:
    if sql is None:
        return False
    # Lex only the declaration's first two words. Comments may occur between
    # them; quoted identifiers and the remainder are left to SQLite's parser.
    cursor = 0
    words: list[str] = []
    while len(words) < 2:
        while cursor < len(sql) and sql[cursor].isspace():
            cursor += 1
        if sql.startswith("--", cursor):
            end = sql.find("\n", cursor + 2)
            if end < 0:
                return False
            cursor = end + 1
        elif sql.startswith("/*", cursor):
            end = sql.find("*/", cursor + 2)
            if end < 0:
                return False
            cursor = end + 2
        else:
            word = re.match(r"[A-Za-z]+", sql[cursor:])
            if word is None:
                return False
            words.append(word.group().upper())
            cursor += len(word.group())
    return words == ["CREATE", "TABLE"]


def _cell(value: object) -> NullCell | IntegerCell | RealCell | TextCell | BlobCell:
    if value is None:
        return NullCell()
    if type(value) is int:
        return IntegerCell(value=str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("SQLite preview contains a nonfinite REAL")
        return RealCell(value=value)
    if isinstance(value, str):
        return TextCell(value=_text(value, "TEXT value", 4096, nul_allowed=True))
    if isinstance(value, bytes):
        if len(value) > 4096:
            raise ValueError("SQLite preview BLOB exceeds 4096 bytes")
        return BlobCell(value=value.hex())
    raise ValueError("SQLite returned an unsupported storage class")


def execute(request: Input, context: OperationContext) -> Output:
    if not hasattr(sqlite3.Connection, "deserialize") or not hasattr(
        sqlite3.Connection, "setlimit"
    ):
        raise ValueError("this Python SQLite build lacks required deserialize/setlimit APIs")
    if sqlite3.sqlite_version_info < (3, 37, 0):
        raise ValueError("SQLite inspection requires SQLite 3.37 or newer")
    content = context.read_bytes(
        request.path, suffixes=(".db", ".sqlite", ".sqlite3"), max_bytes=8_000_000
    )
    if len(content) < 100 or content[:16] != b"SQLite format 3\x00":
        raise ValueError("source is not a SQLite format-3 database image")
    page_size = int.from_bytes(content[16:18], "big")
    page_size = 65536 if page_size == 1 else page_size
    if not 512 <= page_size <= 65536 or page_size & (page_size - 1) or len(content) % page_size:
        raise ValueError("SQLite page size or complete-page file length is invalid")
    page_count = len(content) // page_size
    if page_count > 16_384 or int.from_bytes(content[28:32], "big") != page_count:
        raise ValueError("SQLite header page count must exactly match at most 16384 source pages")
    if content[18:20] != b"\x01\x01" or content[24:28] != content[92:96]:
        raise ValueError(
            "SQLite requires matching change counters and rollback-format bytes; WAL/legacy images are unsupported"
        )
    encoding = int.from_bytes(content[56:60], "big")
    if encoding not in (1, 2, 3):
        raise ValueError("SQLite header must declare a supported text encoding")
    schema: list[SchemaObject] = []
    columns: list[Column] = []
    selected: list[str] = []
    rows: list[list[Cell]] = []
    has_more: bool | None = None
    progress_calls = 0
    defensive = False
    allowed_tables = {"sqlite_master", "sqlite_schema"}
    pragma_table: str | None = None

    def progress() -> int:
        nonlocal progress_calls
        progress_calls += 1
        return int(progress_calls >= request.max_progress_callbacks)

    def authorize(
        action: int, first: str | None, second: str | None, database: str | None, source: str | None
    ) -> int:
        if source is not None:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and database == "main" and first in allowed_tables:
            return sqlite3.SQLITE_OK
        if (
            action == sqlite3.SQLITE_PRAGMA
            and first == "table_xinfo"
            and pragma_table is not None
            and second == pragma_table
            and database in (None, "main")
        ):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    try:
        with closing(
            sqlite3.connect(
                ":memory:", isolation_level=None, detect_types=0, cached_statements=0, timeout=0
            )
        ) as connection:
            connection.set_progress_handler(progress, 1000)
            limits = {
                sqlite3.SQLITE_LIMIT_LENGTH: 262_144,
                sqlite3.SQLITE_LIMIT_SQL_LENGTH: 65_536,
                sqlite3.SQLITE_LIMIT_COLUMN: 64,
                sqlite3.SQLITE_LIMIT_EXPR_DEPTH: 32,
                sqlite3.SQLITE_LIMIT_VDBE_OP: 25_000,
                sqlite3.SQLITE_LIMIT_ATTACHED: 0,
                sqlite3.SQLITE_LIMIT_COMPOUND_SELECT: 1,
                sqlite3.SQLITE_LIMIT_FUNCTION_ARG: 16,
                sqlite3.SQLITE_LIMIT_TRIGGER_DEPTH: 0,
                sqlite3.SQLITE_LIMIT_WORKER_THREADS: 0,
            }
            for category, bound in limits.items():
                connection.setlimit(category, bound)
            if hasattr(connection, "setconfig"):
                for option, enabled in (
                    ("DEFENSIVE", True),
                    ("TRUSTED_SCHEMA", False),
                    ("ENABLE_LOAD_EXTENSION", False),
                    ("ENABLE_VIEW", False),
                    ("ENABLE_TRIGGER", False),
                    ("DQS_DDL", False),
                    ("DQS_DML", False),
                ):
                    code = getattr(sqlite3, "SQLITE_DBCONFIG_" + option, None)
                    if code is not None:
                        connection.setconfig(code, enabled)
                        if option == "DEFENSIVE":
                            defensive = True
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA query_only=ON")
            connection.deserialize(content)
            if connection.execute("PRAGMA query_only").fetchone() != (1,) or connection.execute(
                "PRAGMA trusted_schema"
            ).fetchone() != (0,):
                raise ValueError("SQLite failed to retain query_only/trusted_schema restrictions")
            connection.set_authorizer(authorize)
            schema_bytes = 0
            cursor = connection.execute(
                "SELECT type,name,tbl_name,rootpage,sql FROM main.sqlite_schema LIMIT 129"
            )
            for row in cursor:
                if len(schema) == 128:
                    raise ValueError("SQLite schema exceeds 128 objects")
                kind = _text(row[0], "schema object type", 16)
                if kind not in ("table", "index", "view", "trigger"):
                    raise ValueError("SQLite schema contains an unsupported object type")
                name = _text(row[1], "schema object name", 256)
                table_name = _text(row[2], "schema table name", 256)
                root_page = row[3]
                sql = None if row[4] is None else _text(row[4], "schema SQL", 16_384)
                if type(root_page) is not int or not 0 <= root_page <= page_count:
                    raise ValueError("SQLite schema has an invalid root page")
                item = SchemaObject(
                    type=cast(Literal["table", "index", "view", "trigger"], kind),
                    name=name,
                    table_name=table_name,
                    root_page=root_page,
                    declared_sql=sql,
                    ordinary_table_candidate=(
                        kind == "table"
                        and root_page > 0
                        and not name.lower().startswith("sqlite_")
                        and _ordinary_declaration(sql)
                    ),
                )
                schema_bytes += len(
                    json.dumps(item.model_dump(), ensure_ascii=False).encode("utf-8")
                )
                if schema_bytes > 128_000:
                    raise ValueError("SQLite encoded schema descriptions exceed 128 KB")
                schema.append(item)
            if request.table is not None:
                match = next((item for item in schema if item.name == request.table), None)
                if match is None or not match.ordinary_table_candidate:
                    raise ValueError(
                        "selected SQLite name must be an ordinary user table with a stored root page"
                    )
                # A virtual table's shadow tables have ordinary CREATE TABLE
                # declarations. Reject any database containing virtual tables
                # before previewing, avoiding module-specific shadow guessing.
                if any(item.type == "table" and item.root_page == 0 for item in schema):
                    raise ValueError(
                        "table preview is unsupported for databases containing virtual tables"
                    )
                pragma_table = request.table
                info = connection.execute("PRAGMA main.table_xinfo(" + _quoted(request.table) + ")")
                column_bytes = 0
                for row in info:
                    if len(columns) == 64 or row[6] != 0:
                        raise ValueError(
                            "SQLite selected table exceeds 64 columns or has generated/hidden columns"
                        )
                    column = Column(
                        index=row[0],
                        name=_text(row[1], "column name", 256),
                        declared_type=_text(row[2], "declared column type", 512),
                        not_null=bool(row[3]),
                        default_expression=None
                        if row[4] is None
                        else _text(row[4], "column default", 4096),
                        primary_key_position=row[5],
                    )
                    column_bytes += len(
                        json.dumps(column.model_dump(), ensure_ascii=False).encode("utf-8")
                    )
                    if column_bytes > 128_000:
                        raise ValueError("SQLite encoded column descriptions exceed 128 KB")
                    columns.append(column)
                pragma_table = None
                names = [column.name for column in columns]
                if not names:
                    raise ValueError("SQLite selected table has no readable columns")
                selected = names if request.columns is None else request.columns
                if (
                    any(name not in names for name in selected)
                    or len(selected) * request.limit > 10_000
                ):
                    raise ValueError(
                        "SQLite selection includes unknown columns or exceeds 10000 output cells"
                    )
                allowed_tables.add(request.table)
                sql = (
                    "SELECT "
                    + ",".join(_quoted(name) for name in selected)
                    + " FROM main."
                    + _quoted(request.table)
                    + " NOT INDEXED LIMIT ? OFFSET ?"
                )
                preview = connection.execute(sql, (request.limit + 1, request.offset))
                value_bytes = 0
                has_more = False
                for raw_row in preview:
                    if len(rows) == request.limit:
                        has_more = True
                        break
                    cells = [_cell(value) for value in raw_row]
                    value_bytes += sum(
                        len(
                            json.dumps(
                                cell.model_dump(), ensure_ascii=False, allow_nan=False
                            ).encode("utf-8")
                        )
                        for cell in cells
                    )
                    if value_bytes > 512_000:
                        raise ValueError("SQLite encoded preview values exceed 512 KB")
                    rows.append(cells)
    except (sqlite3.Error, MemoryError) as exc:
        raise ValueError(
            "SQLite rejected the database or exceeded bounded read/query limits: " + str(exc)[:500]
        ) from exc
    return Output(
        schema_objects=schema,
        schema_object_count=len(schema),
        ordinary_table_candidates=sum(item.ordinary_table_candidate for item in schema),
        selected_table=request.table,
        table_columns=columns,
        selected_columns=selected,
        rows=rows,
        returned_rows=len(rows),
        offset=request.offset,
        has_more=has_more,
        page_size=page_size,
        page_count=page_count,
        text_encoding=cast(
            Literal["UTF-8", "UTF-16le", "UTF-16be"],
            ("UTF-8", "UTF-16le", "UTF-16be")[encoding - 1],
        ),
        defensive_mode=defensive,
        progress_callbacks=progress_calls,
        progress_callback_limit=request.max_progress_callbacks,
        sqlite_version=sqlite3.sqlite_version,
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.inspect_sqlite",
    kind="plugin",
    description="Inspect bounded caller-asserted standalone SQLite images via in-memory deserialize and restricted ordinary-table queries; preserve integer/blob encodings and disclose unchecked sidecars, unverified committed/recovered state and incomplete integrity checks.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
