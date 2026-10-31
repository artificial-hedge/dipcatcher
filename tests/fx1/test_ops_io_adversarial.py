"""SYNTHETIC adversarial fixtures for the workspace IO/inspection cluster.

Every byte pattern below is hand-built and deterministic: no network, clock, or
randomness. These probes pin the fail-closed contracts of
``src/fx1/operations/{base,registry,read_jsonl,read_csv,read_toml,inspect_zip,
inspect_numpy_array,inspect_parquet,verify_file_hash}.py`` — path containment,
byte/shape budgets, injection surfaces, hash-verification without TOCTOU, and
registry binding integrity.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from fx1.operations import registry
from fx1.operations.base import (
    InputModel,
    Operation,
    OperationContext,
    OutputModel,
    canonical_json,
)
from fx1.operations.registry import execute_operation, get_operation, invoke_operation_tool

JSONL = "plugins.read_jsonl"
CSV = "plugins.read_csv"
TOML = "plugins.read_toml"
ZIP = "plugins.inspect_zip"
NPY = "plugins.inspect_numpy_array"
PARQUET = "plugins.inspect_parquet"
HASH = "plugins.verify_file_hash"

HEX64 = hashlib.sha256(b"").hexdigest()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def _write(workspace: Path, name: str, data: bytes) -> str:
    (workspace / name).write_bytes(data)
    return name


def _result(operation_id: str, arguments: dict, workspace: Path) -> dict:
    return execute_operation(operation_id, arguments, workspace_root=workspace)["result"]


# --------------------------------------------------------------------------
# base.py — path spelling gate, descriptor traversal, byte budgets
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["", "   ", "/etc/passwd", "//host/share/x.csv", "../x.csv", "a/../../x.csv", "a/../x.csv"],
)
def test_open_binary_rejects_traversal_spellings(context: OperationContext, spelling: str) -> None:
    with (
        pytest.raises(ValueError, match="relative to the workspace"),
        context.open_binary(spelling, suffixes=(".csv",)),
    ):
        pass


@pytest.mark.parametrize("spelling", [".", ".."])
def test_open_binary_rejects_dot_spellings(context: OperationContext, spelling: str) -> None:
    with pytest.raises(ValueError), context.open_binary(spelling, suffixes=(".csv",)):
        pass


@pytest.mark.parametrize("name", ["x.exe", "x.jsonl ", "x.csv.exe", "x.CSV.exe", ".csv", "x.json"])
def test_open_binary_suffix_spelling_must_match_allowlist(
    context: OperationContext, workspace: Path, name: str
) -> None:
    _write(workspace, name, b"a,b\n1,2\n")
    # '.csv' names a hidden dotfile with no suffix; '.json' is outside (".csv",).
    with (
        pytest.raises(ValueError, match="file extension"),
        context.open_binary(name, suffixes=(".csv",)),
    ):
        pass


def test_open_binary_suffix_match_is_case_insensitive(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "data.CSV", b"a,b\n1,2\n")
    with context.open_binary("data.CSV", suffixes=(".csv",)) as stream:
        assert stream.read() == b"a,b\n1,2\n"


def test_open_binary_missing_file_and_missing_parent_fail_closed(
    context: OperationContext,
) -> None:
    for spelling in ("gone.csv", "no-dir/gone.csv"):
        with pytest.raises(OSError), context.open_binary(spelling, suffixes=(".csv",)):
            pass


def test_open_binary_rejects_directory_leaf(context: OperationContext, workspace: Path) -> None:
    (workspace / "adir.csv").mkdir()
    with (
        pytest.raises(ValueError, match="regular file"),
        context.open_binary("adir.csv", suffixes=(".csv",)),
    ):
        pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor traversal")
def test_open_binary_rejects_symlinked_leaf_and_component(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "real.csv", b"a\n1\n")
    os.symlink("real.csv", workspace / "leaf.csv")
    with pytest.raises(OSError), context.open_binary("leaf.csv", suffixes=(".csv",)):
        pass
    (workspace / "sub").mkdir()
    _write(workspace, "sub/inner.csv", b"a\n2\n")
    os.symlink("sub", workspace / "linked")
    with pytest.raises(OSError), context.open_binary("linked/inner.csv", suffixes=(".csv",)):
        pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor traversal")
def test_open_binary_rejects_fifo_leaf(context: OperationContext, workspace: Path) -> None:
    os.mkfifo(workspace / "pipe.csv")
    with (
        pytest.raises(ValueError, match="regular file"),
        context.open_binary("pipe.csv", suffixes=(".csv",)),
    ):
        pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor traversal")
def test_open_binary_accepts_hardlink_inside_workspace(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "real.csv", b"a\n1\n")
    os.link(workspace / "real.csv", workspace / "alias.csv")
    with context.open_binary("alias.csv", suffixes=(".csv",)) as stream:
        assert stream.read() == b"a\n1\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor traversal")
def test_open_binary_read_is_descriptor_pinned_against_rename_swap(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "victim.txt", b"ORIGINAL")
    _write(workspace, "swapped.txt", b"REPLACED")
    with context.open_binary("victim.txt", suffixes=(".txt",)) as stream:
        os.replace(workspace / "swapped.txt", workspace / "victim.txt")
        assert stream.read() == b"ORIGINAL"
    assert stream.source_sha256 == hashlib.sha256(b"ORIGINAL").hexdigest()


def test_open_binary_byte_budget_boundary(context: OperationContext, workspace: Path) -> None:
    payload = b"x" * 4096
    _write(workspace, "exact.bin", payload)
    with context.open_binary("exact.bin", suffixes=(".bin",), max_bytes=4096) as stream:
        assert stream.read() == payload
        assert stream.reached_eof
    with pytest.raises(ValueError, match="exceeds"):
        context.read_bytes("exact.bin", suffixes=(".bin",), max_bytes=4095)
    with (
        pytest.raises(ValueError, match="exceeds"),
        context.open_binary("exact.bin", suffixes=(".bin",), max_bytes=4095),
    ):
        pass


def test_open_binary_max_bytes_must_be_plain_nonnegative_int(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "a.txt", b"1")
    for budget in (True, -1, "5"):
        with (
            pytest.raises(ValueError),
            context.open_binary("a.txt", suffixes=(".txt",), max_bytes=budget),
        ):
            pass
    with (
        pytest.raises(ValueError, match="exceeds"),
        context.open_binary("a.txt", suffixes=(".txt",), max_bytes=0),
    ):
        pass


def test_workspace_reader_digest_requires_full_read(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "d.txt", b"0123456789")
    with context.open_binary("d.txt", suffixes=(".txt",)) as stream:
        assert stream.read(4) == b"0123"
        with pytest.raises(ValueError, match="entire file"):
            stream.source_sha256
        assert stream.read() == b"456789"
        assert stream.source_sha256 == hashlib.sha256(b"0123456789").hexdigest()


def test_resolve_file_is_documented_inspect_only(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "inner.txt", b"x")
    resolved = context.resolve_file("inner.txt", suffixes=(".txt",))
    assert resolved.is_file()
    with pytest.raises(ValueError):
        context.resolve_file("../escape.txt", suffixes=(".txt",))


@pytest.mark.skipif(os.name != "posix", reason="POSIX resolve follows links by contract")
def test_resolve_file_follows_links_but_bounds_target(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "inside.txt", b"x")
    outside = workspace.parent / f"{workspace.name}-outside.txt"
    outside.write_bytes(b"y")
    try:
        os.symlink("inside.txt", workspace / "in-link.txt")
        assert context.resolve_file("in-link.txt", suffixes=(".txt",)).name == "inside.txt"
        os.symlink(outside, workspace / "out-link.txt")
        with pytest.raises(ValueError, match="inside the workspace"):
            context.resolve_file("out-link.txt", suffixes=(".txt",))
    finally:
        outside.unlink()


def test_canonical_json_is_deterministic_and_bounded() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == b'{"a":[2,3],"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})
    with pytest.raises(ValueError, match="budget"):
        canonical_json({"k": "v" * 64}, max_bytes=16)


class _ProbeInput(InputModel):
    value: int


class _ProbeOutput(OutputModel):
    echoed: int


def _probe_handler(request: _ProbeInput, context: OperationContext) -> _ProbeOutput:
    return _ProbeOutput(echoed=request.value)


def test_operation_contract_rejects_bad_identity_or_binding() -> None:
    for bad_id in ("x.y", "plugins.9bad", "plugins.has-hyphen", "plugins.upper_Case"):
        with pytest.raises(ValueError, match="identifier"):
            Operation(
                id=bad_id,
                kind="plugin",
                description="d",
                input_model=_ProbeInput,
                output_model=_ProbeOutput,
                handler=_probe_handler,
            )
    with pytest.raises(ValueError, match="namespace must agree"):
        Operation(
            id="features.probe",
            kind="plugin",
            description="d",
            input_model=_ProbeInput,
            output_model=_ProbeOutput,
            handler=_probe_handler,
        )
    from fx1.operations import read_jsonl

    with pytest.raises(ValueError, match="capability file"):
        Operation(
            id="plugins.probe",
            kind="plugin",
            description="d",
            input_model=read_jsonl.Input,
            output_model=_ProbeOutput,
            handler=_probe_handler,
        )
    with pytest.raises(ValueError, match="capability file"):
        Operation(
            id="plugins.probe",
            kind="plugin",
            description="d",
            input_model=_ProbeInput,
            output_model=read_jsonl.Output,
            handler=_probe_handler,
        )
    with pytest.raises(ValueError, match="nonempty"):
        Operation(
            id="plugins.probe",
            kind="plugin",
            description="  ",
            input_model=_ProbeInput,
            output_model=_ProbeOutput,
            handler=_probe_handler,
        )


def test_invoke_envelope_is_honest_and_deterministic(
    context: OperationContext, workspace: Path
) -> None:
    _write(workspace, "ok.jsonl", b'{"a":1}\n')
    first = execute_operation(JSONL, {"path": "ok.jsonl"}, workspace_root=workspace)
    second = execute_operation(JSONL, {"path": "ok.jsonl"}, workspace_root=workspace)
    assert first == second
    assert first["schema"] == "fx1.operation-result/v1"
    assert first["market_evidence"] is False
    assert first["research_only"] is True
    assert first["live_pnl_claim"] is False
    assert len(first["input_sha256"]) == 64
    assert len(first["output_sha256"]) == 64
    json.dumps(first, allow_nan=False)


def test_invoke_rejects_non_dict_and_extra_arguments(workspace: Path) -> None:
    _write(workspace, "ok.jsonl", b'{"a":1}\n')
    with pytest.raises(ValidationError):
        execute_operation(JSONL, ["ok.jsonl"], workspace_root=workspace)
    with pytest.raises(ValidationError):
        execute_operation(JSONL, {"path": "ok.jsonl", "undeclared": 1}, workspace_root=workspace)
    with pytest.raises(ValidationError):
        execute_operation(JSONL, {"path": "ok.jsonl", "offset": "1"}, workspace_root=workspace)


# --------------------------------------------------------------------------
# registry.py — literal binding, bounded discovery, tool dispatch
# --------------------------------------------------------------------------


def test_registry_rejects_unregistered_ids_and_tools(workspace: Path) -> None:
    for bogus in ("os.system", "plugins.read_jsonl2", "skills.execute", "__main__"):
        with pytest.raises(KeyError, match="unregistered"):
            get_operation(bogus)
    with pytest.raises(KeyError, match="unknown operation tool"):
        invoke_operation_tool("subprocess.run", {}, workspace_root=workspace)
    with pytest.raises(KeyError, match="unregistered"):
        execute_operation("plugins.not_registered", {}, workspace_root=workspace)


def test_every_registered_operation_binds_its_own_module() -> None:
    for entry in registry.list_operations(limit=100)["results"]:
        operation = get_operation(entry["id"])
        namespace = {"feature": "features", "skill": "skills", "plugin": "plugins"}[entry["kind"]]
        assert entry["id"].startswith(f"{namespace}.")
        assert entry["module"] == f"fx1.operations.{entry['id'].split('.', 1)[1]}"
        assert operation.id == entry["id"]
        described = operation.describe()
        assert described["module"] == entry["module"]
        assert described["implementation"] == "independent"
        assert described["market_evidence"] is False
        assert described["input_schema"]["additionalProperties"] is False
        assert described["output_schema"]["additionalProperties"] is False


def test_list_operations_bounds_and_pagination() -> None:
    with pytest.raises(ValueError):
        registry.list_operations(limit=0)
    with pytest.raises(ValueError):
        registry.list_operations(limit=101)
    with pytest.raises(ValueError):
        registry.list_operations(offset=-1)
    with pytest.raises(ValueError):
        registry.list_operations(kind="tool")
    full = registry.list_operations(limit=100)
    assert full["schema"] == "fx1.operation-list/v1"
    assert full["matching_count"] == full["implementation_count"] > 0
    assert not full["has_more"]
    page = registry.list_operations(limit=1)
    assert len(page["results"]) == 1 and page["has_more"]
    ids = [entry["id"] for entry in full["results"]]
    assert ids == sorted(ids)
    assert registry.list_operations(limit=5, offset=10_000)["results"] == []


def test_list_operations_search_is_casefolded_and_conjunctive() -> None:
    hits = registry.list_operations(query="JSONL", limit=100)["results"]
    assert hits and all(
        "jsonl" in entry["id"] or "jsonl" in entry["description"].casefold() for entry in hits
    )
    assert registry.list_operations(query="jsonl no-such-term", limit=100)["matching_count"] == 0
    plugins_only = registry.list_operations(kind="plugin", limit=100)
    assert plugins_only["matching_count"] > 0
    assert all(entry["kind"] == "plugin" for entry in plugins_only["results"])


def test_invoke_operation_tool_validates_arguments(workspace: Path) -> None:
    with pytest.raises(ValidationError):
        invoke_operation_tool("list_operations", {"limit": 0})
    with pytest.raises(ValidationError):
        invoke_operation_tool("describe_operation", {"operation_id": ""})
    with pytest.raises(ValidationError):
        invoke_operation_tool("execute_operation", {"operation_id": JSONL})
    described = invoke_operation_tool("describe_operation", {"operation_id": JSONL})
    assert described["id"] == JSONL


# --------------------------------------------------------------------------
# read_jsonl.py
# --------------------------------------------------------------------------


def test_jsonl_duplicate_keys_rejected_at_any_depth(workspace: Path) -> None:
    _write(workspace, "top.jsonl", b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="duplicate object key"):
        _result(JSONL, {"path": "top.jsonl"}, workspace)
    _write(workspace, "nested.jsonl", b'{"a":{"x":1,"x":2}}\n')
    with pytest.raises(ValueError, match="duplicate object key"):
        _result(JSONL, {"path": "nested.jsonl"}, workspace)


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_jsonl_nonfinite_numbers_rejected(workspace: Path, literal: bytes) -> None:
    _write(workspace, "bad.jsonl", b'{"a":' + literal + b"}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        _result(JSONL, {"path": "bad.jsonl"}, workspace)


@pytest.mark.parametrize("line", [b"[1,2]", b"123", b'"text"', b"null", b"true"])
def test_jsonl_records_must_be_objects(workspace: Path, line: bytes) -> None:
    _write(workspace, "shape.jsonl", line + b"\n")
    with pytest.raises(ValueError, match="must be a JSON object"):
        _result(JSONL, {"path": "shape.jsonl"}, workspace)


def test_jsonl_rejects_surrogates_bare_cr_and_bad_utf8(workspace: Path) -> None:
    _write(workspace, "sur.jsonl", b'{"a":"\\ud800"}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        _result(JSONL, {"path": "sur.jsonl"}, workspace)
    _write(workspace, "cr.jsonl", b'{"a":1}\r{"a":2}\r')
    with pytest.raises(ValueError, match="invalid JSON"):
        _result(JSONL, {"path": "cr.jsonl"}, workspace)
    _write(workspace, "bytes.jsonl", b'{"a":"\xff\xfe"}\n')
    with pytest.raises(ValueError, match="UTF-8"):
        _result(JSONL, {"path": "bytes.jsonl"}, workspace)


def test_jsonl_depth_limit_boundary(workspace: Path) -> None:
    ok = b'{"a":' + b"[" * 63 + b"1" + b"]" * 63 + b"}\n"
    _write(workspace, "d63.jsonl", ok)
    assert _result(JSONL, {"path": "d63.jsonl"}, workspace)["total_rows"] == 1
    _write(workspace, "d66.jsonl", b'{"a":' + b"[" * 66 + b"1" + b"]" * 66 + b"}\n")
    with pytest.raises(ValueError, match="nesting"):
        _result(JSONL, {"path": "d66.jsonl"}, workspace)


def test_jsonl_line_numbering_counts_blanks_and_pages(workspace: Path) -> None:
    _write(workspace, "num.jsonl", b'\n{"a":1}\n\n\n{"a":2}\n')
    result = _result(JSONL, {"path": "num.jsonl"}, workspace)
    assert result["physical_line_numbers"] == [2, 5]
    assert result["blank_lines_skipped"] == 3
    page = _result(JSONL, {"path": "num.jsonl", "offset": 1, "limit": 1}, workspace)
    assert page["rows"] == [{"a": 2}]
    assert page["physical_line_numbers"] == [5]
    assert page["has_more"] is False


def test_jsonl_physical_line_cap(workspace: Path) -> None:
    record = b'{"a":"' + b"x" * 999_990 + b'"}'
    assert len(record) == 999_998
    _write(workspace, "max.jsonl", record + b"\n")  # 999999 chars — under cap
    assert _result(JSONL, {"path": "max.jsonl"}, workspace)["total_rows"] == 1
    _write(workspace, "over.jsonl", b'{"a":"' + b"x" * 1_000_000 + b'"}\n')
    with pytest.raises(ValueError, match="physical lines"):
        _result(JSONL, {"path": "over.jsonl"}, workspace)


def test_jsonl_column_and_page_caps(workspace: Path) -> None:
    wide = "{" + ",".join(f'"k{i}":0' for i in range(1_001)) + "}\n"
    _write(workspace, "wide.jsonl", wide.encode())
    with pytest.raises(ValueError, match="1000 distinct"):
        _result(JSONL, {"path": "wide.jsonl"}, workspace)
    fat = b'{"a":"' + b"x" * 5_000 + b'"}\n'
    _write(workspace, "fat.jsonl", fat * 200)
    with pytest.raises(ValueError, match="page exceeds"):
        _result(JSONL, {"path": "fat.jsonl", "limit": 200}, workspace)


def test_jsonl_record_cap_counts_all_records(workspace: Path) -> None:
    _write(workspace, "many.jsonl", b'{"a":1}\n' * 100_001)
    with pytest.raises(ValueError, match="100000 records"):
        _result(JSONL, {"path": "many.jsonl"}, workspace)


def test_jsonl_suffix_and_offsets(workspace: Path) -> None:
    _write(workspace, "ok.ndjson", b'{"a":1}\n')
    assert _result(JSONL, {"path": "ok.ndjson"}, workspace)["total_rows"] == 1
    _write(workspace, "ok.json", b'{"a":1}\n')
    with pytest.raises(ValueError, match="extension"):
        _result(JSONL, {"path": "ok.json"}, workspace)
    for arguments in ({"offset": -1}, {"offset": 100_001}, {"limit": 201}, {"limit": 1.5}):
        with pytest.raises(ValidationError):
            execute_operation(JSONL, {"path": "ok.ndjson", **arguments}, workspace_root=workspace)


# --------------------------------------------------------------------------
# read_csv.py
# --------------------------------------------------------------------------


def test_csv_delimiter_must_be_single_innocuous_char(workspace: Path) -> None:
    _write(workspace, "ok.csv", b"a,b\n1,2\n")
    for bad in ("\r", "\n", "\0", '"', "ab", ""):
        with pytest.raises(ValidationError):
            execute_operation(CSV, {"path": "ok.csv", "delimiter": bad}, workspace_root=workspace)
    tabbed = _write(workspace, "ok.tsv", b"a\tb\n1\t2\n")
    result = _result(CSV, {"path": tabbed, "delimiter": "\t"}, workspace)
    assert result["columns"] == ["a", "b"]


@pytest.mark.parametrize(
    ("header", "pattern"),
    [
        (b"a,,b\n1,2,3\n", "no blank names"),
        (b"a,a\n1,2\n", "duplicate column"),
        (",".join(f"c{i}" for i in range(257)).encode() + b"\n", "256 columns"),
        (b"a" * 257 + b",b\n1,2\n", "256-character"),
    ],
)
def test_csv_header_contract(workspace: Path, header: bytes, pattern: str) -> None:
    _write(workspace, "bad.csv", header)
    with pytest.raises(ValueError, match=pattern):
        _result(CSV, {"path": "bad.csv"}, workspace)


def test_csv_ragged_rows_unclosed_quotes_and_nuls(workspace: Path) -> None:
    _write(workspace, "ragged.csv", b"a,b\n1,2,3\n")
    with pytest.raises(ValueError, match="expected 2"):
        _result(CSV, {"path": "ragged.csv"}, workspace)
    _write(workspace, "quote.csv", b'a,b\n"unclosed,2\n')
    with pytest.raises(ValueError, match="invalid CSV"):
        _result(CSV, {"path": "quote.csv"}, workspace)
    _write(workspace, "nul.csv", b"a,b\n\x001,2\n")
    with pytest.raises(ValueError, match="NUL"):
        _result(CSV, {"path": "nul.csv"}, workspace)


def test_csv_returns_verbatim_strings_without_interpretation(workspace: Path) -> None:
    _write(workspace, "formula.csv", b"a,b\n\"=cmd|'/c calc'!A0\",+2-3\n")
    result = _result(CSV, {"path": "formula.csv"}, workspace)
    assert result["rows"] == [{"a": "=cmd|'/c calc'!A0", "b": "+2-3"}]


def test_csv_line_cap_blank_records_and_suffix(workspace: Path) -> None:
    _write(workspace, "long.csv", b"a,b\n" + b"x" * 1_000_001 + b",2\n")
    with pytest.raises(ValueError, match="physical lines"):
        _result(CSV, {"path": "long.csv"}, workspace)
    _write(workspace, "blanks.csv", b"a,b\n\n1,2\n\n")
    result = _result(CSV, {"path": "blanks.csv"}, workspace)
    assert result["total_rows"] == 1
    assert result["blank_records_skipped"] == 2
    _write(workspace, "bad.txt", b"a,b\n1,2\n")
    with pytest.raises(ValueError, match="extension"):
        _result(CSV, {"path": "bad.txt"}, workspace)


def test_csv_field_size_limit_and_record_cap(workspace: Path) -> None:
    giant_cell = b"a,b\n" + b"x" * 131_073 + b",2\n"
    _write(workspace, "big-cell.csv", giant_cell)
    with pytest.raises(ValueError, match="invalid CSV"):
        _result(CSV, {"path": "big-cell.csv"}, workspace)
    _write(workspace, "many.csv", b"a\n" + b"1\n" * 100_001)
    with pytest.raises(ValueError, match="100000"):
        _result(CSV, {"path": "many.csv"}, workspace)


# --------------------------------------------------------------------------
# read_toml.py
# --------------------------------------------------------------------------


def test_toml_temporal_values_are_typed_and_path_indexed(workspace: Path) -> None:
    _write(
        workspace,
        "times.toml",
        b"""
offset = 1979-05-27T07:32:00Z
local_dt = 1979-05-27T07:32:00
local_d = 1979-05-27
local_t = 07:32:00
plain = 42
""",
    )
    result = _result(TOML, {"path": "times.toml"}, workspace)
    kinds = {entry["kind"] for entry in result["temporal_values"]}
    assert kinds == {"offset_datetime", "local_datetime", "local_date", "local_time"}
    paths = {tuple(entry["path"]) for entry in result["temporal_values"]}
    assert paths == {("offset",), ("local_dt",), ("local_d",), ("local_t",)}
    assert result["data"]["plain"] == 42
    assert result["data"]["offset"] == "1979-05-27T07:32:00+00:00"


@pytest.mark.parametrize("literal", [b"x = inf", b"x = -inf", b"x = nan"])
def test_toml_nonfinite_floats_rejected(workspace: Path, literal: bytes) -> None:
    _write(workspace, "nf.toml", literal + b"\n")
    with pytest.raises(ValueError, match="nonfinite"):
        _result(TOML, {"path": "nf.toml"}, workspace)


def test_toml_table_path_must_resolve_to_table(workspace: Path) -> None:
    _write(workspace, "nested.toml", b"[a.b]\nx = 1\ny = 1979-05-27\n")
    selected = _result(TOML, {"path": "nested.toml", "table_path": ["a", "b"]}, workspace)
    assert selected["data"] == {"x": 1, "y": "1979-05-27"}
    assert [e["path"] for e in selected["temporal_values"]] == [["a", "b", "y"]]
    with pytest.raises(ValueError, match="does not resolve"):
        _result(TOML, {"path": "nested.toml", "table_path": ["a", "zz"]}, workspace)
    with pytest.raises(ValueError, match="select a TOML table"):
        _result(TOML, {"path": "nested.toml", "table_path": ["a", "b", "x"]}, workspace)
    for arguments in ({"table_path": ["x" * 257]}, {"table_path": ["a"] * 65}):
        with pytest.raises(ValidationError):
            execute_operation(TOML, {"path": "nested.toml", **arguments}, workspace_root=workspace)


def test_toml_node_depth_and_temporal_caps(workspace: Path) -> None:
    deep = "x = " + "{a=" * 66 + "1" + "}" * 66 + "\n"
    _write(workspace, "deep.toml", deep.encode())
    with pytest.raises(ValueError, match="nesting|20000"):
        _result(TOML, {"path": "deep.toml"}, workspace)
    _write(workspace, "many.toml", b"x = [" + b"1," * 20_001 + b"1]\n")
    with pytest.raises(ValueError, match="20000"):
        _result(TOML, {"path": "many.toml"}, workspace)
    times = "[t]\n" + "\n".join(f"k{i} = 1980-01-01" for i in range(201))
    _write(workspace, "times.toml", times.encode())
    with pytest.raises(ValueError, match="200 temporal"):
        _result(TOML, {"path": "times.toml", "table_path": ["t"]}, workspace)


def test_toml_is_pure_parse_no_interpolation(workspace: Path) -> None:
    _write(workspace, "env.toml", b'x = "$HOME"\ny = "$(cat /etc/passwd)"\n')
    result = _result(TOML, {"path": "env.toml"}, workspace)
    assert result["data"] == {"x": "$HOME", "y": "$(cat /etc/passwd)"}
    _write(workspace, "bad.toml", b"x = \n")
    with pytest.raises(ValueError, match="invalid UTF-8 TOML"):
        _result(TOML, {"path": "bad.toml"}, workspace)
    _write(workspace, "bytes.toml", b"x = \xff\n")
    with pytest.raises(ValueError, match="invalid UTF-8 TOML"):
        _result(TOML, {"path": "bytes.toml"}, workspace)
    with pytest.raises(ValueError, match="extension"):
        _result(TOML, {"path": "bad.toml.cfg"}, workspace)


def test_toml_keys_over_256_rejected(workspace: Path) -> None:
    _write(workspace, "key.toml", ('"' + "k" * 257 + '" = 1\n').encode())
    with pytest.raises(ValueError, match="256"):
        _result(TOML, {"path": "key.toml"}, workspace)


# --------------------------------------------------------------------------
# inspect_zip.py — declared metadata, never extracted
# --------------------------------------------------------------------------


def _zip_bytes(members: list[tuple[str, bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data, external_attr in members:
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = external_attr
            info.create_system = 3
            archive.writestr(info, data)
    return buffer.getvalue()


def test_zip_path_issue_taxonomy(workspace: Path) -> None:
    payload = _zip_bytes(
        [
            ("../evil.txt", b"x", 0o100644 << 16),
            ("sub\\..\\evil2.txt", b"x", 0o100644 << 16),
            ("/abs/file.txt", b"x", 0o100644 << 16),
            ("C:\\win\\file.txt", b"x", 0o100644 << 16),
            ("aux.txt", b"x", 0o100644 << 16),
            ("trailing.txt ", b"x", 0o100644 << 16),
            ("ctrl\x01name.txt", b"x", 0o100644 << 16),
            ("del\x7fname.txt", b"x", 0o100644 << 16),
            ("clean.txt", b"x", 0o100644 << 16),
        ]
    )
    _write(workspace, "h.zip", payload)
    result = _result(ZIP, {"path": "h.zip", "limit": 200}, workspace)
    assert result["entries_with_path_issues"] == 8
    # '..' components end with '.', and 'C:\win\file.txt' carries a ':' — both
    # are additionally flagged windows_ambiguous_name.
    assert result["path_issue_counts"] == {
        "absolute_or_drive_path": 2,
        "control_character": 2,
        "parent_traversal": 2,
        "windows_ambiguous_name": 5,
    }
    clean = next(m for m in result["members"] if m["name"] == "clean.txt")
    assert clean["path_issues"] == [] and clean["original_path_issues"] == []


def test_zip_symlink_encrypted_and_duplicate_flags(workspace: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        link = zipfile.ZipInfo("link.txt", date_time=(2024, 1, 1, 0, 0, 0))
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, b"target")
        archive.writestr("enc.txt", b"data")
        archive.writestr("dup.txt", b"1")
        archive.writestr("dup.txt", b"2")
        directory = zipfile.ZipInfo("dir/", date_time=(2024, 1, 1, 0, 0, 0))
        archive.writestr(directory, b"")
    data = bytearray(buffer.getvalue())
    # Set the encrypted flag bit on the enc.txt central-directory record
    # (writestr recomputes flag_bits, so the declaration must be patched in).
    position = 0
    while (position := data.find(b"PK\x01\x02", position)) != -1:
        name_length = int.from_bytes(data[position + 28 : position + 30], "little")
        if data[position + 46 : position + 46 + name_length] == b"enc.txt":
            flags = int.from_bytes(data[position + 8 : position + 10], "little") | 1
            data[position + 8 : position + 10] = flags.to_bytes(2, "little")
        position += 46 + name_length
    _write(workspace, "flags.zip", bytes(data))
    result = _result(ZIP, {"path": "flags.zip", "limit": 200}, workspace)
    members = {m["name"]: m for m in result["members"]}
    assert result["symlink_count"] == 1 and members["link.txt"]["symlink"]
    assert result["encrypted_entry_count"] == 1 and members["enc.txt"]["encrypted"]
    assert result["duplicate_name_count"] == 1 and members["dup.txt"]["duplicate_name"]
    assert result["directory_count"] == 1 and members["dir/"]["directory"]
    assert result["entry_count"] == 5
    assert result["file_count"] == 4


def test_zip_reports_declared_sizes_without_verifying(workspace: Path) -> None:
    payload = _zip_bytes([("huge.txt", b"tiny", 0o100644 << 16)])
    _write(workspace, "declared.zip", payload)
    result = _result(ZIP, {"path": "declared.zip"}, workspace)
    member = result["members"][0]
    assert member["declared_uncompressed_bytes"] == 4
    assert member["compression_method"] == zipfile.ZIP_DEFLATED
    assert member["expansion_ratio"] == 4 / member["compressed_bytes"]
    # The reported CRC/size pair is the archive's declaration, echoed verbatim.
    assert len(member["declared_crc32"]) == 8


def test_zip_entry_count_member_name_and_format_caps(workspace: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(10_001):
            archive.writestr(f"m{index}.txt", b"x")
    _write(workspace, "many.zip", buffer.getvalue())
    with pytest.raises(ValueError, match="10000"):
        _result(ZIP, {"path": "many.zip"}, workspace)
    _write(workspace, "longname.zip", _zip_bytes([("n" * 4_097, b"x", 0o100644 << 16)]))
    with pytest.raises(ValueError, match="4096"):
        _result(ZIP, {"path": "longname.zip"}, workspace)
    valid = _zip_bytes([("a.txt", b"x", 0)])
    for name, blob in (
        ("garbage.zip", b"not a zip file"),
        ("truncated.zip", valid[: len(valid) // 2]),
        ("empty.zip", b""),
    ):
        _write(workspace, name, blob)
        with pytest.raises(ValueError):
            _result(ZIP, {"path": name}, workspace)


def test_zip_npz_suffix_and_pagination(workspace: Path) -> None:
    payload = _zip_bytes([("a.txt", b"1", 0), ("b.txt", b"2", 0)])
    _write(workspace, "arr.npz", payload)
    result = _result(ZIP, {"path": "arr.npz", "limit": 1}, workspace)
    assert len(result["members"]) == 1 and result["has_more"]
    _write(workspace, "arr.jar", payload)
    with pytest.raises(ValueError, match="extension"):
        _result(ZIP, {"path": "arr.jar"}, workspace)


# --------------------------------------------------------------------------
# inspect_numpy_array.py — literal-only header, no pickle, exact payload
# --------------------------------------------------------------------------


def _npy_bytes(header: bytes, payload: bytes = b"", version: tuple[int, int] = (1, 0)) -> bytes:
    length_bytes = 2 if version == (1, 0) else 4
    return (
        b"\x93NUMPY"
        + bytes(version)
        + len(header).to_bytes(length_bytes, "little")
        + header
        + payload
    )


def test_npy_real_roundtrip_reports_verified_shape(workspace: Path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.arange(6).reshape(2, 3).astype(">f8"), allow_pickle=False)
    _write(workspace, "ok.npy", buffer.getvalue())
    result = _result(NPY, {"path": "ok.npy"}, workspace)
    assert result["format_version"] == "1.0"
    assert result["shape"] == [2, 3]
    assert result["element_count"] == 6
    assert result["dtype"] == ">f8"
    assert result["byte_order"] == ">"
    assert result["fortran_order"] is False
    assert result["payload_bytes"] == 48
    assert result["source_bytes"] == len(buffer.getvalue())


def test_npy_structured_dtype_fields_reported(workspace: Path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.zeros(4, dtype=np.dtype([("x", "<i2"), ("y", "<f4")])))
    _write(workspace, "struct.npy", buffer.getvalue())
    result = _result(NPY, {"path": "struct.npy"}, workspace)
    assert result["fields"] == [
        {"name": "x", "dtype": "int16", "byte_offset": 0},
        {"name": "y", "dtype": "float32", "byte_offset": 2},
    ]
    assert result["payload_bytes"] == 24


@pytest.mark.parametrize(
    ("blob", "pattern"),
    [
        (b"", "not an NPY"),
        (b"\x93NUMPY", "not an NPY"),
        (b"\x93NUMPY\x04\x00\x02\x00" + b" " * 40, "unsupported NPY format"),
        (_npy_bytes(b""), "between 1 and 65536"),
        (_npy_bytes(b"{'descr': '<f8'"), "end with a newline"),
        (_npy_bytes(b"[1,2,3]\n"), "dictionary literal"),
        (_npy_bytes(b"{'descr': '<f8', 'fortran_order': False}\n"), "exactly descr"),
        (
            _npy_bytes(b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'x': 0}\n"),
            "exactly",
        ),
        (_npy_bytes(b"{'descr': '<f8', 'fortran_order': 1, 'shape': (1,)}\n"), "boolean"),
        (_npy_bytes(b"{'descr': '<f8', 'fortran_order': False, 'shape': [1]}\n"), "tuple"),
        (
            _npy_bytes(
                b"{'descr': '<f8', 'fortran_order': False, 'shape': (99999999999999999999,)}\n"
            ),
            "dimensions",
        ),
        (
            _npy_bytes(b"{'descr': '<f8', 'fortran_order': False, 'shape': (-1,)}\n"),
            "dimensions",
        ),
        (_npy_bytes(b"{'descr': 42, 'fortran_order': False, 'shape': (1,)}\n"), "string or"),
        (
            _npy_bytes(b"{'descr': '|O8', 'fortran_order': False, 'shape': (1,)}\n", b"x" * 8),
            "pickle",
        ),
        (
            _npy_bytes(
                b"{'descr': __import__('os').system('id'), 'fortran_order': False, 'shape': (1,)}\n"
            ),
            "invalid NPY header",
        ),
        (_npy_bytes(b"{'descr': '2+2', 'fortran_order': False, 'shape': (1,)}\n"), "unsupported"),
        (
            _npy_bytes(b"{'descr': '<f8', 'fortran_order': False, 'shape': (2,)}\n", b"x" * 8),
            "match",
        ),
        (
            _npy_bytes(b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n", b"x" * 16),
            "match",
        ),
    ],
)
def test_npy_hostile_headers_fail_closed(workspace: Path, blob: bytes, pattern: str) -> None:
    _write(workspace, "evil.npy", blob)
    with pytest.raises(ValueError, match=pattern):
        _result(NPY, {"path": "evil.npy"}, workspace)


def test_npy_element_budget_before_payload_check(workspace: Path) -> None:
    blob = _npy_bytes(b"{'descr': '|u1', 'fortran_order': False, 'shape': (2000000000,)}\n")
    _write(workspace, "bomb.npy", blob)
    with pytest.raises(ValueError, match="billion"):
        _result(NPY, {"path": "bomb.npy"}, workspace)


def test_npy_v2_v3_versions(workspace: Path) -> None:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.arange(3), version=(2, 0))
    _write(workspace, "v2.npy", buffer.getvalue())
    assert _result(NPY, {"path": "v2.npy"}, workspace)["format_version"] == "2.0"
    header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    blob = _npy_bytes(header + b" " * padding + b"\n", b"\x00" * 8, version=(3, 0))
    _write(workspace, "v3.npy", blob)
    assert _result(NPY, {"path": "v3.npy"}, workspace)["format_version"] == "3.0"


def test_npy_suffix_allowlist(workspace: Path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.arange(2))
    _write(workspace, "arr.npz", buffer.getvalue())
    with pytest.raises(ValueError, match="extension"):
        _result(NPY, {"path": "arr.npz"}, workspace)


# --------------------------------------------------------------------------
# inspect_parquet.py — footer parse under thrift caps
# --------------------------------------------------------------------------


def test_parquet_roundtrip_schema_rowgroups_and_pagination(workspace: Path) -> None:
    table = pa.table(
        {
            "a": [1, 2, 3],
            "s": pa.array([{"x": 1.5, "y": "p"}, {"x": 2.5, "y": "q"}, {"x": 3.5, "y": "r"}]),
        }
    )
    pq.write_table(table, workspace / "ok.parquet", row_group_size=1)
    result = _result(PARQUET, {"path": "ok.parquet"}, workspace)
    assert [c["name"] for c in result["columns"]] == ["a", "s"]
    assert result["leaf_column_count"] == 3
    assert result["total_rows"] == 3
    assert result["row_group_count"] == 3
    assert result["has_more_row_groups"] is False
    assert result["format_version"] == "2.6"
    page = _result(
        PARQUET, {"path": "ok.parquet", "row_group_offset": 2, "row_group_limit": 1}, workspace
    )
    assert [g["index"] for g in page["row_groups"]] == [2]
    assert page["has_more_row_groups"] is False
    assert page["row_groups"][0]["compression_codecs"] == ["SNAPPY"]


@pytest.mark.parametrize(
    ("blob_name", "blob"),
    [
        ("garbage.parquet", b"PAR1 not really parquet PAR1"),
        ("empty.parquet", b""),
    ],
)
def test_parquet_invalid_bytes_fail_closed(workspace: Path, blob_name: str, blob: bytes) -> None:
    _write(workspace, blob_name, blob)
    with pytest.raises(ValueError, match="invalid or unsupported Parquet"):
        _result(PARQUET, {"path": blob_name}, workspace)


def test_parquet_truncated_file_fails_closed(workspace: Path) -> None:
    table = pa.table({"a": [1, 2]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    _write(workspace, "cut.parquet", buffer.getvalue()[:-4])
    with pytest.raises(ValueError, match="invalid or unsupported Parquet"):
        _result(PARQUET, {"path": "cut.parquet"}, workspace)


def test_parquet_offset_bounds(workspace: Path) -> None:
    table = pa.table({"a": [1]})
    pq.write_table(table, workspace / "one.parquet")
    for arguments in ({"row_group_offset": -1}, {"row_group_limit": 0}, {"row_group_limit": 101}):
        with pytest.raises(ValidationError):
            execute_operation(
                PARQUET, {"path": "one.parquet", **arguments}, workspace_root=workspace
            )


# --------------------------------------------------------------------------
# verify_file_hash.py — same-descriptor digest, tri-state size
# --------------------------------------------------------------------------


def test_hash_match_mismatch_and_size_tristate(workspace: Path) -> None:
    payload = b"hash me"
    _write(workspace, "m.txt", payload)
    digest = hashlib.sha256(payload).hexdigest()
    match = _result(HASH, {"path": "m.txt", "expected_sha256": digest}, workspace)
    assert match["digest_matches"] is True
    assert match["matches"] is True
    assert match["size_matches"] is None  # no expected_bytes supplied
    assert match["actual_bytes"] == len(payload)
    wrong = _result(HASH, {"path": "m.txt", "expected_sha256": HEX64}, workspace)
    assert wrong["digest_matches"] is False and wrong["matches"] is False
    assert wrong["actual_sha256"] == digest
    sized = _result(
        HASH,
        {"path": "m.txt", "expected_sha256": digest, "expected_bytes": len(payload) + 1},
        workspace,
    )
    assert sized["size_matches"] is False and sized["matches"] is False


def test_hash_digest_input_normalization_and_bounds(workspace: Path) -> None:
    _write(workspace, "m.txt", b"hash me")
    digest = hashlib.sha256(b"hash me").hexdigest().upper()
    assert _result(HASH, {"path": "m.txt", "expected_sha256": digest}, workspace)["matches"]
    for bad in ("g" * 64, digest[:63], digest + "0", "Z" * 64):
        with pytest.raises(ValidationError):
            execute_operation(
                HASH, {"path": "m.txt", "expected_sha256": bad}, workspace_root=workspace
            )
    for bad_bytes in (-1, True, 1_000_000_001):
        with pytest.raises(ValidationError):
            execute_operation(
                HASH,
                {"path": "m.txt", "expected_sha256": digest.lower(), "expected_bytes": bad_bytes},
                workspace_root=workspace,
            )
    with pytest.raises(ValidationError):
        execute_operation(
            HASH,
            {"path": "m.txt", "expected_sha256": digest.lower(), "max_bytes": 0},
            workspace_root=workspace,
        )


def test_hash_byte_budget_and_suffix(workspace: Path) -> None:
    _write(workspace, "big.txt", b"x" * 100)
    digest = hashlib.sha256(b"x" * 100).hexdigest()
    with pytest.raises(ValueError, match="exceeds"):
        _result(
            HASH,
            {"path": "big.txt", "expected_sha256": digest, "max_bytes": 50},
            workspace,
        )
    exact = _result(
        HASH,
        {"path": "big.txt", "expected_sha256": digest, "max_bytes": 100},
        workspace,
    )
    assert exact["matches"] is True
    _write(workspace, "blob.exe", b"x")
    with pytest.raises(ValueError, match="extension"):
        _result(HASH, {"path": "blob.exe", "expected_sha256": HEX64}, workspace)


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor traversal")
def test_hash_is_computed_on_the_descriptor_not_the_path(
    workspace: Path, context: OperationContext
) -> None:
    # Swapping the directory entry mid-verification cannot change the result:
    # the digest and size both come from the originally checked descriptor.
    _write(workspace, "victim.txt", b"ORIGINAL")
    _write(workspace, "swapped.txt", b"REPLACED-BYTES")
    with context.open_binary("victim.txt", suffixes=(".txt",)) as stream:
        os.replace(workspace / "swapped.txt", workspace / "victim.txt")
        while stream.read(65_536):
            pass
        assert stream.source_sha256 == hashlib.sha256(b"ORIGINAL").hexdigest()
