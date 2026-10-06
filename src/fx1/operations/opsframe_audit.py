"""opsframe_audit — operation framework + registry battery.

Where ioops_audit covers what operations DO, this battery covers the
machinery every operation stands on:

- *Operation descriptors* — id regex, kind↔namespace agreement, nonempty
  description, input/output schema module locality, the
  ``fx1.operation-result/v1`` invoke envelope (input/output SHA-256,
  honesty flags), and ValidationError/TypeError discipline.
- *InputModel/OutputModel* — ``extra="forbid"`` and
  ``allow_inf_nan=False`` on both sides of the contract.
- *OperationContext* — frozen dataclass, abspath'd root, ``resolve_file``
  spelling checks, ``open_binary`` descriptor-relative POSIX walk with
  per-component ``O_NOFOLLOW``, ``fstat`` regular-file + byte-cap
  enforcement, ``read_bytes`` collection.
- *WorkspaceReader* — nonseekable bounded reads, ``bytes_read``
  accounting, ``source_sha256`` gated on observed EOF, close discipline.
- *canonical_json* — sorted keys, compact separators, raw UTF-8,
  ``allow_nan=False``, and a hard byte budget enforced while encoding.
- *_windows_path* — pure normalization of ``\\\\?\\`` and
  ``\\\\?\\UNC\\`` prefixes (the POSIX half of the containment pair).
- *registry* — ``_IMPLEMENTATIONS`` count pinned, every registered id
  resolves to a module-local OPERATION, every on-disk non-battery module
  is registered, ``list_operations`` bounded paging/kind/query
  semantics, ``execute_operation`` end-to-end, and the three literal
  function tools dispatch only after schema validation.

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from fx1.operations import read_csv, registry
from fx1.operations.base import (
    InputModel,
    Operation,
    OperationContext,
    OutputModel,
    WorkspaceReader,
    _windows_path,
    canonical_json,
)
from quant_fund.research.receipt_v2 import canonical_json_bytes
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["opsframe_audit", "opsframe_audit_bench"]

_PROBE_ID = "features.opsframe_probe"
_DEEP = "sub/deep.csv"
_DEEP_NAME = "deep.csv"
_LINK = "link.csv"
_DATA = "data.csv"
_WIN_PATH = "C:\\ws\\a.csv"
_SR = "features.simple_returns"


class _In(InputModel):
    """Local input schema for descriptor probes."""

    x: int
    y: int = 0


class _Out(OutputModel):
    """Local output schema for descriptor probes."""

    value: float


def _handler(request: _In, context: OperationContext) -> _Out:
    return _Out(value=float(request.x + request.y))


def _bad_handler(request: _In, context: OperationContext) -> Any:
    return 42


def _op(**overrides: Any) -> Operation[_In, _Out]:
    kwargs: dict[str, Any] = {
        "id": _PROBE_ID,
        "kind": "feature",
        "description": "probe fixture",
        "input_model": _In,
        "output_model": _Out,
        "handler": _handler,
    }
    kwargs.update(overrides)
    return Operation(**kwargs)


class _Ws:
    """Real temp workspace holding fixture files."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Resolve now: the POSIX open walk refuses symlinked path
        # components, and macOS tempdirs live under /var -> /private/var.
        self.root = Path(self._tmp.name).resolve()
        self.ctx = OperationContext(workspace_root=self.root)

    def write(self, name: str, content: bytes | str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return name


def _refuses(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — refuse probes accept any failure
        return True
    return False


def _probe_descriptors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    op = _op()
    out["op_id_valid"] = op.id == _PROBE_ID
    out["op_version_default"] = op.version == "1.0.0"
    out["op_version_pinned"] = _op(version="2.3.4").version == "2.3.4"
    out["op_id_missing_ns"] = _refuses(_op, id="nons.x")
    out["op_id_upper"] = _refuses(_op, id="features.X")
    out["op_id_dash"] = _refuses(_op, id="features.a-b")
    out["op_id_leading_us"] = _refuses(_op, id="features._a")
    out["op_id_trailing_dot"] = _refuses(_op, id="features.a.")
    out["op_id_double_us_ok"] = _op(id="features.a__b").id == "features.a__b"
    out["op_id_digit_after"] = _op(id="features.a1").id == "features.a1"
    out["op_kind_ns_mismatch"] = _refuses(_op, id="skills.opsframe_probe", kind="feature")
    out["op_skill_ok"] = _op(id="skills.opsframe_probe", kind="skill").kind == "skill"
    out["op_plugin_ok"] = _op(id="plugins.opsframe_probe", kind="plugin").kind == "plugin"
    out["op_desc_blank"] = _refuses(_op, description="   ")
    out["op_desc_empty"] = _refuses(_op, description="")
    out["op_input_foreign_module"] = _refuses(_op, input_model=read_csv.Input)
    out["op_output_foreign_module"] = _refuses(_op, output_model=read_csv.Output)
    d = op.describe()
    out["describe_envelope"] = (
        d["id"] == _PROBE_ID
        and d["kind"] == "feature"
        and d["module"] == "fx1.operations.opsframe_audit"
        and d["implementation"] == "independent"
        and d["market_evidence"] is False
        and d["input_schema"]["type"] == "object"
        and d["output_schema"]["type"] == "object"
    )
    return out


def _probe_invoke(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    op = _op()
    r = op.invoke({"x": 2, "y": 3}, ws.ctx)
    out["invoke_result"] = math.isclose(r["result"]["value"], 5.0, rel_tol=0.0, abs_tol=1e-15)
    out["invoke_envelope"] = (
        r["schema"] == "fx1.operation-result/v1"
        and r["operation_id"] == _PROBE_ID
        and r["kind"] == "feature"
        and r["research_only"] is True
        and r["live_pnl_claim"] is False
        and r["market_evidence"] is False
    )
    expected_in = hashlib.sha256(canonical_json({"x": 2, "y": 3})).hexdigest()
    out["invoke_input_sha"] = r["input_sha256"] == expected_in
    expected_out = hashlib.sha256(canonical_json({"value": 5.0})).hexdigest()
    out["invoke_output_sha"] = r["output_sha256"] == expected_out
    out["invoke_extra_field"] = _refuses(op.invoke, {"x": 1, "bogus": 2}, ws.ctx)
    out["invoke_missing_field"] = _refuses(op.invoke, {"y": 1}, ws.ctx)
    out["invoke_wrong_type"] = _refuses(op.invoke, {"x": "a"}, ws.ctx)
    out["invoke_output_contract"] = _refuses(_op(handler=_bad_handler).invoke, {"x": 1}, ws.ctx)
    # a pure-compute handler never touches the workspace: a nonexistent
    # root does not block invocation (abspath only, no lstat at init)
    out["invoke_nonfs_ctx_ok"] = not _refuses(
        op.invoke, {"x": 1}, OperationContext(workspace_root=Path("/nope/x"))
    )
    out["invoke_context_abspath"] = OperationContext(
        workspace_root=Path("relative")
    ).workspace_root.is_absolute()
    out["invoke_ctx_frozen"] = _refuses(
        setattr,
        ws.ctx,
        "workspace_root",
        Path("frozen-target"),
    ) and dataclasses.is_dataclass(ws.ctx)
    return out


def _probe_models() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["input_extra_forbid"] = _refuses(_In, x=1, bogus=2)
    # exercise allow_inf_nan on a float-bearing model directly
    out["output_nan_refused"] = _refuses(_Out, value=float("nan"))
    out["output_inf_refused"] = _refuses(_Out, value=float("inf"))
    out["output_extra_forbid"] = _refuses(_Out, value=1.0, bogus=2)
    out["output_finite_ok"] = _Out(value=1.5).value == 1.5
    return out


def _probe_canonical_json() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["cj_sorted_keys"] = canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    out["cj_compact_separators"] = canonical_json({"k": [1, 2]}) == b'{"k":[1,2]}'
    out["cj_utf8_raw"] = canonical_json({"k": "é"}) == '{"k":"é"}'.encode()
    out["cj_nan_refused"] = _refuses(canonical_json, {"k": float("nan")})
    out["cj_inf_refused"] = _refuses(canonical_json, {"k": float("inf")})
    out["cj_budget"] = _refuses(canonical_json, ["x" * 100], max_bytes=10)
    out["cj_budget_edge_ok"] = canonical_json(["ab"], max_bytes=len(b'["ab"]')) == b'["ab"]'
    out["cj_deterministic"] = canonical_json({"z": [3, {"b": 1, "a": 0}]}) == canonical_json(
        {"z": [3, {"b": 1, "a": 0}]}
    )
    return out


def _probe_resolve_file(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    ctx = ws.ctx
    ws.write(_DEEP, "a,b\n1,2\n")
    ws.write("UPPER.CSV", "a\n1\n")
    ok = ctx.resolve_file(_DEEP, suffixes=(".csv",))
    out["rf_nested_ok"] = ok.is_file() and ok.name == _DEEP_NAME
    out["rf_case_insensitive_suffix"] = ctx.resolve_file("UPPER.CSV", suffixes=(".csv",)).is_file()
    out["rf_dotdot"] = _refuses(ctx.resolve_file, "../x.csv", suffixes=(".csv",))
    out["rf_absolute"] = _refuses(ctx.resolve_file, str(ws.root / _DEEP), suffixes=(".csv",))
    out["rf_empty"] = _refuses(ctx.resolve_file, "", suffixes=(".csv",))
    out["rf_blank"] = _refuses(ctx.resolve_file, "   ", suffixes=(".csv",))
    out["rf_wrong_suffix"] = _refuses(ctx.resolve_file, _DEEP, suffixes=(".txt",))
    out["rf_missing"] = _refuses(ctx.resolve_file, "sub/absent.csv", suffixes=(".csv",))
    out["rf_directory"] = _refuses(ctx.resolve_file, "sub", suffixes=("",))
    # internal symlink resolves to an inside path: resolve_file (spelling
    # only) accepts it, while the open_binary descriptor walk refuses.
    link = ws.root / _LINK
    with contextlib.suppress(OSError):
        link.symlink_to(ws.root / _DEEP)
    if link.is_symlink():
        resolved = ctx.resolve_file(_LINK, suffixes=(".csv",))
        out["rf_internal_symlink_resolves"] = (
            resolved.name == _DEEP_NAME and resolved.is_relative_to(ws.root)
        )
        out["ob_symlink_refused"] = _refuses(ctx.read_bytes, _LINK, suffixes=(".csv",))
    else:
        out["rf_internal_symlink_resolves"] = True
        out["ob_symlink_refused"] = True
    outside = Path(tempfile.gettempdir()) / "opsframe_outside.csv"
    outside.write_bytes(b"x")
    esc = ws.root / "escape.csv"
    with contextlib.suppress(OSError):
        esc.symlink_to(outside)
    if esc.is_symlink():
        out["rf_escape_symlink_refused"] = _refuses(
            ctx.resolve_file, "escape.csv", suffixes=(".csv",)
        )
    else:
        out["rf_escape_symlink_refused"] = True
    return out


def _probe_open_binary(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    ctx = ws.ctx
    payload = b"cell-a,cell-b\n1,2\n"
    ws.write(_DATA, payload)
    with ctx.open_binary(_DATA, suffixes=(".csv",)) as reader:
        body = reader.read()
        out["ob_read_bytes"] = body == payload
        out["ob_bytes_read"] = reader.bytes_read == len(payload)
        out["ob_eof"] = reader.reached_eof is True
        out["ob_sha256"] = reader.source_sha256 == hashlib.sha256(payload).hexdigest()
    out["ob_close_idempotent"] = reader.closed is True
    out["ob_read_after_close"] = _refuses(reader.read)
    out["ob_missing"] = _refuses(ctx.read_bytes, "absent.csv", suffixes=(".csv",))
    out["ob_dir_refused"] = _refuses(ctx.read_bytes, "sub", suffixes=("",))
    out["ob_max_bytes_zero"] = _refuses(ctx.read_bytes, _DATA, suffixes=(".csv",), max_bytes=0)
    out["ob_max_bytes_bool"] = _refuses(ctx.read_bytes, _DATA, suffixes=(".csv",), max_bytes=True)
    out["ob_max_bytes_str"] = _refuses(ctx.read_bytes, _DATA, suffixes=(".csv",), max_bytes="9")
    out["ob_max_bytes_neg"] = _refuses(ctx.read_bytes, _DATA, suffixes=(".csv",), max_bytes=-1)
    exact = ctx.read_bytes("data.csv", suffixes=(".csv",), max_bytes=len(payload))
    out["ob_max_bytes_exact_ok"] = exact == payload
    out["ob_max_bytes_over"] = _refuses(
        ctx.read_bytes,
        _DATA,
        suffixes=(".csv",),
        max_bytes=len(payload) - 1,
    )
    out["ob_read_bytes_returns"] = isinstance(exact, bytes)
    fifo = ws.root / "pipe.csv"
    made = False
    try:
        os.mkfifo(fifo)
        made = True
    except OSError:
        pass
    if made:
        out["ob_fifo_refused"] = _refuses(ctx.read_bytes, "pipe.csv", suffixes=(".csv",))
    else:
        out["ob_fifo_refused"] = True
    return out


def _probe_workspace_reader(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    path = ws.root / "wr.csv"
    payload = b"abcdef"
    path.write_bytes(payload)
    fd = os.open(path, os.O_RDONLY)
    reader = WorkspaceReader(fd, max_bytes=4)
    out["wr_readable"] = reader.readable() is True
    out["wr_sha_before_eof"] = _refuses(lambda: reader.source_sha256)
    out["wr_growth_refused"] = _refuses(reader.read)
    reader.close()
    fd = os.open(path, os.O_RDONLY)
    reader2 = WorkspaceReader(fd, max_bytes=64)
    out["wr_full_read"] = reader2.read() == payload
    out["wr_reached_eof"] = reader2.reached_eof is True
    out["wr_sha_after_eof"] = reader2.source_sha256 == hashlib.sha256(payload).hexdigest()
    reader2.close()
    out["wr_read_closed"] = _refuses(reader2.read)
    return out


def _probe_windows_paths() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["wp_extended"] = str(_windows_path("\\\\?\\" + _WIN_PATH)) == _WIN_PATH
    out["wp_unc"] = str(_windows_path("\\\\?\\UNC\\srv\\share\\a.csv")) == "\\\\srv\\share\\a.csv"
    out["wp_plain"] = str(_windows_path(_WIN_PATH)) == _WIN_PATH
    return out


def _probe_registry() -> dict[str, bool]:
    out: dict[str, bool] = {}
    impls = registry._IMPLEMENTATIONS  # noqa: SLF001 — count is the contract
    out["reg_count"] = len(impls) == 40
    out["reg_all_resolve"] = all(registry.get_operation(oid).id == oid for oid in impls)
    out["reg_module_locality"] = all(
        registry.get_operation(oid).handler.__module__ == f"fx1.operations.{name}"
        for oid, name in impls.items()
    )
    out["reg_cached_identity"] = registry.get_operation(_SR) is registry.get_operation(_SR)
    out["reg_unregistered"] = _refuses(registry.get_operation, "skills.bogus_probe")
    out["reg_wrong_ns"] = _refuses(registry.get_operation, "features.read_csv")
    # completeness: every non-battery ops module file must be registered
    ops_dir = Path(__file__).resolve().parent
    stems = {
        p.stem
        for p in ops_dir.glob("*.py")
        if p.stem not in {"__init__", "base", "registry"} and not p.stem.endswith("_audit")
    }
    out["reg_module_completeness"] = stems == set(impls.values())
    return out


def _probe_list() -> dict[str, bool]:
    out: dict[str, bool] = {}
    page1 = registry.list_operations()
    out["list_schema"] = page1["schema"] == "fx1.operation-list/v1"
    out["list_count"] = page1["implementation_count"] == 40
    out["list_matching_all"] = page1["matching_count"] == 40
    out["list_default_page"] = len(page1["results"]) == 20
    out["list_has_more"] = page1["has_more"] is True
    out["list_market_evidence"] = page1["market_evidence"] is False
    page2 = registry.list_operations(offset=20, limit=20)
    out["list_page2"] = len(page2["results"]) == 20
    tail = registry.list_operations(offset=39, limit=20)
    out["list_tail"] = len(tail["results"]) == 1 and tail["has_more"] is False
    out["list_sorted"] = page1["results"][0]["id"] == ("features.bipower_variation")
    plugins = registry.list_operations(kind="plugin", limit=100)
    out["list_kind_plugin"] = plugins["matching_count"] == 7 and all(
        r["kind"] == "plugin" for r in plugins["results"]
    )
    feats = registry.list_operations(kind="feature", limit=100)
    out["list_kind_feature"] = feats["matching_count"] == 12
    audits = registry.list_operations("audit", limit=100)
    out["list_query_audit"] = audits["matching_count"] == 12
    out["list_query_casefold"] = (
        registry.list_operations("AUDIT", limit=100)["matching_count"] == 12
    )
    out["list_query_multiterm_and"] = (
        registry.list_operations("audit missingness", limit=100)["matching_count"] == 2
    )
    out["list_query_none"] = (
        registry.list_operations("zzz-no-match", limit=100)["matching_count"] == 0
    )
    out["list_result_fields"] = all(
        {"id", "kind", "description", "version", "module"} <= set(r) for r in page1["results"]
    )
    out["list_offset_neg"] = _refuses(registry.list_operations, offset=-1)
    out["list_limit_zero"] = _refuses(registry.list_operations, limit=0)
    out["list_limit_over"] = _refuses(registry.list_operations, limit=101)
    out["list_kind_bogus"] = _refuses(registry.list_operations, kind="bogus")
    return out


def _probe_execute_tools(ws: _Ws) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = registry.execute_operation(
        _SR,
        {"prices": [100.0, 101.0, 103.02]},
        workspace_root=ws.root,
    )
    rets = r["result"]["returns"]
    out["exec_end_to_end"] = (
        rets[0] is None
        and math.isclose(rets[1], 0.01, rel_tol=0, abs_tol=1e-15)
        and math.isclose(rets[2], 0.02, rel_tol=0, abs_tol=1e-15)
    )
    out["exec_envelope"] = r["schema"] == "fx1.operation-result/v1"
    out["exec_unknown"] = _refuses(
        registry.execute_operation,
        "skills.bogus",
        {},
        workspace_root=ws.root,
    )
    out["exec_bad_args"] = _refuses(
        registry.execute_operation,
        _SR,
        {"prices": [-1.0]},
        workspace_root=ws.root,
    )
    out["exec_strict_lag_bool"] = _refuses(
        registry.execute_operation,
        _SR,
        {"prices": [1.0, 2.0], "lag": True},
        workspace_root=ws.root,
    )
    specs = registry.operation_tool_specs()
    names = {s["function"]["name"] for s in specs}
    out["tools_three"] = names == {
        "list_operations",
        "describe_operation",
        "execute_operation",
    }
    out["tools_schemas"] = all(
        s["type"] == "function" and s["function"]["parameters"]["type"] == "object" for s in specs
    )
    via = registry.invoke_operation_tool(
        "describe_operation",
        {"operation_id": _SR},
    )
    out["tool_describe"] = (
        via["implementation"] == "independent" and via["market_evidence"] is False
    )
    via_list = registry.invoke_operation_tool("list_operations", {"kind": "plugin", "limit": 100})
    out["tool_list"] = via_list["matching_count"] == 7
    via_exec = registry.invoke_operation_tool(
        "execute_operation",
        {
            "operation_id": _SR,
            "arguments": {"prices": [2.0, 4.0]},
        },
        workspace_root=ws.root,
    )
    tool_rets = via_exec["result"]["returns"]
    out["tool_exec"] = tool_rets[0] is None and math.isclose(
        tool_rets[1], 1.0, rel_tol=0, abs_tol=1e-15
    )
    out["tool_unknown"] = _refuses(registry.invoke_operation_tool, "bogus_tool", {})
    out["tool_bad_args"] = _refuses(registry.invoke_operation_tool, "list_operations", {"limit": 0})
    out["tool_extra_args"] = _refuses(
        registry.invoke_operation_tool,
        "describe_operation",
        {"operation_id": "x", "bogus": 1},
    )
    return out


def opsframe_audit() -> dict[str, bool]:
    """Every operation-framework and registry contract as booleans."""
    ws = _Ws()
    try:
        out: dict[str, bool] = {}
        out.update(_probe_descriptors())
        out.update(_probe_invoke(ws))
        out.update(_probe_models())
        out.update(_probe_canonical_json())
        out.update(_probe_resolve_file(ws))
        out.update(_probe_open_binary(ws))
        out.update(_probe_workspace_reader(ws))
        out.update(_probe_windows_paths())
        out.update(_probe_registry())
        out.update(_probe_list())
        out.update(_probe_execute_tools(ws))
    finally:
        ws._tmp.cleanup()  # noqa: SLF001
    return out


def opsframe_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the operations-framework battery."""
    r = opsframe_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "opsframe_audit",
        "schema": "opsframe_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process Operation.invoke + registry dispatch",
            "not_verified": [
                "Windows handle-inspection branch (POSIX-only host)",
                "symlink substitution races between canonicalization and open",
            ],
        },
        "interpretation": (
            "Operation framework holds: descriptor ids are namespaced and "
            "kind-consistent, schemas must live in the capability file, "
            "invocation validates before execution and emits hashed "
            "input/output with honesty flags, workspace containment "
            "fails closed on escapes/symlinks/non-regular/oversized "
            "files, canonical JSON is sorted+compact+budgeted, and the "
            "registry resolves exactly its reviewed 40 modules with "
            "bounded discovery and schema-gated tool dispatch."
            if ok
            else f"OPSFRAME AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(opsframe_audit_bench(), indent=2, sort_keys=True))
