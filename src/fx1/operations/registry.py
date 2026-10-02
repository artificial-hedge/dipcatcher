"""Explicit registration and bounded discovery of actual capability implementations."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OperationKind

# One entry per independently authored behavior. IDs, parameter values, generated cards,
# wrappers, and alternate names cannot inflate this inventory.
_IMPLEMENTATIONS = {
    "features.simple_returns": "simple_returns",
    "features.rolling_zscore": "rolling_zscore",
    "features.ewma_variance": "ewma_variance",
    "features.drawdown_path": "drawdown_path",
    "features.bipower_variation": "bipower_variation",
    "features.permutation_entropy": "permutation_entropy",
    "skills.audit_bar_integrity": "audit_bar_integrity",
    "skills.audit_point_in_time": "audit_point_in_time",
    "skills.audit_panel_gaps": "audit_panel_gaps",
    "skills.audit_duplicate_keys": "audit_duplicate_keys",
    "skills.score_quantiles": "score_quantiles",
    "skills.score_binary_forecasts": "score_binary_forecasts",
    "skills.score_intervals": "score_intervals",
    "skills.score_empirical_crps": "score_empirical_crps",
    "plugins.read_csv": "read_csv",
    "plugins.read_jsonl": "read_jsonl",
    "plugins.inspect_parquet": "inspect_parquet",
    "features.rolling_rank": "rolling_rank",
    "features.rolling_mad": "rolling_mad",
    "features.rolling_autocorrelation": "rolling_autocorrelation",
    "features.time_weighted_mean": "time_weighted_mean",
    "skills.select_asof_revisions": "select_asof_revisions",
    "skills.join_asof_observations": "join_asof_observations",
    "skills.audit_revision_conflicts": "audit_revision_conflicts",
    "skills.summarize_ingestion_latency": "summarize_ingestion_latency",
    "skills.audit_missingness": "audit_missingness",
    "skills.audit_schema_drift": "audit_schema_drift",
    "skills.audit_referential_integrity": "audit_referential_integrity",
    "skills.audit_source_coverage": "audit_source_coverage",
    "skills.audit_monotonic_sequences": "audit_monotonic_sequences",
    "plugins.verify_file_hash": "verify_file_hash",
    "plugins.read_toml": "read_toml",
    "plugins.inspect_zip": "inspect_zip",
    "plugins.inspect_numpy_array": "inspect_numpy_array",
}


@lru_cache(maxsize=256)
def get_operation(operation_id: str) -> Operation[Any, Any]:
    """Import only a literal path in the reviewed registry."""
    name = _IMPLEMENTATIONS.get(operation_id)
    if name is None:
        raise KeyError(f"unregistered operation {operation_id!r}")
    module_path = f"fx1.operations.{name}"
    module = import_module(module_path)
    operation = getattr(module, "OPERATION", None)
    if not isinstance(operation, Operation):
        raise TypeError(f"{module_path} must export an Operation named OPERATION")
    if operation.id != operation_id or operation.handler.__module__ != module_path:
        raise ValueError(f"{module_path} does not implement its registered operation identity")
    return cast(Operation[Any, Any], operation)


def list_operations(
    query: str = "",
    *,
    kind: OperationKind | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """Return bounded discovery results; generated legacy cards are excluded."""
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
    if kind is not None and kind not in ("feature", "skill", "plugin"):
        raise ValueError("kind must be feature, skill, or plugin")
    terms = query.casefold().split()
    matches: list[dict[str, Any]] = []
    for operation_id in sorted(_IMPLEMENTATIONS):
        operation = get_operation(operation_id)
        if kind is not None and operation.kind != kind:
            continue
        searchable = f"{operation.id} {operation.description}".casefold()
        if all(term in searchable for term in terms):
            matches.append(
                {
                    "id": operation.id,
                    "kind": operation.kind,
                    "description": operation.description,
                    "version": operation.version,
                    "module": operation.handler.__module__,
                }
            )
    return {
        "schema": "fx1.operation-list/v1",
        "implementation_count": len(_IMPLEMENTATIONS),
        "matching_count": len(matches),
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < len(matches),
        "results": matches[offset : offset + limit],
        "market_evidence": False,
    }


def execute_operation(
    operation_id: str,
    arguments: dict[str, Any],
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Execute an approved operation with a host-controlled filesystem boundary."""
    operation = get_operation(operation_id)
    context = OperationContext(workspace_root=workspace_root or Path.cwd())
    return operation.invoke(arguments, context)


class ListArguments(InputModel):
    query: str = Field(default="", max_length=1000)
    kind: OperationKind | None = None
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=20, ge=1, le=100, strict=True)


class DescribeArguments(InputModel):
    operation_id: str = Field(min_length=1, max_length=128)


class ExecuteArguments(DescribeArguments):
    arguments: dict[str, Any]


def operation_tool_specs() -> list[dict[str, Any]]:
    """Discover, inspect, and execute real operations through function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
            },
        }
        for name, description, model in (
            (
                "list_operations",
                "Find independently implemented dipcatcher capabilities.",
                ListArguments,
            ),
            (
                "describe_operation",
                "Read the exact input/output schema before invoking an operation.",
                DescribeArguments,
            ),
            (
                "execute_operation",
                "Run an approved computation, audit, or workspace data reader. Outputs are not market evidence.",
                ExecuteArguments,
            ),
        )
    ]


def invoke_operation_tool(
    name: str, arguments: dict[str, Any], *, workspace_root: Path | None = None
) -> dict[str, Any]:
    """Dispatch only this package's literal tool names after schema validation."""
    if name == "list_operations":
        request = ListArguments.model_validate(arguments)
        return list_operations(
            request.query, kind=request.kind, offset=request.offset, limit=request.limit
        )
    if name == "describe_operation":
        described = DescribeArguments.model_validate(arguments)
        return get_operation(described.operation_id).describe()
    if name == "execute_operation":
        execution = ExecuteArguments.model_validate(arguments)
        return execute_operation(
            execution.operation_id, execution.arguments, workspace_root=workspace_root
        )
    raise KeyError(f"unknown operation tool {name!r}")
