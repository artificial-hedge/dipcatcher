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
    "skills.audit_monotonic_sequences": "audit_monotonic_sequences",
    "plugins.verify_file_hash": "verify_file_hash",
    "plugins.read_toml": "read_toml",
    "plugins.inspect_zip": "inspect_zip",
    "plugins.inspect_numpy_array": "inspect_numpy_array",
    "skills.resolve_security_identity": "resolve_security_identity",
    "skills.select_universe_membership": "select_universe_membership",
    "skills.audit_feature_provenance": "audit_feature_provenance",
    "skills.audit_effective_intervals": "audit_effective_intervals",
    "skills.audit_cross_field_contracts": "audit_cross_field_contracts",
    "skills.audit_source_coverage": "audit_source_coverage",
    "skills.audit_missingness_association": "audit_missingness_association",
    "skills.audit_lineage_graph": "audit_lineage_graph",
    "features.rolling_linear_trend": "rolling_linear_trend",
    "features.permutation_entropy": "permutation_entropy",
    "features.bipower_variation": "bipower_variation",
    "features.spectral_summary": "spectral_summary",
    "skills.score_energy": "score_energy",
    "skills.score_variogram": "score_variogram",
    "skills.score_variance_qlike": "score_variance_qlike",
    "skills.audit_join_fanout": "audit_join_fanout",
    "skills.audit_split_leakage": "audit_split_leakage",
    "skills.select_revision_tombstones": "select_revision_tombstones",
    "skills.audit_vintage_transitions": "audit_vintage_transitions",
    "features.haar_decomposition": "haar_decomposition",
    "features.fractional_difference": "fractional_difference",
    "features.cusum_events": "cusum_events",
    "features.lead_lag_signature": "lead_lag_signature",
    "plugins.read_yaml": "read_yaml",
    "plugins.inspect_gzip": "inspect_gzip",
    "plugins.read_arrow_ipc": "read_arrow_ipc",
    "skills.score_ranked_probability": "score_ranked_probability",
    "skills.build_walk_forward_folds": "build_walk_forward_folds",
    "plugins.inspect_research_receipt": "inspect_research_receipt",
    "plugins.inspect_audit_ledger": "inspect_audit_ledger",
    "skills.audit_forecast_panel": "audit_forecast_panel",
    "skills.join_time_windows": "join_time_windows",
    "skills.audit_resampling_groups": "audit_resampling_groups",
    "features.weighted_covariance": "weighted_covariance",
    "features.isotonic_quantile_repair": "isotonic_quantile_repair",
    "features.empirical_wasserstein": "empirical_wasserstein",
    "skills.build_block_resamples": "build_block_resamples",
    "plugins.inspect_tar": "inspect_tar",
    "plugins.read_xml": "read_xml",
    "plugins.inspect_artifact_bundle": "inspect_artifact_bundle",
    "skills.summarize_pit": "summarize_pit",
    "skills.build_conformal_intervals": "build_conformal_intervals",
    "skills.audit_session_coverage": "audit_session_coverage",
    "skills.assign_horizon_targets": "assign_horizon_targets",
    "skills.audit_release_revisions": "audit_release_revisions",
    "features.label_uniqueness": "label_uniqueness",
    "features.rolling_robust_regression": "rolling_robust_regression",
    "features.empirical_characteristic_function": "empirical_characteristic_function",
    "skills.build_stratified_folds": "build_stratified_folds",
    "plugins.verify_signed_checkpoint": "verify_signed_checkpoint",
    "plugins.inspect_safetensors": "inspect_safetensors",
    "plugins.read_matrix_market": "read_matrix_market",
    "skills.score_histogram_density": "score_histogram_density",
    "skills.score_censored_brier": "score_censored_brier",
    "skills.build_availability_frontier": "build_availability_frontier",
    "skills.audit_stream_offsets": "audit_stream_offsets",
    "skills.audit_temporal_constraints": "audit_temporal_constraints",
    "skills.build_group_stratified_folds": "build_group_stratified_folds",
    "features.hayashi_yoshida_covariance": "hayashi_yoshida_covariance",
    "features.detrended_fluctuation": "detrended_fluctuation",
    "skills.verify_merkle_inclusion": "verify_merkle_inclusion",
    "skills.verify_merkle_consistency": "verify_merkle_consistency",
    "plugins.inspect_sqlite": "inspect_sqlite",
    "features.estimate_product_limit": "estimate_product_limit",
    "skills.score_probability_kernel": "score_probability_kernel",
    "skills.score_competing_risks": "score_competing_risks",
    "skills.align_event_sequences": "align_event_sequences",
    "skills.audit_partition_ranges": "audit_partition_ranges",
    "skills.audit_nested_folds": "audit_nested_folds",
    "skills.build_stationary_resamples": "build_stationary_resamples",
    "features.sample_entropy": "sample_entropy",
    "features.burg_autoregression": "burg_autoregression",
    "plugins.read_fixed_width": "read_fixed_width",
    "plugins.read_libsvm": "read_libsvm",
    "plugins.inspect_netcdf_classic": "inspect_netcdf_classic",
    "skills.score_hierarchical_probabilities": "score_hierarchical_probabilities",
    "skills.score_gaussian_forecasts": "score_gaussian_forecasts",
    "features.principal_coordinates": "principal_coordinates",
    "skills.audit_source_chains": "audit_source_chains",
    "skills.audit_temporal_resamples": "audit_temporal_resamples",
    "skills.audit_schema_compatibility": "audit_schema_compatibility",
    "features.circular_summary": "circular_summary",
    "features.weighted_geometric_median": "weighted_geometric_median",
    "features.partial_autocorrelation": "partial_autocorrelation",
    "plugins.read_arff": "read_arff",
    "plugins.read_dbase": "read_dbase",
    "plugins.read_sparse_npz": "read_sparse_npz",
    "features.weighted_quantile_binning": "weighted_quantile_binning",
    "skills.score_joint_categorical": "score_joint_categorical",
    "skills.score_count_forecasts": "score_count_forecasts",
    "skills.audit_price_basis": "audit_price_basis",
    "skills.reconcile_aggregates": "reconcile_aggregates",
    "skills.audit_unit_conversions": "audit_unit_conversions",
    "features.distance_correlation": "distance_correlation",
    "features.kernel_density_grid": "kernel_density_grid",
    "features.lomb_scargle_periodogram": "lomb_scargle_periodogram",
    "plugins.read_cbor": "read_cbor",
    "plugins.read_messagepack": "read_messagepack",
    "plugins.read_bson": "read_bson",
    "features.empirical_copula": "empirical_copula",
    "features.compositional_logratios": "compositional_logratios",
    "skills.score_piecewise_hazards": "score_piecewise_hazards",
    "skills.audit_transaction_batches": "audit_transaction_batches",
    "skills.audit_observation_windows": "audit_observation_windows",
    "skills.audit_dependency_releases": "audit_dependency_releases",
    "features.segment_mean_changes": "segment_mean_changes",
    "features.dynamic_time_warping": "dynamic_time_warping",
    "features.singular_spectrum_analysis": "singular_spectrum_analysis",
    "plugins.read_avro_container": "read_avro_container",
    "plugins.inspect_hdf5_superblock": "inspect_hdf5_superblock",
    "plugins.read_fits_table": "read_fits_table",
    "features.poisson_binomial_distribution": "poisson_binomial_distribution",
    "skills.score_markov_paths": "score_markov_paths",
    "skills.score_gaussian_mixtures": "score_gaussian_mixtures",
    "skills.audit_source_migrations": "audit_source_migrations",
    "skills.audit_artifact_assembly": "audit_artifact_assembly",
    "skills.audit_snapshot_manifests": "audit_snapshot_manifests",
    "features.markov_absorption": "markov_absorption",
    "features.linear_gaussian_filter": "linear_gaussian_filter",
    "features.graph_effective_resistance": "graph_effective_resistance",
    "plugins.read_wav_pcm": "read_wav_pcm",
    "plugins.read_ply": "read_ply",
    "plugins.inspect_tiff_directory": "inspect_tiff_directory",
    "features.convex_hull_2d": "convex_hull_2d",
    "features.monotone_cubic_interpolation": "monotone_cubic_interpolation",
    "features.discrete_optimal_transport": "discrete_optimal_transport",
    # Batch 13 — previously unregistered implementations. Behavioral cases exist in
    # tests/fx1/operation_cases_chunk{5,7,8}.py; see docs/FX1_CAPABILITY_BASELINE.md
    # for the reconciliation that identified these and docs/FX1_CAPABILITY_PROGRESS.md:632
    # which previously labeled them "unimplemented work directions".
    "features.linear_constraint_feasibility": "linear_constraint_feasibility",
    "features.maximum_flow": "maximum_flow",
    "features.minimum_spanning_forest": "minimum_spanning_forest",
    "features.multilinear_grid_interpolation": "multilinear_grid_interpolation",
    "features.set_function_attribution": "set_function_attribution",
    "features.thin_plate_spline": "thin_plate_spline",
    "plugins.inspect_zarr_metadata": "inspect_zarr_metadata",
    "plugins.read_nrrd": "read_nrrd",
    "plugins.read_stl": "read_stl",
    "skills.audit_resource_reservations": "audit_resource_reservations",
    "skills.audit_task_leases": "audit_task_leases",
    "skills.reconstruct_order_book": "reconstruct_order_book",
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
