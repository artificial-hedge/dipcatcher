"""Research notebook artifact contract, publication, and report summaries.

Serialization keeps unavailable numeric observations explicit and report text
retains the research-only labels used by the lab runner.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np

from quant_fund.research.catalog import (
    dist_crps_eprocess_keys_present,
    family_blob_executed,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
    family_blob_nonempty,
    tail_es_battery_keys_present,
    tail_var_battery_keys_present,
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete text artifact without exposing partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass
class HypothesisResult:
    id: str
    statement: str
    test: str
    statistic: float
    p_value: float
    reject_raw: bool
    reject_fdr: bool
    decision: str
    family: str = "discovery"
    meets_floor: bool | None = None


@dataclass
class ResearchNotebook:
    schema_version: int
    firm: str
    product: str
    version: str
    generated_at: str
    data_source: str
    synthetic: bool
    disclaimer: str
    ranking_target: str
    claim: str
    families: dict[str, Any]
    rankers: list[dict[str, Any]]
    hypotheses: list[HypothesisResult]
    scorecard: dict[str, dict[str, Any]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    backtest_overfitting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))


def format_p_value(value: float) -> str:
    """Render p-values without turning floating-point underflow into ``0``."""
    p = float(value)
    if not np.isfinite(p):
        return "n/a"
    if p <= 0.0:
        return f"<{np.finfo(float).tiny:.3g}"
    return f"{p:.4g}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None:
        return None
    return str(obj)


def _benchmark_scorecard(families: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Classify benchmark evidence without turning diagnostics into passes."""

    scorecard: dict[str, dict[str, Any]] = {}
    for name, payload in families.items():
        entry: dict[str, Any] = {
            "executed": family_blob_executed(payload),
            "nonempty": family_blob_nonempty(payload),
            "finite_observation": family_blob_has_finite_observation(payload),
            "forbidden_metrics_absent": family_blob_forbidden_metrics_absent(payload),
            "claim": "research_metric_only",
        }
        # Soft research diagnostic (Day Wave 18/20/21): never a live promotion gate.
        if name == "tail":
            entry["tail_var_battery_ok"] = tail_var_battery_keys_present(payload)
            entry["tail_es_battery_ok"] = tail_es_battery_keys_present(payload)
        if name == "distribution":
            entry["dist_crps_eprocess_ok"] = dist_crps_eprocess_keys_present(payload)
        scorecard[name] = entry
    return scorecard


def _overfitting_section(block: object) -> str:
    """One-line notebook summary of the backtest-overfitting diagnostics."""
    if not isinstance(block, dict) or not block:
        return "unavailable"

    def _fmt(value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "n/a"
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:.4g}"

    return (
        f"PBO={_fmt(block.get('pbo'))} DSR={_fmt(block.get('dsr'))} "
        f"PSR={_fmt(block.get('psr'))} MinTRL={_fmt(block.get('min_trl'))} "
        f"n_trials={block.get('n_trials')} "
        f"n_trials_effective={block.get('n_trials_effective')} "
        f"research_diagnostic_only"
    )


def _data_snooping_section(blob: object) -> str:
    """One-line notebook summary of the ranker data-snooping battery."""
    if not isinstance(blob, dict) or not blob:
        return "unavailable (needs >=2 rankers with >=10 aligned dates)"
    rc_p = float(blob.get("reality_check_p", float("nan")))
    spa_p = float(blob.get("spa_p_consistent", float("nan")))
    return (
        f"n_trials={blob.get('n_trials')} best={blob.get('best_trial')} "
        f"RC p={format_p_value(rc_p)} SPA(cons) p={format_p_value(spa_p)} "
        f"StepM rejected={blob.get('stepm_n_rejected')} "
        f"MCS included={blob.get('mcs_n_included')}"
    )
