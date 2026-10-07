"""Full promotion-receipt composition — one immutable, hash-bound decision.

A promotion receipt is the single composed record that binds, in one sealed
artifact:

* the **artifact identity** — payload hash, artifact class, feature identity,
  label/horizon identity and full dataset identity (materialized-panel hash,
  source-manifest hash, row counts, time range) as verified from the
  ``model_artifact.v1`` manifest;
* the **evidence report** — the exact ``evidence_report.v1`` bytes by hash;
* the **promotion decision** — the ``promotion.v1`` decision block, its input
  metrics and every gate result;
* the **approving identity** — a named, role-stamped approver with a
  timestamped decision.

Composition is fail-closed end to end: a missing or unverified dataset
identity, a tampered or incomplete evidence report, an unapproved decision,
a missing/failed gate result, synthetic evidence, or a missing/dishonest
approver all raise :class:`PromotionCompositionError` before any bytes are
written. The output is a sealed ``receipt.v2`` envelope whose payload carries
the ``promotion_receipt.v1`` body; the file is written exactly once
(refusing to overwrite) with a ``.sha256`` file sidecar.

Verification is intentionally NOT done here: the independent check lives in
``quant_fund.research.verify`` (``verify-research``), which re-derives every
binding from the files on disk. Composition-side gating and verification-side
checking are separate code paths on purpose.

Research-only: this authorizes a research/paper-challenger promotion at most.
``live_pnl_claim`` is hard-coded ``False``; nothing here authorizes live
trading.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_fund.pipeline.artifact_manifest import (
    ArtifactIdentity,
    ArtifactManifestError,
    manifest_path,
    verify_artifact_manifest,
)
from quant_fund.registry.mlflow_store import promotion_is_approved
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt
from quant_fund.utils.hashing import hash_file

PROMOTION_RECEIPT_SCHEMA: Final[str] = "promotion_receipt.v1"
PROMOTION_RECEIPT_KIND: Final[str] = "promotion_receipt"
PROMOTION_RECEIPT_DATA_LABEL: Final[str] = "promotion_receipt.v1"
EVIDENCE_REPORT_SCHEMA: Final[str] = "evidence_report.v1"
# Every gate must be present and passing; a missing gate result blocks.
REQUIRED_PROMOTION_GATES: Final[tuple[str, ...]] = (
    "artifact_manifest",
    "dataset_identity",
    "evidence_report",
    "research_receipt",
    "leakage",
)
# Names that are not an approving identity.
DISHONEST_APPROVER_NAMES: Final[frozenset[str]] = frozenset(
    {"", "unknown", "anonymous", "none", "null", "n/a", "na", "tbd", "unspecified", "someone"}
)


class PromotionCompositionError(RuntimeError):
    """Raised when a promotion receipt cannot be composed fail-closed."""


class Approver(BaseModel):
    """The approving identity for a promotion decision."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    decided_at: str = Field(min_length=1)

    @field_validator("name", "role")
    @classmethod
    def _honest_identity(cls, value: str) -> str:
        stripped = value.strip()
        if stripped.lower() in DISHONEST_APPROVER_NAMES:
            raise ValueError(f"approver identity is not honest: {value!r}")
        return stripped

    @field_validator("decided_at")
    @classmethod
    def _timestamped(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"approver decided_at is not ISO-8601: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("approver decided_at must carry an explicit timezone")
        return value


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PromotionCompositionError(reason)


def _load_evidence_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PromotionCompositionError(f"evidence report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionCompositionError(f"evidence report is unreadable: {path}") from exc
    if not isinstance(report, dict):
        raise PromotionCompositionError(f"evidence report is not an object: {path}")
    return report


def _gate_results_checked(
    gate_results: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for name, row in gate_results.items():
        if not isinstance(name, str) or not name.strip():
            raise PromotionCompositionError("gate result names must be nonempty")
        if not isinstance(row, Mapping):
            raise PromotionCompositionError(f"gate result {name!r} is not an object")
        status = row.get("status")
        if status != "pass":
            raise PromotionCompositionError(f"gate result {name!r} is not passing: {status!r}")
        normalized[name] = {str(key): str(value) for key, value in row.items()}
    missing = sorted(set(REQUIRED_PROMOTION_GATES) - set(normalized))
    if missing:
        raise PromotionCompositionError(f"gate results missing: {','.join(missing)}")
    return normalized


def compose_promotion_receipt(
    *,
    artifact: Path,
    evidence_report: Path,
    decision: Mapping[str, Any],
    input_metrics: Mapping[str, Any],
    gate_results: Mapping[str, Mapping[str, str]],
    approver: Approver | Mapping[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Compose, seal and persist one immutable promotion receipt.

    Every input is re-derived from disk where a hash is claimed. A missing or
    unverified dataset identity on the artifact manifest blocks composition.
    """
    destination = Path(out_path)
    if destination.exists():
        raise PromotionCompositionError(
            f"promotion receipts are immutable; refusing to overwrite: {destination}"
        )

    # 1. Artifact identity — including dataset identity. This is the gate a
    #    missing/unverified dataset identity cannot pass.
    try:
        identity: ArtifactIdentity = verify_artifact_manifest(Path(artifact))
    except ArtifactManifestError as exc:
        raise PromotionCompositionError(f"artifact identity unverified: {exc}") from exc
    artifact_path = Path(artifact)
    artifact_sha = hash_file(artifact_path)
    manifest_file = manifest_path(artifact_path)
    manifest_sha = hash_file(manifest_file)
    try:
        manifest_record = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionCompositionError(
            f"artifact manifest is unreadable: {manifest_file}"
        ) from exc
    artifact_class = manifest_record.get("class") if isinstance(manifest_record, dict) else None

    # 2. The decision itself must already be an approved promotion.v1.
    if not promotion_is_approved(dict(decision)):
        raise PromotionCompositionError("promotion decision is not an approved promotion.v1")

    # 3. Input metrics must bind the same artifact and the same dataset.
    metrics_data_source = input_metrics.get("data_source")
    _require(
        isinstance(metrics_data_source, str) and metrics_data_source.strip() != "",
        "input metrics carry no data source",
    )
    _require(
        str(metrics_data_source).strip().upper() != "SYNTHETIC",
        "synthetic evidence is not promotable",
    )
    _require(input_metrics.get("synthetic") is not True, "synthetic evidence is not promotable")
    _require(
        input_metrics.get("artifact_sha256") == artifact_sha,
        "input metrics artifact hash does not match the artifact",
    )
    _require(
        input_metrics.get("dataset_content_sha256") == identity.dataset.materialized_panel_sha256,
        "input metrics dataset identity is stale or absent",
    )

    # 4. Gate results — all required gates present and passing.
    gates = _gate_results_checked(gate_results)

    # 5. Approving identity.
    approver_record = (
        approver if isinstance(approver, Approver) else Approver.model_validate(dict(approver))
    )

    # 6. Evidence report — hash-bound and cross-bound to this artifact/dataset.
    report_path = Path(evidence_report)
    report = _load_evidence_report(report_path)
    report_sha = hash_file(report_path)
    sidecar = report_path.with_name(f"{report_path.name}.sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="ascii").strip()
        _require(expected == report_sha, "evidence report hash sidecar mismatch")
    _require(
        report.get("schema") == EVIDENCE_REPORT_SCHEMA,
        "evidence report schema mismatch",
    )
    _require(report.get("research_only") is True, "evidence report is not research-only")
    _require(report.get("live_pnl_claim") is False, "evidence report claims live P&L")
    _require(
        report.get("status") == "complete",
        "evidence report is incomplete; promotion is blocked",
    )
    warnings = report.get("warnings")
    _require(
        not (isinstance(warnings, list) and "synthetic_evidence_not_promotable" in warnings),
        "synthetic evidence is not promotable",
    )
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PromotionCompositionError("evidence report has no provenance")
    _require(
        provenance.get("artifact_sha256") == artifact_sha,
        "evidence report artifact hash does not match the artifact",
    )
    _require(
        provenance.get("dataset_content_sha256") == identity.dataset.materialized_panel_sha256,
        "evidence report dataset identity is stale or absent",
    )
    _require(
        provenance.get("manifest_valid") is True,
        "evidence report does not bind a verified artifact manifest",
    )

    payload: dict[str, Any] = {
        "schema": PROMOTION_RECEIPT_SCHEMA,
        "claim": "research_only",
        "execution_claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_identity": {
            "path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha,
            "artifact_class": artifact_class,
            "identity": identity.model_dump(mode="json"),
        },
        "evidence_report": {
            "path": str(report_path),
            "sha256": report_sha,
            "schema": EVIDENCE_REPORT_SCHEMA,
        },
        "promotion_decision": dict(decision),
        "input_metrics": dict(input_metrics),
        "gate_results": gates,
        "approver": approver_record.model_dump(mode="json"),
    }

    receipt = build_receipt_v2(
        kind=PROMOTION_RECEIPT_KIND,
        data_label=PROMOTION_RECEIPT_DATA_LABEL,
        dataset={
            "artifact_sha256": str(artifact_sha),
            "materialized_panel_sha256": identity.dataset.materialized_panel_sha256,
            "evidence_report_sha256": report_sha,
        },
        params={
            "run_id": str(decision.get("run_id", "")),
            "approver": approver_record.name,
            "gates": sorted(gates),
        },
        code_files=(Path(__file__),),
        verdict="pass",
        payload=payload,
    )
    sealed = seal_receipt(receipt)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: an existing receipt is never rewritten.
    try:
        with open(destination, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PromotionCompositionError(
            f"promotion receipts are immutable; refusing to overwrite: {destination}"
        ) from exc

    file_sha = hash_file(destination)
    sidecar = destination.with_name(f"{destination.name}.sha256")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=sidecar.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            mode="w",
            encoding="ascii",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(file_sha + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, sidecar)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return sealed


__all__ = [
    "Approver",
    "DISHONEST_APPROVER_NAMES",
    "EVIDENCE_REPORT_SCHEMA",
    "PROMOTION_RECEIPT_DATA_LABEL",
    "PROMOTION_RECEIPT_KIND",
    "PROMOTION_RECEIPT_SCHEMA",
    "PromotionCompositionError",
    "REQUIRED_PROMOTION_GATES",
    "compose_promotion_receipt",
]
