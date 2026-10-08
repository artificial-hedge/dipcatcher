"""Fail-closed promotion-receipt composition.

A promotion receipt (``promotion_receipt.v1``) can only be composed when the
artifact manifest carries a verified dataset identity, the decision is
approved and non-synthetic, every gate result is present and passing, the
evidence report binds the same artifact and dataset, and an honest approving
identity is recorded. The first test is the mandated one: missing dataset
identity blocks promotion.

All rows here are SYNTHETIC correctness fixtures (``config.data.source ==
"synthetic"``); the ``data_source: "file"`` marker on the promotion inputs
follows the established positive-path convention of
``tests/unit/hedge_lab/test_promotion_gate.py`` and never claims market
evidence.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import PromotionConfig, ValidationConfig
from quant_fund.models.base import save_joblib_artifact
from quant_fund.pipeline.artifact_manifest import (
    identity_for_training,
    save_training_artifact,
)
from quant_fund.proof.promotion_receipt import (
    PromotionCompositionError,
    compose_promotion_receipt,
)
from quant_fund.registry.mlflow_store import promotion_decision
from quant_fund.reporting.report import write_evidence_report
from quant_fund.utils.hashing import hash_file
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

D0 = datetime(2020, 1, 1, tzinfo=UTC)


def _frame(n_dates: int = 64, n_secs: int = 6) -> pl.DataFrame:
    rng = np.random.default_rng(20261007)
    rows = []
    for day in range(n_dates):
        stamp = D0 + timedelta(days=day)
        for sec in range(n_secs):
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{sec:02d}",
                    "ret_1": float(rng.normal(0.0004, 0.01)),
                    "vol_20": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "mom_20": float(rng.normal(0, 0.01)),
                    "reversal_1": float(rng.normal(0, 0.01)),
                    "future_return_5": float(rng.normal(0, 0.02)),
                }
            )
    return pl.DataFrame(rows)


def _materialize_sources(root: Path, frame: pl.DataFrame) -> None:
    (root / "gold").mkdir(parents=True, exist_ok=True)
    (root / "silver").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(root / "gold" / "features.parquet")
    frame.select(["event_time", "security_id", "future_return_5"]).write_parquet(
        root / "gold" / "labels.parquet"
    )
    frame.select(["event_time", "security_id"]).unique().write_parquet(
        root / "silver" / "universe.parquet"
    )


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.validation = ValidationConfig(
        scheme="expanding", train_bars=16, val_bars=4, test_bars=4, seed=42
    )
    return cfg


def _approver() -> dict[str, str]:
    return {
        "name": "integration-lead",
        "role": "team_lead",
        "decided_at": datetime.now(UTC).isoformat(),
    }


def _gates() -> dict[str, dict[str, str]]:
    stamp = datetime.now(UTC).isoformat()
    return {
        name: {"status": "pass", "checked_at": stamp}
        for name in (
            "artifact_manifest",
            "dataset_identity",
            "evidence_report",
            "research_receipt",
            "leakage",
        )
    }


def _build(
    tmp_path: Path,
    *,
    data_source: str = "file",
    include_promotion: bool = True,
    include_health: bool = True,
) -> dict[str, object]:
    """Build a fully-bound artifact + report + approved decision fixture."""
    frame = _frame()
    cfg = _cfg(tmp_path)
    _materialize_sources(tmp_path, frame)
    artifact = tmp_path / "ranker_fixture.joblib"
    identity = identity_for_training(
        frame,
        config=cfg,
        label="future_return_5",
        features=["mom_20", "vol_20", "reversal_1"],
        label_horizon_bars=5,
    )
    save_training_artifact({"model": 1}, artifact, identity=identity)
    metrics: dict[str, object] = {
        "evidence_complete": True,
        "data_source": data_source,
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
        "fold_ic_stability": 1.0,
        "run_id": "run-fixture-0001",
        "artifact_sha256": hash_file(artifact),
        "manifest_valid": True,
        "research_receipt_valid": True,
        "dataset_content_sha256": identity.dataset.materialized_panel_sha256,
        "synthetic": False,
    }
    decision = promotion_decision(dict(metrics), PromotionConfig(min_folds=2), leakage_ok=True)
    extra_sections: dict[str, object] = {}
    if include_health:
        extra_sections["health"] = {
            "status": "ok",
            "report": "unit-test-fixture",
            "note": "SYNTHETIC correctness fixture; not market evidence",
        }
    if include_promotion:
        extra_sections["promotion"] = decision
    report_paths = write_evidence_report(
        tmp_path,
        candidates={"ranking": {"mean_ic": 0.5}},
        provenance={
            "data_source": data_source,
            "artifact_path": str(artifact),
            "artifact_sha256": hash_file(artifact),
            "manifest_valid": True,
            "config_sha256": identity.config_sha256,
            "dataset_content_sha256": identity.dataset.materialized_panel_sha256,
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
        },
        **extra_sections,
    )
    report_path = next(path for path in report_paths.values() if path.suffix == ".json")
    return {
        "frame": frame,
        "cfg": cfg,
        "artifact": artifact,
        "identity": identity,
        "metrics": metrics,
        "decision": decision,
        "report": report_path,
    }


def _compose(tmp_path: Path, built: dict[str, object], **overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "artifact": built["artifact"],
        "evidence_report": built["report"],
        "decision": built["decision"],
        "input_metrics": built["metrics"],
        "gate_results": _gates(),
        "approver": _approver(),
        "out_path": tmp_path / "promotion_receipts" / "decision.json",
    }
    args.update(overrides)
    return compose_promotion_receipt(**args)  # type: ignore[arg-type]


def test_missing_dataset_identity_blocks_promotion(tmp_path: Path) -> None:
    """Mandated fail-closed test: no dataset identity, no promotion receipt."""
    built = _build(tmp_path)
    bare = tmp_path / "bare.joblib"
    save_joblib_artifact({"model": 2}, bare)  # artifact without any identity block
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built, artifact=bare)


def test_compose_seals_immutable_receipt_with_sidecar(tmp_path: Path) -> None:
    built = _build(tmp_path)
    sealed = _compose(tmp_path, built)
    out_path = tmp_path / "promotion_receipts" / "decision.json"
    assert out_path.is_file()
    assert sealed["receipt_sha256"]
    sidecar = out_path.with_name(f"{out_path.name}.sha256")
    assert sidecar.read_text(encoding="ascii").strip() == hash_file(out_path)
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built)  # receipts are immutable


def test_training_time_report_stage_warning_is_resolved_by_receipt(tmp_path: Path) -> None:
    """Stage-aware scoping: a report missing only this receipt composes.

    The evidence report is written at training time; ``promotion_receipt_missing``
    is the expected stage warning and the composed receipt resolves it. No
    report warning is removed and ``status: complete`` is not loosened.
    """
    built = _build(tmp_path, include_promotion=False)
    report = built["report"]
    assert isinstance(report, Path)
    body = json.loads(report.read_text(encoding="utf-8"))
    assert body["warnings"] == ["promotion_receipt_missing"]
    sealed = _compose(tmp_path, built)
    payload = sealed["payload"]
    assert isinstance(payload, dict)
    binding = payload["evidence_report"]
    assert binding["status_at_composition"] == "insufficient_evidence"
    assert binding["stage_warnings"] == ["promotion_receipt_missing"]
    assert binding["resolved_by"] == "promotion_receipt.v1"


def test_missing_health_report_still_blocks_promotion(tmp_path: Path) -> None:
    """health_report_missing stays blocking — only the stage warning is scoped."""
    built = _build(tmp_path, include_promotion=False, include_health=False)
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built)


def test_tampered_artifact_payload_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    artifact = built["artifact"]
    assert isinstance(artifact, Path)
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built)


def test_tampered_evidence_report_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    report = built["report"]
    assert isinstance(report, Path)
    body = json.loads(report.read_text(encoding="utf-8"))
    body["status"] = "insufficient_evidence"
    report.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built)


def test_missing_gate_result_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    gates = _gates()
    del gates["leakage"]
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built, gate_results=gates)


def test_failing_gate_result_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    gates = _gates()
    gates["dataset_identity"] = {
        "status": "fail",
        "checked_at": gates["dataset_identity"]["checked_at"],
    }
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built, gate_results=gates)


def test_synthetic_evidence_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path, data_source="SYNTHETIC")
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built)


def test_missing_approver_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built, approver={})


def test_dishonest_approver_blocks_promotion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    for name in ("unknown", "someone", "n/a", ""):
        approver = _approver()
        approver["name"] = name
        with pytest.raises(PromotionCompositionError):
            _compose(tmp_path, built, approver=approver, out_path=tmp_path / f"pr_{len(name)}.json")


def test_stale_dataset_identity_blocks_promotion(tmp_path: Path) -> None:
    """Input metrics bound to an older dataset than the manifest must not pass."""
    built = _build(tmp_path)
    metrics = dict(built["metrics"])
    metrics["dataset_content_sha256"] = "0" * 64
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built, input_metrics=metrics)


def test_payload_identity_must_match_manifest(tmp_path: Path) -> None:
    built = _build(tmp_path)
    artifact = built["artifact"]
    assert isinstance(artifact, Path)
    manifest_path = artifact.with_name(f"{artifact.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["dataset"]["materialized_panel_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PromotionCompositionError):
        _compose(tmp_path, built)
