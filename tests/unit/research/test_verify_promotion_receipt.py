"""``verify-research`` acceptance of composed promotion receipts (e2e) + negatives.

The happy path composes a real promotion receipt — verified artifact manifest
with dataset identity, ``evidence_report.v1`` written by the reporting stack
with its hash sidecar, an approved ``promotion.v1`` decision, all gate results,
and an honest approver — and asserts ``verify_research_artifact`` (the exact
function behind ``dipcatcher verify-research``) accepts it.

Every negative reseals its tampered variant with ``seal_receipt`` first, so the
deep semantic binding checks are exercised rather than only the self-seal.

All rows here are SYNTHETIC correctness fixtures; ``data_source: "file"``
follows the established positive-path convention of
``tests/unit/hedge_lab/test_promotion_gate.py`` and claims no market evidence.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import PromotionConfig, ValidationConfig
from quant_fund.pipeline.artifact_manifest import (
    identity_for_training,
    save_training_artifact,
)
from quant_fund.proof.promotion_receipt import compose_promotion_receipt
from quant_fund.registry.mlflow_store import promotion_decision
from quant_fund.reporting.report import write_evidence_report
from quant_fund.research.receipt_v2 import seal_receipt
from quant_fund.research.verify import (
    verify_promotion_receipt,
    verify_research_artifact,
)
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


def _compose_receipt(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Compose one fully-bound, sealed promotion receipt under ``tmp_path``."""
    frame = _frame()
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.validation = ValidationConfig(
        scheme="expanding", train_bars=16, val_bars=4, test_bars=4, seed=42
    )
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
        "data_source": "file",
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
    report_paths = write_evidence_report(
        tmp_path,
        candidates={"ranking": {"mean_ic": 0.5}},
        provenance={
            "data_source": "file",
            "artifact_path": str(artifact),
            "artifact_sha256": hash_file(artifact),
            "manifest_valid": True,
            "config_sha256": identity.config_sha256,
            "dataset_content_sha256": identity.dataset.materialized_panel_sha256,
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
        },
        health={
            "status": "ok",
            "report": "unit-test-fixture",
            "note": "SYNTHETIC correctness fixture; not market evidence",
        },
        promotion=decision,
    )
    report_path = next(path for path in report_paths.values() if path.suffix == ".json")
    out_path = tmp_path / "promotion_receipts" / "decision.json"
    sealed = compose_promotion_receipt(
        artifact=artifact,
        evidence_report=report_path,
        decision=decision,
        input_metrics=metrics,
        gate_results=_gates(),
        approver=_approver(),
        out_path=out_path,
    )
    return out_path, sealed


def _reseal_variant(tmp_path: Path, sealed: dict[str, object], name: str, mutate) -> Path:
    """Copy a sealed receipt, apply ``mutate`` and RESEAL it (deep checks)."""
    body = copy.deepcopy(sealed)
    mutate(body)
    variant = tmp_path / "variants" / f"{name}.json"
    variant.parent.mkdir(parents=True, exist_ok=True)
    variant.write_text(json.dumps(seal_receipt(body), indent=2, sort_keys=True), encoding="utf-8")
    return variant


def test_verify_research_artifact_accepts_promotion_receipt(tmp_path: Path) -> None:
    out_path, _sealed = _compose_receipt(tmp_path)
    result = verify_research_artifact(out_path)
    assert result["valid"] is True, result["errors"]
    assert result["errors"] == []
    assert result["claim"] == "research_only"
    assert result["run_id"] == "run-fixture-0001"
    alias = verify_promotion_receipt(out_path)
    assert alias["valid"] is True


def test_dispatch_is_content_based_not_schema_claimed(tmp_path: Path) -> None:
    out_path, _sealed = _compose_receipt(tmp_path)
    body = json.loads(out_path.read_text(encoding="utf-8"))
    body["kind"] = "notebook_v4"  # claim steering must not dodge promotion checks
    variant = tmp_path / "variants" / "steered.json"
    variant.parent.mkdir(parents=True, exist_ok=True)
    variant.write_text(json.dumps(seal_receipt(body)), encoding="utf-8")
    assert verify_research_artifact(variant)["valid"] is False
    del out_path


def test_unresealed_tampering_is_rejected_by_seal(tmp_path: Path) -> None:
    out_path, sealed = _compose_receipt(tmp_path)
    body = copy.deepcopy(sealed)
    payload = body["payload"]
    assert isinstance(payload, dict)
    payload["gate_results"] = {}
    variant = tmp_path / "variants" / "unsealed.json"
    variant.parent.mkdir(parents=True, exist_ok=True)
    variant.write_text(json.dumps(body), encoding="utf-8")  # NOT resealed
    result = verify_research_artifact(variant)
    assert result["valid"] is False
    assert "promotion_receipt_seal_mismatch" in result["errors"]


NEGATIVE_CASES = (
    (
        "tampered_artifact_payload_hash",
        lambda body: body["payload"]["artifact_identity"].update(artifact_sha256="0" * 64),
        "promotion_artifact_sha256_mismatch",
    ),
    (
        "tampered_evidence_report_hash",
        lambda body: body["payload"]["evidence_report"].update(sha256="0" * 64),
        "promotion_evidence_report_sha256_mismatch",
    ),
    (
        "missing_gate_result",
        lambda body: body["payload"]["gate_results"].pop("leakage"),
        "promotion_gate_result_missing:leakage",
    ),
    (
        "failing_gate_result",
        lambda body: body["payload"]["gate_results"]["dataset_identity"].update(status="fail"),
        "promotion_gate_result_not_passing:dataset_identity",
    ),
    (
        "synthetic_evidence",
        lambda body: body["payload"]["input_metrics"].update(data_source="SYNTHETIC"),
        "promotion_synthetic_evidence",
    ),
    (
        "missing_approver",
        lambda body: body["payload"].update(approver={}),
        "promotion_approver_dishonest",
    ),
    (
        "dishonest_approver",
        lambda body: body["payload"]["approver"].update(name="someone"),
        "promotion_approver_dishonest",
    ),
    (
        "absent_dataset_identity",
        lambda body: body["payload"]["artifact_identity"].pop("identity"),
        "promotion_identity_missing",
    ),
    (
        "stale_dataset_identity",
        lambda body: body["payload"]["input_metrics"].update(dataset_content_sha256="0" * 64),
        "promotion_input_metrics_dataset_stale_or_absent",
    ),
    (
        "identity_drifted_from_manifest",
        lambda body: body["payload"]["artifact_identity"]["identity"]["dataset"].update(
            materialized_panel_sha256="0" * 64
        ),
        "promotion_identity_stale_or_tampered",
    ),
    (
        "research_claim_replaced",
        lambda body: body["payload"].update(live_pnl_claim=True),
        "promotion_live_pnl_claim",
    ),
)


@pytest.mark.parametrize(("name", "mutate", "expected"), NEGATIVE_CASES)
def test_negative_fail_closed(tmp_path: Path, name: str, mutate, expected: str) -> None:
    _out_path, sealed = _compose_receipt(tmp_path)
    variant = _reseal_variant(tmp_path, sealed, name, mutate)
    result = verify_research_artifact(variant)
    assert result["valid"] is False, f"{name} must fail closed"
    assert expected in result["errors"], (name, result["errors"])
    assert result["claim"] == "research_only"
