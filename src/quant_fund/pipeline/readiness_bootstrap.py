"""Plant a SYNTHETIC data manifest and research receipt so lab ``/ready`` can pass.

The receipt is labeled ``SYNTHETIC`` and ``research_only``. It is a startup
fixture for an empty data volume, not market evidence. An existing invalid
manifest or receipt is left in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from quant_fund import __firm__, __version__
from quant_fund.config import load_config
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, hash_file

_RUN_LABEL = b"dipcatcher-synthetic-readiness-bootstrap"
_GENERATED_AT = "2026-11-02T00:00:00+00:00"


class ReadinessBootstrapError(ValueError):
    """The tree could not be made ready. The message does not include file bodies."""


def _run_id() -> str:
    return hashlib.sha256(_RUN_LABEL).hexdigest()


def _existing_problem(root: Path) -> str | None:
    manifest = root / "metadata" / "data_manifest.json"
    receipt = root / "metadata" / "research" / "latest.json"
    if manifest.is_symlink() or receipt.is_symlink():
        return "refusing to replace a symlink"
    return None


def _write_manifest(root: Path, source: str) -> None:
    destination = root / "metadata" / "data_manifest.json"
    if destination.is_file():
        return
    frame_path = root / "silver" / "readiness_bootstrap.parquet"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"a": [1]}).write_parquet(frame_path)
    digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    artifacts = {
        name: {
            "path": str(frame_path.resolve()),
            "sha256": digest,
            "rows": 1,
            "columns": ["a"],
        }
        for name in ("actions", "bars", "master", "silver", "universe")
    }
    body = {"artifacts": artifacts, "schema_version": 1, "source": source}
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")


def _receipt_payload(root: Path, run_id: str) -> dict[str, object]:
    from quant_fund.research.catalog import (
        BENCHMARK_CATALOG_VERSION,
        REQUIRED_BENCHMARK_FAMILIES,
        RESEARCH_RECEIPT_SCHEMA_VERSION,
    )
    from quant_fund.research.receipt_schema import unavailable_overfitting_block

    receipt_dir = root / "metadata" / "research"
    markdown = receipt_dir / f"{run_id}.md"
    payload: dict[str, object] = {
        "schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "firm": __firm__,
        "product": "Dipcatcher",
        "version": __version__,
        "generated_at": _GENERATED_AT,
        "data_source": "SYNTHETIC",
        "synthetic": True,
        "disclaimer": "SYNTHETIC readiness bootstrap. Not market evidence.",
        "ranking_target": "future_return_1",
        "claim": "research_only",
        "rankers": [],
        "hypotheses": [],
        "provenance": {
            "run_id": run_id,
            "git_revision": "readiness-bootstrap",
            "git_worktree_sha256": "e" * 64,
            "config_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "dataset_content_sha256": "d" * 64,
            "northset_inputs_sha256": "e" * 64,
            "row_count": 1,
            "column_count": 1,
            "point_in_time": True,
            "execution_claim": "research_only",
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runtime": {
                "python": "3.12.0",
                "implementation": "CPython",
                "platform": "bootstrap",
                "machine": "bootstrap",
                "byteorder": "little",
                "packages": {
                    "numpy": "bootstrap",
                    "polars": "bootstrap",
                    "scipy": "bootstrap",
                    "scikit-learn": "bootstrap",
                },
            },
        },
        "scorecard": {
            name: {
                "executed": True,
                "nonempty": True,
                "finite_observation": True,
                "forbidden_metrics_absent": True,
                "claim": "research_metric_only",
            }
            for name in REQUIRED_BENCHMARK_FAMILIES
        },
        "families": {name: {"executed": True} for name in REQUIRED_BENCHMARK_FAMILIES},
        "backtest_overfitting": unavailable_overfitting_block(),
        "artifacts": {
            "immutable_json": f"{run_id}.json",
            "immutable_markdown": markdown.name,
            "immutable_markdown_sha256": hash_file(markdown),
        },
    }
    return payload


def _verification_codes(verification: dict[str, object]) -> str:
    errors = verification.get("errors")
    if not isinstance(errors, list):
        return "unknown"
    return ",".join(str(item) for item in errors[:20])


def _write_receipt(root: Path) -> None:
    from quant_fund.research.verify import _receipt_digest, verify_research_artifact

    receipt_dir = root / "metadata" / "research"
    latest = receipt_dir / "latest.json"
    if latest.is_file():
        return
    receipt_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    markdown = receipt_dir / f"{run_id}.md"
    markdown.write_text("# SYNTHETIC readiness bootstrap\n", encoding="utf-8")
    payload = _receipt_payload(root, run_id)
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict):
        raise ReadinessBootstrapError("receipt artifacts were not written")
    artifacts["immutable_json_sha256"] = _receipt_digest(payload)
    text = json.dumps(payload, sort_keys=True)
    latest.write_text(text, encoding="utf-8")
    (receipt_dir / f"{run_id}.json").write_text(text, encoding="utf-8")
    verification = verify_research_artifact(latest)
    if verification.get("valid") is not True:
        latest.unlink(missing_ok=True)
        codes = _verification_codes(verification)
        raise ReadinessBootstrapError(f"synthetic research receipt failed verification:{codes}")


def bootstrap_readiness(config_path: Path) -> dict[str, object]:
    """Write missing SYNTHETIC readiness files under the config's data root."""
    from quant_fund.pipeline.doctor import doctor as run_doctor

    cfg = load_config(config_path)
    root = Path(cfg.data.root)
    problem = _existing_problem(root)
    if problem:
        raise ReadinessBootstrapError(problem)
    _write_manifest(root, str(cfg.data.source))
    _write_receipt(root)
    report = run_doctor(str(config_path))
    if report.get("data_manifest") == "invalid":
        raise ReadinessBootstrapError("data manifest is invalid and was left unchanged")
    if report.get("research_receipt") == "invalid":
        raise ReadinessBootstrapError("research receipt is invalid and was left unchanged")
    return report
