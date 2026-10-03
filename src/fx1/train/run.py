"""Training-run manifest builder for fx-1.

This module does not launch GPUs. It enforces the pipeline contract —
eval-before-train, corpus provenance, cost disclosure — and emits an
immutable run manifest that a cluster launcher consumes. A run that cannot
produce a valid manifest does not start.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fx1.train.config import TrainConfig
from quant_fund.utils.atomicio import atomic_write_text


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_corpus(corpus_path: Path) -> dict[str, int]:
    """Every corpus line must carry receipt provenance. Fail-closed."""
    stats = {"lines": 0, "positive": 0, "negative": 0}
    with corpus_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("receipt_sha256"):
                raise ValueError(f"corpus line {lineno} lacks receipt_sha256 provenance")
            stats["lines"] += 1
            key = "negative" if record.get("negative") else "positive"
            stats[key] += 1
    if stats["lines"] == 0:
        raise ValueError("corpus is empty")
    return stats


def _validate_eval_gate(eval_path: Path) -> None:
    """Re-derive the gate from recorded results — the flag alone is not
    evidence: a hand-written ``honesty_gate_passed: true`` on an empty
    result set must not unblock training."""
    summary = json.loads(eval_path.read_text(encoding="utf-8"))
    raw_results = summary.get("results")
    results = raw_results if isinstance(raw_results, list) else []
    honesty = [r for r in results if isinstance(r, dict) and r.get("kind") == "honesty"]
    # Re-derive violations from EVERY place the suite records them: the
    # honesty:-prefixed task failures, the per-result honesty_violations
    # list (non-enforced tasks), and the summary-level list. Checking only
    # the flag and failures would let a doctored or stale summary pass.
    violations = [
        r
        for r in results
        if isinstance(r, dict)
        and (
            any(str(f).startswith("honesty:") for f in (r.get("failures") or []))
            or bool(r.get("honesty_violations"))
        )
    ]
    if (
        not summary.get("honesty_gate_passed", False)
        or not honesty
        or not all(r.get("passed") for r in honesty)
        or violations
        or summary.get("honesty_violations")
    ):
        raise ValueError(
            "eval harness honesty gate not passed — training is blocked "
            "until eval results with >=1 passing honesty task and zero "
            "contract violations are on record"
        )


def build_training_manifest(config: TrainConfig, out_path: str | Path) -> dict[str, Any]:
    """Validate the run contract and write an immutable manifest."""
    corpus_path = Path(config.corpus_jsonl)
    eval_path = Path(config.eval_results_json)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval results not found: {eval_path} — the eval harness runs before any training"
        )
    corpus_stats = _validate_corpus(corpus_path)
    _validate_eval_gate(eval_path)
    manifest = {
        "run_name": config.run_name,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "corpus_sha256": _file_sha256(corpus_path),
        "corpus_stats": corpus_stats,
        "eval_results_sha256": _file_sha256(eval_path),
        "live_pnl_claim": False,
        "research_only": True,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
