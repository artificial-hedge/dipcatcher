"""Corpus-builder tests: provenance, eligibility, fail-closed behavior."""

import hashlib
import json
from pathlib import Path

from fx1.data import build_corpus, load_receipts


def _write_receipt(path: Path, *, research_only: bool, live_pnl_claim: bool) -> None:
    payload = {
        "schema": "test/v1",
        "research_only": research_only,
        "live_pnl_claim": live_pnl_claim,
        "disclaimer": "test receipt",
        "correctness": {"metric": 1.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_manifest(
    path: Path,
    *,
    claim: str | None,
    synthetic: bool = True,
    live_pnl_claim: bool | None = None,
) -> None:
    payload: dict = {
        "schema_version": 1,
        "synthetic": synthetic,
        "disclaimer": "SYNTHETIC research; simulated correctness evidence only.",
        "families": {},
        "correctness": {"metric": 1.0},
    }
    if claim is not None:
        payload["claim"] = claim
    if live_pnl_claim is not None:
        payload["live_pnl_claim"] = live_pnl_claim
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_receipts_and_eligibility(tmp_path: Path):
    _write_receipt(tmp_path / "good.json", research_only=True, live_pnl_claim=False)
    _write_receipt(tmp_path / "bad.json", research_only=False, live_pnl_claim=True)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    records, refused = load_receipts(tmp_path)
    assert refused == 1 and len(records) == 1
    assert records[0].eligible


def test_build_corpus_provenance_and_negatives(tmp_path: Path):
    _write_receipt(tmp_path / "good.json", research_only=True, live_pnl_claim=False)
    _write_run_manifest(tmp_path / "bad.json", claim=None)
    stats = build_corpus(tmp_path, tmp_path / "corpus.jsonl")
    assert stats == {
        "loaded": 2,
        "refused": 0,
        "positive": 1,
        "negative": 1,
        "skipped": 0,
    }


def test_run_manifest_schema_is_eligible_and_labeled_synthetic(tmp_path: Path):
    _write_run_manifest(tmp_path / "run.json", claim="research_only")
    records, _ = load_receipts(tmp_path)
    assert len(records) == 1 and records[0].eligible


def test_run_manifest_without_claim_fails_closed(tmp_path: Path):
    _write_run_manifest(tmp_path / "run.json", claim=None)
    records, refused = load_receipts(tmp_path)
    assert refused == 0 and not records[0].eligible


def test_explicit_live_claim_overrides_research_only_claim(tmp_path: Path):
    _write_run_manifest(tmp_path / "run.json", claim="research_only", live_pnl_claim=True)
    records, refused = load_receipts(tmp_path)
    assert refused == 1 and records == []


def test_build_corpus_refuses_quarantined_and_rejected_receipts(tmp_path: Path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    quarantined = receipts / "legacy-unsealed"
    quarantined.mkdir()
    _write_receipt(quarantined / "pinned.json", research_only=True, live_pnl_claim=False)
    _write_receipt(receipts / "admitted.json", research_only=True, live_pnl_claim=False)
    _write_receipt(receipts / "reject.json", research_only=True, live_pnl_claim=True)
    digest = hashlib.sha256((quarantined / "pinned.json").read_bytes()).hexdigest()
    registry = {
        "schema": "legacy_quarantine.v2",
        "files": {
            "legacy-unsealed/pinned.json": {
                "sha256": digest,
                "errors": ["research_only_missing"],
                "category": "admission_honesty_gap",
                "reason": "test pin",
            }
        },
        "schema_drift_acks": {},
    }
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "legacy_quarantine.json").write_text(json.dumps(registry))
    stats = build_corpus(receipts, tmp_path / "corpus.jsonl", repo_root=tmp_path)
    assert stats["refused"] == 2 and stats["loaded"] == 1


def test_build_corpus_accepts_multiple_receipt_dirs(tmp_path: Path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_receipt(dir_a / "good.json", research_only=True, live_pnl_claim=False)
    _write_run_manifest(dir_b / "run.json", claim="research_only")
    _write_run_manifest(dir_b / "claimless.json", claim=None)
    stats = build_corpus([dir_a, dir_b], tmp_path / "corpus.jsonl")
    assert stats == {
        "loaded": 3,
        "refused": 0,
        "positive": 2,
        "negative": 1,
        "skipped": 0,
    }
