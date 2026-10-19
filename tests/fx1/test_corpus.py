"""Corpus-builder tests: provenance, eligibility, fail-closed behavior."""

import hashlib
import json
from pathlib import Path

import pytest

from fx1.data import build_corpus, load_receipts

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SFT_QUARANTINE_CATEGORIES = frozenset(
    {"pre_seal_era", "admission_honesty_gap", "claim_absent"},
)


def _quarantine_registry_paths(repo_root: Path) -> set[str]:
    from quant_fund.research.legacy_unsealed import load_quarantine_or_empty

    registry = load_quarantine_or_empty(repo_root / "quality" / "legacy_quarantine.json")
    receipts = repo_root / "receipts"
    return {rel for rel in registry.files if (receipts / rel).is_file()}


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


@pytest.mark.skipif(
    not (_REPO_ROOT / "quality" / "legacy_quarantine.json").is_file(),
    reason="requires committed quarantine registry",
)
def test_build_corpus_refuses_quarantined_registry_receipts_on_real_tree(tmp_path: Path):
    from quant_fund.research.legacy_unsealed import load_quarantine_or_empty

    registry = load_quarantine_or_empty(_REPO_ROOT / "quality" / "legacy_quarantine.json")
    pinned = _quarantine_registry_paths(_REPO_ROOT)
    sft_quarantined = {
        rel for rel in pinned if registry.files[rel].category in _SFT_QUARANTINE_CATEGORIES
    }
    assert len(sft_quarantined) == 9
    records, refused = load_receipts(_REPO_ROOT / "receipts", repo_root=_REPO_ROOT)
    assert refused == len(pinned)
    loaded_rels = {Path(r.path).relative_to(_REPO_ROOT / "receipts").as_posix() for r in records}
    assert pinned.isdisjoint(loaded_rels)
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(_REPO_ROOT / "receipts", out, repo_root=_REPO_ROOT)
    assert stats["refused"] == len(pinned)
    corpus_text = out.read_text(encoding="utf-8")
    for rel in pinned:
        digest = hashlib.sha256((_REPO_ROOT / "receipts" / rel).read_bytes()).hexdigest()
        assert digest not in corpus_text


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


def test_corpus_build_excludes_real_quarantined_registry_receipts(tmp_path: Path) -> None:
    """Would fail before refuse-by-registry-sha / discovered repo_root:

    the 9 eligible quarantined receipts (4 with forbidden_metric_keys)
    trained as positives because ``fx1 corpus build`` never consulted the
    production verifier or ``quality/legacy_quarantine.json``.
    """
    from quant_fund.research.legacy_unsealed import load_quarantine

    repo = Path(__file__).resolve().parents[2]
    registry = load_quarantine(repo / "quality" / "legacy_quarantine.json")
    eligible: list[tuple[str, str, tuple[str, ...]]] = []
    for rel, entry in registry.files.items():
        path = repo / "receipts" / rel
        if not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.sha256:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        research_only = (payload.get("research_only") is True) or (
            payload.get("claim") == "research_only"
        )
        if "live_pnl_claim" in payload:
            live = bool(payload["live_pnl_claim"])
        else:
            live = payload.get("claim") != "research_only"
        if research_only and not live:
            eligible.append((rel, entry.sha256, entry.errors))
    assert len(eligible) == 9
    forbidden = [item for item in eligible if "forbidden_metric_keys" in item[2]]
    assert len(forbidden) == 4
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(repo / "receipts", out)
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    trained = {row["receipt_sha256"] for row in lines}
    blocked = {digest for _rel, digest, _errors in eligible}
    assert blocked.isdisjoint(trained)
    assert stats["refused"] >= 9


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
