"""Admission gate: a valid receipt may still not belong in the corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.admission import (
    ADMISSION_SCHEMA,
    admission_check,
    admission_contract_errors,
    write_admission_receipt,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_file


def _write(dirpath: Path, name: str, body: dict, *, seal: bool = True) -> Path:
    path = dirpath / name
    payload = seal_receipt(dict(body)) if seal else dict(body)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _claim_body(value: float, *, dataset: str | None = None, inputs: str | None = None) -> dict:
    """Minimal claim-bearing receipt body."""
    return {
        "kind": "synthetic_fixture.v1",
        "schema": "synthetic_fixture.v1",  # unregistered kind — no lane contract fires
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "inputs_sha256": inputs or "ab" * 32,
        "params": {"drill": True},
        "results": [{"head": "x", "qlike": value}],
        "dataset_sha256": dataset or "cd" * 32,
    }


@pytest.fixture()
def corpus(tmp_path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _write(root, "a.json", _claim_body(1.0))
    _write(root, "b.json", _claim_body(1.0))
    return root


def test_admit_clean_candidate(corpus: Path) -> None:
    # distinct inputs+dataset fingerprints — no claim group forms vs the corpus
    candidate = _write(
        corpus.parent, "c.json", _claim_body(2.0, dataset="ee" * 32, inputs="ff" * 32)
    )
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "admit"
    assert all(c["ok"] for c in result["checks"])
    assert result["candidate_sha256"]


def test_reject_forged_seal(corpus: Path, tmp_path: Path) -> None:
    candidate = _write(tmp_path, "forged.json", _claim_body(1.0))
    doc = json.loads(candidate.read_text())
    doc["results"][0]["qlike"] = 999.0  # tamper post-seal
    candidate.write_text(json.dumps(doc))
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "reject"


def test_reject_live_pnl_claim(corpus: Path, tmp_path: Path) -> None:
    body = _claim_body(1.0)
    body["live_pnl_claim"] = True
    candidate = _write(tmp_path, "live.json", body)
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "reject"
    honesty = next(c for c in result["checks"] if c["name"] == "honesty")
    assert "live_pnl_claim_true" in honesty["reject_errors"]


def test_quarantine_missing_stamps(corpus: Path, tmp_path: Path) -> None:
    body = _claim_body(1.0)
    del body["research_only"]
    del body["data_label"]
    candidate = _write(tmp_path, "bare.json", body)
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "quarantine"
    honesty = next(c for c in result["checks"] if c["name"] == "honesty")
    assert "research_only_missing" in honesty["quarantine_errors"]
    assert "data_label_missing" in honesty["quarantine_errors"]


def test_quarantine_new_inconsistent_group(corpus: Path, tmp_path: Path) -> None:
    """Same dataset fingerprint, contradictory claim → quarantine, not admit."""
    candidate = _write(
        tmp_path,
        "contra.json",
        _claim_body(9.9, dataset="cd" * 32),
    )
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "quarantine"
    assert result["lattice_new_inconsistent"]


def test_verdict_check_coherence_contract() -> None:
    """The contract pins verdict↔checks coherence — no claim the checks deny."""
    base = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "verdict": "admit",
        "checks": [
            {"name": "seal", "ok": True},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True},
            {"name": "corpus", "ok": True},
        ],
    }
    assert admission_contract_errors(base) == []
    base["verdict"] = "admit"
    base["checks"][1]["quarantine_errors"] = ["data_label_missing"]
    assert "verdict_admit_with_findings" in admission_contract_errors(base)


def test_written_receipt_verifies(corpus: Path, tmp_path: Path) -> None:
    candidate = _write(tmp_path, "c.json", _claim_body(2.0))
    result = admission_check(candidate, corpus)
    out = tmp_path / "out"
    out.mkdir()
    path = write_admission_receipt(result, out)
    ver = verify_receipt_file(path)
    assert ver["valid"], ver["errors"]
    doc = json.loads(path.read_text())
    assert doc["kind"] == ADMISSION_SCHEMA


def test_fails_closed_on_missing_candidate(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        admission_check(tmp_path / "nope.json", corpus)


def test_fails_closed_on_missing_corpus(tmp_path: Path) -> None:
    candidate = _write(tmp_path, "c.json", _claim_body(1.0))
    with pytest.raises(ValueError, match="does not exist"):
        admission_check(candidate, tmp_path / "nope")


def test_wrong_verdict_rejected_by_writer(corpus: Path, tmp_path: Path) -> None:
    result = admission_check(_write(tmp_path, "c.json", _claim_body(1.0)), corpus)
    result["verdict"] = "admit"  # forge the verdict — checks must deny it
    result["checks"][0]["ok"] = False
    with pytest.raises(ValueError, match="contract"):
        write_admission_receipt(result, tmp_path)


def test_empty_corpus_admits_first_clean_receipt(tmp_path: Path) -> None:
    corpus = tmp_path / "empty_corpus"
    corpus.mkdir()
    candidate = _write(tmp_path, "first.json", _claim_body(1.0))
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "admit"
    assert result["n_corpus_receipts"] == 0


def test_candidate_inside_corpus_is_idempotent(corpus: Path) -> None:
    """Re-gating a member must not crash on the shadow dir's self-link."""
    member = corpus / "a.json"
    result = admission_check(member, corpus)
    assert result["verdict"] == "admit"
    assert result["lattice_new_inconsistent"] == []


def test_name_collision_with_corpus_member_fails_closed(corpus: Path, tmp_path: Path) -> None:
    """A foreign receipt sharing a member's filename is refused, not merged."""
    intruder = _write(tmp_path, "a.json", _claim_body(7.7, dataset="ab" * 32))
    with pytest.raises(FileExistsError, match="collision"):
        admission_check(intruder, corpus)
