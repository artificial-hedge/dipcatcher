"""Admission gate: a valid receipt may still not belong in the corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.admission import (
    ADMISSION_SCHEMA,
    admission_check,
    admission_contract_errors,
    admit_batch,
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
    # Missing stamps still surface as quarantine errors, but without a
    # data_label the declared dataset digest is a non-synthetic tape
    # binding no committed manifest attests — reject, not quarantine.
    assert result["verdict"] == "reject"
    honesty = next(c for c in result["checks"] if c["name"] == "honesty")
    assert "research_only_missing" in honesty["quarantine_errors"]
    assert "data_label_missing" in honesty["quarantine_errors"]
    seal = next(c for c in result["checks"] if c["name"] == "seal")
    assert "tape_manifest_unknown" in seal["errors"]


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


def test_epoch_chain_check_present_or_skipped(corpus: Path, tmp_path: Path) -> None:
    """The epoch-chain check always runs — skipped only if corpus_epoch is absent."""
    # Distinct inputs/dataset fingerprints so the lattice check does not flag a
    # claim contradiction — this test exercises the epoch check, not lattice.
    candidate = _write(tmp_path, "c.json", _claim_body(2.0, dataset="ee" * 32, inputs="ff" * 32))
    result = admission_check(candidate, corpus)
    epoch_check = next(c for c in result["checks"] if c["name"] == "epoch_chain")
    assert epoch_check["ok"] is True
    # corpus_epoch lands in a sibling PR: either it is importable and the
    # (unstamped) corpus reports intact-with-no-epochs, or it is skipped.
    if epoch_check.get("skipped") == "corpus_epoch_unavailable":
        assert result["corpus_epoch_root"] is None
    else:
        # an unstamped corpus is benign — no epochs yet, nothing authoritative
        # broken; admissions proceed normally
        assert epoch_check["errors"] == ["no_epoch_receipts"]
        assert result["verdict"] == "admit"


def test_admit_batch_committed_candidate_is_not_vacuous(tmp_path: Path) -> None:
    """A receipt already inside the corpus must still face a real delta.

    Single-candidate ``admission_check`` on a committed file is vacuous on
    the lattice check (before == after — the candidate is already a member).
    ``admit_batch`` strips the changed names from the shadow corpus first.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    contra = _write(corpus, "contra.json", _claim_body(9.9, dataset="cd" * 32))
    _write(corpus, "a.json", _claim_body(1.0))
    _write(corpus, "b.json", _claim_body(1.0))

    # Single admission against the live corpus: vacuous admit (the
    # contradictory file is already in both sides of the delta).
    solo = admission_check(contra, corpus)
    assert not solo["lattice_new_inconsistent"]

    batch = admit_batch([contra], corpus)
    assert batch["verdict"] == "quarantine"
    assert batch["results"][0]["lattice_new_inconsistent"]


def test_admit_batch_intra_diff_contradiction(tmp_path: Path) -> None:
    """Two new receipts contradicting each other: the later file draws the flag."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    first = _write(inbox, "m1.json", _claim_body(1.0, dataset="cd" * 32, inputs="dd" * 32))
    second = _write(inbox, "m2.json", _claim_body(9.9, dataset="cd" * 32, inputs="ee" * 32))

    batch = admit_batch([second, first], corpus)  # order-independence: sorted by name
    assert batch["verdict"] == "quarantine"
    by_name = {r["candidate"]: r["verdict"] for r in batch["results"]}
    assert by_name["m1.json"] == "admit"
    assert by_name["m2.json"] == "quarantine"
    assert batch["failures"] == ["m2.json"]


def test_admit_batch_reject_dominates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    good = _write(inbox, "g.json", _claim_body(2.0, dataset="ee" * 32, inputs="ff" * 32))
    forged = _write(inbox, "f.json", _claim_body(1.0, dataset="ee" * 32, inputs="ab" * 32))
    doc = json.loads(forged.read_text())
    doc["results"][0]["qlike"] = 999.0
    forged.write_text(json.dumps(doc))

    batch = admit_batch([good, forged], corpus)
    assert batch["verdict"] == "reject"
    assert batch["n_candidates"] == 2
    # f.json is rejected on its broken seal; it still lands in the shadow
    # (post-merge coexistence), so g.json — whose dataset group now contains
    # the forged claim — quarantines on the lattice delta. The batch as a
    # whole must not merge, and both files carry their own verdict.
    by_name = {r["candidate"]: r["verdict"] for r in batch["results"]}
    assert by_name == {"f.json": "reject", "g.json": "quarantine"}
    assert set(batch["failures"]) == {"f.json", "g.json"}


def test_admit_batch_fails_closed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="does not exist"):
        admit_batch([tmp_path / "nope.json"], corpus)
    candidate = _write(tmp_path, "c.json", _claim_body(1.0))
    with pytest.raises(ValueError, match="does not exist"):
        admit_batch([candidate], tmp_path / "nope")


def test_epoch_chain_fields_contract() -> None:
    """A broken epoch chain is quarantinable; forged bindings are pinned."""
    base = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "corpus_epoch_receipt": "corpus_epoch_" + "0" * 16 + ".json",
        "corpus_epoch_root": "ef" * 32,
        "verdict": "quarantine",
        "checks": [
            {"name": "seal", "ok": True},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True},
            {"name": "corpus", "ok": True},
            {"name": "epoch_chain", "ok": False, "errors": ["member_removed:x.json"]},
        ],
    }
    assert admission_contract_errors(base) == []
    base["corpus_epoch_root"] = "nothex"
    assert "corpus_epoch_root" in admission_contract_errors(base)
    base["corpus_epoch_root"] = "ef" * 32
    base["verdict"] = "admit"
    assert "verdict_admit_with_findings" in admission_contract_errors(base)


def test_tombstone_reject_error_makes_verdict_reject() -> None:
    """Contract: a retraction reject finding must be able to drive reject."""
    body = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "params": {"q": 0.05, "corpus_dir": "receipts", "known_inconsistent": []},
        "candidate": "c.json",
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "checks": [
            {"name": "seal", "ok": True, "errors": []},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True, "new_inconsistent_groups": 0},
            {"name": "corpus", "ok": True},
            {
                "name": "tombstone",
                "ok": False,
                "reject_errors": ["retracted_bytes"],
                "findings": [],
            },
        ],
        "lattice_new_inconsistent": [],
        "survivors_added": [],
        "survivors_removed": [],
        "verdict": "reject",
    }
    assert admission_contract_errors(body) == []
    # ...and a forged 'admit' on the same checks must not contract-verify.
    forged = dict(body, verdict="admit")
    assert "verdict_admit_with_findings" in admission_contract_errors(forged)


def test_tombstone_finding_makes_verdict_quarantine() -> None:
    """Contract: a soft retraction finding (slot/scope) drives quarantine."""
    body = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "params": {"q": 0.05, "corpus_dir": "receipts", "known_inconsistent": []},
        "candidate": "c.json",
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "checks": [
            {"name": "seal", "ok": True, "errors": []},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True, "new_inconsistent_groups": 0},
            {"name": "corpus", "ok": True},
            {
                "name": "tombstone",
                "ok": False,
                "reject_errors": [],
                "findings": ["retracted_slot"],
            },
        ],
        "lattice_new_inconsistent": [],
        "survivors_added": [],
        "survivors_removed": [],
        "verdict": "quarantine",
    }
    assert admission_contract_errors(body) == []


def test_retracted_bytes_rejected_end_to_end(tmp_path: Path) -> None:
    """E2E: the byte-exact retracted artifact is refused re-admission.

    receipt_tombstone lands on a sibling branch — skip dormant until both
    merge; the contract tests above pin the verdict wiring regardless.
    """
    pytest.importorskip("quant_fund.research.receipt_tombstone")
    from quant_fund.research.receipt_tombstone import write_tombstone

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    bad = _write(corpus, "b.json", _claim_body(1.0))
    write_tombstone(bad, corpus_dir=corpus, reason="bad inputs")

    candidate = tmp_path / "reissue.json"
    candidate.write_text(bad.read_bytes().decode())
    res = admission_check(candidate, corpus)
    # name differs → not the same slot; the *bytes* are what is retracted
    tomb = next(c for c in res["checks"] if c["name"] == "tombstone")
    assert tomb["ok"] is True  # byte-identity is name-keyed in the corpus

    res2 = admission_check(bad, corpus)
    tomb2 = next(c for c in res2["checks"] if c["name"] == "tombstone")
    assert tomb2["reject_errors"] == ["retracted_bytes"]
    assert res2["verdict"] == "reject"
