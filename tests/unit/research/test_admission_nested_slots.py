"""Admission keys every lookup by corpus-relative slot, never by file name.

``make admission-gate`` now routes nested receipts (``legacy-unsealed/``, any
subdirectory) into ``admit-batch``. That engine used to key on the bare file
name: a nested candidate sharing a name with a flat receipt evicted the flat
one from the batch shadow, the second of two same-named candidates was compared
against the first one's bytes, nested corpus members were invisible to the
lattice, and a retraction naming a nested target never matched its candidate.
These tests run the production ``admit_batch`` / ``admission_check`` against
real sealed receipts and the real lattice.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.admission import (
    admission_check,
    admission_contract_errors,
    admit_batch,
)
from quant_fund.research.receipt_tombstone import tombstone_body
from quant_fund.research.receipt_v2 import seal_receipt
from quant_fund.utils.hashing import hash_bytes

DATASET = "cd" * 32


def _body(value: float, *, dataset: str = DATASET, inputs: str = "ab") -> dict[str, Any]:
    """Honest, claim-bearing receipt body (one qlike claim on ``dataset``)."""
    return {
        "kind": "synthetic_fixture.v1",
        "schema": "synthetic_fixture.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "inputs_sha256": inputs * 32,
        "params": {"drill": True},
        "results": [{"head": "x", "qlike": value}],
        "dataset_sha256": dataset,
    }


def _write(root: Path, rel: str, body: Mapping[str, Any]) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal_receipt(dict(body)), indent=2, sort_keys=True))
    return path


def _verdicts(batch: Mapping[str, Any]) -> dict[str, str]:
    return {r["candidate"]: r["verdict"] for r in batch["results"]}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    return root


def test_same_named_nested_candidate_is_checked_against_the_flat_one(corpus: Path) -> None:
    """``zz_a.json`` (1.0) and ``zzz/zz_a.json`` (9.9) contradict each other on one
    dataset. Git sorts the flat file first, so the attacker controls the order:
    the nested one used to be compared against the flat one's own bytes."""
    flat = _write(corpus, "zz_a.json", _body(1.0, inputs="dd"))
    nested = _write(corpus, "zzz/zz_a.json", _body(9.9, inputs="ee"))

    batch = admit_batch([nested, flat], corpus)

    assert _verdicts(batch) == {"zz_a.json": "admit", "zzz/zz_a.json": "quarantine"}
    assert batch["failures"] == ["zzz/zz_a.json"]
    assert batch["verdict"] == "quarantine"


def test_different_names_and_same_names_are_gated_alike(tmp_path: Path) -> None:
    """The control: the identical pair under different file names was already
    caught, so name equality alone must not change the verdict."""
    verdicts = []
    for second in ("zz_b.json", "zz_a.json"):
        corpus = tmp_path / second.removesuffix(".json")
        corpus.mkdir()
        first = _write(corpus, "zz_a.json", _body(1.0, inputs="dd"))
        other = _write(corpus, f"zzz/{second}", _body(9.9, inputs="ee"))
        verdicts.append(_verdicts(admit_batch([first, other], corpus)))
    assert [sorted(v.values()) for v in verdicts] == [["admit", "quarantine"]] * 2


def test_existing_nested_member_is_visible_to_the_lattice(corpus: Path) -> None:
    """A flat candidate contradicting an unchanged *nested* corpus member must
    quarantine, exactly as it does against a flat member."""
    _write(corpus, "zz_sub/n.json", _body(1.0))
    outside = _write(corpus.parent / "inbox", "c.json", _body(9.9, inputs="ee"))
    assert admission_check(outside, corpus)["verdict"] == "quarantine"

    inside = _write(corpus, "c.json", _body(9.9, inputs="ee"))
    assert admit_batch([inside], corpus)["verdict"] == "quarantine"


def test_nested_candidate_does_not_evict_a_flat_member_of_the_same_name(corpus: Path) -> None:
    """``changed_names`` was basename-keyed: ``sub/n.json`` as candidate dropped the
    unchanged flat ``n.json`` from the shadow it is judged against."""
    _write(corpus, "n.json", _body(1.0))
    cand = _write(corpus, "sub/n.json", _body(9.9, inputs="ee"))

    batch = admit_batch([cand], corpus)

    assert _verdicts(batch) == {"sub/n.json": "quarantine"}


def test_a_candidate_is_judged_on_its_own_bytes_not_a_same_named_members(corpus: Path) -> None:
    """Outside candidate ``a.json`` replaces the corpus's ``a.json`` in the
    after-view; it used to be skipped (destination existed) and judged on the
    member's bytes, hiding its contradiction with ``b.json``."""
    _write(corpus, "a.json", _body(1.0, dataset="11" * 32))
    _write(corpus, "b.json", _body(1.0))
    outside = _write(corpus.parent / "inbox", "a.json", _body(9.9, inputs="ee"))

    result = admission_check(outside, corpus)

    assert result["lattice_new_inconsistent"]
    assert result["verdict"] == "quarantine"


def test_two_different_files_claiming_one_slot_fail_closed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    one = _write(tmp_path / "in1", "x.json", _body(1.0, inputs="dd"))
    two = _write(tmp_path / "in2", "x.json", _body(9.9, inputs="ee"))

    with pytest.raises(ValueError, match="same corpus slot"):
        admit_batch([one, two], corpus)
    # The same file named twice is one candidate, not a collision.
    assert admit_batch([one, one], corpus)["n_candidates"] == 1


def test_symlink_candidate_is_refused(corpus: Path, tmp_path: Path) -> None:
    target = _write(tmp_path / "elsewhere", "t.json", _body(1.0))
    link = corpus / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        admit_batch([link], corpus)


def test_corpus_accounting_covers_nested_members_but_not_quarantined_ones(corpus: Path) -> None:
    _write(corpus, "a.json", _body(1.0, dataset="11" * 32))
    _write(corpus, "sub/b.json", _body(1.0, dataset="22" * 32))
    _write(corpus, "legacy-unsealed/q.json", _body(1.0, dataset="33" * 32))
    cand = _write(corpus.parent, "c.json", _body(1.0, dataset="44" * 32))

    result = admission_check(cand, corpus)

    assert result["n_corpus_receipts"] == 2
    # inputs_sha256 binds the nested member: changing it changes the digest.
    _write(corpus, "sub/b.json", _body(2.0, dataset="22" * 32))
    assert admission_check(cand, corpus)["inputs_sha256"] != result["inputs_sha256"]


def test_candidate_rel_must_be_a_corpus_relative_path(corpus: Path, tmp_path: Path) -> None:
    cand = _write(tmp_path, "c.json", _body(1.0))
    for bad in ("", ".", "../c.json", "/abs/c.json"):
        with pytest.raises(ValueError, match="corpus-relative"):
            admission_check(cand, corpus, candidate_rel=bad)


def _tombstone(corpus: Path, target_rel: str, *, scope: str | list[str] = "all") -> None:
    target = corpus / target_rel
    digest = hash_bytes(target.read_bytes())
    body = tombstone_body(
        target_name=target_rel, target_sha256=digest, reason="retracted", scope=scope
    )
    _write(corpus, f"tombstone_{digest[:16]}.json", body)


def _retraction(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    (result,) = batch["results"]
    return next(c for c in result["checks"] if c["name"] == "tombstone")


def test_retraction_naming_a_nested_target_binds_that_nested_candidate(corpus: Path) -> None:
    """``load_tombstones`` keys ``active`` by corpus-relative path; admission looked
    it up by file name, so a nested retraction target never matched."""
    target = _write(corpus, "sub/t.json", _body(1.0))
    _tombstone(corpus, "sub/t.json")

    batch = admit_batch([target], corpus)

    assert batch["verdict"] == "reject"
    assert _retraction(batch)["reject_errors"] == ["retracted_bytes"]


def test_retracted_bytes_moved_into_a_subdirectory_stay_rejected(corpus: Path) -> None:
    """The file-name lookup rejected a retracted flat receipt re-added one level
    down under the same name; slot keying must not loosen that."""
    original = _write(corpus, "t.json", _body(1.0))
    _tombstone(corpus, "t.json")
    moved = corpus / "elsewhere" / "t.json"
    moved.parent.mkdir()
    moved.write_bytes(original.read_bytes())

    batch = admit_batch([moved], corpus)

    assert batch["verdict"] == "reject"
    assert _retraction(batch)["reject_errors"] == ["retracted_bytes"]


def test_a_retracted_slot_may_not_quietly_refill_in_a_subdirectory(corpus: Path) -> None:
    _write(corpus, "t.json", _body(1.0, dataset="11" * 32))
    _tombstone(corpus, "t.json")
    refill = _write(corpus, "sub/t.json", _body(2.0, dataset="22" * 32))

    batch = admit_batch([refill], corpus)

    assert batch["verdict"] == "quarantine"
    assert _retraction(batch)["findings"] == ["retracted_slot"]


def test_an_unrelated_name_is_not_bound_by_a_retraction(corpus: Path) -> None:
    _write(corpus, "t.json", _body(1.0, dataset="11" * 32))
    _tombstone(corpus, "t.json")
    other = _write(corpus, "sub/u.json", _body(2.0, dataset="22" * 32))

    batch = admit_batch([other], corpus)

    assert _retraction(batch) == {
        "name": "tombstone",
        "ok": True,
        "reject_errors": [],
        "findings": [],
    }


# -- a candidate AT a registry-governed slot is rejected, never "admitted" ---------

QUARANTINED_SLOTS = ["legacy-unsealed/c.json", "legacy-unsealed/deep/c.json"]


def _slot_check(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(c for c in result["checks"] if c["name"] == "quarantine_slot")


@pytest.mark.parametrize("slot", QUARANTINED_SLOTS)
def test_an_unretained_candidate_in_a_quarantined_subtree_is_rejected(
    corpus: Path, slot: str
) -> None:
    """The lattice and corpus audit skip ``legacy-unsealed/`` by design, so a
    candidate placed there was measured against nothing and drew ``admit`` --
    while ``suite-health`` fails the same file as ``quarantine_unregistered``.
    It must be rejected outright, even when it contradicts nothing."""
    _write(corpus, "a.json", _body(1.0, dataset="11" * 32))
    cand = _write(corpus, slot, _body(1.0, dataset="22" * 32))

    batch = admit_batch([cand], corpus)

    (result,) = batch["results"]
    assert result["candidate"] == slot
    assert result["verdict"] == "reject" and batch["verdict"] == "reject"
    assert batch["failures"] == [slot]
    assert _slot_check(result) == {
        "name": "quarantine_slot",
        "ok": False,
        "reject_errors": ["quarantined_slot_not_retained"],
    }
    assert admission_contract_errors(result) == []


@pytest.mark.parametrize("slot", QUARANTINED_SLOTS)
def test_a_contradiction_at_a_quarantined_slot_is_still_measured(corpus: Path, slot: str) -> None:
    """Rejection must not hide what the file claims: the lattice delta is taken
    from a visible probe slot, so it matches the same bytes at a visible slot
    (a sealed, honest receipt contradicting a corpus claim)."""
    _write(corpus, "zz_sub/n.json", _body(1.0))
    hidden = _write(corpus, slot, _body(9.9, inputs="ee"))
    (at_slot,) = admit_batch([hidden], corpus)["results"]
    # Same bytes at a visible slot (written only now, so it was not in the view above).
    visible = _write(corpus, "zz_vis/c.json", _body(9.9, inputs="ee"))
    (at_visible,) = admit_batch([visible], corpus)["results"]

    assert at_visible["verdict"] == "quarantine"
    assert at_visible["lattice_new_inconsistent"]
    assert at_slot["verdict"] == "reject"
    assert at_slot["lattice_new_inconsistent"] == at_visible["lattice_new_inconsistent"]
    lattice = next(c for c in at_slot["checks"] if c["name"] == "lattice")
    assert lattice["ok"] is False


def test_a_quarantined_candidate_stays_out_of_later_candidates_view(corpus: Path) -> None:
    """The merged corpus never shows ``legacy-unsealed/`` to the lattice, so a
    later candidate is judged against exactly that -- not against the rejected
    record -- and the shadow dir probe slot does not leak into the accounting."""
    _write(corpus, "a.json", _body(1.0, dataset="11" * 32))
    quarantined = _write(corpus, "legacy-unsealed/c.json", _body(9.9, inputs="ee"))
    later = _write(corpus, "zz_later.json", _body(8.8, inputs="ff", dataset="33" * 32))

    batch = admit_batch([quarantined, later], corpus)

    assert _verdicts(batch) == {"legacy-unsealed/c.json": "reject", "zz_later.json": "admit"}
    assert all(r["n_corpus_receipts"] == 1 for r in batch["results"])


def test_the_contract_refuses_an_admit_for_a_quarantined_slot(corpus: Path) -> None:
    cand = _write(corpus, "legacy-unsealed/c.json", _body(1.0))
    (result,) = admit_batch([cand], corpus)["results"]
    assert admission_contract_errors(result) == []

    forged = {**result, "verdict": "admit"}
    errors = admission_contract_errors(forged)
    assert "quarantined_slot_not_rejected" in errors
    assert "verdict_admit_with_findings" in errors
    # Stripping the check does not help: the candidate path alone demands a reject.
    stripped = {
        **forged,
        "checks": [c for c in result["checks"] if c["name"] != "quarantine_slot"],
    }
    assert "quarantined_slot_not_rejected" in admission_contract_errors(stripped)
    # A rejection that names no cause is still incoherent for a visible slot.
    visible = _write(corpus, "v.json", _body(1.0, dataset="44" * 32))
    (clean,) = admit_batch([visible], corpus)["results"]
    assert admission_contract_errors(clean) == []


# -- retracted bytes stay retracted under any file name ----------------------------


@pytest.mark.parametrize("renamed", ["other_name.json", "sub/other_name.json", "t2.json"])
def test_retracted_bytes_re_added_under_a_different_name_are_rejected(
    corpus: Path, renamed: str
) -> None:
    """``_tombstones_binding`` bound by slot or file name only, so an exact copy
    of the retracted bytes at ``other_name.json`` drew ``admit``."""
    original = _write(corpus, "t.json", _body(1.0))
    _tombstone(corpus, "t.json")
    copy = corpus / renamed
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(original.read_bytes())

    batch = admit_batch([copy], corpus)

    assert batch["verdict"] == "reject"
    assert _retraction(batch)["reject_errors"] == ["retracted_bytes"]


def test_retracted_bytes_are_rejected_in_a_quarantined_subtree_too(corpus: Path) -> None:
    original = _write(corpus, "t.json", _body(1.0))
    _tombstone(corpus, "t.json")
    copy = corpus / "legacy-unsealed" / "other.json"
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())

    (result,) = admit_batch([copy], corpus)["results"]

    assert result["verdict"] == "reject"
    tomb = next(c for c in result["checks"] if c["name"] == "tombstone")
    assert tomb["reject_errors"] == ["retracted_bytes"]


def test_different_bytes_under_a_new_name_are_not_bound_by_a_byte_retraction(
    corpus: Path,
) -> None:
    """The control: only the retracted *bytes* bind across names, not the claim
    they carry -- a different receipt under an unrelated name is untouched."""
    _write(corpus, "t.json", _body(1.0, dataset="11" * 32))
    _tombstone(corpus, "t.json")
    other = _write(corpus, "sub/other_name.json", _body(1.0000001, dataset="22" * 32))

    batch = admit_batch([other], corpus)

    assert _retraction(batch)["reject_errors"] == []
    assert _retraction(batch)["findings"] == []


# -- admission removes every shadow it builds ---------------------------------------


@pytest.fixture()
def scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the interpreter's temp dir (the documented ``tempfile.tempdir``
    override) at an empty dir so leaked shadows are countable."""
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def test_admission_check_leaves_no_shadow_dirs(corpus: Path, scratch_root: Path) -> None:
    _write(corpus, "zz_sub/n.json", _body(1.0))
    outside = _write(corpus.parent / "inbox", "c.json", _body(9.9, inputs="ee"))

    assert admission_check(outside, corpus)["verdict"] == "quarantine"

    assert list(scratch_root.iterdir()) == []


def test_admit_batch_leaves_no_shadow_dirs(corpus: Path, scratch_root: Path) -> None:
    _write(corpus, "a.json", _body(1.0, dataset="11" * 32))
    batch = [
        _write(corpus, "b.json", _body(2.0, dataset="22" * 32)),
        _write(corpus, "zz_sub/c.json", _body(3.0, dataset="33" * 32)),
        _write(corpus, "legacy-unsealed/d.json", _body(4.0, dataset="44" * 32)),
    ]

    result = admit_batch(batch, corpus)

    assert result["n_gated"] == 3
    assert list(scratch_root.iterdir()) == []


def test_shadow_dirs_are_removed_when_the_check_fails_midway(
    corpus: Path, scratch_root: Path
) -> None:
    """``q`` outside (0, 1) makes ``corpus_audit`` raise after both shadows exist."""
    _write(corpus, "a.json", _body(1.0, dataset="11" * 32))
    cand = _write(corpus, "b.json", _body(2.0, dataset="22" * 32))

    with pytest.raises(ValueError, match="q must lie in"):
        admission_check(cand, corpus, q=2.0)
    with pytest.raises(ValueError, match="q must lie in"):
        admit_batch([cand], corpus, q=2.0)

    assert list(scratch_root.iterdir()) == []
