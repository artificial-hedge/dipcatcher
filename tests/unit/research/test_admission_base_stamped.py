"""Admission of a diff that restores stamped receipts vs. one that changes them.

A receipt whose bytes the *base* branch's pinned head epoch already stamped is
a corpus member, not an arrival: ``admit_batch`` lists it under
``already_stamped`` instead of re-admitting it under today's honesty policy.
Only the base pin can grant that — a candidate changed to *new* bytes is gated
as usual, and a pin that does not authenticate its head epoch fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.admission import admit_batch
from quant_fund.research.corpus_epoch import (
    chain_ancestry,
    corpus_epoch,
    load_heads_pin,
    pinned_head_members,
    update_heads_pin,
    write_epoch_receipt,
)
from quant_fund.utils.receipt import seal_receipt


def _write(path: Path, body: dict[str, object]) -> Path:
    path.write_text(json.dumps(seal_receipt(body), indent=2, sort_keys=True) + "\n")
    return path


def _legacy_body(marker: str) -> dict[str, object]:
    # Sealed and verifiable, but without the honesty stamps today's admission
    # requires (research_only / data_label / live_pnl_claim) — the shape of a
    # historical receipt restored to its stamped bytes.
    return {"kind": "synthetic_fixture.v1", "marker": marker}


def _stamped_corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _write(corpus / "old.json", _legacy_body("v1"))
    head = write_epoch_receipt(corpus_epoch(corpus), corpus)
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", head)
    return corpus, pin


def _base_entry(corpus: Path, pin: Path) -> dict[str, str]:
    return load_heads_pin(pin)[f"{corpus.as_posix()}/*.json"]


def test_restored_bytes_are_already_stamped_not_readmitted(tmp_path: Path) -> None:
    corpus, pin = _stamped_corpus(tmp_path)
    stamped = pinned_head_members(corpus, _base_entry(corpus, pin))
    batch = admit_batch([corpus / "old.json"], corpus, base_stamped=stamped)
    assert batch["already_stamped"] == ["old.json"]
    assert batch["n_gated"] == 0
    assert batch["verdict"] == "admit"


def test_changed_bytes_are_gated_even_with_base_pin(tmp_path: Path) -> None:
    """Negative control: a stamped name with NEW bytes is an arrival — it is
    gated, and without honesty stamps it is not admitted."""
    corpus, pin = _stamped_corpus(tmp_path)
    stamped = pinned_head_members(corpus, _base_entry(corpus, pin))
    _write(corpus / "old.json", _legacy_body("v2-rewritten"))
    batch = admit_batch([corpus / "old.json"], corpus, base_stamped=stamped)
    assert batch["already_stamped"] == []
    assert batch["n_gated"] == 1
    assert batch["verdict"] != "admit"


def test_without_base_pin_restorations_are_gated(tmp_path: Path) -> None:
    corpus, _pin = _stamped_corpus(tmp_path)
    batch = admit_batch([corpus / "old.json"], corpus)
    assert batch["already_stamped"] == []
    assert batch["n_gated"] == 1


def test_base_pin_must_authenticate_its_head_epoch(tmp_path: Path) -> None:
    corpus, pin = _stamped_corpus(tmp_path)
    entry = dict(_base_entry(corpus, pin))
    head = corpus / entry["receipt"]
    # Digest that does not match the head epoch's bytes.
    with pytest.raises(ValueError, match="does not match the pinned digest"):
        pinned_head_members(corpus, {**entry, "sha256": "0" * 64})
    # Head epoch missing from the corpus.
    head.unlink()
    with pytest.raises(ValueError, match="missing"):
        pinned_head_members(corpus, entry)
    # Path-like receipt names are refused.
    with pytest.raises(ValueError, match="missing"):
        pinned_head_members(corpus, {**entry, "receipt": "../epoch_heads.json"})


def _mutation_ack(member: str, epoch: str, frm: str, to: str) -> dict[str, object]:
    return {
        "corpus": "receipts/*.json",
        "member": member,
        "transition": "mutated",
        "from_sha256": frm,
        "to_sha256": to,
        "epoch": epoch,
        "reason": "test fixture",
    }


def test_mutation_ack_must_predate_the_base_chain(tmp_path: Path) -> None:
    """Stamp-and-excuse: a change that records a mutation of an append-only
    receipt in a NEW epoch and acknowledges it in the same diff is rejected;
    an ack naming an epoch the base chain already held is accepted. (Removal
    acks follow the same rule: ``test_epoch_ack_base_rules``.)"""
    corpus, pin = _stamped_corpus(tmp_path)
    base_head = _base_entry(corpus, pin)["receipt"]
    base = chain_ancestry(corpus, _base_entry(corpus, pin))
    assert base == frozenset({base_head})
    old_sha = pinned_head_members(corpus, _base_entry(corpus, pin))["old.json"]
    # The "PR": rewrite the stamped receipt, stamp the mutation, acknowledge it.
    _write(corpus / "old.json", _legacy_body("tampered"))
    new_epoch = write_epoch_receipt(corpus_epoch(corpus), corpus).name
    new_sha = pinned_head_members(
        corpus, {"receipt": new_epoch, "sha256": _sha(corpus / new_epoch)}
    )["old.json"]
    laundering = [_mutation_ack("old.json", new_epoch, old_sha, new_sha)]
    batch = admit_batch([corpus / "old.json"], corpus, epoch_acks=laundering, base_epochs=base)
    assert batch["verdict"] == "reject"
    assert batch["ack_errors"] == [
        f"epoch_ack_postdates_base:'receipts/*.json':'old.json'@{new_epoch!a}"
    ]
    # Positive control: an ack for history the base already recorded passes
    # this rule (the receipt itself is still gated on its own merits).
    historical = [_mutation_ack("old.json", base_head, "1" * 64, "2" * 64)]
    batch = admit_batch([corpus / "old.json"], corpus, epoch_acks=historical, base_epochs=base)
    assert batch["ack_errors"] == []


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
