"""SYNTHETIC adversarial probes for the eval stores — journal-first
durability (a failed append changes nothing), eviction-vs-worker
resurrection, spec update ordering, and wire-object containment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import pytest

from fx1.serve.evals import EvalRecord, EvalSpec, EvalSpecStore, EvalStore, spec_wire
from fx1.serve.journal import JobJournal


def _store(path: Path | None = None, max_entries: int = 2) -> EvalStore:
    journal = JobJournal(path) if path is not None else None
    return EvalStore(max_entries=max_entries, journal=journal)


def _specs(path: Path | None = None, max_entries: int = 2) -> EvalSpecStore:
    journal = JobJournal(path) if path is not None else None
    return EvalSpecStore(max_entries=max_entries, journal=journal)


def _kill(journal: JobJournal, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every append now fails before durably writing — the contract
    under test is that a failed append changes no live state."""

    def dead(payload: dict[str, Any]) -> None:
        raise OSError("synthetic journal loss")

    monkeypatch.setattr(journal, "append", dead)


_EVAL_STATUS = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def _rec(eval_id: str = "eval-a", status: _EVAL_STATUS = "queued") -> EvalRecord:
    return EvalRecord(
        eval_id=eval_id,
        suite="capability",
        backend="local_fx1",
        seed=0,
        status=status,
        created_at=time.time(),
    )


def _spec(spec_id: str = "spec-a", name: str = "n") -> EvalSpec:
    return EvalSpec(
        spec_id=spec_id,
        name=name,
        data_source_config={"type": "custom", "item_schema": {"suite": "capability"}},
        testing_criteria=[{"type": "score_model", "name": "grader"}],
        metadata={"env": "synthetic"},
        created_at=time.time(),
    )


def test_put_failure_is_fully_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed put append must leave the store exactly as before —
    no record, no idempotency mapping."""
    journal = JobJournal(tmp_path / "evals.jsonl")
    store = EvalStore(max_entries=4, journal=journal)
    _kill(journal, monkeypatch)
    with pytest.raises(OSError):
        store.put(_rec("eval-new"), "idem-2", "fp2")
    assert store.get("eval-new") is None
    assert store.get_key("idem-2") is None


def test_evicted_victim_survives_a_failed_put(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the put append fails after evictions were *selected*, the
    victim must not have been dropped — it stays live until the journal
    acknowledges the tombstone (journal-first, not evict-first)."""
    journal = JobJournal(tmp_path / "evals.jsonl")
    store = EvalStore(max_entries=1, journal=journal)
    victim = _rec("eval-victim")
    store.put(victim, None, None)
    _kill(journal, monkeypatch)
    with pytest.raises(OSError):
        store.put(_rec("eval-new"), None, None)
    assert store.get("eval-victim") is victim  # never dropped
    assert store.get("eval-new") is None


def test_delete_failure_keeps_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tombstone-first delete: a failed append must leave the record
    live — not dropped in memory while the journal still replays it."""
    journal = JobJournal(tmp_path / "evals.jsonl")
    store = EvalStore(max_entries=4, journal=journal)
    rec = _rec("eval-doomed", status="succeeded")
    store.put(rec, None, None)
    _kill(journal, monkeypatch)
    with pytest.raises(OSError):
        store.delete(rec.eval_id)
    assert store.get(rec.eval_id) is rec


def test_mark_on_evicted_record_never_resurrects(tmp_path: Path) -> None:
    """A late worker mark on an evicted record must not re-journal it —
    eviction is the store's verdict and replay honors it."""
    path = tmp_path / "evals.jsonl"
    store = _store(path)
    doomed = _rec("eval-doomed", status="succeeded")
    store.put(doomed, "idem-x", "fpx")
    store.put(_rec("eval-b"), None, None)
    store.put(_rec("eval-c"), None, None)  # evicts doomed (max=2)
    assert store.get(doomed.eval_id) is None

    doomed.status = "failed"  # a stale worker write lands late
    store.mark(doomed)
    revived = _store(path)
    assert revived.get(doomed.eval_id) is None  # no resurrection
    assert revived.get_key("idem-x") is None


def test_start_loses_to_cancel(tmp_path: Path) -> None:
    """The atomic claim: a queued record cancelled before the worker
    claims it can never flip back to running."""
    store = _store(tmp_path / "evals.jsonl")
    rec = _rec("eval-q")
    store.put(rec, None, None)
    got, outcome = store.cancel(rec.eval_id)
    assert outcome == "cancelled" and got is rec
    assert store.start(rec.eval_id) is None
    assert rec.status == "cancelled"


def test_spec_put_delete_are_journal_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The spec store holds the same boundary: failed appends are inert,
    deletes tombstone before dropping."""
    journal = JobJournal(tmp_path / "specs.jsonl")
    store = EvalSpecStore(max_entries=1, journal=journal)
    keep = _spec("spec-keep")
    store.put(keep)
    _kill(journal, monkeypatch)
    with pytest.raises(OSError):
        store.put(_spec("spec-new"))
    assert store.get("spec-new") is None
    assert store.get("spec-keep") is keep
    with pytest.raises(OSError):
        store.delete("spec-keep")
    assert store.get("spec-keep") is keep


def test_spec_update_is_ordered_and_never_resurrects(tmp_path: Path) -> None:
    """update() computes the next record under the lock, journals it,
    then mutates — a deleted spec can never be journaled back to life,
    and the live object only carries what the journal confirmed."""
    path = tmp_path / "specs.jsonl"
    store = _specs(path)
    spec = _spec("spec-e", name="orig")
    store.put(spec)
    out = store.update("spec-e", name="renamed", metadata={"env": "prod"})
    assert out is spec and spec.name == "renamed"
    store.delete("spec-e")
    assert store.update("spec-e", name="zombie") is None
    revived = _specs(path)
    assert revived.get("spec-e") is None  # update never resurrects


def test_spec_update_failure_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed update append leaves the live spec untouched — never a
    mutated record the journal doesn't know about."""
    journal = JobJournal(tmp_path / "specs.jsonl")
    store = EvalSpecStore(max_entries=2, journal=journal)
    spec = _spec("spec-u", name="orig")
    store.put(spec)
    _kill(journal, monkeypatch)
    with pytest.raises(OSError):
        store.update("spec-u", name="changed")
    assert spec.name == "orig"  # journal-first: nothing applied


def test_spec_wire_never_shares_live_containers() -> None:
    """The wire object owns its nested dicts — mutating a response must
    not reach into the stored spec."""
    spec = _spec("spec-w")
    wire = spec_wire(spec)
    wire["data_source_config"]["injected"] = True
    wire["testing_criteria"][0]["name"] = "tampered"
    wire["metadata"]["env"] = "tampered"
    assert "injected" not in spec.data_source_config
    assert spec.testing_criteria[0]["name"] == "grader"
    assert spec.metadata["env"] == "synthetic"


def test_queued_eval_recovers_failed_not_queued(tmp_path: Path) -> None:
    """An eval non-terminal at crash never comes back queued or running —
    it fails with the honest restart error."""
    path = tmp_path / "evals.jsonl"
    store = _store(path)
    rec = _rec("eval-crash")
    store.put(rec, "idem-c", "fpc")
    revived = _store(path)
    got = revived.get("eval-crash")
    assert got is not None
    assert got.status == "failed"
    assert "restart" in (got.error or "")
    assert revived.get_key("idem-c") == ("fpc", "eval-crash")
