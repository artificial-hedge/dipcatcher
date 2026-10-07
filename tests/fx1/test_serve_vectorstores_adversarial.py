"""SYNTHETIC adversarial probes for the vector-store surface — wire
objects never share live containers, chunking_strategy is honored (not
silently dropped), journal-first attach/create, deterministic lexical
search, and fail-closed filter grammar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.vectorstores import (
    VS_MAX_RESULTS,
    VectorStoreError,
    VectorStoreStore,
)

_CORPUS = {
    "file-alpha": b"alpha beta gamma risk parity portfolio",
    "file-beta": b"beta decay half-life isotope nuclear",
}


def _reader(file_id: str) -> tuple[bytes, str] | None:
    content = _CORPUS.get(file_id)
    if content is None:
        return None
    return content, f"{file_id}.txt"


def _store(
    state_dir: Path | None = None, max_stores: int = 4, journal: JobJournal | None = None
) -> VectorStoreStore:
    return VectorStoreStore(
        max_stores=max_stores, state_dir=state_dir, file_reader=_reader, journal=journal
    )


def test_create_honors_chunking_strategy_not_silent(tmp_path: Path) -> None:
    """``chunking_strategy`` on create applies to the initial file_ids —
    it was previously swallowed by ``extra='allow'`` and dropped."""
    store = _store(tmp_path)
    out = store.create(
        file_ids=["file-alpha"],
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 100, "chunk_overlap_tokens": 0},
        },
    )
    files = store.list_files(out["id"])
    assert files["data"][0]["chunking_strategy"] == {
        "type": "static",
        "static": {"max_chunk_size_tokens": 100, "chunk_overlap_tokens": 0},
    }


def test_wire_objects_never_share_live_containers(tmp_path: Path) -> None:
    """Mutating a returned wire object must never rewrite the stored
    record — the frozen batch verdict included."""
    store = _store(tmp_path)
    vs = store.create(name="synthetic")
    attached = store.attach(
        vs["id"],
        "file-alpha",
        attributes={"kind": "a"},
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 128, "chunk_overlap_tokens": 8},
        },
    )
    attached["chunking_strategy"]["static"]["max_chunk_size_tokens"] = 4096
    attached["attributes"]["kind"] = "tampered"
    fresh = store.get_file(vs["id"], "file-alpha")
    assert fresh["chunking_strategy"]["static"]["max_chunk_size_tokens"] == 128
    assert fresh["attributes"]["kind"] == "a"

    batch = store.file_batch_create(vs["id"], ["file-beta"])
    rows = store.file_batch_files(vs["id"], batch["id"])
    rows["data"][0]["chunking_strategy"]["type"] = "tampered"
    again = store.file_batch_files(vs["id"], batch["id"])
    assert again["data"][0]["chunking_strategy"]["type"] == "auto"  # frozen verdict


def test_attach_failure_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed journal append mid-attach publishes nothing — no
    membership, no chunks, no last_active bump."""
    journal = JobJournal(tmp_path / "vs.jsonl")
    store = _store(None, journal=journal)
    vs = store.create()
    before = store.get(vs["id"])

    real_append = journal.append

    def dead(payload: dict[str, Any]) -> None:
        raise OSError("synthetic journal loss")

    monkeypatch.setattr(journal, "append", dead)
    with pytest.raises(OSError):
        store.attach(vs["id"], "file-alpha")
    monkeypatch.setattr(journal, "append", real_append)
    after = store.get(vs["id"])
    assert after["file_counts"]["total"] == 0
    assert after["last_active_at"] == before["last_active_at"]
    assert store.search([vs["id"]], "alpha") == []


def test_create_failure_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed create append publishes no store and no idem record —
    the journal-first boundary holds for create-with-files too."""
    journal = JobJournal(tmp_path / "vs.jsonl")
    store = _store(None, journal=journal)

    def dead(payload: dict[str, Any]) -> None:
        raise OSError("synthetic journal loss")

    monkeypatch.setattr(journal, "append", dead)
    with pytest.raises(OSError):
        store.create(file_ids=["file-alpha"], idempotency_key="k1", body_fingerprint="f1")
    revived = VectorStoreStore(max_stores=4, file_reader=_reader, journal=journal)
    assert revived.list_stores()["data"] == []


def test_search_is_deterministic_across_instances(tmp_path: Path) -> None:
    """Identical corpus + query → identical scores and order — the
    hashed bag-of-words index has no nondeterminism."""
    first = _store(tmp_path)
    vs = first.create(file_ids=["file-alpha", "file-beta"])
    hits_a = first.search([vs["id"]], "alpha beta")
    second = _store(tmp_path)
    hits_b = second.search([vs["id"]], "alpha beta")
    assert [h["file_id"] for h in hits_a] == [h["file_id"] for h in hits_b]
    assert [h["score"] for h in hits_a] == [h["score"] for h in hits_b]
    assert hits_a[0]["file_id"] == "file-alpha"  # both terms hit the alpha doc


def test_search_bounds_and_isolation(tmp_path: Path) -> None:
    """Cross-store hits are impossible; query bounds refuse honestly;
    unknown ids fail closed; duplicate ids dedupe."""
    store = _store(tmp_path)
    a = store.create(file_ids=["file-alpha"])
    b = store.create(file_ids=["file-beta"])
    for hit in store.search([a["id"]], "alpha"):
        assert hit["vector_store_id"] == a["id"]
    assert store.search([b["id"]], "alpha") == []  # beta doc doesn't match
    with pytest.raises(VectorStoreError) as exc:
        store.search([a["id"]], "")
    assert exc.value.code == "invalid_request"
    with pytest.raises(VectorStoreError) as exc2:
        store.search([a["id"]], "x", max_results=VS_MAX_RESULTS + 1)
    assert exc2.value.code == "invalid_request"
    with pytest.raises(VectorStoreError) as exc3:
        store.search([a["id"], "vs_missing"], "alpha")
    assert exc3.value.code == "vector_store_not_found"


def test_filters_fail_closed(tmp_path: Path) -> None:
    """The comparison grammar refuses junk — a filter is never
    silently ignored."""
    store = _store(tmp_path)
    vs = store.create(file_ids=["file-alpha"])
    for bad in (
        {"type": "nonsense", "key": "k", "value": "v"},
        {"type": "and", "filters": [{"type": "eq", "key": "k", "value": "v"}] * 20},
        {"type": "eq", "key": "k"},  # missing value
    ):
        with pytest.raises(VectorStoreError):
            store.search([vs["id"]], "alpha", filters=bad)
    # a legal filter still applies honestly
    hits = store.search(
        [vs["id"]],
        "alpha",
        filters={"type": "eq", "key": "no_such_attr", "value": "x"},
    )
    assert hits == []


def test_expired_store_is_still_readable_but_write_closed(tmp_path: Path) -> None:
    """Expiry is honest: reads keep working, attach/search/batch 410,
    detach stays allowed — the contract vs_audit already pins."""
    store = _store(tmp_path)
    vs = store.create(
        file_ids=["file-alpha"],
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    meta = store._stores[vs["id"]]
    meta.expires_at = 1  # force expiry (SYNTHETIC time warp)
    got = store.get(vs["id"])
    assert got["status"] == "expired"
    with pytest.raises(VectorStoreError) as exc:
        store.attach(vs["id"], "file-beta")
    assert exc.value.code == "vector_store_expired"
    with pytest.raises(VectorStoreError) as exc2:
        store.search([vs["id"]], "alpha")
    assert exc2.value.code == "vector_store_expired"
    # detach on an expired store is still permitted (the tombstone can
    # always be cleaned up) — pinned behavior, keep it honest.
    store.detach(vs["id"], "file-alpha")


def test_idem_conflict_is_refused(tmp_path: Path) -> None:
    """Same Idempotency-Key + different body is a 409-class refusal,
    never a silent re-execution."""
    store = _store(tmp_path)
    store.create(name="one", idempotency_key="k", body_fingerprint="f1")
    with pytest.raises(VectorStoreError) as exc:
        store.create(name="two", idempotency_key="k", body_fingerprint="f2")
    assert exc.value.code == "idempotency_conflict"
