"""Flash context — store, retrieval, refresh. Offline; no network, no model."""

import threading
from pathlib import Path

import pytest

from fx1.flash.refresh import refresh
from fx1.flash.retrieve import context_block, retrieve
from fx1.flash.store import FlashStore


@pytest.fixture()
def store(tmp_path: Path) -> FlashStore:
    return FlashStore(tmp_path / "flash" / "entries.jsonl")


def _add_inflation(store: FlashStore) -> str:
    entry = store.add(
        text="US inflation was 3.4% in August 2026, measured by CPI year-over-year.",
        task="inflation research",
        tags=["macro", "us", "cpi"],
        sources=[{"url": "https://inflationtool.com/rates/usa", "title": "US inflation"}],
        uncertainty="low",
        verified=True,
    )
    return entry.id


# ---------------------------------------------------------------------------
# store: CRUD, provenance, tombstones, error reporting
# ---------------------------------------------------------------------------


def test_add_and_get_roundtrip(store: FlashStore) -> None:
    entry_id = _add_inflation(store)
    entry = store.get(entry_id)
    assert entry is not None
    assert entry.text.startswith("US inflation")
    assert entry.uncertainty == "low"
    assert entry.sources[0].url == "https://inflationtool.com/rates/usa"
    assert entry.verified is True
    assert entry.schema_version == "fx1.flash-entry/v1"


def test_add_rejects_empty_text(store: FlashStore) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        store.add(text="   ")


def test_update_preserves_history(store: FlashStore) -> None:
    entry_id = _add_inflation(store)
    revised = store.update(entry_id, {"uncertainty": "medium"})
    assert revised is not None
    assert revised.uncertainty == "medium"
    assert revised.text.startswith("US inflation")  # untouched fields survive
    # the file keeps both revisions as lines: original + tombstone + revision
    lines = [raw for raw in store.path.read_text().splitlines() if raw.strip()]
    assert len(lines) == 3


def test_update_rejects_unknown_fields(store: FlashStore) -> None:
    entry_id = _add_inflation(store)
    with pytest.raises(ValueError, match="cannot correct fields"):
        store.update(entry_id, {"id": "spoofed", "schema_version": "evil"})


def test_remove_tombstones(store: FlashStore) -> None:
    entry_id = _add_inflation(store)
    assert store.remove(entry_id) is True
    assert store.get(entry_id) is None
    assert store.remove(entry_id) is False


def test_torn_last_line_reported_not_fatal(store: FlashStore) -> None:
    _add_inflation(store)
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "fx1.flash-entry/v1", "id": "x", "te\n')
    assert store.get("x") is None
    assert store.read_errors() == ["line 2: not JSON"]


def test_mark_used_increments(store: FlashStore) -> None:
    entry_id = _add_inflation(store)
    marked = store.mark_used(entry_id)
    assert marked is not None
    assert marked.uses == 1
    assert marked.last_used_at
    assert store.mark_used("missing-id") is None


def test_list_filters_by_task_and_tags(store: FlashStore) -> None:
    _add_inflation(store)
    store.add(text="conformal prediction coverage", task="conformal bench", tags=["ml"])
    assert len(store.list()) == 2
    assert len(store.list(task="inflation")) == 1
    assert len(store.list(tags=["ml"])) == 1
    assert len(store.list(tags=["ml", "macro"])) == 2


# ---------------------------------------------------------------------------
# retrieval: relevance, recency, staleness, explainability
# ---------------------------------------------------------------------------


def test_retrieve_ranks_relevant_above_unrelated(store: FlashStore) -> None:
    inflation_id = _add_inflation(store)
    store.add(text="quantum chromodynamics binds quarks", task="physics", tags=["qcd"])
    hits = retrieve(store, "what is the current US inflation rate", k=3)
    assert hits
    assert hits[0].entry.id == inflation_id
    assert "term-match" in hits[0].reasons
    # unrelated physics entry must not surface
    assert all("physics" not in h.entry.task for h in hits)


def test_retrieve_excludes_below_threshold(store: FlashStore) -> None:
    store.add(text="quantum chromodynamics binds quarks", task="physics", tags=["qcd"])
    hits = retrieve(store, "US inflation rate now", k=3)
    assert hits == []


def test_retrieve_tag_and_task_boosts(store: FlashStore) -> None:
    store.add(text="some generic macro comment about the economy", task="markets")
    tagged = store.add(text="a note without the query words", task="other", tags=["inflation"])
    hits = retrieve(store, "inflation", k=3)
    ids = [h.entry.id for h in hits]
    assert tagged.id in ids
    tag_hit = next(h for h in hits if h.entry.id == tagged.id)
    assert "tag-match" in tag_hit.reasons


def test_stale_entries_demoted_and_flagged(store: FlashStore) -> None:
    entry = store.add(
        text="US inflation was 3.4% in August 2026 (CPI)",
        task="inflation",
        tags=["macro"],
        stale_after_s=60,
    )
    from datetime import UTC, datetime

    updated_at = datetime.fromisoformat(entry.updated_at)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    future = updated_at.timestamp() + 3600  # 1h later > 60s staleness window
    hits = retrieve(store, "US inflation rate", k=3, now=future)
    assert hits and hits[0].entry.id == entry.id
    assert hits[0].stale
    assert "stale" in hits[0].reasons


def test_private_entries_flagged(store: FlashStore) -> None:
    entry = store.add(
        text="internal volatility model secrets for the fund",
        task="internal",
        tags=["private"],
        private=True,
    )
    hits = retrieve(store, "volatility model", k=3)
    assert hits and hits[0].entry.id == entry.id
    assert "private" in hits[0].reasons


def test_context_block_carries_provenance(store: FlashStore) -> None:
    _add_inflation(store)
    hits = retrieve(store, "US inflation", k=2)
    block = context_block(hits)
    assert "https://inflationtool.com/rates/usa" in block
    assert "uncertainty: low" in block


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def test_refresh_records_reachable_sources(store: FlashStore) -> None:
    entry_id = _add_inflation(store)

    class FakeResult:
        status = 200
        error = ""

    report = refresh(store, entry_id, fetch_fn=lambda url: FakeResult())
    assert report is not None
    assert report.all_reachable is True
    entry = store.get(entry_id)
    assert entry is not None
    assert entry.refreshed_at
    assert "reachable" in entry.refresh_note


def test_refresh_marks_unreachable_honestly(store: FlashStore) -> None:
    entry_id = _add_inflation(store)

    class FakeResult:
        status = 404
        error = "not found"

    report = refresh(store, entry_id, fetch_fn=lambda url: FakeResult())
    assert report is not None
    assert report.all_reachable is False
    entry = store.get(entry_id)
    assert entry is not None
    assert "unreachable" in entry.refresh_note


def test_refresh_missing_id(store: FlashStore) -> None:
    assert refresh(store, "nope", fetch_fn=lambda url: None) is None


# ---------------------------------------------------------------------------
# cross-session persistence
# ---------------------------------------------------------------------------


def test_store_survives_reopen(store: FlashStore) -> None:
    entry_id = _add_inflation(store)
    reopened = FlashStore(store.path)
    entry = reopened.get(entry_id)
    assert entry is not None
    assert entry.text.startswith("US inflation")


def test_threaded_appends_all_survive(store: FlashStore) -> None:
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(n):
                store.add(text=f"finding {i} about volatility regimes", task="regime")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(10,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.all()) == 40
    assert store.read_errors() == []
