"""SYNTHETIC journal faults must not publish or resurrect conversation data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.openai_compat import OpenAIEnvelopeStore


def _envelope(eid: str = "conv_first") -> dict[str, Any]:
    return {
        "id": eid,
        "object": "conversation",
        "status": "queued",
        "metadata": {"source": "synthetic"},
        "output": [{"content": [{"text": "original"}]}],
    }


def _items() -> list[dict[str, Any]]:
    return [{"id": "item_original", "content": [{"text": "private synthetic text"}]}]


def _ids(store: OpenAIEnvelopeStore) -> list[str]:
    return [envelope["id"] for envelope in store.list_envelopes("conversation")]


@pytest.mark.parametrize("operation", ["status", "metadata", "repin", "items", "delete", "put"])
def test_failed_append_preserves_live_state_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    journal = JobJournal(tmp_path / "conversations.jsonl")
    store = OpenAIEnvelopeStore(cap=2, journal=journal)
    store.put(_envelope(), items={"items": _items()})
    store.put(_envelope("conv_second"))
    before_envelopes = store.list_envelopes("conversation")
    before_items = store.get_items("conv_first", "items")
    before_bytes = journal.path.read_bytes()

    def fail_append(payload: dict[str, Any]) -> None:
        raise OSError("SYNTHETIC failed journal append")

    monkeypatch.setattr(journal, "append", fail_append)
    with pytest.raises(OSError, match="SYNTHETIC"):
        if operation == "status":
            store.transition_status("conv_first", expect={"queued"}, status="in_progress")
        elif operation == "metadata":
            store.update_metadata("conv_first", {"source": "changed"})
        elif operation == "repin":
            store.repin(_envelope())
        elif operation == "items":
            store.mutate_items("conv_first", "items", lambda items: [*items, {"id": "new"}])
        elif operation == "delete":
            store.delete("conv_first")
        else:
            store.put(_envelope("conv_third"))
    assert store.list_envelopes("conversation") == before_envelopes
    assert store.get_items("conv_first", "items") == before_items
    assert journal.path.read_bytes() == before_bytes
    recovered = OpenAIEnvelopeStore(cap=2, journal=JobJournal(journal.path))
    assert recovered.list_envelopes("conversation") == before_envelopes
    assert recovered.get_items("conv_first", "items") == before_items


@pytest.mark.parametrize("corruption", ["delete_tail", "after_delete", "first_line"])
def test_damaged_journal_refuses_startup_without_rewriting(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "conversations.jsonl"
    store = OpenAIEnvelopeStore(journal=JobJournal(path))
    store.put(_envelope(), items={"items": _items()})
    assert store.delete("conv_first")
    lines = path.read_bytes().splitlines(keepends=True)
    if corruption == "delete_tail":
        damaged = lines[0] + lines[1][: len(lines[1]) // 2]
    elif corruption == "after_delete":
        damaged = b"".join(lines) + b'{"seq":'
    else:
        damaged = b'{"broken":true}\n' + lines[1]
    path.write_bytes(damaged)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="envelope journal is damaged"):
            OpenAIEnvelopeStore(journal=JobJournal(path))
        assert path.read_bytes() == damaged


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"op": "unknown"},
        {"op": "put", "envelope": {}},
        {"op": "put", "envelope": {"id": ""}},
        {"op": "put", "envelope": {"id": "conv_a"}, "items": []},
        {"op": "put", "envelope": {"id": "conv_a"}, "items": {"items": [7]}},
        {"op": "put", "envelope": {"id": "conv_a"}, "refresh": "false"},
        {"op": "set_items", "id": "conv_first", "key": "items", "items": [None]},
        {"op": "set_items", "id": "conv_first", "key": 5, "items": []},
        {"op": "delete", "id": None},
    ],
)
def test_verified_but_malformed_record_refuses_startup_without_compaction(
    tmp_path: Path, payload: Any
) -> None:
    path = tmp_path / "conversations.jsonl"
    journal = JobJournal(path)
    store = OpenAIEnvelopeStore(journal=journal)
    store.put(_envelope(), items={"items": _items()})
    journal.append(payload)
    original = path.read_bytes()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="invalid operation"):
            OpenAIEnvelopeStore(journal=JobJournal(path))
        assert path.read_bytes() == original


def test_nested_inputs_outputs_and_mutator_results_are_isolated(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    store = OpenAIEnvelopeStore(journal=JobJournal(path))
    env, items = _envelope(), _items()
    expected_env, expected_items = deepcopy(env), deepcopy(items)
    store.put(env, items={"items": items})
    env["output"][0]["content"][0]["text"] = "mutated caller envelope"
    items[0]["content"][0]["text"] = "mutated caller items"
    assert store.get("conv_first") == expected_env
    assert store.get_items("conv_first", "items") == expected_items
    result = store.get("conv_first")
    assert result is not None
    result["metadata"]["source"] = "mutated get"
    result["output"][0]["content"].clear()
    listed = store.list_envelopes("conversation")
    listed[0]["output"].clear()
    got_items = store.get_items("conv_first", "items")
    assert got_items is not None
    got_items[0]["content"].clear()
    assert store.get("conv_first") == expected_env
    assert store.get_items("conv_first", "items") == expected_items
    external_item = {"id": "new", "content": [{"text": "new text"}]}
    merged = store.mutate_items("conv_first", "items", lambda old: [*old, external_item])
    expected_items.append(deepcopy(external_item))
    external_item["content"].clear()
    assert merged is not None
    merged[0]["content"].clear()
    assert store.get_items("conv_first", "items") == expected_items
    repinned = store.repin(_envelope())
    repinned["output"].clear()
    assert store.get("conv_first") == expected_env
    recovered = OpenAIEnvelopeStore(journal=JobJournal(path))
    assert recovered.get("conv_first") == expected_env
    assert recovered.get_items("conv_first", "items") == expected_items


def test_throwing_mutator_does_not_modify_nested_live_state(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    store = OpenAIEnvelopeStore(journal=JobJournal(path))
    store.put(_envelope(), items={"items": _items()})
    original = path.read_bytes()

    def mutate_then_throw(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items[0]["content"][0]["text"] = "should never publish"
        raise ValueError("SYNTHETIC mutation failure")

    with pytest.raises(ValueError, match="SYNTHETIC"):
        store.mutate_items("conv_first", "items", mutate_then_throw)
    assert store.get_items("conv_first", "items") == _items()
    assert path.read_bytes() == original


@pytest.mark.parametrize("operation", ["status", "metadata", "items"])
def test_mutation_eviction_order_matches_replay(tmp_path: Path, operation: str) -> None:
    path = tmp_path / "conversations.jsonl"
    store = OpenAIEnvelopeStore(cap=2, journal=JobJournal(path))
    store.put(_envelope())
    store.put(_envelope("conv_second"))
    if operation == "status":
        assert store.transition_status("conv_first", expect={"queued"}, status="in_progress")
        expected_order = ["conv_first", "conv_second"]
    elif operation == "metadata":
        assert store.update_metadata("conv_first", {"new": "value"}) is not None
        expected_order = ["conv_first", "conv_second"]
    else:
        assert store.mutate_items("conv_first", "items", lambda _: _items()) == _items()
        expected_order = ["conv_second", "conv_first"]
    assert _ids(store) == expected_order
    recovered = OpenAIEnvelopeStore(cap=2, journal=JobJournal(path))
    assert _ids(recovered) == expected_order
    recovered.put(_envelope("conv_third"))
    assert _ids(recovered) == [expected_order[-1], "conv_third"]
    assert OpenAIEnvelopeStore(cap=2, journal=JobJournal(path)).get(expected_order[0]) is None


def test_parallel_item_mutations_preserve_each_write_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    store = OpenAIEnvelopeStore(journal=JobJournal(path))
    store.put(_envelope())

    def add(index: int) -> None:
        store.mutate_items("conv_first", "items", lambda old: [*old, {"id": str(index)}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(32)))
    items = store.get_items("conv_first", "items")
    assert items is not None
    assert {item["id"] for item in items} == {str(index) for index in range(32)}
    assert len(items) == 32
    recovered = OpenAIEnvelopeStore(journal=JobJournal(path))
    assert recovered.get_items("conv_first", "items") == items
    assert recovered.delete("conv_first")
    assert recovered.mutate_items("conv_first", "items", lambda _: _items()) is None
    assert OpenAIEnvelopeStore(journal=JobJournal(path)).get("conv_first") is None
