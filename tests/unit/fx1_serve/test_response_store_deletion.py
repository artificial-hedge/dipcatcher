"""SYNTHETIC response-store races: completed work cannot undo deletion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from fx1.serve.openai_compat import OpenAIEnvelopeStore


@pytest.mark.parametrize("late_status", ["in_progress", "completed", "cancelled", "failed"])
def test_late_background_write_does_not_resurrect_deleted_response(late_status: str) -> None:
    store = OpenAIEnvelopeStore()
    store.put(
        {"id": "resp_deleted", "status": "queued"},
        items={"input_items": [{"id": "input_deleted"}]},
    )
    stale = store.get("resp_deleted")
    assert stale is not None
    assert store.delete("resp_deleted") is True
    stale["status"] = late_status
    assert store.put_if_present(stale, items={"input_items": [{"id": "late_input"}]}) is False
    assert store.get("resp_deleted") is None
    assert store.get_items("resp_deleted", "input_items") is None
    assert len(store) == 0


def test_late_background_write_does_not_undo_eviction() -> None:
    store = OpenAIEnvelopeStore(cap=1)
    store.put({"id": "resp_old", "status": "queued"})
    stale = store.get("resp_old")
    assert stale is not None
    replacement = {"id": "resp_new", "status": "completed"}
    store.put(replacement, items={"input_items": [{"id": "input_new"}]})
    stale["status"] = "completed"
    assert store.put_if_present(stale) is False
    assert store.get("resp_old") is None
    assert store.get("resp_new") == replacement
    assert store.get_items("resp_new", "input_items") == [{"id": "input_new"}]
    assert len(store) == 1


def test_present_update_keeps_payload_items_and_refreshes_eviction_order() -> None:
    store = OpenAIEnvelopeStore(cap=2)
    store.put({"id": "resp_a", "status": "queued"})
    store.put({"id": "resp_b", "status": "queued"})
    terminal = {"id": "resp_a", "status": "completed", "output": [{"text": "answer"}]}
    items = {"input_items": [{"id": "input_a"}]}
    assert store.put_if_present(terminal, items=items) is True
    store.put({"id": "resp_c", "status": "queued"})
    assert store.get("resp_a") == terminal
    assert store.get_items("resp_a", "input_items") == items["input_items"]
    assert store.get("resp_b") is None
    assert store.get("resp_c") is not None
    assert len(store) == 2


def test_delete_between_worker_read_and_write_wins() -> None:
    store = OpenAIEnvelopeStore()
    store.put({"id": "resp_race", "status": "queued"})
    read_done = Event()
    allow_write = Event()

    def late_worker() -> bool:
        stale = store.get("resp_race")
        assert stale is not None
        read_done.set()
        if not allow_write.wait(timeout=2):
            raise TimeoutError("test did not release the held background write")
        stale["status"] = "completed"
        return store.put_if_present(stale)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(late_worker)
        try:
            assert read_done.wait(timeout=2)
            assert store.delete("resp_race") is True
        finally:
            allow_write.set()
        assert future.result(timeout=2) is False
    assert store.get("resp_race") is None
