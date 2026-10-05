"""Generated conversation item IDs must not reuse a deleted list position."""

from __future__ import annotations

import pytest

from fx1.harness import Harness
from fx1.sdk import Fx1Harness
from fx1.serve import conv_audit as audit


@pytest.mark.parametrize("delete_index", [0, 1, 2])
def test_http_append_ids_remain_fresh_after_delete_and_restart(delete_index: int) -> None:
    with audit._audit_context():
        state = audit._temporary_directory()
        first, _ = audit._client({audit._MODEL: audit._StubBackend}, state_dir=state)
        cid = audit._conv_create(first, items=[audit._msg(str(i)) for i in range(3)])["id"]
        original = audit._item_ids(audit._items(first, cid))
        assert (
            first.delete(f"/v1/conversations/{cid}/items/{original[delete_index]}").status_code
            == 200
        )
        after_delete = [value for i, value in enumerate(original) if i != delete_index]
        next_client, _ = audit._client({audit._MODEL: audit._StubBackend}, state_dir=state)
        minted = audit._items_add(next_client, cid, [audit._msg("new")])["data"][0]["id"]
        assert minted not in original
        assert audit._item_ids(audit._items(next_client, cid)) == [*after_delete, minted]
        assert next_client.delete(f"/v1/conversations/{cid}/items/{minted}").status_code == 200
        assert audit._item_ids(audit._items(next_client, cid)) == after_delete
        journal = state / "conversations.jsonl"
        before = journal.read_bytes()
        assert next_client.delete(f"/v1/conversations/{cid}/items/missing").status_code == 404
        assert journal.read_bytes() == before


def test_sdk_append_ids_remain_fresh_after_delete_and_restart() -> None:
    with audit._audit_context():
        state = audit._temporary_directory()

        def harness() -> Fx1Harness:
            return Fx1Harness(
                harness=Harness(runner=lambda argv, timeout_s: (0, "synthetic", "")),
                backend_resolver=lambda name, *args, **kwargs: audit._StubBackend(),
                state_dir=state,
            )

        first = harness()
        cid = first.openai_conversation_create(items=[audit._msg("a"), audit._msg("b")])["id"]
        original = audit._item_ids(first.openai_conversation_items(cid))
        first.openai_conversation_item_delete(cid, original[0])
        second = harness()
        minted = second.openai_conversation_items_add(cid, [audit._msg("new")])["data"][0]["id"]
        assert minted not in original
        assert audit._item_ids(second.openai_conversation_items(cid)) == [original[1], minted]
