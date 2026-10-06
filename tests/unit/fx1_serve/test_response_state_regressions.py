"""SYNTHETIC, deterministic schedules for background-response state races.

Compile the unchanged route bodies with controlled dependencies so each
interleaving is reproducible without real backends or timing assumptions.
"""

from __future__ import annotations

import ast
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import openai_compat
from fx1.serve.api import _REQUEST_KEY_ID
from fx1.serve.openai_compat import OpenAIEnvelopeStore


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str, **kwargs: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code


def _handler(name: str, bindings: dict[str, Any]) -> Any:
    source_path = Path(openai_compat.__file__).with_name("api.py")
    tree = ast.parse(source_path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    bindings.update(
        Any=Any,
        ApiError=ApiError,
        OPENAI_RESPONSE_TERMINAL=openai_compat.OPENAI_RESPONSE_TERMINAL,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), bindings)
    return bindings[name]


@pytest.mark.parametrize("terminal_status", ["completed", "incomplete"])
def test_completion_winner_does_not_signal_cancel(terminal_status: str) -> None:
    rid = "resp_review"

    class CompletionWinsStore(OpenAIEnvelopeStore):
        def put_if_present(self, env: dict[str, Any], **kwargs: Any) -> bool:
            super().put({"id": rid, "object": "response", "status": terminal_status})
            return super().put_if_present(env, **kwargs)

        def transition_status(self, envelope_id: str, **kwargs: Any) -> bool:
            super().put({"id": rid, "object": "response", "status": terminal_status})
            return super().transition_status(envelope_id, **kwargs)

    store = CompletionWinsStore()
    store.put({"id": rid, "object": "response", "status": "queued"})
    event = threading.Event()
    bindings = dict(envelope_store=store, bg_cancel={rid: event})
    bindings["_stored_envelope"] = _handler("_stored_envelope", bindings)
    cancel = _handler("openai_response_cancel", bindings)
    status = 200
    try:
        cancel(rid)
    except ApiError as exc:
        status = exc.status_code
    stored = store.get(rid)
    assert stored is not None
    assert status == 409
    assert not event.is_set()
    assert stored["status"] == terminal_status


@pytest.mark.parametrize("intervention", ["cancelled", "deleted"])
def test_inactive_before_start_does_not_call_backend(intervention: str) -> None:
    rid = "resp_review"
    event = threading.Event()

    class InactiveBeforeStartStore(OpenAIEnvelopeStore):
        def _intervene(self) -> None:
            if intervention == "deleted":
                self.delete(rid)
            else:
                super().put({"id": rid, "object": "response", "status": "cancelled"})
            event.set()

        def get(self, envelope_id: str) -> dict[str, Any] | None:
            stale = super().get(envelope_id)
            if stale is not None and stale.get("status") == "queued":
                self._intervene()
            return stale

        def transition_status(self, envelope_id: str, **kwargs: Any) -> bool:
            if kwargs.get("status") == "in_progress":
                self._intervene()
            return super().transition_status(envelope_id, **kwargs)

    store = InactiveBeforeStartStore()
    store.put({"id": rid, "object": "response", "status": "queued"})
    calls: list[str] = []

    def core(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str, dict[str, int]]:
        calls.append("backend")
        return {"id": rid, "status": "completed"}, "completion", {}

    bindings = dict(
        envelope_store=store,
        cancel_ev=event,
        bg_cancel={rid: event},
        inflight=threading.Semaphore(1),
        rid=rid,
        queued={"created_at": 1},
        body=object(),
        request=types.SimpleNamespace(headers={}),
        key=None,
        _openai_response_core=core,
        bg_key_id=None,
        _REQUEST_KEY_ID=_REQUEST_KEY_ID,
    )
    _handler("_bg_run", bindings)()
    assert not calls
    stored = store.get(rid)
    if intervention == "deleted":
        assert stored is None
    else:
        assert stored is not None and stored["status"] == "cancelled"


@pytest.mark.parametrize("winner", ["deleted", "cancelled", "completed", "incomplete"])
def test_failed_worker_preserves_delete_or_terminal_winner(winner: str) -> None:
    rid = "resp_review"

    class WinnerBeforeWriteStore(OpenAIEnvelopeStore):
        def _intervene(self) -> None:
            if winner == "deleted":
                self.delete(rid)
            else:
                super().put({"id": rid, "object": "response", "status": winner})

        def put_if_present(self, env: dict[str, Any], **kwargs: Any) -> bool:
            self._intervene()
            return super().put_if_present(env, **kwargs)

        def put_unless_status(self, env: dict[str, Any], **kwargs: Any) -> bool:
            self._intervene()
            return super().put_unless_status(env, **kwargs)

    store = WinnerBeforeWriteStore()
    store.put({"id": rid, "object": "response", "status": "in_progress"})
    bindings = dict(envelope_store=store, cancel_ev=threading.Event())
    _handler("_bg_fail", bindings)(rid, error={"message": "synthetic test failure"})
    stored = store.get(rid)
    if winner == "deleted":
        assert stored is None
    else:
        assert stored is not None and stored["status"] == winner
        assert "error" not in stored


@pytest.mark.parametrize("intervention", ["active", "cancelled", "deleted"])
def test_only_committed_background_results_append_to_conversation(intervention: str) -> None:
    rid = "resp_review"
    store = OpenAIEnvelopeStore()
    store.put({"id": rid, "object": "response", "status": "in_progress"})
    conversations = OpenAIEnvelopeStore()
    original_items = [{"id": "prior_item"}]
    conversations.put(
        {"id": "conv_review", "object": "conversation"}, items={"items": original_items}
    )
    body = types.SimpleNamespace(
        previous_response_id=None,
        conversation="conv_review",
        input=[{"id": "new_input"}],
        store=True,
    )
    body.model_copy = lambda *, update: types.SimpleNamespace(**{**vars(body), **update})
    output = types.SimpleNamespace(
        content="synthetic answer",
        tool_calls=None,
        completion_id="completion_review",
        logprobs=None,
        model="fx1",
        usage={"input_tokens": 1},
    )

    def complete(**kwargs: Any) -> Any:
        if intervention == "deleted":
            store.delete(rid)
        elif intervention == "cancelled":
            store.put({"id": rid, "object": "response", "status": "cancelled"})
        return output

    bindings = dict(
        envelope_store=store,
        conv_store=conversations,
        conversation_id_of=lambda value: value,
        chained_response_input=lambda envelope, old, current: current,
        _file_search_turn=lambda request, effective: ([], effective),
        response_to_kwargs=lambda *args, **kwargs: {},
        ft_store=types.SimpleNamespace(checkpoint_for=lambda _: None),
        CompleteRequest=lambda **kwargs: types.SimpleNamespace(**kwargs),
        complete=complete,
        Response=object,
        validate_response_format=lambda *args: None,
        response_text_format=lambda request: None,
        response_cap_call_items=lambda request, calls: ([], None),
        uuid=types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex="synthetic")),
        openai_response_object=lambda **kwargs: {
            "id": kwargs["rid"],
            "object": "response",
            "status": kwargs["status"],
            "output": [{"id": "new_output", "text": kwargs["content"]}],
        },
        response_input_items_for_store=lambda items, **kwargs: [dict(item) for item in items],
    )
    bindings["_complete_request_from_kwargs"] = _handler("_complete_request_from_kwargs", bindings)
    envelope, completion_id, usage = _handler("_openai_response_core", bindings)(body, {}, rid=rid)
    assert completion_id == "completion_review" and usage == output.usage
    stored = store.get(rid)
    items = conversations.get_items("conv_review", "items")
    if intervention == "active":
        assert stored is not None and stored["status"] == "completed"
        assert stored["_fx1_completion_id"] == completion_id
        assert items == [*original_items, *body.input, *envelope["output"]]
    else:
        assert items == original_items
        if intervention == "deleted":
            assert stored is None
        else:
            assert stored is not None and stored["status"] == "cancelled"
