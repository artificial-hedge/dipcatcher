"""Deterministic schedules for the nested background-response handlers.

The route factory closes over stores/executors. These unit tests compile the
unchanged handler body with controlled dependencies to exercise precise race
windows without timing assumptions or live backends.
"""

from __future__ import annotations

import ast
import threading
import types
from pathlib import Path
from typing import Any

from fx1.serve import openai_compat
from fx1.serve.openai_compat import OpenAIEnvelopeStore


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str, **kwargs: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code


def _handler(name: str, bindings: dict[str, Any]) -> Any:
    source = Path(openai_compat.__file__).with_name("api.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    bindings.update(
        Any=Any,
        ApiError=ApiError,
        OPENAI_RESPONSE_TERMINAL=openai_compat.OPENAI_RESPONSE_TERMINAL,
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(Path(openai_compat.__file__).with_name("api.py")),
            "exec",
        ),
        bindings,
    )
    return bindings[name]


def test_completion_winner_does_not_signal_cancel() -> None:
    rid = "resp_review"

    class CompletionWinsStore(OpenAIEnvelopeStore):
        def put_unless_status(self, env: dict[str, Any], **kwargs: Any) -> bool:
            super().put({"id": rid, "object": "response", "status": "completed"})
            return super().put_unless_status(env, **kwargs)

        def transition_status(self, envelope_id: str, **kwargs: Any) -> bool:
            super().put({"id": rid, "object": "response", "status": "completed"})
            return super().transition_status(envelope_id, **kwargs)

    store = CompletionWinsStore()
    store.put({"id": rid, "object": "response", "status": "queued"})
    event = threading.Event()
    ns = dict(envelope_store=store, bg_cancel={rid: event})
    ns["_stored_envelope"] = _handler("_stored_envelope", ns)
    cancel = _handler("openai_response_cancel", ns)
    status = 200
    try:
        cancel(rid)
    except ApiError as exc:
        status = exc.status_code
    assert status == 409 and not event.is_set() and store.get(rid)["status"] == "completed"


def test_cancelled_before_start_does_not_call_backend() -> None:
    rid = "resp_review"
    event = threading.Event()

    class CancelBeforeStartStore(OpenAIEnvelopeStore):
        def transition_status(self, envelope_id: str, **kwargs: Any) -> bool:
            if kwargs.get("status") == "in_progress":
                super().put({"id": rid, "object": "response", "status": "cancelled"})
                event.set()
            return super().transition_status(envelope_id, **kwargs)

    store = CancelBeforeStartStore()
    store.put({"id": rid, "object": "response", "status": "queued"})
    calls: list[str] = []

    def core(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str, dict[str, int]]:
        calls.append("backend")
        return {"id": rid, "status": "completed"}, "completion", {}

    ns = dict(
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
    )
    worker = _handler("_bg_run", ns)
    worker()
    assert not calls and store.get(rid)["status"] == "cancelled"


def test_failed_worker_does_not_resurrect_deleted_response() -> None:
    rid = "resp_review"

    class DeleteBeforeWriteStore(OpenAIEnvelopeStore):
        def put_unless_status(self, env: dict[str, Any], **kwargs: Any) -> bool:
            self.delete(env["id"])
            return super().put_unless_status(env, **kwargs)

    store = DeleteBeforeWriteStore()
    store.put({"id": rid, "object": "response", "status": "in_progress"})
    ns = dict(envelope_store=store, cancel_ev=threading.Event())
    fail = _handler("_bg_fail", ns)
    fail(rid, error={"message": "synthetic test failure"})
    assert store.get(rid) is None
