"""Adversarial regressions for batch idempotency and lifecycle races."""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import api
from fx1.serve.batch_audit import (
    _abatch_create,
    _abatch_item,
    _abatch_rows,
    _audit_resources,
    _batch_create,
    _chat_body,
    _client,
    _GateBackend,
    _line,
    _root_h,
    _StubBackend,
    _upload,
    _wait_abatch,
)
from fx1.serve.journal import JobJournal


def _endpoint(app: Any, path: str, method: str) -> Callable[..., Any]:
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def _nonlocal(fn: Callable[..., Any], name: str) -> Any:
    return inspect.getclosurevars(fn).nonlocals[name]


@contextmanager
def _clients() -> Iterator[None]:
    with _audit_resources():
        yield


def _concurrent_posts(
    client: Any, path: str, body: dict[str, Any], headers: dict[str, str]
) -> list[Any]:
    responses: list[Any] = []
    lock = threading.Lock()

    def submit() -> None:
        response = client.post(path, json=body, headers=headers)
        with lock:
            responses.append(response)

    threads = [threading.Thread(target=submit), threading.Thread(target=submit)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "concurrent idempotency request deadlocked"
    return responses


def test_openai_concurrent_first_use_executes_once() -> None:
    """A simultaneous first claim must create one batch, not two."""
    with _clients():
        gate = _GateBackend()
        client, app = _client(backend_map={"hosted_k3": lambda: gate}, max_inflight=4)
        create = _endpoint(app, "/v1/batches", "POST")
        store = _nonlocal(create, "batch_store")
        original_put = store.put
        entered = threading.Event()
        release = threading.Event()
        put_count = 0
        put_lock = threading.Lock()

        def slow_first_put(batch: Any) -> None:
            nonlocal put_count
            with put_lock:
                put_count += 1
                current = put_count
            if current == 1:
                entered.set()
                assert release.wait(5)
            original_put(batch)

        store.put = slow_first_put
        fid = _upload(client, [_line("same", _chat_body("same"))])
        body = {"input_file_id": fid, "endpoint": "/v1/chat/completions"}
        headers = {**_root_h(), "Idempotency-Key": "simultaneous-openai"}

        responses: list[Any] = []

        def first() -> None:
            responses.append(client.post("/v1/batches", json=body, headers=headers))

        first_thread = threading.Thread(target=first)
        first_thread.start()
        assert entered.wait(5)
        second_thread = threading.Thread(target=first)
        second_thread.start()
        time.sleep(0.1)
        observed_puts_while_first_blocked = put_count
        release.set()
        for thread in (first_thread, second_thread):
            thread.join(timeout=10)
            assert not thread.is_alive()
        gate.release.set()

        assert len(responses) == 2
        assert observed_puts_while_first_blocked == 1, (
            "a waiter entered creation before the first response was stored"
        )
        assert {response.status_code for response in responses} == {200}
        assert len({response.json()["id"] for response in responses}) == 1
        assert put_count == 1
        assert (
            sum(response.headers.get("X-Fx1-Idempotent-Replay") == "true" for response in responses)
            == 1
        )


def test_anthropic_concurrent_first_use_executes_once() -> None:
    """The Anthropic batch dialect uses the same atomic claim contract."""
    with _clients():
        gate = _GateBackend()
        client, app = _client(backend_map={"hosted_k3": lambda: gate}, max_inflight=4)
        create = _endpoint(app, "/v1/messages/batches", "POST")
        store = _nonlocal(create, "abatch_store")
        original_put = store.put
        entered = threading.Event()
        release = threading.Event()
        put_count = 0
        put_lock = threading.Lock()

        def slow_first_put(batch: Any) -> None:
            nonlocal put_count
            with put_lock:
                put_count += 1
                current = put_count
            if current == 1:
                entered.set()
                assert release.wait(5)
            original_put(batch)

        store.put = slow_first_put
        body = {"requests": [_abatch_item("same")]}
        headers = {
            **_root_h(),
            "anthropic-version": "2023-06-01",
            "Idempotency-Key": "simultaneous-anthropic",
        }
        responses: list[Any] = []

        def submit() -> None:
            responses.append(client.post("/v1/messages/batches", json=body, headers=headers))

        first_thread = threading.Thread(target=submit)
        first_thread.start()
        assert entered.wait(5)
        second_thread = threading.Thread(target=submit)
        second_thread.start()
        time.sleep(0.1)
        observed_puts_while_first_blocked = put_count
        release.set()
        for thread in (first_thread, second_thread):
            thread.join(timeout=10)
            assert not thread.is_alive()
        gate.release.set()

        assert len(responses) == 2
        assert observed_puts_while_first_blocked == 1
        assert {response.status_code for response in responses} == {200}
        assert len({response.json()["id"] for response in responses}) == 1
        assert put_count == 1
        assert (
            sum(response.headers.get("X-Fx1-Idempotent-Replay") == "true" for response in responses)
            == 1
        )


@pytest.mark.parametrize("dialect", ["openai", "anthropic"])
def test_batch_journal_failure_never_starts_provider_work(dialect: str, tmp_path: Path) -> None:
    """A failed durable publication must not leave an untracked worker."""
    with _clients():
        backend = _StubBackend()
        client, app = _client(
            backend_map={"hosted_k3": lambda: backend},
            max_inflight=1,
            state_dir=str(tmp_path / dialect),
        )
        if dialect == "openai":
            path = "/v1/batches"
            headers = _root_h()
            fid = _upload(client, [_line("unpublished", _chat_body("unpublished"))])
            body = {"input_file_id": fid, "endpoint": "/v1/chat/completions"}
            store_name = "batch_store"
        else:
            path = "/v1/messages/batches"
            headers = {**_root_h(), "anthropic-version": "2023-06-01"}
            body = {"requests": [_abatch_item("unpublished")]}
            store_name = "abatch_store"

        create = _endpoint(app, path, "POST")
        store = _nonlocal(create, store_name)
        submitted: list[Callable[[], None]] = []

        def capture_submit(fn: Callable[..., None], *args: Any) -> object:
            submitted.append(lambda: fn(*args))
            return object()

        app.state.jobs_executor.submit = capture_submit
        assert store._journal is not None

        def failed_journal_append(_payload: dict[str, Any]) -> None:
            raise OSError("synthetic journal write failure")

        store._journal.append = failed_journal_append
        response = client.post(path, json=body, headers=headers)
        assert response.status_code == 500

        assert len(submitted) == 1
        submitted[0]()
        # The assertion is about effects, not merely response shape: even a
        # queued callable that runs after the failed request must not reach
        # the provider or mutate an unjournaled record.
        assert backend.calls == 0
        assert store.list() == []


@pytest.mark.parametrize("dialect", ["openai", "anthropic"])
def test_batch_journal_failure_preserves_existing_capacity_entry(
    dialect: str, tmp_path: Path
) -> None:
    """A rejected replacement must neither publish nor evict live state."""
    journal_path = tmp_path / f"{dialect}.jsonl"
    journal = JobJournal(journal_path)
    if dialect == "openai":
        store: Any = api._BatchStore(1, journal=journal)
        old: Any = api._BatchRecord(
            batch_id="batch_old",
            input_file_id="file_old",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            status="completed",
            created_at=1,
            expires_at=2,
        )
        new: Any = api._BatchRecord(
            batch_id="batch_new",
            input_file_id="file_new",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            status="completed",
            created_at=1,
            expires_at=2,
        )
    else:
        store = api._AnthropicBatchStore(1, journal=journal)
        old = api._AnthropicBatchRecord(
            batch_id="msgbatch_old",
            status="ended",
            created_at=1,
            expires_at=2,
            ended_at=2,
        )
        new = api._AnthropicBatchRecord(
            batch_id="msgbatch_new",
            status="ended",
            created_at=1,
            expires_at=2,
            ended_at=2,
        )

    store.put(old)

    def failed_journal_append(_payload: dict[str, Any]) -> None:
        raise OSError("synthetic journal write failure")

    journal.append = failed_journal_append
    with pytest.raises(OSError, match="synthetic journal write failure"):
        store.put(new)

    assert [record.batch_id for record in store.list()] == [old.batch_id]
    recovered = type(store)(1, journal=JobJournal(journal_path))
    assert [record.batch_id for record in recovered.list()] == [old.batch_id]


class _SecondCallGate(_StubBackend):
    """Complete item one, then hold item two to expose the durable prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.second_entered = threading.Event()
        self.release = threading.Event()

    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        del sampling
        with self._lock:
            self.calls += 1
            call = self.calls
        if call == 2:
            self.second_entered.set()
            self.release.wait(timeout=10)
        return f"ok:{messages[-1]['content']}"


def test_anthropic_restart_preserves_committed_result_prefix(tmp_path: Path) -> None:
    """A crash after item N keeps item N instead of rewriting it as errored."""
    with _clients():
        backend = _SecondCallGate()
        client1, _ = _client(
            backend_map={"hosted_k3": lambda: backend},
            state_dir=str(tmp_path),
        )
        created = _abatch_create(client1, [_abatch_item("done"), _abatch_item("pending")])
        assert backend.second_entered.wait(5)

        client2, _ = _client(
            backend_map={"hosted_k3": lambda: _StubBackend()},
            state_dir=str(tmp_path),
        )
        recovered = _wait_abatch(client2, created["id"])
        rows = {row["custom_id"]: row["result"] for row in _abatch_rows(client2, created["id"])}
        backend.release.set()

        assert recovered["processing_status"] == "ended"
        assert rows["done"]["type"] == "succeeded"
        assert rows["pending"]["type"] == "errored"


def test_anthropic_delete_beats_late_terminal_mark(tmp_path: Path) -> None:
    """A delayed worker mark cannot resurrect a deleted batch on replay."""
    with _clients():
        client1, app1 = _client(
            backend_map={"hosted_k3": lambda: _StubBackend()},
            state_dir=str(tmp_path),
        )
        create = _endpoint(app1, "/v1/messages/batches", "POST")
        store = _nonlocal(create, "abatch_store")
        original_mark = store.mark
        terminal_mark_entered = threading.Event()
        release_terminal_mark = threading.Event()

        def delayed_terminal_mark(batch: Any) -> None:
            if batch.status == "ended" and not terminal_mark_entered.is_set():
                terminal_mark_entered.set()
                assert release_terminal_mark.wait(5)
            original_mark(batch)

        store.mark = delayed_terminal_mark
        created = _abatch_create(client1, [_abatch_item("done")])
        assert terminal_mark_entered.wait(5)
        deleted = client1.delete(
            f"/v1/messages/batches/{created['id']}",
            headers={**_root_h(), "anthropic-version": "2023-06-01"},
        )
        assert deleted.status_code == 200
        release_terminal_mark.set()
        time.sleep(0.1)

        client2, _ = _client(
            backend_map={"hosted_k3": lambda: _StubBackend()},
            state_dir=str(tmp_path),
        )
        recovered = client2.get(
            f"/v1/messages/batches/{created['id']}",
            headers={**_root_h(), "anthropic-version": "2023-06-01"},
        )
        assert recovered.status_code == 404


def test_terminal_batch_statuses_are_immutable() -> None:
    """Concurrent cancel attempts cannot regress completed/ended records."""
    with _clients():
        client, _ = _client(backend_map={"hosted_k3": lambda: _StubBackend()})
        openai = _batch_create(client, [_line("done", _chat_body("done"))])
        anthropic = _abatch_create(client, [_abatch_item("done")])
        # Wait for both workers before racing terminal cancels.
        from fx1.serve.batch_audit import _wait_batch  # noqa: PLC0415

        assert _wait_batch(client, openai["id"])["status"] == "completed"
        assert _wait_abatch(client, anthropic["id"])["processing_status"] == "ended"

        openai_codes: list[int] = []
        anthropic_codes: list[int] = []

        def cancel_openai() -> None:
            openai_codes.append(
                client.post(f"/v1/batches/{openai['id']}/cancel", headers=_root_h()).status_code
            )

        def cancel_anthropic() -> None:
            anthropic_codes.append(
                client.post(
                    f"/v1/messages/batches/{anthropic['id']}/cancel",
                    headers={**_root_h(), "anthropic-version": "2023-06-01"},
                ).status_code
            )

        threads = [threading.Thread(target=cancel_openai) for _ in range(8)] + [
            threading.Thread(target=cancel_anthropic) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert openai_codes == [409] * 8
        assert anthropic_codes == [400] * 8
        assert (
            client.get(f"/v1/batches/{openai['id']}", headers=_root_h()).json()["status"]
            == "completed"
        )
        assert (
            client.get(
                f"/v1/messages/batches/{anthropic['id']}",
                headers={**_root_h(), "anthropic-version": "2023-06-01"},
            ).json()["processing_status"]
            == "ended"
        )
