"""SYNTHETIC fault injection for eval admission and worker resource ownership."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Response

from fx1.harness import Harness
from fx1.serve import api
from fx1.serve.evals import EvalRecord
from fx1.serve.journal import JobJournal


class _Backend:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def probe(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[FastAPI, list[Callable[[], None]], list[_Backend]]]:
    monkeypatch.delenv("FX1_API_STATE_DIR", raising=False)
    pending: list[Callable[[], None]] = []
    backends: list[_Backend] = []

    def resolve(*args: Any, **kwargs: Any) -> _Backend:
        backend = _Backend()
        backends.append(backend)
        return backend

    app = api.create_app(
        harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
        backend_resolver=resolve,
        max_inflight=1,
        breaker_threshold=0,
    )

    def enqueue(fn: Callable[[], None]) -> Future[None]:
        pending.append(fn)
        return Future()

    def run(record: EvalRecord, **kwargs: Any) -> None:
        record.status = "succeeded"
        record.report = {"data_label": "SYNTHETIC"}
        record.finished_at = time.time()

    monkeypatch.setattr(app.state.jobs_executor, "submit", enqueue)
    monkeypatch.setattr(api, "run_eval_record", run)
    try:
        yield app, pending, backends
    finally:
        app.state.jobs_executor.shutdown(wait=True, cancel_futures=True)


def _submit(app: FastAPI, *, judge: bool = False) -> Any:
    route = next(
        r
        for r in app.routes
        if getattr(r, "path", None) == "/harness/evals" and "POST" in getattr(r, "methods", set())
    )
    return route.endpoint(
        body=api.EvalSubmitRequest(
            suite="capability" if judge else "calibration",
            backend="hosted_k3",
            judge_backend="hosted_k3" if judge else None,
        ),
        response=Response(),
        idempotency_key="synthetic-eval",
    )


def _assert_capacity_released(app: FastAPI) -> None:
    assert app.state.metrics.snapshot().inflight == 0
    assert app.state.inflight_slots.acquire(blocking=False)
    app.state.inflight_slots.release()


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_journal_failure_never_leaks_eval_capacity(
    probe: tuple[FastAPI, list[Callable[[], None]], list[_Backend]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_at: int,
) -> None:
    """Fail the queued write, running write, and final write independently."""
    app, pending, backends = probe
    journal = JobJournal(tmp_path / "evals.jsonl")
    append = journal.append
    calls = 0

    def fail_once(payload: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("SYNTHETIC journal failure")
        append(payload)

    monkeypatch.setattr(journal, "append", fail_once)
    monkeypatch.setattr(app.state.eval_store, "_journal", journal)
    if fail_at == 1:
        with pytest.raises(OSError, match="SYNTHETIC journal failure"):
            _submit(app)
        assert not pending
        assert app.state.eval_store.get_key("synthetic-eval") is None
    else:
        submitted = _submit(app)
        if fail_at == 3:
            with pytest.raises(OSError, match="SYNTHETIC journal failure"):
                pending.pop()()
        else:
            pending.pop()()
        record = app.state.eval_store.get(submitted.eval_id)
        assert record.status == ("failed" if fail_at == 2 else "succeeded")
        if fail_at == 2:
            assert not backends  # A failed start never performs model work.
    _assert_capacity_released(app)


def test_executor_refusal_releases_capacity_even_if_tombstone_write_fails(
    probe: tuple[FastAPI, list[Callable[[], None]], list[_Backend]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app, pending, _ = probe
    journal = JobJournal(tmp_path / "evals.jsonl")
    append = journal.append

    def tombstone_fails(payload: dict[str, Any]) -> None:
        if "deleted" in payload:
            raise OSError("SYNTHETIC tombstone failure")
        append(payload)

    def closed_executor(fn: Callable[[], None]) -> Future[None]:
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(journal, "append", tombstone_fails)
    monkeypatch.setattr(app.state.eval_store, "_journal", journal)
    monkeypatch.setattr(app.state.jobs_executor, "submit", closed_executor)
    with pytest.raises(OSError, match="SYNTHETIC tombstone failure"):
        _submit(app)
    assert not pending
    _assert_capacity_released(app)


@pytest.mark.parametrize("fail_runner", [False, True])
def test_eval_closes_primary_and_judge_backends(
    probe: tuple[FastAPI, list[Callable[[], None]], list[_Backend]],
    monkeypatch: pytest.MonkeyPatch,
    fail_runner: bool,
) -> None:
    app, pending, backends = probe
    if fail_runner:

        def fail(record: EvalRecord, **kwargs: Any) -> None:
            raise RuntimeError("SYNTHETIC runner failure")

        monkeypatch.setattr(api, "run_eval_record", fail)
    submitted = _submit(app, judge=True)
    pending.pop()()
    assert len(backends) == 2
    assert all(backend.closed for backend in backends)
    assert app.state.eval_store.get(submitted.eval_id).status == (
        "failed" if fail_runner else "succeeded"
    )
    _assert_capacity_released(app)


def test_queued_cancellation_releases_capacity_without_resolving_backend(
    probe: tuple[FastAPI, list[Callable[[], None]], list[_Backend]],
) -> None:
    app, pending, backends = probe
    submitted = _submit(app)
    record, outcome = app.state.eval_store.cancel(submitted.eval_id)
    assert outcome == "cancelled"
    pending.pop()()
    assert record.status == "cancelled"
    assert not backends
    _assert_capacity_released(app)


@pytest.mark.parametrize("failing_backend", [0, 1])
def test_backend_close_failure_still_finalizes_and_releases_capacity(
    probe: tuple[FastAPI, list[Callable[[], None]], list[_Backend]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_backend: int,
) -> None:
    app, pending, backends = probe
    journal = JobJournal(tmp_path / "evals.jsonl")
    monkeypatch.setattr(app.state.eval_store, "_journal", journal)

    def close(backend: _Backend) -> None:
        backend.closed = True
        if backend is backends[failing_backend]:
            raise OSError("SYNTHETIC backend close failure")

    monkeypatch.setattr(_Backend, "close", close)
    submitted = _submit(app, judge=True)
    with pytest.raises(OSError, match="SYNTHETIC backend close failure"):
        pending.pop()()
    assert all(backend.closed for backend in backends)
    latest = journal.replay().payloads[-1]["record"]
    assert latest["eval_id"] == submitted.eval_id
    assert latest["status"] == "succeeded"
    _assert_capacity_released(app)
