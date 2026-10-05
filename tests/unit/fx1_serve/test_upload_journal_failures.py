"""Native UploadStore regressions for an append that fails before writing.

Only the append boundary and wall clock are fault-injected. Real JobJournal,
blob persistence, replay, garbage collection and retry are exercised.
"""

from __future__ import annotations

import itertools
import shutil
from types import SimpleNamespace

import pytest

import fx1.serve.uploads as uploads


class InjectedAppendFailure(OSError):
    pass


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(uploads.time, "time", lambda: clock[0])
    return clock


@pytest.fixture(autouse=True)
def fixed_ids(monkeypatch):
    ids = itertools.count(1)
    monkeypatch.setattr(
        uploads, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=f"{next(ids):032x}"))
    )


def create(store, *, nbytes=4):
    return store.create(
        purpose="evals", filename="data.jsonl", nbytes=nbytes, mime_type="application/jsonl"
    )


def inject_before_write(monkeypatch, store, operation):
    """The injected exception demonstrably occurs before any journal write."""
    journal = store._journal
    before = journal.path.read_bytes()
    calls = []

    def fail(payload):
        calls.append(payload)
        raise InjectedAppendFailure("injected before journal write")

    with monkeypatch.context() as patch:
        patch.setattr(journal, "append", fail)
        with pytest.raises(InjectedAppendFailure, match="before journal write"):
            operation()
    assert len(calls) == 1
    assert journal.path.read_bytes() == before


def restart_copy(source, destination, *, max_entries=8):
    """Replay a disk snapshot without changing a still-live store's journal."""
    shutil.copytree(source, destination)
    return uploads.UploadStore(
        max_entries=max_entries, max_bytes=100, state_dir=destination, ttl_s=10
    )


def observe(issues, label, condition, actual):
    if not condition:
        issues.append(f"{label}: {actual!r}")


def try_operation(issues, label, operation):
    try:
        return operation()
    except Exception as exc:
        issues.append(f"{label}: {type(exc).__name__}: {exc}")
        return None


def observe_pending(issues, label, store, uid, part_id, data):
    meta = store._uploads.get(uid)
    observe(
        issues,
        f"{label} metadata",
        meta is not None
        and meta.status == "pending"
        and meta.file_id is None
        and meta.parts == {part_id: len(data)},
        None if meta is None else meta.model_dump(),
    )
    blob = store._blob(uid, part_id)
    observe(
        issues,
        f"{label} old durable bytes",
        blob.is_file() and blob.read_bytes() == data,
        blob.read_bytes() if blob.is_file() else "missing blob",
    )
    observe(issues, f"{label} replay warnings", not store.recover_warnings, store.recover_warnings)


def test_add_part_append_failure_does_not_publish_or_consume_capacity(tmp_path, monkeypatch):
    state = tmp_path / "state"
    store = uploads.UploadStore(8, 100, state_dir=state, ttl_s=10)
    meta = create(store)
    old_part = store.add_part(meta.upload_id, b"ab")["id"]
    inject_before_write(monkeypatch, store, lambda: store.add_part(meta.upload_id, b"cd"))

    issues = []
    observe_pending(issues, "live after failed add", store, meta.upload_id, old_part, b"ab")
    restored = restart_copy(state, tmp_path / "restarted")
    observe_pending(issues, "restarted after failed add", restored, meta.upload_id, old_part, b"ab")
    for name, current in (("live", store), ("restarted", restored)):
        part = try_operation(
            issues,
            f"{name} retry add",
            lambda current=current: current.add_part(meta.upload_id, b"cd"),
        )
        if part is not None:
            actual = try_operation(
                issues,
                f"{name} assembly after retry",
                lambda current=current, part=part: current.assemble(
                    meta.upload_id, [old_part, part["id"]]
                ),
            )
            observe(issues, f"{name} assembly", actual == b"abcd", actual)
    assert not issues, "\n".join(issues)


@pytest.mark.parametrize("transition", ["complete", "cancel", "expiry"])
def test_terminal_append_failure_preserves_pending_bytes_and_retry(
    tmp_path, monkeypatch, fixed_clock, transition
):
    state = tmp_path / "state"
    store = uploads.UploadStore(8, 100, state_dir=state, ttl_s=10)
    meta = create(store)
    pid = store.add_part(meta.upload_id, b"data")["id"]

    def transition_on(current):
        if transition == "complete":
            # Exercise the ordinary assemble-before-complete sequence on retry.
            data = current.assemble(meta.upload_id, [pid])
            return current.complete(meta.upload_id, [pid], content=data, file_id="file_test")
        if transition == "cancel":
            return current.cancel(meta.upload_id)
        fixed_clock[0] = 1011.0
        return current.get(meta.upload_id)

    inject_before_write(monkeypatch, store, lambda: transition_on(store))
    issues = []
    observe_pending(issues, "live after failed terminal", store, meta.upload_id, pid, b"data")
    restored = restart_copy(state, tmp_path / "restarted")
    observe_pending(
        issues, "restarted after failed terminal", restored, meta.upload_id, pid, b"data"
    )
    expected_status = {"complete": "completed", "cancel": "cancelled", "expiry": "expired"}[
        transition
    ]

    for name, current in (("live", store), ("restarted", restored)):
        retried = try_operation(
            issues, f"{name} terminal retry", lambda current=current: transition_on(current)
        )
        observe(
            issues,
            f"{name} retry status",
            retried is not None and retried.status == expected_status,
            None if retried is None else retried.status,
        )
        if retried is not None:
            again = restart_copy(current._journal.path.parent, tmp_path / f"after_retry_{name}")
            committed = again._uploads[meta.upload_id]
            observe(
                issues,
                f"{name} retried terminal durability",
                committed.status == expected_status,
                committed.model_dump(),
            )
    assert not issues, "\n".join(issues)


@pytest.mark.parametrize("max_entries", [1, 8], ids=["with_eviction", "without_eviction"])
def test_create_append_failure_does_not_publish_or_evict(tmp_path, monkeypatch, max_entries):
    state = tmp_path / "state"
    store = uploads.UploadStore(max_entries, 100, state_dir=state, ttl_s=10)
    old = create(store)
    pid = store.add_part(old.upload_id, b"data")["id"]
    inject_before_write(monkeypatch, store, lambda: create(store))

    issues = []
    observe(
        issues,
        "live membership after failed create",
        list(store._uploads) == [old.upload_id],
        list(store._uploads),
    )
    observe_pending(issues, "live after failed create", store, old.upload_id, pid, b"data")
    restored = restart_copy(state, tmp_path / "restarted", max_entries=max_entries)
    observe(
        issues,
        "restarted membership after failed create",
        list(restored._uploads) == [old.upload_id],
        list(restored._uploads),
    )
    observe_pending(issues, "restarted after failed create", restored, old.upload_id, pid, b"data")

    for name, current in (("live", store), ("restarted", restored)):
        retried = try_operation(
            issues, f"{name} create retry", lambda current=current: create(current)
        )
        if retried is not None:
            expected_ids = (
                [retried.upload_id] if max_entries == 1 else [old.upload_id, retried.upload_id]
            )
            observe(
                issues,
                f"{name} membership after create retry",
                list(current._uploads) == expected_ids,
                list(current._uploads),
            )
            again = restart_copy(
                current._journal.path.parent,
                tmp_path / f"after_retry_{name}",
                max_entries=max_entries,
            )
            # Existing native contract keeps evicted intents as expired after
            # replay, although the running bounded store returns 404 for them.
            observe(
                issues,
                f"{name} recovered membership after retry",
                list(again._uploads) == [old.upload_id, retried.upload_id],
                list(again._uploads),
            )
            expected_old_status = "expired" if max_entries == 1 else "pending"
            recovered_old = again._uploads.get(old.upload_id)
            observe(
                issues,
                f"{name} recovered prior upload after retry",
                recovered_old is not None and recovered_old.status == expected_old_status,
                None if recovered_old is None else recovered_old.status,
            )
    assert not issues, "\n".join(issues)
