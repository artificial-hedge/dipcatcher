"""Upload completion serializes file publication with terminal transitions."""

from __future__ import annotations

import inspect
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import fx1.sdk as sdk
import fx1.serve.api as api
import fx1.serve.uploads as uploads
from fx1.harness import Harness


class UploadLock:
    def __init__(self, original, publishing, contender):
        self.original, self.publishing, self.contender = original, publishing, contender

    def __enter__(self):
        if self.publishing.is_set() and self.original.locked():
            self.contender.set()
        self.original.acquire()
        return self

    def __exit__(self, *args):
        self.original.release()


@contextmanager
def surface(kind, state, monkeypatch):
    for key in list(os.environ):
        if key.startswith("FX1_API_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("FX1_API_KEY", "test-upload-completion")

    def unused(*args, **kwargs):
        raise AssertionError("test must not call a backend or harness command")

    if kind == "api":
        app = api.create_app(
            harness=Harness(runner=unused), backend_resolver=unused, state_dir=state
        )
        endpoint = next(
            route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/uploads"
        )
        store = inspect.getclosurevars(endpoint).nonlocals["upload_store"]
        with TestClient(app, headers={"X-API-Key": "test-upload-completion"}) as client:
            created = client.post(
                "/v1/uploads",
                json={
                    "purpose": "batch",
                    "filename": "test.jsonl",
                    "bytes": 2,
                    "mime_type": "application/jsonl",
                },
            )
            assert created.status_code == 200
            uid = created.json()["id"]
            part = client.post(f"/v1/uploads/{uid}/parts", files={"data": ("part.bin", b"ab")})
            assert part.status_code == 200
            pid = part.json()["id"]

            def complete(md5=None):
                body = {"part_ids": [pid]}
                if md5 is not None:
                    body["md5"] = md5
                r = client.post(f"/v1/uploads/{uid}/complete", json=body)
                return r.status_code, r.json()

            def cancel():
                r = client.post(f"/v1/uploads/{uid}/cancel")
                return r.status_code, r.json()

            yield SimpleNamespace(
                store=store,
                uid=uid,
                pid=pid,
                complete=complete,
                cancel=cancel,
                files=lambda: app.state.file_store.list(),
                file_store=app.state.file_store,
                sdk=None,
            )
    else:
        harness = sdk.Fx1Harness(state_dir=state, backend_resolver=unused)
        uid = harness.upload_create(bytes=2)["id"]
        pid = harness.upload_part(uid, b"ab")["id"]

        def wire(fn):
            try:
                return 200, fn()
            except uploads.UploadStoreError as exc:
                return exc.status, {"error": {"code": exc.code}}

        yield SimpleNamespace(
            store=harness._upload_store,
            uid=uid,
            pid=pid,
            complete=lambda md5=None: wire(lambda: harness.upload_complete(uid, [pid], md5=md5)),
            cancel=lambda: wire(lambda: harness.upload_cancel(uid)),
            files=lambda: list(harness._files.values()),
            file_store=None,
            sdk=harness,
        )


def block_publication(s, publishing, contender, release, monkeypatch):
    calls = []

    def after_publish():
        calls.append(None)
        if len(calls) == 1:
            publishing.set()
            assert release.wait(5), "first publisher was not released"
        else:
            contender.set()

    monkeypatch.setattr(s.store, "_lock", UploadLock(s.store._lock, publishing, contender))
    if s.file_store is not None:
        original = s.file_store.put

        def blocked_put(**kwargs):
            rec = original(**kwargs)
            after_publish()
            return rec

        monkeypatch.setattr(s.file_store, "put", blocked_put)
    else:
        original = s.sdk._files_lock

        class FileLock:
            def __enter__(self):
                original.acquire()

            def __exit__(self, *args):
                original.release()
                after_publish()

        monkeypatch.setattr(s.sdk, "_files_lock", FileLock())
    return calls


@pytest.mark.parametrize("kind", ["api", "sdk"])
@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("competitor", ["complete", "cancel"])
def test_competing_lifecycle_cannot_publish_an_orphan(
    kind, durable, competitor, tmp_path, monkeypatch
):
    with surface(kind, tmp_path / "state" if durable else None, monkeypatch) as s:
        publishing, contender, release = threading.Event(), threading.Event(), threading.Event()
        calls = block_publication(s, publishing, contender, release, monkeypatch)

        def compete():
            result = s.complete() if competitor == "complete" else s.cancel()
            contender.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(s.complete)
            try:
                assert publishing.wait(5)
                second = pool.submit(compete)
                assert contender.wait(5), (
                    "competitor never reached publication or locked transition"
                )
            finally:
                release.set()
            responses = [first.result(timeout=5), second.result(timeout=5)]
        assert sorted(status for status, _ in responses) == [200, 409]
        assert len(calls) == 1, f"published {len(calls)} files: {responses}"
        assert len(s.files()) == 1
        assert s.store.get(s.uid).status == "completed"
        assert responses[0][0] == 200


@pytest.mark.parametrize("kind", ["api", "sdk"])
def test_expiry_during_publication_does_not_orphan_file(kind, tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(uploads, "time", SimpleNamespace(time=lambda: clock[0]))
    with surface(kind, None, monkeypatch) as s:
        publishing, contender, release = threading.Event(), threading.Event(), threading.Event()
        block_publication(s, publishing, contender, release, monkeypatch)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(s.complete)
            try:
                assert publishing.wait(5)
                clock[0] = 5000.0
            finally:
                release.set()
            status, result = future.result(timeout=5)
        assert status == 200, result
        assert s.store.get(s.uid).status == "completed"
        assert len(s.files()) == 1


@pytest.mark.parametrize("kind", ["api", "sdk"])
@pytest.mark.parametrize("durable", [False, True])
def test_eviction_waits_for_completion_publication(kind, durable, tmp_path, monkeypatch):
    with surface(kind, tmp_path / "state" if durable else None, monkeypatch) as s:
        monkeypatch.setattr(s.store, "_max", 1)
        publishing, contender, release = threading.Event(), threading.Event(), threading.Event()
        block_publication(s, publishing, contender, release, monkeypatch)

        def evict():
            result = s.store.create(
                purpose="batch", filename="new.jsonl", nbytes=2, mime_type="application/jsonl"
            )
            contender.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(s.complete)
            try:
                assert publishing.wait(5)
                second = pool.submit(evict)
                assert contender.wait(5)
            finally:
                release.set()
            status, result = first.result(timeout=5)
            replacement = second.result(timeout=5)
        assert status == 200, result
        assert s.store.get(s.uid) is None
        assert s.store.get(replacement.upload_id).status == "pending"
        assert len(s.files()) == 1


@pytest.mark.parametrize("kind", ["api", "sdk"])
def test_upload_journal_failure_rolls_back_file_and_preserves_retry(kind, tmp_path, monkeypatch):
    with surface(kind, tmp_path / "state", monkeypatch) as s:
        append = s.store._journal.append

        def fail(payload):
            raise OSError("injected before append")

        with monkeypatch.context() as fault:
            fault.setattr(s.store._journal, "append", fail)
            with pytest.raises(OSError, match="injected before append"):
                s.complete()
        assert s.store.get(s.uid).status == "pending"
        assert s.store.assemble(s.uid, [s.pid]) == b"ab"
        assert not s.files()
        assert s.store._journal.append == append
        assert s.complete()[0] == 200
        assert len(s.files()) == 1


@pytest.mark.parametrize("kind", ["api", "sdk"])
def test_bad_checksum_is_retryable_without_publication(kind, tmp_path, monkeypatch):
    with surface(kind, None, monkeypatch) as s:
        status, result = s.complete(md5="0" * 32)
        assert status == 400
        assert result["error"]["code"] == "checksum_mismatch"
        assert not s.files()
        assert s.store.get(s.uid).status == "pending"
        assert s.complete()[0] == 200


@pytest.mark.parametrize("kind", ["api", "sdk"])
def test_postwrite_failure_does_not_delete_journal_referenced_file(kind, tmp_path, monkeypatch):
    state = tmp_path / "state"
    with surface(kind, state, monkeypatch) as s:
        append = s.store._journal.append

        def fail_after_write(payload):
            append(payload)
            raise OSError("injected after append")

        with monkeypatch.context() as fault:
            fault.setattr(s.store._journal, "append", fail_after_write)
            with pytest.raises(OSError, match="injected after append"):
                s.complete()
        files = s.files()
        assert len(files) == 1
        file_id = files[0].file_id if kind == "api" else files[0]["id"]
        replay_path = tmp_path / "replayed"
        shutil.copytree(state, replay_path)
        recovered = uploads.UploadStore(256, 1024, state_dir=replay_path)
        assert recovered.get(s.uid).status == "completed"
        assert recovered.get(s.uid).file_id == file_id
