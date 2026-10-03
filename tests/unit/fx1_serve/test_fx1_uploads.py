"""KATs for the /v1/uploads chunked-file surface — store, API, SDK twin."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.sdk import Fx1Harness
from fx1.serve.api import create_app
from fx1.serve.uploads import (
    UPLOAD_MAX_PARTS,
    UploadStore,
    UploadStoreError,
)


class _B:
    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


def _client(**kw: Any) -> TestClient:
    return TestClient(
        create_app(backend_resolver=lambda *a, **k: _B(), **kw),
        raise_server_exceptions=False,
    )


_BODY = b'{"l":1}\n{"l":2}\n{"l":3}\n'


def _create(client: TestClient, nbytes: int = len(_BODY)) -> str:
    resp = client.post(
        "/v1/uploads",
        json={
            "purpose": "batch",
            "filename": "big.jsonl",
            "bytes": nbytes,
            "mime_type": "application/jsonl",
        },
    )
    assert resp.status_code == 200
    return resp.json()["id"]


def _part(client: TestClient, uid: str, data: bytes) -> dict[str, Any]:
    return client.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", data)})


class TestUploadLifecycle:
    def test_wire_lifecycle_caller_order(self) -> None:
        c = _client()
        uid = _create(c)
        p1 = _part(c, uid, _BODY[:8]).json()["id"]
        p2 = _part(c, uid, _BODY[8:18]).json()["id"]
        p3 = _part(c, uid, _BODY[18:]).json()["id"]
        done = c.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [p3, p1, p2]})
        assert done.status_code == 200
        body = done.json()
        assert body["status"] == "completed" and body["file"]["id"].startswith("file-")
        content = c.get(f"/v1/files/{body['file']['id']}/content")
        assert content.content == _BODY[18:] + _BODY[:8] + _BODY[8:18]

    def test_create_envelope_shape(self) -> None:
        c = _client()
        uid = _create(c)
        resp = c.post(
            "/v1/uploads",
            json={
                "purpose": "fine-tune",
                "filename": "t.jsonl",
                "bytes": 4,
                "mime_type": "t",
            },
        ).json()
        assert (
            resp["object"] == "upload"
            and resp["status"] == "pending"
            and resp["file"] is None
            and resp["expires_at"] > resp["created_at"]
            and resp["id"].startswith("upload_")
        )
        assert c.post(f"/v1/uploads/{uid}/cancel").json()["status"] == "cancelled"


class TestUploadFailClosed:
    def test_purpose_and_extension(self) -> None:
        c = _client()
        assert (
            c.post(
                "/v1/uploads",
                json={
                    "purpose": "user_data",
                    "filename": "x.jsonl",
                    "bytes": 1,
                    "mime_type": "t",
                },
            ).status_code
            == 400
        )
        assert (
            c.post(
                "/v1/uploads",
                json={
                    "purpose": "batch",
                    "filename": "x.txt",
                    "bytes": 1,
                    "mime_type": "t",
                },
            ).status_code
            == 400
        )

    def test_cumulative_over_declared(self) -> None:
        c = _client()
        uid = _create(c, nbytes=4)
        r = _part(c, uid, b"abcde")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "part_exceeds_declared_bytes"

    def test_under_declared_complete(self) -> None:
        c = _client()
        uid = _create(c, nbytes=64)
        pid = _part(c, uid, b"short").json()["id"]
        r = c.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "upload_incomplete"

    def test_missing_part(self) -> None:
        c = _client()
        uid = _create(c)
        _part(c, uid, b"ab")
        r = c.post(f"/v1/uploads/{uid}/complete", json={"part_ids": ["part_nope"]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "part_not_found"

    def test_md5_mismatch_mints_no_file(self) -> None:
        c = _client()
        uid = _create(c, nbytes=2)
        pid = _part(c, uid, b"ab").json()["id"]
        r = c.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid], "md5": "0" * 32})
        assert r.status_code == 400 and r.json()["error"]["code"] == "checksum_mismatch"
        # pre-terminal: still accepts parts; correct md5 then completes
        ok = c.post(
            f"/v1/uploads/{uid}/complete",
            json={
                "part_ids": [pid],
                "md5": hashlib.md5(b"ab", usedforsecurity=False).hexdigest(),
            },
        )
        assert ok.status_code == 200 and ok.json()["file"]["bytes"] == 2

    def test_terminal_records_refuse_parts(self) -> None:
        c = _client()
        uid = _create(c)
        c.post(f"/v1/uploads/{uid}/cancel")
        r = _part(c, uid, b"ab")
        assert r.status_code == 409 and r.json()["error"]["code"] == "upload_terminal"
        assert c.post(f"/v1/uploads/{uid}/cancel").json()["status"] == "cancelled"

    def test_missing_upload_404(self) -> None:
        c = _client()
        assert _part(c, "upload_nope", b"x").status_code == 404
        assert c.post("/v1/uploads/upload_nope/cancel").status_code == 404


class TestUploadStoreDurability:
    def test_restart_recovers_pending_and_parts(self, tmp_path: Path) -> None:
        c1 = _client(state_dir=str(tmp_path))
        uid = _create(c1, nbytes=4)
        pid = _part(c1, uid, b"ab").json()["id"]
        c2 = _client(state_dir=str(tmp_path))
        p2 = _part(c2, uid, b"cd").json()["id"]
        done = c2.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid, p2]})
        assert done.status_code == 200
        assert c2.get(f"/v1/files/{done.json()['file']['id']}/content").content == b"abcd"

    def test_cancel_survives_restart(self, tmp_path: Path) -> None:
        c1 = _client(state_dir=str(tmp_path))
        uid = _create(c1)
        c1.post(f"/v1/uploads/{uid}/cancel")
        c2 = _client(state_dir=str(tmp_path))
        r = _part(c2, uid, b"ab")
        assert r.status_code == 409


class TestUploadStoreBounds:
    def test_too_many_parts(self) -> None:
        store = UploadStore(max_entries=4, max_bytes=1 << 20)
        meta = store.create(
            purpose="batch", filename="x.jsonl", nbytes=UPLOAD_MAX_PARTS, mime_type="t"
        )
        for _ in range(UPLOAD_MAX_PARTS):
            store.add_part(meta.upload_id, b"x")
        with pytest.raises(UploadStoreError) as exc:
            store.add_part(meta.upload_id, b"x")
        assert exc.value.code == "too_many_parts"

    def test_declared_bounds(self) -> None:
        store = UploadStore(max_entries=4, max_bytes=8)
        with pytest.raises(UploadStoreError):
            store.create(purpose="batch", filename="x.jsonl", nbytes=0, mime_type="t")
        with pytest.raises(UploadStoreError):
            store.create(purpose="batch", filename="x.jsonl", nbytes=9, mime_type="t")

    def test_lru_evict_drops_pending(self) -> None:
        store = UploadStore(max_entries=1, max_bytes=64)
        m1 = store.create(purpose="batch", filename="a.jsonl", nbytes=2, mime_type="t")
        store.create(purpose="batch", filename="b.jsonl", nbytes=2, mime_type="t")
        with pytest.raises(UploadStoreError, match="not found"):
            store.add_part(m1.upload_id, b"ab")


class TestUploadSdkTwin:
    def test_in_process_lifecycle(self) -> None:
        sdk = Fx1Harness(backend_resolver=lambda *a, **k: _B())
        up = sdk.upload_create(bytes=len(_BODY))
        p1 = sdk.upload_part(up["id"], _BODY[:8])
        p2 = sdk.upload_part(up["id"], _BODY[8:])
        done = sdk.upload_complete(up["id"], [p1["id"], p2["id"]])
        assert done["status"] == "completed" and done["file"]["bytes"] == len(_BODY)
        assert sdk.file_content(done["file"]["id"]) == _BODY
        assert "_content" not in sdk.file_card(done["file"]["id"])

    def test_in_process_fail_closed(self) -> None:
        sdk = Fx1Harness(backend_resolver=lambda *a, **k: _B())
        with pytest.raises(UploadStoreError):
            sdk.upload_create(bytes=0)
        up = sdk.upload_create(bytes=4)
        with pytest.raises(UploadStoreError, match="not found"):
            sdk.upload_complete(up["id"], ["part_ghost"])
        assert sdk.upload_cancel(up["id"])["status"] == "cancelled"
        with pytest.raises(UploadStoreError, match="cancelled"):
            sdk.upload_part(up["id"], b"ab")
