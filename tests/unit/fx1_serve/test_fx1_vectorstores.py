"""KATs for /v1/vector_stores + the server-side file_search lane —
store, wire, SDK twin, stream frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.sdk import Fx1Harness
from fx1.serve.api import create_app
from fx1.serve.openai_compat import OpenAICompatError
from fx1.serve.vectorstores import (
    VS_MAX_ATTRS,
    VS_MAX_FILES,
    VectorStoreError,
    VectorStoreStore,
)


class _B:
    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


def _client(**kw: Any) -> TestClient:
    return TestClient(
        create_app(backend_resolver=lambda *a, **k: _B(), **kw),
        raise_server_exceptions=False,
    )


def _upload(client: TestClient, content: bytes, name: str = "doc.jsonl") -> str:
    resp = client.post(
        "/v1/files",
        files={"file": (name, content, "text/plain")},
        data={"purpose": "batch"},
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["id"])


def _store(tmp_path: Path | None = None, **kw: Any) -> VectorStoreStore:
    return VectorStoreStore(512, state_dir=tmp_path, **kw)


def _files_reader(docs: dict[str, bytes]):
    def read(file_id: str) -> tuple[bytes, str] | None:
        data = docs.get(file_id)
        if data is None:
            return None
        return data, f"{file_id}.txt"

    return read


class TestVectorStoreStore:
    def test_create_get_update_delete(self) -> None:
        s = _store()
        vs = s.create(name="kb", metadata={"team": "fx"})
        assert vs["id"].startswith("vs_") and vs["object"] == "vector_store"
        assert vs["name"] == "kb" and vs["file_counts"]["total"] == 0
        got = s.get(vs["id"])
        assert got["id"] == vs["id"]
        upd = s.update(vs["id"], name="kb2", metadata={"a": "1"})
        assert upd["name"] == "kb2" and upd["metadata"] == {"a": "1"}
        assert s.delete(vs["id"])["deleted"] is True
        with pytest.raises(VectorStoreError) as e:
            s.get(vs["id"])
        assert e.value.status == 404

    def test_list_paging(self) -> None:
        s = _store()
        ids = [s.create(name=f"s{i}")["id"] for i in range(3)]
        page = s.list_stores(limit=2)
        assert [o["id"] for o in page["data"]] == ids[::-1][:2]
        assert page["has_more"] is True
        page2 = s.list_stores(limit=2, after=page["data"][-1]["id"])
        assert [o["id"] for o in page2["data"]] == [ids[0]]
        asc = s.list_stores(order="asc")
        assert [o["id"] for o in asc["data"]] == ids

    def test_attach_and_search(self) -> None:
        s = _store(file_reader=_files_reader({"file-a": b"alpha beta gamma delta"}))
        vs = s.create(name="kb")
        rec = s.attach(vs["id"], "file-a")
        assert rec["status"] == "completed" and rec["indexed_chunks"] >= 1
        hits = s.search([vs["id"]], "beta gamma")
        assert hits and hits[0]["file_id"] == "file-a"
        assert hits[0]["score"] > 0 and "beta" in hits[0]["text"]
        assert s.search([vs["id"]], "zzz notfound") == []

    def test_attach_missing_file_fails(self) -> None:
        s = _store(file_reader=_files_reader({}))
        vs = s.create()
        with pytest.raises(VectorStoreError) as e:
            s.attach(vs["id"], "file-none")
        assert e.value.status == 404 and e.value.code == "file_not_found"

    def test_attach_missing_store_fails(self) -> None:
        s = _store(file_reader=_files_reader({"file-a": b"x"}))
        with pytest.raises(VectorStoreError) as e:
            s.attach("vs_nope", "file-a")
        assert e.value.status == 404 and e.value.code == "vector_store_not_found"

    def test_attach_twice_conflicts(self) -> None:
        s = _store(file_reader=_files_reader({"file-a": b"x"}))
        vs = s.create()
        s.attach(vs["id"], "file-a")
        with pytest.raises(VectorStoreError) as e:
            s.attach(vs["id"], "file-a")
        assert e.value.status == 409

    def test_attach_caps(self) -> None:
        docs = {f"file-{i}": b"w w w" for i in range(VS_MAX_FILES)}
        s2 = _store(file_reader=_files_reader(docs))
        vs2 = s2.create()
        for i in range(VS_MAX_FILES):
            s2.attach(vs2["id"], f"file-{i}")
        with pytest.raises(VectorStoreError) as e:
            s2.attach(vs2["id"], "file-extra")
        assert e.value.status == 409 and e.value.code == "vector_store_full"

    def test_empty_text_lands_failed(self) -> None:
        s = _store(file_reader=_files_reader({"file-e": b"   \n"}))
        vs = s.create()
        rec = s.attach(vs["id"], "file-e")
        assert rec["status"] == "failed" and rec["last_error"]

    def test_static_chunking_validated(self) -> None:
        s = _store(file_reader=_files_reader({"f": b"a b c d e"}))
        vs = s.create()
        with pytest.raises(VectorStoreError) as e:
            s.attach(
                vs["id"],
                "f",
                chunking_strategy={"type": "static", "static": {"max_chunk_size_tokens": 99}},
            )
        assert e.value.status == 400
        with pytest.raises(VectorStoreError) as e:
            s.attach(vs["id"], "f", chunking_strategy={"type": "weird"})
        assert e.value.status == 400

    def test_attributes_cap_and_types(self) -> None:
        s = _store(file_reader=_files_reader({"f": b"a"}))
        vs = s.create()
        with pytest.raises(VectorStoreError) as e:
            s.attach(vs["id"], "f", attributes={f"k{i}": 1 for i in range(VS_MAX_ATTRS + 1)})
        assert e.value.status == 400
        with pytest.raises(VectorStoreError) as e:
            s.attach(vs["id"], "f", attributes={"k": {"nested": 1}})
        assert e.value.status == 400

    def test_filters_narrow_hits(self) -> None:
        docs = {
            "file-a": b"quant risk alpha",
            "file-b": b"quant beta tape",
        }
        s = _store(file_reader=_files_reader(docs))
        vs = s.create()
        s.attach(vs["id"], "file-a", attributes={"kind": "notes"})
        s.attach(vs["id"], "file-b", attributes={"kind": "code"})
        hits = s.search([vs["id"]], "quant", filters={"type": "eq", "key": "kind", "value": "code"})
        assert hits and {h["file_id"] for h in hits} == {"file-b"}
        all_hits = s.search([vs["id"]], "quant")
        assert {h["file_id"] for h in all_hits} == {"file-a", "file-b"}

    def test_score_threshold_and_max(self) -> None:
        s = _store(file_reader=_files_reader({"f": b"a a a"}))
        vs = s.create()
        s.attach(vs["id"], "f")
        with pytest.raises(VectorStoreError):
            s.search([vs["id"]], "a", score_threshold=1.5)
        with pytest.raises(VectorStoreError):
            s.search([vs["id"]], "a", max_results=0)
        hits = s.search([vs["id"]], "a", score_threshold=0.999)
        assert all(h["score"] >= 0.999 for h in hits)

    def test_detach_and_list_filter(self) -> None:
        s = _store(file_reader=_files_reader({"f": b"w"}))
        vs = s.create()
        s.attach(vs["id"], "f")
        files = s.list_files(vs["id"])
        assert [r["id"] for r in files["data"]] == ["f"]
        with pytest.raises(VectorStoreError) as e:
            s.list_files(vs["id"], filter="bogus")
        assert e.value.status == 400
        assert s.detach(vs["id"], "f")["deleted"] is True
        assert s.list_files(vs["id"])["data"] == []
        with pytest.raises(VectorStoreError):
            s.detach(vs["id"], "f")

    def test_file_content_page(self) -> None:
        s = _store(file_reader=_files_reader({"f": b"hello world"}))
        vs = s.create()
        s.attach(vs["id"], "f")
        page = s.file_content(vs["id"], "f")
        assert page["object"] == "vector_store.file_content.page"
        assert page["data"][0]["type"] == "text" and "hello" in page["data"][0]["text"]

    def test_journal_replay(self, tmp_path: Path) -> None:
        s = _store(tmp_path, file_reader=_files_reader({"f": b"a b c"}))
        vs = s.create(name="kb")
        s.attach(vs["id"], "f")
        s2 = _store(tmp_path, file_reader=_files_reader({"f": b"a b c"}))
        assert s2.get(vs["id"])["name"] == "kb"
        assert s2.list_files(vs["id"])["data"][0]["id"] == "f"
        assert s2.search([vs["id"]], "a b")

    def test_store_cap_evicts_lru(self) -> None:
        s = VectorStoreStore(2)
        first = s.create()["id"]
        s.create()
        s.create()
        with pytest.raises(VectorStoreError) as e:
            s.get(first)
        assert e.value.status == 404
        assert len(s.list_stores()["data"]) == 2


def _mk_vs_with_file(client: TestClient, text: bytes = b"quant risk alpha beta") -> tuple[str, str]:
    fid = _upload(client, text)
    vs = client.post("/v1/vector_stores", json={"name": "kb"})
    assert vs.status_code == 200, vs.text
    vs_id = vs.json()["id"]
    att = client.post(f"/v1/vector_stores/{vs_id}/files", json={"file_id": fid})
    assert att.status_code == 200, att.text
    return vs_id, fid


class TestVectorStoreRoutes:
    def test_crud_wire(self) -> None:
        c = _client()
        vs = c.post("/v1/vector_stores", json={"name": "kb"})
        assert vs.status_code == 200 and vs.json()["object"] == "vector_store"
        vid = vs.json()["id"]
        assert c.get(f"/v1/vector_stores/{vid}").json()["id"] == vid
        upd = c.post(f"/v1/vector_stores/{vid}", json={"name": "kb2"})
        assert upd.json()["name"] == "kb2"
        listed = c.get("/v1/vector_stores").json()
        assert any(o["id"] == vid for o in listed["data"])
        assert c.delete(f"/v1/vector_stores/{vid}").json()["deleted"] is True
        assert c.get(f"/v1/vector_stores/{vid}").status_code == 404

    def test_file_routes(self) -> None:
        c = _client()
        vs_id, fid = _mk_vs_with_file(c)
        got = c.get(f"/v1/vector_stores/{vs_id}/files/{fid}")
        assert got.status_code == 200 and got.json()["status"] == "completed"
        files = c.get(f"/v1/vector_stores/{vs_id}/files?filter=completed")
        assert files.status_code == 200 and files.json()["data"][0]["id"] == fid
        assert c.get(f"/v1/vector_stores/{vs_id}/files?filter=bogus").status_code == 400
        content = c.get(f"/v1/vector_stores/{vs_id}/files/{fid}/content")
        assert content.status_code == 200
        assert content.json()["object"] == "vector_store.file_content.page"
        dele = c.delete(f"/v1/vector_stores/{vs_id}/files/{fid}")
        assert dele.status_code == 200 and dele.json()["deleted"] is True
        assert c.get(f"/v1/vector_stores/{vs_id}/files/{fid}").status_code == 404

    def test_attach_unknown_file_404(self) -> None:
        c = _client()
        vid = c.post("/v1/vector_stores", json={}).json()["id"]
        r = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": "file-nope"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "file_not_found"

    def test_create_with_file_ids(self) -> None:
        c = _client()
        fid = _upload(c, b"alpha")
        vs = c.post("/v1/vector_stores", json={"file_ids": [fid]})
        assert vs.status_code == 200 and vs.json()["file_counts"]["total"] == 1
        bad = c.post("/v1/vector_stores", json={"file_ids": ["file-nope"]})
        assert bad.status_code == 404

    def test_respond_file_search(self) -> None:
        c = _client()
        vs_id, fid = _mk_vs_with_file(c, b"the halting problem quant alpha")
        r = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "what is the halting problem",
                "tools": [{"type": "file_search", "vector_store_ids": [vs_id]}],
                "include": ["file_search_call.results"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        fs_items = [o for o in body["output"] if o["type"] == "file_search_call"]
        assert fs_items and body["output"][0]["type"] == "file_search_call"
        fs = fs_items[0]
        assert fs["status"] == "completed" and fs["queries"] == ["what is the halting problem"]
        assert fs["results"] and fs["results"][0]["file_id"] == fid
        assert fs["results"][0]["score"] > 0
        assert body["output"][-1]["type"] == "message"
        items = c.get(f"/v1/responses/{body['id']}/input_items").json()
        assert any(
            it.get("role") == "developer"
            and "file_search results" in json.dumps(it)
            and "halting" in json.dumps(it)
            for it in items["data"]
        )

    def test_respond_file_search_include_gate(self) -> None:
        c = _client()
        vs_id, _ = _mk_vs_with_file(c)
        r = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "quant alpha",
                "tools": [{"type": "file_search", "vector_store_ids": [vs_id]}],
            },
        )
        fs = [o for o in r.json()["output"] if o["type"] == "file_search_call"][0]
        assert fs["results"] is None
        # context still injected even without include
        items = c.get(f"/v1/responses/{r.json()['id']}/input_items").json()
        assert any(it.get("role") == "developer" for it in items["data"])

    def test_respond_file_search_unknown_store_400(self) -> None:
        c = _client()
        r = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "q",
                "tools": [{"type": "file_search", "vector_store_ids": ["vs_nope"]}],
            },
        )
        assert r.status_code == 404 or r.status_code == 400

    def test_tool_choice_file_search(self) -> None:
        c = _client()
        vs_id, _ = _mk_vs_with_file(c)
        r = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "quant",
                "tools": [{"type": "file_search", "vector_store_ids": [vs_id]}],
                "tool_choice": {"type": "file_search"},
            },
        )
        assert r.status_code == 200, r.text
        bad = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "quant",
                "tool_choice": {"type": "file_search"},
            },
        )
        assert bad.status_code in (400, 422)

    def test_respond_stream_file_search_frames(self) -> None:
        c = _client()
        vs_id, _ = _mk_vs_with_file(c)
        r = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "quant alpha",
                "stream": True,
                "tools": [{"type": "file_search", "vector_store_ids": [vs_id]}],
                "include": ["file_search_call.results"],
            },
        )
        assert r.status_code == 200, r.text
        types = [
            json.loads(line.split("data: ", 1)[1])["type"]
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        assert "response.output_item.added" in types
        assert "response.file_search_call.in_progress" in types
        assert "response.file_search_call.searching" in types
        assert "response.file_search_call.completed" in types
        assert types[-1] == "response.completed"
        fs_idx = types.index("response.file_search_call.completed")
        msg_idx = next(
            i for i, t in enumerate(types) if t == "response.output_item.added" and i > fs_idx
        )
        assert msg_idx > fs_idx

    def test_file_search_input_item_roundtrip(self) -> None:
        """A file_search_call input item folds into context, not a 501."""
        c = _client()
        r = c.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": [
                    {
                        "type": "file_search_call",
                        "queries": ["q"],
                        "results": [{"file_id": "f1", "text": "alpha context", "score": 0.9}],
                    },
                    {"type": "message", "role": "user", "content": "go"},
                ],
            },
        )
        assert r.status_code == 200, r.text


class TestSdkVectorStores:
    def test_sdk_crud_and_search(self) -> None:
        h = Fx1Harness(backend_resolver=lambda *a, **k: _B())
        vs = h.vector_store_create(name="kb")
        assert vs["id"].startswith("vs_")
        up = h.openai_file_create(content=b"quant risk alpha", filename="d.jsonl")
        rec = h.vector_store_file_create(vs["id"], up["id"])
        assert rec["status"] == "completed"
        listed = h.vector_store_file_list(vs["id"])
        assert listed["data"][0]["id"] == up["id"]
        page = h.vector_store_file_content(vs["id"], up["id"])
        assert "quant" in page["data"][0]["text"]
        assert h.vector_store_file_delete(vs["id"], up["id"])["deleted"] is True
        assert h.vector_store_delete(vs["id"])["deleted"] is True
        with pytest.raises(VectorStoreError):
            h.vector_store_get(vs["id"])

    def test_sdk_respond_file_search(self) -> None:
        h = Fx1Harness(backend_resolver=lambda *a, **k: _B())
        up = h.openai_file_create(content=b"the halting problem quant", filename="d.jsonl")
        vs = h.vector_store_create()
        h.vector_store_file_create(vs["id"], up["id"])
        out, _cid = h.openai_response(
            {
                "model": "fx1",
                "input": "what is the halting problem",
                "tools": [{"type": "file_search", "vector_store_ids": [vs["id"]]}],
                "include": ["file_search_call.results"],
            }
        )
        fs = [o for o in out["output"] if o["type"] == "file_search_call"]
        assert fs and fs[0]["results"] and fs[0]["results"][0]["file_id"] == up["id"]

    def test_sdk_state_dir_replays(self, tmp_path: Path) -> None:
        h = Fx1Harness(state_dir=tmp_path, backend_resolver=lambda *a, **k: _B())
        up = h.openai_file_create(content=b"a b", filename="d.jsonl")
        vs = h.vector_store_create(name="kb")
        h.vector_store_file_create(vs["id"], up["id"])
        h2 = Fx1Harness(state_dir=tmp_path, backend_resolver=lambda *a, **k: _B())
        assert h2.vector_store_get(vs["id"])["name"] == "kb"
        assert h2.vector_store_file_list(vs["id"])["data"][0]["id"] == up["id"]

    def test_vs_search_route(self) -> None:
        client = _client()
        fid = _upload(client, b"epsilon transitions drive drift\n")
        vs = client.post("/v1/vector_stores", json={"name": "kb"}).json()
        client.post(f"/v1/vector_stores/{vs['id']}/files", json={"file_id": fid})
        out = client.post(f"/v1/vector_stores/{vs['id']}/search", json={"query": "epsilon"})
        assert out.status_code == 200, out.text
        page = out.json()
        assert page["object"] == "vector_store.search_results.page"
        assert page["search_query"] == "epsilon"
        assert page["data"][0]["file_id"] == fid
        assert page["data"][0]["content"][0]["type"] == "text"
        assert "epsilon" in page["data"][0]["content"][0]["text"]
        assert page["has_more"] is False and page["next_page"] is None

    def test_vs_search_query_list_and_score_threshold(self) -> None:
        client = _client()
        fid = _upload(client, b"epsilon transitions drive drift\n")
        vs = client.post("/v1/vector_stores", json={"name": "kb"}).json()
        client.post(f"/v1/vector_stores/{vs['id']}/files", json={"file_id": fid})
        out = client.post(
            f"/v1/vector_stores/{vs['id']}/search",
            json={"query": ["epsilon", "drift"], "max_num_results": 5},
        )
        assert out.status_code == 200
        assert out.json()["search_query"] == "epsilon drift"
        # threshold above any lexical hit fails closed to empty, not an error
        out2 = client.post(
            f"/v1/vector_stores/{vs['id']}/search",
            json={
                "query": "epsilon",
                "ranking_options": {"score_threshold": 0.999},
            },
        )
        assert out2.status_code == 200
        assert out2.json()["data"] == []

    def test_vs_search_fail_closed(self) -> None:
        client = _client()
        vs = client.post("/v1/vector_stores", json={"name": "kb"}).json()
        # ghost store → 404 vector_store_not_found
        ghost = client.post("/v1/vector_stores/vs_ghost/search", json={"query": "x"})
        assert ghost.status_code == 404
        assert ghost.json()["error"]["code"] == "vector_store_not_found"
        # empty query → 400
        assert (
            client.post(f"/v1/vector_stores/{vs['id']}/search", json={"query": "  "}).status_code
            == 400
        )
        # rewrite_query refused
        assert (
            client.post(
                f"/v1/vector_stores/{vs['id']}/search",
                json={"query": "x", "rewrite_query": True},
            ).status_code
            == 422
        )
        # non-auto ranker refused; out-of-range threshold refused
        assert (
            client.post(
                f"/v1/vector_stores/{vs['id']}/search",
                json={"query": "x", "ranking_options": {"ranker": "bm25"}},
            ).status_code
            == 422
        )
        assert client.post(
            f"/v1/vector_stores/{vs['id']}/search",
            json={"query": "x", "ranking_options": {"score_threshold": 1.5}},
        ).status_code in (400, 422)
        # max_num_results over the bound → 422
        assert (
            client.post(
                f"/v1/vector_stores/{vs['id']}/search",
                json={"query": "x", "max_num_results": 51},
            ).status_code
            == 422
        )

    def test_sdk_vector_store_search_parity(self) -> None:
        h = Fx1Harness(backend_resolver=lambda *a, **k: _B())
        up = h.openai_file_create(content=b"epsilon transitions drive drift", filename="d.jsonl")
        vs = h.vector_store_create(name="kb")
        h.vector_store_file_create(vs["id"], up["id"], attributes={"team": "q"})
        page = h.vector_store_search(vs["id"], "epsilon")
        assert page["object"] == "vector_store.search_results.page"
        assert page["data"][0]["file_id"] == up["id"]
        assert page["data"][0]["attributes"] == {"team": "q"}
        filtered = h.vector_store_search(
            vs["id"], "epsilon", filters={"type": "eq", "key": "team", "value": "q"}
        )
        assert filtered["data"][0]["file_id"] == up["id"]
        filtered_out = h.vector_store_search(
            vs["id"], "epsilon", filters={"type": "eq", "key": "team", "value": "other"}
        )
        assert filtered_out["data"] == []
        with pytest.raises(OpenAICompatError) as bad_filters:
            h.vector_store_search(vs["id"], "epsilon", filters={"wrong": "shape"})
        assert bad_filters.value.code == "invalid_filters"
        with pytest.raises(ValueError, match="rewrite_query"):
            h.vector_store_search(vs["id"], "x", rewrite_query=True)
        with pytest.raises(OpenAICompatError) as ghost:
            h.vector_store_search("vs_ghost", "x")
        assert ghost.value.status == 404

    def test_file_batch_create_counts_failures_not_abort(self) -> None:
        s = _store(file_reader=_files_reader({"file-a": b"alpha beta", "file-b": b"gamma"}))
        vs = s.create(name="kb")
        batch = s.file_batch_create(vs["id"], ["file-a", "file-none", "file-b"])
        assert batch["object"] == "vector_store.files_batch"
        assert batch["id"].startswith("vsfb_")
        assert batch["status"] == "completed"
        c = batch["file_counts"]
        assert (c["completed"], c["failed"], c["total"]) == (2, 1, 3)
        # the failed member carries the per-file refusal as last_error
        files = s.file_batch_files(vs["id"], batch["id"], filter="failed")
        assert files["data"][0]["id"] == "file-none"
        assert files["data"][0]["status"] == "failed"
        assert files["data"][0]["last_error"]["code"] == "file_not_found"
        ok = s.file_batch_files(vs["id"], batch["id"], filter="completed")
        assert {r["id"] for r in ok["data"]} == {"file-a", "file-b"}
        assert ok["data"][0]["indexed_chunks"] >= 1

    def test_file_batch_all_failed_and_bad_input(self) -> None:
        s = _store(file_reader=_files_reader({}))
        vs = s.create()
        batch = s.file_batch_create(vs["id"], ["file-none", "file-nada"])
        assert batch["status"] == "failed"
        assert batch["file_counts"]["completed"] == 0
        with pytest.raises(VectorStoreError) as e:
            s.file_batch_create(vs["id"], [])
        assert e.value.status == 400
        with pytest.raises(VectorStoreError) as e2:
            s.file_batch_create(vs["id"], ["x"] * 501)
        assert e2.value.status == 400
        with pytest.raises(VectorStoreError) as ghost:
            s.file_batch_create("vs_ghost", ["file-a"])
        assert ghost.value.code == "vector_store_not_found"

    def test_file_batch_get_cancel_files(self) -> None:
        s = _store(file_reader=_files_reader({"file-a": b"alpha"}))
        vs = s.create()
        batch = s.file_batch_create(vs["id"], ["file-a"])
        got = s.file_batch_get(vs["id"], batch["id"])
        assert got["id"] == batch["id"] and got["status"] == "completed"
        # terminal at create → cancel is an honest 409, not a fake window
        with pytest.raises(VectorStoreError) as e:
            s.file_batch_cancel(vs["id"], batch["id"])
        assert e.value.status == 409 and e.value.code == "file_batch_terminal"
        with pytest.raises(VectorStoreError) as e2:
            s.file_batch_get(vs["id"], "vsfb_nope")
        assert e2.value.code == "file_batch_not_found"
        with pytest.raises(VectorStoreError) as e3:
            s.file_batch_files(vs["id"], batch["id"], filter="bogus")
        assert e3.value.code == "invalid_filters"
        # dropping the store drops its batches too
        s.delete(vs["id"])
        with pytest.raises(VectorStoreError) as e4:
            s.file_batch_get(vs["id"], batch["id"])
        assert e4.value.code == "vector_store_not_found"

    def test_file_batch_replays_from_journal(self, tmp_path: Path) -> None:
        docs = {"file-a": b"alpha"}
        s = _store(tmp_path, file_reader=_files_reader(docs))
        vs = s.create(name="kb")
        batch = s.file_batch_create(vs["id"], ["file-a"])
        s2 = _store(tmp_path, file_reader=_files_reader(docs))
        got = s2.file_batch_get(vs["id"], batch["id"])
        assert got["file_counts"]["completed"] == 1

    def test_file_batch_routes(self) -> None:
        client = _client()
        fid = _upload(client, b"epsilon transitions\n")
        vs = client.post("/v1/vector_stores", json={"name": "kb"}).json()
        created = client.post(
            f"/v1/vector_stores/{vs['id']}/file_batches",
            json={"file_ids": [fid, "file-ghost"]},
        )
        assert created.status_code == 200, created.text
        batch = created.json()
        assert batch["object"] == "vector_store.files_batch"
        assert batch["file_counts"] == {
            "in_progress": 0,
            "completed": 1,
            "failed": 1,
            "cancelled": 0,
            "total": 2,
        }
        got = client.get(f"/v1/vector_stores/{vs['id']}/file_batches/{batch['id']}")
        assert got.status_code == 200 and got.json()["id"] == batch["id"]
        files = client.get(
            f"/v1/vector_stores/{vs['id']}/file_batches/{batch['id']}/files",
            params={"filter": "failed"},
        )
        assert files.status_code == 200
        assert [r["id"] for r in files.json()["data"]] == ["file-ghost"]
        cancel = client.post(f"/v1/vector_stores/{vs['id']}/file_batches/{batch['id']}/cancel")
        assert cancel.status_code == 409
        assert cancel.json()["error"]["code"] == "file_batch_terminal"
        ghost = client.get(f"/v1/vector_stores/{vs['id']}/file_batches/vsfb_x")
        assert ghost.status_code == 404
        bad = client.post(f"/v1/vector_stores/{vs['id']}/file_batches", json={"file_ids": []})
        assert bad.status_code == 422

    def test_sdk_file_batch_parity(self) -> None:
        h = Fx1Harness(backend_resolver=lambda *a, **k: _B())
        up = h.openai_file_create(content=b"alpha beta", filename="a.jsonl")
        vs = h.vector_store_create(name="kb")
        batch = h.vector_store_file_batch_create(vs["id"], [up["id"], "file-ghost"])
        assert batch["object"] == "vector_store.files_batch"
        assert batch["file_counts"]["completed"] == 1
        assert batch["file_counts"]["failed"] == 1
        got = h.vector_store_file_batch_get(vs["id"], batch["id"])
        assert got["id"] == batch["id"]
        rows = h.vector_store_file_batch_files(vs["id"], batch["id"], filter="failed")
        assert [r["id"] for r in rows["data"]] == ["file-ghost"]
        with pytest.raises(OpenAICompatError) as e:
            h.vector_store_file_batch_cancel(vs["id"], batch["id"])
        assert e.value.status == 409 and e.value.code == "file_batch_terminal"
        with pytest.raises(ValueError):
            h.vector_store_file_batch_create(vs["id"], [])
