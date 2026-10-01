"""Tests for registry/artifact_store.py — content-addressed + hash-chained."""

from __future__ import annotations

import pytest

from quant_fund.registry.artifact_store import ArtifactStore, ArtifactStoreError


def test_store_fetch_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    sha = store.store(b"hello evidence", kind="receipt")
    assert store.fetch(sha) == b"hello evidence"
    assert len(sha) == 64


def test_store_dedups_objects_but_appends_index(tmp_path):
    store = ArtifactStore(tmp_path)
    a = store.store(b"same bytes", kind="receipt")
    b = store.store(b"same bytes", kind="receipt")
    assert a == b
    listing = store.list()
    assert len(listing) == 2  # two history records, one object
    assert all(r["sha256"] == a for r in listing)


def test_store_rejects_non_bytes(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(TypeError):
        store.store("not bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        store.store(b"x", kind="")


def test_fetch_missing_and_drift(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError, match="not present"):
        store.fetch("f" * 64)
    sha = store.store(b"original", kind="blob")
    obj = tmp_path / "objects" / sha[:2] / sha[2:]
    obj.write_bytes(b"tampered")
    with pytest.raises(ArtifactStoreError, match="drifted"):
        store.fetch(sha)


def test_verify_clean(tmp_path):
    store = ArtifactStore(tmp_path)
    store.store(b"a", kind="x")
    store.store(b"b", kind="y")
    rep = store.verify()
    assert rep["ok"] and rep["n_objects"] == 2 and rep["n_index_records"] == 2


def test_verify_catches_object_tamper(tmp_path):
    store = ArtifactStore(tmp_path)
    sha = store.store(b"payloadd", kind="x")
    obj = tmp_path / "objects" / sha[:2] / sha[2:]
    obj.write_bytes(b"payloade")
    rep = store.verify()
    assert not rep["ok"]
    assert any("digest drift" in e for e in rep["errors"])


def test_verify_catches_index_edit(tmp_path):
    store = ArtifactStore(tmp_path)
    store.store(b"one", kind="x")
    store.store(b"two", kind="x")
    index = tmp_path / "index.jsonl"
    raw = index.read_bytes()
    # rewrite second record's bytes field 3 -> 9 (same length, chain-untouched)
    lines = raw.split(b"\n")
    assert b'"bytes":3' in lines[1]
    lines[1] = lines[1].replace(b'"bytes":3', b'"bytes":9')
    index.write_bytes(b"\n".join(lines))
    rep = store.verify()
    assert not rep["ok"]
    assert any("size mismatch" in e or "chain break" in e for e in rep["errors"])


def test_verify_catches_deleted_index_line(tmp_path):
    store = ArtifactStore(tmp_path)
    store.store(b"one", kind="x")
    store.store(b"two", kind="x")
    index = tmp_path / "index.jsonl"
    lines = index.read_bytes().split(b"\n")
    index.write_bytes(b"\n".join(lines[:1] + lines[2:]))
    rep = store.verify()
    assert not rep["ok"]


def test_verify_flags_orphan_object(tmp_path):
    store = ArtifactStore(tmp_path)
    store.store(b"tracked", kind="x")
    # hand-write an unindexed object
    import hashlib

    junk = b"unindexed"
    sha = hashlib.sha256(junk).hexdigest()
    obj = tmp_path / "objects" / sha[:2] / sha[2:]
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(junk)
    rep = store.verify()
    assert not rep["ok"]
    assert rep["n_orphans"] == 1
