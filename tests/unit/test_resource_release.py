"""Fail-safe release of sockets, descriptors, and private scratch trees."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import pytest

from fx1.bench.run_audit import _empty_dir_outcome
from quant_fund.cli.live_cmds import _key
from quant_fund.config.inherit_audit import inherit_audit_bench
from quant_fund.data.concurrent_io import ConnectionPool, pooled_request
from quant_fund.live.gateway_arm import load_hmac_key
from quant_fund.pit.asof_audit import asof_storm_audit
from quant_fund.research.admission import _shadow_corpus, admit_batch
from quant_fund.research.custody import CUSTODY_SCHEMA, verify_custody_bundle
from quant_fund.utils.hashing import hash_bytes


class _DeadConn:
    def __init__(self) -> None:
        self.timeout = 1.0
        self.closed = False

    def request(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("exchange failed")

    def getresponse(self) -> None:
        raise AssertionError("response is unread after a failed request")

    def close(self) -> None:
        self.closed = True


def test_pool_close_drops_idle_connections() -> None:
    pool = ConnectionPool()
    first = pool.checkout("http", "127.0.0.1", 9, timeout=1.0)
    pool.release("http", "127.0.0.1", 9, first)
    pool.close()
    assert pool._idle == {}
    second = pool.checkout("http", "127.0.0.1", 9, timeout=1.0)
    assert second is not first
    pool.discard(second)


def test_pool_context_closes_idle_connections() -> None:
    with ConnectionPool() as pool:
        conn = pool.checkout("http", "127.0.0.1", 9, timeout=1.0)
        pool.release("http", "127.0.0.1", 9, conn)
    assert pool._idle == {}


def test_pooled_request_closes_socket_on_unexpected_error() -> None:
    pool = ConnectionPool()
    conn = _DeadConn()
    pool.checkout = lambda *_args, **_kwargs: conn  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="exchange failed"):
        pooled_request("GET", "http://pool.test/x", timeout=1, max_bytes=32, pool=pool)
    assert conn.closed is True


def test_empty_bench_dir_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "empty"

    def make_empty(**_kwargs: object) -> str:
        target.mkdir(exist_ok=True)
        return str(target)

    monkeypatch.setattr("fx1.bench.run_audit.tempfile.mkdtemp", make_empty)

    def boom(path: str) -> None:
        assert Path(path).is_dir()
        raise FileNotFoundError(path)

    assert _empty_dir_outcome(boom) == "raise:FileNotFoundError"
    assert not target.exists()

    assert _empty_dir_outcome(lambda _path: None) == "no-raise"
    assert not target.exists()


def test_hmac_key_write_failure_closes_and_unlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    real_close = os.close

    def spy_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr(os, "close", spy_close)
    monkeypatch.setattr(os, "write", fail_write)
    path = tmp_path / "hmac.key"
    with pytest.raises(RuntimeError, match="disk"):
        load_hmac_key(path)
    assert closed
    assert not path.exists()


def test_live_desk_key_write_failure_closes_and_unlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    real_close = os.close

    def spy_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr(os, "close", spy_close)
    monkeypatch.setattr(os, "write", fail_write)
    with pytest.raises(RuntimeError, match="disk"):
        _key(tmp_path)
    assert closed
    assert not (tmp_path / "hmac.key").exists()


def test_shadow_corpus_and_admit_batch_remove_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.json").write_text("{}\n", encoding="utf-8")
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*_args: object, **kwargs: object) -> str:
        path = real_mkdtemp(prefix=str(kwargs.get("prefix", "")), dir=tmp_path)
        created.append(path)
        return path

    monkeypatch.setattr("quant_fund.research.admission.tempfile.mkdtemp", spy)
    with _shadow_corpus(corpus) as shadow:
        held = shadow
        assert (shadow / "a.json").is_file()
    assert not held.exists()
    report = admit_batch([], corpus)
    assert report["verdict"] == "admit"
    assert report["n_candidates"] == 0
    assert created
    assert all(not Path(path).exists() for path in created)


def test_inherit_audit_bench_removes_generated_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*_args: object, **kwargs: object) -> str:
        path = real_mkdtemp(prefix=str(kwargs.get("prefix", "")), dir=tmp_path)
        created.append(path)
        return path

    monkeypatch.setattr("quant_fund.config.inherit_audit.tempfile.mkdtemp", spy)
    payload = inherit_audit_bench()
    assert payload["live_pnl_claim"] is False
    assert len(payload["receipt_sha256"]) == 64
    assert created
    assert not Path(created[0]).exists()


def test_asof_storm_removes_private_root_and_keeps_caller_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*_args: object, **kwargs: object) -> str:
        path = real_mkdtemp(prefix=str(kwargs.get("prefix", "")), dir=tmp_path)
        created.append(path)
        return path

    monkeypatch.setattr("quant_fund.pit.asof_audit.tempfile.mkdtemp", spy)
    private = asof_storm_audit(seed=0, n_secs=1, n_events=1, n_restatements=0, n_probes=1)
    assert private["n_restatements"] == 0
    assert created and not Path(created[0]).exists()
    supplied = tmp_path / "kept"
    supplied.mkdir()
    asof_storm_audit(supplied, seed=0, n_secs=1, n_events=1, n_restatements=0, n_probes=1)
    assert supplied.is_dir()


def test_verify_custody_bundle_removes_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*_args: object, **kwargs: object) -> str:
        path = real_mkdtemp(prefix=str(kwargs.get("prefix", "")), dir=tmp_path)
        created.append(path)
        return path

    monkeypatch.setattr("quant_fund.research.custody.tempfile.mkdtemp", spy)
    raw = b'{"members":[]}\n'
    member = b"member-bytes"
    bundle = {
        "kind": CUSTODY_SCHEMA,
        "schema": CUSTODY_SCHEMA,
        "member_sha256": hash_bytes(member),
        "member": "member.json",
        "corpus_dir": "receipts",
        "pattern": "*.json",
        "first_epoch": "epoch.json",
        "chain_head": "epoch.json",
        "hops": [
            {
                "name": "epoch.json",
                "bytes_b64": base64.b64encode(raw).decode("ascii"),
                "sha256": hash_bytes(raw),
            }
        ],
        "inclusion": {
            "merkle_root": "not-a-root",
            "member": "epoch.json",
            "member_sha256": hash_bytes(raw),
        },
        "embedded_files": {},
        "witness_proofs": {},
    }
    result = verify_custody_bundle(bundle, member)
    assert result["layers"]["member"]["ok"] is True
    assert result["ok"] is False
    assert created
    assert not Path(created[0]).exists()
