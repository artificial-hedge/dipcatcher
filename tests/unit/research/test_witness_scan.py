"""Tests for witness_scan — Rekor key-misuse oracle."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

import quant_fund.research.witness_scan as mod


def _hashedrekord(digest: str) -> str:
    """A Rekor entry body: hashedrekord spec attesting ``digest``."""
    return base64.b64encode(
        json.dumps(
            {
                "kind": "hashedrekord",
                "apiVersion": "0.0.1",
                "spec": {
                    "data": {"hash": {"algorithm": "sha256", "value": digest}},
                    "signature": {"content": "AA==", "publicKey": {"content": "AA=="}},
                },
            }
        ).encode()
    ).decode()


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "quality/checkpoints").mkdir(parents=True)
    (tmp_path / "quality/witness").mkdir(parents=True)
    (tmp_path / "quality/witness_signing.pub").write_bytes(b"PUBKEY")
    (tmp_path / "quality/checkpoint.json").write_bytes(b'{"pins":1}')
    return tmp_path


def _mock_net(monkeypatch: pytest.MonkeyPatch, entries: dict[str, str]) -> None:
    """Mock _http: index returns the UUID list; entries/{uuid} returns each."""
    uuids = list(entries)

    def fake(url: str, timeout: float, data: bytes | None = None) -> bytes:
        if url.endswith("/api/v1/index/retrieve"):
            assert data is not None
            return json.dumps(uuids).encode()
        uuid = url.rsplit("/", 1)[-1]
        body = entries[uuid]
        return json.dumps({uuid: {"body": body}}).encode()

    monkeypatch.setattr(mod, "_http", fake)


def test_scan_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    known = hashlib.sha256((root / "quality/checkpoint.json").read_bytes()).hexdigest()
    _mock_net(monkeypatch, {"uuid-1": _hashedrekord(known)})
    res = mod.scan_witness_log(root)
    assert res["ok"] and res["scanned"] == 1 and res["foreign"] == []


def test_scan_foreign_checkpoint_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    known = hashlib.sha256((root / "quality/checkpoint.json").read_bytes()).hexdigest()
    foreign_digest = hashlib.sha256(b'{"pins":"attacker"}').hexdigest()
    _mock_net(
        monkeypatch,
        {"uuid-1": _hashedrekord(known), "uuid-2": _hashedrekord(foreign_digest)},
    )
    res = mod.scan_witness_log(root)
    assert not res["ok"]
    assert res["foreign"] == [foreign_digest]


def test_scan_unrecognized_and_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    # Garbage body → unrecognized, not an accusation.
    _mock_net(monkeypatch, {"uuid-x": "!!!not-base64!!!"})
    res = mod.scan_witness_log(root)
    assert res["ok"] and res["unrecognized"] == 1

    def _dead(url: str, timeout: float, data: bytes | None = None) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr(mod, "_http", _dead)
    res = mod.scan_witness_log(root)
    # Unreachable index is fail-closed (ok=False) but flagged `online` —
    # the misuse oracle must not wave through a stolen-key fork while blind.
    assert not res["ok"] and res["online"] is False
    assert res["errors"][0].startswith("index_unreachable")


def test_scan_orphan_registry_explains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    orphan = hashlib.sha256(b'{"pins":"superseded"}').hexdigest()
    (root / "quality/witness_orphans.json").write_text(
        json.dumps(
            {
                "schema": "witness_orphans.v1",
                "orphans": {orphan: {"note": "pre-spine checkpoint"}},
            }
        )
    )
    _mock_net(monkeypatch, {"uuid-9": _hashedrekord(orphan)})
    res = mod.scan_witness_log(root)
    assert res["ok"] and res["foreign"] == [] and res["explained"] == [orphan]
    # An unregistered foreign digest still fails even with a registry present.
    extra = hashlib.sha256(b'{"pins":"other"}').hexdigest()
    _mock_net(
        monkeypatch,
        {"uuid-9": _hashedrekord(orphan), "uuid-10": _hashedrekord(extra)},
    )
    res = mod.scan_witness_log(root)
    assert not res["ok"] and res["foreign"] == [extra]


def test_scan_witness_proof_digests_count_as_spine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest pinned inside a committed witness proof is known even when
    its checkpoint record predates the archive era."""
    root = _repo(tmp_path)
    (root / "quality/checkpoint.json").unlink()  # pre-archive era
    old_digest = "ab" * 32
    (root / "quality/witness/proof_1.json").write_text(
        json.dumps({"target": {"sha256": old_digest}})
    )
    _mock_net(monkeypatch, {"uuid-1": _hashedrekord(old_digest)})
    res = mod.scan_witness_log(root)
    assert res["ok"] and res["foreign"] == []
