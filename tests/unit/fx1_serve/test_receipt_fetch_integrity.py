"""Fail-closed receipt download integrity regressions."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from fx1.serve.api import create_app
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "kind": "receipt_fetch_test",
        "schema": "receipt_fetch_test.v1",
        "git_revision": "test",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"ok": True},
    }
    receipt["receipt_sha256"] = hash_bytes(canonical_json_bytes(receipt))
    return receipt


def _client(tmp_path: Path) -> tuple[TestClient, Path, str]:
    receipt = _receipt()
    sha = str(receipt["receipt_sha256"])
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return TestClient(create_app(receipts_dir=tmp_path)), path, sha


def test_valid_receipt_is_served_and_conditionally_cached(tmp_path: Path) -> None:
    client, path, sha = _client(tmp_path)

    fetched = client.get(f"/receipts/{sha}")
    assert fetched.status_code == 200
    assert fetched.content == path.read_bytes()
    assert fetched.headers["etag"] == f'"{sha}"'
    assert fetched.headers["x-fx1-receipt-valid"] == "true"

    conditional = client.get(f"/receipts/{sha}", headers={"If-None-Match": f'"{sha}"'})
    assert conditional.status_code == 304
    assert conditional.headers["x-fx1-receipt-valid"] == "true"


def test_corrupted_receipt_never_uses_immutable_identity(tmp_path: Path) -> None:
    client, path, sha = _client(tmp_path)
    assert client.get(f"/receipts/{sha}").status_code == 200

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["git_revision"] = "evil"
    path.write_text(json.dumps(payload), encoding="utf-8")

    fetched = client.get(f"/receipts/{sha}")
    assert fetched.status_code == 409
    assert fetched.json()["code"] == "receipt_integrity_failed"
    assert "etag" not in fetched.headers
    assert "immutable" not in fetched.headers.get("cache-control", "")

    conditional = client.get(f"/receipts/{sha}", headers={"If-None-Match": f'"{sha}"'})
    assert conditional.status_code == 409
    assert conditional.json()["code"] == "receipt_integrity_failed"
    assert "etag" not in conditional.headers
    assert "immutable" not in conditional.headers.get("cache-control", "")
