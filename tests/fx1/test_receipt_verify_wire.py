"""Byte-level pin for the receipt verify routes' verdict mapping.

``verify_receipt`` and ``verify_receipts_batch`` build the same field set.
These cases lock that JSON, including the fail-closed exception branch,
before the two copies are folded into one helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fx1.serve.api as api_mod


def _wire(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")


def _ok(
    *,
    valid: bool = True,
    path: str = "<api>",
    schema: str = "receipt.v2",
    kind: object = "online_fdr.v1",
    verdict: object = "ok",
    digest_convention: str | None = "canonical",
    errors: object = (),
    warnings: object = (),
) -> dict[str, Any]:
    return {
        "valid": valid,
        "path": path,
        "schema": schema,
        "kind": kind,
        "verdict": verdict,
        "digest_convention": digest_convention,
        "errors": errors,
        "warnings": warnings,
    }


def _fields(
    *,
    valid: bool,
    path: str,
    schema_tag: str,
    kind: str | None,
    verdict: str | None,
    digest_convention: str | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "valid": valid,
        "path": path,
        "schema_tag": schema_tag,
        "kind": kind,
        "verdict": verdict,
        "digest_convention": digest_convention,
        "errors": errors,
        "warnings": warnings,
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(api_mod.create_app(receipts_dir=tmp_path / "receipts", public_release=False))


def test_verify_routes_map_verdicts_byte_identically(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    script: list[dict[str, Any] | BaseException] = [
        _ok(warnings=("leak-canary",), errors=["kept"]),
        _ok(
            valid=False,
            path="from-verifier",
            schema="legacy",
            kind=1,
            verdict=True,
            digest_convention=None,
            errors=(),
            warnings=(),
        ),
        _ok(kind="", verdict="0", digest_convention="strict"),
        RuntimeError('say "no" café'),
        ValueError(),
    ]
    seen: list[tuple[object, object]] = []

    def fake(payload: object, path: Path | str = Path("<memory>")) -> dict[str, Any]:
        seen.append((payload, path))
        action = script.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    monkeypatch.setattr(api_mod, "verify_receipt_payload", fake)

    success = client.post("/receipts/verify", json={"receipt": {"id": "a"}})
    expected_success = _fields(
        valid=True,
        path="<api>",
        schema_tag="receipt.v2",
        kind="online_fdr.v1",
        verdict="ok",
        digest_convention="canonical",
        errors=["kept"],
        warnings=["leak-canary"],
    )
    assert success.status_code == 200
    assert success.json() == expected_success
    assert success.content == _wire(expected_success)

    coerced = client.post("/receipts/verify", json={"receipt": {"id": "b"}})
    expected_coerced = _fields(
        valid=False,
        path="from-verifier",
        schema_tag="legacy",
        kind=None,
        verdict=None,
        digest_convention=None,
        errors=[],
        warnings=[],
    )
    assert coerced.status_code == 200
    assert coerced.json() == expected_coerced
    assert coerced.content == _wire(expected_coerced)

    blanks = client.post("/receipts/verify", json={"receipt": {"id": "c"}})
    expected_blanks = _fields(
        valid=True,
        path="<api>",
        schema_tag="receipt.v2",
        kind="",
        verdict="0",
        digest_convention="strict",
        errors=[],
        warnings=[],
    )
    assert blanks.status_code == 200
    assert blanks.content == _wire(expected_blanks)

    quoted = client.post("/receipts/verify", json={"receipt": {"id": "d"}})
    expected_quoted = _fields(
        valid=False,
        path="<api>",
        schema_tag="",
        kind=None,
        verdict=None,
        digest_convention=None,
        errors=['say "no" café'],
        warnings=[],
    )
    assert quoted.status_code == 200
    assert quoted.json() == expected_quoted
    assert quoted.content == _wire(expected_quoted)

    empty_exc = client.post("/receipts/verify", json={"receipt": {"id": "e"}})
    expected_empty = _fields(
        valid=False,
        path="<api>",
        schema_tag="",
        kind=None,
        verdict=None,
        digest_convention=None,
        errors=[""],
        warnings=[],
    )
    assert empty_exc.status_code == 200
    assert empty_exc.content == _wire(expected_empty)

    def _item(index: int, fields: dict[str, Any]) -> dict[str, Any]:
        # Parent fields, then index — the subclass field order on the wire.
        return {**fields, "index": index}

    script.extend(
        [
            _ok(warnings=("w",)),
            RuntimeError("item-local"),
            _ok(valid=False, kind={"nested": True}, verdict=None, errors=["bad"], warnings=[]),
        ]
    )
    batch = client.post(
        "/receipts/verify/batch",
        json={"receipts": [{"id": "a"}, {"id": "b"}, {"id": "boom"}]},
    )
    expected_batch = {
        "verified": 1,
        "failed": 2,
        "results": [
            _item(0, {**expected_success, "warnings": ["w"], "errors": []}),
            _item(
                1,
                _fields(
                    valid=False,
                    path="<api>",
                    schema_tag="",
                    kind=None,
                    verdict=None,
                    digest_convention=None,
                    errors=["item-local"],
                    warnings=[],
                ),
            ),
            _item(
                2,
                _fields(
                    valid=False,
                    path="<api>",
                    schema_tag="receipt.v2",
                    kind=None,
                    verdict=None,
                    digest_convention="canonical",
                    errors=["bad"],
                    warnings=[],
                ),
            ),
        ],
    }
    assert batch.status_code == 200
    assert batch.json() == expected_batch
    assert batch.content == _wire(expected_batch)
    assert script == []
    assert [path for _payload, path in seen] == [Path("<api>")] * len(seen)
    assert [payload for payload, _path in seen] == [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
        {"id": "d"},
        {"id": "e"},
        {"id": "a"},
        {"id": "b"},
        {"id": "boom"},
    ]
