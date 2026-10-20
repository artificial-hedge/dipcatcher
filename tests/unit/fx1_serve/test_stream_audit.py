"""Tests for fx1.serve.stream_audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fx1.serve.stream_audit import (
    _ENV_KEYS,
    _make_ctx,
    _StubBackend,
    stream_audit,
    stream_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = stream_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = stream_audit_bench()
    failures = {k: v for k, v in blob["claim"]["results"].items() if v is not True}
    assert blob["claim"]["ok"] is True, f"stream_audit probes failed: {failures}"
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = stream_audit_bench()
    b = stream_audit_bench()
    different = {
        key: (a["claim"]["results"].get(key), b["claim"]["results"].get(key))
        for key in a["claim"]["results"].keys() | b["claim"]["results"].keys()
        if a["claim"]["results"].get(key) != b["claim"]["results"].get(key)
    }
    assert a["receipt_sha256"] == b["receipt_sha256"], (
        f"stream_audit receipt changed; differing probes: {different}; "
        f"revisions: {a['git_revision']}, {b['git_revision']}"
    )


def test_empty_audit_result_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.stream_audit.stream_audit", lambda: {})
    blob = stream_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]


def test_keyed_chat_stream_replays_envelope_creation_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fx1.serve.api as api_mod

    original = api_mod.openai_envelope

    def pinned_envelope(*args: Any, **kwargs: Any) -> dict[str, Any]:
        envelope = original(*args, **kwargs)
        envelope["created"] = 1_000_000_000
        return envelope

    monkeypatch.setattr(api_mod, "openai_envelope", pinned_envelope)
    ctx = _make_ctx(tmp_path, _StubBackend())
    request = {
        "model": "fx1",
        "messages": [{"role": "user", "content": "timestamp boundary"}],
        "stream": True,
    }
    headers = {"Idempotency-Key": "creation-time"}
    first = ctx.client.post("/v1/chat/completions", json=request, headers=headers)
    replay = ctx.client.post("/v1/chat/completions", json=request, headers=headers)
    assert first.status_code == replay.status_code == 200
    assert replay.headers.get("x-fx1-idempotent-replay") == "true"
    assert '"created":1000000000' in first.text
    assert first.text == replay.text
