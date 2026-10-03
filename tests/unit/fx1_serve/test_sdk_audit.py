"""Tests for fx1.sdk + sdk_audit."""

from __future__ import annotations

import os

from fx1.sdk import Fx1Harness
from fx1.sdk_audit import sdk_audit, sdk_audit_bench
from fx1.serve.backends import SamplingParams
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = sdk_audit()
    assert len(results) >= 20
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = sdk_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = sdk_audit_bench()
    b = sdk_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_complete_closes_backend() -> None:
    class _B:
        _model = "m0"
        closed = 0

        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            return "clean"

        def close(self) -> None:
            self.closed += 1

    backend = _B()
    sdk = Fx1Harness(backend_resolver=lambda *a, **k: backend)
    out = sdk.complete([{"role": "user", "content": "hi"}], backend="byok")
    assert out.content == "clean"
    assert backend.closed == 1


def test_health_never_leaks_env_values() -> None:
    os.environ["MOONSHOT_API_KEY"] = "sentinel-secret"
    try:
        h = Fx1Harness().health()
    finally:
        os.environ.pop("MOONSHOT_API_KEY")
    assert h.backends["hosted_k3"] is True
    assert all(isinstance(v, bool) for v in h.backends.values())
