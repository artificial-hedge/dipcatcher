"""Tests for fx1.sdk + sdk_audit."""

from __future__ import annotations

import inspect
import os
import sys
import threading

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


def test_response_headers_are_published_atomically() -> None:
    sdk = Fx1Harness()
    vulnerable_line = next(
        (
            Fx1Harness._record_call.__code__.co_firstlineno + offset
            for offset, line in enumerate(inspect.getsourcelines(Fx1Harness._record_call)[0])
            if 'self._last_response_headers["x-fx1-completion-id"]' in line
        ),
        None,
    )
    parked = threading.Event()
    release = threading.Event()

    def trace(frame: object, event: str, arg: object) -> object:
        del arg
        if (
            vulnerable_line is not None
            and event == "line"
            and getattr(frame, "f_code", None) == Fx1Harness._record_call.__code__
            and getattr(frame, "f_lineno", None) == vulnerable_line
        ):
            parked.set()
            release.wait(2)
        return trace

    def record() -> None:
        sys.settrace(trace)
        try:
            sdk._record_call(
                "byok",
                "model",
                True,
                1.0,
                None,
                None,
                None,
                "a" * 64,
                "b" * 64,
            )
        finally:
            sys.settrace(None)

    writer = threading.Thread(target=record)
    writer.start()
    try:
        if vulnerable_line is not None:
            assert parked.wait(2)
        headers = sdk.last_response_headers
        if vulnerable_line is not None:
            assert "x-fx1-completion-id" in headers
    finally:
        release.set()
        writer.join(2)

    assert not writer.is_alive()
    assert set(sdk.last_response_headers) == {
        "x-request-id",
        "x-fx1-api-version",
        "openai-processing-ms",
        "x-fx1-completion-id",
    }


def test_health_never_leaks_env_values() -> None:
    os.environ["MOONSHOT_API_KEY"] = "sentinel-secret"
    try:
        h = Fx1Harness().health()
    finally:
        os.environ.pop("MOONSHOT_API_KEY")
    assert h.backends["hosted_k3"] is True
    assert all(isinstance(v, bool) for v in h.backends.values())
