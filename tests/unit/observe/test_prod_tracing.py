from __future__ import annotations

import pytest

from quant_fund.observe.metrics import render_prometheus, reset_metrics
from quant_fund.observe.tracing import clear_spans, span


def test_malformed_otlp_port_does_not_escape_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "http://127.0.0.1:not-a-port/v1/traces")
    reset_metrics()
    clear_spans()

    def work() -> int:
        with span("decision"):
            return 7

    assert work() == 7
    assert 'dipcatcher_errors_total{stage="otel_export"} 1' in render_prometheus()
