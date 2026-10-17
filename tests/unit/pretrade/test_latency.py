"""Check latency report consistency; the dedicated CI job measures its gate.

The required ``pretrade-risk`` job runs ``python -m quant_fund.pretrade.bench
--gate`` without pytest workers or coverage instrumentation.
"""

from __future__ import annotations

from quant_fund.pretrade.bench import P50_LIMIT_NS, P99_LIMIT_NS, run_benchmark


def test_allow_path_reports_latency_gate_consistently() -> None:
    report = run_benchmark(trials=3, samples=3000, warmup=800, gate=True)
    assert report["implementation"] == "cpython-slots"
    assert report["numba"] is False
    assert report["rust"] is False
    assert report["min_ns"] <= report["p50_ns"] <= report["p99_ns"] <= report["max_ns"]
    assert len(report["trials"]) == 3
    assert "kill_switch" in report["checks"]
    assert "good_faith_violation" in report["checks"]
    within = report["p50_ns"] < P50_LIMIT_NS and report["p99_ns"] < P99_LIMIT_NS
    assert report["gate_pass"] is within
