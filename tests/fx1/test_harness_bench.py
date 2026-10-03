"""KATs for the harness perf bench (fx1.harness_bench + ops receipt)."""

from __future__ import annotations

import pytest

from fx1.harness_bench import DEFAULT_BENCH_PROMPT, run_bench
from fx1.sdk import CompletionResult
from fx1.serve.ops_receipt import BENCH_RESULT_SCHEMA, bench_receipt
from quant_fund.research.receipt_v2 import verify_receipt_payload


class _Stub:
    """Completes instantly; records kwargs; reports usage."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, **kw):
        self.calls.append({"messages": messages, **dict(kw)})
        return CompletionResult(
            backend="byok",
            model="stub-v0",
            content="ok",
            usage={"prompt_tokens": 5, "completion_tokens": 7},
        )


class _Flaky:
    """Fails every other call."""

    def __init__(self) -> None:
        self.i = 0

    def complete(self, messages, **kw):
        self.i += 1
        if self.i % 2:
            raise RuntimeError("boom")
        return CompletionResult(backend="x", model="m", content="c")


def test_bench_counts_and_usage() -> None:
    stub = _Stub()
    rec = run_bench(
        stub,
        n=8,
        concurrency=4,
        warmup=1,
        prompt="hi",
        max_tokens=9,
        timeout_s=5.0,
        backend="byok",
        seed=3,
        mode="remote",
    )
    m = rec["metrics"]
    assert m["measured_requests"] == 8
    assert m["error_count"] == 0 and m["error_rate"] == 0.0 and m["errors"] == {}
    assert m["prompt_tokens_total"] == 40
    assert m["completion_tokens_total"] == 56
    assert m["usage_reported"] == 8
    assert m["models"] == ["stub-v0"] and m["backends"] == ["byok"]
    assert rec["params"]["n"] == 8 and rec["params"]["seed"] == 3
    # warmup + measured == 9 calls; kwargs forward verbatim
    assert len(stub.calls) == 9
    assert stub.calls[0]["max_tokens"] == 9 and stub.calls[0]["backend"] == "byok"


def test_bench_prompt_is_digested_not_embedded() -> None:
    rec = run_bench(_Stub(), n=1, concurrency=1, warmup=0, prompt="secret probe text")
    blob = repr(rec)
    assert "secret probe text" not in blob
    assert rec["params"]["prompt_sha256"]
    assert rec["params"]["prompt_chars"] == len("secret probe text")


def test_bench_error_histogram_and_rate() -> None:
    rec = run_bench(_Flaky(), n=6, concurrency=1, warmup=0)
    m = rec["metrics"]
    assert m["error_count"] == 3
    assert m["error_rate"] == pytest.approx(0.5)
    assert m["errors"] == {"RuntimeError": 3}
    assert m["measured_requests"] == 6


def test_bench_all_failed_still_reports() -> None:
    class _Dead:
        def complete(self, messages, **kw):
            raise ConnectionError("down")

    rec = run_bench(_Dead(), n=3, concurrency=2, warmup=0)
    m = rec["metrics"]
    assert m["error_count"] == 3 and m["errors"] == {"ConnectionError": 3}
    assert m["models"] == [] and m["completion_tokens_total"] == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"n": 0},
        {"n": 4097},
        {"concurrency": 0},
        {"concurrency": 257},
        {"warmup": -1},
        {"warmup": 257},
        {"max_tokens": 0},
        {"timeout_s": 0.0},
        {"prompt": "   "},
        {"prompt": "x" * 40000},
        {"mode": "sideways"},
    ],
)
def test_bench_param_bounds_fail_closed(kw: dict) -> None:
    with pytest.raises(ValueError):
        run_bench(_Stub(), **kw)


def test_bench_receipt_seals_and_verifies() -> None:
    rec = run_bench(_Stub(), n=2, concurrency=1, warmup=0)
    doc = bench_receipt(rec)
    assert doc["schema"] == BENCH_RESULT_SCHEMA == "fx1_bench_result.v1"
    assert doc["kind"] == "fx1_bench_result"
    assert doc["research_only"] is True and doc["live_pnl_claim"] is False
    assert doc["receipt_sha256"]
    v = verify_receipt_payload(doc)
    assert v["valid"], v["errors"]


def test_bench_default_prompt_constant() -> None:
    assert DEFAULT_BENCH_PROMPT.strip()
    rec = run_bench(_Stub(), n=1, concurrency=1, warmup=0)
    assert rec["mode"] == "in_process"
