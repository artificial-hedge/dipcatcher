"""Focused fail-closed input validation for fx1 operator surfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from fx1.eval.ext_bench_banks import ExternalBenchmarkSchemaError, load_mtbench_bank
from fx1.forecast.evaluate import forward_simple_returns
from fx1.train.config import TrainConfig
from fx1.train.run import _validate_corpus, _validate_eval_gate
from fx1.webresearch.engine import ResearchBudget
from fx1.webresearch.fetch import fetch_page, url_policy_error
from fx1.webresearch.search import DuckDuckGoLiteSearcher


def _train_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_name": "synthetic",
        "stage": "proxy",
        "corpus_jsonl": "corpus.jsonl",
        "eval_results_json": "eval.json",
        "estimated_nodes": 1,
        "estimated_gpu_hours": 1.0,
        "estimated_cost_usd": 0.0,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("field", ["learning_rate", "estimated_gpu_hours", "estimated_cost_usd"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_train_config_rejects_nonfinite_numbers(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(**_train_payload(**{field: value}))


def test_train_config_rejects_undeclared_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainConfig(**_train_payload(learning_rat=1e-3))


def test_manifest_validators_reject_nonobject_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus line 1 must be a JSON object"):
        _validate_corpus(corpus)

    evaluation = tmp_path / "eval.json"
    evaluation.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="eval results must be a JSON object"):
        _validate_eval_gate(evaluation)


def test_eval_gate_does_not_accept_truthy_pass_flags(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "honesty_gate_passed": "true",
                "results": [{"kind": "honesty", "passed": "false"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="honesty gate not passed"):
        _validate_eval_gate(path)


def test_benchmark_loader_wraps_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_bytes(b"\xff")
    with pytest.raises(ExternalBenchmarkSchemaError, match="cannot read benchmark file"):
        load_mtbench_bank(path)


def test_benchmark_loader_wraps_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    def deny_read(path: Path, *, encoding: str) -> str:
        raise PermissionError("synthetic read failure")

    monkeypatch.setattr(Path, "read_text", deny_read)
    with pytest.raises(ExternalBenchmarkSchemaError, match="PermissionError"):
        load_mtbench_bank(path)


@pytest.mark.parametrize("horizon", [float("nan"), float("inf"), 1.5, True, 0])
def test_forward_returns_rejects_noninteger_horizons(horizon: Any) -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [stamp],
            "available_time": [stamp],
            "close": [1.0],
        }
    )
    with pytest.raises(ValueError, match="horizon must be an integer"):
        forward_simple_returns(bars, horizon)


def _no_open(request: object, timeout_s: float) -> object:
    raise AssertionError("network opener must not be called")


def _no_resolve(host: str, port: int) -> list[tuple[Any, ...]]:
    raise AssertionError("DNS must not be called")


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_s": 0.0},
        {"timeout_s": float("nan")},
        {"timeout_s": float("inf")},
        {"max_bytes": 0},
        {"max_bytes": -1},
        {"max_bytes": True},
        {"max_redirects": -1},
        {"max_redirects": True},
    ],
)
def test_fetch_rejects_invalid_limits_before_network(options: dict[str, Any]) -> None:
    result = fetch_page("https://example.com", open_fn=_no_open, resolve_fn=_no_resolve, **options)
    assert not result.ok
    assert result.error.startswith("invalid ")


def test_fetch_malformed_url_is_an_error_not_an_exception() -> None:
    result = fetch_page("http://[::1", open_fn=_no_open, resolve_fn=_no_resolve)
    assert not result.ok
    assert "invalid URL" in result.error


def test_empty_dns_answer_is_not_public() -> None:
    problem = url_policy_error("https://empty.example", resolve_fn=lambda host, port: [])
    assert problem is not None
    assert "did not resolve" in problem


@pytest.mark.parametrize(
    ("query", "options"),
    [
        (" ", {}),
        ("query", {"max_results": 0}),
        ("query", {"max_results": -1}),
        ("query", {"max_results": True}),
        ("query", {"timeout_s": 0.0}),
        ("query", {"timeout_s": float("nan")}),
        ("query", {"timeout_s": float("inf")}),
    ],
)
def test_search_rejects_invalid_inputs_before_network(query: str, options: dict[str, Any]) -> None:
    searcher = DuckDuckGoLiteSearcher(open_fn=_no_open)
    result = searcher.search(query, **options)
    assert result.hits == []
    assert result.error


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_duration_s", 0.0),
        ("max_duration_s", float("nan")),
        ("max_duration_s", float("inf")),
        ("max_searches", 0),
        ("max_sources", True),
        ("max_bytes_per_source", -1),
        ("max_depth", -1),
        ("max_results_per_search", 0),
        ("search_pause_s", -0.1),
        ("search_pause_s", float("nan")),
        ("search_pause_s", "invalid"),
    ],
)
def test_research_budget_rejects_invalid_limits(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        ResearchBudget(**{field: value})
