"""Offline SYNTHETIC fixtures validate import integrity and fail-closed scoring.

Scripted receipts below test protocol accounting. They are neither generated
market evidence nor an actual sandbox/model benchmark run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fx1.eval.quantcode_bench import (
    OFFICIAL_PIN,
    BenchmarkPin,
    EvaluationProtocol,
    ExecutionReceipt,
    JudgeReceipt,
    QuantCodeManifest,
    QuantCodeTask,
    TaskDataReceipt,
    load_quantcode_bench,
    run_quantcode_eval,
)

SHA = "a" * 64


def _files(
    tmp_path: Path,
    n: int = 2,
    *,
    rows: list[dict] | None = None,
    expected_tasks: int | None = None,
) -> tuple[Path, Path, BenchmarkPin]:
    if rows is None:
        rows = [
            {
                "id": task_id,
                "reformulated_task": f"SYNTHETIC specification {task_id}",
                "source": "synthetic",
                "difficulty": "easy",
                "ticker": "generic",
                "yf_symbol": "AAPL",
                "timeframe": "1d",
                "original_ticker": None,
                "original_timeframe": None,
                "was_reformulated": False,
            }
            for task_id in range(1, n + 1)
        ]
    requirements = [
        {
            "task_id": row["id"],
            "ticker": row["ticker"],
            "yf_symbol": row["yf_symbol"],
            "timeframe": row["timeframe"],
            "data_available": True,
            "needs_reformulation": False,
            "has_hardcoded_prices": False,
            "requires_session_timing": False,
            "requires_multi_asset": False,
            "requires_external_data": False,
            "reasoning": "SYNTHETIC fixture only",
        }
        for row in rows
    ]
    task_path, req_path = tmp_path / "tasks.json", tmp_path / "requirements.json"
    task_path.write_text(json.dumps(rows))
    req_path.write_text(json.dumps(requirements))
    pin = BenchmarkPin(
        "https://example.test/SYNTHETIC-fixture",
        "a" * 40,
        hashlib.sha256(task_path.read_bytes()).hexdigest(),
        hashlib.sha256(req_path.read_bytes()).hexdigest(),
        expected_tasks or n,
    )
    return task_path, req_path, pin


def _manifest(tmp_path: Path, n: int = 2) -> QuantCodeManifest:
    task_path, req_path, pin = _files(tmp_path, n)
    return load_quantcode_bench(task_path, pin=pin, requirements_path=req_path)


def _protocol(ids: tuple[int, ...] = (1, 2), **kwargs) -> EvaluationProtocol:
    return EvaluationProtocol(
        "scripted-SYNTHETIC-model", "fixture-v1", "scripted-judge", "v1", ids, **kwargs
    )


def _data(manifest: QuantCodeManifest) -> dict[int, TaskDataReceipt]:
    now = datetime(2026, 10, 1, tzinfo=UTC)
    return {
        task.task_id: TaskDataReceipt(
            task.task_id,
            SHA,
            task.requirements_sha256 or SHA,
            "SYNTHETIC-fixture",
            now,
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 2, 1, tzinfo=UTC),
            "SYNTHETIC bar-close convention",
            task.required_capabilities,
            True,
            True,
        )
        for task in manifest.tasks
    }


def _model(task: QuantCodeTask, feedback: tuple[str, ...]) -> str:
    # This deliberate raising payload proves the module treats code as data.
    return "raise RuntimeError('this generated payload must never execute on the host')"


def _executor(task: QuantCodeTask, code: str, data: TaskDataReceipt) -> ExecutionReceipt:
    return ExecutionReceipt(
        task.content_sha256,
        hashlib.sha256(code.encode()).hexdigest(),
        data.bundle_sha256,
        SHA,
        SHA,
        SHA,
        True,
        True,
        1,
        "scripted-SYNTHETIC-executor",
        True,
    )


def _judge(task: QuantCodeTask, code: str, execution: ExecutionReceipt) -> JudgeReceipt:
    return JudgeReceipt(
        task.content_sha256,
        hashlib.sha256(code.encode()).hexdigest(),
        execution.receipt_sha256,
        SHA,
        "scripted-judge",
        "v1",
        True,
    )


def _run(manifest: QuantCodeManifest, **kwargs):
    adapters = dict(model=_model, executor=_executor, judge=_judge, data_receipts=_data(manifest))
    adapters.update(kwargs)
    return run_quantcode_eval(manifest, _protocol(), **adapters)


def test_import_records_byte_semantic_schema_source_and_adapter_hashes(tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest.complete_import
    assert not manifest.official_tasks_complete
    assert manifest.file_sha256 == manifest.pin.task_sha256
    assert manifest.requirements_file_sha256 == manifest.pin.requirements_sha256
    for sha in (
        manifest.content_sha256,
        manifest.schema_sha256,
        manifest.source_sha256,
        manifest.adapter_sha256,
    ):
        assert len(sha) == 64
    assert manifest == _manifest(tmp_path)


def test_official_pin_rejects_locally_fabricated_tasks(tmp_path):
    path, _, _ = _files(tmp_path, 400)
    with pytest.raises(ValueError, match="source bytes"):
        load_quantcode_bench(path, pin=OFFICIAL_PIN)


def test_pin_requires_immutable_source_and_explicit_hashes():
    with pytest.raises(ValueError, match="full Git revision"):
        BenchmarkPin("https://example.test", "main", SHA, SHA)
    with pytest.raises(ValueError, match="SHA-256"):
        BenchmarkPin("https://example.test", "a" * 40, "unknown", SHA)


def test_changed_download_is_rejected(tmp_path):
    path, requirements, pin = _files(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source bytes"):
        load_quantcode_bench(path, pin=pin, requirements_path=requirements)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", True),
        ("source", "unknown"),
        ("timeframe", "4h"),
        ("was_reformulated", 1),
        ("reformulated_task", ""),
        ("difficulty", "extreme"),
    ],
)
def test_schema_rejects_wrong_types_and_unknown_values(tmp_path, field, value):
    path, _, _ = _files(tmp_path)
    rows = json.loads(path.read_text())
    rows[0][field] = value
    path, requirements, pin = _files(tmp_path, rows=rows)
    with pytest.raises(ValueError):
        load_quantcode_bench(path, pin=pin, requirements_path=requirements)


def test_duplicate_ids_and_duplicate_json_keys_rejected(tmp_path):
    path, _, _ = _files(tmp_path)
    rows = json.loads(path.read_text())
    rows[1]["id"] = rows[0]["id"]
    path, _, pin = _files(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="unique positive"):
        load_quantcode_bench(path, pin=pin)
    path.write_text('[{"id":1,"id":2}]')
    pin = replace(pin, task_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_quantcode_bench(path, pin=pin)


def test_partial_import_requires_explicit_opt_in(tmp_path):
    path, _, pin = _files(tmp_path, expected_tasks=400)
    with pytest.raises(ValueError, match="exactly ids"):
        load_quantcode_bench(path, pin=pin)
    manifest = load_quantcode_bench(path, pin=pin, require_complete=False)
    assert not manifest.complete_import
    assert not manifest.official_tasks_complete


def test_partial_import_still_rejects_ids_outside_pin(tmp_path):
    path, _, _ = _files(tmp_path)
    rows = json.loads(path.read_text())
    rows[1]["id"] = 999
    path, _, pin = _files(tmp_path, rows=rows, expected_tasks=400)
    with pytest.raises(ValueError, match="exactly ids"):
        load_quantcode_bench(path, pin=pin, require_complete=False)


def test_data_requirements_bind_instrument_and_unique_ids(tmp_path):
    path, requirements, pin = _files(tmp_path)
    rows = json.loads(requirements.read_text())
    rows[0]["yf_symbol"] = "MSFT"
    requirements.write_text(json.dumps(rows))
    pin = replace(pin, requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="disagree"):
        load_quantcode_bench(path, pin=pin, requirements_path=requirements)


def test_missing_adapters_are_unmeasured_and_do_not_invoke_model(tmp_path):
    manifest = _manifest(tmp_path)

    def must_not_call(task, feedback):
        raise AssertionError("model must not be called without sandbox and judge")

    report = run_quantcode_eval(manifest, _protocol(), model=must_not_call)
    assert report.status == "not_evaluated"
    assert report.judge_pass_rate is None
    assert report.n_evaluated == 0
    assert all(outcome.turns == 0 for outcome in report.outcomes)


def test_missing_requirements_blocks_evaluation(tmp_path):
    path, _, pin = _files(tmp_path)
    manifest = load_quantcode_bench(path, pin=pin)
    report = _run(manifest)
    assert report.status == "not_evaluated"
    assert "data_requirements" in report.outcomes[0].reason


def test_scripted_pipeline_counts_correctness_only_and_never_executes_code(tmp_path):
    manifest = _manifest(tmp_path)
    report = _run(manifest)
    assert report.status == "complete"
    assert report.n_evaluated == report.n_passed == 2
    assert report.judge_pass_rate == 1.0
    assert report.compilation_rate == report.backtest_rate == report.trade_rate == 1.0
    assert not report.all_400_evaluated
    assert not report.official_score
    assert not report.market_evidence
    assert report.research_only
    assert all(outcome.synthetic_inputs for outcome in report.outcomes)
    assert len(report.data_provenance_sha256) == 64
    assert report.to_dict()["data_receipts"][0]["retrieved_at"].endswith("+00:00")


def test_success_on_subset_remains_partial(tmp_path):
    manifest = _manifest(tmp_path)
    report = run_quantcode_eval(
        manifest,
        _protocol((1,)),
        model=_model,
        executor=_executor,
        judge=_judge,
        data_receipts=_data(manifest),
    )
    assert report.status == "partial"
    assert report.judge_pass_rate == 1.0
    assert not report.all_400_evaluated


def test_splits_reject_overlap_missing_ids_and_unsupported_holdout_claim(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        _protocol(training_ids=(1,))
    with pytest.raises(ValueError, match="contamination audit"):
        _protocol(split_label="heldout")
    with pytest.raises(ValueError, match="outside"):
        run_quantcode_eval(_manifest(tmp_path), _protocol((3,)))


def test_duplicate_specifications_cannot_cross_splits(tmp_path):
    path, _, _ = _files(tmp_path)
    rows = json.loads(path.read_text())
    rows[1]["reformulated_task"] = rows[0]["reformulated_task"]
    path, requirements, pin = _files(tmp_path, rows=rows)
    manifest = load_quantcode_bench(path, pin=pin, requirements_path=requirements)
    with pytest.raises(ValueError, match="specifications cross"):
        run_quantcode_eval(manifest, _protocol((1,), training_ids=(2,)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("sandboxed", False),
        ("code_sha256", "b" * 64),
        ("total_trades", -1),
        ("compiled", "yes"),
        ("receipt_sha256", "missing"),
    ],
)
def test_execution_receipt_failure_never_scores_pass(tmp_path, field, value):
    def bad_executor(task, code, data):
        return replace(_executor(task, code, data), **{field: value})

    report = _run(_manifest(tmp_path), executor=bad_executor)
    assert report.status == "not_evaluated"
    assert report.n_passed == 0


def test_judge_exception_is_unmeasured_not_passed(tmp_path):
    def failing_judge(task, code, execution):
        raise RuntimeError("judge infrastructure failed")

    report = _run(_manifest(tmp_path), judge=failing_judge)
    assert report.status == "not_evaluated"
    assert report.n_passed == 0
    assert all(outcome.reason == "semantic_judge_error" for outcome in report.outcomes)
    assert report.compilation_rate == report.backtest_rate == report.trade_rate == 1.0
    assert report.judge_pass_rate is None


def test_wrong_judge_revision_rejected(tmp_path):
    def wrong_judge(task, code, execution):
        return replace(_judge(task, code, execution), judge_revision="unrecorded")

    report = _run(_manifest(tmp_path), judge=wrong_judge)
    assert report.status == "not_evaluated"
    assert report.outcomes[0].reason == "judge_receipt_rejected"


def test_no_trades_is_observed_failure_without_judge_call(tmp_path):
    def no_trades(task, code, data):
        return replace(_executor(task, code, data), total_trades=0)

    def must_not_judge(task, code, execution):
        raise AssertionError("no trades must stop before judging")

    report = _run(_manifest(tmp_path), executor=no_trades, judge=must_not_judge)
    assert report.status == "complete"
    assert report.judge_pass_rate == 0.0
    assert report.compilation_rate == report.backtest_rate == 1.0
    assert report.trade_rate == 0.0
    assert all(outcome.reason == "no_trades" for outcome in report.outcomes)


def test_agentic_repair_uses_feedback_and_records_turn_count(tmp_path):
    manifest = _manifest(tmp_path)
    feedback_seen = []

    def repair_model(task, feedback):
        feedback_seen.append(feedback)
        return "fixed" if feedback else "broken"

    def repair_executor(task, code, data):
        receipt = _executor(task, code, data)
        return (
            receipt
            if code == "fixed"
            else replace(receipt, compiled=False, backtest_success=False, total_trades=0)
        )

    report = run_quantcode_eval(
        manifest,
        _protocol(max_turns=2),
        model=repair_model,
        executor=repair_executor,
        judge=_judge,
        data_receipts=_data(manifest),
    )
    assert report.n_passed == 2
    assert all(outcome.turns == 2 for outcome in report.outcomes)
    assert ("compilation_failed",) in feedback_seen


def test_data_rights_requirements_and_timezone_fail_closed(tmp_path):
    manifest = _manifest(tmp_path)
    data = _data(manifest)
    data[1] = replace(data[1], rights_confirmed=False)
    report = _run(manifest, data_receipts=data)
    assert report.status == "partial"
    assert report.n_evaluated == report.n_passed == 1
    assert report.judge_pass_rate == 0.5
    assert report.outcomes[0].state == "not_evaluated"
    with pytest.raises(ValueError, match="timezone"):
        replace(data[1], retrieved_at=datetime(2026, 10, 1))


def test_requirement_capabilities_cannot_be_silently_dropped(tmp_path):
    manifest = _manifest(tmp_path)
    task = replace(manifest.tasks[0], required_capabilities=("requires_multi_asset",))
    manifest = replace(manifest, tasks=(task, manifest.tasks[1]))
    data = _data(manifest)
    data[1] = replace(data[1], supported_capabilities=())
    report = _run(manifest, data_receipts=data)
    assert report.outcomes[0].reason == "data_provenance_or_coverage_rejected"
