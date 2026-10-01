"""Pinned QuantCode-Bench import and injected code-correctness evaluation.

No network, subprocess, ``exec``, pickle loading, or strategy execution occurs
here. A caller must supply a sandbox executor and a semantic judge. Their
receipts bind each observation to the exact task, generated code, market-input
bundle, and execution environment. An attestation hash is a provenance link,
not proof that an arbitrary injected callable implements an OS sandbox.

This is a local evaluation protocol over imported public tasks, not a claim of
upstream-harness equivalence or an official leaderboard score. Public tasks are
not automatically uncontaminated holdout data. Missing adapters, data rights,
data requirements, and infrastructure failures remain unmeasured; judge errors
never become passes. Success measures implementation fidelity, never market
performance. Synthetic inputs remain correctness evidence only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

UPSTREAM_REPOSITORY = "https://github.com/LimexAILab/QuantCode-Bench"
UPSTREAM_REVISION = "f8bda951addb409a81aa316c00401dbde60774ae"
UPSTREAM_TASK_SHA256 = "b197e0271779f332c6808ea40167615e3b90061563544b8bdf3c48237a9f17d3"
UPSTREAM_REQUIREMENTS_SHA256 = "7bc4039cfe971ec04de3618c652eca268c95ce07030c5f597c594209344f38b9"
UPSTREAM_TASK_COUNT = 400
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TIMEFRAMES = frozenset({"1m", "5m", "15m", "30m", "1h", "1d"})
_SOURCES = frozenset({"reddit", "tradingview", "stackexchange", "github", "synthetic"})
_REQUIREMENT_FLAGS = (
    "has_hardcoded_prices",
    "requires_session_timing",
    "requires_multi_asset",
    "requires_external_data",
)
_SCHEMA = {
    "version": "quantcode-bench-import-v1",
    "task_required": {
        "id": "positive-integer",
        "reformulated_task": "nonempty-string",
        "source": sorted(_SOURCES),
        "difficulty": ["easy", "medium", "hard"],
        "ticker": "nonempty-string",
        "yf_symbol": "nonempty-string",
        "timeframe": sorted(_TIMEFRAMES),
    },
    "task_optional": {
        "original_ticker": "string-or-null",
        "original_timeframe": "string-or-null",
        "was_reformulated": "boolean",
    },
    "requirements": [
        "task_id",
        "ticker",
        "yf_symbol",
        "timeframe",
        "data_available",
        "needs_reformulation",
        "reasoning",
        *_REQUIREMENT_FLAGS,
    ],
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ).encode()
    ).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


def _hash(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


@dataclass(frozen=True)
class BenchmarkPin:
    """Expected bytes from a revision-addressed source, never a moving branch."""

    repository_url: str
    revision: str
    task_sha256: str
    requirements_sha256: str
    expected_tasks: int = UPSTREAM_TASK_COUNT
    license_id: str = "MIT"

    def __post_init__(self) -> None:
        if not self.repository_url.startswith("https://") or not _REVISION.fullmatch(self.revision):
            raise ValueError("source must have an HTTPS URL and a full Git revision")
        _hash(self.task_sha256, "task_sha256")
        _hash(self.requirements_sha256, "requirements_sha256")
        if type(self.expected_tasks) is not int or self.expected_tasks < 1 or not self.license_id:
            raise ValueError("expected_tasks and license_id must be explicit")

    @property
    def official(self) -> bool:
        return self == OFFICIAL_PIN


OFFICIAL_PIN = BenchmarkPin(
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    UPSTREAM_TASK_SHA256,
    UPSTREAM_REQUIREMENTS_SHA256,
)


@dataclass(frozen=True)
class QuantCodeTask:
    task_id: int
    specification: str
    source: str
    difficulty: str
    ticker: str
    yf_symbol: str
    timeframe: str
    content_sha256: str
    requirements_sha256: str | None
    required_capabilities: tuple[str, ...]
    upstream_data_available: bool | None
    needs_reformulation: bool | None


@dataclass(frozen=True)
class QuantCodeManifest:
    pin: BenchmarkPin
    tasks: tuple[QuantCodeTask, ...]
    file_sha256: str
    content_sha256: str
    schema_sha256: str
    source_sha256: str
    adapter_sha256: str
    requirements_file_sha256: str | None
    complete_import: bool

    @property
    def official_tasks_complete(self) -> bool:
        return self.pin.official and self.complete_import


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read_rows(path: Path, expected_sha256: str) -> tuple[list[dict[str, Any]], str]:
    if path.stat().st_size > 8_000_000:
        raise ValueError("benchmark file exceeds 8 MB import limit")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise ValueError("benchmark source bytes do not match the pin")
    rows = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("benchmark must be a nonempty JSON array of objects")
    return rows, sha


def _requirements(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if type(task_id) is not int or task_id < 1 or task_id in result:
            raise ValueError("requirements task ids must be unique positive integers")
        for field in ("data_available", "needs_reformulation", *_REQUIREMENT_FLAGS):
            if type(row.get(field)) is not bool:
                raise ValueError(f"requirement {field} must be boolean")
        for field in ("ticker", "yf_symbol", "timeframe", "reasoning"):
            _text(row, field)
        result[task_id] = row
    return result


def load_quantcode_bench(
    tasks_path: str | Path,
    *,
    pin: BenchmarkPin = OFFICIAL_PIN,
    requirements_path: str | Path | None = None,
    require_complete: bool = True,
) -> QuantCodeManifest:
    """Import downloaded JSON with byte, semantic, schema and source hashes.

    The official pin requires the actual externally published 400-task file.
    A custom pin permits a separately identified subset or fixture; it cannot
    claim official import status. Missing requirement metadata permits import
    inspection but blocks evaluation. Downloads are the caller's responsibility.
    """
    rows, file_sha = _read_rows(Path(tasks_path), pin.task_sha256)
    requirements_sha: str | None = None
    requirements: dict[int, dict[str, Any]] = {}
    if requirements_path is not None:
        req_rows, requirements_sha = _read_rows(Path(requirements_path), pin.requirements_sha256)
        requirements = _requirements(req_rows)
    tasks: list[QuantCodeTask] = []
    seen: set[int] = set()
    for row in rows:
        task_id = row.get("id")
        if type(task_id) is not int or task_id < 1 or task_id in seen:
            raise ValueError("task ids must be unique positive integers")
        seen.add(task_id)
        for field in (
            "reformulated_task",
            "source",
            "difficulty",
            "ticker",
            "yf_symbol",
            "timeframe",
        ):
            _text(row, field)
        if row["source"] not in _SOURCES or row["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError("unknown task source or difficulty")
        if row["timeframe"] not in _TIMEFRAMES:
            raise ValueError("unsupported task timeframe")
        for field in ("original_ticker", "original_timeframe"):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise ValueError(f"{field} must be string or null")
        if "was_reformulated" in row and type(row["was_reformulated"]) is not bool:
            raise ValueError("was_reformulated must be boolean")
        req = requirements.get(task_id)
        if requirements_path is not None and req is None:
            raise ValueError("task is missing its data requirement record")
        if req is not None and any(
            req[field] != row[field] for field in ("ticker", "yf_symbol", "timeframe")
        ):
            raise ValueError("task and requirement instrument/timeframe disagree")
        tasks.append(
            QuantCodeTask(
                task_id,
                row["reformulated_task"],
                row["source"],
                row["difficulty"],
                row["ticker"],
                row["yf_symbol"],
                row["timeframe"],
                _digest(row),
                _digest(req) if req is not None else None,
                tuple(field for field in _REQUIREMENT_FLAGS if req is not None and req[field]),
                req["data_available"] if req is not None else None,
                req["needs_reformulation"] if req is not None else None,
            )
        )
    if requirements_path is not None and set(requirements) != seen:
        raise ValueError("task and requirement id sets disagree")
    complete = seen == set(range(1, pin.expected_tasks + 1))
    if not seen <= set(range(1, pin.expected_tasks + 1)) or (require_complete and not complete):
        raise ValueError(f"benchmark requires exactly ids 1..{pin.expected_tasks}")
    return QuantCodeManifest(
        pin,
        tuple(sorted(tasks, key=lambda task: task.task_id)),
        file_sha,
        _digest(rows),
        _digest(_SCHEMA),
        _digest(asdict(pin)),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requirements_sha,
        complete,
    )


@dataclass(frozen=True)
class EvaluationProtocol:
    model_id: str
    model_revision: str
    judge_id: str
    judge_revision: str
    evaluation_ids: tuple[int, ...]
    development_ids: tuple[int, ...] = ()
    training_ids: tuple[int, ...] = ()
    training_corpus_sha256: tuple[str, ...] = ()
    contamination_status: Literal["unknown", "audited", "disclosed_overlap"] = "unknown"
    split_label: Literal["public_benchmark", "development", "heldout"] = "public_benchmark"
    max_turns: int = 1

    def __post_init__(self) -> None:
        identities = (self.model_id, self.model_revision, self.judge_id, self.judge_revision)
        if not all(isinstance(value, str) and value.strip() for value in identities):
            raise ValueError("model and judge identities/revisions must be explicit")
        if type(self.max_turns) is not int or not 1 <= self.max_turns <= 10:
            raise ValueError("max_turns must be between 1 and 10")
        if self.contamination_status not in {"unknown", "audited", "disclosed_overlap"}:
            raise ValueError("unknown contamination status")
        if self.split_label not in {"public_benchmark", "development", "heldout"}:
            raise ValueError("unknown split label")
        parts = (self.evaluation_ids, self.development_ids, self.training_ids)
        if not self.evaluation_ids or any(len(set(part)) != len(part) for part in parts):
            raise ValueError("evaluation ids must be nonempty and split ids unique")
        if any(type(task_id) is not int or task_id < 1 for part in parts for task_id in part):
            raise ValueError("split ids must be positive integers")
        if any(set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("training, development and evaluation task ids must be disjoint")
        if self.split_label == "heldout" and self.contamination_status != "audited":
            raise ValueError("heldout label requires a contamination audit")
        for sha in self.training_corpus_sha256:
            _hash(sha, "training_corpus_sha256")


@dataclass(frozen=True)
class TaskDataReceipt:
    task_id: int
    bundle_sha256: str
    requirements_sha256: str
    provider: str
    retrieved_at: datetime
    start_time: datetime
    end_time: datetime
    availability_convention: str
    supported_capabilities: tuple[str, ...] = ()
    rights_confirmed: bool = False
    synthetic: bool = False

    def __post_init__(self) -> None:
        _hash(self.bundle_sha256, "bundle_sha256")
        _hash(self.requirements_sha256, "requirements_sha256")
        if type(self.task_id) is not int or self.task_id < 1:
            raise ValueError("data receipt task id must be a positive integer")
        if type(self.rights_confirmed) is not bool or type(self.synthetic) is not bool:
            raise ValueError("rights and synthetic labels must be boolean")
        if not set(self.supported_capabilities) <= set(_REQUIREMENT_FLAGS):
            raise ValueError("unsupported data capability label")
        for value in (self.retrieved_at, self.start_time, self.end_time):
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("data receipt timestamps must have a timezone")
        if self.start_time > self.end_time or self.end_time > self.retrieved_at:
            raise ValueError("data receipt date range is inconsistent")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.provider,
                self.availability_convention,
            )
        ):
            raise ValueError("provider and availability convention must be explicit")


@dataclass(frozen=True)
class ExecutionReceipt:
    task_sha256: str
    code_sha256: str
    data_bundle_sha256: str
    environment_sha256: str
    sandbox_attestation_sha256: str
    receipt_sha256: str
    compiled: bool
    backtest_success: bool
    total_trades: int
    backend_id: str
    sandboxed: bool


@dataclass(frozen=True)
class JudgeReceipt:
    task_sha256: str
    code_sha256: str
    execution_receipt_sha256: str
    receipt_sha256: str
    judge_id: str
    judge_revision: str
    semantic_aligned: bool


class StrategyModel(Protocol):
    def __call__(self, task: QuantCodeTask, feedback: tuple[str, ...]) -> str: ...


class SandboxExecutor(Protocol):
    def __call__(
        self,
        task: QuantCodeTask,
        code: str,
        data: TaskDataReceipt,
    ) -> ExecutionReceipt: ...


class SemanticJudge(Protocol):
    def __call__(
        self,
        task: QuantCodeTask,
        code: str,
        execution: ExecutionReceipt,
    ) -> JudgeReceipt: ...


@dataclass(frozen=True)
class QuantCodeOutcome:
    task_id: int
    state: Literal["passed", "failed", "not_evaluated"]
    turns: int
    reason: str
    execution: ExecutionReceipt | None = None
    judge: JudgeReceipt | None = None
    synthetic_inputs: bool | None = None


@dataclass(frozen=True)
class QuantCodeReport:
    status: Literal["complete", "partial", "not_evaluated"]
    manifest: QuantCodeManifest
    protocol: EvaluationProtocol
    protocol_sha256: str
    data_provenance_sha256: str
    data_receipts: tuple[TaskDataReceipt, ...]
    outcomes: tuple[QuantCodeOutcome, ...]
    n_evaluated: int
    n_passed: int
    compilation_rate: float | None
    backtest_rate: float | None
    trade_rate: float | None
    judge_pass_rate: float | None
    all_400_evaluated: bool
    research_only: bool = True
    market_evidence: bool = False
    official_score: bool = False
    upstream_protocol_equivalence: str = "UNVERIFIED"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible full receipt, including dates and source identity."""
        result: dict[str, Any] = json.loads(json.dumps(asdict(self), default=_json_default))
        return result


def _execution_valid(
    receipt: ExecutionReceipt, task: QuantCodeTask, code_sha: str, data: TaskDataReceipt
) -> bool:
    for field in (
        "task_sha256",
        "code_sha256",
        "data_bundle_sha256",
        "environment_sha256",
        "sandbox_attestation_sha256",
        "receipt_sha256",
    ):
        _hash(getattr(receipt, field), field)
    return (
        receipt.task_sha256 == task.content_sha256
        and receipt.code_sha256 == code_sha
        and receipt.data_bundle_sha256 == data.bundle_sha256
        and type(receipt.sandboxed) is bool
        and receipt.sandboxed
        and isinstance(receipt.backend_id, str)
        and bool(receipt.backend_id.strip())
        and type(receipt.compiled) is bool
        and type(receipt.backtest_success) is bool
        and type(receipt.total_trades) is int
        and receipt.total_trades >= 0
        and (receipt.compiled or not receipt.backtest_success)
        and (receipt.backtest_success or receipt.total_trades == 0)
    )


def _evaluate_task(
    task: QuantCodeTask,
    protocol: EvaluationProtocol,
    model: StrategyModel | None,
    executor: SandboxExecutor | None,
    judge: SemanticJudge | None,
    data: TaskDataReceipt | None,
) -> QuantCodeOutcome:
    missing = []
    if model is None:
        missing.append("model")
    if executor is None:
        missing.append("sandbox_executor")
    if judge is None:
        missing.append("semantic_judge")
    if task.requirements_sha256 is None:
        missing.append("data_requirements")
    if data is None:
        missing.append("data_bundle")
    if missing:
        return QuantCodeOutcome(task.task_id, "not_evaluated", 0, "missing:" + ",".join(missing))
    assert model is not None and executor is not None and judge is not None and data is not None
    if (
        data.task_id != task.task_id
        or data.requirements_sha256 != task.requirements_sha256
        or data.rights_confirmed is not True
        or type(data.synthetic) is not bool
        or not set(task.required_capabilities) <= set(data.supported_capabilities)
    ):
        return QuantCodeOutcome(
            task.task_id, "not_evaluated", 0, "data_provenance_or_coverage_rejected"
        )
    feedback: tuple[str, ...] = ()
    execution: ExecutionReceipt | None = None
    judgement: JudgeReceipt | None = None
    for turn in range(1, protocol.max_turns + 1):
        try:
            code = model(task, feedback)
        except Exception:
            return QuantCodeOutcome(task.task_id, "not_evaluated", turn, "model_error")
        if not isinstance(code, str) or not code.strip():
            feedback += ("empty_or_invalid_model_code",)
            continue
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        try:
            execution = executor(task, code, data)
            if not isinstance(execution, ExecutionReceipt) or not _execution_valid(
                execution, task, code_sha, data
            ):
                return QuantCodeOutcome(
                    task.task_id, "not_evaluated", turn, "execution_receipt_rejected"
                )
        except Exception:
            return QuantCodeOutcome(task.task_id, "not_evaluated", turn, "sandbox_executor_error")
        if not execution.compiled:
            feedback += ("compilation_failed",)
            continue
        if not execution.backtest_success:
            feedback += ("backtest_failed",)
            continue
        if execution.total_trades == 0:
            feedback += ("no_trades",)
            continue
        try:
            judgement = judge(task, code, execution)
            if not isinstance(judgement, JudgeReceipt):
                raise ValueError("wrong judge receipt type")
            for field in (
                "task_sha256",
                "code_sha256",
                "execution_receipt_sha256",
                "receipt_sha256",
            ):
                _hash(getattr(judgement, field), field)
            if (
                judgement.task_sha256 != task.content_sha256
                or judgement.code_sha256 != code_sha
                or judgement.execution_receipt_sha256 != execution.receipt_sha256
                or judgement.judge_id != protocol.judge_id
                or judgement.judge_revision != protocol.judge_revision
                or type(judgement.semantic_aligned) is not bool
            ):
                return QuantCodeOutcome(
                    task.task_id,
                    "not_evaluated",
                    turn,
                    "judge_receipt_rejected",
                    execution=execution,
                    synthetic_inputs=data.synthetic,
                )
        except Exception:
            return QuantCodeOutcome(
                task.task_id,
                "not_evaluated",
                turn,
                "semantic_judge_error",
                execution=execution,
                synthetic_inputs=data.synthetic,
            )
        if judgement.semantic_aligned:
            return QuantCodeOutcome(
                task.task_id,
                "passed",
                turn,
                "semantic_alignment_observed",
                execution,
                judgement,
                data.synthetic,
            )
        feedback += ("semantic_alignment_failed",)
    return QuantCodeOutcome(
        task.task_id,
        "failed",
        protocol.max_turns,
        feedback[-1],
        execution,
        judgement,
        data.synthetic,
    )


def run_quantcode_eval(
    manifest: QuantCodeManifest,
    protocol: EvaluationProtocol,
    *,
    model: StrategyModel | None = None,
    executor: SandboxExecutor | None = None,
    judge: SemanticJudge | None = None,
    data_receipts: Mapping[int, TaskDataReceipt] | None = None,
) -> QuantCodeReport:
    """Measure selected imported tasks; never execute generated code internally.

    ``judge_pass_rate`` uses the planned selected-task denominator, counting
    unknown observations conservatively as no pass. It is None when none were
    evaluated. ``n_evaluated`` and status expose missing coverage. A 100% result
    on a selected subset never sets ``all_400_evaluated``. An injected executor
    must enforce OS isolation, no credentials, resource limits, and constrained
    market-data access outside this module; a timeout alone is not a sandbox.
    """
    by_id = {task.task_id: task for task in manifest.tasks}
    all_ids = (*protocol.evaluation_ids, *protocol.development_ids, *protocol.training_ids)
    if not set(all_ids) <= set(by_id):
        raise ValueError("protocol split contains ids outside the imported benchmark")
    parts = (protocol.evaluation_ids, protocol.development_ids, protocol.training_ids)
    specs = [
        {re.sub(r"\s+", " ", by_id[task_id].specification).strip() for task_id in part}
        for part in parts
    ]
    if any(specs[i] & specs[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("duplicate specifications cross the declared split")
    data = data_receipts or {}
    used_data = tuple(data[task_id] for task_id in protocol.evaluation_ids if task_id in data)
    outcomes = tuple(
        _evaluate_task(by_id[task_id], protocol, model, executor, judge, data.get(task_id))
        for task_id in protocol.evaluation_ids
    )
    n_evaluated = sum(outcome.state != "not_evaluated" for outcome in outcomes)
    n_passed = sum(outcome.state == "passed" for outcome in outcomes)
    executions = [outcome.execution for outcome in outcomes if outcome.execution is not None]
    stage_observed = bool(executions) or n_evaluated > 0
    all_tasks = manifest.complete_import and set(protocol.evaluation_ids) == set(by_id)
    status: Literal["complete", "partial", "not_evaluated"] = (
        "not_evaluated"
        if n_evaluated == 0
        else "complete"
        if all_tasks and n_evaluated == len(outcomes)
        else "partial"
    )
    return QuantCodeReport(
        status,
        manifest,
        protocol,
        _digest(asdict(protocol)),
        _digest([asdict(receipt) for receipt in used_data]),
        used_data,
        outcomes,
        n_evaluated,
        n_passed,
        sum(receipt.compiled for receipt in executions) / len(outcomes) if stage_observed else None,
        sum(receipt.backtest_success for receipt in executions) / len(outcomes)
        if stage_observed
        else None,
        sum(receipt.total_trades > 0 for receipt in executions) / len(outcomes)
        if stage_observed
        else None,
        n_passed / len(outcomes) if n_evaluated else None,
        manifest.official_tasks_complete
        and status == "complete"
        and n_evaluated == UPSTREAM_TASK_COUNT,
    )
