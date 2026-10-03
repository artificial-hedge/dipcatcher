"""fx-1 eval submissions — the seeded suite battery over the wire.

Every eval the harness ships takes a ``ModelFn``; a resolved serving
backend *is* one. This module is the bridge between them:

- ``EVAL_SUITES`` — lazily-loaded registry (suite name -> runner) so the
  eval import chain never loads for a server that only serves
  completions. ``accepts_judge`` marks the suites that take a grader
  ``ModelFn`` (capability and the external-bench adapter).
- ``EvalRecord`` — the bounded-store record a submission produces:
  suite/seed/backend + the serialized report, sealed on export as
  ``fx1_eval_record.v1``.
- ``EvalStore`` — LRU of records with an Idempotency-Key index (the
  same contract ``_JobStore`` gives run jobs).
- ``metered_model`` — wraps the resolved backend so every model call
  inside an eval lands on the metrics ledger under
  ``eval:{suite}:{backend}`` — never silent.
- ``run_eval_record`` — the worker: drive the runner, serialize the
  report fail-closed, stamp finish. Called from the jobs executor under
  the shared inflight/drain contract.

Evals always run under the decode pin ``{"temperature": 0.0}`` —
recorded on the record so a sealed eval receipt states the exact decode
config the evidence was produced under.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from fx1.eval.suite import ModelFn
from fx1.serve.backends import SamplingParams

__all__ = [
    "EVAL_SUITES",
    "EvalDiff",
    "EvalDiffDelta",
    "EvalRecord",
    "EvalStore",
    "EvalTaskTransition",
    "diff_eval_records",
    "eval_record_receipt",
    "eval_runner",
    "metered_model",
    "run_eval_record",
]

# suite -> (module, function, accepts_judge). Lazy: importing the eval
# package chain (bank seeding, adapters) costs real time at import, so
# the registry stores dotted paths and resolves on first run.
_EVAL_RUNNERS: dict[str, tuple[str, str, bool]] = {
    "capability": ("fx1.eval.capability", "run_capability_eval", True),
    "calibration": ("fx1.eval.calibration_eval", "run_calibration_eval", False),
    "tooluse": ("fx1.eval.tooluse_eval", "run_tooluse_eval", False),
    "retrieval": ("fx1.eval.retrieval_eval", "run_retrieval_eval", False),
    "ts_reasoning": ("fx1.eval.ts_reasoning", "run_ts_reasoning_eval", False),
    "ext_bench": ("fx1.eval.ext_bench", "run_ext_bench_eval", True),
    "options_reasoning": (
        "fx1.eval.options_reasoning_eval",
        "run_options_reasoning_eval",
        False,
    ),
}

EVAL_SUITES: tuple[str, ...] = tuple(_EVAL_RUNNERS)

EvalSuiteName = Literal[
    "capability",
    "calibration",
    "tooluse",
    "retrieval",
    "ts_reasoning",
    "ext_bench",
    "options_reasoning",
]

# Decode pin every eval runs under — recorded verbatim on the record so
# the sealed receipt states the config the evidence came from.
EVAL_SAMPLING = SamplingParams()


def eval_runner(suite: str) -> Callable[..., Any]:
    """Resolve a suite name to its ``run_*_eval`` callable.

    Fail closed: unknown suite names are a KeyError at submit time, never
    a worker fault. Import happens here (not module top) so the eval
    chain loads only when an eval actually runs.
    """
    module_name, fn_name, _accepts_judge = _EVAL_RUNNERS[suite]
    fn: Callable[..., Any] = getattr(importlib.import_module(module_name), fn_name)
    return fn


def suite_accepts_judge(suite: str) -> bool:
    return _EVAL_RUNNERS[suite][2]


def report_dump(report: Any) -> dict[str, Any]:
    """Serialize a suite report fail-closed — pydantic or dataclass,
    nothing else seals into a record."""
    if isinstance(report, BaseModel):
        return report.model_dump(mode="json")
    if is_dataclass(report) and not isinstance(report, type):
        return asdict(report)
    raise TypeError(f"eval report is not serializable: {type(report).__name__}")


def metered_model(
    backend: Any,
    *,
    metric_key: str,
    record: Callable[[str, bool, float, dict[str, int] | None], None],
) -> ModelFn:
    """Wrap ``backend`` as a ModelFn that meters every call.

    Each suite call lands on the complete ledger under ``metric_key``
    (``eval:{suite}:{backend}``) — eval traffic is metered model work,
    visible but never conflated with user completions.
    """

    def _fn(messages: list[dict[str, str]]) -> str:
        started = time.monotonic()
        try:
            out = backend.complete(messages, sampling=EVAL_SAMPLING)
        except Exception:
            record(metric_key, False, (time.monotonic() - started) * 1000.0, None)
            raise
        record(
            metric_key,
            True,
            (time.monotonic() - started) * 1000.0,
            getattr(backend, "last_usage", None),
        )
        out_str: str = out
        return out_str

    return _fn


class EvalRecord(BaseModel):
    """One submitted eval's durable record — sealed on receipt export.

    Never carries request-side credentials (``byok``/keys): the record is
    evidence, and evidence must be safe to export.
    """

    model_config = ConfigDict(extra="forbid")

    eval_id: str
    suite: str
    # The serving link actually used (attempts carries the full chain).
    backend: str
    seed: int
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: float
    finished_at: float | None = None
    report: dict[str, Any] | None = None
    error: str | None = None
    # Ordered chain trace — {backend, ok, error_class} per link tried.
    attempts: list[dict[str, Any]] | None = None
    # The decode pin the eval ran under (always the 0.0 default today).
    sampling: dict[str, Any] | None = None
    # Terminal-state webhook (the job contract): on every terminal
    # transition the record is POSTed to ``callback_url`` — never the
    # secret, which lives only as a signing key.
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    callback_error: str | None = None
    callback_attempts: int = 0
    _callback_secret: str | None = PrivateAttr(default=None)


class EvalStore:
    """Bounded LRU of eval records + an Idempotency-Key index — the
    ``_JobStore`` contract for eval submissions. Eviction drops the
    idempotency mapping with the record."""

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._records: OrderedDict[str, EvalRecord] = OrderedDict()
        self._keys: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._record_key: dict[str, str] = {}

    @property
    def capacity(self) -> int:
        return self._max

    def get(self, eval_id: str) -> EvalRecord | None:
        with self._lock:
            return self._records.get(eval_id)

    def list_records(
        self,
        status: str | None = None,
        suite: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[EvalRecord], int]:
        """Newest-first snapshot, optionally filtered; returns
        (page, total-before-paging) like the jobs list."""
        with self._lock:
            records = list(self._records.values())
        records.reverse()
        if status is not None:
            records = [r for r in records if r.status == status]
        if suite is not None:
            records = [r for r in records if r.suite == suite]
        total = len(records)
        if limit is not None:
            records = records[:limit]
        return records, total

    def cancel(self, eval_id: str) -> tuple[EvalRecord | None, str]:
        """Cooperative cancel: a 'queued' record flips to 'cancelled' (the
        worker frees its slot on dequeue without running). Running and
        terminal records report their status so the route can 409."""
        with self._lock:
            rec = self._records.get(eval_id)
            if rec is None:
                return None, "missing"
            if rec.status == "queued":
                rec.status = "cancelled"
                rec.finished_at = time.time()
                return rec, "cancelled"
            return rec, rec.status

    def cancel_pending(self) -> list[EvalRecord]:
        """Shutdown path: queued records flip to cancelled so their futures
        never start."""
        with self._lock:
            out = [r for r in self._records.values() if r.status == "queued"]
            for rec in out:
                rec.status = "cancelled"
                rec.finished_at = time.time()
            return out

    def get_key(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            hit = self._keys.get(key)
            if hit is not None:
                self._keys.move_to_end(key)
            return hit

    def put(
        self,
        record: EvalRecord,
        key: str | None,
        fingerprint: str | None,
    ) -> None:
        with self._lock:
            self._records[record.eval_id] = record
            self._records.move_to_end(record.eval_id)
            if key is not None and fingerprint is not None:
                self._keys[key] = (fingerprint, record.eval_id)
                self._record_key[record.eval_id] = key
            while len(self._records) > self._max:
                evicted_id, _ = self._records.popitem(last=False)
                old_key = self._record_key.pop(evicted_id, None)
                if old_key is not None:
                    self._keys.pop(old_key, None)


def run_eval_record(
    record: EvalRecord,
    *,
    model: ModelFn,
    judge: ModelFn | None = None,
) -> None:
    """Worker: drive the suite runner, serialize the report, stamp finish.

    Called under the shared inflight slot by the jobs executor; the
    caller owns status transitions ('running' in, terminal out). A runner
    fault lands on the record as ``failed`` + error — never propagates
    into the worker loop.
    """
    try:
        fn = eval_runner(record.suite)
        kwargs: dict[str, Any] = {}
        if suite_accepts_judge(record.suite):
            kwargs["judge"] = judge
        report = fn(model, seed=record.seed, **kwargs)
        record.report = report_dump(report)
        record.status = "succeeded"
    except Exception as exc:  # noqa: BLE001 — worker faults land in the record
        record.error = f"{type(exc).__name__}: {exc}"
        record.status = "failed"
    finally:
        record.finished_at = time.time()


def eval_record_receipt(record: dict[str, Any]) -> dict[str, Any]:
    """Seal an eval record as ``fx1_eval_record.v1`` — the same claim as
    the completion/job receipts: these bytes were the recorded run."""
    from fx1.serve.ops_receipt import _ops_receipt

    return _ops_receipt("fx1_eval_record", "fx1_eval_record.v1", record)


# ---- eval diffs: the promotion-gate primitive -------------------------------
#
# ``GET /harness/evals/{base}/diff/{candidate}`` and the SDK twin diff two
# terminal eval records: which seeded tasks flipped, which direction the
# honesty/ship gate moved, and the by_kind counter deltas. A diff is
# *comparable* only when both records ran the same suite over the same
# bank (``eval_bank_sha256``) at the same seed — a cross-bank diff is still
# served but reads ``comparable: false`` / ``verdict: "unknown"`` rather
# than pretending the numbers mean anything.


class EvalTaskTransition(BaseModel):
    """One seeded task whose pass/fail flipped between base and candidate."""

    model_config = ConfigDict(extra="forbid")

    task: str
    base: bool
    candidate: bool
    direction: Literal["fixed", "regressed"]


class EvalDiffDelta(BaseModel):
    """A numeric leaf that moved between the two reports' ``by_kind``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    base: float
    candidate: float
    delta: float


class EvalDiff(BaseModel):
    """Deterministic comparison of two terminal eval records."""

    model_config = ConfigDict(extra="forbid")

    object: Literal["eval_diff"] = "eval_diff"
    base_eval_id: str
    candidate_eval_id: str
    base_backend: str
    candidate_backend: str
    same_suite: bool
    same_seed: bool
    same_bank: bool
    comparable: bool
    gate_base: bool | None
    gate_candidate: bool | None
    gate_transition: Literal["opened", "closed", "unchanged", "unknown"]
    tasks_fixed: list[str]
    tasks_regressed: list[str]
    tasks_only_base: list[str]
    tasks_only_candidate: list[str]
    deltas: list[EvalDiffDelta]
    verdict: Literal["improved", "regressed", "unchanged", "unknown"]


# Per-task verdict shapes by suite: run_suite-style ``results`` carry
# ``task``/``passed``; tooluse's ``outcomes`` carry ``task_id``/``completed``;
# retrieval's ``results`` carry ``question_id``/``correct``. Each pair is
# the suite's declared per-task verdict — the diffs read all three.
_TASK_VERDICT_SHAPES: tuple[tuple[str, str, str], ...] = (
    ("results", "task", "passed"),
    ("outcomes", "task_id", "completed"),
    ("results", "question_id", "correct"),
)


def _report_tasks(report: dict[str, Any]) -> dict[str, bool]:
    """task name -> passed, over the report's per-task arrays; absent → empty."""
    out: dict[str, bool] = {}
    for container, name_key, flag_key in _TASK_VERDICT_SHAPES:
        items = report.get(container)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            flag = item.get(flag_key)
            name = item.get(name_key)
            if isinstance(flag, bool) and isinstance(name, str):
                out.setdefault(name, flag)
    return out


def _numeric_leaves(node: Any, prefix: str = "") -> dict[str, float]:
    """Flatten numeric leaves of a report subtree; bools are not numbers."""
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for key in node:
            out.update(_numeric_leaves(node[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[prefix] = float(node)
    return out


def diff_eval_records(base: EvalRecord, candidate: EvalRecord) -> EvalDiff:
    """Diff two eval records into a promotion-gate verdict.

    Fails closed on the caller's side: both records must be terminal with
    a serialized report — the route 409s otherwise; this function assumes
    the contract and never invents a diff over a missing report.
    """
    base_report = base.report or {}
    cand_report = candidate.report or {}

    same_suite = base.suite == candidate.suite
    same_seed = base.seed == candidate.seed
    base_bank = base_report.get("eval_bank_sha256")
    cand_bank = cand_report.get("eval_bank_sha256")
    same_bank = isinstance(base_bank, str) and isinstance(cand_bank, str) and base_bank == cand_bank
    # Banks are seeded by construction: same suite + same seed pins the
    # bank. A stamped mismatch overrides that; absent stamps don't —
    # suites that don't emit ``eval_bank_sha256`` stay diffable while
    # ``same_bank`` still reports the stamp evidence honestly.
    bank_mismatch = (
        isinstance(base_bank, str) and isinstance(cand_bank, str) and base_bank != cand_bank
    )
    comparable = same_suite and same_seed and not bank_mismatch

    base_tasks = _report_tasks(base_report)
    cand_tasks = _report_tasks(cand_report)
    if not comparable:
        # A mismatched bank makes task-name pairing meaningless — the
        # same name in a different bank is a different task.
        base_tasks = {}
        cand_tasks = {}

    transitions: list[EvalTaskTransition] = []
    for name in sorted(base_tasks.keys() & cand_tasks.keys()):
        if base_tasks[name] != cand_tasks[name]:
            transitions.append(
                EvalTaskTransition(
                    task=name,
                    base=base_tasks[name],
                    candidate=cand_tasks[name],
                    direction="fixed" if cand_tasks[name] else "regressed",
                )
            )
    tasks_regressed = [t.task for t in transitions if t.direction == "regressed"]
    tasks_fixed = [t.task for t in transitions if t.direction == "fixed"]

    gate_base = base_report.get("honesty_gate_passed")
    gate_cand = cand_report.get("honesty_gate_passed")
    gate_b = gate_base if isinstance(gate_base, bool) else None
    gate_c = gate_cand if isinstance(gate_cand, bool) else None
    gate_transition: Literal["opened", "closed", "unchanged", "unknown"]
    if gate_b is None or gate_c is None:
        gate_transition = "unknown"
    elif gate_b == gate_c:
        gate_transition = "unchanged"
    else:
        gate_transition = "opened" if gate_c else "closed"

    base_leaves = _numeric_leaves(base_report.get("by_kind"))
    cand_leaves = _numeric_leaves(cand_report.get("by_kind"))
    deltas = [
        EvalDiffDelta(
            path=p,
            base=base_leaves[p],
            candidate=cand_leaves[p],
            delta=cand_leaves[p] - base_leaves[p],
        )
        for p in sorted(base_leaves.keys() & cand_leaves.keys())
        if base_leaves[p] != cand_leaves[p]
    ]

    if not comparable:
        verdict: Literal["improved", "regressed", "unchanged", "unknown"] = "unknown"
    elif tasks_regressed or gate_transition == "closed":
        verdict = "regressed"
    elif tasks_fixed or gate_transition == "opened":
        verdict = "improved"
    else:
        verdict = "unchanged"

    return EvalDiff(
        base_eval_id=base.eval_id,
        candidate_eval_id=candidate.eval_id,
        base_backend=base.backend,
        candidate_backend=candidate.backend,
        same_suite=same_suite,
        same_seed=same_seed,
        same_bank=same_bank,
        comparable=comparable,
        gate_base=gate_b,
        gate_candidate=gate_c,
        gate_transition=gate_transition,
        tasks_fixed=tasks_fixed,
        tasks_regressed=tasks_regressed,
        tasks_only_base=sorted(base_tasks.keys() - cand_tasks.keys()),
        tasks_only_candidate=sorted(cand_tasks.keys() - base_tasks.keys()),
        deltas=deltas,
        verdict=verdict,
    )
