"""OpenAI-compatible fine-tuning surface over the staged fx-1 Pipeline.

``POST /v1/fine_tuning/jobs`` queues a gated training run — quality gate,
frozen split, base eval, receipted training, candidate eval — against an
uploaded training file. The contract keeps the harness's fail-closed
posture:

* ``training_file`` must exist in the files store and parse as chat-format
  JSONL *before* the job queues (validation is synchronous — faster and
  more honest than OpenAI's async ``validating_files`` limbo).
* ``model`` must be a trainable fx-1 lineage name — ``fx1`` or
  ``local_fx1``. ``hosted_k3`` is a remote model and ``byok`` is the
  caller's own weights; neither is trainable through this surface.
* ``trained_tokens`` stays ``null`` — there is no tokenizer in the repo,
  and a fabricated count would be worse than none.
* Cancellation is cooperative: a queued job is cancelled immediately; a
  running job is checked at stage boundaries — an in-flight trainer call
  is never killed mid-write.
* Result artifacts (comparison, candidate eval, training receipt) are
  registered back into the files store so ``GET /v1/files/{id}/content``
  downloads them — real artifacts, real ids.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fx1.serve.backends import InferenceBackend

TRAINABLE_MODELS = ("fx1", "local_fx1")

FT_JOB_STATUS = Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]
_FT_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
_FT_EVENT_CAP = 256
_FT_EVENTS_KIND = "fine_tuning.job.event"


class FTHyperparameters(BaseModel, extra="forbid"):
    n_epochs: int | None = Field(default=None, ge=1, le=50)
    batch_size: int | None = Field(default=None, ge=1, le=512)
    learning_rate_multiplier: float | None = Field(default=None, gt=0, le=10)


class FTJobRequest(BaseModel, extra="forbid"):
    model: str = Field(min_length=1, max_length=64)
    training_file: str = Field(min_length=1, max_length=128)
    hyperparameters: FTHyperparameters | None = None
    suffix: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    validation_file: str | None = Field(default=None, min_length=1, max_length=128)
    seed: int | None = Field(default=None, ge=0)
    method: Literal["supervised"] | None = None
    metadata: dict[str, str] | None = Field(default=None, max_length=16)


class FTJobError(BaseModel, extra="forbid"):
    code: str
    message: str
    param: str | None = None


class FTJobEvent(BaseModel, extra="forbid"):
    object: Literal["fine_tuning.job.event"] = "fine_tuning.job.event"
    id: str
    created_at: int
    level: Literal["info", "warn", "error"]
    message: str
    data: dict[str, Any] | None = None


class FTJob(BaseModel, extra="forbid"):
    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    id: str
    model: str
    created_at: int
    finished_at: int | None = None
    status: FT_JOB_STATUS = "queued"
    fine_tuned_model: str | None = None
    organization_id: str = "fx1-harness"
    result_files: list[str] = Field(default_factory=list)
    training_file: str
    validation_file: str | None = None
    trained_tokens: int | None = None
    hyperparameters: FTHyperparameters = Field(default_factory=FTHyperparameters)
    seed: int | None = None
    error: FTJobError | None = None
    method: dict[str, Any] = Field(default_factory=lambda: {"type": "supervised"})
    integrations: list[Any] = Field(default_factory=list)
    estimated_finish: int | None = None
    metadata: dict[str, str] | None = None
    user_provided_suffix: str | None = None


class FTJobList(BaseModel, extra="forbid"):
    object: Literal["list"] = "list"
    data: list[FTJob]
    has_more: bool


class FTEventList(BaseModel, extra="forbid"):
    object: Literal["list"] = "list"
    data: list[FTJobEvent]
    has_more: bool


class FTJobSpec(BaseModel, extra="forbid"):
    """Everything a runner needs to execute one fine-tuning job."""

    job_id: str
    model: str
    corpus_path: Path
    val_path: Path | None
    hyperparameters: dict[str, Any]
    seed: int
    work_dir: Path
    ft_model_name: str


class FTJobOutcome(BaseModel, extra="forbid"):
    """What a runner hands back: the produced model name, the artifacts
    worth registering as result files, and an honest token count (or
    ``None`` when it cannot be measured — no fabricated numbers)."""

    fine_tuned_model: str | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    trained_tokens: int | None = None
    checkpoint: str | None = None


# Runner signature: ``(spec, *, emit, should_cancel) -> FTJobOutcome``.
# The worker calls it inside the inflight slot. ``emit`` appends a job
# event; ``should_cancel`` is the cooperative-cancel check — a runner
# that returns early because it saw True gets its job marked cancelled,
# not failed.
FTJobRunner = Callable[..., FTJobOutcome]


def validate_chat_jsonl(content: bytes, *, file_id: str) -> int:
    """Parse the upload as OpenAI chat-format JSONL (``{"messages": [...]}``).

    Returns the example count. Fails closed on the first malformed line —
    a job whose corpus can't be read never reaches the queue.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"training file {file_id!r} is not utf-8: {exc}") from exc
    roles = {"system", "user", "assistant", "tool"}
    n = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"training file {file_id!r} line {lineno} is not JSON: {exc}") from exc
        messages = obj.get("messages") if isinstance(obj, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"training file {file_id!r} line {lineno} is not a chat record "
                '(expected {"messages": [...]})'
            )
        for m in messages:
            if (
                not isinstance(m, dict)
                or m.get("role") not in roles
                or not isinstance(m.get("content"), str)
                or not m["content"]
            ):
                raise ValueError(
                    f"training file {file_id!r} line {lineno} has an invalid "
                    "message (role must be system|user|assistant|tool with a "
                    "non-empty string content)"
                )
        n += 1
    if n == 0:
        raise ValueError(f"training file {file_id!r} contains no examples")
    return n


class FTJobEntry:
    """A stored job plus its event feed and cooperative-cancel flag."""

    __slots__ = ("job", "events", "cancel", "idem_key", "body_fp")

    def __init__(self, job: FTJob, idem_key: str | None, body_fp: str) -> None:
        self.job = job
        self.events: list[FTJobEvent] = []
        self.cancel = threading.Event()
        self.idem_key = idem_key
        self.body_fp = body_fp


class FTJobStore:
    """Bounded LRU store for fine-tuning jobs — same posture as the job
    and eval stores: newest-first listing, silent eviction, idempotency
    keys replaying to the original record."""

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._max = max(1, max_entries)
        self._entries: OrderedDict[str, FTJobEntry] = OrderedDict()
        self._keys: dict[str, str] = {}

    def put(self, job: FTJob, idem_key: str | None, body_fp: str) -> FTJobEntry:
        entry = FTJobEntry(job, idem_key, body_fp)
        with self._lock:
            self._entries[job.id] = entry
            self._entries.move_to_end(job.id)
            if idem_key is not None:
                self._keys[f"ft:{idem_key}"] = job.id
            while len(self._entries) > self._max:
                _old_id, old_entry = self._entries.popitem(last=False)
                if old_entry.idem_key is not None:
                    self._keys.pop(f"ft:{old_entry.idem_key}", None)
        return entry

    def lookup_idem(self, key: str) -> FTJobEntry | None:
        with self._lock:
            job_id = self._keys.get(f"ft:{key}")
            if job_id is None:
                return None
            entry = self._entries.get(job_id)
            if entry is not None:
                self._entries.move_to_end(job_id)
            return entry

    def get(self, job_id: str) -> FTJobEntry | None:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is not None:
                self._entries.move_to_end(job_id)
            return entry

    def list_jobs(self, *, limit: int, after: str | None) -> tuple[list[FTJob], bool]:
        """Newest-first page; ``after`` is the id cursor (exclusive)."""
        with self._lock:
            entries = list(self._entries.values())
        entries.reverse()
        if after is not None:
            idx = next((i for i, e in enumerate(entries) if e.job.id == after), None)
            if idx is not None:
                entries = entries[idx + 1 :]
            else:
                entries = []
        has_more = len(entries) > limit
        return [e.job for e in entries[:limit]], has_more

    def add_event(
        self,
        job_id: str,
        level: Literal["info", "warn", "error"],
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = FTJobEvent(
            id=f"ftev-{uuid.uuid4().hex}",
            created_at=int(time.time()),
            level=level,
            message=message,
            data=data,
        )
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return
            entry.events.append(event)
            while len(entry.events) > _FT_EVENT_CAP:
                entry.events.pop(0)

    def list_events(
        self, job_id: str, *, limit: int, after: str | None
    ) -> tuple[list[FTJobEvent], bool]:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return [], False
            events = list(entry.events)
        # oldest-first, like OpenAI's events feed
        if after is not None:
            idx = next((i for i, e in enumerate(events) if e.id == after), None)
            events = events[idx + 1 :] if idx is not None else []
        has_more = len(events) > limit
        return events[:limit], has_more

    def request_cancel(self, job_id: str) -> Literal["queued", "running", "terminal", "missing"]:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return "missing"
            if entry.job.status in _FT_TERMINAL:
                return "terminal"
            if entry.job.status == "queued":
                entry.job.status = "cancelled"
                entry.job.finished_at = int(time.time())
                entry.cancel.set()
                return "queued"
            entry.cancel.set()
            return "running"

    def cancel_pending(self) -> int:
        with self._lock:
            pending = [e for e in self._entries.values() if e.job.status == "queued"]
            for e in pending:
                e.job.status = "cancelled"
                e.job.finished_at = int(time.time())
                e.cancel.set()
            running = [e for e in self._entries.values() if e.job.status == "running"]
            for e in running:
                e.cancel.set()
        return len(pending) + len(running)


def default_ft_runner(
    resolver: Callable[..., InferenceBackend],
) -> Callable[..., FTJobOutcome]:
    """Wire the staged Pipeline as the runner.

    Base and candidate model functions resolve through the app's backend
    resolver (``local_fx1`` for the weights lineage); the trainer is the
    pipeline's own default, which raises with GPU setup instructions when
    no trainer is configured — that failure lands on the job record, not
    in a fabricated success.
    """

    def _runner(
        spec: FTJobSpec,
        *,
        emit: Callable[[str, str, dict[str, Any] | None], None],
        should_cancel: Callable[[], bool],
    ) -> FTJobOutcome:
        from fx1.train.config import LadderStage, TrainConfig  # noqa: PLC0415
        from fx1.train.pipeline import Pipeline  # noqa: PLC0415

        hp = spec.hyperparameters
        config = TrainConfig(
            run_name=spec.ft_model_name,
            stage=LadderStage.PROXY,
            corpus_jsonl=str(spec.corpus_path),
            eval_results_json=str(spec.work_dir / "eval_base.json"),
            epochs=int(hp["n_epochs"]) if hp.get("n_epochs") else 1,
            learning_rate=1e-4
            * (
                float(hp["learning_rate_multiplier"]) if hp.get("learning_rate_multiplier") else 1.0
            ),
            estimated_nodes=1,
            estimated_gpu_hours=0.1,
            estimated_cost_usd=0.0,
        )
        pipe = Pipeline(config, spec.work_dir)
        emit("info", "quality gate: dedup + decontamination + frozen split", None)
        pipe.run_quality_gate()
        if should_cancel():
            return FTJobOutcome()
        base_backend = resolver("local_fx1", None, None, None)

        def base_fn(messages: list[dict[str, str]]) -> str:
            return base_backend.complete(messages)

        emit("info", "base eval on the canonical bank", None)
        pipe.run_eval_base(base_fn)
        if should_cancel():
            return FTJobOutcome()
        emit("info", "training run (receipted)", None)
        checkpoint = pipe.run_training(seed=spec.seed)
        if should_cancel():
            return FTJobOutcome()
        from fx1.serve.backends import LocalFx1Backend  # noqa: PLC0415

        cand_backend = LocalFx1Backend(checkpoint_dir=checkpoint)

        def cand_fn(messages: list[dict[str, str]]) -> str:
            return cand_backend.complete(messages)

        emit("info", "candidate eval + ship-gate comparison", None)
        pipe.run_eval_candidate(cand_fn)
        artifacts = {
            name: Path(path) for name, path in pipe.state.artifacts.items() if Path(path).is_file()
        }
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts=artifacts,
            trained_tokens=None,
            checkpoint=str(checkpoint),
        )

    return _runner
