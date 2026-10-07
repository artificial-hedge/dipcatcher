"""OpenAI-compatible fine-tuning surface over the staged fx-1 Pipeline.

``POST /v1/fine_tuning/jobs`` queues a gated training run — quality gate,
frozen split, base eval, receipted training, candidate eval — against an
uploaded training file. The trainer is real: ``fx1.train.tiny_lm``
fine-tunes the base fx-1 checkpoint weights on the job corpus and writes
a servable checkpoint under the job's work dir, so a succeeded job's
``ft:`` model name actually serves trained weights through
``LocalWeightsEngine``. The contract keeps the harness's fail-closed
posture:

* ``training_file`` must exist in the files store and parse as chat-format
  JSONL *before* the job queues (validation is synchronous — faster and
  more honest than OpenAI's async ``validating_files`` limbo).
* ``model`` must be a trainable fx-1 lineage name — ``fx1`` or
  ``local_fx1``. ``hosted_k3`` is a remote model and ``byok`` is the
  caller's own weights; neither is trainable through this surface.
* ``trained_tokens`` is a real measured count — ``steps * seq_len`` of
  corpus windows consumed by the trainer — read back from the produced
  checkpoint's manifest, not fabricated.
* Cancellation is cooperative: a queued job is cancelled immediately; a
  running job is checked at stage boundaries and between optimization
  steps — an in-flight tensor write is never killed mid-write.
* Pause/resume shares that contract: ``POST .../pause`` parks a queued
  job before it starts and a running job at the next stage boundary
  (status ``paused`` — non-terminal); ``POST .../resume`` restores it.
  A paused job still honours cancel and drain. On restart a job replayed
  as ``paused`` recovers to ``failed`` like any other non-terminal state.
* Eval stages produce evidence, not ship gates: the fixture-scale model's
  real bank scores are sealed into ``eval_base.json`` /
  ``eval_candidate.json`` / ``comparison.json`` result files, and the
  measured pair is stamped onto the produced ``modelcard.json`` (the
  card's ``eval_delta`` is real measured evidence — the fixture
  placeholder never ships on a job-produced card) plus sealed into
  ``eval_receipt.json``. Serving the minted ``ft:`` name is gated the
  same way the base checkpoint is — the card must load and pass
  ``ship_eligible``, and the engine verifies the weights sha pin. A
  candidate whose measured delta fails the gate is honestly refused at
  serve; the gate's verdict reflects reality, not a fixture constant.
* Result artifacts (comparison, candidate eval, training receipt, the
  checkpoint's weights/manifest/card) are registered back into the files
  store so ``GET /v1/files/{id}/content`` downloads them — real
  artifacts, real ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from fx1.serve.backends import InferenceBackend
from fx1.serve.journal import JobJournal, _ClaimLocks
from fx1.serve.webhooks import check_callback_url

TRAINABLE_MODELS = ("fx1", "local_fx1")

FT_JOB_STATUS = Literal[
    "validating_files", "queued", "running", "succeeded", "failed", "cancelled", "paused"
]
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
    # fx1 extension: terminal-state webhook — the finished job record is
    # POSTed to ``callback_url`` on succeeded/failed/cancelled, signed with
    # ``callback_secret`` via the X-Fx1-Webhook-* headers (never echoed).
    callback_url: str | None = None
    callback_secret: str | None = None

    @field_validator("callback_url")
    @classmethod
    def _callback_url_http(cls, v: str | None) -> str | None:
        return check_callback_url(v)

    @model_validator(mode="after")
    def _callback_secret_needs_url(self) -> FTJobRequest:
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError("callback_secret requires callback_url")
        return self


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
    # fx1 extension — terminal webhook bookkeeping (same contract as the
    # /harness/* jobs): the finished record is POSTed to callback_url;
    # delivery state is readable on the job itself.
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    callback_attempts: int = 0
    callback_error: str | None = None
    _callback_secret: str | None = PrivateAttr(default=None)
    _callback_fired: bool = PrivateAttr(default=False)
    _callback_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)


class FTJobList(BaseModel, extra="forbid"):
    object: Literal["list"] = "list"
    data: list[FTJob]
    has_more: bool


class FTEventList(BaseModel, extra="forbid"):
    object: Literal["list"] = "list"
    data: list[FTJobEvent]
    has_more: bool


class FTJobCheckpoint(BaseModel, extra="forbid"):
    """OpenAI's ``fine_tuning.job.checkpoint`` — one registered model
    artifact a job produced. The harness records the model name and its
    checkpoint dir, not intermediate step metrics — ``step_number`` and
    ``metrics`` stay empty rather than fabricating numbers."""

    object: Literal["fine_tuning.job.checkpoint"] = "fine_tuning.job.checkpoint"
    id: str
    created_at: int
    fine_tuned_model_checkpoint: str
    step_number: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class FTJobCheckpointList(BaseModel, extra="forbid"):
    """OpenAI's checkpoints list envelope — ``first_id``/``last_id`` are
    the page's edge ids (``after`` cursors), null on an empty page."""

    object: Literal["list"] = "list"
    data: list[FTJobCheckpoint]
    first_id: str | None = None
    last_id: str | None = None
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


# Runner signature: ``(spec, *, emit, should_cancel, pause_gate) ->
# FTJobOutcome``. The worker calls it inside the inflight slot. ``emit``
# appends a job event; ``should_cancel`` is the cooperative-cancel check
# — a runner that returns early because it saw True gets its job marked
# cancelled, not failed. ``pause_gate`` blocks while the job is paused
# and returns True when a cancel landed while parked — call it between
# stages; a runner that returns early on True unwinds to ``cancelled``.
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
    """A stored job plus its event feed, cooperative-cancel flag, and
    pause gate. ``resume`` is set while unpaused — a worker parks on
    ``resume.wait()`` at stage boundaries; ``paused_from`` remembers the
    status resume restores (``queued`` parked pre-start vs ``running``
    parked mid-pipeline)."""

    __slots__ = ("job", "events", "cancel", "pause", "resume", "paused_from", "idem_key", "body_fp")

    def __init__(self, job: FTJob, idem_key: str | None, body_fp: str) -> None:
        self.job = job
        self.events: list[FTJobEvent] = []
        self.cancel = threading.Event()
        self.pause = threading.Event()
        self.resume = threading.Event()
        self.resume.set()
        self.paused_from: Literal["queued", "running"] | None = None
        self.idem_key = idem_key
        self.body_fp = body_fp


class FTJobStore:
    """Bounded LRU store for fine-tuning jobs — same posture as the job
    and eval stores: newest-first listing, silent eviction, idempotency
    keys replaying to the original record.

    With a ``JobJournal`` bound (``--state-dir``) every transition,
    event, model registration, cancel, and eviction appends to a
    hash-chained ``ft_jobs.jsonl``; boot replays the chain: terminal
    jobs return as-was with their event feed, jobs still
    ``queued``/``running`` at the crash recover as ``failed`` with a
    restart-explaining error (specs aren't journaled — nothing is
    silently re-run), ``ft:`` idempotency keys still resolve, and the
    ``ft:`` model registry rebuilds minus refs whose producing job was
    evicted. ``callback_secret`` never touches disk, so a recovered job
    keeps ``callback_url`` for audit but cannot deliver post-restart."""

    def __init__(self, max_entries: int, journal: JobJournal | None = None) -> None:
        self._lock = threading.Lock()
        self._max = max(1, max_entries)
        self._claims = _ClaimLocks(max_entries)
        self._entries: OrderedDict[str, FTJobEntry] = OrderedDict()
        self._keys: dict[str, str] = {}
        # fine-tuned model registry: ft:<model>:<suffix>:<job12> -> ref
        # (``{id, job_id, checkpoint, created}``). A registration survives
        # only while its producing job does — evicting the job drops the
        # card so a listed model can never point at forgotten provenance.
        self._models: dict[str, dict[str, Any]] = {}
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            now = int(time.time())
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._drop(str(evict))
                if "ft_model" in payload:
                    ref = payload["ft_model"]
                    self._models[str(ref["id"])] = dict(ref)
                if "ft_model_delete" in payload:
                    self._models.pop(str(payload["ft_model_delete"]), None)
                if "ft_event" in payload:
                    ev = payload["ft_event"]
                    host = self._entries.get(str(ev["job_id"]))
                    if host is not None:
                        host.events.append(FTJobEvent.model_validate(ev["event"]))
                        while len(host.events) > _FT_EVENT_CAP:
                            host.events.pop(0)
                    continue
                if "ft_job" not in payload:
                    continue
                job = FTJob.model_validate(payload["ft_job"])
                # Signing secrets are not journaled; recovered records never re-deliver.
                job._callback_fired = True
                key = payload.get("key")
                fp = payload.get("fp")
                entry = FTJobEntry(
                    job,
                    str(key) if key is not None else None,
                    str(fp) if fp is not None else "",
                )
                for ev in payload.get("events") or ():
                    entry.events.append(FTJobEvent.model_validate(ev))
                self._entries[job.id] = entry
                self._entries.move_to_end(job.id)
                if entry.idem_key is not None:
                    self._keys[f"ft:{entry.idem_key}"] = job.id
            for entry in self._entries.values():
                if entry.job.status not in _FT_TERMINAL:
                    entry.job.status = "failed"
                    entry.job.finished_at = now
                    entry.job.error = FTJobError(
                        code="job_failed",
                        message="process restarted before the job reached a terminal state",
                    )
            # a model card can never outlive its producing job — drop refs
            # whose job was evicted mid-journal
            for mname, mref in list(self._models.items()):
                if str(mref["job_id"]) not in self._entries:
                    del self._models[mname]
            self._compact_locked()

    def _drop(self, job_id: str) -> None:
        """Evict one entry plus its idem key and any model cards it minted."""
        old = self._entries.pop(job_id, None)
        if old is not None and old.idem_key is not None:
            self._keys.pop(f"ft:{old.idem_key}", None)
        for mname, mref in list(self._models.items()):
            if str(mref["job_id"]) == job_id:
                del self._models[mname]

    def _record(self, entry: FTJobEntry) -> dict[str, Any]:
        return {
            "ft_job": entry.job.model_dump(mode="json"),
            "events": [e.model_dump(mode="json") for e in entry.events],
            "key": entry.idem_key,
            "fp": entry.body_fp,
        }

    def _compact_locked(self) -> None:
        """Rewrite the journal with only the live state — boot post-replay
        so dead history and torn tails don't accumulate."""
        if self._journal is not None:
            live = [self._record(e) for e in self._entries.values()]
            live.extend({"ft_model": ref} for ref in self._models.values())
            self._journal.compact(live)

    def mark(self, entry: FTJobEntry) -> None:
        """Journal a status transition made outside the store (the worker
        mutates ``entry.job`` in place; this makes each hop durable)."""
        if self._journal is not None:
            with self._lock:
                # A bounded-store eviction wins over a late worker mark.
                # Journaling an entry no longer present would resurrect the
                # job (and potentially its model card) after restart.
                if self._entries.get(entry.job.id) is entry:
                    self._journal.append(self._record(entry))

    def put(self, job: FTJob, idem_key: str | None, body_fp: str) -> FTJobEntry:
        entry = FTJobEntry(job, idem_key, body_fp)
        with self._lock:
            # Determine the bounded-store transition without publishing it.
            # The journal must acknowledge the put+evictions first; a failed
            # append leaves the live store and its existing registry intact.
            order = [job_id for job_id in self._entries if job_id != job.id]
            evicted = order[: max(0, len(order) + 1 - self._max)]
            if self._journal is not None:
                payload = self._record(entry)
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)
            for old_id in evicted:
                self._drop(old_id)
            self._entries[job.id] = entry
            self._entries.move_to_end(job.id)
            if idem_key is not None:
                self._keys[f"ft:{idem_key}"] = job.id
        return entry

    def register_model(self, name: str, *, job_id: str, checkpoint: str, created: int) -> None:
        """Bind an ``ft:`` model name to its producing job + checkpoint.
        Only called on succeeded jobs with a real checkpoint — a card is
        never minted for a model the harness cannot serve."""
        with self._lock:
            # A worker finishing after its job was evicted must not publish a
            # provenance-free model.  The eviction is the terminal verdict.
            if job_id not in self._entries:
                return
            ref = {
                "id": name,
                "job_id": job_id,
                "checkpoint": checkpoint,
                "created": created,
            }
            if self._journal is not None:
                self._journal.append({"ft_model": dict(ref)})
            self._models[name] = ref

    def get_model(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            return self._models.get(name)

    def unregister_model(self, name: str) -> dict[str, Any] | None:
        """Remove an ``ft:`` model registration — the ``DELETE
        /v1/models/{id}`` store op. Returns the dropped card or None when
        the name was never registered; the deletion journals so a restart
        never resurrects a deleted model."""
        with self._lock:
            ref = self._models.get(name)
            if ref is not None and self._journal is not None:
                self._journal.append({"ft_model_delete": name})
            return self._models.pop(name, None)

    def models(self) -> list[dict[str, Any]]:
        """All registered ft models, sorted by id (stable list order)."""
        with self._lock:
            return [dict(self._models[k]) for k in sorted(self._models)]

    def checkpoints_for(
        self, job_id: str, *, limit: int, after: str | None
    ) -> tuple[list[FTJobCheckpoint], bool]:
        """Checkpoints a job registered, oldest-first — one entry per
        model card the job produced (``GET
        /v1/fine_tuning/jobs/{id}/checkpoints``). Deleted registrations
        drop off the listing: a tombstone never fabricates history for a
        model that is gone."""
        with self._lock:
            cards = [m for m in self._models.values() if m["job_id"] == job_id]
        cards.sort(key=lambda m: (int(m["created"]), str(m["id"])))
        items = [
            FTJobCheckpoint(
                id="ftckpt-" + hashlib.sha256(f"{job_id}:{m['id']}".encode()).hexdigest()[:24],
                created_at=int(m["created"]),
                fine_tuned_model_checkpoint=str(m["id"]),
            )
            for m in cards
        ]
        if after is not None:
            idx = next((i for i, c in enumerate(items) if c.id == after), None)
            items = items[idx + 1 :] if idx is not None else []
        has_more = len(items) > limit
        return items[:limit], has_more

    def checkpoint_for(self, name: str) -> str | None:
        """The checkpoint dir an ``ft:`` name resolves to — the piece
        ``_resolve_openai_link`` needs to route the request at the
        fine-tuned weights instead of the default link."""
        ref = self.get_model(name)
        return str(ref["checkpoint"]) if ref is not None else None

    def async_claim_lock(self, key: str | None) -> AbstractAsyncContextManager[None]:
        """Per-key submit claim — a keyed retry waits for an in-flight
        twin's ``lookup_idem`` + ``put`` instead of double-executing
        next to it (same contract as the job/eval stores)."""
        return self._claims.ahold(key)

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
            if self._journal is not None:
                self._journal.append(
                    {"ft_event": {"job_id": job_id, "event": event.model_dump(mode="json")}}
                )

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

    def request_cancel(
        self, job_id: str
    ) -> Literal["queued", "running", "paused", "terminal", "missing"]:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return "missing"
            if entry.job.status in _FT_TERMINAL:
                return "terminal"
            if entry.job.status == "paused":
                # like a queued cancel: the terminal write lands now; any
                # parked worker wakes on resume and exits through its
                # cancel check without re-emitting the event
                entry.job.status = "cancelled"
                entry.job.finished_at = int(time.time())
                entry.paused_from = None
                entry.cancel.set()
                entry.resume.set()
                if self._journal is not None:
                    self._journal.append(self._record(entry))
                return "paused"
            if entry.job.status == "queued":
                entry.job.status = "cancelled"
                entry.job.finished_at = int(time.time())
                entry.cancel.set()
                if self._journal is not None:
                    self._journal.append(self._record(entry))
                return "queued"
            entry.cancel.set()
            return "running"

    def request_pause(
        self, job_id: str
    ) -> Literal["queued", "running", "paused", "terminal", "missing"]:
        """``POST .../pause``: queued parks before start, running parks at
        the next stage boundary — both report ``paused`` immediately. A
        second pause is idempotent (returns ``paused``)."""
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return "missing"
            if entry.job.status in _FT_TERMINAL:
                return "terminal"
            if entry.job.status == "paused":
                return "paused"
            paused_from: Literal["queued", "running"] = (
                "running" if entry.job.status == "running" else "queued"
            )
            entry.paused_from = paused_from
            entry.job.status = "paused"
            entry.pause.set()
            entry.resume.clear()
            if self._journal is not None:
                self._journal.append(self._record(entry))
            return paused_from

    def request_resume(
        self, job_id: str
    ) -> Literal["queued", "running", "terminal", "missing", "not_paused"]:
        """``POST .../resume``: restores the status pause captured and
        opens the gate — a parked worker proceeds on its next check."""
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return "missing"
            if entry.job.status in _FT_TERMINAL:
                return "terminal"
            if entry.job.status != "paused":
                return "not_paused"
            restored = entry.paused_from or "queued"
            entry.job.status = restored
            entry.paused_from = None
            entry.pause.clear()
            entry.resume.set()
            if self._journal is not None:
                self._journal.append(self._record(entry))
            return restored

    def cancel_pending(self) -> list[FTJob]:
        """Mass-cancel on drain: queued jobs flip to ``cancelled`` outright
        (their workers never started — the returned records owe any
        terminal webhook); running jobs only get the flag — their workers
        own the terminal transition."""
        with self._lock:
            pending = [
                e
                for e in self._entries.values()
                if e.job.status == "queued"
                or (e.job.status == "paused" and e.paused_from == "queued")
            ]
            for e in pending:
                e.job.status = "cancelled"
                e.job.finished_at = int(time.time())
                e.cancel.set()
                e.resume.set()
                if self._journal is not None:
                    self._journal.append(self._record(e))
            running = [
                e
                for e in self._entries.values()
                if e.job.status == "running"
                or (e.job.status == "paused" and e.paused_from == "running")
            ]
            for e in running:
                e.cancel.set()
                e.resume.set()
        return [e.job for e in pending]


def _base_checkpoint_dir() -> Path:
    """Resolve the base fx-1 weights every fine-tune starts from.

    ``FX1_CHECKPOINT_DIR`` is authoritative when set (it must name a
    complete checkpoint dir — a bad pin fails loudly rather than silently
    falling through). Otherwise the committed ``artifacts/fx1_tiny_lm``
    fixture is the in-repo base for both trainable names.
    """
    candidates: list[Path] = []
    env = os.environ.get("FX1_CHECKPOINT_DIR")
    if env:
        pinned = Path(env)
        if all(
            (pinned / member).is_file()
            for member in ("weights.safetensors", "weights.manifest.json", "modelcard.json")
        ):
            return pinned
        raise RuntimeError(
            f"FX1_CHECKPOINT_DIR={pinned} is not a complete checkpoint dir "
            "(needs weights.safetensors + weights.manifest.json + modelcard.json)"
        )
    candidates.append(Path(__file__).resolve().parents[3] / "artifacts" / "fx1_tiny_lm")
    candidates.append(Path.cwd() / "artifacts" / "fx1_tiny_lm")
    for cand in candidates:
        if all(
            (cand / member).is_file()
            for member in ("weights.safetensors", "weights.manifest.json", "modelcard.json")
        ):
            return cand
    raise RuntimeError(
        "no base fx-1 checkpoint resolvable for fine-tuning — set "
        "FX1_CHECKPOINT_DIR or restore artifacts/fx1_tiny_lm"
    )


def default_ft_runner(
    resolver: Callable[..., InferenceBackend],
) -> Callable[..., FTJobOutcome]:
    """Wire the staged Pipeline over the real in-repo tiny-LM trainer.

    The job corpus flows through the pipeline's real gates — dedup,
    eval-contamination, frozen split — then ``fx1.train.tiny_lm``
    fine-tunes the (sha-verified) base checkpoint weights on it and writes
    a servable checkpoint into the job's work dir: ``weights.safetensors``
    + ``weights.manifest.json`` + a ``modelcard.json`` naming the minted
    ``ft:`` model. Base/candidate evals run in-process on real weights and
    are sealed as evidence (a fixture-scale model's bank scores are honest
    data, not a pass/fail gate); the ship gate for the minted model is the
    card contract at serve time. ``resolver`` is kept for the runner
    contract's shape — this lane evals in-process, not through backends.
    """
    del resolver  # the evidence evals run in-process on real weights

    def _runner(
        spec: FTJobSpec,
        *,
        emit: Callable[[str, str, dict[str, Any] | None], None],
        should_cancel: Callable[[], bool],
        pause_gate: Callable[[], bool],
    ) -> FTJobOutcome:
        from fx1.eval.bank import DEFAULT_BANK  # noqa: PLC0415
        from fx1.eval.suite import run_suite  # noqa: PLC0415
        from fx1.modelcard import EvalDelta  # noqa: PLC0415
        from fx1.serve.local_engine import LocalWeightsEngine  # noqa: PLC0415
        from fx1.train.config import LadderStage, TrainConfig  # noqa: PLC0415
        from fx1.train.pipeline import Pipeline, Stage  # noqa: PLC0415
        from fx1.train.receipts import loads_receipt  # noqa: PLC0415
        from fx1.train.tiny_lm import (  # noqa: PLC0415
            FT_LR_BASE,
            make_tiny_lm_trainer,
            sha256_file,
            stamp_measured_eval_delta,
        )
        from quant_fund.utils.hashing import (  # noqa: PLC0415
            canonical_json_bytes,
            hash_bytes,
        )
        from quant_fund.utils.reproducibility import git_revision  # noqa: PLC0415

        class _EvidencePipeline(Pipeline):
            """Eval stages seal measured evidence instead of ship-gating.

            The fixture-scale trainer produces a ~31k-param byte LM whose
            real bank scores sit near zero — the standard ``run_eval_base``
            / ``run_eval_candidate`` refusals would make every job fail
            honestly *before* producing anything. This lane's contract is
            different: run the same evals on the same bank, seal the same
            artifacts (eval_base.json, eval_candidate.json,
            comparison.json), and keep the receipt-pinned integrity checks
            — while ship-gating the minted model stays where it belongs,
            at card validation on serve.
            """

            def run_eval_base_evidence(self, model_fn: Any) -> dict[str, Any]:
                summary = run_suite(model_fn, list(DEFAULT_BANK))
                out = self._write("eval_base.json", dict(summary))
                self._advance(Stage.EVAL_BASE, eval_base=str(out))
                return dict(summary)

            def run_eval_candidate_evidence(self, candidate_fn: Any) -> dict[str, Any]:
                base_path = Path(self.state.artifacts["eval_base"])
                receipt_path = self.state.artifacts.get("training_receipt")
                if receipt_path:
                    receipt = loads_receipt(receipt_path)
                    actual = hashlib.sha256(base_path.read_bytes()).hexdigest()
                    if actual != receipt.eval_base_sha256:
                        raise RuntimeError(
                            "eval_base.json no longer matches the hash pinned "
                            "in the training receipt; re-run the base eval "
                            "rather than re-baselining on tampered or stale "
                            "results"
                        )
                base_summary = json.loads(base_path.read_text(encoding="utf-8"))
                cand_summary = run_suite(candidate_fn, list(DEFAULT_BANK))
                cand_out = self._write("eval_candidate.json", dict(cand_summary))
                if not isinstance(base_summary, dict) or not isinstance(
                    base_summary.get("results"), list
                ):
                    raise RuntimeError("eval_base.json is not a suite summary")
                base_bank = base_summary.get("eval_bank_sha256")
                if not base_bank or base_bank != cand_summary.get("eval_bank_sha256"):
                    raise RuntimeError(
                        "base and candidate eval summaries pin different eval "
                        "banks; comparing them would certify nothing"
                    )
                base_by_name: dict[str, Any] = {}
                for r in base_summary["results"]:
                    name = r.get("task")
                    if not isinstance(name, str) or not name:
                        raise RuntimeError("eval_base.json contains a result without a task name")
                    base_by_name[name] = r
                cand_by_name = {str(r.get("task")): r for r in cand_summary.results}
                comparison = self._compare_by_kind(base_by_name, cand_by_name, kind="domain")
                if not comparison:
                    raise RuntimeError(
                        "no domain tasks in the eval bank — nothing to compare "
                        "the candidate against"
                    )
                general = self._compare_by_kind(base_by_name, cand_by_name, kind="general")
                if general:
                    comparison["general"] = general
                comp_out = self._write("comparison.json", comparison)
                self._advance(
                    Stage.EVAL_CANDIDATE,
                    eval_candidate=str(cand_out),
                    comparison=str(comp_out),
                )
                return dict(cand_summary)

            def run_card_stamp(
                self,
                checkpoint: Path,
                base_summary: dict[str, Any],
                cand_summary: dict[str, Any],
            ) -> EvalDelta:
                """CARD: replace the pending eval_delta with the measured pair.

                The trainer writes the ``eval_pending_delta`` sentinel — a
                card that fails the ship gate closed — so the checkpoint on
                disk can never claim eligibility it has not earned. This is
                the only place a produced card gains real eval numbers: the
                measured domain/general pass rates from the sealed
                ``comparison.json`` plus the candidate's measured honesty
                gate. A candidate that measured worse carries an honest
                negative delta and a ``ship_eligible=False`` card — the
                serve-time gate reads exactly this verdict. The sealed
                ``eval_receipt.json`` (eval name, task counts, bank sha,
                seeds, every measured number) lands in the job's
                ``result_files`` so the stamped card is auditable.
                """
                comparison = json.loads(
                    Path(self.state.artifacts["comparison"]).read_text(encoding="utf-8")
                )
                general = comparison.get("general")
                if not isinstance(general, dict):
                    general = {}
                measured = EvalDelta(
                    domain_pass_rate_base=float(comparison["base_pass_rate"]),
                    domain_pass_rate_candidate=float(comparison["candidate_pass_rate"]),
                    general_pass_rate_base=float(general.get("base_pass_rate", 0.0)),
                    general_pass_rate_candidate=float(general.get("candidate_pass_rate", 0.0)),
                    honesty_gate_candidate=bool(cand_summary.get("honesty_gate_passed")),
                )
                bank_sha = str(cand_summary.get("eval_bank_sha256") or "")
                by_kind = cand_summary.get("by_kind") or {}
                n_tasks = len(cand_summary.get("results") or [])
                n_by_kind = {
                    str(k): int(v.get("total") or 0)
                    for k, v in by_kind.items()
                    if isinstance(v, dict)
                }
                base_results = base_summary.get("results") or []
                cand_results = cand_summary.get("results") or []
                receipt: dict[str, Any] = {
                    "kind": "fx1_ft_eval",
                    "schema": "fx1_ft_eval.v1",
                    "git_revision": git_revision(),
                    "data_label": "SYNTHETIC",
                    "research_only": True,
                    "live_pnl_claim": False,
                    "job_id": spec.job_id,
                    "fine_tuned_model": spec.ft_model_name,
                    "eval": {
                        "suite": "fx1.eval.suite.run_suite",
                        "bank": "DEFAULT_BANK",
                        "eval_bank_sha256": bank_sha,
                        "n_tasks": n_tasks,
                        "n_by_kind": n_by_kind,
                        "max_tokens": 24,
                        "job_seed": spec.seed,
                        "comparison_seed": 7,
                        "base_checkpoint": str(base_ckpt),
                        "base_weights_sha256": base_weights_sha,
                        "candidate_checkpoint": str(checkpoint),
                    },
                    "measured": {
                        "domain_pass_rate_base": measured.domain_pass_rate_base,
                        "domain_pass_rate_candidate": measured.domain_pass_rate_candidate,
                        "domain_delta": float(comparison["delta"]),
                        "domain_delta_ci_low": float(comparison["delta_ci_low"]),
                        "domain_delta_ci_high": float(comparison["delta_ci_high"]),
                        "domain_mcnemar_statistic": float(comparison["mcnemar_statistic"]),
                        "general_pass_rate_base": measured.general_pass_rate_base,
                        "general_pass_rate_candidate": (measured.general_pass_rate_candidate),
                        "general_delta": float(general.get("delta", 0.0)),
                        "honesty_gate_base": bool(base_summary.get("honesty_gate_passed")),
                        "honesty_gate_candidate": measured.honesty_gate_candidate,
                        "honesty_violations_candidate": sorted(
                            str(t) for t in (cand_summary.get("honesty_violations") or [])
                        ),
                        "base_tasks_passed": sum(1 for r in base_results if r.get("passed")),
                        "candidate_tasks_passed": sum(1 for r in cand_results if r.get("passed")),
                        "ship_eligible": measured.ship_eligible,
                    },
                    "interpretation": (
                        "the ft job's eval_delta is measured: "
                        "fx1.eval.suite.run_suite scored base and candidate "
                        "on the same DEFAULT_BANK, and the card carries the "
                        f"result (ship_eligible={measured.ship_eligible})"
                    ),
                }
                receipt["receipt_sha256"] = hash_bytes(canonical_json_bytes(receipt))
                receipt_out = self._write("eval_receipt.json", receipt)
                # the card itself ships as the ``checkpoint/modelcard.json``
                # member below — registering it here too would duplicate the
                # file under the same result name
                stamp_measured_eval_delta(
                    checkpoint,
                    measured,
                    evidence_note=(
                        "eval_delta measured by fx1.eval.suite.run_suite over "
                        f"DEFAULT_BANK ({n_tasks} seeded tasks "
                        f"{n_by_kind}; eval_bank_sha256 {bank_sha[:16]}\u2026; "
                        f"job seed {spec.seed}; max_tokens 24; "
                        f"ship_eligible={measured.ship_eligible}); evidence "
                        "sealed in the job's eval_receipt.json result file"
                    ),
                )
                self._advance(Stage.CARD, eval_receipt=str(receipt_out))
                return measured

        hp = spec.hyperparameters
        config = TrainConfig(
            run_name=spec.ft_model_name,
            stage=LadderStage.PROXY,
            corpus_jsonl=str(spec.corpus_path),
            eval_results_json=str(spec.work_dir / "eval_base.json"),
            epochs=int(hp["n_epochs"]) if hp.get("n_epochs") else 1,
            learning_rate=FT_LR_BASE
            * (
                float(hp["learning_rate_multiplier"]) if hp.get("learning_rate_multiplier") else 1.0
            ),
            estimated_nodes=1,
            estimated_gpu_hours=0.1,
            estimated_cost_usd=0.0,
        )
        base_ckpt = _base_checkpoint_dir()
        base_weights_sha = sha256_file(base_ckpt / "weights.safetensors")
        pipe = _EvidencePipeline(
            config,
            spec.work_dir,
            trainer=make_tiny_lm_trainer(
                seed=spec.seed,
                fine_tuned_model=spec.ft_model_name,
                job_id=spec.job_id,
                corpus_label=spec.corpus_path.name,
                base_checkpoint=base_ckpt,
                base_weights_sha256=base_weights_sha,
                lr=config.learning_rate,
                val_path=spec.val_path,
                should_continue=lambda: not (should_cancel() or pause_gate()),
            ),
        )
        emit("info", "quality gate: dedup + decontamination + frozen split", None)
        pipe.run_quality_gate()
        emit(
            "info",
            "quality gate passed",
            {
                "corpus_kept": pipe.state.metrics.get("corpus_kept"),
                "val_count": pipe.state.metrics.get("val_count"),
            },
        )
        if should_cancel() or pause_gate():
            return FTJobOutcome()

        # the eval path measures weights regardless of the card's own
        # ship-gate claim — the base card is the fixture placeholder and
        # the candidate card is the eval-pending sentinel at this point
        base_engine = LocalWeightsEngine(base_ckpt, enforce_ship_gate=False)

        def base_fn(messages: list[dict[str, str]]) -> str:
            return base_engine.complete_messages(messages, max_tokens=24).text

        emit("info", "base eval on the canonical bank (measured evidence)", None)
        base_summary = pipe.run_eval_base_evidence(base_fn)
        emit(
            "info",
            "base eval complete (evidence, not a ship gate)",
            _eval_event_data(base_summary),
        )
        if should_cancel() or pause_gate():
            return FTJobOutcome()

        emit("info", "training run (receipted): tiny-LM fine-tune of the base weights", None)
        checkpoint = pipe.run_training(seed=spec.seed)
        manifest = json.loads((checkpoint / "weights.manifest.json").read_text(encoding="utf-8"))
        training_stats = manifest.get("training") or {}
        emit(
            "info",
            "training complete",
            {
                "steps": training_stats.get("steps"),
                "trained_tokens": training_stats.get("trained_tokens"),
                "final_loss": training_stats.get("final_loss"),
                "val_loss": training_stats.get("val_loss"),
            },
        )
        if should_cancel() or pause_gate():
            return FTJobOutcome()

        cand_engine = LocalWeightsEngine(checkpoint, enforce_ship_gate=False)

        def cand_fn(messages: list[dict[str, str]]) -> str:
            return cand_engine.complete_messages(messages, max_tokens=24).text

        emit("info", "candidate eval + measured comparison", None)
        if pause_gate():
            return FTJobOutcome()
        cand_summary = pipe.run_eval_candidate_evidence(cand_fn)
        emit(
            "info",
            "candidate eval complete (evidence, not a ship gate)",
            _eval_event_data(cand_summary),
        )
        if should_cancel() or pause_gate():
            return FTJobOutcome()

        measured = pipe.run_card_stamp(checkpoint, base_summary, cand_summary)
        emit(
            "info" if measured.ship_eligible else "warn",
            (
                "eval delta measured + stamped on the model card"
                if measured.ship_eligible
                else "eval delta measured: card is not ship-eligible — the "
                "gate will honestly refuse to serve this checkpoint"
            ),
            {
                "domain_pass_rate_base": measured.domain_pass_rate_base,
                "domain_pass_rate_candidate": measured.domain_pass_rate_candidate,
                "general_pass_rate_base": measured.general_pass_rate_base,
                "general_pass_rate_candidate": measured.general_pass_rate_candidate,
                "honesty_gate_candidate": measured.honesty_gate_candidate,
                "ship_eligible": measured.ship_eligible,
            },
        )
        artifacts = {
            name: Path(path) for name, path in pipe.state.artifacts.items() if Path(path).is_file()
        }
        for member in ("weights.safetensors", "weights.manifest.json", "modelcard.json"):
            member_path = checkpoint / member
            if member_path.is_file():
                artifacts[f"checkpoint/{member}"] = member_path
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts=artifacts,
            trained_tokens=(
                int(training_stats["trained_tokens"])
                if training_stats.get("trained_tokens") is not None
                else None
            ),
            checkpoint=str(checkpoint),
        )

    return _runner


def _eval_event_data(summary: dict[str, Any]) -> dict[str, Any]:
    """The honest eval numbers a job event can carry."""
    results = summary.get("results") or []
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "bank_tasks": len(results),
        "bank_passed": passed,
        "honesty_gate_passed": summary.get("honesty_gate_passed"),
        "honesty_violations": summary.get("honesty_violations"),
        "eval_bank_sha256": summary.get("eval_bank_sha256"),
    }
