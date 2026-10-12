"""The concierge — dipcatcher's conversational orchestration shell.

This is the front door the product goal describes: chat with the active
fx-series model, ask follow-up questions, and delegate real work through
slash commands:

- ``/superpower <goal>`` — plan, select, coordinate, and explain research
  capabilities (background by default; ``--sync`` for scripts).
- ``/research <goal>`` — deep background web research with budgets,
  progress, ``/cancel`` and ``/redirect``.
- ``/flash ...`` — inspect, correct, remove, search, and refresh persistent
  research memory.
- plain text — a conversational turn with tool access to flash context,
  a bounded web lookup, and approval-gated harness commands.

Responsiveness contract: background jobs run on their own thread. On POSIX
terminals the console polls stdin so progress renders while waiting for a
line. Windows terminals and piped (non-TTY) stdin use blocking reads;
background work continues, but its queued progress renders between lines.

Every seam is injectable (backend, harness, researcher, store, approvals,
input/output) so the whole surface is testable offline; nothing here
fabricates results when a seam is missing — it says so.
"""

from __future__ import annotations

import json
import select
import shlex
import sys  # noqa: F401 — tests patch this module's sys; console shares the object
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

from fx1.interactive import actions, profiles
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.chat_tools import ToolEnv, run_registered_harness
from fx1.interactive.chat_turn import cancel_chat_keyboard, run_chat_turn
from fx1.interactive.data_ux import (
    Attachment,
    annotate_summary,
    attachment_payload,
    dataset_refs,
    handle_data,
    merge_attachments,
    model_context,
    rehash_rows,
    restore_stored,
    stale_warnings,
)
from fx1.interactive.diagnostics_ux import (
    render_diagnostics,
    telemetry_owner,
    write_diagnostics_bundle,
)
from fx1.interactive.diagnostics_ux import (
    startup_lines as startup_check_lines,
)
from fx1.interactive.intent import looks_like_orchestration
from fx1.interactive.jobs import JobLedger
from fx1.interactive.model_client import ModelClient, ModelIdentity
from fx1.interactive.model_ux import (
    CallSnapshot,
    charge_reported_usage,
    completion_kwargs,
    endpoint_stored_line,
    models_command,
    open_backend,
    outbound_messages,
    public_error,
    render_approvals,
    startup_lines,
    use_lines,
)
from fx1.interactive.orb import OrbAnimator, OrbState
from fx1.interactive.profiles import DEFAULT_MODEL
from fx1.interactive.runctx import new_session_id
from fx1.interactive.runs_ux import render_runs
from fx1.interactive.session_ux import (
    export_session,
    format_session_line,
    format_session_list,
    interrupted_hint,
    record_turn,
    stage_record,
    turn_detail,
    turn_outcome,
)
from fx1.interactive.sessions import SessionConflict, SessionError
from fx1.interactive.superpower import (
    SuperpowerResult,
    describe_capabilities,
    run_plan,
)
from fx1.interactive.superpower_glue import (
    PreparedRun,
    RunSeams,
    build_context,
    cancel_lines,
    evidence_report,
    evidence_status_line,
    make_plan,
    operator_status,
    prepare_qualified_run,
)
from fx1.interactive.usage_budget import PRESET_CHAT_TURN, UsageLimits, UsageTracker

HELP_TEXT = """\
Chat
  plain text          talk to the active model (flash, web, harness)
  /use <model>        switch fx1 | fx1-lite; keep this conversation
  /model              show the active model id
  /models             status table; check [--live] [fx1|fx1-lite]

Research
  /superpower <goal>  plan and run capabilities (--sync runs inline; help lists them)
  /data ...           attach|list|remove|inspect a CSV or Parquet
  /runs [id]          persisted jobs, newest first
  /research <goal>    background web research
  /status             job state and evidence status
  /cancel             cancel the background job
  /redirect <goal>    retarget background research
  /evidence [E1]      ledger items, or one item's detail
  /budget             ceilings; tokens are not a spend cap

Session
  /session            id, model, job state, and turn count
  /sessions [N]       newest sessions
  /resume [id]        prefix, last, or id; a job is not restarted
  /export [md|json]   write a redacted transcript
  /new                fresh session; keep the active model

Lab
  /flash ...          add|list|show|remove|correct|search|refresh
  /harness ...        list|describe|run registered commands
  /keys ...           list|set|remove endpoints (presence-only)
  /approvals          decisions; /approvals revoke [key]
  /diagnostics        local report; --bundle PATH writes a zip
  /help               this help
  /exit               leave
"""


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


@dataclass
class BackgroundJob:
    """A cancellable background run with an event queue the console drains."""

    label: str
    run_fn: Callable[[threading.Event, Callable[[str, str], None]], Any]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    events: deque[tuple[str, str]] = field(default_factory=deque)
    result: Any = None
    error: str = ""
    announced: bool = False
    thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> BackgroundJob:
        def on_event(event: str, detail: str) -> None:
            self.events.append((event, detail))

        def target() -> None:
            try:
                self.result = self.run_fn(self.cancel_event, on_event)
            except Exception as exc:  # noqa: BLE001 — report, never crash the shell
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.done_event.set()

        self.thread = threading.Thread(target=target, daemon=True, name=f"dip-{self.label}")
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return not self.done_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def join(self, timeout_s: float | None = None) -> bool:
        """Wait for the worker and report whether it actually stopped."""
        if self.thread is not None:
            self.thread.join(timeout_s)
        return self.done_event.is_set()

    def drain(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        while self.events:
            out.append(self.events.popleft())
        return out


# ---------------------------------------------------------------------------
# The concierge
# ---------------------------------------------------------------------------


def _safe_slash_error(exc: Exception) -> str:
    """Operator-facing slash errors. Do not echo credential-shaped text."""
    return public_error(exc)


class _ApprovalTap:
    """Forwards approvals and emits the key plus the decision, never the question."""

    def __init__(self, gate: ApprovalGate, emit: Callable[[str, Mapping[str, Any]], None]) -> None:
        self._gate = gate
        self._emit = emit

    def ask(self, question: str, *, kind: str = "action", key: str | None = None) -> bool:
        approved = bool(self._gate.ask(question, kind=kind, key=key))
        self._emit(
            "concierge.approval",
            {"key": key or "", "kind": kind, "approved": approved},
        )
        return approved

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gate, name)


class Concierge:
    """Conversational orchestrator; one instance per session."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        store: Any | None = None,
        researcher: Any | None = None,
        research_budget: Any | None = None,
        harness: Any | None = None,
        approvals: ApprovalGate | None = None,
        backend_factory: Callable[..., Any] | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[..., None] = typer.echo,
        animated: bool | None = None,
        session_store: Any | None = None,
        telemetry_emit: Callable[[str, Mapping[str, Any]], None] | None = None,
        harness_validator: Callable[[str, list[str]], list[str]] | None = None,
        usage_preset: UsageLimits | None = None,
        clock: Callable[[], float] | None = None,
        telemetry: Any | None = None,
    ) -> None:
        from fx1.flash.store import FlashStore
        from fx1.webresearch.engine import ResearchBudget, Researcher

        self.model = model
        self._store = store if store is not None else FlashStore()
        self._researcher = researcher if researcher is not None else Researcher()
        self._research_budget = research_budget or ResearchBudget()
        self._harness = harness
        self._approvals: Any = approvals if approvals is not None else ApprovalGate()
        self._backend_factory = backend_factory or ModelClient
        self._input = input_fn
        self._output = output_fn
        self._animated = True if animated is None else animated
        self._messages: list[dict[str, Any]] = []
        self._job: BackgroundJob | None = None
        self._session_store = session_store
        self._session: dict[str, Any] | None = None
        self._budget_overrides: dict[str, Any] = {}
        self.resume_notice = ""
        self._telemetry = telemetry_emit
        self._telemetry_box = (
            telemetry if telemetry is not None else telemetry_owner(telemetry_emit)
        )
        if telemetry_emit is not None:
            self._approvals = _ApprovalTap(self._approvals, self._emit)
        self._jobs: JobLedger | None = None
        self._evidence: Any | None = None
        self._prepared: PreparedRun | None = None
        self._turn_label = "turn"
        self._skip_turn_record = False
        self._harness_validator = harness_validator
        self._usage_preset = PRESET_CHAT_TURN if usage_preset is None else usage_preset
        self._clock = clock or time.monotonic
        self._pass_deadline = clock is None or clock is time.monotonic
        self._state_lock = threading.Lock()
        self._turn_cancel: threading.Event | None = None
        self._turn_mark: int | None = None
        self._status_explicit = False
        self._last_answer = ""
        self._last_status = "none"
        self._last_usage: dict[str, int] | None = None
        self._last_identity: ModelIdentity | None = None
        self._last_latency_s: float | None = None
        self._last_result: Any | None = None
        self._model_stats: dict[str, dict[str, Any]] = {}
        self._attachments: list[Attachment] = []
        self._turn_verified: set[str] = set()
        self._corr_session = new_session_id()
        self._active_tracker: UsageTracker | None = None
        self._pending_snapshot: CallSnapshot | None = None
        self._superpower_cancel: dict[str, threading.Event | None] | None = None

    # -- output helpers ------------------------------------------------------

    def _say(self, text: str = "") -> None:
        self._output(text)

    def _say_error(self, exc: Exception) -> None:
        typer.secho(f"error: {public_error(exc)}", err=True, fg=typer.colors.RED)

    def _refuse_long_input(self, count: int) -> None:
        self._say(f"input too long ({count} characters; limit {_MAX_INPUT_CHARS})")
        self._set_status("error")

    def begin_chat_turn(self) -> None:
        """Drop verified-receipt paths from the previous chat turn."""
        self._turn_verified = set()

    def turn_verified_receipts(self) -> set[str]:
        """Receipt paths this chat turn has verified. Empty until one verifies."""
        return set(self._turn_verified)

    def note_verified_receipt(self, path: str) -> None:
        """Record *path* only when the file verifies under its own directory."""
        self._turn_verified.update(_verified_receipt_paths(path))

    def _backend(self) -> Any | None:
        opened = open_backend(self.model, self._backend_factory)
        if opened.status == "error" and opened.detail:
            self._say(opened.detail)
            self._set_status("error")
        return opened.backend

    # -- read-only headless API ---------------------------------------------

    @property
    def active_model(self) -> str:
        """Selected model id, ``fx1`` or ``fx1-lite``."""
        with self._state_lock:
            return self.model

    @property
    def last_answer(self) -> str:
        """Validated text of the last chat turn, or the last superpower summary."""
        with self._state_lock:
            return self._last_answer

    @property
    def last_status(self) -> str:
        """``ok|withheld|error|cancelled|budget|no_endpoint|none`` for the last input."""
        with self._state_lock:
            return self._last_status

    @property
    def last_usage(self) -> dict[str, int] | None:
        """Token counts from the last model call, or None when unreported."""
        with self._state_lock:
            if self._last_usage is None:
                return None
            return dict(self._last_usage)

    @property
    def last_identity(self) -> ModelIdentity | None:
        """Served-model identity from the last model call, or None."""
        with self._state_lock:
            return self._last_identity

    @property
    def last_latency_s(self) -> float | None:
        """Elapsed seconds of the last model call, or None before any call."""
        with self._state_lock:
            return self._last_latency_s

    @property
    def last_result(self) -> Any | None:
        """Last ``SuperpowerResult``, or None when superpower has not finished."""
        with self._state_lock:
            return self._last_result

    def messages(self) -> list[dict[str, Any]]:
        """Deep copy of the transcript. The built system prompt is not stored."""
        import copy

        with self._state_lock:
            return copy.deepcopy(self._messages)

    def cancel_turn(self) -> None:
        """Cancel the in-flight model turn. Safe to call from another thread."""
        with self._state_lock:
            event = self._turn_cancel
        if event is not None:
            event.set()

    @property
    def job_running(self) -> bool:
        """True while this session's background job has not finished."""
        job = self._job
        return job is not None and job.running

    def cancel_job(self) -> bool:
        """Ask the background job to stop. False when nothing is running."""
        job = self._job
        if job is None or not job.running:
            return False
        job.cancel()
        self._emit("concierge.job", {"label": job.label, "state": "cancel-requested"})
        return True

    def startup_checks(self) -> list[str]:
        """Store-health and job-ledger lines. Empty when both are clean."""
        return startup_check_lines(profiles.store_health, self._job_ledger().reconcile)

    # -- session loop --------------------------------------------------------

    def handle_line(self, line: str) -> bool:
        """Dispatch one input line. Returns True when the session should end."""
        if len(line) > _MAX_INPUT_CHARS:
            self._refuse_long_input(len(line))
            return False
        text = line.strip()
        if not text:
            return False
        self._status_explicit = False
        self._emit("concierge.turn", {"phase": "start", "model": self.model})
        try:
            ended = self._handle_text(text)
        except KeyboardInterrupt:
            self._on_keyboard_interrupt()
            self._persist_session()
            self._finish_turn()
            return False
        if not self._status_explicit:
            self._assign_status("ok")
        self._finish_turn()
        return ended

    def _handle_text(self, text: str) -> bool:
        if text.startswith("/"):
            self._turn_label = text[1:].split(maxsplit=1)[0][:32]
            return self._slash(text)
        if text.lower() in {"exit", "quit"}:
            self.shutdown()
            self._persist_session()
            return True
        if looks_like_orchestration(text):
            self._turn_label = "superpower"
            self._say("routing this to research orchestration (same planner as /superpower).")
            ended = self._cmd_superpower(text)
            self._persist_session()
            return ended
        self._turn_label = "chat"
        self._chat_turn(text)
        self._persist_session()
        return False

    def _finish_turn(self) -> None:
        usage = self.last_usage or {}
        self._emit(
            "concierge.turn",
            {
                "phase": "end",
                "model": self.model,
                "outcome": turn_outcome(self.last_status),
                "latency_s": self.last_latency_s,
                "usage": usage,
            },
        )
        self._note_turn()

    def tick(self) -> None:
        """Drain background progress and announce completion — call between
        typed lines so the conversation stays responsive during research."""
        if self._job is None:
            return
        for event, detail in self._job.drain():
            self._say(f"[{self._job.label}] {event}: {detail}")
        if self._job.done_event.is_set() and not self._job.announced:
            self._job.announced = True
            self._announce_job(self._job)

    def _announce_job(self, job: BackgroundJob) -> None:
        if job.error:
            self._say(f"[{job.label}] failed: {job.error}")
            self._set_status("error")
            return
        result = job.result
        if isinstance(result, SuperpowerResult):
            self._publish_superpower(result)
            self._say("")
            self._say(result.summary or "(no summary)")
            self._say(evidence_status_line(result))
            self._say("")
            self._say(result.explain())
            self._emit(
                "concierge.job",
                {"label": job.label, "state": operator_status(result)},
            )
            return
        if result is not None and hasattr(result, "to_markdown"):
            self._say(result.to_markdown())
            return
        self._say(f"[{job.label}] finished")

    def join_job(self, timeout_s: float | None = None) -> None:
        """Block until the background job settles (used by --sync and tests)."""
        if self._job is not None and self._job.running:
            self._job.join(timeout_s)
        self.tick()

    def shutdown(self, timeout_s: float = 5.0) -> bool:
        """Cancel and join the session-owned worker before the session exits."""
        job = self._job
        if job is None:
            self._persist_session()
            return True
        stopped = True
        if job.running:
            job.cancel()
            stopped = job.join(timeout_s)
            if not stopped:
                self._say(f"[{job.label}] did not stop within {timeout_s:.1f}s")
        self.tick()
        self._persist_session()
        return stopped

    def _job_slot_available(self) -> bool:
        """Refuse to orphan an active worker by replacing its only handle."""
        if self._job is not None and self._job.running:
            self._say(
                f"[{self._job.label}] already running — wait for it or use /cancel before "
                "starting another background job"
            )
            return False
        # Announce/drain a completed predecessor before replacing its handle.
        self.tick()
        return True

    # -- slash commands ------------------------------------------------------

    def _slash(self, text: str) -> bool:
        command, _, rest = text[1:].partition(" ")
        self._emit("concierge.command", {"command": command})
        handler = getattr(self, f"_cmd_{command.replace('-', '_')}", None)
        if handler is None:
            self._say_error(ValueError(_unknown_command(command)))
            return False
        try:
            ended = bool(handler(rest))
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            self._say_error(ValueError(_safe_slash_error(exc)))
            self._set_status("error")
            self._persist_session()
            return False
        self._persist_session()
        return ended

    def _cmd_help(self, rest: str) -> bool:
        del rest
        self._say(HELP_TEXT)
        return False

    def _cmd_exit(self, rest: str) -> bool:
        del rest
        self.shutdown()
        return True

    def _cmd_model(self, rest: str) -> bool:
        del rest
        self._say(f"active model: {self.model}")
        return False

    def _cmd_use(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError("usage: /use <fx1|fx1-lite>")
        model = profiles.check_model(tokens[0])
        previous = self.model
        with self._state_lock:
            self.model = model
        if previous != model:
            self._append_message(
                {
                    "role": "system",
                    "content": (
                        f"active model changed from {previous} to {model}; "
                        "earlier turns stay in this session"
                    ),
                }
            )
        for line in use_lines(model):
            self._say(line)
        return False

    def _cmd_models(self, rest: str) -> bool:
        with self._state_lock:
            stats = {key: dict(value) for key, value in self._model_stats.items()}
            active = self.model
        text, updates = models_command(rest, active=active, stats=stats)
        if updates:
            with self._state_lock:
                self._model_stats.update(updates)
        self._say(text)
        return False

    def _cmd_data(self, rest: str) -> bool:
        rows, text = handle_data(rest, self._attachment_snapshot(), self._harness_obj())
        self._set_attachments(rows)
        self._say(text)
        return False

    def _cmd_runs(self, rest: str) -> bool:
        self._say(render_runs(self._job_ledger(), rest.strip()))
        return False

    def _cmd_diagnostics(self, rest: str) -> bool:
        bundle = _bundle_path(rest)
        diag, text = render_diagnostics(
            config_dir=profiles.config_dir(),
            session_id=self._corr_session,
            model=self.model,
            endpoint_rows=profiles.list_models,
            job_counts=self._job_counts,
            session_counts=self._session_counts,
            telemetry=self._telemetry_box,
        )
        self._say(text)
        if bundle is not None:
            self._say(write_diagnostics_bundle(diag, bundle, telemetry=self._telemetry_box))
        return False

    def _job_counts(self) -> dict[str, int]:
        records = self._job_ledger().list(limit=500)
        counts: dict[str, int] = {}
        for record in records:
            key = str(record.state)
            counts[key] = counts.get(key, 0) + 1
        counts["total"] = len(records)
        return counts

    def _session_counts(self) -> dict[str, int]:
        store = self._session_store
        if store is None:
            return {"total": 0}
        rows, _warnings = store.list(limit=500)
        return {"total": len(rows)}

    def _ledger_state(self) -> str | None:
        result = self._job.result if self._job is not None else None
        job_id = getattr(result, "job_id", None)
        if not isinstance(job_id, str) or self._jobs is None:
            return None
        try:
            return str(self._jobs.get(job_id).state)
        except Exception:  # noqa: BLE001 — cancel text stays honest if the ledger is unreadable
            return None

    def _cmd_approvals(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            self._say(render_approvals(self._approvals.snapshot()))
            return False
        if tokens[0] != "revoke" or len(tokens) > 2:
            raise ValueError("usage: /approvals [revoke [key]]")
        self._approvals.revoke(tokens[1] if len(tokens) == 2 else None)
        self._say("approvals revoked" if len(tokens) == 1 else "approval revoked")
        return False

    def _cmd_session(self, rest: str) -> bool:
        del rest
        if self._session is None:
            self._say(f"no persisted session; active model {self.model}")
            return False
        self._say(format_session_line(self._session, self.model))
        return False

    def _cmd_sessions(self, rest: str) -> bool:
        if self._session_store is None:
            raise ValueError("session store is not configured")
        limit = _session_limit(rest)
        rows, warnings = self._session_store.list(limit)
        self._say(format_session_list(rows))
        for warning in warnings:
            self._say(f"warning: {warning}")
        return False

    def _cmd_resume(self, rest: str) -> bool:
        record = self._load_resume(rest.strip())
        self._adopt_resume(record)
        return False

    def _load_resume(self, text: str) -> dict[str, Any]:
        store = self._session_store
        if store is None:
            raise ValueError("session store is not configured")
        if not text:
            record = store.latest()
        else:
            record = store.load(store.resolve_prefix(text))
        if not isinstance(record, dict):
            raise ValueError("no saved session to resume")
        return record

    def _adopt_resume(self, record: dict[str, Any]) -> None:
        job = record.get("job") or {}
        if job.get("state") == "running" and self._session_store is not None:
            record = self._session_store.mark_interrupted(str(record["id"]))
            self.resume_notice = (
                f"session {record.get('id')} was running; marked interrupted and not restarted"
            )
            self._say(self.resume_notice)
        self._install_record(record)
        self._say(f"resumed session {record.get('id')} on model {self.model}")
        if (record.get("job") or {}).get("state") == "interrupted":
            self._say(interrupted_hint())

    def _install_record(self, record: dict[str, Any]) -> None:
        self._session = record
        restored = str(record.get("model") or self.model)
        profiles.check_model(restored)
        with self._state_lock:
            self.model = restored
        self._messages = [
            dict(item) for item in record.get("messages") or [] if isinstance(item, dict)
        ]
        stored_budget = record.get("budget")
        self._budget_overrides = dict(stored_budget) if isinstance(stored_budget, dict) else {}
        self._restore_attachments(record.get("attachments"))

    def _restore_attachments(self, stored: object) -> None:
        rows, warnings = restore_stored(stored)
        self._set_attachments(rows)
        for warning in warnings:
            self._say(warning)

    def _set_attachments(self, rows: Sequence[Attachment]) -> None:
        with self._state_lock:
            self._attachments = list(rows)

    def _attachment_snapshot(self) -> list[Attachment]:
        with self._state_lock:
            return list(self._attachments)

    def dataset_model_note(self) -> str:
        """Sanitized dataset labels for the model, already wrapped as untrusted."""
        return model_context(self._attachment_snapshot())

    def attachment_warnings(self) -> tuple[str, ...]:
        """Stale-dataset lines. Empty when every attached digest still matches."""
        return stale_warnings(self._attachment_snapshot())

    def adopt_attachments(self, rows: Sequence[Attachment]) -> str | None:
        """Add already-profiled datasets. A cap refusal leaves the list unchanged."""
        with self._state_lock:
            merged, problem = merge_attachments(self._attachments, rows)
            if problem:
                return problem
            self._attachments = merged
        return None

    def _refuse_datasets(self) -> bool:
        reason = self._dataset_block()
        if not reason:
            return False
        self._say(reason)
        return True

    def _dataset_block(self) -> str | None:
        with self._state_lock:
            rows, reason = rehash_rows(self._attachments)
            self._attachments = rows
        return reason

    def _cmd_export(self, rest: str) -> bool:
        if self._session_store is None or self._session is None:
            raise ValueError("no persisted session to export")
        kind, dest = _export_args(rest)
        _path, message = export_session(
            self._session_store, self._session, kind, dest, profiles.config_dir()
        )
        self._say(message)
        return False

    def _cmd_new(self, rest: str) -> bool:
        del rest
        if self.job_running:
            self._say("a background job is running; /cancel it before /new")
            return False
        self._persist_session()
        self._messages = []
        self._budget_overrides = {}
        self._set_attachments([])
        self._job = None
        self._session = None
        if self._session_store is not None:
            self._session = self._session_store.create(model=self.model, messages=[])
        self._say(f"new session on model {self.model}")
        return False

    def restore(self, session_id: str, *, preferred_model: str | None = None) -> None:
        """Load one session by id or unique prefix. A running job is not restarted."""
        if self._session_store is None:
            self.resume_notice = "session store is not configured"
            return
        self._install_restored(self._load_by_reference(session_id), preferred_model)

    def restore_latest(self, *, preferred_model: str | None = None) -> None:
        """Load the newest session. A running job is marked interrupted, not restarted."""
        store = self._session_store
        if store is None:
            self.resume_notice = "session store is not configured"
            return
        record = store.latest()
        if not record:
            self.resume_notice = "no saved session to resume"
            return
        self._install_restored(record, preferred_model)

    def _load_by_reference(self, session_id: str) -> dict[str, Any]:
        store = self._session_store
        if store is None:
            raise SessionError("session store is not configured")
        try:
            loaded = store.load(store.resolve_prefix(session_id))
        except SessionError as exc:
            raise SessionError(public_error(exc)) from exc
        if not isinstance(loaded, dict):
            raise SessionError("session record must be an object")
        return loaded

    def _install_restored(self, record: dict[str, Any], preferred_model: str | None) -> None:
        store = self._session_store
        loaded = record
        job = record.get("job") or {}
        if job.get("state") == "running" and store is not None:
            loaded = store.mark_interrupted(str(record["id"]))
        if preferred_model is not None:
            profiles.check_model(preferred_model)
            loaded["model"] = preferred_model
        self._install_record(loaded)
        notice = f"resumed session {loaded.get('id')} on model {self.model}"
        if (loaded.get("job") or {}).get("state") == "interrupted":
            notice += "; the previous job was not restarted"
        self.resume_notice = notice

    def _cmd_evidence(self, rest: str) -> bool:
        result = self._job.result if self._job is not None else None
        self._say(evidence_report(result, self._evidence, rest.strip()))
        return False

    def _cmd_budget(self, rest: str) -> bool:
        """Show the run ceilings, or store one override on this session."""
        from fx1.interactive.budgets import BudgetError, RunBudget, merge_budget

        tokens = shlex.split(rest)
        if not tokens:
            budget = self._run_budget()
            if budget is None:
                self._say("budget unavailable")
                return False
            public = budget.as_public()
            self._say(
                "budget "
                f"max_steps={public['max_steps']} max_seconds={public['max_seconds']} "
                f"max_families={public['max_families']} max_harness={public['max_harness']} "
                f"max_retries={public['max_retries']}; {public['spend_cap_note']}"
            )
            self._say(UsageTracker(self._usage_preset).describe())
            return False
        if len(tokens) != 2:
            raise ValueError(
                "usage: /budget [max-steps N | max-seconds N | max-families N | "
                "max-harness N | max-retries N]"
            )
        key = tokens[0].replace("-", "_")
        if key not in {"max_steps", "max_seconds", "max_families", "max_harness", "max_retries"}:
            raise ValueError(f"unknown budget field {tokens[0]!r}")
        try:
            number: int | float = float(tokens[1]) if key == "max_seconds" else int(tokens[1])
        except ValueError as exc:
            raise ValueError(f"{tokens[0]} needs a number") from exc
        current = dict(self._budget_overrides)
        current[key] = number
        try:
            merge_budget(RunBudget(), current)
        except BudgetError as exc:
            raise ValueError(str(exc)) from exc
        self._budget_overrides = current
        if isinstance(self._session, dict):
            self._session["budget"] = dict(current)
        self._say(f"budget {key}={number}")
        return False

    def _persist_session(self) -> None:
        store = self._session_store
        if store is None:
            return
        self._skip_turn_record = False
        try:
            self._write_session(store)
        except SessionConflict:
            self._skip_turn_record = True
            self._say("session changed on disk; not saved over the newer revision")
        except SessionError:
            self._skip_turn_record = True
            self._say("session was not saved")

    def _write_session(self, store: Any) -> None:
        record = self._session
        if record is None:
            record = store.create(model=self.model, messages=list(self._messages))
            self._session = record
        job = self._job
        stage_record(
            record,
            model=self.model,
            messages=list(self._messages),
            job_running=job is not None and job.running,
            job_error="" if job is None else job.error,
            job_present=job is not None,
            budget=self._budget_overrides,
            approvals=self._approvals.snapshot(),
            attachments=attachment_payload(self._attachment_snapshot()),
        )
        store.save(record)

    def _note_turn(self) -> None:
        store = self._session_store
        record = self._session
        if store is None or not isinstance(record, dict) or self._skip_turn_record:
            return
        record["approvals"] = self._approvals.snapshot()
        served = None if self.last_identity is None else self.last_identity.served
        warning = record_turn(
            store,
            record,
            model=self.model,
            served_model=served,
            usage=self.last_usage,
            outcome=self.last_status,
            detail=turn_detail(self._turn_label),
        )
        if warning:
            self._say(warning)

    def _cmd_keys(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens or tokens[0] == "list":
            self._say(actions.render_table(profiles.list_models()))
            return False
        sub = tokens[0]
        if sub == "remove" and len(tokens) == 2:
            removed = profiles.remove_endpoint(tokens[1])
            self._say(
                f"removed endpoint {tokens[1]!r}"
                if removed
                else f"endpoint {tokens[1]!r} was not stored"
            )
            return False
        if sub == "set" and len(tokens) in {1, 2}:
            model = tokens[1] if len(tokens) == 2 else self.model
            return self._keys_set(model)
        raise ValueError("usage: /keys [list | set [model] | remove <model>]")

    def _keys_set(self, model: str) -> bool:
        profiles.check_model(model)
        prompted = self._prompt_endpoint(model)
        if prompted is None:
            return False
        api_key, base_url = prompted
        try:
            path, spec = profiles.set_endpoint_spec(model, api_key, base_url)
        except (ValueError, KeyError) as exc:
            raise type(exc)(public_error(exc)) from None
        self._say(endpoint_stored_line(model, path, spec.describe()))
        self._say_store_warnings()
        return False

    def _prompt_endpoint(self, model: str) -> tuple[str, str] | None:
        import getpass

        endpoint = profiles.resolve_endpoint(model)
        default_url = endpoint[1] if endpoint else profiles.default_base_url()
        mark = _prompt_mark()
        try:
            api_key = getpass.getpass(f"{mark} {model} API key: ")
            base_url = self._input(f"{mark} {model} endpoint URL [{default_url}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            self._say(f"endpoint setup for {model!r} cancelled; no changes saved")
            return None
        return api_key, base_url or default_url

    def _say_store_warnings(self) -> None:
        for warning in profiles.store_health():
            self._say(f"warning: {warning}")

    def _cmd_harness(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens or tokens[0] == "list":
            self._say(actions.render_table(actions.harness_rows()))
            return False
        if tokens[0] == "describe" and len(tokens) == 2:
            self._say(actions.status_json(actions.harness_get(tokens[1])))
            return False
        if tokens[0] == "run" and len(tokens) >= 2:
            self._run_harness(tokens[1], tokens[2:])
            return False
        raise ValueError("usage: /harness [list | describe <name> | run <name> [args...]]")

    def _run_harness(self, name: str, extra_args: list[str]) -> str:
        """Approval-gated, registry-constrained harness execution."""
        return run_registered_harness(name, extra_args, self._tool_env(self._current_cancel()))

    # -- /superpower ---------------------------------------------------------

    def _cmd_superpower(self, rest: str) -> bool:
        goal, sync = _parse_superpower(rest)
        if goal is None:
            self._say(describe_capabilities())
            return False
        if not self._job_slot_available():
            return False
        return self._launch_superpower(goal, sync=sync)

    def _launch_superpower(self, goal: str, *, sync: bool) -> bool:
        if self._refuse_datasets():
            return False
        prepared = self._prepare_superpower()
        model_fn = self._bind_model_fn(self._backend())
        plan = make_plan(
            goal,
            prepared,
            model_fn=model_fn,
            family_selector=self._family_selector(),
            max_harness=self._max_families(),
        )
        self._say_superpower_plan(plan)
        if sync:
            self._superpower_sync(prepared, plan, model_fn)
            return False
        if self._refuse_datasets():
            return False
        self._superpower_async(prepared, plan, model_fn)
        return False

    def _prepare_superpower(self) -> PreparedRun:
        prepared = prepare_qualified_run(
            self._harness_obj(),
            config_dir=profiles.config_dir(),
            datasets=dataset_refs(self._attachment_snapshot()),
        )
        self._prepared = prepared
        self._evidence = prepared.evidence
        return prepared

    def _say_superpower_plan(self, plan: Any) -> None:
        self._say(plan.describe())
        budget = self._run_budget()
        if budget is not None:
            public = budget.as_public()
            self._say(
                "budget: "
                f"max_steps={public['max_steps']} max_seconds={public['max_seconds']} "
                f"max_families={public['max_families']}; {public['spend_cap_note']}"
            )
        self._say("")

    def _superpower_sync(
        self, prepared: PreparedRun, plan: Any, model_fn: Callable[[str], str] | None
    ) -> None:
        if self._refuse_datasets():
            return
        self._emit("concierge.job", {"label": "superpower", "state": "started"})
        self._arm_superpower_cancel(threading.Event())
        result = self._executed_plan(prepared, plan, model_fn, None, None)
        job = BackgroundJob(label="superpower", run_fn=lambda *_args: result)
        job.result = result
        job.done_event.set()
        job.announced = True
        self._job = job
        self._announce_job(job)

    def _superpower_async(
        self, prepared: PreparedRun, plan: Any, model_fn: Callable[[str], str] | None
    ) -> None:
        def run_fn(
            cancel_event: threading.Event, on_event: Callable[[str, str], None]
        ) -> SuperpowerResult:
            self._arm_superpower_cancel(cancel_event)
            blocked = self._dataset_block()
            if blocked:
                return _blocked_result(plan, blocked)
            return self._executed_plan(prepared, plan, model_fn, on_event, cancel_event)

        self._emit("concierge.job", {"label": "superpower", "state": "started"})
        self._job = BackgroundJob(label="superpower", run_fn=run_fn).start()
        self._say("[superpower] running in the background — /status, /cancel; chat stays open")

    def _bind_model_fn(self, backend: Any | None) -> Callable[[str], str] | None:
        if backend is None:
            self._superpower_cancel = None
            return None
        holder: dict[str, threading.Event | None] = {"cancel": None}
        self._superpower_cancel = holder

        def model_fn(prompt: str) -> str:
            return self._synthesize(backend, prompt, holder["cancel"])

        return model_fn

    def _synthesize(self, backend: Any, prompt: str, cancel: threading.Event | None) -> str:
        tracker = UsageTracker(self._usage_preset, monotonic=self._clock)
        tracker.start()
        messages = outbound_messages(
            [{"role": "user", "content": _with_dataset_context(prompt, self.dataset_model_note())}],
            self.model,
            tools=(),
            mode="superpower",
        )
        kwargs = completion_kwargs(
            tracker, backend, cancel, pass_deadline=self._pass_deadline, tools=None
        )
        tracker.charge_model_call()
        with self._orb():
            text = str(backend.complete(messages, **kwargs))
        charge_reported_usage(tracker, backend)
        return text

    def _arm_superpower_cancel(self, cancel: threading.Event | None) -> None:
        holder = self._superpower_cancel
        if holder is not None:
            holder["cancel"] = cancel

    def _context_for(
        self,
        prepared: PreparedRun,
        model_fn: Callable[[str], str] | None,
        on_event: Callable[[str, str], None] | None,
        cancel_event: threading.Event | None,
    ) -> Any:
        return build_context(
            prepared,
            RunSeams(
                model_fn=model_fn,
                harness=self._harness_obj(),
                approvals=self._approvals,
                researcher=self._researcher,
                research_budget=self._research_budget,
                flash_store=self._store,
                on_event=on_event,
                cancel_event=cancel_event,
                budget=self._run_budget(),
                job_ledger=self._job_ledger(),
                session_id=self._corr_session,
                telemetry_emit=self._telemetry,
            ),
        )

    def _executed_plan(
        self,
        prepared: PreparedRun,
        plan: Any,
        model_fn: Callable[[str], str] | None,
        on_event: Callable[[str, str], None] | None,
        cancel_event: threading.Event | None,
    ) -> SuperpowerResult:
        blocked = self._dataset_block()
        if blocked:
            return _blocked_result(plan, blocked)
        result = run_plan(plan, self._context_for(prepared, model_fn, on_event, cancel_event))
        result.summary = annotate_summary(
            result.summary or "",
            unchecked=_score_without_issue_time(result),
        )
        return result

    def _harness_obj(self) -> Any:
        from fx1.harness import Harness

        if self._harness is not None:
            return self._harness
        return Harness()

    def _job_ledger(self) -> JobLedger:
        if self._jobs is None:
            self._jobs = JobLedger()
        return self._jobs

    def _max_families(self) -> int:
        budget = self._run_budget()
        limit = getattr(budget, "max_families", 4)
        return int(limit) if isinstance(limit, int) else 4

    def _run_budget(self) -> Any | None:
        try:
            from fx1.interactive.budgets import RunBudget
        except ImportError:
            return None
        budget = RunBudget()
        overrides = {str(key): value for key, value in self._budget_overrides.items()}
        if overrides:
            from fx1.interactive.budgets import merge_budget

            return merge_budget(budget, overrides)
        budget.validate()
        return budget

    def _family_selector(self) -> Callable[[str], dict[str, Any]]:
        def select(goal: str) -> dict[str, Any]:
            from fx1.harness import Harness

            harness = self._harness if self._harness is not None else Harness()
            try:
                from fx1.interactive.families import FamilyClient, harness_family_runner
            except ImportError:
                return _family_payload_from_harness(harness, goal, self._max_families())
            return FamilyClient(harness_family_runner(harness)).select(
                goal, max_families=self._max_families()
            )

        return select

    # -- /research -----------------------------------------------------------

    def _cmd_research(self, rest: str) -> bool:
        options, goal = _parse_research(shlex.split(rest))
        if not self._job_slot_available():
            return False
        budget = _research_budget_for(self._research_budget, options)
        job = BackgroundJob(
            label="research",
            run_fn=self._research_runner(goal, budget, options.get("queries")),
        )
        return self._dispatch_research(job, goal, budget, sync=bool(options.get("sync")))

    def _research_runner(
        self, goal: str, budget: Any, queries: Any
    ) -> Callable[[threading.Event, Callable[[str, str], None]], Any]:
        def run_fn(cancel_event: threading.Event, on_event: Callable[[str, str], None]) -> Any:
            return self._researcher.run(
                goal,
                budget=budget,
                queries=queries,
                cancel_event=cancel_event,
                on_progress=lambda progress: on_event(progress.event, progress.detail),
            )

        return run_fn

    def _dispatch_research(self, job: BackgroundJob, goal: str, budget: Any, *, sync: bool) -> bool:
        if sync:
            job.start()
            job.done_event.wait()
            self._job = job
            self.tick()
            return False
        self._job = job.start()
        self._say(
            f"[research] {goal!r} running in the background "
            f"(budget {budget.max_duration_s:.0f}s / {budget.max_sources} sources) — "
            "/status, /cancel, /redirect; chat stays open"
        )
        return False

    def _cmd_status(self, rest: str) -> bool:
        del rest
        if self._job is None:
            self._say("no background job")
            return False
        state = "running" if self._job.running else "finished"
        self._say(f"[{self._job.label}] {state}")
        result = self._job.result
        if isinstance(result, SuperpowerResult):
            self._say(evidence_status_line(result))
        for event, detail in self._job.drain():
            self._say(f"[{self._job.label}] {event}: {detail}")
        if not self._job.running and not self._job.announced:
            self._job.announced = True
            self._announce_job(self._job)
        return False

    def _cmd_cancel(self, rest: str) -> bool:
        del rest
        if not self.cancel_job():
            self._say("no running background job")
            return False
        label = self._job.label if self._job is not None else "job"
        for line in cancel_lines(label, self._ledger_state()):
            self._say(line)
        return False

    def _cmd_redirect(self, rest: str) -> bool:
        goal = rest.strip()
        if not goal:
            raise ValueError("usage: /redirect <new goal>")
        if self._job is None or not self._job.running:
            self._say("no running background job to redirect")
            return False
        if hasattr(self._researcher, "redirect"):
            self._researcher.redirect(goal)
            self._say(f"[{self._job.label}] redirect requested -> {goal!r}")
        else:
            self._say_error(RuntimeError("active job cannot be redirected"))
        return False

    # -- /flash --------------------------------------------------------------

    def _cmd_flash(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError("usage: /flash add|list|show|remove|correct|search|refresh ...")
        sub = tokens[0]
        args = tokens[1:]
        if sub == "add":
            return self._flash_add(args)
        if sub == "list":
            self._flash_list(args)
            return False
        if sub == "show" and len(args) == 1:
            self._flash_show(args[0])
            return False
        if sub == "remove" and len(args) == 1:
            removed = self._store.remove(args[0])
            self._say(f"removed {args[0]!r}" if removed else f"no entry {args[0]!r}")
            return False
        if sub == "correct":
            return self._flash_correct(args)
        if sub == "search" and args:
            self._flash_search(" ".join(args))
            return False
        if sub == "refresh" and len(args) == 1:
            self._flash_refresh(args[0])
            return False
        raise ValueError("usage: /flash add|list|show|remove|correct|search|refresh ...")

    def _flash_add(self, args: list[str]) -> bool:
        options: dict[str, Any] = {}
        positional: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token in {
                "--task",
                "--tags",
                "--source",
                "--uncertainty",
                "--conflict",
            } and i + 1 < len(args):
                options[token[2:]] = args[i + 1]
                i += 2
                continue
            if token in {"--private", "--verified"}:
                options[token[2:]] = True
                i += 1
                continue
            positional.append(token)
            i += 1
        text = " ".join(positional)
        if not text.strip():
            raise ValueError("usage: /flash add <text> [--task T] [--tags a,b] [--source URL] ...")
        from fx1.interactive.sessions import scrub_secret_text

        entry = self._store.add(
            text=scrub_secret_text(text),
            task=options.get("task", ""),
            tags=[t.strip() for t in options.get("tags", "").split(",") if t.strip()],
            sources=[{"url": options["source"]}] if options.get("source") else [],
            uncertainty=options.get("uncertainty"),
            conflicts=[options["conflict"]] if options.get("conflict") else [],
            private=bool(options.get("private")),
            verified=bool(options.get("verified")),
        )
        self._say(f"saved flash entry {entry.id} ({entry.id[:8]})")
        return False

    def _flash_list(self, args: list[str]) -> None:
        from fx1.flash.retrieve import retrieve

        if args and args[0] == "all":
            entries = self._store.all()
        elif args:
            entries = [hit.entry for hit in retrieve(self._store, " ".join(args), k=10)]
        else:
            entries = self._store.all()
        if not entries:
            self._say("(no flash entries)")
            return
        for entry in entries[:30]:
            flags = []
            if entry.private:
                flags.append("private")
            if entry.verified:
                flags.append("verified")
            if entry.uncertainty:
                flags.append(entry.uncertainty)
            self._say(
                f"{entry.id[:8]}  {entry.updated_at[:10]}  "
                f"{('[' + ','.join(flags) + '] ') if flags else ''}{entry.text[:90]}"
            )

    def _flash_show(self, entry_id: str) -> None:
        entry = self._store.get(entry_id)
        if entry is None:
            self._say_error(KeyError(f"no flash entry {entry_id!r}"))
            return
        self._say(entry.model_dump_json(indent=2))

    def _flash_correct(self, args: list[str]) -> bool:
        if not args or not args[0]:
            raise ValueError("usage: /flash correct <id> --text ... --uncertainty ... --tags ...")
        entry_id = args[0]
        patch: dict[str, Any] = {}
        i = 1
        while i < len(args):
            token = args[i]
            if token in {"--text", "--task", "--uncertainty"} and i + 1 < len(args):
                patch[token[2:]] = args[i + 1]
                i += 2
                continue
            if token == "--tags" and i + 1 < len(args):
                patch["tags"] = [t.strip() for t in args[i + 1].split(",") if t.strip()]
                i += 2
                continue
            if token == "--verified":
                patch["verified"] = True
                i += 1
                continue
            if token == "--conflict" and i + 1 < len(args):
                entry = self._store.get(entry_id)
                if entry is None:
                    raise KeyError(f"no flash entry {entry_id!r}")
                patch["conflicts"] = [*entry.conflicts, args[i + 1]]
                i += 2
                continue
            raise ValueError(f"unknown /flash correct option {token!r}")
        revised = self._store.update(entry_id, patch)
        if revised is None:
            raise KeyError(f"no flash entry {entry_id!r}")
        self._say(f"corrected {entry_id} (revision kept; history preserved)")
        return False

    def _flash_search(self, query: str) -> None:
        from fx1.flash.retrieve import retrieve

        hits = retrieve(self._store, query, k=8)
        if not hits:
            self._say(f"no flash context above threshold for {query!r}")
            return
        for hit in hits:
            self._say(
                f"{hit.entry.id[:8]}  score={hit.score:.3f}  {hit.entry.text[:100]}"
                f"  [{', '.join(hit.reasons)}]"
            )

    def _flash_refresh(self, entry_id: str) -> None:
        from fx1.flash.refresh import refresh

        report = refresh(self._store, entry_id, fetch_fn=None)
        if report is None:
            self._say_error(KeyError(f"no flash entry {entry_id!r}"))
            return
        self._say(report.note)

    # -- plain chat ----------------------------------------------------------

    def _chat_turn(self, text: str) -> None:
        run_chat_turn(self, text)

    def _on_keyboard_interrupt(self) -> None:
        cancel_chat_keyboard(self)

    def _tool_superpower(self, goal: str) -> str:
        prepared = self._prepare_superpower()
        plan = make_plan(
            goal,
            prepared,
            model_fn=None,
            family_selector=self._family_selector(),
            max_harness=self._max_families(),
        )
        result = run_plan(plan, self._context_for(prepared, None, None, self._current_cancel()))
        self._publish_superpower(result)
        return f"{result.summary}\n\n{result.explain()}"

    def _tool_env(self, cancel: threading.Event | None) -> ToolEnv:
        return ToolEnv(
            store=self._store,
            researcher=self._researcher,
            harness=self._harness,
            approvals=self._approvals,
            validator=self._harness_validator,
            cancel_event=cancel,
            session_id=self._corr_session,
            say=self._say,
            say_error=self._say_error,
            run_superpower=self._tool_superpower,
            cwd=Path.cwd(),
            config_dir=profiles.config_dir(),
            emit=self._emit,
        )

    def _orb(self) -> OrbAnimator | _null:
        if not self._animated:
            return _null()
        return OrbAnimator(OrbState.COMPOSING, preset="inline")

    def _append_message(self, message: dict[str, Any]) -> None:
        with self._state_lock:
            self._messages.append(message)

    def _truncate_messages(self, start: int) -> None:
        with self._state_lock:
            del self._messages[start:]

    def _transcript(self) -> list[dict[str, Any]]:
        with self._state_lock:
            return [dict(item) for item in self._messages]

    def _arm_turn(self, event: threading.Event) -> None:
        with self._state_lock:
            self._turn_cancel = event

    def _disarm_turn(self) -> None:
        with self._state_lock:
            self._turn_cancel = None

    def _current_cancel(self) -> threading.Event | None:
        with self._state_lock:
            return self._turn_cancel

    def _assign_status(self, status: str) -> None:
        with self._state_lock:
            self._last_status = status

    def _set_status(self, status: str) -> None:
        self._assign_status(status)
        self._status_explicit = True
        self._emit("concierge.status", {"status": status, "model": self.model})

    def _remember_answer(self, content: str) -> None:
        with self._state_lock:
            self._last_answer = content
            self._last_status = "ok"
        self._status_explicit = True

    def _note_call(self, snapshot: CallSnapshot) -> None:
        served = None if snapshot.identity is None else snapshot.identity.served
        matched = None if snapshot.identity is None else snapshot.identity.matched
        with self._state_lock:
            self._last_usage = None if snapshot.usage is None else dict(snapshot.usage)
            self._last_identity = snapshot.identity
            self._last_latency_s = snapshot.latency_s
            self._model_stats[self.model] = {
                "served": served,
                "latency_s": snapshot.latency_s,
                "matched": matched,
            }

    def _publish_superpower(self, result: SuperpowerResult) -> None:
        with self._state_lock:
            self._last_result = result
            self._last_answer = result.summary or ""
            self._last_status = operator_status(result)
        self._status_explicit = True

    def _emit(self, event: str, fields: Mapping[str, Any]) -> None:
        emit = self._telemetry
        if emit is None:
            return
        try:
            emit(event, dict(fields))
        except Exception:  # noqa: BLE001 — telemetry must not break a turn
            return


class _null:
    """Context manager that does nothing (animation disabled / non-TTY)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Console driver
# ---------------------------------------------------------------------------


def _family_payload_from_harness(harness: Any, goal: str, max_families: int) -> dict[str, Any]:
    """Read the lab family surface when the typed client module is absent."""
    result = harness.run("families", ["--select", goal, "--max", str(max_families)])
    if int(getattr(result, "exit_code", 1)) != 0:
        detail = getattr(result, "stderr", "") or "families command failed"
        raise RuntimeError(str(detail)[:500])
    payload = json.loads(getattr(result, "stdout", "") or "")
    if not isinstance(payload, dict):
        raise RuntimeError("families payload must be an object")
    payload.setdefault("schema", "dipcatcher.family_surface.v1")
    return payload


def _session_limit(rest: str) -> int:
    tokens = shlex.split(rest)
    if not tokens:
        return 20
    if len(tokens) != 1:
        raise ValueError("usage: /sessions [N]")
    try:
        limit = int(tokens[0])
    except ValueError as exc:
        raise ValueError("usage: /sessions [N]") from exc
    if limit < 1:
        raise ValueError("usage: /sessions [N]")
    return limit


def _export_args(rest: str) -> tuple[str, Path | None]:
    tokens = shlex.split(rest)
    kind = "md"
    if tokens and tokens[0] in {"md", "json", "markdown"}:
        kind = "json" if tokens[0] == "json" else "md"
        tokens = tokens[1:]
    if len(tokens) > 1:
        raise ValueError("usage: /export [md|json] [path]")
    if not tokens:
        return kind, None
    return kind, Path(tokens[0])


def _bundle_path(rest: str) -> Path | None:
    tokens = shlex.split(rest)
    if not tokens:
        return None
    if len(tokens) != 2 or tokens[0] != "--bundle":
        raise ValueError("usage: /diagnostics [--bundle PATH]")
    return Path(tokens[1])


def _tty_polling_read(input_fn: Callable[[str], str], prompt: str, timeout_s: float) -> str | None:
    """Poll a POSIX TTY; Windows and non-TTY input use blocking reads.

    Returns None on timeout. Implemented in ``fx1.interactive.console``.
    ``select`` stays referenced so tests can monkeypatch this module's import.
    """
    _ = select
    from fx1.interactive.console import _tty_polling_read as _read

    return _read(input_fn, prompt, timeout_s)


def run_console(*args: Any, **kwargs: Any) -> None:
    """Open the chat console. See ``fx1.interactive.console.run_console``.

    ``startup_lines`` stays referenced so this module's import is not dropped.
    """
    _ = startup_lines
    from fx1.interactive.console import run_console as _run

    return _run(*args, **kwargs)


_MAX_INPUT_CHARS = 16_000


def _verified_receipt_paths(path: str) -> set[str]:
    if not isinstance(path, str) or not path or len(path) > 512:
        return set()
    try:
        item = _verify_one(path)
    except (OSError, ValueError, RuntimeError):
        return set()
    from fx1.interactive.evidence_ledger import Verdict

    if item.verdict is not Verdict.VERIFIED:
        return set()
    found = {path}
    ref = getattr(item, "ref", None)
    if isinstance(ref, str) and ref:
        found.add(ref)
    return found


def _verify_one(path: str) -> Any:
    from fx1.interactive.evidence_ledger import verify_receipt

    return verify_receipt(path, roots=[_receipt_parent(path)])


def _receipt_parent(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent


def _unknown_command(command: str) -> str:
    shown = _visible_command(command)
    match = _closest_command(shown)
    if match is None:
        return f"unknown command /{shown}; try /help"
    return f"unknown command /{shown}; did you mean /{match}? try /help"


def _visible_command(command: str) -> str:
    from fx1.interactive.terminal import sanitize_for_terminal

    text = sanitize_for_terminal(command)
    visible = "".join(ch for ch in text if ch.isprintable())
    collapsed = " ".join(visible.split())
    return collapsed or "command"


def _closest_command(shown: str) -> str | None:
    import difflib

    folded = shown.replace("_", "-").casefold()
    found = difflib.get_close_matches(folded, _slash_names(), n=1, cutoff=0.6)
    if not found:
        return None
    return found[0]


def _slash_names() -> list[str]:
    return sorted(
        name.removeprefix("_cmd_").replace("_", "-")
        for name in dir(Concierge)
        if name.startswith("_cmd_")
    )


def _prompt_mark() -> str:
    from fx1.interactive.terminal import unicode_stdout

    if unicode_stdout():
        return "›"
    return ">"


def _with_dataset_context(prompt: str, note: str) -> str:
    if not note:
        return prompt
    return f"{note}\n\n{prompt}"


def _blocked_result(plan: Any, reason: str) -> SuperpowerResult:
    goal = getattr(plan, "goal", "")
    if not isinstance(goal, str):
        goal = ""
    return SuperpowerResult(goal=goal, plan=plan, summary=reason, status="no-evidence")


def _score_without_issue_time(result: SuperpowerResult) -> bool:
    if not _plan_scored(result):
        return False
    return not _bound_issue_time(result)


def _plan_scored(result: SuperpowerResult) -> bool:
    if _ran_score(result.runs):
        return True
    steps = getattr(result.plan, "typed_steps", ())
    return _planned_score(steps)


def _bound_issue_time(result: SuperpowerResult) -> bool:
    steps = getattr(result.plan, "typed_steps", ())
    for step in steps:
        if getattr(step, "capability_id", "") != "score-forecasts":
            continue
        args = getattr(step, "args", {})
        if isinstance(args, Mapping) and "--available-time-col" in args:
            return True
    return False


def _ran_score(runs: Sequence[Any]) -> bool:
    return any(getattr(run, "id", "") == "score-forecasts" for run in runs)


def _planned_score(steps: Sequence[Any]) -> bool:
    return any(getattr(step, "capability_id", "") == "score-forecasts" for step in steps)


def _parse_superpower(rest: str) -> tuple[str | None, bool]:
    tokens = shlex.split(rest)
    if not tokens:
        raise ValueError("usage: /superpower <goal> [--sync]")
    if tokens[0] == "help":
        return None, False
    sync = "--sync" in tokens
    goal = " ".join(token for token in tokens if token != "--sync").strip()
    if not goal:
        raise ValueError("usage: /superpower <goal> [--sync]")
    return goal, sync


_RESEARCH_USAGE = (
    "usage: /research <goal> [--budget-s N] [--max-sources N] [--queries q1|q2|q3] [--sync]"
)
_RESEARCH_INT = {"--budget-s": "budget_s", "--max-sources": "max_sources"}


def _parse_research(tokens: list[str]) -> tuple[dict[str, Any], str]:
    if not tokens:
        raise ValueError(_RESEARCH_USAGE)
    options: dict[str, Any] = {}
    positional: list[str] = []
    index = 0
    while index < len(tokens):
        nxt = _take_research_option(tokens, index, options)
        if nxt < 0:
            positional.append(tokens[index])
            index += 1
            continue
        index = nxt
    goal = " ".join(positional).strip()
    if not goal:
        raise ValueError("usage: /research <goal> ...")
    return options, goal


def _take_research_option(tokens: list[str], index: int, options: dict[str, Any]) -> int:
    token = tokens[index]
    if token in _RESEARCH_INT and index + 1 < len(tokens):
        return _research_int_option(token, tokens[index + 1], options, index)
    if token == "--queries" and index + 1 < len(tokens):
        options["queries"] = [part for part in tokens[index + 1].split("|") if part]
        return index + 2
    if token == "--sync":
        options["sync"] = True
        return index + 1
    return -1


def _research_int_option(token: str, raw: str, options: dict[str, Any], index: int) -> int:
    try:
        options[_RESEARCH_INT[token]] = int(raw)
    except ValueError:
        raise ValueError(f"{token} needs an integer") from None
    return index + 2


def _research_budget_for(base: Any, options: Mapping[str, Any]) -> Any:
    from fx1.webresearch.engine import ResearchBudget

    return ResearchBudget(
        max_duration_s=options.get("budget_s", base.max_duration_s),
        max_sources=options.get("max_sources", base.max_sources),
        max_searches=base.max_searches,
        max_depth=base.max_depth,
        search_pause_s=base.search_pause_s,
    )
