"""Dataset, run, and model-check commands for the concierge.

``Concierge`` subclasses :class:`DataCommands` so these handlers stay out of
the chat loop module.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fx1.interactive import profiles
from fx1.interactive.data_ux import (
    Attachment,
    annotate_summary,
    dataset_refs,
    handle_data,
    merge_attachments,
    model_context,
    rehash_rows,
    restore_stored,
    stale_warnings,
)
from fx1.interactive.datasets import dataset_rows
from fx1.interactive.model_ux import models_command
from fx1.interactive.planner import DatasetRef
from fx1.interactive.runs_ux import render_runs
from fx1.interactive.superpower import SuperpowerResult
from fx1.interactive.superpower_glue import prepare_qualified_run


class DataCommands:
    """Attachment state and the slash commands that read it."""

    _attachments: list[Attachment]
    _state_lock: threading.Lock
    _model_stats: dict[str, dict[str, Any]]
    model: str
    _prepared: Any
    _evidence: Any
    _job: Any

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

    def _restore_attachments(self, stored: object, legacy: object = None) -> None:
        rows, warnings = restore_stored(stored, cwd=Path.cwd(), legacy=legacy)
        self._set_attachments(rows)
        for warning in warnings:
            self._say(warning)

    def _set_attachments(self, rows: Sequence[Attachment]) -> None:
        with self._state_lock:
            self._attachments = list(rows)

    def _attachment_snapshot(self) -> list[Attachment]:
        with self._state_lock:
            return list(self._attachments)

    @property
    def _datasets(self) -> list[DatasetRef]:
        return list(dataset_refs(self._attachment_snapshot()))

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

    def _prepare_superpower(self) -> Any:
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
        self, prepared: Any, plan: Any, model_fn: Callable[[str], str] | None
    ) -> None:
        if self._refuse_datasets():
            return
        self._emit("concierge.job", {"label": "superpower", "state": "started"})
        result = self._executed_plan(prepared, plan, model_fn, None, self._current_cancel())
        job = _finished_job(result)
        self._job = job
        self._announce_job(job)

    def _superpower_async(
        self, prepared: Any, plan: Any, model_fn: Callable[[str], str] | None
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
        self._job = _started_job(run_fn)
        self._say("[superpower] running in the background — /status, /cancel; chat stays open")

    def _stage_datasets(self, record: dict[str, Any]) -> None:
        rows = dataset_rows(self._datasets)
        if rows or "datasets" in record:
            record["datasets"] = rows

    def _executed_plan(
        self,
        prepared: Any,
        plan: Any,
        model_fn: Callable[[str], str] | None,
        on_event: Callable[[str, str], None] | None,
        cancel_event: threading.Event | None,
    ) -> SuperpowerResult:
        blocked = self._dataset_block()
        if blocked:
            return _blocked_result(plan, blocked)
        from fx1.interactive.superpower import run_plan

        result = run_plan(plan, self._context_for(prepared, model_fn, on_event, cancel_event))
        result.summary = annotate_summary(
            result.summary or "",
            unchecked=_score_without_issue_time(result),
        )
        return result

    def _say(self, text: str = "") -> None:
        raise NotImplementedError

    def _harness_obj(self) -> Any:
        raise NotImplementedError

    def _job_ledger(self) -> Any:
        raise NotImplementedError

    def _context_for(
        self,
        prepared: Any,
        model_fn: Callable[[str], str] | None,
        on_event: Callable[[str, str], None] | None,
        cancel_event: threading.Event | None,
    ) -> Any:
        raise NotImplementedError

    def _emit(self, event: str, fields: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def _current_cancel(self) -> threading.Event | None:
        raise NotImplementedError

    def _announce_job(self, job: Any) -> None:
        raise NotImplementedError

    def _arm_superpower_cancel(self, cancel: threading.Event | None) -> None:
        raise NotImplementedError

    def _run_budget(self) -> Any:
        raise NotImplementedError


def _finished_job(result: SuperpowerResult) -> Any:
    from fx1.interactive.concierge import BackgroundJob

    job = BackgroundJob(label="superpower", run_fn=lambda *_args: result)
    job.result = result
    job.done_event.set()
    job.announced = True
    return job


def _started_job(
    run_fn: Callable[[threading.Event, Callable[[str, str], None]], SuperpowerResult],
) -> Any:
    from fx1.interactive.concierge import BackgroundJob

    return BackgroundJob(label="superpower", run_fn=run_fn).start()


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
