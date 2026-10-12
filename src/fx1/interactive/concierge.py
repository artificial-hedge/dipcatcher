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
import re
import select
import shlex
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import typer

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive import actions, profiles
from fx1.interactive.approvals import ApprovalGate, console_ask_fn
from fx1.interactive.intent import looks_like_orchestration
from fx1.interactive.orb import OrbAnimator, OrbState, orbs_enabled
from fx1.interactive.orb import banner as orb_banner
from fx1.interactive.profiles import DEFAULT_MODEL
from fx1.interactive.superpower import (
    SuperpowerContext,
    SuperpowerResult,
    describe_capabilities,
    plan_goal,
    run_plan,
)
from fx1.serve.backends import HostedK3Backend


def _scrub_tool_calls(calls: Any) -> list[dict[str, Any]]:
    """Copy tool calls with credential values removed from argument JSON."""
    from fx1.interactive.sessions import scrub_secret_text

    scrubbed: list[dict[str, Any]] = []
    for call in calls or ():
        if not isinstance(call, dict):
            continue
        copied = dict(call)
        function = copied.get("function")
        if isinstance(function, dict):
            fn = dict(function)
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                fn["arguments"] = scrub_secret_text(arguments)
            copied["function"] = fn
        scrubbed.append(copied)
    return scrubbed


HELP_TEXT = """\
conversation          plain text chats with the active model (tools: flash, web, harness)
/superpower <goal>    plan + run the relevant research capabilities (--sync to run inline)
/research <goal>      deep web research in the background (--budget-s, --max-sources, --queries)
/status               show the background job's progress
/cancel               cancel the background job (research or superpower)
/redirect <goal>      re-target the background research at a new goal
/flash add|list|show|remove|correct|search|refresh ...
/superpower help      list every capability the planner may select
/use <model>          activate fx1 | fx1-lite and keep this conversation
/model                show the active model
/session              show the persisted session id, model, and job state
/resume [id]          reload a session; an interrupted job is not restarted
/evidence             show receipt paths from the last orchestration job
/budget               show or set step, family, harness, and duration ceilings
/keys list|set|remove  manage model endpoints (presence-only)
/harness list|describe|run  registered dipcatcher harness commands (fail-closed)
/help                 this help
/exit                 leave
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

_SECRET_ERROR = re.compile(r"(?i)(?:\bsk-[A-Za-z0-9]+|\bapi[_-]?key\b|\bbearer\b|\bsecret\b)")


def _safe_slash_error(exc: Exception) -> str:
    """Operator-facing slash errors. Do not echo credential-shaped text."""
    text = str(exc)
    if _SECRET_ERROR.search(text):
        return f"{type(exc).__name__}: command failed"
    return text


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
    ) -> None:
        from fx1.flash.store import FlashStore
        from fx1.webresearch.engine import ResearchBudget, Researcher

        self.model = model
        self._store = store if store is not None else FlashStore()
        self._researcher = researcher if researcher is not None else Researcher()
        self._research_budget = research_budget or ResearchBudget()
        self._harness = harness
        self._approvals = approvals if approvals is not None else ApprovalGate()
        self._backend_factory = backend_factory or HostedK3Backend
        self._input = input_fn
        self._output = output_fn
        self._animated = True if animated is None else animated
        self._messages: list[dict[str, Any]] = []
        self._job: BackgroundJob | None = None
        self._session_store = session_store
        self._session: dict[str, Any] | None = None
        self._budget_overrides: dict[str, Any] = {}
        self.resume_notice = ""

    # -- output helpers ------------------------------------------------------

    def _say(self, text: str = "") -> None:
        self._output(text)

    def _say_error(self, exc: Exception) -> None:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)

    def _backend(self) -> Any | None:
        resolved = profiles.resolve_endpoint(self.model)
        if not resolved:
            return None
        api_key, base_url = resolved
        try:
            return self._backend_factory(api_key=api_key, model=self.model, api_url=base_url)
        except Exception:  # noqa: BLE001
            return None

    # -- session loop --------------------------------------------------------

    def handle_line(self, line: str) -> bool:
        """Dispatch one input line. Returns True when the session should end."""
        text = line.strip()
        if not text:
            return False
        if text.startswith("/"):
            return self._slash(text)
        if text.lower() in {"exit", "quit"}:
            self.shutdown()
            self._persist_session()
            return True
        if looks_like_orchestration(text):
            self._say("routing this to research orchestration (same planner as /superpower).")
            ended = self._cmd_superpower(text)
            self._persist_session()
            return ended
        self._chat_turn(text)
        self._persist_session()
        return False

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
            return
        result = job.result
        if isinstance(result, SuperpowerResult):
            self._say("")
            self._say(result.summary or "(no summary)")
            self._say("")
            self._say(result.explain())
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
        handler = getattr(self, f"_cmd_{command.replace('-', '_')}", None)
        if handler is None:
            self._say_error(ValueError(f"unknown command /{command}; try /help"))
            return False
        try:
            ended = bool(handler(rest))
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            self._say_error(ValueError(_safe_slash_error(exc)))
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
        model = tokens[0]
        previous = self.model
        profiles.apply_profile(model)
        self.model = model
        if previous != model:
            self._messages.append(
                {
                    "role": "system",
                    "content": (
                        f"active model changed from {previous} to {model}; "
                        "earlier turns stay in this session"
                    ),
                }
            )
        self._say(
            f"active model {model!r}; conversation kept; "
            f"{profiles.ENV_API_KEY} + {profiles.ENV_BASE_URL} injected into this session"
        )
        return False

    def _cmd_session(self, rest: str) -> bool:
        del rest
        if self._session is None:
            self._say(f"no persisted session; active model {self.model}")
            return False
        job = self._session.get("job") or {}
        self._say(
            f"session {self._session.get('id')} model {self.model} job {job.get('state', 'idle')}"
        )
        return False

    def _cmd_resume(self, rest: str) -> bool:
        if self._session_store is None:
            raise ValueError("session store is not configured")
        session_id = rest.strip()
        record = (
            self._session_store.load(session_id) if session_id else self._session_store.latest()
        )
        if not record:
            raise ValueError("no saved session to resume")
        job = record.get("job") or {}
        if job.get("state") == "running":
            record = self._session_store.mark_interrupted(str(record["id"]))
            self.resume_notice = (
                f"session {record.get('id')} was running; marked interrupted and not restarted"
            )
            self._say(self.resume_notice)
        self._session = record
        restored = str(record.get("model") or self.model)
        profiles.check_model(restored)
        self.model = restored
        self._messages = [
            dict(item) for item in record.get("messages") or [] if isinstance(item, dict)
        ]
        stored_budget = record.get("budget")
        self._budget_overrides = dict(stored_budget) if isinstance(stored_budget, dict) else {}
        self._say(f"resumed session {record.get('id')} on model {self.model}")
        return False

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
        job = record.get("job") or {}
        if job.get("state") == "running":
            record = store.mark_interrupted(str(record["id"]))
        self._session = record
        restored = str(record.get("model") or self.model)
        chosen = preferred_model or restored
        profiles.check_model(chosen)
        self.model = chosen
        if preferred_model is not None:
            record["model"] = preferred_model
        self._messages = [
            dict(item) for item in record.get("messages") or [] if isinstance(item, dict)
        ]
        notice = f"resumed session {record.get('id')} on model {self.model}"
        if (record.get("job") or {}).get("state") == "interrupted":
            notice += "; the previous job was not restarted"
        self.resume_notice = notice

    def _cmd_evidence(self, rest: str) -> bool:
        del rest
        result = self._job.result if self._job is not None else None
        paths: list[str] = []
        if isinstance(result, SuperpowerResult):
            paths = [artifact for run in result.runs for artifact in run.artifacts if artifact]
        self._say("\n".join(paths) if paths else "no receipt paths on the last job")
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
        if self._session is None:
            self._session = store.create(model=self.model, messages=list(self._messages))
            return
        self._session["model"] = self.model
        self._session["messages"] = list(self._messages)
        job = self._job
        state = "running" if job is not None and job.running else "idle"
        if job is not None and not job.running:
            state = "finished" if not job.error else "idle"
        current = dict(self._session.get("job") or {})
        if current.get("state") == "interrupted" and state == "idle":
            state = "interrupted"
        current["state"] = state
        current.setdefault("resume", "manual" if state == "interrupted" else "none")
        current.setdefault("steps_completed", [])
        self._session["job"] = current
        if self._budget_overrides:
            self._session["budget"] = dict(self._budget_overrides)
        store.save(self._session)

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
            profiles.check_model(model)
            import getpass

            endpoint = profiles.resolve_endpoint(model)
            default_url = endpoint[1] if endpoint else profiles.default_base_url()
            try:
                api_key = getpass.getpass(f"› {model} API key: ")
                base_url = self._input(f"› {model} endpoint URL [{default_url}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._say(f"endpoint setup for {model!r} cancelled; no changes saved")
                return False
            base_url = base_url or default_url
            path = profiles.set_endpoint(model, api_key, base_url)
            self._say(
                f"stored endpoint for {model!r} -> {profiles.host_of(base_url)} "
                f"in {path} (mode 0600)"
            )
            return False
        raise ValueError("usage: /keys [list | set [model] | remove <model>]")

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
        from fx1.harness import HARNESS_REGISTRY, Harness

        command = next((c for c in HARNESS_REGISTRY if c.name == name), None)
        if command is None:
            raise KeyError(f"{name!r} is not a registered harness command")
        consequential = command.role.value == "model_training" or command.name in {
            "train",
            "optimize",
            "paper",
        }
        if consequential and not self._approvals.ask(
            f"run harness command {name!r} (consequential)?", kind="consequential"
        ):
            self._say(f"[harness] {name}: not approved — skipped")
            return ""
        harness = self._harness if self._harness is not None else Harness()
        result = harness.run(name, extra_args)
        if result.stdout:
            self._say(result.stdout.rstrip()[-4000:])
        if result.stderr:
            self._say_error(RuntimeError(result.stderr.strip()[-4000:]))
        return str(result.stdout)

    # -- /superpower ---------------------------------------------------------

    def _cmd_superpower(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError("usage: /superpower <goal> [--sync]")
        if tokens[0] == "help":
            self._say(describe_capabilities())
            return False
        sync = "--sync" in tokens
        goal = " ".join(t for t in tokens if t != "--sync")
        if not goal.strip():
            raise ValueError("usage: /superpower <goal> [--sync]")
        if not self._job_slot_available():
            return False
        backend = self._backend()
        model_fn = None
        if backend is not None:

            def model_fn(prompt: str) -> str:
                with (
                    OrbAnimator(OrbState.COMPOSING, preset="inline") if self._animated else _null()
                ):
                    return str(backend.complete([{"role": "user", "content": prompt}]))

        plan = plan_goal(
            goal,
            model_fn=model_fn,
            family_selector=self._family_selector(),
            max_harness=self._max_families(),
        )
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
        if sync:
            ctx = self._superpower_context(model_fn)
            result = run_plan(plan, ctx)
            job = BackgroundJob(label="superpower", run_fn=lambda *_: result)
            job.result = result
            job.done_event.set()
            job.announced = True
            self._job = job
            self._announce_job(job)
            return False

        def run_fn(
            cancel_event: threading.Event, on_event: Callable[[str, str], None]
        ) -> SuperpowerResult:
            job_ctx = self._superpower_context(
                model_fn, on_event=on_event, cancel_event=cancel_event
            )
            return run_plan(plan, job_ctx)

        self._job = BackgroundJob(label="superpower", run_fn=run_fn).start()
        self._say("[superpower] running in the background — /status, /cancel; chat stays open")
        return False

    def _superpower_context(
        self,
        model_fn: Callable[[str], str] | None,
        *,
        on_event: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SuperpowerContext:
        return SuperpowerContext(
            model_fn=model_fn,
            harness=self._harness,
            approvals=self._approvals,
            researcher=self._researcher,
            research_budget=self._research_budget,
            flash_store=self._store,
            on_event=on_event,
            cancel_event=cancel_event,
            budget=self._run_budget(),
        )

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
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError(
                "usage: /research <goal> [--budget-s N] [--max-sources N] "
                "[--queries q1|q2|q3] [--sync]"
            )
        options: dict[str, Any] = {}
        positional: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"--budget-s", "--max-sources"} and i + 1 < len(tokens):
                try:
                    options[token[2:].replace("-", "_")] = int(tokens[i + 1])
                except ValueError:
                    raise ValueError(f"{token} needs an integer") from None
                i += 2
                continue
            if token == "--queries" and i + 1 < len(tokens):
                options["queries"] = [q for q in tokens[i + 1].split("|") if q]
                i += 2
                continue
            if token == "--sync":
                options["sync"] = True
                i += 1
                continue
            positional.append(token)
            i += 1
        goal = " ".join(positional)
        if not goal.strip():
            raise ValueError("usage: /research <goal> ...")
        if not self._job_slot_available():
            return False
        from fx1.webresearch.engine import ResearchBudget

        base = self._research_budget
        budget = ResearchBudget(
            max_duration_s=options.get("budget_s", base.max_duration_s),
            max_sources=options.get("max_sources", base.max_sources),
            max_searches=base.max_searches,
            max_depth=base.max_depth,
            search_pause_s=base.search_pause_s,
        )
        queries = options.get("queries")

        def run_fn(cancel_event: threading.Event, on_event: Callable[[str, str], None]) -> Any:
            return self._researcher.run(
                goal,
                budget=budget,
                queries=queries,
                cancel_event=cancel_event,
                on_progress=lambda p: on_event(p.event, p.detail),
            )

        job = BackgroundJob(label="research", run_fn=run_fn)
        if options.get("sync"):
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
        for event, detail in self._job.drain():
            self._say(f"[{self._job.label}] {event}: {detail}")
        if not self._job.running and not self._job.announced:
            self._job.announced = True
            self._announce_job(self._job)
        return False

    def _cmd_cancel(self, rest: str) -> bool:
        del rest
        if self._job is None or not self._job.running:
            self._say("no running background job")
            return False
        self._job.cancel()
        self._say(f"[{self._job.label}] cancel requested — will stop at the next step")
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
        backend = self._backend()
        if backend is None:
            self._say(
                f"no model endpoint configured — set one with `/keys set {self.model}`, or export "
                "MOONSHOT_API_KEY / FX1_BASE_URL. Slash commands still work."
            )
            return
        start = len(self._messages)
        from fx1.interactive.sessions import scrub_secret_text

        self._messages.append({"role": "user", "content": scrub_secret_text(text)})
        tools = self._chat_tools()
        try:
            for _ in range(6):
                with (
                    OrbAnimator(OrbState.COMPOSING, preset="inline") if self._animated else _null()
                ):
                    completion = backend.complete_with_tools(
                        self._messages, tools=tools, tool_choice="auto"
                    )
                content = scrub_secret_text(validate_fx1_output(completion.content or ""))
                if not completion.tool_calls:
                    self._messages.append({"role": "assistant", "content": content})
                    self._say(content)
                    return
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": _scrub_tool_calls(completion.tool_calls),
                    }
                )
                for call in completion.tool_calls:
                    result_text = scrub_secret_text(self._execute_tool(call))
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result_text,
                        }
                    )
            self._say("(tool loop budget exhausted — ask a follow-up to continue)")
        except Fx1HonestyError as exc:
            refusal = (
                f"answer withheld by the fx-1 honesty gate ({exc}). Ask a follow-up to continue."
            )
            if any(message.get("role") == "tool" for message in self._messages[start:]):
                # A blocked answer cannot undo completed tool actions. Keep
                # their validated call/result transcript for the next turn.
                self._messages.append({"role": "assistant", "content": refusal})
            else:
                del self._messages[start:]
            self._say(refusal)
        except Exception as exc:  # noqa: BLE001 — a dead network must not kill the shell
            del self._messages[start:]
            self._say_error(exc)

    def _chat_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "flash_retrieve",
                    "description": (
                        "Search persistent research memory (flash context) for "
                        "prior findings relevant to a query. Returns findings with "
                        "sources, uncertainty, and conflicts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_lookup",
                    "description": (
                        "A bounded public-web lookup: search and read up to two "
                        "sources. Use for facts outside the conversation; deep "
                        "research belongs to /research."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "flash_save",
                    "description": "Persist a finding to flash context for later sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "task": {"type": "string"},
                            "uncertainty": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "superpower_run",
                    "description": (
                        "Plan and run the qualified research families for a goal. "
                        "Uses the same selector and honesty gate as /superpower. "
                        "Retired, unknown, and unqualified families are refused."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"goal": {"type": "string"}},
                        "required": ["goal"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "harness_run",
                    "description": (
                        "Run one registered dipcatcher harness command "
                        "(e.g. doctor, verify-research, research). Consequential "
                        "commands require operator approval. Fail-closed on "
                        "unknown names."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name"],
                    },
                },
            },
        ]

    def _execute_tool(self, call: dict[str, Any]) -> str:
        fn = call.get("function", {})
        name = fn.get("name", "")
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                arguments = {}
        except json.JSONDecodeError:
            arguments = {}
        if name == "flash_retrieve":
            from fx1.flash.retrieve import context_block, release_private, retrieve

            hits = release_private(
                retrieve(self._store, str(arguments.get("query", "")), k=4),
                self._approvals,
            )
            if not hits:
                return "no relevant flash context found"
            for hit in hits:
                self._store.mark_used(hit.entry.id)
            return context_block(hits)
        if name == "flash_save":
            from fx1.interactive.sessions import scrub_secret_text

            entry = self._store.add(
                text=scrub_secret_text(str(arguments.get("text", ""))),
                task=scrub_secret_text(str(arguments.get("task", ""))),
                uncertainty=arguments.get("uncertainty"),
                tags=["model-saved"],
            )
            return f"saved flash entry {entry.id}"
        if name == "web_lookup":
            return self._web_lookup(str(arguments.get("query", "")))
        if name == "superpower_run":
            goal = str(arguments.get("goal", "")).strip()
            if not goal:
                return "superpower_run requires a goal"
            plan = plan_goal(goal, family_selector=self._family_selector())
            result = run_plan(plan, self._superpower_context(None))
            return f"{result.summary}\n\n{result.explain()}"
        if name == "harness_run":
            harness_name = str(arguments.get("name", ""))
            args_list = [
                str(a) for a in arguments.get("args", []) if isinstance(a, (str, int, float))
            ]
            try:
                stdout = self._run_harness(harness_name, args_list)
            except KeyError as exc:
                return f"harness refused: {exc}"
            return stdout[-6000:] or "(command produced no output)"
        return f"unknown tool {name!r}"

    def _web_lookup(self, query: str) -> str:
        """One search + at most two fetches, capped — the chat-scale lookup."""
        from fx1.webresearch.engine import ResearchBudget

        if not query.strip():
            return "empty query"
        budget = ResearchBudget(
            max_duration_s=30,
            max_searches=1,
            max_sources=2,
            max_depth=0,
            search_pause_s=0.0,
        )
        report = self._researcher.run(query, budget=budget, queries=[query])
        parts: list[str] = []
        for source in report.sources:
            if source.ok:
                parts.append(f"SOURCE {source.url}\n{source.text[:2500]}")
        if not parts:
            return "web lookup found nothing (or the network is unavailable)"
        return "\n\n".join(parts)


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


def _tty_polling_read(input_fn: Callable[[str], str], prompt: str, timeout_s: float) -> str | None:
    """Poll a POSIX TTY; Windows and non-TTY input use blocking reads.

    Returns None on timeout. The caller supplies an empty prompt while
    waiting for a previously prompted line.
    """
    if sys.platform == "win32" or not sys.stdin.isatty():
        return input_fn(prompt)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":  # EOF
        raise EOFError
    return line.rstrip("\n")


def _console_concierge(
    *,
    model: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[..., None],
    resume: bool = False,
    model_explicit: bool = False,
) -> Concierge:
    """Production console: persist sessions and mark a crashed run interrupted."""
    session_store = None
    try:
        from fx1.interactive.sessions import SessionStore
    except ImportError:
        session_store = None
    else:
        session_store = SessionStore()
    concierge = Concierge(
        model=model,
        input_fn=input_fn,
        output_fn=output_fn,
        approvals=ApprovalGate(ask_fn=console_ask_fn(input_fn)),
        session_store=session_store,
    )
    if session_store is None:
        return concierge
    if resume:
        preferred = model if model_explicit else None
        concierge.restore_latest(preferred_model=preferred)
        return concierge
    latest = session_store.latest()
    job = (latest or {}).get("job") or {}
    if latest and job.get("state") == "running":
        record = session_store.mark_interrupted(str(latest["id"]))
        concierge.resume_notice = (
            f"session {record.get('id')} was interrupted and was not restarted. "
            "Use /resume to inspect it."
        )
    return concierge


def run_console(
    model: str = DEFAULT_MODEL,
    *,
    concierge_factory: Callable[..., Concierge] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = typer.echo,
    poll_timeout_s: float = 0.15,
    resume: bool = False,
    model_explicit: bool = False,
) -> None:
    """Read chat input with idle progress ticks on POSIX terminals.

    Windows and non-TTY input render queued progress between submitted lines.
    """
    concierge = (
        concierge_factory()
        if concierge_factory is not None
        else _console_concierge(
            model=model,
            input_fn=input_fn,
            output_fn=output_fn,
            resume=resume,
            model_explicit=model_explicit,
        )
    )
    model = concierge.model
    endpoint = profiles.resolve_endpoint(model)
    host = profiles.host_of(endpoint[1]) if endpoint else "unconfigured"
    output_fn(orb_banner(model, host, color=orbs_enabled()))
    output_fn(
        f"dipcatcher concierge — model {model}. Plain text chats; "
        "/superpower, /research, /flash, /help. Ctrl-D or /exit to leave."
    )
    if concierge.resume_notice:
        output_fn(concierge.resume_notice)
    if endpoint is None:
        output_fn(f"No endpoint configured. Run /keys set {model} to add your API key and URL.")
    if sys.platform == "win32":
        output_fn("On Windows, background progress updates after you submit a line.")
    prompt_pending = True
    while True:
        try:
            line = _tty_polling_read(input_fn, "dip › " if prompt_pending else "", poll_timeout_s)
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            concierge.shutdown()
            return
        if line is None:  # poll timeout — render background progress
            prompt_pending = False
            concierge.tick()
            continue
        prompt_pending = True
        try:
            if concierge.handle_line(line):
                return
        except Exception as exc:  # noqa: BLE001 — a bad turn must not kill the shell
            typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        concierge.tick()
