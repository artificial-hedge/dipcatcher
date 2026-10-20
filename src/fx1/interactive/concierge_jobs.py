"""Background job types for the interactive concierge."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fx1.interactive.approvals import ActionBounds, ApprovalGate


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
    job_id: str = ""
    goal: str = ""
    started_at: float = field(default_factory=time.monotonic)
    ledger: Any | None = None
    thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> BackgroundJob:
        from fx1.interactive.redact import redact_text

        def on_event(event: str, detail: str) -> None:
            if self.done_event.is_set():
                return
            self.events.append((redact_text(event), redact_text(detail)))

        def target() -> None:
            try:
                self.result = self.run_fn(self.cancel_event, on_event)
            except Exception as exc:  # noqa: BLE001 — report, never crash the shell
                self.error = redact_text(f"{type(exc).__name__}: {exc}")
            finally:
                self.done_event.set()

        self.thread = threading.Thread(target=target, daemon=True, name=f"dip-{self.label}")
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return not self.done_event.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def cancel(self) -> None:
        self.cancel_event.set()

    def join(self, timeout_s: float | None = None) -> bool:
        """Wait for the worker and report whether it actually stopped."""
        if self.thread is not None:
            self.thread.join(timeout_s)
        return self.done_event.is_set()

    def drain(self, limit: int | None = None) -> list[tuple[str, str]]:
        """Drain queued progress events, coalescing consecutive duplicates."""
        out: list[tuple[str, str]] = []
        cap = limit if limit is not None else len(self.events)
        while self.events and len(out) < cap:
            event, detail = self.events.popleft()
            if out and out[-1] == (event, detail):
                continue
            out.append((event, detail))
        return out


@dataclass
class JournaledApprovalGate(ApprovalGate):
    """An approval gate that also records every decision in the session journal."""

    journal: Any | None = None

    def ask(
        self,
        question: str,
        *,
        kind: str = "action",
        key: str | None = None,
        bounds: ActionBounds | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        approved = super().ask(question, kind=kind, key=key, bounds=bounds, timeout_s=timeout_s)
        if self.journal is not None:
            # Journaling never blocks a decision.
            with contextlib.suppress(Exception):
                self.journal.approval(kind, question, approved)
        return approved

    @classmethod
    def wrap(cls, gate: ApprovalGate, journal: Any | None) -> JournaledApprovalGate:
        if isinstance(gate, cls):
            gate.journal = journal
            return gate
        return cls(
            ask_fn=gate.ask_fn,
            records=gate.records,
            decide_fn=gate.decide_fn,
            policy=gate.policy,
            interactive=gate.interactive,
            clock=gate.clock,
            prompt_timeout_s=gate.prompt_timeout_s,
            journal=journal,
        )
