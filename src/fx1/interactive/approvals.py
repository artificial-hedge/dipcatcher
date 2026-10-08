"""Approval gates for consequential actions and private-data sharing.

The front door keeps the lab's fail-closed posture: approvals default to
*deny* whenever no interactive operator can answer (piped stdin, tests,
scripts). An interactive session asks on the console. Two reasons are
first-class:

- ``consequential`` — running a harness command that trains/optimizes or
  otherwise mutates lab state beyond cheap diagnostics;
- ``private_data`` — moving private flash context (or other machine-local
  material) into a model prompt or a web-search query.

The gate records every decision so a session can explain what was (and was
not) authorized.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

AskFn = Callable[[str], bool]  # question -> approved


@dataclass
class ApprovalRecord:
    kind: str
    question: str
    approved: bool
    timestamp: str


@dataclass
class ApprovalGate:
    """Fail-closed approval: ``ask`` returns False unless a prompter approves."""

    ask_fn: AskFn | None = None
    records: list[ApprovalRecord] = field(default_factory=list)

    def ask(self, question: str, *, kind: str = "action") -> bool:
        from datetime import UTC, datetime

        approved = False
        if self.ask_fn is not None:
            try:
                approved = bool(self.ask_fn(question))
            except (EOFError, KeyboardInterrupt):
                approved = False
        self.records.append(
            ApprovalRecord(
                kind=kind,
                question=question,
                approved=approved,
                timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
        return approved

    def denied(self) -> int:
        return sum(1 for record in self.records if not record.approved)

    def granted(self) -> int:
        return sum(1 for record in self.records if record.approved)


def console_ask_fn(input_fn: Callable[[str], str] = input) -> AskFn:
    """Interactive y/N prompter; anything but an explicit yes is a deny."""

    def ask(question: str) -> bool:
        try:
            answer = input_fn(f"approval required — {question} [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

    return ask
