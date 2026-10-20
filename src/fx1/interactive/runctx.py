"""Correlation identifiers for sessions, runs, jobs, and steps.

Identifiers are time-sortable, carry no secrets, and are validated before they
are ever used as a file name, so a hostile or corrupted id can never become a
path traversal. ``RunContext`` bundles the ids that every diagnostic event,
job record, and receipt reference should carry.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, replace
from datetime import UTC, datetime

KINDS: tuple[str, ...] = ("s", "r", "j")  # session, run, job
_ID_RE = re.compile(r"^(?P<kind>[srj])-\d{8}T\d{6}Z-[0-9a-f]{8}$")
_STEP_RE = re.compile(r"^r-\d{8}T\d{6}Z-[0-9a-f]{8}\.\d{3}$")


def new_id(kind: str, *, now: datetime | None = None, rand: str | None = None) -> str:
    """Mint a sortable id like ``s-20261012T024500Z-1a2b3c4d``."""
    if kind not in KINDS:
        raise ValueError(f"unknown id kind {kind!r}; expected one of {KINDS}")
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = rand if rand is not None else secrets.token_hex(4)
    candidate = f"{kind}-{stamp}-{suffix}"
    if not _ID_RE.match(candidate):
        raise ValueError(f"random suffix {suffix!r} is not 8 lowercase hex characters")
    return candidate


def new_session_id(*, now: datetime | None = None) -> str:
    return new_id("s", now=now)


def new_run_id(*, now: datetime | None = None) -> str:
    return new_id("r", now=now)


def new_job_id(*, now: datetime | None = None) -> str:
    return new_id("j", now=now)


def step_id(run_id: str, index: int) -> str:
    """Deterministic step id ``<run_id>.<NNN>`` (index 0-999)."""
    require_id(run_id, "r")
    if not 0 <= index <= 999:
        raise ValueError(f"step index {index} outside 0..999")
    return f"{run_id}.{index:03d}"


def is_valid_id(value: str, kind: str | None = None) -> bool:
    if not isinstance(value, str):
        return False
    match = _ID_RE.match(value)
    return match is not None and (kind is None or match.group("kind") == kind)


def is_valid_step_id(value: str) -> bool:
    return isinstance(value, str) and _STEP_RE.match(value) is not None


def require_id(value: str, kind: str | None = None) -> str:
    """Return *value* unchanged when it is a well-formed id; else ``ValueError``.

    Call before building any path from an externally supplied id.
    """
    if not is_valid_id(value, kind):
        expected = f" of kind {kind!r}" if kind else ""
        raise ValueError(f"malformed identifier{expected}")
    return value


@dataclass(frozen=True)
class RunContext:
    """Correlation fields attached to every event, job, and receipt reference."""

    session_id: str
    run_id: str | None = None
    step: str | None = None
    model: str | None = None
    family: str | None = None

    def __post_init__(self) -> None:
        require_id(self.session_id, "s")
        if self.run_id is not None:
            require_id(self.run_id, "r")
        if self.step is not None and not is_valid_step_id(self.step):
            raise ValueError("malformed step identifier")

    def with_run(self, run_id: str | None = None) -> RunContext:
        return replace(self, run_id=run_id or new_run_id(), step=None)

    def with_step(self, index: int, *, family: str | None = None) -> RunContext:
        if self.run_id is None:
            raise ValueError("with_step requires a run id; call with_run() first")
        return replace(self, step=step_id(self.run_id, index), family=family or self.family)

    def with_model(self, model: str | None) -> RunContext:
        return replace(self, model=model)

    def as_fields(self) -> dict[str, str]:
        """Non-empty fields for structured logs (never contains secrets)."""
        fields = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "step": self.step,
            "model": self.model,
            "family": self.family,
        }
        return {key: value for key, value in fields.items() if value}
