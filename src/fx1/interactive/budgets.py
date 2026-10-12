"""Hard ceilings for dipcatcher /superpower from planning through synthesis.

Counters cover steps, retries, research families, harness commands, and wall
time. No price catalog is configured, so an unknown price is not a dollar
spend cap: ``as_public`` always reports ``spend_cap`` as ``None``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

_COUNT_CEILING = 32
_SECONDS_CEILING = 86400
_OVERRIDE_KEYS = frozenset(
    {
        "max_steps",
        "max_seconds",
        "max_retries",
        "max_families",
        "max_harness",
    }
)
_SPEND_CAP_NOTE = (
    "no price catalog is configured; this budget is time/step limits only, not a dollar cap"
)


class BudgetError(ValueError):
    """A run budget is invalid or a charge would pass a hard ceiling."""


def _reject_non_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BudgetError(f"{name} must be a real number")
    if isinstance(value, float) and math.isnan(value):
        raise BudgetError(f"{name} must be a real number")


@dataclass
class RunBudget:
    """Step, retry, family, harness, and wall-time ceilings for one run.

    ``started_at`` is ``time.monotonic`` and is omitted from the public view.
    It is a process clock, not a price or a calendar timestamp.
    """

    max_steps: int = 12
    max_seconds: float = 3600
    max_retries: int = 1
    max_families: int = 4
    max_harness: int = 4
    started_at: float = field(default_factory=time.monotonic)
    steps: int = 0
    retries: int = 0

    def validate(self) -> None:
        """Raise ``BudgetError`` when a ceiling is negative or above its hard max."""
        limits: dict[str, int | float] = {
            "max_steps": self.max_steps,
            "max_seconds": self.max_seconds,
            "max_retries": self.max_retries,
            "max_families": self.max_families,
            "max_harness": self.max_harness,
        }
        for name, value in limits.items():
            _reject_non_real(name, value)
            if value < 0:
                raise BudgetError(f"{name} must not be negative")
        if self.max_steps > _COUNT_CEILING:
            raise BudgetError(f"max_steps exceeds {_COUNT_CEILING}")
        if self.max_families > _COUNT_CEILING:
            raise BudgetError(f"max_families exceeds {_COUNT_CEILING}")
        if self.max_harness > _COUNT_CEILING:
            raise BudgetError(f"max_harness exceeds {_COUNT_CEILING}")
        if self.max_seconds > _SECONDS_CEILING:
            raise BudgetError(f"max_seconds exceeds {_SECONDS_CEILING}")

    def expired(self, now: float | None = None) -> bool:
        """Return whether elapsed monotonic time has reached ``max_seconds``."""
        clock = time.monotonic() if now is None else now
        return (clock - self.started_at) >= self.max_seconds

    def charge_step(self) -> None:
        """Record one step, or raise if the step ceiling or the clock is spent."""
        self.validate()
        blocked: list[str] = []
        if self.steps + 1 > self.max_steps:
            blocked.append("step budget exceeded")
        if self.expired():
            blocked.append("run budget expired")
        if blocked:
            raise BudgetError("; ".join(blocked))
        self.steps += 1

    def charge_retry(self) -> None:
        """Record one retry, or raise if another retry would pass ``max_retries``."""
        self.validate()
        if self.retries + 1 > self.max_retries:
            raise BudgetError("retry budget exceeded")
        self.retries += 1

    def as_public(self) -> dict[str, Any]:
        """Return limits and progress with no secrets and no dollar spend cap.

        ``spend_cap`` is ``None`` because an unknown price must not be treated
        as a guaranteed dollar ceiling.
        """
        return {
            "steps": self.steps,
            "retries": self.retries,
            "max_steps": self.max_steps,
            "max_seconds": self.max_seconds,
            "max_retries": self.max_retries,
            "max_families": self.max_families,
            "max_harness": self.max_harness,
            "expired": self.expired(),
            "spend_cap": None,
            "spend_cap_note": _SPEND_CAP_NOTE,
        }


def merge_budget(base: RunBudget, overrides: dict[str, Any]) -> RunBudget:
    """Build a new budget from ``base`` ceilings plus allowed overrides.

    Only ``max_steps``, ``max_seconds``, ``max_retries``, ``max_families``,
    and ``max_harness`` may be overridden. Unknown keys raise ``BudgetError``.
    Progress counters on the result are zero and the monotonic clock restarts.
    ``validate`` runs before the budget is returned.
    """
    if not isinstance(overrides, dict):
        raise BudgetError("budget overrides must be a dict")
    unknown = [key for key in overrides if key not in _OVERRIDE_KEYS]
    if unknown:
        rendered = ", ".join(sorted(str(key) for key in unknown))
        raise BudgetError(f"unknown budget override: {rendered}")
    budget = RunBudget(
        max_steps=overrides.get("max_steps", base.max_steps),
        max_seconds=overrides.get("max_seconds", base.max_seconds),
        max_retries=overrides.get("max_retries", base.max_retries),
        max_families=overrides.get("max_families", base.max_families),
        max_harness=overrides.get("max_harness", base.max_harness),
        steps=0,
        retries=0,
    )
    budget.validate()
    return budget
