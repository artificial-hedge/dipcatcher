"""Hard ceilings for one chat turn, superpower run, or exec step.

Counts are operational limits. Token unit price is unknown, so a token ceiling
is not a spend cap and calls that omit provider ``usage`` are counted as
unmetered rather than assigned a fabricated total.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

BudgetKind = Literal["wall", "steps", "model_calls", "retries", "tokens"]

_BUDGET_FIELDS = (
    "max_wall_s",
    "max_steps",
    "max_model_calls",
    "max_retries_per_step",
    "max_total_tokens",
)
_SNAPSHOT_VERSION = 1


def _opt_wall(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_wall_s must be a finite number or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("max_wall_s must be a finite number >= 0")
    return number


def _opt_count(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int or null")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an int >= 0")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an int >= 1")
    return value


def _nonneg_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number >= 0")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite number >= 0")
    return number


def _token_count(value: Any, label: str) -> int:
    return _nonneg_int(value, label)


def _usage_tokens(usage: Mapping[str, int] | None) -> int | None:
    """Return accounted tokens, or ``None`` when the provider omitted usage."""
    if usage is None:
        return None
    data = _as_mapping(usage, "usage")
    if "total_tokens" in data:
        return _token_count(data["total_tokens"], "total_tokens")
    if "prompt_tokens" not in data and "completion_tokens" not in data:
        return None
    prompt = _token_count(data.get("prompt_tokens", 0), "prompt_tokens")
    completion = _token_count(data.get("completion_tokens", 0), "completion_tokens")
    return prompt + completion


def _step_label(step: str) -> str:
    if not isinstance(step, str) or not step.strip():
        raise ValueError("step must be a non-empty string")
    return step.strip()


def _retry_map(value: Any) -> dict[str, int]:
    data = _as_mapping(value, "retries")
    out: dict[str, int] = {}
    for key, count in data.items():
        if not isinstance(key, str):
            raise ValueError("retry step names must be strings")
        out[_step_label(key)] = _nonneg_int(count, "retries")
    return out


def _fmt(value: float | int) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, float):
        return str(int(value))
    return str(value)


def _dim(label: str, used: float | int, limit: float | int | None) -> str:
    cap = "unbounded" if limit is None else _fmt(limit)
    return f"{label} {_fmt(used)}/{cap}"


def _retry_text(retries: Mapping[str, int], limit: int | None) -> str:
    cap = "unbounded" if limit is None else str(limit)
    if not retries:
        return f"retries 0 (max {cap} per step)"
    rendered = ",".join(f"{name}={count}" for name, count in sorted(retries.items()))
    return f"retries {rendered} (max {cap} per step)"


def _token_text(used: int, limit: int | None, unmetered: int) -> str:
    cap = "unbounded" if limit is None else str(limit)
    return (
        f"tokens {used}/{cap} (tokens only; unit price is unknown so no spend cap is implied); "
        f"{unmetered} calls reported no usage"
    )


@dataclass(frozen=True)
class UsageLimits:
    """Ceilings for one run. ``None`` leaves that dimension unconstrained.

    These limits do not price tokens. A token ceiling is not a spend cap.
    """

    max_wall_s: float | None = None
    max_steps: int | None = None
    max_model_calls: int | None = None
    max_retries_per_step: int | None = None
    max_total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_wall_s": self.max_wall_s,
            "max_steps": self.max_steps,
            "max_model_calls": self.max_model_calls,
            "max_retries_per_step": self.max_retries_per_step,
            "max_total_tokens": self.max_total_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UsageLimits:
        """Validate types and ranges. Unknown keys are rejected."""
        data = _as_mapping(payload, "budget")
        unknown = sorted(set(data) - set(_BUDGET_FIELDS))
        if unknown:
            raise ValueError(f"unknown budget fields: {unknown}")
        return cls(
            max_wall_s=_opt_wall(data.get("max_wall_s")),
            max_steps=_opt_count(data.get("max_steps"), "max_steps"),
            max_model_calls=_opt_count(data.get("max_model_calls"), "max_model_calls"),
            max_retries_per_step=_opt_count(
                data.get("max_retries_per_step"), "max_retries_per_step"
            ),
            max_total_tokens=_opt_count(data.get("max_total_tokens"), "max_total_tokens"),
        )


# Operational defaults for the console. They are not prices and not a spend cap.
PRESET_CHAT_TURN = UsageLimits(
    max_wall_s=180.0,
    max_steps=6,
    max_model_calls=4,
    max_retries_per_step=1,
    max_total_tokens=24_000,
)
PRESET_SUPERPOWER = UsageLimits(
    max_wall_s=900.0,
    max_steps=16,
    max_model_calls=8,
    max_retries_per_step=2,
    max_total_tokens=80_000,
)
PRESET_EXEC = UsageLimits(
    max_wall_s=300.0,
    max_steps=2,
    max_model_calls=1,
    max_retries_per_step=0,
    max_total_tokens=4_000,
)


class BudgetExceeded(RuntimeError):
    """A hard ceiling was crossed. ``used`` may be larger than ``limit``."""

    def __init__(self, kind: BudgetKind, limit: float, used: float) -> None:
        self.kind: BudgetKind = kind
        self.limit = limit
        self.used = used
        super().__init__(f"budget exceeded: {kind} used {used} limit {limit}")


class UsageTracker:
    """Thread-safe counters for one :class:`UsageLimits`.

    Wall time starts at :meth:`start` on the injected monotonic clock.
    :meth:`charge_usage` never invents a token total: a missing ``usage``
    object increments ``unmetered_calls``. ``charge_*`` and :meth:`check`
    raise :class:`BudgetExceeded` once a dimension is over its ceiling.
    """

    def __init__(
        self,
        budget: UsageLimits,
        *,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(budget, UsageLimits):
            raise TypeError("budget must be a UsageLimits")
        self._budget = budget
        self._monotonic: Callable[[], float] = monotonic or time.monotonic
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._steps = 0
        self._model_calls = 0
        self._retries: dict[str, int] = {}
        self._tokens = 0
        self._unmetered = 0

    @property
    def budget(self) -> UsageLimits:
        return self._budget

    def start(self) -> None:
        """Begin the wall clock. A second call leaves the original start in place."""
        with self._lock:
            if self._started_at is None:
                self._started_at = float(self._monotonic())

    def remaining_s(self) -> float | None:
        """Seconds left on the wall ceiling, or ``None`` when wall is unconstrained."""
        with self._lock:
            return self._remaining_s()

    def deadline(self) -> float | None:
        """Monotonic timestamp when the wall ceiling is reached, if the clock is running."""
        with self._lock:
            return self._deadline()

    def charge_step(self, n: int = 1) -> None:
        count = _positive_int(n, "n")
        with self._lock:
            self._steps += count
            self._check()

    def charge_model_call(self) -> None:
        with self._lock:
            self._model_calls += 1
            self._check()

    def charge_retry(self, step: str) -> None:
        label = _step_label(step)
        with self._lock:
            self._retries[label] = self._retries.get(label, 0) + 1
            self._check()

    def charge_usage(self, usage: Mapping[str, int] | None) -> None:
        """Add ``total_tokens``, else prompt+completion. ``None`` is unmetered."""
        tokens = _usage_tokens(usage)
        with self._lock:
            if tokens is None:
                self._unmetered += 1
            else:
                self._tokens += tokens
            self._check()

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` when any dimension is over its ceiling."""
        with self._lock:
            self._check()

    def request_max_tokens(self, wanted: int | None = None) -> int | None:
        """Cap a provider ``max_tokens`` by tokens still unused.

        ``None`` in and ``None`` out means the token dimension is unconstrained.
        A exhausted ceiling raises :class:`BudgetExceeded` instead of requesting
        a non-positive completion length.
        """
        if wanted is not None:
            wanted = _nonneg_int(wanted, "wanted")
        with self._lock:
            return self._capped_tokens(wanted)

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready counters. Wall progress is elapsed seconds, not a raw monotonic value."""
        with self._lock:
            return {
                "version": _SNAPSHOT_VERSION,
                "budget": self._budget.to_dict(),
                "started": self._started_at is not None,
                "elapsed_s": self._elapsed(),
                "steps": self._steps,
                "model_calls": self._model_calls,
                "retries": dict(self._retries),
                "total_tokens": self._tokens,
                "unmetered_calls": self._unmetered,
            }

    @classmethod
    def restore(
        cls,
        payload: Mapping[str, Any],
        *,
        monotonic: Callable[[], float] | None = None,
    ) -> UsageTracker:
        """Rebuild a tracker. Elapsed wall time is rebased onto ``monotonic``."""
        data = _snapshot_fields(payload)
        budget = UsageLimits.from_dict(_as_mapping(data["budget"], "budget"))
        tracker = cls(budget, monotonic=monotonic)
        elapsed = _nonneg_float(data["elapsed_s"], "elapsed_s")
        with tracker._lock:
            _restore_into(tracker, data, elapsed)
        return tracker

    def describe(self) -> str:
        """Honest one-line status. Always states that token price is unknown."""
        snap = self.snapshot()
        budget = self._budget
        wall = _dim("wall_s", snap["elapsed_s"], budget.max_wall_s)
        if not snap["started"]:
            wall += " (not started)"
        parts = [
            wall,
            _dim("steps", snap["steps"], budget.max_steps),
            _dim("model_calls", snap["model_calls"], budget.max_model_calls),
            _retry_text(snap["retries"], budget.max_retries_per_step),
            _token_text(snap["total_tokens"], budget.max_total_tokens, snap["unmetered_calls"]),
        ]
        return "; ".join(parts)

    def _remaining_s(self) -> float | None:
        limit = self._budget.max_wall_s
        if limit is None:
            return None
        if self._started_at is None:
            return float(limit)
        return max(0.0, float(limit) - self._elapsed())

    def _deadline(self) -> float | None:
        limit = self._budget.max_wall_s
        if limit is None or self._started_at is None:
            return None
        now = float(self._monotonic())
        # A backward clock must not advertise more slack than the ceiling.
        start = self._started_at if now >= self._started_at else now
        return start + float(limit)

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, float(self._monotonic()) - self._started_at)

    def _check(self) -> None:
        self._check_wall()
        _raise_if_over("steps", self._steps, self._budget.max_steps)
        _raise_if_over("model_calls", self._model_calls, self._budget.max_model_calls)
        _raise_if_over("tokens", self._tokens, self._budget.max_total_tokens)
        self._check_retries()

    def _check_wall(self) -> None:
        limit = self._budget.max_wall_s
        if limit is None or self._started_at is None:
            return
        elapsed = float(self._monotonic()) - self._started_at
        if elapsed >= float(limit):
            raise BudgetExceeded("wall", float(limit), float(elapsed))

    def _check_retries(self) -> None:
        limit = self._budget.max_retries_per_step
        if limit is None:
            return
        for used in self._retries.values():
            if used > limit:
                raise BudgetExceeded("retries", float(limit), float(used))

    def _capped_tokens(self, wanted: int | None) -> int | None:
        limit = self._budget.max_total_tokens
        if limit is None:
            return wanted
        remaining = limit - self._tokens
        if remaining <= 0:
            raise BudgetExceeded("tokens", float(limit), float(self._tokens))
        if wanted is None:
            return remaining
        return min(wanted, remaining)


def _raise_if_over(kind: BudgetKind, used: int, limit: int | None) -> None:
    if limit is not None and used > limit:
        raise BudgetExceeded(kind, float(limit), float(used))


_SNAPSHOT_FIELDS = frozenset(
    {
        "version",
        "budget",
        "started",
        "elapsed_s",
        "steps",
        "model_calls",
        "retries",
        "total_tokens",
        "unmetered_calls",
    }
)


def _snapshot_fields(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = _as_mapping(payload, "snapshot")
    missing = _SNAPSHOT_FIELDS - set(data)
    if missing:
        raise ValueError("snapshot is missing required fields")
    if data["version"] != _SNAPSHOT_VERSION:
        raise ValueError("unsupported budget snapshot version")
    if not isinstance(data["started"], bool):
        raise ValueError("snapshot started must be a bool")
    return data


def _restore_into(tracker: UsageTracker, data: Mapping[str, Any], elapsed: float) -> None:
    if data["started"]:
        tracker._started_at = float(tracker._monotonic()) - elapsed
    tracker._steps = _nonneg_int(data["steps"], "steps")
    tracker._model_calls = _nonneg_int(data["model_calls"], "model_calls")
    tracker._tokens = _nonneg_int(data["total_tokens"], "total_tokens")
    tracker._unmetered = _nonneg_int(data["unmetered_calls"], "unmetered_calls")
    tracker._retries = _retry_map(data["retries"])
