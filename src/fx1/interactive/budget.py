"""Per-run resource budgets for the interactive harness.

A :class:`RunBudget` is the configured ceiling. A :class:`BudgetLedger`
charges against it, thread-safely, and fails closed: a charge that would
cross a limit raises :class:`BudgetExceeded` and is not recorded. Prompt
tokens stay uncapped when the provider limit is unknown so the ledger never
invents a spend ceiling.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_ENV_WALL = "FX1_BUDGET_WALL_S"
_ENV_HARNESS_STEPS = "FX1_BUDGET_HARNESS_STEPS"
_ENV_MODEL_CALLS = "FX1_BUDGET_MODEL_CALLS"
_ENV_RETRIES = "FX1_BUDGET_RETRIES"
_ENV_WEB_SOURCES = "FX1_BUDGET_WEB_SOURCES"
_ENV_PROMPT_TOKENS = "FX1_BUDGET_PROMPT_TOKENS"

_FIELD_TO_ENV: dict[str, str] = {
    "max_wall_s": _ENV_WALL,
    "max_harness_steps": _ENV_HARNESS_STEPS,
    "max_model_calls": _ENV_MODEL_CALLS,
    "max_retries_per_call": _ENV_RETRIES,
    "max_web_sources": _ENV_WEB_SOURCES,
    "max_prompt_tokens": _ENV_PROMPT_TOKENS,
}


def _plain(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    rendered = format(number, ".6f").rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _env_float(source: Mapping[str, str], var: str, default: float) -> float:
    if var not in source:
        return default
    raw = source[var]
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid {var}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"invalid {var}={raw!r}")
    return value


def _env_int(source: Mapping[str, str], var: str, default: int) -> int:
    if var not in source:
        return default
    raw = source[var]
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid {var}={raw!r}") from exc


def _env_optional_int(source: Mapping[str, str], var: str, default: int | None) -> int | None:
    if var not in source:
        return default
    raw = source[var]
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid {var}={raw!r}") from exc


def _named_env_error(exc: ValueError) -> ValueError:
    message = str(exc)
    for field, var in _FIELD_TO_ENV.items():
        if message.startswith(field):
            return ValueError(f"invalid {var}: {message}")
    return ValueError(message)


@dataclass(frozen=True)
class RunBudget:
    """Ceilings for one interactive run. ``None`` prompt tokens means uncapped."""

    max_wall_s: float = 1800.0
    max_harness_steps: int = 4
    max_model_calls: int = 8
    max_retries_per_call: int = 2
    max_web_sources: int = 8
    max_prompt_tokens: int | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` when a ceiling is not a usable positive bound."""
        wall = self.max_wall_s
        if isinstance(wall, bool) or not isinstance(wall, int | float):
            raise ValueError("max_wall_s must be positive")
        if not math.isfinite(float(wall)) or wall <= 0:
            raise ValueError("max_wall_s must be positive")
        _require_count("max_harness_steps", self.max_harness_steps, allow_zero=False)
        _require_count("max_model_calls", self.max_model_calls, allow_zero=False)
        _require_count("max_retries_per_call", self.max_retries_per_call, allow_zero=True)
        _require_count("max_web_sources", self.max_web_sources, allow_zero=False)
        _require_prompt_tokens(self.max_prompt_tokens)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RunBudget:
        """Build a budget from ``FX1_BUDGET_*`` variables.

        Unset variables keep the dataclass defaults. A present but unusable
        value raises ``ValueError`` naming that variable. The result is
        validated.
        """
        source: Mapping[str, str] = os.environ if environ is None else environ
        base = cls()
        budget = cls(
            max_wall_s=_env_float(source, _ENV_WALL, base.max_wall_s),
            max_harness_steps=_env_int(source, _ENV_HARNESS_STEPS, base.max_harness_steps),
            max_model_calls=_env_int(source, _ENV_MODEL_CALLS, base.max_model_calls),
            max_retries_per_call=_env_int(source, _ENV_RETRIES, base.max_retries_per_call),
            max_web_sources=_env_int(source, _ENV_WEB_SOURCES, base.max_web_sources),
            max_prompt_tokens=_env_optional_int(source, _ENV_PROMPT_TOKENS, base.max_prompt_tokens),
        )
        try:
            budget.validate()
        except ValueError as exc:
            raise _named_env_error(exc) from exc
        return budget

    def describe(self) -> str:
        if self.max_prompt_tokens is None:
            tokens = "prompt tokens: uncapped (provider limits unknown)"
        else:
            tokens = f"prompt tokens: {self.max_prompt_tokens}"
        return (
            f"wall {_plain(self.max_wall_s)}s · "
            f"harness steps {self.max_harness_steps} · "
            f"model calls {self.max_model_calls} · "
            f"retries/call {self.max_retries_per_call} · "
            f"web sources {self.max_web_sources} · "
            f"{tokens}"
        )


def _require_count(name: str, value: int, *, allow_zero: bool) -> None:
    legal = isinstance(value, int) and not isinstance(value, bool)
    if not legal or value < 0 or (value == 0 and not allow_zero):
        if allow_zero:
            raise ValueError(f"{name} must be >= 0")
        raise ValueError(f"{name} must be positive")


def _require_prompt_tokens(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_prompt_tokens must be positive or None")


class BudgetExceeded(RuntimeError):
    """A charge would have crossed a single budget dimension."""

    def __init__(self, dimension: str, limit: float | int, used: float | int) -> None:
        self.dimension = dimension
        self.limit = limit
        self.used = used
        super().__init__(f"budget exceeded: {dimension} {used} > {limit}")


class BudgetLedger:
    """Thread-safe per-run accounting; every charge checks the limit first and fails closed."""

    def __init__(self, budget: RunBudget, *, clock: Callable[[], float] = time.monotonic) -> None:
        budget.validate()
        self._budget = budget
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._step_ids: list[str] = []
        self._model_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._retries: dict[str, int] = {}
        self._web_sources = 0

    @property
    def budget(self) -> RunBudget:
        return self._budget

    def _elapsed(self) -> float:
        return max(0.0, self._clock() - self._started)

    def elapsed_s(self) -> float:
        with self._lock:
            return self._elapsed()

    def remaining_s(self) -> float:
        with self._lock:
            return max(0.0, self._budget.max_wall_s - self._elapsed())

    def deadline_reached(self) -> bool:
        with self._lock:
            return self._elapsed() >= self._budget.max_wall_s

    def check_wall(self) -> None:
        with self._lock:
            elapsed = self._elapsed()
            limit = self._budget.max_wall_s
            if elapsed >= limit:
                raise BudgetExceeded("wall_s", limit, elapsed)

    def charge_harness_step(self, step_id: str) -> None:
        with self._lock:
            limit = self._budget.max_harness_steps
            used = len(self._step_ids) + 1
            if used > limit:
                raise BudgetExceeded("harness_steps", limit, used)
            self._step_ids.append(step_id)

    def charge_model_call(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        prompt = 0 if prompt_tokens is None else prompt_tokens
        completion = 0 if completion_tokens is None else completion_tokens
        if prompt < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if completion < 0:
            raise ValueError("completion_tokens must be >= 0")
        with self._lock:
            call_limit = self._budget.max_model_calls
            calls = self._model_calls + 1
            if calls > call_limit:
                raise BudgetExceeded("model_calls", call_limit, calls)
            token_limit = self._budget.max_prompt_tokens
            prompt_total = self._prompt_tokens + prompt
            if token_limit is not None and prompt_total > token_limit:
                raise BudgetExceeded("prompt_tokens", token_limit, prompt_total)
            self._model_calls = calls
            self._prompt_tokens = prompt_total
            self._completion_tokens += completion

    def charge_retry(self, call_id: str) -> None:
        with self._lock:
            limit = self._budget.max_retries_per_call
            used = self._retries.get(call_id, 0) + 1
            if used > limit:
                raise BudgetExceeded("retries", limit, used)
            self._retries[call_id] = used

    def charge_web_source(self) -> None:
        with self._lock:
            limit = self._budget.max_web_sources
            used = self._web_sources + 1
            if used > limit:
                raise BudgetExceeded("web_sources", limit, used)
            self._web_sources = used

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = self._elapsed()
            return {
                "elapsed_s": elapsed,
                "remaining_s": max(0.0, self._budget.max_wall_s - elapsed),
                "harness_steps": {
                    "used": len(self._step_ids),
                    "limit": self._budget.max_harness_steps,
                    "ids": list(self._step_ids),
                },
                "model_calls": {
                    "used": self._model_calls,
                    "limit": self._budget.max_model_calls,
                },
                "tokens": {
                    "prompt": self._prompt_tokens,
                    "completion": self._completion_tokens,
                    "prompt_limit": self._budget.max_prompt_tokens,
                },
                "retries": dict(self._retries),
                "web_sources": {
                    "used": self._web_sources,
                    "limit": self._budget.max_web_sources,
                },
            }

    def describe(self) -> str:
        with self._lock:
            elapsed = self._elapsed()
            return (
                f"budget: {_plain(elapsed)}s/{_plain(self._budget.max_wall_s)}s wall · "
                f"{len(self._step_ids)}/{self._budget.max_harness_steps} harness · "
                f"{self._model_calls}/{self._budget.max_model_calls} model calls · "
                f"tokens p={self._prompt_tokens} c={self._completion_tokens} · "
                f"web {self._web_sources}/{self._budget.max_web_sources}"
            )
