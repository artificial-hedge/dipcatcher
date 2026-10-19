"""Process-env isolation: mutating test modules must not touch import time."""

from __future__ import annotations

import importlib
import os
import sys

from tests.conftest import GUARDED_ENV_BASELINE

_GUARDED = tuple(GUARDED_ENV_BASELINE)

_MUTATING_MODULES = (
    "tests.property.test_replay_determinism",
    "tests.unit.fx1_serve.test_idem_audit",
    "tests.unit.fx1_serve.test_batch_audit",
)


def _snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _GUARDED}


def test_guarded_env_unchanged_after_importing_mutating_modules() -> None:
    before = _snapshot()
    for name in _MUTATING_MODULES:
        sys.modules.pop(name, None)
        importlib.import_module(name)
    after = _snapshot()
    assert after == before, f"import-time env leak: before={before} after={after}"


def test_guarded_env_matches_session_baseline_after_collection() -> None:
    assert _snapshot() == GUARDED_ENV_BASELINE
