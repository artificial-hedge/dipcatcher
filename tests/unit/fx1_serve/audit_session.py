"""Run one audit bench with the same env scrub the test fixtures use.

The determinism tests compare a second, live bench against this result.
Both executions call the bench. The session fixture holds the first one
so the determinism test pays for a single run.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any


def run_with_clean_env(
    bench: Callable[[], dict[str, Any]],
    env_keys: Iterable[str],
) -> dict[str, Any]:
    keys = tuple(env_keys)
    saved = {name: os.environ.get(name) for name in keys}
    for name in keys:
        os.environ.pop(name, None)
    try:
        return bench()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
