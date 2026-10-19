"""Formal-lane hooks: locale pinned per test, not at collection import time."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _formal_process_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hypothesis formal shards must not depend on the runner's TZ/LC_ALL."""
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("LC_ALL", "C")
