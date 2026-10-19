"""fx1_serve lane hooks shared across audit batteries."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_numba_jit_for_audit_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
