"""The matching core compiler must not hang, and a timeout is not a built library."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quant_fund.market_sim import native


def test_compile_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") == native._COMPILE_TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=native._COMPILE_TIMEOUT_S)

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out"):
        native._compile(tmp_path / "core.c", tmp_path / "core.so")
