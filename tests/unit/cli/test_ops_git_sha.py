"""sim-live's git probe must finish, and a failed probe is not a revision."""

from __future__ import annotations

import subprocess

import pytest

from quant_fund.cli import ops_cmds


def test_probe_git_sha_timeout_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") == ops_cmds._GIT_SHA_TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=ops_cmds._GIT_SHA_TIMEOUT_S)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ops_cmds._probe_git_sha() is None


def test_probe_git_sha_nonzero_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="deadbeef\n", stderr="no")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ops_cmds._probe_git_sha() is None


def test_probe_git_sha_returns_short_head(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ops_cmds._probe_git_sha() == "abc1234"
