"""POSIX-only stdlib names stay callable under mypy --strict on Windows."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from fx1 import harness
from fx1.flash import store


def test_fchmod_skips_when_the_platform_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(store.os, "fchmod", raising=False)
    store._fchmod(3, 0o600)


def test_posix_signal_kills_the_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = Mock()
    proc.pid = 42
    killpg = Mock()
    getpgid = Mock(return_value=7)
    monkeypatch.setattr(harness.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(harness.os, "getpgid", getpgid, raising=False)
    monkeypatch.setattr(harness.signal, "SIGKILL", 9, raising=False)
    harness._signal_posix(proc, kill=True)
    getpgid.assert_called_once_with(42)
    killpg.assert_called_once_with(7, 9)
    proc.kill.assert_not_called()


def test_posix_signal_falls_back_without_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(harness.os, "killpg", raising=False)
    monkeypatch.delattr(harness.os, "getpgid", raising=False)
    proc = Mock()
    proc.pid = 42
    harness._signal_posix(proc, kill=True)
    proc.kill.assert_called_once_with()


def test_windows_signal_path_does_not_need_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.os, "name", "nt")
    proc = Mock()
    harness._signal_process(proc, kill=False)
    proc.terminate.assert_called_once_with()
    proc.kill.assert_not_called()
