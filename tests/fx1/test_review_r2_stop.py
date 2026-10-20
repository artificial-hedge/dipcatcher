"""A terminal stop must not retry a step or start another owned process."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fx1.interactive.budgets import RunBudget
from fx1.interactive.procs import (
    REGISTRY,
    ProcessRegistry,
    ProcLimits,
    run_owned,
)
from fx1.interactive.superpower import SuperpowerContext, plan_goal, run_plan
from tests.fx1.capabilities_fixture import sample_catalog


@pytest.fixture(autouse=True)
def _dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))


def _limits() -> ProcLimits:
    return ProcLimits(timeout_s=20.0, kill_grace_s=0.5, cancel_poll_s=0.05)


def _wait_file(path: Path) -> None:
    deadline = time.monotonic() + 20
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.is_file()


def test_stop_cancels_the_live_run_and_refuses_another_spawn(tmp_path: Path) -> None:
    registry = ProcessRegistry()
    ready = tmp_path / "ready"
    code = "import sys,time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('1')\ntime.sleep(30)\n"
    box: dict[str, Any] = {}

    def go() -> None:
        box["result"] = run_owned(
            [sys.executable, "-c", code, str(ready)],
            _limits(),
            registry=registry,
        )

    thread = threading.Thread(target=go)
    thread.start()
    try:
        _wait_file(ready)
        registry.terminate_all(0.5, stop=True)
        thread.join(20)
        assert not thread.is_alive()
        result = box["result"]
        assert result.cancelled is True
        spawned: list[int] = []

        def popen(_argv: list[str], **_kwargs: object) -> None:
            spawned.append(1)
            raise AssertionError("spawned while stopping")

        refused = run_owned(
            ["/bin/true"],
            _limits(),
            registry=registry,
            popen=popen,
        )
        assert refused.cancelled is True
        assert refused.spawn_error == "process registry is stopping"
        assert spawned == []
    finally:
        registry.terminate_all(0.2)
        thread.join(5)
        registry.resume()


def test_reap_without_stop_still_allows_the_next_spawn() -> None:
    registry = ProcessRegistry()
    registry.terminate_all(0.2)
    result = run_owned([sys.executable, "-c", "print('ok')"], _limits(), registry=registry)
    assert result.exit_code == 0
    assert result.cancelled is False
    assert result.spawn_error is None


def test_cancel_then_death_is_not_a_failed_exit(tmp_path: Path) -> None:
    registry = ProcessRegistry()
    ready = tmp_path / "ready"
    code = "import sys,time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('1')\ntime.sleep(30)\n"
    cancel = threading.Event()
    box: dict[str, Any] = {}

    def go() -> None:
        box["result"] = run_owned(
            [sys.executable, "-c", code, str(ready)],
            _limits(),
            registry=registry,
            cancel_event=cancel,
        )

    thread = threading.Thread(target=go)
    thread.start()
    try:
        _wait_file(ready)
        pid = registry.live()[0].pid
        cancel.set()
        os_kill(pid)
        thread.join(20)
        assert not thread.is_alive()
        assert box["result"].cancelled is True
    finally:
        registry.terminate_all(0.2)
        thread.join(5)


def os_kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


class _Flaky:
    def __init__(self) -> None:
        self.n = 0

    def run(self, name: str, extra: list[str] | None = None, **_kwargs: Any) -> SimpleNamespace:
        del name, extra
        self.n += 1
        code = 2 if self.n == 1 else 0
        return SimpleNamespace(
            exit_code=code,
            stdout="DATA_LABEL=SYNTHETIC\nSYNTHETIC\nok\n",
            stderr="transient",
            timed_out=False,
            cancelled=False,
            truncated=False,
        )


def test_stopping_registry_does_not_retry_a_failed_step(tmp_path: Path) -> None:
    flaky = _Flaky()
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    REGISTRY.terminate_all(stop=True)
    try:
        result = run_plan(
            plan,
            SuperpowerContext(
                harness=flaky,
                run_dir=tmp_path / "run",
                budget=RunBudget(max_retries=1, max_steps=8),
                catalog=sample_catalog(),
            ),
        )
    finally:
        REGISTRY.resume()
    assert flaky.n == 1
    vol = next(run for run in result.runs if run.id == "vol-bench")
    assert vol.status == "failed"
