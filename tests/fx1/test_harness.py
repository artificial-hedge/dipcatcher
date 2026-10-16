"""Harness tests: registry fail-closed, runner injection, config containment.

Subprocess cases use ``sys.executable -c`` through
:func:`fx1.harness.make_subprocess_runner` so they stay offline and bounded.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fx1.harness import (
    Harness,
    HarnessResult,
    HarnessRole,
    RunnerOutcome,
    _signal_windows_tree,
    _subprocess_runner,
    default_runner_available,
    make_subprocess_runner,
)


def _fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, f"ran: {' '.join(argv)}", ""


def test_registry_covers_all_four_roles():
    harness = Harness(runner=_fake_runner)
    roles = {c.role for c in harness.list_commands()}
    assert roles == {
        HarnessRole.DATA_ENGINE,
        HarnessRole.EVALUATION,
        HarnessRole.VERIFICATION,
        HarnessRole.MODEL_TRAINING,
    }


def test_registry_covers_full_lab_surface():
    names = {c.name for c in Harness(runner=_fake_runner).list_commands()}
    expected = {
        "doctor",
        "verify-research",
        "validate",
        "monitor",
        "ingest",
        "collect",
        "build-features",
        "build-labels",
        "research",
        "northset",
        "backtest",
        "forecast",
        "kronos-forecast",
        "candle-book",
        "kyle-ofi",
        "session-book",
        "vendor-book-map",
        "book-panel",
        "report",
        "tearsheet",
        "train",
        "optimize",
        "paper",
    }
    assert expected <= names
    # Hidden/network-serving surfaces are deliberately unreachable by fx-1.
    assert "lab" not in names and "api" not in names


def test_role_filter():
    harness = Harness(runner=_fake_runner)
    verification = harness.list_commands(role=HarnessRole.VERIFICATION)
    assert verification and all(c.role == HarnessRole.VERIFICATION for c in verification)


def test_unregistered_command_fails_closed():
    harness = Harness(runner=_fake_runner)
    with pytest.raises(KeyError, match="not a registered harness command"):
        harness.run("live-trading")


def test_run_dispatches_through_injected_runner():
    harness = Harness(runner=_fake_runner)
    result = harness.run("verify-research")
    assert result.ok
    assert "verify-research" in result.stdout


def test_config_contained_to_configs_dir(tmp_path: Path):
    harness = Harness(runner=_fake_runner)
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("research", config=tmp_path / "evil.yaml")


def test_default_runner_targets_dipcatcher_cli():
    harness = Harness()
    # Registry argv always targets the dipcatcher harness CLI surface.
    assert all(isinstance(c.argv, list) and c.argv for c in harness.list_commands())


def _run_script(
    script: str,
    *,
    timeout_s: int = 30,
    cancel_event: threading.Event | None = None,
    grace_s: float = 5.0,
    poll_s: float = 0.05,
    max_output_chars: int = 2_000_000,
    command: str = "doctor",
) -> HarnessResult:
    inner = make_subprocess_runner(
        sys.executable,
        grace_s=grace_s,
        poll_s=poll_s,
        max_output_chars=max_output_chars,
    )

    def runner(
        argv: list[str],
        budget_s: int,
        event: threading.Event | None,
    ) -> RunnerOutcome:
        del argv
        return inner(["-c", script], budget_s, event)

    return Harness(cancellable_runner=runner).run(
        command,
        cancel_event=cancel_event,
        timeout_s=timeout_s,
    )


def test_ok_requires_clean_exit() -> None:
    clean = HarnessResult(command="doctor", exit_code=0, stdout="", stderr="")
    assert clean.ok
    assert clean.cancelled is False
    assert clean.timed_out is False
    assert clean.duration_s == 0.0
    cancelled = HarnessResult(command="doctor", exit_code=0, stdout="", stderr="", cancelled=True)
    assert not cancelled.ok
    timed_out = HarnessResult(command="doctor", exit_code=0, stdout="", stderr="", timed_out=True)
    assert not timed_out.ok
    failed = HarnessResult(command="doctor", exit_code=2, stdout="", stderr="")
    assert not failed.ok


def test_subprocess_normal_completion_and_stderr() -> None:
    script = "import sys; print('hello'); print('warn', file=sys.stderr)"
    runner = make_subprocess_runner(sys.executable)
    outcome = runner(["-c", script], 30, None)
    assert outcome.exit_code == 0
    assert outcome.stdout.strip() == "hello"
    assert outcome.stderr.strip() == "warn"
    assert outcome.cancelled is False
    assert outcome.timed_out is False

    result = _run_script(script)
    assert result.ok
    assert result.stdout.strip() == "hello"
    assert result.stderr.strip() == "warn"
    assert result.duration_s > 0
    assert result.cancelled is False
    assert result.timed_out is False


def test_cancel_returns_partial_stdout() -> None:
    event = threading.Event()
    timer = threading.Timer(0.3, event.set)
    timer.daemon = True
    started = time.monotonic()
    timer.start()
    try:
        result = _run_script(
            "print('started', flush=True); import time; time.sleep(30)",
            timeout_s=8,
            cancel_event=event,
            grace_s=1.0,
        )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    assert elapsed < 6.0  # far below the child's 30s sleep; slack for slow CI runners
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.ok is False
    assert "started" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows taskkill process-tree contract")
def test_windows_cancel_stops_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived.txt"
    child = (
        "import time; from pathlib import Path; "
        f"time.sleep(2); Path({str(marker)!r}).write_text('orphan')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "print('child started', flush=True); time.sleep(30)"
    )
    event = threading.Event()
    timer = threading.Timer(1.0, event.set)
    timer.daemon = True
    timer.start()
    try:
        result = _run_script(parent, timeout_s=8, cancel_event=event, grace_s=0.5)
    finally:
        timer.cancel()

    assert result.cancelled is True
    assert "child started" in result.stdout
    time.sleep(2.5)
    assert not marker.exists()


def test_windows_tree_termination_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[bool] = []

    class Child:
        pid = 123

        def kill(self) -> None:
            killed.append(True)

    monkeypatch.setattr(
        "fx1.harness.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )
    with pytest.raises(RuntimeError, match="process-tree termination failed"):
        _signal_windows_tree(Child())  # type: ignore[arg-type]
    assert killed == [True]


def test_deadline_marks_timed_out() -> None:
    started = time.monotonic()
    result = _run_script("import time; time.sleep(30)", timeout_s=1, grace_s=1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 8.0  # 1s deadline + 1s grace, with slack for slow CI runners
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.ok is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group signals")
def test_sigterm_ignored_child_is_killed() -> None:
    event = threading.Event()
    timer = threading.Timer(0.4, event.set)
    timer.daemon = True
    started = time.monotonic()
    timer.start()
    try:
        result = _run_script(
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('x', flush=True); "
            "time.sleep(30)",
            timeout_s=8,
            cancel_event=event,
            grace_s=0.5,
        )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    assert elapsed < 8.0  # SIGTERM ignored; SIGKILL after 0.5s grace, with CI slack
    assert result.cancelled is True
    assert result.ok is False
    assert result.exit_code == -signal.SIGKILL
    assert "x" in result.stdout


def test_large_output_is_truncated_without_deadlock() -> None:
    script = "import sys; sys.stdout.write('A' * 3_000_000); sys.stderr.write('B' * 3_000_000)"
    started = time.monotonic()
    result = _run_script(script, timeout_s=8, grace_s=0.5, max_output_chars=100_000)
    assert time.monotonic() - started < 15.0
    marker = "\n[output truncated at 100000 chars]"
    assert result.exit_code == 0
    assert result.ok is True
    assert result.cancelled is False
    assert result.timed_out is False
    assert result.stdout.endswith(marker)
    assert result.stderr.endswith(marker)
    assert result.stdout[: -len(marker)] == "A" * 100_000
    assert result.stderr[: -len(marker)] == "B" * 100_000


def test_output_truncation_boundary() -> None:
    runner = make_subprocess_runner(sys.executable, max_output_chars=100)
    exact = runner(["-c", "import sys; sys.stdout.write('A' * 100)"], 10, None)
    assert exact.exit_code == 0
    assert exact.stdout == "A" * 100
    assert exact.timed_out is False
    over = runner(["-c", "import sys; sys.stdout.write('A' * 101)"], 10, None)
    assert over.exit_code == 0
    assert over.stdout == ("A" * 100) + "\n[output truncated at 100 chars]"


def test_legacy_runner_wrapper() -> None:
    seen: list[tuple[list[str], int]] = []

    def legacy(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen.append((list(argv), timeout_s))
        return 0, "out", "err"

    event = threading.Event()
    event.set()
    result = Harness(runner=legacy).run(
        "monitor",
        extra_args=["--x"],
        cancel_event=event,
        timeout_s=40,
    )
    assert result.ok
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.cancelled is False
    assert result.timed_out is False
    assert result.duration_s >= 0
    assert seen == [(["monitor", "--x"], 40)]


def test_legacy_subprocess_runner_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(*_args: object, **_kwargs: object) -> object:
        def run(
            argv: list[str],
            timeout_s: int,
            cancel_event: threading.Event | None,
        ) -> RunnerOutcome:
            assert argv == ["doctor"]
            assert timeout_s == 3
            assert cancel_event is None
            return RunnerOutcome(exit_code=7, stdout="o", stderr="e", timed_out=True)

        return run

    monkeypatch.setattr("fx1.harness.make_subprocess_runner", factory)
    assert _subprocess_runner(["doctor"], 3) == (7, "o", "e")


def test_missing_executable_fails_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.harness.shutil.which", lambda _name: None)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess spawned")

    monkeypatch.setattr("fx1.harness.subprocess.Popen", boom)
    harness = Harness()
    assert harness.list_commands()
    with pytest.raises(RuntimeError, match="dipcatcher executable not found on PATH") as caught:
        harness.run("doctor")
    assert str(caught.value) == (
        "dipcatcher executable not found on PATH; install the fx-1 distribution "
        "(uv sync) or inject a runner"
    )


def test_injected_runner_skips_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.harness.shutil.which", lambda _name: None)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess spawned")

    monkeypatch.setattr("fx1.harness.subprocess.Popen", boom)
    legacy = Harness(runner=_fake_runner).run("doctor")
    assert legacy.ok
    assert legacy.cancelled is False
    assert legacy.timed_out is False

    def cancellable(
        argv: list[str],
        timeout_s: int,
        cancel_event: threading.Event | None,
    ) -> RunnerOutcome:
        del argv, timeout_s, cancel_event
        return RunnerOutcome(exit_code=0, stdout="c", stderr="")

    injected = Harness(cancellable_runner=cancellable).run("doctor")
    assert injected.ok
    assert injected.stdout == "c"


def test_default_runner_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fx1.harness.shutil.which",
        lambda name: "/bin/dipcatcher" if name == "dipcatcher" else None,
    )
    assert default_runner_available() is True
    monkeypatch.setattr("fx1.harness.shutil.which", lambda _name: None)
    assert default_runner_available() is False


def test_timeout_s_only_tightens() -> None:
    seen: list[int] = []

    def runner(
        argv: list[str],
        timeout_s: int,
        cancel_event: threading.Event | None,
    ) -> RunnerOutcome:
        del argv, cancel_event
        seen.append(timeout_s)
        return RunnerOutcome(exit_code=0, stdout="", stderr="")

    harness = Harness(cancellable_runner=runner)
    registered = harness.get("train").timeout_s
    harness.run("train", timeout_s=registered + 100)
    harness.run("train", timeout_s=9)
    assert seen == [registered, 9]
    with pytest.raises(ValueError, match="timeout_s"):
        harness.run("train", timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        harness.run("train", timeout_s=-3)
    assert seen == [registered, 9]


def test_run_forwards_cancel_event() -> None:
    seen: list[threading.Event | None] = []

    def runner(
        argv: list[str],
        timeout_s: int,
        cancel_event: threading.Event | None,
    ) -> RunnerOutcome:
        del argv, timeout_s
        seen.append(cancel_event)
        return RunnerOutcome(exit_code=0, stdout="", stderr="")

    event = threading.Event()
    event.set()
    result = Harness(cancellable_runner=runner).run("doctor", cancel_event=event)
    assert result.ok
    assert seen == [event]
