"""The parent-death wrapper must resolve bare command names like ``Popen`` does."""

from __future__ import annotations

import shutil
import sys

import pytest

from fx1.interactive.procs import ProcLimits, run_owned

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="parent-death wrapper is Linux-only"
)


@linux_only
def test_bare_command_name_resolves_through_path() -> None:
    if shutil.which("echo") is None:
        pytest.skip("no echo on PATH")
    result = run_owned(["echo", "bare-name-ok"], ProcLimits(timeout_s=30))
    assert result.exit_code == 0
    assert result.spawn_error is None
    assert result.stdout.strip() == "bare-name-ok"


@linux_only
def test_absolute_path_still_runs() -> None:
    result = run_owned([sys.executable, "-c", "print('abs-ok')"], ProcLimits(timeout_s=30))
    assert result.exit_code == 0
    assert result.stdout.strip() == "abs-ok"


@linux_only
def test_missing_binary_is_a_spawn_failure_not_a_traceback() -> None:
    result = run_owned(["definitely-not-a-real-binary-xyz"], ProcLimits(timeout_s=30))
    assert result.exit_code == 127
    assert "Traceback" not in result.stderr
