"""Child trees: SIGTERM escalation, forks, and the double-fork escapee."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fx1.interactive.procs import (
    EXIT_CANCEL,
    ProcessRegistry,
    ProcLimits,
    install_terminal_stop,
    run_owned,
)
from tests.fx1.chaos_support import HANG_S, pid_alive, record, wait_until

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")


def _py(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


def test_chaos_parent_death_reaps_the_direct_child(tmp_path: Path) -> None:
    """SIGKILL of the owner kills the direct child via the Linux parent-death signal."""
    ready = tmp_path / "ready"
    pidfile = tmp_path / "child.pid"
    parent = subprocess.Popen(
        _py(_PARENT, str(ready), str(pidfile)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = 0
    try:
        wait_until(ready.is_file, HANG_S, "child did not start")
        child = int(pidfile.read_text(encoding="utf-8").strip())
        assert pid_alive(child)
        started = time.monotonic()
        assert parent.pid is not None
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=HANG_S)
        wait_until(lambda: not pid_alive(child), HANG_S, "direct child survived parent SIGKILL")
        elapsed = time.monotonic() - started
        record(
            "parent_death",
            iterations=1,
            passed=1,
            elapsed_s=round(elapsed, 3),
            orphans=0,
            duplicates=0,
        )
    finally:
        if parent.poll() is None and parent.pid:
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
        if child and pid_alive(child):
            os.kill(child, signal.SIGKILL)


def test_chaos_sigterm_ignored_escalates_to_sigkill(tmp_path: Path) -> None:
    """A child and grandchild that ignore SIGTERM are SIGKILLed inside the grace budget."""
    ready = tmp_path / "ready"
    gpath = tmp_path / "grand.pid"
    cancel = __import__("threading").Event()
    result = None
    grand = 0
    try:
        result = _run_until_ready(
            _IGNORE,
            ready,
            gpath,
            cancel,
            ProcLimits(timeout_s=30, kill_grace_s=0.4, cancel_poll_s=0.05),
        )
        grand = int(gpath.read_text(encoding="utf-8").strip()) if gpath.is_file() else 0
        assert result.cancelled is True
        assert result.exit_code == EXIT_CANCEL
        assert result.pid is not None
        assert not pid_alive(result.pid)
        assert grand > 0
        assert not pid_alive(grand)
        assert result.duration_s < 15
        record(
            "sigterm_escalate",
            iterations=1,
            passed=1,
            elapsed_s=round(result.duration_s, 3),
            orphans=0,
            duplicates=0,
        )
    finally:
        if result is not None and result.pid and pid_alive(result.pid):
            os.kill(result.pid, signal.SIGKILL)
        if grand and pid_alive(grand):
            os.kill(grand, signal.SIGKILL)


def test_chaos_double_fork_setsid_escapee_is_measured(tmp_path: Path) -> None:
    """A double-fork plus setsid grandchild is not in the descendant sweep.

    This is the documented residual risk. The test kills that exact pid itself.
    """
    ready = tmp_path / "ready"
    pidfile = tmp_path / "escape.pid"
    cancel = __import__("threading").Event()
    result = None
    escapee = 0
    try:
        result = _run_until_ready(
            _ESCAPE,
            ready,
            pidfile,
            cancel,
            ProcLimits(timeout_s=30, kill_grace_s=0.4, cancel_poll_s=0.05),
        )
        escapee = int(pidfile.read_text(encoding="utf-8").strip())
        assert result.cancelled is True
        assert result.pid is not None
        assert not pid_alive(result.pid)
        still = pid_alive(escapee)
        record(
            "double_fork_setsid",
            iterations=1,
            passed=1,
            escapee_alive_after_cancel=still,
            elapsed_s=round(result.duration_s, 3),
            orphans=1 if still else 0,
            duplicates=0,
        )
        assert still, "expected the setsid escapee to outlive the owned child"
    finally:
        if result is not None and result.pid and pid_alive(result.pid):
            os.kill(result.pid, signal.SIGKILL)
        if escapee and pid_alive(escapee):
            os.kill(escapee, signal.SIGKILL)
        if escapee:
            wait_until(lambda: not pid_alive(escapee), HANG_S, "escapee was not killed")


def test_chaos_terminal_stop_exits_143_and_reaps(tmp_path: Path) -> None:
    """install_terminal_stop is the handler the chat console should arm."""
    ready = tmp_path / "ready"
    pidfile = tmp_path / "child.pid"
    parent = subprocess.Popen(
        _py(_STOP, str(ready), str(pidfile)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = 0
    try:
        wait_until(ready.is_file, HANG_S, "stop child did not start")
        child = int(pidfile.read_text(encoding="utf-8").strip())
        assert parent.pid is not None
        started = time.monotonic()
        os.kill(parent.pid, signal.SIGTERM)
        code = parent.wait(timeout=HANG_S)
        wait_until(lambda: not pid_alive(child), HANG_S, "child survived SIGTERM")
        assert code == 143
        record(
            "terminal_stop",
            iterations=1,
            passed=1,
            exit=code,
            elapsed_s=round(time.monotonic() - started, 3),
            orphans=0,
            duplicates=0,
        )
    finally:
        if parent.poll() is None and parent.pid:
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
        if child and pid_alive(child):
            os.kill(child, signal.SIGKILL)


def _run_until_ready(
    code: str,
    ready: Path,
    arg: Path,
    cancel: threading.Event,
    limits: ProcLimits,
) -> object:
    box: dict[str, object] = {}

    def _target() -> None:
        box["result"] = run_owned(
            _py(code, str(ready), str(arg)),
            limits,
            cancel_event=cancel,
            registry=ProcessRegistry(),
        )

    thread = threading.Thread(target=_target, name="chaos-owned")
    thread.start()
    wait_until(ready.is_file, HANG_S, "owned child did not become ready")
    cancel.set()
    thread.join(HANG_S)
    assert not thread.is_alive()
    return box["result"]


_PARENT = r"""
import sys
from fx1.interactive.procs import ProcLimits, ProcessRegistry, run_owned
ready, pidfile = sys.argv[1], sys.argv[2]
code = (
    "import os,sys,time,pathlib\n"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "pathlib.Path(sys.argv[2]).write_text('1')\n"
    "time.sleep(180)\n"
)
run_owned(
    [sys.executable, "-c", code, pidfile, ready],
    ProcLimits(timeout_s=180, kill_grace_s=1.0),
    registry=ProcessRegistry(),
)
"""

_STOP = r"""
import sys
from fx1.interactive.procs import (
    ProcLimits,
    ProcessRegistry,
    install_terminal_stop,
    run_owned,
)
install_terminal_stop()
ready, pidfile = sys.argv[1], sys.argv[2]
code = (
    "import os,sys,time,pathlib\n"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "pathlib.Path(sys.argv[2]).write_text('1')\n"
    "time.sleep(180)\n"
)
run_owned(
    [sys.executable, "-c", code, pidfile, ready],
    ProcLimits(timeout_s=180, kill_grace_s=1.0),
    registry=ProcessRegistry(),
)
"""

_IGNORE = r"""
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
ready = Path(sys.argv[1])
gpath = Path(sys.argv[2])
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpath.write_text(str(os.getpid()))
    ready.write_text("1")
    time.sleep(180)
    os._exit(0)
while not ready.exists():
    time.sleep(0.01)
time.sleep(180)
"""

_ESCAPE = r"""
import os, sys, time
from pathlib import Path
ready = Path(sys.argv[1])
pidfile = Path(sys.argv[2])
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        pidfile.write_text(str(os.getpid()))
        ready.write_text("1")
        time.sleep(180)
        os._exit(0)
    os._exit(0)
while not ready.exists():
    time.sleep(0.01)
time.sleep(180)
"""


def test_chaos_install_terminal_stop_skips_an_existing_handler() -> None:
    """A handler that is already installed, such as exec's, is not replaced."""
    term = signal.getsignal(signal.SIGTERM)
    hup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        install_terminal_stop(ProcessRegistry())
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, term)
        if hup is not None and hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, hup)
