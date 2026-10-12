"""Live and adversarial checks for the owned subprocess runner and harness.

These tests spawn real processes, real pipes, and (where noted) a pty.
They set FX1_CONFIG_DIR and FX1_SESSION_DIR to temp directories.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import tracemalloc
from pathlib import Path

import pytest

from fx1.harness import Harness
from fx1.interactive.procs import (
    EXIT_CANCEL,
    EXIT_TIMEOUT,
    REGISTRY,
    TRUNCATION_MARKER,
    ProcessRegistry,
    ProcLimits,
    ProcResult,
    run_owned,
    scrub_env,
)

_SECRET = "sk-r3-live-secret-value"
_ROOT = Path("/workspace")
_HAS_FORK = hasattr(os, "fork") and hasattr(os, "setsid")

_SECRET_NAMES = (
    "moonshot_api_key",
    "MoonShot_Api_Key",
    "DB_PASSWORD",
    "db_password",
    "MY_SECRET",
    "my_secret",
    "AWS_SECRET_ACCESS_KEY",
    "aws_access_key_id",
    "GITHUB_TOKEN",
    "github_token",
    "HF_TOKEN",
    "hf_token",
    "OPENAI_API_KEY",
    "openai_api_key",
    "ANTHROPIC_API_KEY",
    "anthropic_api_key",
)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "fx1-config"
    sess = tmp_path / "fx1-sessions"
    cfg.mkdir()
    sess.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("FX1_SESSION_DIR", str(sess))
    yield
    if REGISTRY.live():
        REGISTRY.terminate_all(grace_s=0.2)


def _py(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


def _limits(**overrides: float | int) -> ProcLimits:
    values: dict[str, float | int] = {"timeout_s": 8.0, "kill_grace_s": 0.4, "cancel_poll_s": 0.02}
    values.update(overrides)
    return ProcLimits(**values)  # type: ignore[arg-type]


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    try:
        text = stat.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    parts = text.rpartition(")")[2].split()
    return not (parts and parts[0] == "Z")


def _kill(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _wait_dead(pid: int, budget: float = 1.5) -> bool:
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


def _readers() -> list[threading.Thread]:
    return [
        item for item in threading.enumerate() if item.name.startswith("proc-") and item.is_alive()
    ]


def _child_env() -> str:
    return "import json,os,sys; json.dump(dict(os.environ), sys.stdout)"


def _runner_capture(store: dict[str, list[str]]):
    def _run(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        store["argv"] = list(argv)
        store["timeout"] = [str(timeout_s)]
        return 0, "ok", ""

    return _run


# ---------------------------------------------------------------------------
# Environment scrubbing
# ---------------------------------------------------------------------------


def test_scrub_drops_secret_names_and_pins_threads(tmp_path: Path) -> None:
    base = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "USER": "ada",
        "TERM": "xterm",
        "SYSTEMROOT": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE",
        "USERPROFILE": r"C:\Users\ada",
    }
    for name in _SECRET_NAMES:
        base[name] = _SECRET
    extra = {
        "HF_TOKEN": _SECRET,
        "OPENAI_API_KEY": _SECRET,
        "CUSTOM_OK": "kept",
        "LD_PRELOAD": "/tmp/evil.so",
        "BASH_ENV": "/tmp/evil-bashrc",
        "PYTHONSTARTUP": "/tmp/evil.py",
        "LD_LIBRARY_PATH": "/tmp/evil-lib",
    }
    env = scrub_env(base, extra=extra)
    result = run_owned(_py(_child_env()), _limits(), env=env, registry=ProcessRegistry())
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    for name in _SECRET_NAMES:
        assert name not in payload
    assert _SECRET not in result.stdout
    assert _SECRET not in result.stderr
    assert payload["HOME"] == str(tmp_path)
    assert payload["LANG"] == "C.UTF-8"
    assert payload["LC_ALL"] == "C.UTF-8"
    assert payload["USER"] == "ada"
    assert payload["SYSTEMROOT"] == r"C:\Windows"
    assert payload["COMSPEC"].endswith("cmd.exe")
    assert payload["PATHEXT"] == ".COM;.EXE"
    assert payload["USERPROFILE"] == r"C:\Users\ada"
    assert payload["CUSTOM_OK"] == "kept"
    for blocked in ("LD_PRELOAD", "BASH_ENV", "PYTHONSTARTUP", "LD_LIBRARY_PATH"):
        assert blocked not in payload
    assert payload["DIPCATCHER_CHILD"] == "1"
    assert payload["PYTHONUNBUFFERED"] == "1"
    assert payload["OMP_NUM_THREADS"] == "1"
    assert payload["MKL_NUM_THREADS"] == "1"
    assert payload["NUMEXPR_NUM_THREADS"] == "1"
    assert payload["VECLIB_MAXIMUM_THREADS"] == "1"
    assert payload["TOKENIZERS_PARALLELISM"] == "false"


def test_thread_pin_keeps_parent_value_and_extra_override() -> None:
    pinned = scrub_env({"PATH": "/bin", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "4"})
    assert pinned["OMP_NUM_THREADS"] == "8"
    assert pinned["MKL_NUM_THREADS"] == "4"
    assert pinned["TOKENIZERS_PARALLELISM"] == "false"
    overridden = scrub_env({"PATH": "/bin"}, extra={"OMP_NUM_THREADS": "3"})
    assert overridden["OMP_NUM_THREADS"] == "3"


def test_env_none_scrubs_lowercase_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("moonshot_api_key", _SECRET)
    monkeypatch.setenv("Anthropic_Api_Key", _SECRET)
    monkeypatch.setenv("db_password", _SECRET)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    result = run_owned(_py(_child_env()), _limits(), registry=ProcessRegistry())
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "moonshot_api_key" not in payload
    assert "Anthropic_Api_Key" not in payload
    assert "db_password" not in payload
    assert _SECRET not in result.stdout
    assert payload["OMP_NUM_THREADS"] == "1"
    assert payload["DIPCATCHER_CHILD"] == "1"


def test_windows_names_survive_platform_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    captured: dict[str, object] = {}

    class _Done:
        def __init__(self) -> None:
            self.pid = 4242
            self.returncode: int | None = 0
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int | None:
            return self.returncode

    def _popen(argv: list[str], **kwargs: object) -> _Done:
        captured["env"] = kwargs.get("env")
        captured["creationflags"] = kwargs.get("creationflags")
        return _Done()

    fake = {
        "PATH": r"C:\Windows",
        "SYSTEMROOT": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".EXE",
        "USERPROFILE": r"C:\Users\ada",
        "HOME": r"C:\Users\ada",
        "moonshot_api_key": _SECRET,
        "AWS_SECRET_ACCESS_KEY": _SECRET,
    }
    result = run_owned(
        ["tool"],
        _limits(),
        env=scrub_env(fake),
        popen=_popen,
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["COMSPEC"].endswith("cmd.exe")
    assert env["PATHEXT"] == ".EXE"
    assert env["USERPROFILE"] == r"C:\Users\ada"
    assert "moonshot_api_key" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert _SECRET not in env.values()
    assert captured["creationflags"] == getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


def test_popen_cancellable_scrubs_parent_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.harness import popen_cancellable

    monkeypatch.setenv("MOONSHOT_API_KEY", _SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _SECRET)
    monkeypatch.setenv("HF_TOKEN", _SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", _SECRET)
    monkeypatch.setenv("GITHUB_TOKEN", _SECRET)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", _SECRET)
    monkeypatch.setenv("db_password", _SECRET)
    code, out, err = popen_cancellable([sys.executable, "-c", _child_env()], timeout_s=15)
    assert code == 0, err
    payload = json.loads(out)
    for name in (
        "MOONSHOT_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "db_password",
    ):
        assert name not in payload
    assert _SECRET not in out
    assert _SECRET not in err
    assert payload["DIPCATCHER_CHILD"] == "1"
    assert payload["PYTHONUNBUFFERED"] == "1"
    assert "OMP_NUM_THREADS" in payload


# ---------------------------------------------------------------------------
# stdin, output bounds, reader threads
# ---------------------------------------------------------------------------


def test_input_sees_immediate_eof_and_devtty_fails() -> None:
    code = (
        "import sys\n"
        "try:\n"
        "    input()\n"
        "    print('GOT')\n"
        "except EOFError:\n"
        "    print('EOF')\n"
        "try:\n"
        "    open('/dev/tty')\n"
        "    print('TTY')\n"
        "except OSError:\n"
        "    print('NO-TTY')\n"
    )
    result = run_owned(_py(code), _limits(), registry=ProcessRegistry())
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == ["EOF", "NO-TTY"]
    assert _readers() == []


def test_parent_pty_bytes_do_not_reach_child(tmp_path: Path) -> None:
    script = tmp_path / "pty_parent.py"
    script.write_text(
        textwrap.dedent(
            """
            import os, pty, sys
            sys.path.insert(0, "/workspace/src")
            from fx1.interactive.procs import ProcessRegistry, ProcLimits, run_owned
            master, slave = pty.openpty()
            os.dup2(slave, 0)
            os.write(master, b"PTY-BYTES-NOT-FOR-CHILD\\n")
            code = (
                "import sys\\n"
                "print('STDIN=' + repr(sys.stdin.read()))\\n"
                "try:\\n"
                "    open('/dev/tty')\\n"
                "    print('TTY')\\n"
                "except OSError:\\n"
                "    print('NO-TTY')\\n"
            )
            result = run_owned(
                [sys.executable, "-c", code],
                ProcLimits(timeout_s=8),
                registry=ProcessRegistry(),
            )
            if "PTY-BYTES-NOT-FOR-CHILD" in result.stdout:
                sys.stderr.write(result.stdout)
                raise SystemExit(3)
            if "STDIN=''" not in result.stdout or "NO-TTY" not in result.stdout:
                sys.stderr.write(result.stdout + result.stderr)
                raise SystemExit(4)
            """
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/src"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_hundred_megabytes_stays_bounded() -> None:
    code = (
        "import sys\n"
        "chunk = b'x' * (1024 * 1024)\n"
        "for _ in range(100):\n"
        "    sys.stdout.buffer.write(chunk)\n"
    )
    cap = 200_000
    tracemalloc.start()
    result = run_owned(
        _py(code),
        _limits(timeout_s=18, max_output_bytes=cap),
        registry=ProcessRegistry(),
    )
    _traced, peak = tracemalloc.get_traced_memory()
    assert _traced >= 0
    tracemalloc.stop()
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False
    assert result.truncated is True
    assert result.stdout.startswith("x" * 64)
    assert len(result.stdout) <= cap + len(TRUNCATION_MARKER)
    assert peak < 20 * 1024 * 1024
    assert _readers() == []


def test_giant_line_without_newline_is_captured() -> None:
    code = "import sys; sys.stdout.buffer.write(b'Q' * 4000000)"
    result = run_owned(
        _py(code),
        _limits(timeout_s=10, max_output_bytes=5_000_000),
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0, result.stderr
    assert result.truncated is False
    assert len(result.stdout) == 4_000_000
    assert set(result.stdout) == {"Q"}


def test_binary_garbage_and_split_utf8() -> None:
    binary = "import sys; sys.stdout.buffer.write(b'ok\\x00\\xff\\xfe')"
    binary_result = run_owned(_py(binary), _limits(), registry=ProcessRegistry())
    assert binary_result.exit_code == 0
    assert "ok" in binary_result.stdout
    assert "\ufffd" in binary_result.stdout
    split = "import sys; sys.stdout.buffer.write(b'A'*65535 + b'\\xe2\\x82\\xac' + b'Z')"
    split_result = run_owned(
        _py(split),
        _limits(max_output_bytes=2_000_000),
        registry=ProcessRegistry(),
    )
    assert split_result.exit_code == 0, split_result.stderr
    assert split_result.truncated is False
    assert split_result.stdout.endswith("Z")
    assert "€" in split_result.stdout
    cut = "import sys; sys.stdout.buffer.write(b'\\xe2\\x82\\xac')"
    cut_result = run_owned(_py(cut), _limits(max_output_bytes=2), registry=ProcessRegistry())
    assert cut_result.exit_code == 0
    assert cut_result.truncated is True


def test_interleaved_streams_and_early_stdout_close() -> None:
    interleaved = (
        "import sys\n"
        "for i in range(200):\n"
        "    sys.stdout.write(f'o{i}\\n')\n"
        "    sys.stderr.write(f'e{i}\\n')\n"
    )
    both = run_owned(_py(interleaved), _limits(), registry=ProcessRegistry())
    assert both.exit_code == 0, both.stderr
    assert both.stdout.splitlines()[0] == "o0"
    assert both.stdout.splitlines()[-1] == "o199"
    assert both.stderr.splitlines()[0] == "e0"
    assert both.stderr.splitlines()[-1] == "e199"
    early = (
        "import sys, time\n"
        "sys.stdout.write('early\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stdout.close()\n"
        "time.sleep(0.2)\n"
        "sys.stderr.write('later\\n')\n"
    )
    closed = run_owned(_py(early), _limits(), registry=ProcessRegistry())
    assert closed.exit_code == 0, closed.stderr
    assert "early" in closed.stdout
    assert "later" in closed.stderr
    assert closed.duration_s < 5
    assert _readers() == []


def test_repeated_short_runs_do_not_leak_readers() -> None:
    registry = ProcessRegistry()
    for _ in range(4):
        result = run_owned(_py("print('x')"), _limits(), registry=registry)
        assert result.exit_code == 0
    assert registry.live() == []
    assert _readers() == []


# ---------------------------------------------------------------------------
# Tree kill, cancel, zombies, setsid limit
# ---------------------------------------------------------------------------


def test_cancel_during_startup_stops_within_a_second() -> None:
    registry = ProcessRegistry()
    cancel = threading.Event()

    def _fire() -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not registry.live():
            time.sleep(0.01)
        cancel.set()

    threading.Thread(target=_fire, daemon=True).start()
    started = time.monotonic()
    result = run_owned(
        _py("import time; time.sleep(30)"),
        _limits(timeout_s=8),
        cancel_event=cancel,
        registry=registry,
    )
    elapsed = time.monotonic() - started
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.exit_code == EXIT_CANCEL
    assert "[cancel]" in result.stderr
    assert "pending" not in result.stderr
    assert elapsed < 5
    assert result.pid is not None
    assert _wait_dead(result.pid)
    assert registry.live() == []
    assert _readers() == []


def test_cancel_during_heavy_output_is_bounded() -> None:
    registry = ProcessRegistry()
    cancel = threading.Event()

    def _fire() -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not registry.live():
            time.sleep(0.01)
        time.sleep(0.05)
        cancel.set()

    code = "import sys\nwhile True:\n    sys.stdout.buffer.write(b'y' * 65536)\n"
    threading.Thread(target=_fire, daemon=True).start()
    result = run_owned(
        _py(code),
        _limits(timeout_s=6, max_output_bytes=100_000),
        cancel_event=cancel,
        registry=registry,
    )
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.exit_code == EXIT_CANCEL
    assert len(result.stdout) <= 100_000 + len(TRUNCATION_MARKER)
    assert result.pid is not None
    assert _wait_dead(result.pid)
    assert registry.live() == []
    assert _readers() == []


def test_sigterm_ignored_child_dies_on_sigkill() -> None:
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "time.sleep(60)\n"
    )
    result = run_owned(
        _py(code),
        _limits(timeout_s=0.4, kill_grace_s=0.2),
        registry=ProcessRegistry(),
    )
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.exit_code == EXIT_TIMEOUT
    assert "terminated" in result.stderr
    assert "pending" not in result.stderr
    assert result.duration_s < 5
    assert result.pid is not None
    assert _wait_dead(result.pid)


def test_subprocess_grandchild_is_killed(tmp_path: Path) -> None:
    gpath = tmp_path / "grand.pid"
    code = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "inner = ('import os,sys,time\\nfrom pathlib import Path\\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\\ntime.sleep(60)\\n')\n"
        "subprocess.Popen([sys.executable, '-c', inner, sys.argv[1]])\n"
        "path = Path(sys.argv[1])\n"
        "for _ in range(200):\n"
        "    if path.exists():\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "time.sleep(60)\n"
    )
    result = None
    try:
        result = run_owned(
            _py(code, str(gpath)),
            _limits(timeout_s=3, kill_grace_s=0.3),
            registry=ProcessRegistry(),
        )
        assert result.timed_out is True
        assert gpath.exists(), result.stderr
        grand = int(gpath.read_text())
        assert result.pid is not None
        assert _wait_dead(result.pid)
        assert _wait_dead(grand)
    finally:
        if gpath.exists():
            _kill(int(gpath.read_text() or "0"))
        if result is not None and result.pid is not None:
            _kill(result.pid)


@pytest.mark.skipif(not _HAS_FORK, reason="POSIX fork")
def test_fork_grandchild_in_group_is_killed(tmp_path: Path) -> None:
    gpath = tmp_path / "fork.pid"
    code = (
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "path = Path(sys.argv[1])\n"
        "for _ in range(200):\n"
        "    if path.exists():\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "time.sleep(60)\n"
    )
    result = None
    try:
        result = run_owned(
            _py(code, str(gpath)),
            _limits(timeout_s=3, kill_grace_s=0.25),
            registry=ProcessRegistry(),
        )
        assert result.timed_out is True
        grand = int(gpath.read_text())
        assert result.pid is not None
        assert _wait_dead(result.pid)
        assert _wait_dead(grand)
    finally:
        if gpath.exists():
            _kill(int(gpath.read_text() or "0"))


@pytest.mark.skipif(not _HAS_FORK, reason="POSIX fork")
def test_multiprocessing_child_is_killed(tmp_path: Path) -> None:
    gpath = tmp_path / "mp.pid"
    code = (
        "import multiprocessing, os, sys, time\n"
        "from pathlib import Path\n"
        "def work(path):\n"
        "    Path(path).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "if __name__ == '__main__':\n"
        "    multiprocessing.set_start_method('fork')\n"
        "    proc = multiprocessing.Process(target=work, args=(sys.argv[1],))\n"
        "    proc.start()\n"
        "    path = Path(sys.argv[1])\n"
        "    for _ in range(200):\n"
        "        if path.exists():\n"
        "            break\n"
        "        time.sleep(0.01)\n"
        "    time.sleep(60)\n"
    )
    result = None
    try:
        result = run_owned(
            _py(code, str(gpath)),
            _limits(timeout_s=4, kill_grace_s=0.4),
            registry=ProcessRegistry(),
        )
        assert result.timed_out is True, result.stderr
        assert gpath.exists(), result.stderr
        grand = int(gpath.read_text())
        assert result.pid is not None
        assert _wait_dead(result.pid)
        assert _wait_dead(grand)
    finally:
        if gpath.exists():
            _kill(int(gpath.read_text() or "0"))


@pytest.mark.skipif(not _HAS_FORK, reason="POSIX fork")
def test_setsid_grandchild_of_living_parent_is_killed(tmp_path: Path) -> None:
    gpath = tmp_path / "setsid.pid"
    code = (
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "    Path(sys.argv[1]).write_text(f'{os.getpid()} {os.getsid(0)}')\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "path = Path(sys.argv[1])\n"
        "for _ in range(200):\n"
        "    if path.exists():\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "time.sleep(60)\n"
    )
    result = None
    grand = 0
    try:
        result = run_owned(
            _py(code, str(gpath)),
            _limits(timeout_s=3, kill_grace_s=0.25),
            registry=ProcessRegistry(),
        )
        assert result.timed_out is True, result.stderr
        assert gpath.exists(), result.stderr
        grand_text, sid_text = gpath.read_text().split()
        grand = int(grand_text)
        assert int(sid_text) != result.pgid
        assert result.pid is not None
        assert _wait_dead(result.pid)
        assert _wait_dead(grand)
        assert str(grand) not in result.stderr
        assert _readers() == []
    finally:
        _kill(grand)
        if result is not None and result.pid is not None:
            _kill(result.pid)


@pytest.mark.skipif(not _HAS_FORK, reason="POSIX fork")
def test_double_fork_setsid_is_not_claimed_killed(tmp_path: Path) -> None:
    """A reparented setsid descendant is outside the tracked tree.

    The result must not name that pid as killed. Reader threads still exit.
    """
    gpath = tmp_path / "daemon.pid"
    code = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    path.write_text(f'{os.getpid()} {os.getsid(0)}')\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "for _ in range(200):\n"
        "    if path.exists():\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "os._exit(0)\n"
    )
    registry = ProcessRegistry()
    result = None
    grand = 0
    try:
        result = run_owned(
            _py(code, str(gpath)),
            _limits(timeout_s=5, kill_grace_s=0.2),
            registry=registry,
        )
        assert result.exit_code == 0, result.stderr
        assert result.timed_out is False
        assert result.cancelled is False
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not gpath.exists():
            time.sleep(0.02)
        assert gpath.exists()
        grand_text, sid_text = gpath.read_text().split()
        grand = int(grand_text)
        assert int(sid_text) != result.pgid
        assert _alive(grand)
        assert str(grand) not in result.stderr
        assert "killed_tree" not in result.stderr
        assert registry.live() == []
        assert _readers() == []
    finally:
        _kill(grand)


def test_zombie_grandchild_is_reaped(tmp_path: Path) -> None:
    zpath = tmp_path / "zombie.pid"
    code = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "Path(sys.argv[1]).write_text(str(pid))\n"
        "time.sleep(60)\n"
    )
    if not hasattr(os, "fork"):
        pytest.skip("POSIX fork")
    result = run_owned(
        _py(code, str(zpath)),
        _limits(timeout_s=0.6, kill_grace_s=0.2),
        registry=ProcessRegistry(),
    )
    assert result.timed_out is True
    zombie = int(zpath.read_text())
    assert result.pid is not None
    assert _wait_dead(result.pid)
    assert _wait_dead(zombie)


def test_failed_kill_reports_pending_and_stays_registered() -> None:
    registry = ProcessRegistry()
    cancel = threading.Event()

    def _noop(_pid: int) -> None:
        return None

    def _fire() -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not registry.live():
            time.sleep(0.01)
        cancel.set()

    threading.Thread(target=_fire, daemon=True).start()
    result = None
    try:
        result = run_owned(
            _py("import time; time.sleep(60)"),
            _limits(timeout_s=4, kill_grace_s=0.15),
            cancel_event=cancel,
            killer=_noop,
            registry=registry,
        )
        assert result.cancelled is True
        assert result.timed_out is False
        assert result.exit_code == EXIT_CANCEL
        assert "pending" in result.stderr
        assert "still alive" in result.stderr
        assert "terminated" not in result.stderr
        assert result.pid is not None
        assert _alive(result.pid)
        live = registry.live()
        assert [item.pid for item in live] == [result.pid]
    finally:
        registry.terminate_all(grace_s=0.3)
        if result is not None and result.pid is not None:
            _kill(result.pid)


def test_atexit_kills_a_live_child(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "atexit_parent.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, threading, time
            from pathlib import Path
            sys.path.insert(0, "/workspace/src")
            from fx1.interactive.procs import ProcessRegistry, ProcLimits, run_owned
            pidfile = Path({str(pidfile)!r})
            registry = ProcessRegistry()
            registry.install_atexit()

            def work() -> None:
                run_owned(
                    [sys.executable, "-c",
                     "import os,sys,time; from pathlib import Path; "
                     "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)",
                     str(pidfile)],
                    ProcLimits(timeout_s=20, kill_grace_s=0.3, cancel_poll_s=0.05),
                    registry=registry,
                )

            threading.Thread(target=work, daemon=True).start()
            for _ in range(100):
                if pidfile.exists():
                    break
                time.sleep(0.05)
            else:
                raise SystemExit(2)
            """
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/src"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    try:
        assert _wait_dead(pid, budget=2.0)
    finally:
        _kill(pid)


def test_harness_result_timeout_and_cancel_are_coherent() -> None:
    class _Bridge:
        supports_extended = True

        def __call__(
            self,
            argv: list[str],
            timeout_s: int,
            *,
            cancel_event: threading.Event | None = None,
            env_extra: object = None,
            step: str | None = None,
        ) -> ProcResult:
            return run_owned(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                ProcLimits(timeout_s=float(timeout_s), kill_grace_s=0.2, cancel_poll_s=0.02),
                cancel_event=cancel_event,
                registry=ProcessRegistry(),
            )

    timed = Harness(runner=_Bridge()).run("doctor", timeout_s=1)
    assert timed.exit_code == EXIT_TIMEOUT
    assert timed.timed_out is True
    assert timed.cancelled is False
    assert timed.ok is False
    assert "[timeout]" in timed.stderr
    assert timed.duration_s < 5
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    stopped = Harness(runner=_Bridge()).run("monitor", timeout_s=5, cancel_event=cancel)
    assert stopped.exit_code == EXIT_CANCEL
    assert stopped.cancelled is True
    assert stopped.timed_out is False
    assert stopped.duration_s < 5


# ---------------------------------------------------------------------------
# Harness containment, ceiling, launcher
# ---------------------------------------------------------------------------


def test_timeout_cannot_exceed_registry_ceiling() -> None:
    seen: dict[str, int] = {}

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen["timeout"] = timeout_s
        return 0, "", ""

    harness = Harness(runner=_runner)
    command = harness.get("doctor")
    ceiling = command.timeout_s
    try:
        harness.run("doctor", timeout_s=ceiling + 9_999)
        assert seen["timeout"] == ceiling
        harness.run("doctor", timeout_s=5)
        assert seen["timeout"] == 5
        command.timeout_s = ceiling + 4_000
        harness.run("doctor")
        assert seen["timeout"] == ceiling
        with pytest.raises(ValueError, match="positive int"):
            harness.run("doctor", timeout_s=True)  # type: ignore[arg-type]
    finally:
        command.timeout_s = ceiling


def test_unknown_command_does_not_spawn() -> None:
    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        raise AssertionError("spawned")

    with pytest.raises(KeyError, match="not a registered harness command"):
        Harness(runner=_runner).run("live-trading")


def test_legacy_runner_still_receives_two_arguments() -> None:
    seen: dict[str, object] = {}

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen["argv"] = list(argv)
        seen["timeout"] = timeout_s
        return 0, "legacy", ""

    result = Harness(runner=_runner).run(
        "doctor",
        ["--help"],
        timeout_s=12,
        cancel_event=threading.Event(),
        env_extra={"QUANT_MODE": "lab"},
        step="s",
    )
    assert result.ok
    assert result.stdout == "legacy"
    assert result.timed_out is False
    assert result.cancelled is False
    assert seen["argv"] == ["doctor", "--help"]
    assert seen["timeout"] == 12


def test_config_symlink_dotdot_repeat_and_trailing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    inside = configs / "ok.yaml"
    inside.write_text("a: 1\n", encoding="utf-8")
    outside = tmp_path / "outside.yaml"
    outside.write_text("nope: 1\n", encoding="utf-8")
    link = configs / "evil.yaml"
    link.symlink_to(outside)
    monkeypatch.chdir(tmp_path)
    store: dict[str, list[str]] = {}
    harness = Harness(runner=_runner_capture(store))
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", config=link)
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config", "../outside.yaml"])
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config=../outside.yaml"])
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config", str(inside), "--config", str(outside)])
    assert "argv" not in store
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--help", "--config"])
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config="])
    assert "argv" not in store
    result = harness.run("doctor", extra_args=["--config=configs/ok.yaml", "--seed", "1"])
    assert result.ok
    assert store["argv"] == ["doctor", "--config=configs/ok.yaml", "--seed", "1"]


def test_configs_directory_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.yaml").write_text("a: 1\n", encoding="utf-8")
    linked = tmp_path / "configs"
    linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    harness = Harness(runner=lambda argv, timeout_s: (0, "", ""))
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", config=linked / "evil.yaml")


def test_nested_cwd_uses_ancestor_repo_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    configs = repo / "configs"
    configs.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    ok = configs / "ok.yaml"
    ok.write_text("a: 1\n", encoding="utf-8")
    nested = repo / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    store: dict[str, list[str]] = {}
    harness = Harness(runner=_runner_capture(store))
    result = harness.run("doctor", config=ok)
    assert result.ok
    assert str(ok.resolve()) in store["argv"]
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", config=tmp_path / "outside.yaml")


def test_nested_configs_cannot_shadow_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    ancestor = repo / "configs" / "ok.yaml"
    ancestor.write_text("a: 1\n", encoding="utf-8")
    nested = repo / "nested"
    local = nested / "configs"
    local.mkdir(parents=True)
    local_file = local / "local.yaml"
    local_file.write_text("a: 1\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    store: dict[str, list[str]] = {}
    harness = Harness(runner=_runner_capture(store))
    result = harness.run("doctor", config=ancestor)
    assert result.ok
    assert str(ancestor.resolve()) in store["argv"]
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", config=local_file)


def test_out_dir_and_data_are_not_contained() -> None:
    """Path-like flags other than --config are forwarded unchanged."""
    store: dict[str, list[str]] = {}
    harness = Harness(runner=_runner_capture(store))
    result = harness.run(
        "doctor",
        extra_args=["--out-dir", "/tmp/outside-out", "--data", "/tmp/outside-data"],
    )
    assert result.ok
    assert store["argv"] == [
        "doctor",
        "--out-dir",
        "/tmp/outside-out",
        "--data",
        "/tmp/outside-data",
    ]


def test_doctor_help_starts_quickly_and_does_not_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", _SECRET)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    result = Harness().run("doctor", extra_args=["--help"])
    elapsed = time.monotonic() - started
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False
    assert result.cancelled is False
    assert "doctor" in result.stdout.lower()
    assert _SECRET not in result.stdout
    assert _SECRET not in result.stderr
    assert "Start a conversation" not in result.stdout
    assert elapsed < 25, elapsed
    assert REGISTRY.live() == []


def test_bare_launcher_does_not_recurse_or_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", _SECRET)
    monkeypatch.setenv("TERM", "xterm-256color")
    started = time.monotonic()
    result = run_owned(
        [sys.executable, "-m", "dipcatcher_cli"],
        ProcLimits(timeout_s=25, kill_grace_s=0.4, cancel_poll_s=0.05),
        registry=ProcessRegistry(),
    )
    elapsed = time.monotonic() - started
    assert result.timed_out is False, result.stderr
    assert result.exit_code == 0, result.stderr
    assert result.stdout.count("Start a conversation") == 1
    assert "interactive terminal" in result.stdout
    assert _SECRET not in result.stdout
    assert _SECRET not in result.stderr
    assert elapsed < 25, elapsed
    assert _readers() == []
