"""Owned subprocess runner: real children, scrubbing, cancel, and Windows branches."""

from __future__ import annotations

import atexit
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fx1.interactive import procs
from fx1.interactive.procs import (
    DEFAULT_ENV_ALLOW,
    ENV_DENY_EXACT,
    ENV_DENY_PREFIXES,
    ENV_DENY_SUFFIXES,
    EXIT_CANCEL,
    EXIT_SPAWN,
    EXIT_TIMEOUT,
    REGISTRY,
    TRUNCATION_MARKER,
    LiveProc,
    ProcessRegistry,
    ProcLimits,
    run_owned,
    scrub_env,
)
from fx1.interactive.redact import REDACTED

FAKE_PID = 2_000_000_000
_SECRET = "sk-test-secret-value"


def _py(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


def _limits(**overrides: float | int) -> ProcLimits:
    values: dict[str, float | int] = {"timeout_s": 8.0}
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
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    parts = text.rpartition(")")[2].split()
    return not (parts and parts[0] == "Z")


def _wait_dead(pid: int) -> bool:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


def _sigkill(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


@pytest.fixture(autouse=True)
def _registry_quiet() -> None:
    assert REGISTRY.live() == []
    yield
    if REGISTRY.live():
        REGISTRY.terminate_all(grace_s=0.2)
    assert REGISTRY.live() == []


class _Immediate:
    def __init__(self, pid: int = FAKE_PID, stdout: bytes = b"hi\n", stderr: bytes = b"") -> None:
        self.pid = pid
        self.returncode: int | None = 0
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


class _Hang:
    def __init__(self, pid: int = FAKE_PID) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")
        self._lock = threading.Lock()

    def poll(self) -> int | None:
        with self._lock:
            return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        with self._lock:
            if self.returncode is not None:
                return self.returncode
        if timeout and timeout > 0:
            time.sleep(min(float(timeout), 0.02))
        with self._lock:
            if self.returncode is not None:
                return self.returncode
        raise subprocess.TimeoutExpired("tool", timeout or 0)

    def mark(self, code: int) -> None:
        with self._lock:
            self.returncode = code


def test_limits_defaults_and_frozen() -> None:
    limits = ProcLimits(timeout_s=3)
    assert limits.kill_grace_s == 2.0
    assert limits.max_output_bytes == 2 * 1024 * 1024
    assert limits.cancel_poll_s == 0.05
    with pytest.raises(AttributeError):
        limits.timeout_s = 1  # type: ignore[misc]


def test_allow_and_deny_lists_are_data() -> None:
    assert isinstance(DEFAULT_ENV_ALLOW, tuple)
    for name in (
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "FX1_CONFIG_DIR",
        "PYTHON*",
        "LC_*",
        "QUANT_*",
        "DIPCATCHER_*",
        "MLFLOW_*",
        "HTTP_PROXY",
        "NO_COLOR",
        "TOKENIZERS_PARALLELISM",
    ):
        assert name in DEFAULT_ENV_ALLOW
    assert "FX1_BASE_URL" in ENV_DENY_EXACT
    assert "GITHUB_TOKEN" in ENV_DENY_EXACT
    for prefix in ("MOONSHOT_", "AWS_", "GATE_SIGNING_KEY", "WITNESS_SIGNING_KEY"):
        assert prefix in ENV_DENY_PREFIXES
    assert "_API_KEY" in ENV_DENY_SUFFIXES
    assert "_TOKEN" in ENV_DENY_SUFFIXES
    assert "PROOFCORE_SIGNING_KEY" in ENV_DENY_PREFIXES


@pytest.mark.parametrize(
    ("base", "extra", "present", "absent"),
    [
        pytest.param({"PATH": "/bin"}, None, {"PATH": "/bin"}, ["HOME"], id="path"),
        pytest.param({"HOME": "/h"}, None, {"HOME": "/h"}, ["PATH"], id="home"),
        pytest.param(
            {"USERPROFILE": r"C:\u", "TEMP": "/t", "TMP": "/t2", "TMPDIR": "/t3"},
            None,
            {"USERPROFILE": r"C:\u", "TEMP": "/t", "TMP": "/t2", "TMPDIR": "/t3"},
            [],
            id="temp-dirs",
        ),
        pytest.param(
            {"LANG": "C.UTF-8", "LC_ALL": "C", "LANGUAGE": "en"},
            None,
            {"LANG": "C.UTF-8", "LC_ALL": "C", "LANGUAGE": "en"},
            ["LC"],
            id="locale",
        ),
        pytest.param(
            {"TERM": "xterm", "TZ": "UTC", "VIRTUAL_ENV": "/venv"},
            None,
            {"TERM": "xterm", "TZ": "UTC", "VIRTUAL_ENV": "/venv"},
            [],
            id="term-tz-venv",
        ),
        pytest.param(
            {"FX1_CONFIG_DIR": "/cfg"},
            None,
            {"FX1_CONFIG_DIR": "/cfg"},
            ["FX1_BASE_URL"],
            id="fx1-config",
        ),
        pytest.param(
            {"PYTHONPATH": "/src", "PYTHONUNBUFFERED": "0", "PYTHONHASHSEED": "1"},
            None,
            {"PYTHONPATH": "/src", "PYTHONHASHSEED": "1"},
            [],
            id="python-prefix",
        ),
        pytest.param({"QUANT_MODE": "lab"}, None, {"QUANT_MODE": "lab"}, [], id="quant"),
        pytest.param(
            {"DIPCATCHER_SESSION_ID": "sess"},
            None,
            {"DIPCATCHER_SESSION_ID": "sess"},
            [],
            id="dipcatcher-from-base",
        ),
        pytest.param(
            {"MLFLOW_TRACKING_URI": "file:/ml"},
            None,
            {"MLFLOW_TRACKING_URI": "file:/ml"},
            [],
            id="mlflow",
        ),
        pytest.param(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            None,
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            [],
            id="thread-knobs",
        ),
        pytest.param({"NO_COLOR": "1"}, None, {"NO_COLOR": "1"}, [], id="no-color"),
        pytest.param(
            {"http_proxy": "http://p", "HTTPS_PROXY": "http://s", "NO_PROXY": "localhost"},
            None,
            {"http_proxy": "http://p", "HTTPS_PROXY": "http://s", "NO_PROXY": "localhost"},
            [],
            id="proxies",
        ),
        pytest.param(
            {"SYSTEMROOT": r"C:\Windows", "COMSPEC": "cmd", "PATHEXT": ".EXE", "WINDIR": r"C:\W"},
            None,
            {"SYSTEMROOT": r"C:\Windows", "COMSPEC": "cmd", "PATHEXT": ".EXE", "WINDIR": r"C:\W"},
            [],
            id="windows-basics",
        ),
        pytest.param(
            {"PATH": "/bin", "MOONSHOT_API_KEY": _SECRET},
            None,
            {"PATH": "/bin"},
            ["MOONSHOT_API_KEY"],
            id="moonshot-api-key",
        ),
        pytest.param(
            {"MOONSHOT_BASE_URL": "https://example.invalid"},
            None,
            {},
            ["MOONSHOT_BASE_URL"],
            id="moonshot-base",
        ),
        pytest.param(
            {
                "AWS_ACCESS_KEY_ID": "AKIA",
                "AWS_SECRET_ACCESS_KEY": "sek",
                "AWS_REGION": "us-east-1",
            },
            None,
            {},
            ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            id="aws",
        ),
        pytest.param({"GITHUB_TOKEN": "ghp"}, None, {}, ["GITHUB_TOKEN"], id="github-token"),
        pytest.param(
            {"FX1_BASE_URL": "https://example.invalid/v1"},
            None,
            {},
            ["FX1_BASE_URL"],
            id="fx1-base-url",
        ),
        pytest.param(
            {
                "GATE_SIGNING_KEY": "a",
                "WITNESS_SIGNING_KEY": "b",
                "PROOFCORE_SIGNING_KEY": "c",
            },
            None,
            {},
            ["GATE_SIGNING_KEY", "WITNESS_SIGNING_KEY", "PROOFCORE_SIGNING_KEY"],
            id="signing-keys",
        ),
        pytest.param(
            {"OPENAI_API_KEY": "sk", "HF_TOKEN": "hf"},
            None,
            {},
            ["OPENAI_API_KEY", "HF_TOKEN"],
            id="provider-keys",
        ),
        pytest.param(
            {"DB_PASSWORD": "hunter2", "password": "nope"},
            None,
            {},
            ["DB_PASSWORD", "password"],
            id="password",
        ),
        pytest.param({"QUANT_API_KEY": "q"}, None, {}, ["QUANT_API_KEY"], id="quant-api-key"),
        pytest.param({"PYTHON_API_KEY": "p"}, None, {}, ["PYTHON_API_KEY"], id="python-api-key"),
        pytest.param(
            {"MLFLOW_TRACKING_TOKEN": "t"},
            None,
            {},
            ["MLFLOW_TRACKING_TOKEN"],
            id="mlflow-token",
        ),
        pytest.param(
            {"LD_PRELOAD": "/evil.so", "UNRELATED": "x"},
            None,
            {},
            ["LD_PRELOAD", "UNRELATED"],
            id="not-allow-listed",
        ),
        pytest.param({"Path": "/bin"}, None, {"Path": "/bin"}, [], id="casefold-path"),
        pytest.param(
            {},
            {
                "DIPCATCHER_SESSION_ID": "s1",
                "DIPCATCHER_RUN_ID": "r1",
                "DIPCATCHER_STEP": "r1.000",
            },
            {
                "DIPCATCHER_SESSION_ID": "s1",
                "DIPCATCHER_RUN_ID": "r1",
                "DIPCATCHER_STEP": "r1.000",
            },
            [],
            id="correlation-extra",
        ),
        pytest.param({}, {"CUSTOM_KNOB": "1"}, {"CUSTOM_KNOB": "1"}, [], id="extra-knob"),
        pytest.param({}, {"GH_TOKEN": "x"}, {}, ["GH_TOKEN"], id="extra-token-refused"),
        pytest.param({}, None, {}, ["PATH"], id="empty-base"),
        pytest.param(
            {"USER": "ada", "USERNAME": "ada", "LOGNAME": "ada"},
            None,
            {"USER": "ada", "USERNAME": "ada", "LOGNAME": "ada"},
            [],
            id="user-names",
        ),
    ],
)
def test_scrub_env_table(
    base: dict[str, str],
    extra: dict[str, str] | None,
    present: dict[str, str],
    absent: list[str],
) -> None:
    env = scrub_env(base, extra=extra)
    for key, value in present.items():
        assert env[key] == value
    for key in absent:
        assert key not in env
    assert env["DIPCATCHER_CHILD"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_scrub_env_allow_secrets_only_applies_to_extra() -> None:
    both = scrub_env(
        {"MOONSHOT_API_KEY": "from-base"},
        extra={"MOONSHOT_API_KEY": "from-extra"},
        allow_secrets=["MOONSHOT_API_KEY"],
    )
    assert both["MOONSHOT_API_KEY"] == "from-extra"
    base_only = scrub_env({"MOONSHOT_API_KEY": "from-base"}, allow_secrets=["MOONSHOT_API_KEY"])
    assert "MOONSHOT_API_KEY" not in base_only


def test_scrub_env_custom_allow_replaces_exact_patterns() -> None:
    env = scrub_env({"PATH": "/bin", "HOME": "/h"}, allow=("PATH",))
    assert env["PATH"] == "/bin"
    assert "HOME" not in env
    assert env["DIPCATCHER_CHILD"] == "1"


def test_scrub_env_forces_child_markers_last() -> None:
    env = scrub_env(
        {"DIPCATCHER_CHILD": "0", "PYTHONUNBUFFERED": "0"},
        extra={"DIPCATCHER_CHILD": "0"},
    )
    assert env["DIPCATCHER_CHILD"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_scrub_env_does_not_mutate_input_and_is_idempotent() -> None:
    base = {"PATH": "/bin", "MOONSHOT_API_KEY": _SECRET, "HOME": "/h"}
    snapshot = dict(base)
    once = scrub_env(base)
    assert base == snapshot
    assert scrub_env(once) == once
    assert _SECRET not in once.values()


def test_success_captures_both_streams() -> None:
    code = "import sys; print('hello-out'); print('hello-err', file=sys.stderr)"
    result = run_owned(_py(code), _limits(), registry=ProcessRegistry())
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello-out"
    assert result.stderr.strip() == "hello-err"
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.truncated is False
    assert result.spawn_error is None
    assert result.pid is not None
    assert result.duration_s >= 0


def test_nonzero_exit() -> None:
    result = run_owned(_py("import sys; sys.exit(3)"), _limits(), registry=ProcessRegistry())
    assert result.exit_code == 3
    assert result.timed_out is False
    assert result.spawn_error is None


def test_exact_output_limit_is_not_truncated() -> None:
    result = run_owned(
        _py("import sys; sys.stdout.write('C' * 64)"),
        _limits(max_output_bytes=64),
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0
    assert result.truncated is False
    assert result.stdout == "C" * 64
    assert TRUNCATION_MARKER not in result.stdout


def test_truncation_marker_and_process_still_exits() -> None:
    code = "import sys; sys.stdout.write('A' * 5000); sys.stderr.write('B' * 5000)"
    result = run_owned(_py(code), _limits(max_output_bytes=128), registry=ProcessRegistry())
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.truncated is True
    assert result.stdout.startswith("A" * 128)
    assert result.stderr.startswith("B" * 128)
    assert TRUNCATION_MARKER in result.stdout
    assert TRUNCATION_MARKER in result.stderr
    assert "A" * 129 not in result.stdout


def test_invalid_utf8_is_replaced() -> None:
    code = "import sys; sys.stdout.buffer.write(b'ok\\xff\\xfe')"
    result = run_owned(_py(code), _limits(), registry=ProcessRegistry())
    assert result.exit_code == 0
    assert "ok" in result.stdout
    assert "\ufffd" in result.stdout


def test_large_output_both_streams_does_not_deadlock() -> None:
    code = (
        "import sys\n"
        "chunk = 1024\n"
        "for _ in range(1200):\n"
        "    sys.stdout.buffer.write(b'x' * chunk)\n"
        "    sys.stderr.buffer.write(b'y' * chunk)\n"
    )
    result = run_owned(
        _py(code),
        _limits(timeout_s=8, max_output_bytes=3 * 1024 * 1024),
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.truncated is False
    assert len(result.stdout) >= 1_000_000
    assert len(result.stderr) >= 1_000_000
    assert set(result.stdout) == {"x"}
    assert set(result.stderr) == {"y"}


def test_spawn_failure_returns_127_and_clears_registry() -> None:
    registry = ProcessRegistry()
    events: list[str] = []
    result = run_owned(
        ["/no/such/fx1-wp-c-binary"],
        _limits(),
        registry=registry,
        emit=lambda kind, _fields: events.append(kind),
    )
    assert result.exit_code == EXIT_SPAWN
    assert result.pid is None
    assert result.pgid is None
    assert result.spawn_error
    assert result.timed_out is False
    assert registry.live() == []
    assert events == ["proc.exit"]


def test_spawn_error_text_is_redacted() -> None:
    def _boom(argv: list[str], **kwargs: object) -> None:
        raise OSError("failed api_key=abcdefghijklmnop nearby")

    events: list[dict[str, object]] = []
    result = run_owned(
        ["tool", "--token", "othersecretvalue"],
        _limits(),
        popen=_boom,
        registry=ProcessRegistry(),
        emit=lambda _kind, fields: events.append(dict(fields)),
    )
    assert result.exit_code == EXIT_SPAWN
    assert result.spawn_error is not None
    assert "abcdefghijklmnop" not in result.spawn_error
    assert "othersecretvalue" not in " ".join(result.argv_summary)
    blob = json.dumps(events)
    assert "abcdefghijklmnop" not in blob
    assert "othersecretvalue" not in blob


def test_cwd_is_honored(tmp_path: Path) -> None:
    result = run_owned(
        _py("import os; print(os.getcwd())"),
        _limits(),
        cwd=tmp_path,
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0
    assert Path(result.stdout.strip()) == tmp_path.resolve()


def test_session_leader_pgid_matches_pid() -> None:
    code = "import os; print(os.getpid()); print(os.getpgid(0))"
    result = run_owned(_py(code), _limits(), registry=ProcessRegistry())
    lines = result.stdout.strip().splitlines()
    assert result.exit_code == 0
    assert lines[0] == lines[1]
    assert result.pid == int(lines[0])
    assert result.pgid == int(lines[1])


def test_provided_env_is_used_verbatim() -> None:
    code = (
        "import json, os, sys; "
        "json.dump({'custom': os.environ.get('CUSTOM_KNOB'), "
        "'child': os.environ.get('DIPCATCHER_CHILD')}, sys.stdout)"
    )
    env = {"CUSTOM_KNOB": "yes", "PATH": os.environ.get("PATH", "")}
    result = run_owned(_py(code), _limits(), env=env, registry=ProcessRegistry())
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["custom"] == "yes"
    assert payload["child"] is None


def test_env_none_scrubs_process_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", _SECRET)
    monkeypatch.setenv("FX1_BASE_URL", "https://example.invalid/secret")
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    code = "import json, os, sys; json.dump(dict(os.environ), sys.stdout)"
    result = run_owned(_py(code), _limits(), registry=ProcessRegistry())
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "MOONSHOT_API_KEY" not in payload
    assert "FX1_BASE_URL" not in payload
    assert payload["FX1_CONFIG_DIR"] == str(tmp_path)
    assert payload["DIPCATCHER_CHILD"] == "1"
    assert payload["PYTHONUNBUFFERED"] == "1"
    assert _SECRET not in result.stdout
    assert _SECRET not in result.stderr


def test_scrubbed_child_environment_keeps_allow_list(tmp_path: Path) -> None:
    base = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        "LC_ALL": "C",
        "FX1_CONFIG_DIR": str(tmp_path),
        "OMP_NUM_THREADS": "1",
        "NO_COLOR": "1",
        "HTTP_PROXY": "http://proxy.invalid:8080",
        "QUANT_MODE": "lab",
        "MOONSHOT_API_KEY": _SECRET,
        "AWS_ACCESS_KEY_ID": "AKIATEST",
        "GITHUB_TOKEN": "ghp_test",
        "FX1_BASE_URL": "https://example.invalid",
        "LD_PRELOAD": "/tmp/evil.so",
        "UNRELATED_FLAG": "nope",
    }
    extra = {"DIPCATCHER_SESSION_ID": "sess-1", "CUSTOM_KNOB": "kept"}
    env = scrub_env(base, extra=extra)
    code = "import json, os, sys; json.dump(dict(os.environ), sys.stdout)"
    result = run_owned(_py(code), _limits(), env=env, registry=ProcessRegistry())
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["DIPCATCHER_CHILD"] == "1"
    assert payload["HOME"] == str(tmp_path)
    assert payload["FX1_CONFIG_DIR"] == str(tmp_path)
    assert payload["QUANT_MODE"] == "lab"
    assert payload["DIPCATCHER_SESSION_ID"] == "sess-1"
    assert payload["CUSTOM_KNOB"] == "kept"
    assert payload["HTTP_PROXY"] == "http://proxy.invalid:8080"
    assert payload["OMP_NUM_THREADS"] == "1"
    for banned in (
        "MOONSHOT_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "GITHUB_TOKEN",
        "FX1_BASE_URL",
        "LD_PRELOAD",
        "UNRELATED_FLAG",
    ):
        assert banned not in payload
    assert _SECRET not in result.stdout


def test_pre_cancel_does_not_spawn() -> None:
    cancel = threading.Event()
    cancel.set()
    calls: list[int] = []

    def _popen(argv: list[str], **kwargs: object) -> None:
        calls.append(1)
        raise AssertionError("spawned")

    result = run_owned(
        ["tool"],
        _limits(),
        cancel_event=cancel,
        popen=_popen,
        registry=ProcessRegistry(),
    )
    assert calls == []
    assert result.cancelled is True
    assert result.exit_code == EXIT_CANCEL
    assert result.pid is None
    assert "[cancel]" in result.stderr
    assert result.timed_out is False


def test_cancel_acks_within_half_second() -> None:
    registry = ProcessRegistry()
    cancel = threading.Event()
    marked: dict[str, float] = {}

    def _fire() -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not registry.live():
            time.sleep(0.01)
        marked["t"] = time.monotonic()
        cancel.set()

    threading.Thread(target=_fire, daemon=True).start()
    result = run_owned(
        _py("import time; time.sleep(30)"),
        ProcLimits(timeout_s=5, kill_grace_s=2.0, cancel_poll_s=0.05),
        cancel_event=cancel,
        registry=registry,
    )
    acked = time.monotonic() - marked["t"]
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.exit_code == EXIT_CANCEL
    assert "[cancel]" in result.stderr
    assert acked <= 0.5
    assert result.pid is not None
    assert _wait_dead(result.pid)
    assert registry.live() == []


def test_timeout_does_not_wait_full_grace_when_child_dies(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[int] = []
    real_killpg = os.killpg

    def _spy(pgid: int, sig: int) -> None:
        signals.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", _spy)
    events: list[tuple[str, object]] = []
    result = run_owned(
        _py("import time; time.sleep(30)"),
        ProcLimits(timeout_s=0.25, kill_grace_s=2.0, cancel_poll_s=0.02),
        registry=ProcessRegistry(),
        emit=lambda kind, fields: events.append((kind, fields.get("reason"))),
    )
    assert result.timed_out is True
    assert result.cancelled is False
    assert result.exit_code == EXIT_TIMEOUT
    assert "[timeout]" in result.stderr
    assert "exceeded 0.25s" in result.stderr
    assert signal.SIGTERM in signals
    assert signal.SIGKILL not in signals
    assert result.duration_s < 1.5
    assert ("proc.kill", "timeout") in events
    assert result.pid is not None
    assert _wait_dead(result.pid)


_GRANDCHILD_TIMEOUT = """
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
    time.sleep(120)
    os._exit(0)
for _ in range(500):
    if ready.exists():
        break
    time.sleep(0.01)
time.sleep(120)
"""

_GRANDCHILD_ORPHAN = """
import os, signal, sys, time
from pathlib import Path
ready = Path(sys.argv[1])
gpath = Path(sys.argv[2])
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpath.write_text(str(os.getpid()))
    ready.write_text("1")
    time.sleep(120)
    os._exit(0)
for _ in range(500):
    if ready.exists():
        os._exit(0)
    time.sleep(0.01)
os._exit(2)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")
def test_timeout_kills_grandchild_that_ignores_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "ready"
    gpath = tmp_path / "grand.pid"
    signals: list[int] = []
    real_killpg = os.killpg

    def _spy(pgid: int, sig: int) -> None:
        signals.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", _spy)
    result = None
    try:
        result = run_owned(
            _py(_GRANDCHILD_TIMEOUT, str(ready), str(gpath)),
            ProcLimits(timeout_s=3.0, kill_grace_s=0.25, cancel_poll_s=0.05),
            registry=ProcessRegistry(),
        )
        assert result.timed_out is True
        assert result.exit_code == EXIT_TIMEOUT
        assert ready.exists(), result.stderr
        assert gpath.exists()
        grand = int(gpath.read_text())
        assert result.pid is not None
        assert _wait_dead(result.pid)
        assert _wait_dead(grand)
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals
    finally:
        if gpath.exists():
            _sigkill(int(gpath.read_text() or "0"))
        if result is not None and result.pid is not None:
            _sigkill(result.pid)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")
def test_clean_exit_reaps_orphan_grandchild(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    gpath = tmp_path / "grand.pid"
    events: list[object] = []
    result = None
    try:
        result = run_owned(
            _py(_GRANDCHILD_ORPHAN, str(ready), str(gpath)),
            ProcLimits(timeout_s=5, kill_grace_s=0.25, cancel_poll_s=0.02),
            registry=ProcessRegistry(),
            emit=lambda _kind, fields: events.append(fields.get("reason")),
        )
        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.cancelled is False
        assert gpath.exists(), result.stderr
        assert _wait_dead(int(gpath.read_text()))
        assert "orphan" in events
    finally:
        if gpath.exists():
            _sigkill(int(gpath.read_text() or "0"))


def test_registry_tracks_step_and_redacted_argv_then_clears() -> None:
    registry = ProcessRegistry()
    cancel = threading.Event()

    def _worker() -> None:
        run_owned(
            _py("import time; time.sleep(30)") + ["--api-key", _SECRET],
            ProcLimits(timeout_s=8, kill_grace_s=0.2, cancel_poll_s=0.02),
            registry=registry,
            cancel_event=cancel,
            step="step-7",
        )

    thread = threading.Thread(target=_worker)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not registry.live():
        time.sleep(0.01)
    live = registry.live()
    assert len(live) == 1
    assert live[0].step == "step-7"
    assert live[0].pid > 0
    assert _SECRET not in " ".join(live[0].argv_summary)
    assert REDACTED in live[0].argv_summary
    cancel.set()
    thread.join(timeout=3)
    assert thread.is_alive() is False
    assert registry.live() == []


def test_terminate_all_stops_live_child() -> None:
    registry = ProcessRegistry()
    holder: dict[str, object] = {}

    def _worker() -> None:
        holder["result"] = run_owned(
            _py("import time; time.sleep(30)"),
            ProcLimits(timeout_s=8, kill_grace_s=0.2, cancel_poll_s=0.02),
            registry=registry,
        )

    thread = threading.Thread(target=_worker)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not registry.live():
        time.sleep(0.01)
    assert registry.live()
    pid = registry.live()[0].pid
    stopped = registry.terminate_all(grace_s=0.3)
    thread.join(timeout=3)
    assert stopped == 1
    assert thread.is_alive() is False
    assert registry.live() == []
    assert registry.terminate_all() == 0
    assert _wait_dead(pid)


def test_default_registry_is_used_when_omitted() -> None:
    cancel = threading.Event()

    def _worker() -> None:
        run_owned(
            _py("import time; time.sleep(30)"),
            ProcLimits(timeout_s=8, kill_grace_s=0.2, cancel_poll_s=0.02),
            cancel_event=cancel,
        )

    thread = threading.Thread(target=_worker)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not REGISTRY.live():
        time.sleep(0.01)
    assert REGISTRY.live()
    cancel.set()
    thread.join(timeout=3)
    assert thread.is_alive() is False
    assert REGISTRY.live() == []


def test_live_returns_a_copy() -> None:
    registry = ProcessRegistry()
    registry.add(
        LiveProc(pid=FAKE_PID, pgid=FAKE_PID, started_at=2.0, argv_summary=["b"], step=None)
    )
    registry.add(
        LiveProc(pid=FAKE_PID - 1, pgid=None, started_at=1.0, argv_summary=["a"], step="s")
    )
    snap = registry.live()
    assert [item.pid for item in snap] == [FAKE_PID - 1, FAKE_PID]
    snap.clear()
    snap_argv = registry.live()[0].argv_summary
    snap_argv.append("mutated")
    assert registry.live()[0].argv_summary == ["a"]
    registry.discard(FAKE_PID)
    registry.discard(FAKE_PID - 1)
    assert registry.live() == []


def test_install_atexit_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ProcessRegistry()
    calls: list[object] = []
    real_register = atexit.register

    def _spy(func: object, *args: object, **kwargs: object) -> object:
        calls.append(func)
        return real_register(func, *args, **kwargs)

    monkeypatch.setattr(atexit, "register", _spy)
    registry.install_atexit()
    registry.install_atexit()
    assert calls == [registry.terminate_all]


def test_argv_redacted_in_result_and_events() -> None:
    events: list[tuple[str, object]] = []
    result = run_owned(
        _py("print('hi')") + ["--api-key", _SECRET],
        _limits(),
        registry=ProcessRegistry(),
        emit=lambda kind, fields: events.append((kind, fields.get("argv"))),
    )
    assert result.exit_code == 0
    assert _SECRET not in " ".join(result.argv_summary)
    assert REDACTED in result.argv_summary
    names = [kind for kind, _argv in events]
    assert names == ["proc.spawn", "proc.exit"]
    blob = json.dumps(events)
    assert _SECRET not in blob


def test_posix_spawn_uses_new_session() -> None:
    captured: dict[str, object] = {}
    proc = _Immediate()

    def _popen(argv: list[str], **kwargs: object) -> _Immediate:
        captured.update(kwargs)
        return proc

    result = run_owned(
        ["tool"],
        _limits(),
        popen=_popen,
        registry=ProcessRegistry(),
        env={"PATH": "/bin"},
    )
    assert captured["start_new_session"] is True
    assert "creationflags" not in captured
    assert captured["stdin"] is subprocess.DEVNULL
    assert result.stdout == "hi\n"
    assert result.exit_code == 0


def test_windows_spawn_uses_new_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    captured: dict[str, object] = {}
    proc = _Immediate(stdout=b"win\n", stderr=b"e")

    def _popen(argv: list[str], **kwargs: object) -> _Immediate:
        captured.update(kwargs)
        return proc

    result = run_owned(
        ["tool"],
        _limits(),
        popen=_popen,
        registry=ProcessRegistry(),
        env={"PATH": "/bin"},
    )
    group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    assert captured["creationflags"] == group_flag
    assert group_flag == 0x00000200
    assert "start_new_session" not in captured
    assert result.pgid is None
    assert result.stdout == "win\n"
    assert result.stderr == "e"


def test_windows_timeout_uses_injected_killer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    proc = _Hang()
    killed: list[int] = []

    def _killer(pid: int) -> None:
        killed.append(pid)
        proc.mark(-9)

    def _popen(argv: list[str], **kwargs: object) -> _Hang:
        return proc

    result = run_owned(
        ["tool"],
        ProcLimits(timeout_s=0.05, cancel_poll_s=0.01, kill_grace_s=0.05),
        popen=_popen,
        killer=_killer,
        registry=ProcessRegistry(),
        env={"PATH": "/bin"},
    )
    assert killed == [FAKE_PID]
    assert result.timed_out is True
    assert result.exit_code == EXIT_TIMEOUT
    assert "[timeout]" in result.stderr


def test_windows_cancel_uses_injected_killer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    proc = _Hang()
    killed: list[int] = []
    cancel = threading.Event()

    def _killer(pid: int) -> None:
        killed.append(pid)
        proc.mark(-15)

    def _popen(argv: list[str], **kwargs: object) -> _Hang:
        cancel.set()
        return proc

    result = run_owned(
        ["tool"],
        ProcLimits(timeout_s=2, cancel_poll_s=0.01, kill_grace_s=0.05),
        popen=_popen,
        killer=_killer,
        cancel_event=cancel,
        registry=ProcessRegistry(),
        env={"PATH": "/bin"},
    )
    assert killed == [FAKE_PID]
    assert result.cancelled is True
    assert result.exit_code == EXIT_CANCEL
    assert result.timed_out is False


def test_windows_terminate_all_uses_taskkill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    killed: list[int] = []
    monkeypatch.setattr(procs, "_taskkill", lambda pid: killed.append(pid))
    registry = ProcessRegistry()
    registry.add(LiveProc(pid=77, pgid=None, started_at=0.0, argv_summary=["a"], step=None))
    assert registry.terminate_all(grace_s=0.0) == 1
    assert killed == [77]
    assert registry.live() == []
    assert registry.terminate_all() == 0


def test_taskkill_argv_and_refuses_self(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(procs.subprocess, "run", _fake_run)
    procs._taskkill(42)
    assert seen["argv"] == ["taskkill", "/PID", "42", "/T", "/F"]
    seen.clear()
    procs._taskkill(os.getpid())
    assert seen == {}


def test_taskkill_missing_binary_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(argv: list[str], **kwargs: object) -> None:
        raise FileNotFoundError("taskkill")

    monkeypatch.setattr(procs.subprocess, "run", _missing)
    procs._taskkill(42)
