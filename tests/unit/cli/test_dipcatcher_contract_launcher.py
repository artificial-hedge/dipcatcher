"""Golden subprocess contract for the dipcatcher launcher."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fx1 import __version__
from tests.fx1.ux_pty import hermetic_network, prepare, run_pty

ROOT = Path(__file__).resolve().parents[3]
_SECRET = "sk-contract-planted-secret"
_PY = sys.executable


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not _sensitive(key)}
    env.update(
        {
            "FX1_CONFIG_DIR": str(tmp_path / "cfg"),
            "FX1_SESSION_DIR": str(tmp_path / "sessions"),
            "HOME": str(tmp_path / "home"),
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.update(hermetic_network())
    env.update(extra)
    for name in ("cfg", "sessions", "home"):
        (tmp_path / name).mkdir(exist_ok=True)
    return env


def _sensitive(key: str) -> bool:
    folded = key.casefold()
    return folded.endswith(("_api_key", "_token", "_secret")) or "password" in folded


def _run(
    args: list[str], env: dict[str, str], *, stdin: bytes | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PY, "-m", "dipcatcher_cli", *args],
        cwd=ROOT,
        env=env,
        input=stdin.decode() if isinstance(stdin, bytes) else None,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_dipcatcher_contract_help_preface(tmp_path: Path, flag: str) -> None:
    proc = _run([flag], _env(tmp_path, MOONSHOT_API_KEY=_SECRET))
    assert proc.returncode == 0
    assert "Start a conversation: dipcatcher" in proc.stdout
    assert "Usage:" in proc.stdout
    assert proc.stdout.count("Start a conversation") == 1
    assert _SECRET not in proc.stdout + proc.stderr


def test_dipcatcher_contract_bare_pipe_does_not_block(tmp_path: Path) -> None:
    proc = subprocess.Popen(
        [_PY, "-m", "dipcatcher_cli"],
        cwd=ROOT,
        env=_env(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    deadline = time.monotonic() + 60
    out = bytearray()
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            import select

            ready, _, _ = select.select([proc.stdout], [], [], 0.2)
            if ready:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if chunk:
                    out.extend(chunk)
        if proc.poll() is None:
            raise AssertionError("bare dipcatcher blocked on an open pipe")
        rest = proc.stdout.read() or b""
        err = proc.stderr.read() if proc.stderr is not None else b""
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        proc.stdin.close()
    text = (bytes(out) + rest).decode("utf-8", "replace")
    assert proc.returncode == 0
    assert "Start a conversation" in text
    assert _SECRET not in text + err.decode("utf-8", "replace")


def test_dipcatcher_contract_version(tmp_path: Path) -> None:
    exact = _run(["version"], _env(tmp_path))
    assert exact.returncode == 0
    assert exact.stdout.strip() == f"dipcatcher {__version__}"
    flagged = _run(["--version"], _env(tmp_path))
    assert flagged.returncode == 0
    assert __version__ in flagged.stdout
    extra = _run(["version", "--nope"], _env(tmp_path))
    assert extra.returncode == 2


def test_dipcatcher_contract_lab_passthrough(tmp_path: Path) -> None:
    env = _env(tmp_path, DIPCATCHER_CHILD="1", MOONSHOT_API_KEY=_SECRET)
    doctor = _run(["doctor", "--help"], env)
    assert doctor.returncode == 0
    assert "doctor" in doctor.stdout.lower()
    assert "Start a conversation" not in doctor.stdout
    assert "fx1 ›" not in doctor.stdout
    assert _SECRET not in doctor.stdout + doctor.stderr
    unknown = _run(["not-a-dipcatcher-command"], _env(tmp_path))
    assert unknown.returncode == 2
    chat = _run(["chat"], _env(tmp_path))
    assert chat.returncode == 2
    assert "interactive terminal" in chat.stderr


def test_dipcatcher_contract_child_tty_does_not_open_chat(tmp_path: Path) -> None:
    root = prepare(tmp_path, "child")
    result = run_pty(
        root,
        [("wait", b"Start a conversation")],
        args=(),
        extra={"DIPCATCHER_CHILD": "1", "TERM": "xterm-256color"},
        wrap=False,
    )
    assert result.status == 0
    assert "fx1 ›" not in result.text
    assert "fx1 >" not in result.text
