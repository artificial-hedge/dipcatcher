"""The dataset scoring journey through a real pty and ``dipcatcher chat``."""

from __future__ import annotations

import contextlib
import os
import pty
import re
import select
import sys
import time
from pathlib import Path

import pytest

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX pty required")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[78]")
_PROMPT = re.compile(r"fx1 › $", re.M)
_KEY = "sk-test-fx1-secret-value-aaa111"
_LITE = "sk-test-lite-secret-value-bbb222"
_HANG_S = 180.0


def _plain(text: str) -> str:
    return _ANSI.sub("", text).replace("\r", "")


def _csv(path: Path, rows: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["timestamp,realized,forecast_var"]
    for index in range(rows):
        day = index // 24 + 1
        hour = index % 24
        lines.append(f"2020-01-{day:02d}T{hour:02d}:00:00+00:00,1.0,1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _env(root: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "INPUTRC": "/dev/null",
        "NO_COLOR": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "FX1_CONFIG_DIR": str(root / "cfg"),
        "FX1_SESSION_DIR": str(root / "sessions"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


class _Pty:
    def __init__(self, env: dict[str, str], cwd: Path) -> None:
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(cwd)
            os.execvpe(
                sys.executable,
                [sys.executable, "-m", "dipcatcher_cli", "chat", "--model", "fx1"],
                env,
            )
            os._exit(127)
        self.buf = ""

    def read_until(self, timeout: float) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, 65536).decode("utf-8", "replace")
            except OSError:
                break
            if not chunk:
                break
            self.buf += chunk
            if _PROMPT.search(_plain(self.buf)):
                return _plain(self.buf)
        return _plain(self.buf)

    def say(self, line: str, timeout: float) -> str:
        self.buf = ""
        os.write(self.fd, (line + "\n").encode())
        text = self.read_until(timeout)
        if not _PROMPT.search(text):
            raise AssertionError(f"no prompt after {line!r}: {text[-800:]}")
        return text

    def eof(self, timeout: float) -> int:
        os.write(self.fd, b"\x04")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            finished, status = os.waitpid(self.pid, os.WNOHANG)
            if finished:
                self.pid = 0
                return os.waitstatus_to_exitcode(status)
            time.sleep(0.05)
        raise AssertionError("child did not exit")

    def close(self) -> None:
        pid = self.pid
        if pid:
            with contextlib.suppress(OSError):
                os.kill(pid, 15)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            self.pid = 0
        with contextlib.suppress(OSError):
            os.close(self.fd)


@posix_only
def test_pty_scores_the_attached_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.interactive import profiles
    from tests.fx1.stub_model_server import StubModelServer

    work = tmp_path / "work"
    work.mkdir()
    _csv(work / "data" / "vol.csv")
    export = work / "session.md"
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with StubModelServer({"fx1": _KEY, "fx1-lite": _LITE}) as server:
        profiles.set_endpoint("fx1", _KEY, server.url)
        profiles.set_endpoint("fx1-lite", _LITE, server.url)
        session = _Pty(env, work)
        try:
            banner = session.read_until(60)
            assert _PROMPT.search(banner), banner[-800:]
            chunks = [banner]
            chunks.append(session.say("/data attach vol.csv", 90))
            chunks.append(
                session.say(
                    "/superpower Check whether my volatility forecasts are calibrated "
                    "using QLIKE --sync",
                    _HANG_S,
                )
            )
            chunks.append(session.say("/evidence", 60))
            chunks.append(session.say("/runs", 60))
            chunks.append(session.say(f"/export {export}", 60))
            code = session.eof(60)
        finally:
            session.close()
    full = "\n".join(chunks)
    artifact = Path("/opt/cursor/artifacts")
    if artifact.is_dir():
        (artifact / "chat_data_pty.txt").write_text(full + f"\nexit={code}\n", encoding="utf-8")
    assert code == 0
    assert not any(line.strip().casefold().startswith("error:") for line in full.splitlines())
    assert _KEY not in full
    assert _LITE not in full
    evidence = full.split("/evidence", 1)[-1].split("/runs", 1)[0]
    runs = full.split("/runs", 1)[-1].split("/export", 1)[0]
    assert len(re.findall(r"VERIFIED", evidence)) >= 1
    assert len(re.findall(r"^j-\d{8}T\S+", runs, re.M)) == 1
    assert export.is_file()
    assert export.stat().st_mode & 0o777 == 0o600
    exported = export.read_text(encoding="utf-8")
    assert _KEY not in exported
    assert _LITE not in exported
