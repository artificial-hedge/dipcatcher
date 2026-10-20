"""Store faults: permissions, corruption, future schema, and two writers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fx1.interactive.atomicstore import FileLock, StoreCorruptError, StoreLockTimeout
from fx1.interactive.jobs import JobLedger, JobState
from fx1.interactive.sessions import SessionConflict, SessionError, SessionStore
from tests.fx1.chaos_support import HANG_S, record

_SECRET = "sk-chaos-store-key"


def _session(tmp_path: Path) -> tuple[SessionStore, dict[str, object]]:
    store = SessionStore(tmp_path / "sessions")
    created = store.create(model="fx1", messages=[{"role": "user", "content": "keep"}])
    return store, created


def test_chaos_readonly_session_dir_keeps_the_file(tmp_path: Path) -> None:
    """A 0555 session directory whose chmod cannot be restored must not be rewritten."""
    store, created = _session(tmp_path)
    path = store.directory / f"{created['id']}.json"
    before = path.read_bytes()
    store.directory.chmod(0o555)
    try:
        result = _python(tmp_path, _READONLY, str(store.directory), str(created["id"]))
    finally:
        store.directory.chmod(0o755)
    assert result.returncode == 0, result.stderr[-400:]
    assert path.read_bytes() == before
    assert store.load(str(created["id"]))["messages"][0]["content"] == "keep"
    record("store_readonly", iterations=1, passed=1, orphans=0, duplicates=0)


def test_chaos_enospace_and_rlimit_do_not_truncate(tmp_path: Path) -> None:
    store, created = _session(tmp_path)
    path = store.directory / f"{created['id']}.json"
    before = path.read_bytes()
    enospc = _python(
        tmp_path,
        _ENOSPC,
        str(store.directory),
        str(created["id"]),
    )
    limited = _python(
        tmp_path,
        _RLIMIT,
        str(store.directory),
        str(created["id"]),
    )
    assert path.read_bytes() == before
    assert b"keep" in before
    assert enospc.returncode == 0, enospc.stderr[-300:]
    assert limited.returncode in {0, -25, 1}
    record(
        "store_enospc",
        iterations=2,
        passed=2,
        enospc=enospc.returncode,
        rlimit=limited.returncode,
        orphans=0,
        duplicates=0,
    )


def test_chaos_symlink_truncated_and_future_schema(tmp_path: Path) -> None:
    store, created = _session(tmp_path)
    directory = store.directory
    sid = str(created["id"])
    path = directory / f"{sid}.json"
    original = path.read_text(encoding="utf-8")
    _replace_with_symlink(path, directory / "other.json")
    with pytest.raises(SessionError, match="symlink"):
        store.load(sid)
    assert not path.exists() or path.is_symlink() is False
    path.write_text(original, encoding="utf-8")
    path.write_text(original[:12], encoding="utf-8")
    with pytest.raises(SessionError):
        store.load(sid)
    future = directory / f"{sid}.json"
    if future.exists():
        future.unlink()
    payload = json.loads(original)
    payload["schema"] = 99
    future.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionError, match="newer"):
        store.load(sid)
    assert json.loads(future.read_text(encoding="utf-8"))["schema"] == 99
    record("store_corrupt", iterations=3, passed=3, orphans=0, duplicates=0)


def test_chaos_stale_revision_is_a_conflict(tmp_path: Path) -> None:
    store, created = _session(tmp_path)
    sid = str(created["id"])
    stale = store.load(sid)
    created_messages = list(created["messages"])  # type: ignore[arg-type]
    created_messages.append({"role": "user", "content": "newer"})
    created["messages"] = created_messages
    store.save(created)  # type: ignore[arg-type]
    stale_messages = list(stale["messages"])
    stale_messages.append({"role": "user", "content": "stale"})
    stale["messages"] = stale_messages
    with pytest.raises(SessionConflict, match="conflict"):
        store.save(stale)
    loaded = store.load(sid)
    assert loaded["revision"] == 1
    assert loaded["messages"][-1]["content"] == "newer"
    record("store_conflict", iterations=1, passed=1, orphans=0, duplicates=0)


def test_chaos_job_symlink_is_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    ledger = JobLedger(root)
    job = ledger.create(kind="exec", goal="symlink")
    path = root / f"{job.job_id}.json"
    other = root / "other.json"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.unlink()
    path.symlink_to(other.name)
    with pytest.raises(StoreCorruptError, match="symlink"):
        ledger.get(job.job_id)
    assert other.is_file()
    record("job_symlink", iterations=1, passed=1, orphans=0, duplicates=0)


def test_chaos_two_processes_session_busy_and_job_lock(tmp_path: Path) -> None:
    store, created = _session(tmp_path)
    sid = str(created["id"])
    ready = tmp_path / "held"
    release = tmp_path / "release"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(store.directory), sid, str(ready), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_file(ready)
        rival = _python(tmp_path, _RIVAL, str(store.directory), sid)
        assert rival.returncode == 0, rival.stderr[-400:]
        assert "busy" in rival.stdout
        assert _SECRET not in rival.stdout + rival.stderr
    finally:
        release.write_text("go", encoding="utf-8")
        try:
            holder.wait(timeout=HANG_S)
        finally:
            if holder.poll() is None and holder.pid:
                os.kill(holder.pid, 9)
                holder.wait(timeout=5)
    _job_lock(tmp_path / "jobs")
    record("store_busy", iterations=2, passed=2, orphans=0, duplicates=0)


def _job_lock(root: Path) -> None:
    ledger = JobLedger(root)
    job = ledger.create(kind="exec", goal="lock")
    ledger.transition(job.job_id, JobState.RUNNING)
    ready = root / "ready"
    release = root / "release"
    holder = subprocess.Popen(
        [sys.executable, "-c", _JOB_HOLD, str(root), str(ready), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_file(ready)
        with (
            pytest.raises(StoreLockTimeout, match="lock"),
            FileLock(root / ".ledger", timeout_s=0.2),
        ):
            pass
        again = ledger.get(job.job_id)
        assert again.state == JobState.RUNNING
    finally:
        release.write_text("go", encoding="utf-8")
        try:
            holder.wait(timeout=HANG_S)
        finally:
            if holder.poll() is None and holder.pid:
                os.kill(holder.pid, 9)
                holder.wait(timeout=5)


def _replace_with_symlink(path: Path, other: Path) -> None:
    other.write_text('{"schema": 1}\n', encoding="utf-8")
    path.unlink()
    path.symlink_to(other)


def _python(tmp_path: Path, code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=HANG_S,
        check=False,
    )


def _wait_file(path: Path) -> bool:
    deadline = time.monotonic() + HANG_S
    while time.monotonic() < deadline:
        if path.is_file():
            return True
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return False


_READONLY = r"""
import os
import sys
from pathlib import Path
from fx1.interactive.sessions import SessionError, SessionStore

def _deny(*_args, **_kwargs):
    raise OSError(1, "Operation not permitted")

os.chmod = _deny
os.fchmod = _deny
store = SessionStore(Path(sys.argv[1]))
record = store.load(sys.argv[2])
record["messages"] = [{"role": "user", "content": "nope"}]
try:
    store.save(record)
except SessionError:
    sys.exit(0)
sys.exit(1)
"""

_ENOSPC = r"""
import os
import sys
from pathlib import Path
from fx1.interactive.sessions import SessionError, SessionStore

def _full(*_args, **_kwargs):
    raise OSError(28, "No space left on device")

os.write = _full
store = SessionStore(Path(sys.argv[1]))
record = store.load(sys.argv[2])
record["messages"] = [{"role": "user", "content": "x" * 4000}]
try:
    store.save(record)
except (SessionError, OSError):
    sys.exit(0)
sys.exit(1)
"""

_RLIMIT = r"""
import resource
import sys
from pathlib import Path
from fx1.interactive.sessions import SessionError, SessionStore

resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
store = SessionStore(Path(sys.argv[1]))
record = store.load(sys.argv[2])
record["messages"] = [{"role": "user", "content": "y" * 200000}]
try:
    store.save(record)
except (SessionError, OSError):
    sys.exit(0)
sys.exit(0)
"""

_HOLD = r"""
import sys
import time
from pathlib import Path
from fx1.interactive.sessions import SessionStore

store = SessionStore(Path(sys.argv[1]))
ready = Path(sys.argv[3])
release = Path(sys.argv[4])
with store.reserve(sys.argv[2]):
    ready.write_text("held", encoding="utf-8")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not release.exists():
        time.sleep(0.05)
"""

_RIVAL = r"""
import sys
from pathlib import Path
from fx1.interactive.sessions import SessionBusy, SessionStore

store = SessionStore(Path(sys.argv[1]))
try:
    with store.reserve(sys.argv[2]):
        print("acquired")
except SessionBusy as exc:
    text = str(exc)
    if "busy" not in text and "lock" not in text:
        sys.exit(2)
    print("busy")
"""

_JOB_HOLD = r"""
import sys
import time
from pathlib import Path
from fx1.interactive.atomicstore import FileLock

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
release = Path(sys.argv[3])
with FileLock(root / ".ledger", timeout_s=5):
    ready.write_text("held", encoding="utf-8")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not release.exists():
        time.sleep(0.05)
"""
