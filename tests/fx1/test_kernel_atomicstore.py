"""Atomic writes survive crashes; locks serialize writers; corruption is explicit."""

from __future__ import annotations

import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive import atomicstore
from fx1.interactive.atomicstore import (
    FileLock,
    StoreCorruptError,
    StoreLockTimeout,
    atomic_write_json,
    atomic_write_text,
    permission_problem,
    read_json,
    update_json,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


def test_roundtrip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "state.json"
    atomic_write_json(path, {"b": 2, "a": [1, "é"]})
    assert read_json(path) == {"a": [1, "é"], "b": 2}
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index(
        '"b"'
    )


@posix_only
def test_file_and_new_dir_are_private(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "state.json"
    atomic_write_json(path, {"k": 1})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert permission_problem(path) is None
    path.chmod(0o644)
    problem = permission_problem(path)
    assert problem is not None and "chmod 600" in problem


def test_failed_replace_keeps_old_content_and_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"v": "old"})

    def boom(src: str, dst: Any) -> None:
        raise OSError("disk exploded")

    monkeypatch.setattr(atomicstore.os, "replace", boom)
    with pytest.raises(OSError, match="disk exploded"):
        atomic_write_json(path, {"v": "new"})
    monkeypatch.undo()
    assert read_json(path) == {"v": "old"}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_corrupt_file_is_explicit_and_quarantinable(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_text(path, "{not json")
    with pytest.raises(StoreCorruptError) as info:
        read_json(path)
    assert info.value.path == path and path.exists()
    with pytest.raises(StoreCorruptError):
        read_json(path, quarantine_corrupt=True)
    assert not path.exists()
    assert any(p.name.startswith("state.json.corrupt-") for p in tmp_path.iterdir())
    with pytest.raises(FileNotFoundError):
        read_json(path)


def test_non_utf8_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bin.json"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(StoreCorruptError):
        read_json(path)


def test_lock_times_out_while_held(tmp_path: Path) -> None:
    target = tmp_path / "ledger.json"
    with (
        FileLock(target),
        pytest.raises(StoreLockTimeout),
        FileLock(target, timeout_s=0.1, poll_s=0.01),
    ):
        pass
    with FileLock(target, timeout_s=0.1):
        pass


def test_update_json_has_no_lost_updates_across_threads(tmp_path: Path) -> None:
    path = tmp_path / "counter.json"
    n_threads, per_thread = 6, 15

    def work() -> None:
        for _ in range(per_thread):
            update_json(path, lambda cur: {"n": cur["n"] + 1}, default={"n": 0}, lock_timeout_s=10)

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert read_json(path) == {"n": n_threads * per_thread}


def test_update_json_does_not_overwrite_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_text(path, "garbage")
    with pytest.raises(StoreCorruptError):
        update_json(path, lambda cur: cur, default={})
    assert path.read_text(encoding="utf-8") == "garbage"


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl cross-process semantics")
def test_lock_excludes_other_processes(tmp_path: Path) -> None:
    import subprocess

    target = tmp_path / "x.json"
    code = (
        "import sys; from pathlib import Path; "
        "from fx1.interactive.atomicstore import FileLock, StoreLockTimeout\n"
        "try:\n"
        "    FileLock(Path(sys.argv[1]), timeout_s=0.2).__enter__()\n"
        "    print('acquired')\n"
        "except StoreLockTimeout:\n"
        "    print('timeout')\n"
    )
    with FileLock(target):
        held = subprocess.run(
            [sys.executable, "-c", code, str(target)], capture_output=True, text=True, check=True
        )
    free = subprocess.run(
        [sys.executable, "-c", code, str(target)], capture_output=True, text=True, check=True
    )
    assert held.stdout.strip() == "timeout"
    assert free.stdout.strip() == "acquired"
