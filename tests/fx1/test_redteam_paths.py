"""Adversarial path, file, and archive checks for the chat harness.

Each test states the safe outcome. A planted symlink, hard link, FIFO,
device, or race must not redirect a write, block a read, or publish a
secret.
"""

from __future__ import annotations

import io
import json
import os
import stat
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import fx1.interactive.evidence_ledger as evidence_mod
from dipcatcher_cli.sessions_cmd import _destination, _UsageError, _write_export
from fx1.interactive import jobs as jobs_mod
from fx1.interactive import sessions as sessions_mod
from fx1.interactive.atomicstore import (
    FILE_MODE,
    FileLock,
    StoreError,
    atomic_write_bytes,
    atomic_write_json,
    ensure_private_dir,
    read_json,
)
from fx1.interactive.contracts import ContractViolation
from fx1.interactive.diagnostics import write_bundle
from fx1.interactive.evidence_ledger import Verdict, verify_receipt
from fx1.interactive.evidence_ledger import _file_size as receipt_file_size
from fx1.interactive.headless import _write_last_message
from fx1.interactive.jobs import IllegalTransition, JobLedger
from fx1.interactive.orchestration import RunState, _argv_for, _prepare_dir
from fx1.interactive.planner import PlanStep
from fx1.interactive.policy import PolicyViolation, contain_path
from fx1.interactive.profiles import set_endpoint, store_health
from fx1.interactive.runctx import new_run_id, new_session_id
from fx1.interactive.session_ux import export_session
from fx1.interactive.sessions import SessionError, SessionStore
from fx1.interactive.superpower import SuperpowerContext, SuperpowerPlan, SuperpowerResult
from fx1.interactive.telemetry import Telemetry
from fx1.interactive.terminal import ConsoleWriter, LineReader, TermCaps
from tests.fx1.capabilities_fixture import sample_catalog

SECRET = "RT3-SECRET-PATH-7f3a"
_POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode checks")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _bounded(fn: Callable[[], None], timeout: float = 0.3) -> BaseException | None:
    holder: dict[str, Any] = {"err": None, "done": False}

    def run() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - the assertion is the outcome
            holder["err"] = exc
        holder["done"] = True

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert holder["done"], "call blocked"
    assert not thread.is_alive()
    err = holder["err"]
    return err if isinstance(err, BaseException) else None


def _lie_missing(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    real_exists = Path.exists
    real_lexists = os.path.lexists

    def exists(self: Path) -> bool:
        if self == target:
            return False
        return real_exists(self)

    def lexists(path: os.PathLike[str] | str) -> bool:
        if Path(path) == target:
            return False
        return real_lexists(path)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(os.path, "lexists", lexists)


def _export_store() -> Any:
    class _Store:
        def export_markdown(self, record: object) -> str:
            del record
            return "transcript\n"

        def export_json(self, record: object) -> str:
            del record
            return "{}\n"

    return _Store()


def _caps() -> TermCaps:
    return TermCaps(False, False, False, False, True, 80, 24, False)


@_POSIX
def test_ancestor_symlink_is_not_a_private_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoreError, match="symlink"):
        ensure_private_dir(link / "private")
    assert not (outside / "private").exists()


@_POSIX
def test_new_parents_stay_private_under_umask_zero(tmp_path: Path) -> None:
    previous = os.umask(0)
    try:
        path = tmp_path / "a" / "b" / "state.json"
        atomic_write_json(path, {"k": 1})
    finally:
        os.umask(previous)
    assert _mode(tmp_path / "a") == 0o700
    assert _mode(tmp_path / "a" / "b") == 0o700
    assert _mode(path) == FILE_MODE
    assert read_json(path) == {"k": 1}


@_POSIX
def test_read_json_does_not_follow_a_symlink(tmp_path: Path) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"v": SECRET}), encoding="utf-8")
    link = tmp_path / "state.json"
    link.symlink_to(secret)
    with pytest.raises(OSError):
        read_json(link)
    assert secret.read_text(encoding="utf-8").find(SECRET) >= 0


@_POSIX
def test_read_json_does_not_block_on_a_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "state.json"
    os.mkfifo(fifo)
    err = _bounded(lambda: read_json(fifo))
    assert isinstance(err, OSError)
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@_POSIX
def test_read_json_does_not_drain_a_device(tmp_path: Path) -> None:
    del tmp_path
    zero = Path("/dev/zero")
    if not zero.exists():
        pytest.skip("no /dev/zero")
    err = _bounded(lambda: read_json(zero))
    assert isinstance(err, OSError)


@_POSIX
def test_read_json_rejects_a_huge_sparse_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    os.truncate(path, 32 * 1024 * 1024)
    reads: list[int] = []
    real_read = os.read

    def spy(fd: int, size: int) -> bytes:
        block = real_read(fd, size)
        reads.append(len(block))
        return block

    monkey_read = spy
    original = os.read
    os.read = monkey_read  # type: ignore[assignment]
    try:
        with pytest.raises(OSError):
            read_json(path)
    finally:
        os.read = original
    assert sum(reads) < 1024 * 1024


@_POSIX
def test_read_json_does_not_open_a_hardlink_target(tmp_path: Path) -> None:
    victim = tmp_path / "victim.json"
    victim.write_text(json.dumps({"v": SECRET}), encoding="utf-8")
    os.chmod(victim, 0o644)
    link = tmp_path / "state.json"
    os.link(victim, link)
    with pytest.raises(OSError):
        read_json(link)
    assert _mode(victim) == 0o644
    assert SECRET in victim.read_text(encoding="utf-8")


@_POSIX
def test_lock_file_symlink_is_not_followed(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text(SECRET, encoding="utf-8")
    os.chmod(victim, 0o644)
    lock = tmp_path / "state.json.lock"
    lock.symlink_to(victim)
    with pytest.raises(OSError), FileLock(tmp_path / "state.json", timeout_s=0.2):
        pass
    assert victim.read_text(encoding="utf-8") == SECRET
    assert _mode(victim) == 0o644
    assert lock.is_symlink()


@_POSIX
def test_lock_file_fifo_does_not_block(tmp_path: Path) -> None:
    fifo = tmp_path / "state.json.lock"
    os.mkfifo(fifo)

    def take() -> None:
        with FileLock(tmp_path / "state.json", timeout_s=0.2):
            pass

    err = _bounded(take)
    assert isinstance(err, (OSError, StoreError))
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@_POSIX
def test_existing_lock_is_tightened_to_owner_only(tmp_path: Path) -> None:
    lock = tmp_path / "state.json.lock"
    lock.write_bytes(b"\0")
    os.chmod(lock, 0o666)
    with FileLock(tmp_path / "state.json", timeout_s=0.2):
        assert _mode(lock) == 0o600


@_POSIX
def test_exclusive_publish_ignores_a_lost_exists_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    _lie_missing(monkeypatch, target)
    with pytest.raises(FileExistsError):
        atomic_write_bytes(target, b"NEW", exclusive=True)
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


@_POSIX
def test_export_does_not_follow_an_ancestor_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    dest = link / "nested" / "chat.md"
    written, message = export_session(_export_store(), {"id": "s"}, "md", dest, tmp_path / "cfg")
    assert written is None
    assert "refusing" in message
    assert not (outside / "nested").exists()


@_POSIX
def test_export_overwrite_race_keeps_the_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "chat.md"
    target.write_text("ORIGINAL", encoding="utf-8")
    _lie_missing(monkeypatch, target)
    written, _message = export_session(_export_store(), {"id": "s"}, "md", target, tmp_path / "cfg")
    assert written is None
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


@_POSIX
def test_export_directories_and_file_are_private_under_umask_zero(tmp_path: Path) -> None:
    dest = tmp_path / "exports" / "nested" / "chat.md"
    previous = os.umask(0)
    try:
        written, _message = export_session(
            _export_store(), {"id": "s"}, "md", dest, tmp_path / "cfg"
        )
    finally:
        os.umask(previous)
    assert written == dest
    assert _mode(tmp_path / "exports") == 0o700
    assert _mode(dest.parent) == 0o700
    assert _mode(dest) == 0o600


def test_export_rejects_windows_and_unc_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for raw in (r"C:\Windows\system32\rt3.md", r"\\server\share\rt3.md", "//server/share/rt3.md"):
        written, message = export_session(
            _export_store(), {"id": "s"}, "md", Path(raw), tmp_path / "cfg"
        )
        assert written is None
        assert "refusing" in message
    names = [item.name for item in tmp_path.iterdir()]
    assert names == [] or names == ["cfg"] or "cfg" in names
    assert not any("\\" in name or name.startswith("C:") for name in names)


def test_export_tilde_is_not_expanded_into_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(tmp_path)
    dest = Path("~/rt3-export.md")
    written, _message = export_session(_export_store(), {"id": "s"}, "md", dest, tmp_path / "cfg")
    assert written is not None
    assert not (home / "rt3-export.md").exists()
    assert SECRET not in home.as_posix()


def test_sessions_out_rejects_empty_null_and_tilde_expansion(tmp_path: Path) -> None:
    with pytest.raises(_UsageError):
        _destination("  ")
    with pytest.raises(_UsageError):
        _destination("a\x00b")
    with pytest.raises(_UsageError):
        _destination("a\nb")
    chosen = _destination("~/rt3-out.md")
    assert chosen is not None
    assert chosen.parts[0] == "~"
    assert not str(chosen).startswith(str(tmp_path))


@_POSIX
def test_sessions_out_exclusive_write_does_not_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.md"
    target.write_text("ORIGINAL", encoding="utf-8")
    _lie_missing(monkeypatch, target)
    with pytest.raises(_UsageError):
        _write_export(target, "NEW")
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


@_POSIX
def test_last_message_ancestor_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    err = _write_last_message(link / "nested" / "last.txt", "answer")
    assert err is not None
    assert "could not write" in err
    assert not (outside / "nested").exists()


@_POSIX
def test_last_message_symlink_does_not_replace_its_target(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    target.write_text(SECRET, encoding="utf-8")
    link = tmp_path / "last.txt"
    link.symlink_to(target)
    err = _write_last_message(link, "answer")
    assert err is not None
    assert target.read_text(encoding="utf-8") == SECRET
    assert link.is_symlink()


@_POSIX
def test_last_message_does_not_replace_a_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "last.txt"
    os.mkfifo(fifo)
    err = _bounded(lambda: _write_last_message(fifo, "answer"))
    assert err is None or isinstance(err, OSError)
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


def test_last_message_rejects_null_and_windows_strings(tmp_path: Path) -> None:
    del tmp_path
    assert _write_last_message("a\x00b", "answer") is not None
    err = _write_last_message(r"C:\Windows\rt3-last.txt", "answer")
    assert err is not None
    unc = _write_last_message(r"\\server\share\rt3-last.txt", "answer")
    assert unc is not None


@_POSIX
def test_bundle_hardlink_does_not_copy_the_target(tmp_path: Path) -> None:
    secret = tmp_path / "secret.jsonl"
    secret.write_text(json.dumps({"note": SECRET}) + "\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    link = logs / "events.jsonl"
    os.link(secret, link)
    bundle = tmp_path / "support.zip"
    write_bundle({"model": "fx1"}, bundle, telemetry_files=[link])
    raw = bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        members = b"\n".join(archive.read(name) for name in names)
    assert SECRET.encode() not in raw
    assert SECRET.encode() not in members
    assert _mode(bundle) == 0o600


@_POSIX
def test_bundle_names_cannot_zip_slip(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    hostile = (
        "..jsonl",
        "foo..bar.jsonl",
        "a" * 200 + ".jsonl",
        "foo\\..\\bar.jsonl",
        ".jsonl",
    )
    paths: list[Path] = []
    for name in hostile:
        path = logs / name
        path.write_text("{}\n", encoding="utf-8")
        paths.append(path)
    odd = logs / os.fsdecode(b"\xff\xfe.jsonl")
    fd = os.open(odd, os.O_CREAT | os.O_WRONLY, 0o600)
    os.write(fd, b"{}\n")
    os.close(fd)
    paths.append(odd)
    bundle = tmp_path / "names.zip"
    write_bundle({"model": "fx1"}, bundle, telemetry_files=paths)
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            assert ".." not in name
            assert "\\" not in name
            assert not name.startswith("/")
            assert not name.startswith("telemetry/") or name.count("/") == 1


@_POSIX
def test_bundle_parents_are_private_under_umask_zero(tmp_path: Path) -> None:
    bundle = tmp_path / "out" / "nested" / "support.zip"
    previous = os.umask(0)
    try:
        write_bundle({"model": "fx1"}, bundle)
    finally:
        os.umask(previous)
    assert _mode(tmp_path / "out") == 0o700
    assert _mode(bundle.parent) == 0o700
    assert _mode(bundle) == 0o600


@_POSIX
def test_bundle_symlink_destination_is_not_replaced(tmp_path: Path) -> None:
    victim = tmp_path / "victim.zip"
    victim.write_bytes(b"ORIGINAL")
    link = tmp_path / "support.zip"
    link.symlink_to(victim)
    with pytest.raises(FileExistsError):
        write_bundle({"model": "fx1"}, link)
    assert victim.read_bytes() == b"ORIGINAL"
    assert link.is_symlink()


def test_contain_path_rejects_overlong_null_tilde_and_sibling(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "inner.txt").write_text("ok", encoding="utf-8")
    with pytest.raises(PolicyViolation) as too_long:
        contain_path("a" * 5000, [root])
    assert too_long.value.code == "path_escape"
    for raw in ("a\x00b", "~/secret", "sub/~/x", r"C:\Windows\system32", r"\\server\share\a"):
        with pytest.raises(PolicyViolation) as exc:
            contain_path(raw, [root])
        assert exc.value.code == "path_escape"
    sibling = tmp_path / "root-evil"
    sibling.mkdir()
    with pytest.raises(PolicyViolation):
        contain_path(str(sibling / "inner.txt"), [root])
    assert contain_path("inner.txt", [root]) == (root / "inner.txt").resolve()


@_POSIX
def test_contain_path_rejects_a_symlink_leaf_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text(SECRET, encoding="utf-8")
    link = root / "alias.txt"
    link.symlink_to(secret)
    with pytest.raises(PolicyViolation) as exc:
        contain_path("alias.txt", [root])
    assert exc.value.code == "path_escape"
    assert SECRET not in str(exc.value)


@_POSIX
def test_receipt_parent_swap_does_not_read_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True)
    receipt = sub / "r.json"
    receipt.write_text('{"data_label":"REAL"}\n', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "r.json").write_text(json.dumps({"note": SECRET}), encoding="utf-8")
    swapped = {"done": False}

    def swap_size(path: Path) -> int | None:
        size = receipt_file_size(path)
        if path == receipt and not swapped["done"]:
            swapped["done"] = True
            os.rename(sub, tmp_path / "sub-real")
            sub.symlink_to(outside, target_is_directory=True)
        return size

    reads: list[bytes] = []
    real_read = evidence_mod.os.read

    def spy(fd: int, size: int) -> bytes:
        block = real_read(fd, size)
        reads.append(block)
        return block

    monkeypatch.setattr(evidence_mod, "_file_size", swap_size)
    monkeypatch.setattr(evidence_mod.os, "read", spy)
    calls: list[Path] = []

    def verifier(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {"valid": True}

    item = verify_receipt(receipt, roots=[root], verifier=verifier)
    assert swapped["done"]
    assert calls == []
    assert item.verdict is not Verdict.VERIFIED
    assert SECRET.encode() not in b"".join(reads)
    assert SECRET not in item.detail
    assert (outside / "r.json").read_text(encoding="utf-8").find(SECRET) >= 0


@_POSIX
def test_receipt_fifo_is_unverifiable_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    fifo = root / "r.json"
    os.mkfifo(fifo)
    item = verify_receipt(fifo, roots=[root])
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "not a regular file"
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@_POSIX
def test_receipt_hardlink_is_not_verified(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"data_label": "REAL", "note": SECRET}), encoding="utf-8")
    link = root / "r.json"
    os.link(outside, link)
    calls: list[Path] = []

    def verifier(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {"valid": True}

    item = verify_receipt(link, roots=[root], verifier=verifier)
    assert calls == []
    assert item.verdict is not Verdict.VERIFIED
    assert SECRET not in item.detail
    assert SECRET in outside.read_text(encoding="utf-8")


@_POSIX
def test_receipt_huge_sparse_file_is_not_read(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "big.json"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    os.truncate(path, 32 * 1024 * 1024)
    reads: list[int] = []
    real_read = evidence_mod.os.read

    def spy(fd: int, size: int) -> bytes:
        block = real_read(fd, size)
        reads.append(len(block))
        return block

    evidence_mod.os.read = spy  # type: ignore[assignment]
    try:
        item = verify_receipt(path, roots=[root], max_bytes=1024)
    finally:
        evidence_mod.os.read = real_read
    assert item.verdict is Verdict.UNVERIFIABLE
    assert "size" in item.detail
    assert sum(reads) == 0


def test_receipt_dotdot_tilde_and_windows_strings_stay_outside(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "nope.json"
    outside.write_text(SECRET, encoding="utf-8")
    for raw in (
        str(root / ".." / "nope.json"),
        "~/nope.json",
        r"C:\Windows\nope.json",
        r"\\server\share\nope.json",
        "a\x00b.json",
    ):
        item = verify_receipt(raw, roots=[root])
        assert item.verdict is Verdict.UNVERIFIABLE
        assert SECRET not in item.detail
        assert SECRET not in item.ref


def test_dev_zero_under_its_own_root_is_not_a_receipt() -> None:
    zero = Path("/dev/zero")
    if not zero.exists():
        pytest.skip("no /dev/zero")
    err_holder: dict[str, Any] = {}

    def run() -> None:
        err_holder["item"] = verify_receipt(zero, roots=[Path("/dev")])

    assert _bounded(run) is None
    item = err_holder["item"]
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.sha256 is None


@_POSIX
def test_telemetry_fifo_append_does_not_block(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    session_id = new_session_id()
    fifo = logs / f"{session_id}.jsonl"
    os.mkfifo(fifo)
    tel = Telemetry(session_id=session_id, directory=logs)

    def emit() -> None:
        tel.emit("app.tick", {"n": 1})

    assert _bounded(emit) is None
    assert tel.dropped >= 1
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@_POSIX
def test_telemetry_hardlink_is_not_appended_or_chmodded(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    victim = tmp_path / "victim"
    victim.write_text(SECRET, encoding="utf-8")
    os.chmod(victim, 0o644)
    session_id = new_session_id()
    os.link(victim, logs / f"{session_id}.jsonl")
    tel = Telemetry(session_id=session_id, directory=logs)
    tel.emit("app.tick", {"n": 1})
    assert victim.read_text(encoding="utf-8") == SECRET
    assert _mode(victim) == 0o644
    assert tel.dropped >= 1


@_POSIX
def test_telemetry_ancestor_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    session_id = new_session_id()
    tel = Telemetry(session_id=session_id, directory=link / "logs")
    tel.emit("app.tick", {"n": 1})
    assert tel.dropped >= 1
    assert list(outside.rglob("*")) == []


@_POSIX
def test_telemetry_tail_does_not_block_on_a_fifo(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    session_id = new_session_id()
    fifo = logs / f"{session_id}.jsonl"
    os.mkfifo(fifo)
    tel = Telemetry(session_id=session_id, directory=logs)
    assert _bounded(lambda: tel.tail(5)) is None


@_POSIX
def test_job_root_ancestor_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    ledger = JobLedger(link / "jobs")
    with pytest.raises((StoreError, OSError)):
        ledger.create(kind="turn", goal="pinball")
    assert list(outside.rglob("*")) == []


@_POSIX
def test_job_create_does_not_overwrite_when_exists_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    fixed = "j-20260101T000000Z-abcdef12"
    path = root / f"{fixed}.json"
    path.write_text("ORIGINAL", encoding="utf-8")
    monkeypatch.setattr(jobs_mod, "new_job_id", lambda **_kwargs: fixed)
    _lie_missing(monkeypatch, path)
    ledger = JobLedger(root)
    with pytest.raises(IllegalTransition):
        ledger.create(kind="turn", goal="pinball")
    assert path.read_text(encoding="utf-8") == "ORIGINAL"


def test_job_ids_reject_traversal_and_windows_strings(tmp_path: Path) -> None:
    ledger = JobLedger(tmp_path / "jobs")
    for bad in ("../x", "/tmp/j", "j-20260101T000000Z-abcdef1", "C:\\jobs\\a", "~\n", "a\x00b"):
        with pytest.raises(ValueError):
            ledger.get(bad)


@_POSIX
def test_session_directory_ancestor_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SessionError, match="symlink"):
        SessionStore(link / "sessions").create(model="fx1")
    assert list(outside.rglob("*")) == []


@_POSIX
def test_session_create_does_not_overwrite_when_exists_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "sessions"
    directory.mkdir()
    fixed = "ab" * 16
    path = directory / f"{fixed}.json"
    path.write_text("ORIGINAL", encoding="utf-8")

    class _UUID:
        hex = fixed

    monkeypatch.setattr(sessions_mod.uuid, "uuid4", lambda: _UUID())
    _lie_missing(monkeypatch, path)
    with pytest.raises(SessionError):
        SessionStore(directory).create(model="fx1")
    assert path.read_text(encoding="utf-8") == "ORIGINAL"


@_POSIX
def test_session_tree_is_private_under_umask_zero(tmp_path: Path) -> None:
    previous = os.umask(0)
    try:
        store = SessionStore(tmp_path / "a" / "sessions")
        record = store.create(model="fx1-lite")
    finally:
        os.umask(previous)
    path = tmp_path / "a" / "sessions" / f"{record['id']}.json"
    assert _mode(tmp_path / "a") == 0o700
    assert _mode(path.parent) == 0o700
    assert _mode(path) == 0o600


def test_session_ids_reject_traversal_null_tilde_and_windows(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    hostile = (
        "../x",
        "..",
        ".",
        "/etc/passwd",
        "a" * 31 + "\x00",
        "~/.ssh/id_rsa",
        r"C:\sessions\a",
        r"\\server\share\a",
        "é" * 16,
        "A" * 32,
        "s-not-an-id",
        "ab" * 16 + "/../../etc/passwd",
    )
    for bad in hostile:
        with pytest.raises(SessionError):
            store.load(bad)


@_POSIX
def test_group_readable_credential_dir_is_tightened(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    os.chmod(cfg, 0o755)
    store = cfg / "credentials.json"
    store_health(store)
    assert _mode(cfg) == 0o700
    os.chmod(cfg, 0o750)
    store_health(store)
    assert _mode(cfg) == 0o700


@_POSIX
def test_credential_hardlink_does_not_change_the_victim(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text(SECRET, encoding="utf-8")
    os.chmod(victim, 0o644)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    os.chmod(cfg, 0o700)
    store = cfg / "credentials.json"
    os.link(victim, store)
    with pytest.raises(ValueError, match="hard link"):
        set_endpoint(
            "fx1",
            "sk-rt3-path-key",
            "https://stored.example/v1/chat/completions",
            path=store,
        )
    assert victim.read_text(encoding="utf-8") == SECRET
    assert _mode(victim) == 0o644


@_POSIX
def test_credential_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    os.chmod(cfg, 0o700)
    store = cfg / "credentials.json"
    os.mkfifo(store)

    def write() -> None:
        set_endpoint(
            "fx1",
            "sk-rt3-path-key",
            "https://stored.example/v1/chat/completions",
            path=store,
        )

    err = _bounded(write)
    assert isinstance(err, ValueError)
    assert stat.S_ISFIFO(store.lstat().st_mode)


@_POSIX
def test_credential_store_symlink_is_not_followed(tmp_path: Path) -> None:
    victim = tmp_path / "victim.json"
    victim.write_text(SECRET, encoding="utf-8")
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    store = cfg / "credentials.json"
    store.symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        set_endpoint(
            "fx1",
            "sk-rt3-path-key",
            "https://stored.example/v1/chat/completions",
            path=store,
        )
    assert victim.read_text(encoding="utf-8") == SECRET


@_POSIX
def test_history_symlink_is_not_loaded_or_written_through(tmp_path: Path) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps([SECRET]), encoding="utf-8")
    link = tmp_path / "history"
    link.symlink_to(secret)
    reader = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()), history_path=link)
    reader.load_history()
    assert SECRET not in "\n".join(reader.history())
    armed = LineReader(
        _caps(),
        ConsoleWriter(_caps(), out=io.StringIO()),
        history_path=link,
        input_fn=lambda _prompt: "hello-history",
    )
    armed.read_line("> ")
    armed.save_history()
    assert secret.read_text(encoding="utf-8") == json.dumps([SECRET])


@_POSIX
def test_history_fifo_load_does_not_block(tmp_path: Path) -> None:
    fifo = tmp_path / "history"
    os.mkfifo(fifo)
    reader = LineReader(_caps(), ConsoleWriter(_caps(), out=io.StringIO()), history_path=fifo)
    assert _bounded(reader.load_history) is None
    assert reader.history() == ()


@_POSIX
def test_run_dir_ancestor_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoreError, match="symlink"):
        _prepare_dir(link / "runs" / "nested")
    assert list(outside.rglob("*")) == []


@_POSIX
def test_out_dir_swap_after_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / SECRET).write_text("nope", encoding="utf-8")
    catalog = sample_catalog()
    cap = catalog.get("research")
    assert cap is not None
    plan = SuperpowerPlan(goal="pinball")
    run_id = new_run_id()
    state = RunState(
        plan=plan,
        ctx=SuperpowerContext(run_dir=run_dir, roots={}),
        result=SuperpowerResult(goal="pinball", plan=plan, run_id=run_id),
        ledger=evidence_mod.EvidenceLedger(),
        run_id=run_id,
        run_dir=run_dir,
    )
    step = PlanStep(
        capability_id="research",
        args={},
        rationale="score",
        depends_on=(),
        evidence_class="SYNTHETIC",
        consequential=False,
        est_runtime_s=1.0,
        step_key="k",
    )
    real_validate = __import__(
        "fx1.interactive.orchestration", fromlist=["validate_args"]
    ).validate_args

    def swapping(capability: object, args: dict[str, Any], roots: object = None) -> dict[str, Any]:
        validated = real_validate(capability, args, roots=roots)
        out = Path(str(args["--out-dir"]))
        if out.exists() and not out.is_symlink():
            held = tmp_path / "held-out"
            os.rename(out, held)
            out.symlink_to(outside, target_is_directory=True)
        return validated

    monkeypatch.setattr("fx1.interactive.orchestration.validate_args", swapping)
    with pytest.raises(ContractViolation) as exc:
        _argv_for(state, step, cap)
    assert exc.value.code == "path_escape"
    assert SECRET not in str(exc.value)
    assert list(outside.iterdir()) == [outside / SECRET]
