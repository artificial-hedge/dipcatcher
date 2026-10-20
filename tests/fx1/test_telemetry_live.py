"""Live telemetry and diagnostics: secrets, hangs, concurrency, bundle bounds.

Absolute paths that remain in a diagnostics report are ``sys.executable`` and
``shutil.which`` results when they sit outside the home directory. Paths under
``HOME`` are rewritten with a ``~`` prefix in ``collect_diagnostics``,
``format_report``, and zip members. The on-disk telemetry JSONL is not rewritten.
"""

from __future__ import annotations

import base64
import errno
import json
import multiprocessing as mp
import os
import stat
import threading
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fx1.harness import HarnessCommand, HarnessRole
from fx1.interactive import diagnostics as diag_mod
from fx1.interactive.diagnostics import (
    DiagnosticsError,
    collect_diagnostics,
    format_report,
    write_bundle,
)
from fx1.interactive.runctx import new_id
from fx1.interactive.telemetry import SCHEMA_VERSION, Telemetry

NOW = datetime(2026, 10, 12, 2, 45, tzinfo=UTC)
SECRET = "r7-secret-7f3c9a1e-notakey"
BODY = "r7-body-secret-aa91"
HOME_MARK = "r7-home-marker"


@pytest.fixture(autouse=True)
def _dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("FX1_API_KEY", SECRET)


def _ids() -> tuple[str, str]:
    return new_id("s", now=NOW, rand="abc12345"), new_id("r", now=NOW, rand="def67890")


def _open(tmp_path: Path, session_id: str, **overrides: Any) -> Telemetry:
    options: dict[str, Any] = {
        "session_id": session_id,
        "directory": tmp_path / "logs",
        "clock": lambda: NOW,
    }
    options.update(overrides)
    return Telemetry(**options)


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _blob(*parts: bytes | str) -> bytes:
    chunks: list[bytes] = []
    for part in parts:
        chunks.append(part if isinstance(part, bytes) else part.encode())
    return b"\n".join(chunks)


def _assert_absent(haystack: bytes, *needles: str) -> None:
    for needle in needles:
        assert needle.encode() not in haystack
        assert base64.b64encode(needle.encode()) not in haystack


def _join(thread: threading.Thread) -> None:
    thread.join(2)
    assert not thread.is_alive()


def test_events_and_bundle_omit_secrets_and_keep_correlation(tmp_path: Path) -> None:
    session_id, run_id = _ids()
    step = f"{run_id}.001"
    tel = _open(tmp_path, session_id)
    child = tel.bind(run_id=run_id, model="fx1", family="volatility")
    try:
        raise RuntimeError(BODY)
    except RuntimeError as exc:
        child.emit(
            "model.error",
            {
                "step": step,
                "error": str(exc),
                "output": BODY,
                "exception": f"boom {SECRET}",
                "message": SECRET,
                "note": f"echo {SECRET}",
            },
        )
    raw_log = tel.path.read_bytes()
    _assert_absent(raw_log, SECRET, BODY)
    row = tel.tail(1)[0]
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["session_id"] == session_id
    assert row["run_id"] == run_id
    assert row["model"] == "fx1"
    assert row["family"] == "volatility"
    assert row["step"] == step
    assert str(row["error"]).startswith("[omitted:")
    assert SECRET not in row["note"]
    config = tmp_path / "cfg"
    (config / "sessions").mkdir(parents=True)
    (config / "profiles.json").write_text(
        json.dumps({"api_key": SECRET, "note": BODY}), encoding="utf-8"
    )
    (config / "sessions" / "s.json").write_text(SECRET, encoding="utf-8")
    hostile = HarnessCommand(
        name="doctor",
        role=HarnessRole.VERIFICATION,
        argv=["doctor", "--api-key", SECRET],
        timeout_s=30,
        description="probe",
    )

    def endpoints() -> list[dict[str, str]]:
        return [{"model": "fx1", "host": "api.example", "status": "ready", "fingerprint": "abcd"}]

    def boom() -> dict[str, int]:
        raise RuntimeError(SECRET)

    diag = collect_diagnostics(
        config_dir=config,
        session_id=session_id,
        model="fx1",
        endpoint_rows=endpoints,
        job_counts=boom,
        telemetry=tel,
        environ={
            "FX1_API_KEY": SECRET,
            "FX1_CONFIG_DIR": str(config),
            "DIPCATCHER_FAMILY": "volatility",
        },
        which=lambda _name: None,
        clock=lambda: NOW,
    )
    monkeypatch_registry = (hostile,)
    saved = diag_mod.HARNESS_REGISTRY
    diag_mod.HARNESS_REGISTRY = monkeypatch_registry
    try:
        armed = collect_diagnostics(
            config_dir=config,
            session_id=session_id,
            model="fx1",
            telemetry=tel,
            environ={"FX1_API_KEY": SECRET},
            which=lambda _name: None,
            clock=lambda: NOW,
        )
    finally:
        diag_mod.HARNESS_REGISTRY = saved
    report = format_report(diag)
    bundle = tmp_path / "support.zip"
    write_bundle(armed, bundle, telemetry_files=tel.log_files())
    pieces = _blob(json.dumps(diag), json.dumps(armed), report, bundle.read_bytes())
    pieces += b"\n" + b"\n".join(_members(bundle).values())
    _assert_absent(pieces, SECRET, BODY)
    assert session_id.encode() in pieces
    assert run_id.encode() in pieces
    assert step.encode() in pieces
    assert b"fx1" in pieces
    assert b"volatility" in pieces
    assert b"error_type" in pieces
    assert diag["jobs"]["error_type"] == "RuntimeError"


def test_schema_version_cannot_be_overridden(tmp_path: Path) -> None:
    session_id, _run = _ids()
    tel = _open(tmp_path, session_id)
    tel.emit("app.tick", {"schema_version": 99, "note": "ok"})
    row = tel.tail(1)[0]
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["note"] == "ok"


def test_home_paths_are_rewritten_in_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = home / ".fx1"
    session_id, _run = _ids()
    logs = tmp_path / "logs"
    tel = Telemetry(session_id=session_id, directory=logs, clock=lambda: NOW)
    tel.emit("app.path", {"note": str(home / HOME_MARK)})
    diag = collect_diagnostics(
        config_dir=config,
        session_id=session_id,
        model="fx1",
        environ={"FX1_CONFIG_DIR": str(config)},
        which=lambda name: str(home / name),
        clock=lambda: NOW,
    )
    bundle = tmp_path / "home.zip"
    write_bundle(diag, bundle, telemetry_files=tel.log_files())
    raw = bundle.read_bytes() + b"\n" + b"\n".join(_members(bundle).values())
    report = format_report(diag).encode()
    assert str(home).encode() not in raw
    assert str(home).encode() not in report
    assert b"~/" in raw
    assert tel.path.read_bytes().find(str(home).encode()) >= 0


def test_symlink_fifo_device_and_huge_inputs_do_not_escape(tmp_path: Path) -> None:
    session_id, _run = _ids()
    secret_file = tmp_path / "secret.jsonl"
    secret_file.write_text(json.dumps({"note": SECRET}) + "\n", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(secret_file)
    passwd = tmp_path / "passwd.jsonl"
    passwd.symlink_to("/etc/passwd")
    fifo = tmp_path / "fifo.jsonl"
    os.mkfifo(fifo)
    huge = tmp_path / "huge.jsonl"
    with huge.open("wb") as handle:
        handle.truncate(3 * 1024 * 1024)
    blocked = tmp_path / "blocked.jsonl"
    blocked.write_text(json.dumps({"note": SECRET}) + "\n", encoding="utf-8")
    blocked.chmod(0)
    bundle = tmp_path / "bound.zip"
    holder: dict[str, Any] = {}

    def build() -> None:
        try:
            write_bundle(
                {"model": "fx1", "session_id": session_id},
                bundle,
                telemetry_files=[link, passwd, fifo, Path("/dev/zero"), huge, blocked],
            )
            holder["ok"] = True
        except Exception as exc:  # noqa: BLE001 - the assertion is that this returns
            holder["err"] = exc

    thread = threading.Thread(target=build, daemon=True)
    try:
        thread.start()
        _join(thread)
        assert holder.get("err") is None
        assert bundle.is_file()
        assert bundle.stat().st_size < 200_000
        raw = bundle.read_bytes() + b"\n" + b"\n".join(_members(bundle).values())
        _assert_absent(raw, SECRET)
        assert b"root:" not in raw
        assert b"telemetry.unreadable" in raw
    finally:
        blocked.chmod(0o600)


def test_permission_and_disk_full_messages_omit_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(DiagnosticsError) as denied:
        write_bundle({"model": "fx1"}, Path("/etc") / f"{SECRET}.zip")
    assert SECRET not in str(denied.value)
    assert "permission denied" in str(denied.value)

    def full(_path: Path, _data: bytes) -> None:
        raise OSError(
            errno.ENOSPC, "No space left on device", str(tmp_path / SECRET / "bundle.zip")
        )

    monkeypatch.setattr(diag_mod, "_publish_private", full)
    with pytest.raises(DiagnosticsError) as nospace:
        write_bundle({"model": "fx1"}, tmp_path / "fresh.zip")
    assert SECRET not in str(nospace.value)
    assert "no space" in str(nospace.value)
    existing = tmp_path / f"{SECRET}.zip"
    existing.write_bytes(b"already")
    with pytest.raises(FileExistsError) as exists:
        write_bundle({"model": "fx1"}, existing)
    assert SECRET not in str(exists.value)
    assert existing.read_bytes() == b"already"


def _telemetry_worker(index: int, directory: str, session_id: str, max_bytes: int) -> None:
    tel = Telemetry(
        session_id=session_id, directory=Path(directory), max_bytes=max_bytes, max_files=3
    )
    for n in range(12):
        tel.emit("worker.event", {"proc": index, "n": n, "pad": "y" * 40})
    (Path(directory) / f"dropped-{index}.txt").write_text(str(tel.dropped), encoding="utf-8")


def _run_workers(directory: Path, session_id: str, max_bytes: int) -> list[int]:
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_telemetry_worker, args=(index, str(directory), session_id, max_bytes))
        for index in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(15)
        assert proc.exitcode == 0
    dropped: list[int] = []
    for index in range(4):
        text = (directory / f"dropped-{index}.txt").read_text(encoding="utf-8")
        dropped.append(int(text))
    return dropped


def _is_log(path: Path) -> bool:
    if path.name.endswith(".lock") or path.name.startswith("dropped-"):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    return path.suffix == ".jsonl" or ".jsonl." in path.name


def _jsonl_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in directory.iterdir():
        if not _is_log(path):
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    return rows


def test_four_processes_append_without_losing_events(tmp_path: Path) -> None:
    session_id, _run = _ids()
    logs = tmp_path / "logs"
    logs.mkdir()
    dropped = _run_workers(logs, session_id, 5_000_000)
    assert dropped == [0, 0, 0, 0]
    rows = [row for row in _jsonl_rows(logs) if row.get("event") == "worker.event"]
    assert {(row["proc"], row["n"]) for row in rows} == {
        (i, n) for i in range(4) for n in range(12)
    }
    assert {row["schema_version"] for row in rows} == {SCHEMA_VERSION}


def test_four_processes_rotate_without_dropping_or_tearing(tmp_path: Path) -> None:
    session_id, _run = _ids()
    logs = tmp_path / "logs"
    logs.mkdir()
    dropped = _run_workers(logs, session_id, 500)
    assert dropped == [0, 0, 0, 0]
    names = [path.name for path in logs.iterdir() if _is_log(path)]
    assert len(names) <= 3
    rows = _jsonl_rows(logs)
    assert rows
    assert all(row["schema_version"] == SCHEMA_VERSION for row in rows)
