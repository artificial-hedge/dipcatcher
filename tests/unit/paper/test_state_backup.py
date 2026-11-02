"""Paper ledger and checkpoint backup round trip."""

from __future__ import annotations

import importlib.util
import json
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from quant_fund.paper.state_backup import PaperBackupError, backup_paper_state, restore_paper_state

_ROOT = Path(__file__).resolve().parents[3]


def _cli():
    path = _ROOT / "scripts" / "paper_state_backup.py"
    spec = importlib.util.spec_from_file_location("paper_state_backup_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path) -> None:
    run = root / "metadata" / "paper" / "run-1"
    run.mkdir(parents=True)
    (run / "broker_state.json").write_text('{"schema":"paper"}\n', encoding="utf-8")
    (run / "meta.json").write_text("{}\n", encoding="utf-8")
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "ckpt.json").write_bytes(b"checkpoint-bytes")
    (root / "notes.txt").write_text("leave-me\n", encoding="utf-8")


def test_round_trip_restores_bytes_and_skips_the_rest(tmp_path: Path) -> None:
    source = tmp_path / "data"
    _tree(source)
    archive = tmp_path / "archive"
    backup_paper_state(source, archive)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["files"]}
    assert "metadata/paper/run-1/broker_state.json" in paths
    assert "checkpoints/ckpt.json" in paths
    assert "notes.txt" not in paths
    shutil.rmtree(source)
    restored = tmp_path / "restored"
    restore_paper_state(archive, restored)
    assert (restored / "metadata" / "paper" / "run-1" / "broker_state.json").read_text(
        encoding="utf-8"
    ) == '{"schema":"paper"}\n'
    assert (restored / "checkpoints" / "ckpt.json").read_bytes() == b"checkpoint-bytes"
    assert not (restored / "notes.txt").exists()


def test_cli_round_trip_and_tamper_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "data"
    _tree(source)
    archive = tmp_path / "archive"
    cli = _cli()
    assert cli.main(["backup", "--source", str(source), "--dest", str(archive)]) == 0
    payload = archive / "payload" / "checkpoints" / "ckpt.json"
    payload.write_bytes(b"tampered")
    restored = tmp_path / "restored"
    assert cli.main(["restore", "--archive", str(archive), "--dest", str(restored)]) == 1
    assert capsys.readouterr().err.strip() == "hash_mismatch"
    assert not restored.exists()


def test_symlink_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "data"
    _tree(source)
    outside = tmp_path / "outside"
    outside.write_text("secret-looking-but-not\n", encoding="utf-8")
    (source / "metadata" / "paper" / "run-1" / "link.json").symlink_to(outside)
    archive = tmp_path / "archive"
    with pytest.raises(PaperBackupError, match="symlink"):
        backup_paper_state(source, archive)
    assert not archive.exists()


def test_divergent_destination_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "data"
    _tree(source)
    archive = tmp_path / "archive"
    backup_paper_state(source, archive)
    dest = tmp_path / "dest"
    existing = dest / "checkpoints" / "ckpt.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"other")
    with pytest.raises(PaperBackupError, match="dest_differs"):
        restore_paper_state(archive, dest)
    assert existing.read_bytes() == b"other"
    assert not (dest / "metadata").exists()
    existing.write_bytes(b"checkpoint-bytes")
    restore_paper_state(archive, dest)
    assert (dest / "metadata" / "paper" / "run-1" / "broker_state.json").is_file()
    assert existing.read_bytes() == b"checkpoint-bytes"


def test_run_directory_and_missing_ledger(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "broker_state.json").write_text("state\n", encoding="utf-8")
    (run / "meta.json").write_text("meta\n", encoding="utf-8")
    archive = tmp_path / "archive"
    backup_paper_state(run, archive)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert {entry["path"] for entry in manifest["files"]} == {"broker_state.json", "meta.json"}
    with pytest.raises(PaperBackupError, match="paper_state_missing"):
        backup_paper_state(tmp_path / "empty", tmp_path / "nope")


def test_backup_skips_dotfiles_and_still_refuses_a_dot_symlink(tmp_path: Path) -> None:
    source = tmp_path / "data"
    _tree(source)
    torn = source / "metadata" / "paper" / "run-1" / ".broker_state.json.torn"
    torn.write_text("partial-not-archived\n", encoding="utf-8")
    archive = tmp_path / "archive"
    backup_paper_state(source, archive)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    names = [Path(entry["path"]).name for entry in manifest["files"]]
    assert "broker_state.json" in names
    assert all(not name.startswith(".") for name in names)
    payload = archive / "payload"
    assert not any(path.name.startswith(".") for path in payload.rglob("*"))
    outside = tmp_path / "outside"
    outside.write_text("leave\n", encoding="utf-8")
    (source / "metadata" / "paper" / "run-1" / ".secret").symlink_to(outside)
    with pytest.raises(PaperBackupError, match="symlink"):
        backup_paper_state(source, tmp_path / "archive-2")


def test_sigkill_during_restore_does_not_publish_a_partial_file(tmp_path: Path) -> None:
    source = tmp_path / "data"
    _tree(source)
    archive = tmp_path / "archive"
    backup_paper_state(source, archive)
    dest = tmp_path / "restored"
    code = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "import quant_fund.paper.state_backup as backup\n"
        "def boom(_fd):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "backup.os.fsync = boom\n"
        "backup.restore_paper_state(Path(sys.argv[1]), Path(sys.argv[2]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(archive), str(dest)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    canonical = dest / "metadata" / "paper" / "run-1" / "broker_state.json"
    assert not canonical.exists()
    restore_paper_state(archive, dest)
    assert canonical.read_text(encoding="utf-8") == '{"schema":"paper"}\n'
    assert (dest / "checkpoints" / "ckpt.json").read_bytes() == b"checkpoint-bytes"


def test_manifest_escape_writes_nothing(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    payload = archive / "payload"
    payload.mkdir(parents=True)
    (payload / "ok.json").write_text("ok\n", encoding="utf-8")
    (archive / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "paper_state_backup.v1",
                "files": [
                    {
                        "path": "../escape.json",
                        "sha256": "ab" * 32,
                        "size": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "dest"
    with pytest.raises(PaperBackupError, match="path"):
        restore_paper_state(archive, dest)
    assert not (tmp_path / "escape.json").exists()
    assert not dest.exists()
