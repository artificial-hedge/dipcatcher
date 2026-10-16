"""Windows pin ceremony runs the Makefile order without exposing key material."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "owner_pin_ceremony.ps1"
POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None

pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="requires Windows PowerShell")


def _invoke(
    tmp_path: Path,
    *,
    fail_corpus: str = "",
    fail_command: str = "",
    key_file: Path | None = None,
    anchor_checkpoint: bool = False,
    fake_common_git_dir: Path | None = None,
):
    shim = tmp_path / "uv.cmd"
    shim.write_bytes(
        b'@echo off\r\n>>"%FAKE_UV_LOG%" echo %*\r\n'
        b'if "%FAKE_UV_FAIL_CORPUS%"=="%6" exit /b 23\r\n'
        b'if "%FAKE_UV_FAIL_COMMAND%"=="%4" exit /b 29\r\nexit /b 0\r\n'
    )
    log = tmp_path / "uv.log"
    if fake_common_git_dir is not None:
        (tmp_path / "git.cmd").write_bytes(b'@echo off\r\necho %FAKE_GIT_COMMON_DIR%\r\n')
    key = key_file or tmp_path / "owner-key.txt"
    if key_file is None:
        key.write_text("private-key-material-never-print", encoding="ascii")
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    env["FAKE_UV_LOG"] = str(log)
    env["FAKE_UV_FAIL_CORPUS"] = fail_corpus or "__no_failure__"
    env["FAKE_UV_FAIL_COMMAND"] = fail_command or "__no_failure__"
    if fake_common_git_dir is not None:
        env["FAKE_GIT_COMMON_DIR"] = str(fake_common_git_dir)
    command = [
        str(POWERSHELL),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-KeyFile",
        str(key),
    ]
    if anchor_checkpoint:
        command.append("-AnchorCheckpoint")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result, log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _makefile_specs() -> list[tuple[str, str | None]]:
    recipe = (
        (ROOT / "Makefile")
        .read_text(encoding="utf-8")
        .split("\nstamp-epochs:", 1)[1]
        .split("\nsign-pins:", 1)[0]
    )
    specs: list[tuple[str, str | None]] = []
    for line in recipe.splitlines():
        match = re.search(
            r"uv run dipcatcher corpus-epoch --corpus-dir ([\w./-]+)"
            r"(?: --glob '([^']+)')? --out-dir ([\w./-]+)"
            r" --heads-pin quality/epoch_heads.json",
            line,
        )
        if match:
            assert match.group(1) == match.group(3)
            specs.append((match.group(1), match.group(2)))
    loop = re.search(r"for spec in ([\w.\- ]+); do", recipe)
    assert loop is not None
    specs.extend((directory, "*") for directory in loop.group(1).split())
    return specs


def test_windows_ceremony_matches_makefile_sequence(tmp_path: Path) -> None:
    result, commands = _invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert commands[0].startswith("run --frozen dipcatcher quorum-check-key --key-file ")
    assert commands[1] == "run --frozen dipcatcher crown-jewels --write"

    specs = _makefile_specs()
    stamp_commands = commands[2 : 2 + len(specs)]
    assert len(stamp_commands) == len(specs)
    for command, (directory, pattern) in zip(stamp_commands, specs, strict=True):
        parts = command.replace('"', "").split()
        expected = ["run", "--frozen", "dipcatcher", "corpus-epoch", "--corpus-dir", directory]
        if pattern is not None:
            expected.extend(("--glob", pattern))
        expected.extend(("--out-dir", directory, "--heads-pin", "quality/epoch_heads.json"))
        assert parts == expected

    tail = commands[2 + len(specs) :]
    assert len(tail) == 7
    assert tail[0].startswith("run --frozen dipcatcher quorum-sign --key-file ")
    assert tail[1] == "run --frozen dipcatcher anchor-timestamp --file quality/epoch_heads.json"
    assert tail[2] == "run --frozen dipcatcher anchor-timestamp --file quality/crown_jewels.json"
    assert tail[3].startswith("run --frozen dipcatcher checkpoint --key-file ")
    assert tail[4:] == [
        "run --frozen dipcatcher crown-jewels --check",
        "run --frozen dipcatcher verify-repo",
        "run --frozen dipcatcher checkpoint-chain",
    ]
    assert "private-key-material-never-print" not in result.stdout + result.stderr


def test_windows_ceremony_stops_at_failed_stamp(tmp_path: Path) -> None:
    result, commands = _invoke(tmp_path, fail_corpus="verifier")
    assert result.returncode != 0
    assert len(commands) == 4  # preflight, crown jewels, receipts, then verifier
    assert "Stamp verifier failed with exit code 23" in result.stderr
    assert "private-key-material-never-print" not in result.stdout + result.stderr


def test_windows_ceremony_stops_before_pins_on_invalid_signer(tmp_path: Path) -> None:
    result, commands = _invoke(tmp_path, fail_command="quorum-check-key")
    assert result.returncode != 0
    assert len(commands) == 1
    assert commands[0].startswith("run --frozen dipcatcher quorum-check-key --key-file ")
    assert "Validate quorum signer failed with exit code 29" in result.stderr
    assert "private-key-material-never-print" not in result.stdout + result.stderr


def test_windows_ceremony_stops_before_anchor_on_signing_failure(tmp_path: Path) -> None:
    result, commands = _invoke(tmp_path, fail_command="quorum-sign")
    assert result.returncode != 0
    assert commands[-1].startswith("run --frozen dipcatcher quorum-sign --key-file ")
    assert all("anchor-timestamp" not in command for command in commands)
    assert "Quorum-sign pins failed with exit code 29" in result.stderr


def test_windows_ceremony_can_anchor_checkpoint(tmp_path: Path) -> None:
    result, commands = _invoke(tmp_path, anchor_checkpoint=True)
    assert result.returncode == 0, result.stderr
    assert any(
        command.startswith("run --frozen dipcatcher checkpoint --key-file ")
        and command.endswith(" --anchor")
        for command in commands
    )


def test_windows_ceremony_rejects_repo_key_file(tmp_path: Path) -> None:
    result, commands = _invoke(tmp_path, key_file=ROOT / "quality" / "gate_quorum.json")
    assert result.returncode != 0
    assert "KeyFile must be kept outside the repository" in result.stderr
    assert commands == []


def test_windows_ceremony_rejects_key_in_primary_checkout(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    (primary / ".git").mkdir(parents=True)
    key = primary / "owner-key.txt"
    key.write_text("private-key-material-never-print", encoding="ascii")
    result, commands = _invoke(
        tmp_path, key_file=key, fake_common_git_dir=primary / ".git"
    )
    assert result.returncode != 0
    assert "KeyFile must be kept outside the repository" in result.stderr
    assert commands == []
    assert "private-key-material-never-print" not in result.stdout + result.stderr


def test_windows_ceremony_rejects_symlinked_key_parent(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    key = actual / "owner-key.txt"
    key.write_text("private-key-material-never-print", encoding="ascii")
    link = tmp_path / "linked"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        env = os.environ.copy()
        env["CEREMONY_TEST_LINK"] = str(link)
        env["CEREMONY_TEST_TARGET"] = str(actual)
        junction = subprocess.run(
            [
                str(POWERSHELL),
                "-NoProfile",
                "-Command",
                "New-Item -ItemType Junction -Path $env:CEREMONY_TEST_LINK "
                "-Target $env:CEREMONY_TEST_TARGET | Out-Null",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if junction.returncode != 0:
            pytest.skip(f"directory reparse point unavailable: {junction.stderr}")
    result, commands = _invoke(tmp_path, key_file=link / key.name)
    assert result.returncode != 0
    assert "KeyFile path must not traverse a symlink or reparse point" in result.stderr
    assert commands == []
