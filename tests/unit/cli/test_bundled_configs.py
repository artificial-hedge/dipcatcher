"""Bundled lab configs, doctor failure mode, and harness allowlist."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_fund import __version__
from quant_fund.config import load_config
from quant_fund.pipeline.doctor import doctor
from quant_fund.utils.bundled_configs import bundled_file, resolve_lab_config

ROOT = Path(__file__).resolve().parents[3]


def _env(tmp: Path) -> dict[str, str]:
    home = tmp / "home"
    config = tmp / "cfg"
    home.mkdir()
    config.mkdir()
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "FX1_CONFIG_DIR": str(config),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
    }


def test_cwd_config_wins_over_the_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    local = configs / "research.yaml"
    local.write_text(
        "runtime:\n  mode: research\ndata:\n  root: data\n  source: synthetic\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    found = resolve_lab_config(Path("configs/research.yaml"))
    assert found.resolve() == local.resolve()


def test_missing_relative_config_uses_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_lab_config(Path("configs/research.yaml"))
    bundled = bundled_file("research.yaml")
    assert bundled is not None
    assert resolved == bundled
    assert resolved.is_file()
    loaded = load_config(resolved)
    assert loaded.runtime.mode.value == "research"


def test_absolute_missing_path_is_not_rewritten(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert resolve_lab_config(missing) == missing


def test_doctor_reports_missing_lake_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    info = doctor(str(resolve_lab_config(Path("configs/research.yaml"))))
    assert info["dir_raw"] == "missing"
    assert info["data_manifest"] == "missing"
    assert info["research_receipt"] == "missing"
    assert not (tmp_path / "data").exists()


def test_cli_doctor_empty_cwd_is_honest_and_clean(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "doctor"],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 1, proc.stderr
    assert "data_manifest: missing" in proc.stdout
    assert "dir_raw: missing" in proc.stdout
    assert "Traceback" not in proc.stderr
    assert "FileNotFoundError" not in proc.stderr
    assert not (tmp_path / "data").exists()


def test_cli_doctor_missing_config_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    proc = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "doctor", "--config", str(missing)],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert f"config not found: {missing}" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_quant_version_from_empty_cwd(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "quant_fund.cli.main", "--version"],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == __version__


def test_harness_allows_bundled_configs_and_rejects_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.harness import Harness

    monkeypatch.chdir(tmp_path)
    bundled = bundled_file("base.yaml")
    assert bundled is not None
    store: dict[str, list[str]] = {}

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del timeout_s
        store["argv"] = list(argv)
        return 0, "", ""

    result = Harness(runner=_runner).run("doctor", config=bundled)
    assert result.ok
    assert str(bundled.resolve()) in store["argv"]
    with pytest.raises(ValueError, match="allowlist"):
        Harness(runner=_runner).run("doctor", config=tmp_path / "outside.yaml")
