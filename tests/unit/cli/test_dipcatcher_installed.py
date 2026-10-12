"""Locks for dipcatcher when the process cwd is not the checkout.

Fast tests use the editable install. The slow test builds a wheel, installs it
into a throwaway venv, and re-runs the smoke driver.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER_PATH = ROOT / "scripts" / "dipcatcher_install_smoke.py"


def _load_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dipcatcher_install_smoke", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DRIVER = _load_driver()


def _env(tmp: Path) -> dict[str, str]:
    home = tmp / "home"
    config = tmp / "cfg"
    home.mkdir(exist_ok=True)
    config.mkdir(exist_ok=True)
    (config / "sessions").mkdir(exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "FX1_CONFIG_DIR": str(config),
        "FX1_SESSION_DIR": str(config / "sessions"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
    }


def _run(tmp: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    empty = tmp / "empty"
    empty.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", *args],
        cwd=empty,
        env=_env(tmp),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_console_script_targets_are_importable() -> None:
    scripts = DRIVER.layout_gaps(ROOT)["scripts"]
    assert set(scripts) == {"fx1", "quant", "dipcatcher", "fxi", "verify-ledger", "mc-engine"}
    for target in scripts.values():
        module_name, attr = str(target).split(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr))


def test_version_from_empty_cwd(tmp_path: Path) -> None:
    from quant_fund import __version__

    result = _run(tmp_path, ["--version"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__
    assert "Traceback" not in result.stderr
    assert list((tmp_path / "empty").iterdir()) == []


def test_help_and_version_skip_heavy_imports(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER.launcher_import_probe_source()],
        cwd=empty,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    help_row = next(item for item in rows if item["args"] == ["--help"])
    version_row = next(item for item in rows if item["args"] == ["--version"])
    assert help_row["exit"] == 0
    assert help_row["heavy"] == []
    assert help_row["interactive"] is False
    assert help_row["stdout_has_usage"] is True
    assert version_row["exit"] == 0
    assert version_row["heavy"] == []
    assert version_row["interactive"] is False
    assert version_row["stdout"]
    assert "Usage:" not in version_row["stdout"]
    assert list(empty.iterdir()) == []


def test_bare_and_chat_from_empty_cwd(tmp_path: Path) -> None:
    bare = _run(tmp_path, [])
    assert bare.returncode == 0
    assert "Start a conversation: dipcatcher" in bare.stdout
    assert "Usage:" in bare.stdout
    assert "requires an interactive terminal" not in bare.stderr
    chat = _run(tmp_path, ["chat"])
    assert chat.returncode == 2
    assert "requires an interactive terminal" in chat.stderr
    assert list((tmp_path / "empty").iterdir()) == []


def test_wheel_force_include_ships_config_closure() -> None:
    gaps = DRIVER.layout_gaps(ROOT)
    force = gaps["force_include"]
    for name in DRIVER.LAYOUT_FILES:
        leaf = name.split("/", 1)[1]
        assert force[name] == f"quant_fund/_bundled/configs/{leaf}"
    shipped = " ".join(force)
    assert "arch_boundaries.toml" not in shipped
    assert "corpus_epoch" not in shipped
    assert gaps["in_wheel"] == list(DRIVER.PACKAGED_FILES)
    assert gaps["sdist_includes_configs"] is True
    assert gaps["packages"] == list(DRIVER.WHEEL_PACKAGES)
    for row in DRIVER.source_anchors(ROOT):
        assert row["present"] is True


def test_importlib_resources_live_in_wheel_packages() -> None:
    rows = DRIVER.importlib_resource_targets(ROOT)
    assert rows
    for row in rows:
        assert row["exists"] is True
        assert row["in_wheel_config"] is True


@pytest.mark.slow
def test_installed_wheel_smoke(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is not available")
    report_path = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--report", str(report_path)],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode == 2:
        pytest.skip(proc.stdout[-500:] or proc.stderr[-500:])
    assert proc.returncode == 0, proc.stderr[-2000:] + "\n" + proc.stdout[-2000:]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["shadows_checkout"] is True
    assert report["empty_cwd_untouched"] is True
    assert report["defects"] == []
    assert not str(report["installed"]["fx1"]).startswith(str(ROOT / "src"))
    by_name = {item["name"]: item for item in report["probes"]}
    assert by_name["quant --version"]["exit_code"] == 0
    assert by_name["dipcatcher --help"]["exit_code"] == 0
    assert by_name["dipcatcher chat"]["exit_code"] == 2
    assert by_name["dipcatcher"]["exit_code"] == 0
    assert by_name["families"]["exit_code"] == 0
    assert by_name["verify-receipt"]["exit_code"] == 0
    assert by_name["wheel ships bundled configs"]["matched"] is True
    doctor = by_name["python -m dipcatcher_cli doctor"]
    assert doctor["exit_code"] == 1
    assert doctor["matched"] is True
    assert "Traceback" not in doctor["stderr_head"]
