"""Local mirror of the CI package job: wheel plus installed CLI --help."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from quant_fund.paper.trading_session import module_network_imports
from quant_fund.simtest.release_dry_run import run_release_dry_run

pytestmark = pytest.mark.release_dry_run


def test_release_module_has_no_network_imports() -> None:
    source = Path("src/quant_fund/simtest/release_dry_run.py").read_text(encoding="utf-8")
    assert module_network_imports(source) == []


def test_built_wheel_cli_help(tmp_path: Path) -> None:
    report = run_release_dry_run(tmp_path)
    assert report["wheel"].endswith(".whl")
    assert report["sdist"].endswith(".tar.gz")
    for name in ("dipcatcher", "fx1"):
        entry = report["entrypoints"][name]
        assert entry["exit_code"] == 0
        assert entry["help_bytes"] > 0
    probe = subprocess.run(
        [sys.executable, "-c", "import quant_fund; print(quant_fund.__file__)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "/src/quant_fund/" in probe.stdout.replace("\\", "/")
