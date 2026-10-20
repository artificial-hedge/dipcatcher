"""fx-1 coverage floor is pinned in pyproject and wired into Make/CI."""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _fail_under() -> int:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = data["tool"]["fx1"]["coverage"]["fail_under"]
    assert isinstance(floor, int)
    return floor


def test_fx1_coverage_floor_matches_make_and_proofcore() -> None:
    floor = _fail_under()
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    proofcore = (_ROOT / ".github/workflows/proofcore.yml").read_text(encoding="utf-8")
    assert f"--cov-fail-under={floor}" in makefile
    assert f"--cov-fail-under={floor}" in proofcore


def test_fx1_coverage_floor_is_measured_minus_one() -> None:
    """Documented measurement 91.19% (proofcore fx1-coverage, 2026-10-15)."""
    floor = _fail_under()
    measured = 91.19
    assert floor == int(measured - 1)
