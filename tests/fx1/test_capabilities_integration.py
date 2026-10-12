"""Live catalog: the lab producer and the capability client agree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fx1.harness import HARNESS_REGISTRY
from fx1.interactive.contracts import Capability, CapabilityCatalog, build_argv, validate_args
from fx1.interactive.planner import plan_goal
from quant_fund.cli.capability_overlay import qualified_ids

_GOALS = (
    ("volatility forecast QLIKE scoring", "vol-bench"),
    ("calibration/PIT coverage", "calibration-eval"),
    ("rank IC", "rankic"),
    ("serial dependence", "serial-watch"),
    ("detector power", "lane-power"),
    ("selective conformal/volatility family", "research-only"),
    ("receipt verification", "verify-receipt"),
    ("health check", "doctor"),
)


@pytest.fixture(scope="module")
def live_catalog() -> CapabilityCatalog:
    env = {"PYTHONPATH": "src", "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"}
    completed = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "capabilities", "list", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return CapabilityCatalog.from_json(completed.stdout)


def test_live_catalog_parses_and_matches_qualified_harness_ids(
    live_catalog: CapabilityCatalog,
) -> None:
    registry = {command.name for command in HARNESS_REGISTRY}
    qualified = set(qualified_ids())
    expected = {
        cap.id
        for cap in live_catalog.capabilities.values()
        if cap.id in qualified and cap.argv and (cap.id in registry or cap.argv[0] in registry)
    }
    assert {cap.id for cap in live_catalog.executable()} == expected
    assert "verify-all" not in expected
    assert live_catalog.get("verify-all") is not None
    assert live_catalog.get("verify-all").executable is False  # type: ignore[union-attr]
    research = live_catalog.get("research")
    selective = live_catalog.get("research-only")
    assert research is not None and selective is not None
    assert research.id != selective.id
    assert research.argv == selective.argv == ("research",)
    only = next(arg for arg in selective.args if arg.name == "--only")
    assert only.choices
    assert only.required is True


def test_executable_rows_round_trip_argv(live_catalog: CapabilityCatalog, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    for cap in live_catalog.executable():
        validated = validate_args(cap, _required_values(cap, roots), roots=roots)
        argv = build_argv(cap, validated, roots=roots)
        assert list(argv[: len(cap.argv)]) == list(cap.argv)


@pytest.mark.parametrize(("goal", "expected"), _GOALS)
def test_live_planner_selects_representative_goals(
    live_catalog: CapabilityCatalog, tmp_path: Path, goal: str, expected: str
) -> None:
    plan = plan_goal(goal, live_catalog, roots=_roots(tmp_path))
    assert plan.steps
    assert plan.steps[0].capability_id == expected
    assert plan.steps[0].step_key
    assert expected in plan.steps[0].step_key or plan.steps[0].capability_id == expected


def test_live_planner_explains_an_empty_training_goal(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    plan = plan_goal("train a transformer on my data", live_catalog, roots=_roots(tmp_path))
    assert plan.steps == []
    text = plan.describe()
    assert "No suitable capability" in text
    assert plan.exclusions


def _roots(tmp_path: Path) -> dict[str, list[Path]]:
    for name in ("input", "output", "config"):
        (tmp_path / name).mkdir(exist_ok=True)
    return {
        "config": [tmp_path / "config"],
        "input": [tmp_path / "input"],
        "output_dir": [tmp_path / "output"],
    }


def _required_values(cap: Capability, roots: dict[str, list[Path]]) -> dict[str, object]:
    supplied: dict[str, object] = {}
    for spec in cap.args:
        if spec.has_default or not spec.required:
            continue
        if spec.type == "path":
            role = spec.path_role if spec.path_role in roots else "input"
            supplied[spec.name] = str(roots[role][0])
            continue
        if spec.choices:
            supplied[spec.name] = [spec.choices[0]] if spec.type == "csv" else spec.choices[0]
    return supplied
