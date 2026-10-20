"""Issue-time column binding for score-forecasts on the live catalog."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from fx1.harness import Harness
from fx1.interactive.contracts import (
    CapabilityCatalog,
    build_argv,
    contains_shell_metacharacters,
    fetch_catalog,
)
from fx1.interactive.planner import (
    _COLUMN_IGNORED,
    DatasetRef,
    plan_goal,
    validate_plan,
)

_GOAL = "score my volatility forecasts with QLIKE"
_HOSTILE = (
    "--out-dir",
    "x --out /etc",
    "\x1b[31missued",
    "issued\x00at",
    "   ",
    " issued_at",
    "-" + "a" * 128,
    "a" * 129,
)


@pytest.fixture(scope="module")
def live_catalog(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapabilityCatalog]:
    folder = tmp_path_factory.mktemp("available-time")
    previous = _isolate(folder)
    try:
        yield _load_catalog()
    finally:
        _restore(previous)


def test_live_catalog_declares_available_time_col(live_catalog: CapabilityCatalog) -> None:
    cap = live_catalog.get("score-forecasts")
    assert cap is not None
    spec = next(item for item in cap.args if item.name == "--available-time-col")
    assert spec.required is False
    assert spec.type == "str"


def test_issue_time_column_is_bound_and_deterministic(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "var.csv"
    csv_path.write_text("timestamp,issued_at,realized,forecast\n", encoding="utf-8")
    datasets = (DatasetRef(str(csv_path), "csv", "variance", "issued_at"),)
    plan = plan_goal(_GOAL, live_catalog, datasets=datasets, roots=roots)
    again = plan_goal(_GOAL, live_catalog, datasets=datasets, roots=roots)
    assert plan.to_dict() == again.to_dict()
    assert [step.capability_id for step in plan.steps] == ["score-forecasts"]
    step = plan.steps[0]
    assert step.args["--available-time-col"] == "issued_at"
    assert _COLUMN_IGNORED not in plan.notes
    assert validate_plan(plan, live_catalog, max_steps=4, budget_s=None, roots=roots) == []
    cap = live_catalog.get("score-forecasts")
    assert cap is not None
    argv = build_argv(cap, step.args, roots=roots)
    index = argv.index("--available-time-col")
    assert argv[index + 1] == "issued_at"
    assert not any(contains_shell_metacharacters(token) for token in argv)


def test_absent_issue_time_column_omits_the_flag(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "var.csv"
    csv_path.write_text("timestamp,realized,forecast\n", encoding="utf-8")
    plan = plan_goal(
        _GOAL,
        live_catalog,
        datasets=(DatasetRef(str(csv_path), "csv", "variance"),),
        roots=roots,
    )
    assert [step.capability_id for step in plan.steps] == ["score-forecasts"]
    assert "--available-time-col" not in plan.steps[0].args
    assert _COLUMN_IGNORED not in plan.notes


@pytest.mark.parametrize("column", _HOSTILE)
def test_hostile_issue_time_column_never_reaches_argv(
    live_catalog: CapabilityCatalog, tmp_path: Path, column: str
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "var.csv"
    csv_path.write_text("timestamp,realized,forecast\n", encoding="utf-8")
    plan = plan_goal(
        _GOAL,
        live_catalog,
        datasets=(DatasetRef(str(csv_path), "csv", "variance", column),),
        roots=roots,
    )
    assert [step.capability_id for step in plan.steps] == ["score-forecasts"]
    assert "--available-time-col" not in plan.steps[0].args
    assert _COLUMN_IGNORED in plan.notes
    cap = live_catalog.get("score-forecasts")
    assert cap is not None
    argv = build_argv(cap, plan.steps[0].args, roots=roots)
    assert "--available-time-col" not in argv
    assert ("--available-time-col", column) not in list(zip(argv, argv[1:], strict=False))
    if column != "--out-dir":
        assert column not in argv
    assert all("\x00" not in token and "\x1b" not in token for token in argv)
    assert argv.count("--out-dir") == 1


def test_undeclared_flag_is_not_invented(live_catalog: CapabilityCatalog, tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "var.csv"
    csv_path.write_text("timestamp,issued_at,realized,forecast\n", encoding="utf-8")
    catalog = _drop_available_time(live_catalog)
    plan = plan_goal(
        _GOAL,
        catalog,
        datasets=(DatasetRef(str(csv_path), "csv", "variance", "issued_at"),),
        roots=roots,
    )
    assert [step.capability_id for step in plan.steps] == ["score-forecasts"]
    assert "--available-time-col" not in plan.steps[0].args
    assert _COLUMN_IGNORED not in plan.notes
    assert validate_plan(plan, catalog, max_steps=4, budget_s=None, roots=roots) == []


def _drop_available_time(catalog: CapabilityCatalog) -> CapabilityCatalog:
    document = json.loads(json.dumps(catalog.document))
    for entry in document["capabilities"]:
        if entry.get("id") != "score-forecasts":
            continue
        entry["args"] = [arg for arg in entry["args"] if arg.get("name") != "--available-time-col"]
        _drop_allowed(entry)
    return CapabilityCatalog.from_json(document)


def _drop_allowed(entry: dict[str, Any]) -> None:
    allowed = entry.get("allowed_args")
    if isinstance(allowed, list):
        entry["allowed_args"] = [name for name in allowed if name != "--available-time-col"]
        return
    if isinstance(allowed, dict):
        entry["allowed_args"] = {
            key: value for key, value in allowed.items() if key != "--available-time-col"
        }


def _load_catalog() -> CapabilityCatalog:
    harness = Harness()

    def runner(name: str, args: list[str]) -> tuple[int, str]:
        result = harness.run(name, list(args), timeout_s=90)
        return int(result.exit_code), result.stdout

    return fetch_catalog(runner)


def _roots(tmp_path: Path) -> dict[str, list[Path]]:
    paths = {name: tmp_path / name for name in ("config", "input", "output")}
    for path in paths.values():
        path.mkdir()
    return {"config": [paths["config"]], "input": [paths["input"]], "output_dir": [paths["output"]]}


def _isolate(folder: Path) -> Mapping[str, str | None]:
    keys = ("FX1_CONFIG_DIR", "FX1_SESSION_DIR", "NO_COLOR", "TERM", "FORCE_COLOR")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["FX1_CONFIG_DIR"] = str(folder / "fx1-config")
    os.environ["FX1_SESSION_DIR"] = str(folder / "fx1-sessions")
    os.environ["NO_COLOR"] = "1"
    os.environ["TERM"] = "dumb"
    os.environ["FORCE_COLOR"] = "0"
    return previous


def _restore(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        os.environ[key] = value
