"""Producer/consumer lockstep for argument binding keys."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fx1.interactive.contracts import CapabilityCatalog, ContractError
from quant_fund.cli.capability_cmds import (
    LabCommand,
    audit_capabilities,
    build_capability_document,
)

_EVIDENCE = "tests/unit/cli/test_capabilities_cmd.py"
_TASKS = {
    "volatility": ["volatility", "qlike"],
    "quantile": ["quantile", "pinball"],
    "probability": ["probability", "brier", "log loss"],
}


def _arg(name: str, *, path_role: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "kind": "option",
        "type": "path" if path_role else "str",
        "required": True,
        "default": None,
        "repeatable": False,
        "help": "Argument.",
    }
    if path_role:
        payload["path_role"] = path_role
    return payload


def _command() -> LabCommand:
    return LabCommand(
        argv=("score-demo",),
        help_text="Score a file.",
        hidden=False,
        args=(_arg("--task"), _arg("--data", path_role="input")),
    )


def _entry() -> dict[str, object]:
    task_hint: dict[str, object] = {
        "choices": ["volatility", "quantile", "probability"],
        "bind": "goal",
        "goal_terms": {key: list(value) for key, value in _TASKS.items()},
    }
    data_hint: dict[str, object] = {"path_role": "input", "bind": "dataset"}
    return {
        "id": "score-demo",
        "version": "1",
        "purpose": "Score an attached forecast file with a proper score.",
        "tasks": ["forecast_scoring"],
        "aliases": ["score demo"],
        "command": {
            "argv": ["score-demo"],
            "role": "evaluation",
            "read_only": True,
            "writes": [],
            "consequential": False,
            "timeout_s": 30,
        },
        "allowed_args": ["--task", "--data"],
        "arg_hints": {"--task": task_hint, "--data": data_hint},
        "inputs": {
            "data_labels": ["file"],
            "requires_dataset": True,
            "availability_timing": "caller-supplied file",
        },
        "outputs": {
            "scores": ["QLIKE"],
            "receipt_schema": "",
            "receipt_flag": [],
            "receipt_stdout_marker": "",
        },
        "dependencies": [],
        "post_checks": [{"kind": "exit_code"}],
        "limits": {"max_runtime_s": 30, "max_output_chars": 1000},
        "qualification": {
            "status": "qualified",
            "evidence": [_EVIDENCE],
            "ruleset": "overlay-v1",
            "audited_on": "2026-10-12",
        },
        "evidence_class": "RESEARCH",
    }


def _problems(entry: dict[str, object]) -> list[str]:
    return audit_capabilities(overlay=[entry], commands=[_command()], discover_repo=True)


def _document(entry: dict[str, object] | None = None) -> dict[str, Any]:
    doc = build_capability_document(
        overlay=[_entry() if entry is None else entry],
        commands=[_command()],
        discover_repo=True,
        executable_only=True,
    )
    return doc


def test_real_score_forecasts_publishes_bindings_the_client_parses() -> None:
    assert audit_capabilities(include_families=False) == []
    doc = build_capability_document(executable_only=True)
    catalog = CapabilityCatalog.from_json(doc)
    cap = catalog.get("score-forecasts")
    assert cap is not None and cap.executable
    by_name = {arg.name: arg for arg in cap.args}
    assert by_name["--data"].bind == "dataset"
    assert by_name["--data"].path_role == "input"
    task = by_name["--task"]
    assert task.bind == "goal"
    assert set(task.goal_terms) <= set(task.choices or ())
    assert "qlike" in task.goal_terms["volatility"]
    assert "ece" in task.goal_terms["probability"]


def test_valid_bind_round_trips_and_invalid_kinds_fail_both_sides() -> None:
    assert _problems(_entry()) == []
    catalog = CapabilityCatalog.from_json(_document())
    cap = catalog.get("score-demo")
    assert cap is not None and cap.executable
    by_name = {arg.name: arg for arg in cap.args}
    assert by_name["--data"].bind == "dataset"
    assert by_name["--task"].goal_terms["quantile"] == ("quantile", "pinball")

    bad_bind = json.loads(json.dumps(_entry()))
    bad_bind["arg_hints"]["--task"]["bind"] = "env"
    assert any("bind is invalid" in problem for problem in _problems(bad_bind))
    published = _document()
    task = next(arg for arg in published["capabilities"][0]["args"] if arg["name"] == "--task")
    task["bind"] = "env"
    with pytest.raises(ContractError, match="bind is invalid"):
        CapabilityCatalog.from_json(published)


def test_goal_terms_must_stay_inside_choices_and_size_bounds() -> None:
    widened = json.loads(json.dumps(_entry()))
    widened["arg_hints"]["--task"]["goal_terms"]["nope"] = ["qlike"]
    assert any("subset of choices" in problem for problem in _problems(widened))

    shouted = json.loads(json.dumps(_entry()))
    shouted["arg_hints"]["--task"]["goal_terms"]["volatility"] = ["QLIKE"]
    assert any("invalid term" in problem for problem in _problems(shouted))

    huge = json.loads(json.dumps(_entry()))
    huge["arg_hints"]["--task"]["goal_terms"]["volatility"] = [f"t{index}" for index in range(13)]
    assert any("exceeds 12 terms" in problem for problem in _problems(huge))

    published = _document()
    task = next(arg for arg in published["capabilities"][0]["args"] if arg["name"] == "--task")
    task["goal_terms"]["volatility"] = [f"t{index}" for index in range(13)]
    with pytest.raises(ContractError, match="exceeds 12 terms"):
        CapabilityCatalog.from_json(published)

    many = json.loads(json.dumps(_entry()))
    choices = [f"c{index}" for index in range(9)]
    many["arg_hints"]["--task"]["choices"] = choices
    many["arg_hints"]["--task"]["goal_terms"] = {choice: ["term"] for choice in choices}
    assert any("exceeds 8 choices" in problem for problem in _problems(many))


def test_unknown_hint_keys_and_misplaced_bindings_are_rejected() -> None:
    unknown = json.loads(json.dumps(_entry()))
    unknown["arg_hints"]["--data"]["from_env"] = "HOME"
    assert any("unknown arg hint key from_env" in problem for problem in _problems(unknown))

    misplaced = json.loads(json.dumps(_entry()))
    misplaced["arg_hints"]["--data"]["goal_terms"] = {"volatility": ["qlike"]}
    assert any("only valid for bind goal" in problem for problem in _problems(misplaced))

    not_path = json.loads(json.dumps(_entry()))
    not_path["arg_hints"]["--task"] = {"bind": "dataset"}
    assert any("input path" in problem for problem in _problems(not_path))

    published = _document()
    data = next(arg for arg in published["capabilities"][0]["args"] if arg["name"] == "--data")
    data["from_env"] = "HOME"
    with pytest.raises(ContractError, match="unknown key from_env"):
        CapabilityCatalog.from_json(published)
