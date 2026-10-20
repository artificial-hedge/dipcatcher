"""The fx1 contract client must parse what the lab command really prints.

``test_contracts.py`` pins the client against fixtures; this test pins the two
halves together so a producer/consumer drift (for example ``"default": null``
on an optional option) cannot ship unnoticed.
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 — fixed argv, the lab command is the system under test
import sys
from pathlib import Path

import pytest

from fx1.interactive.contracts import CapabilityCatalog, ContractError

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def live_document() -> dict[str, object]:
    env = {**os.environ, "PYTHONPATH": str(_ROOT / "src"), "NO_COLOR": "1"}
    proc = subprocess.run(  # noqa: S603 — argv is a fixed local list
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
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    document: dict[str, object] = json.loads(proc.stdout)
    return document


def test_client_parses_the_live_executable_capability_contract(
    live_document: dict[str, object],
) -> None:
    catalog = CapabilityCatalog.from_json(live_document)
    assert {"score-forecasts", "vol-bench", "verify-receipt", "research"} <= set(
        catalog.capabilities
    )
    assert all(cap.executable for cap in catalog.capabilities.values())


def test_optional_options_without_a_default_parse_as_having_none(
    live_document: dict[str, object],
) -> None:
    catalog = CapabilityCatalog.from_json(live_document)
    cap = catalog.get("score-forecasts")
    assert cap is not None
    by_name = {arg.name: arg for arg in cap.args}
    assert by_name["--forecast-col"].has_default is False
    assert by_name["--time-col"].has_default is True
    assert by_name["--data"].required is True


def _live_arg(document: dict[str, object], cap_id: str, name: str) -> dict[str, object]:
    caps = document["capabilities"]
    assert isinstance(caps, list)
    for cap in caps:
        assert isinstance(cap, dict)
        if cap.get("id") != cap_id:
            continue
        args = cap["args"]
        assert isinstance(args, list)
        for arg in args:
            assert isinstance(arg, dict)
            if arg.get("name") == name:
                return arg
    raise AssertionError(f"{cap_id} {name}")


def test_live_score_forecasts_binding_round_trips(live_document: dict[str, object]) -> None:
    catalog = CapabilityCatalog.from_json(live_document)
    cap = catalog.get("score-forecasts")
    assert cap is not None
    by_name = {arg.name: arg for arg in cap.args}
    assert by_name["--data"].bind == "dataset"
    assert not by_name["--data"].goal_terms
    task = by_name["--task"]
    assert task.bind == "goal"
    assert set(task.goal_terms) == {"volatility", "quantile", "probability"}
    assert "qlike" in task.goal_terms["volatility"]
    assert "pinball" in task.goal_terms["quantile"]
    assert "log loss" in task.goal_terms["probability"]
    assert by_name["--available-time-col"].bind is None
    assert by_name["--out-dir"].bind is None


def test_live_catalog_rejects_unknown_bind_and_argument_keys(
    live_document: dict[str, object],
) -> None:
    shell = json.loads(json.dumps(live_document))
    _live_arg(shell, "score-forecasts", "--task")["bind"] = "shell"
    with pytest.raises(ContractError, match="bind is invalid"):
        CapabilityCatalog.from_json(shell)

    extra = json.loads(json.dumps(live_document))
    _live_arg(extra, "score-forecasts", "--data")["from_env"] = "HOME"
    with pytest.raises(ContractError, match="unknown key"):
        CapabilityCatalog.from_json(extra)

    outside = json.loads(json.dumps(live_document))
    terms = _live_arg(outside, "score-forecasts", "--task")["goal_terms"]
    assert isinstance(terms, dict)
    terms["nope"] = ["qlike"]
    with pytest.raises(ContractError, match="subset of choices"):
        CapabilityCatalog.from_json(outside)

    shouted = json.loads(json.dumps(live_document))
    shouted_terms = _live_arg(shouted, "score-forecasts", "--task")["goal_terms"]
    assert isinstance(shouted_terms, dict)
    shouted_terms["volatility"] = ["QLIKE"]
    with pytest.raises(ContractError, match="invalid term"):
        CapabilityCatalog.from_json(shouted)
