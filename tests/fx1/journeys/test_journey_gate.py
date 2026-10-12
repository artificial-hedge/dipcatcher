"""Aggregate P1 gate. Unfrozen suites must be green; the 99/100 floor applies once frozen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fx1.journeys.engine import SCOPE, gate_problems, run_suite
from tests.fx1.journeys.schema import CATEGORIES, load_journeys


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, Path]:
    root = tmp_path_factory.mktemp("journey-gate")
    report_path = root / "report.json"
    return run_suite(report_path=report_path), report_path


def test_aggregate_gate(suite: tuple[object, Path]) -> None:
    report, _path = suite
    problems = gate_problems(report)  # type: ignore[arg-type]
    assert problems == []


def test_report_states_it_is_not_the_five_user_trial(suite: tuple[object, Path]) -> None:
    report, path = suite
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scope"] == SCOPE
    assert "not the moderated five-user trial" in payload["scope"]
    assert payload["total"] == len(load_journeys())
    assert payload["total"] <= 100
    assert set(CATEGORIES).issubset(payload["categories"])
    assert report.scope == SCOPE  # type: ignore[attr-defined]
