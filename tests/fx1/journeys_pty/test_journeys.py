"""Parametrized frozen journeys. Set JOURNEYS_FILTER to run a slice."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fx1.journeys_pty.load import (
    CATALOG_PATH,
    CATEGORIES,
    load_catalog,
    select_journeys,
    suite_sha256,
)
from tests.fx1.journeys_pty.runner import run_journey
from tests.fx1.journeys_pty.textutil import FX1_KEY, redact

_FILTER = os.environ.get("JOURNEYS_FILTER", "")


def _params() -> list[object]:
    rows = select_journeys(load_catalog(), _FILTER)
    chosen: list[object] = []
    for row in rows:
        marks = (pytest.mark.slow,) if row["mode"] == "pty" else ()
        chosen.append(pytest.param(row["id"], id=f"{row['id']}-{row['mode']}", marks=marks))
    return chosen


@pytest.mark.parametrize("journey_id", _params())
def test_journey(journey_id: str, tmp_path: Path) -> None:
    journey = next(row for row in load_catalog() if row["id"] == journey_id)
    result = run_journey(journey, tmp_path)
    assert result["status"] == "passed", result["reason"]


def test_catalog_is_frozen_and_balanced() -> None:
    rows = load_catalog()
    assert [row["id"] for row in rows] == [f"J{number:03d}" for number in range(1, 101)]
    for category in CATEGORIES:
        assert sum(1 for row in rows if row["category"] == category) == 10
    critical = [row for row in rows if row["critical"]]
    assert len(critical) >= 20
    assert all(row["mode"] == "pty" for row in critical)
    assert sum(1 for row in rows if row["mode"] == "pty") >= 12
    assert sum(1 for row in rows if row["executes_lab"]) <= 15
    digest = suite_sha256()
    assert len(digest) == 64
    assert FX1_KEY not in CATALOG_PATH.read_text(encoding="utf-8")


def test_failure_snippets_redact_fixture_keys() -> None:
    assert FX1_KEY not in redact(f"leaked {FX1_KEY} tail")
