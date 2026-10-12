"""Frozen goal/data relevance suite against the live catalog and planner."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from tests.fx1.relevance.runner import SCHEMA, run_suite, suite_sha256

_KINDS = (
    "supported",
    "paraphrase",
    "ambiguous",
    "missing_data",
    "retired",
    "ineligible",
    "unsafe",
    "unsupported",
)
_ENV = ("FX1_CONFIG_DIR", "FX1_SESSION_DIR", "NO_COLOR", "TERM", "FORCE_COLOR")


@pytest.fixture(scope="module")
def relevance_result(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    folder = tmp_path_factory.mktemp("relevance")
    previous = {key: os.environ.get(key) for key in _ENV}
    os.environ["FX1_CONFIG_DIR"] = str(folder / "fx1-config")
    os.environ["FX1_SESSION_DIR"] = str(folder / "fx1-sessions")
    os.environ["NO_COLOR"] = "1"
    os.environ["TERM"] = "dumb"
    os.environ["FORCE_COLOR"] = "0"
    try:
        yield run_suite(work_dir=folder / "work")
    finally:
        _restore(previous)


def test_relevance_suite_holds_invariants(relevance_result: dict[str, Any]) -> None:
    result = relevance_result
    per_case = _rows(result["per_case"])
    failed = [row for row in per_case if not row["passed"]]
    assert result["schema"] == SCHEMA
    assert int(result["case_count"]) >= 40
    assert result["suite_sha256"] == suite_sha256()
    assert result["invariant_failures"] == []
    assert result["model_refinement_ok"] is True
    assert result["unsafe_refused_rate"] == 1.0
    assert result["ok"] is True
    assert failed == []
    assert result["supported_case_pass_rate"] == 1.0
    assert result["paraphrase_consistency"] == 1.0
    assert max(float(row["elapsed_s"]) for row in per_case) < 20.0


def test_relevance_suite_covers_kinds_and_intent(relevance_result: dict[str, Any]) -> None:
    result = relevance_result
    counts = result["counts_by_kind"]
    assert isinstance(counts, dict)
    for kind in _KINDS:
        assert counts.get(kind, 0) >= 1
    intent = result["intent"]
    assert isinstance(intent, dict)
    assert int(intent["positive"]) >= 30
    assert int(intent["negative"]) >= 30
    assert intent["fp"] == 0
    assert intent["precision"] == 1.0
    assert intent["recall"] is not None
    assert 0.0 <= float(intent["recall"]) <= 1.0
    scope = str(result["scope"])
    assert "OPEN" in scope
    assert "state-of-the-art" in scope


def _rows(value: object) -> list[dict[str, Any]]:
    assert isinstance(value, list)
    return [row for row in value if isinstance(row, dict)]


def _restore(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        os.environ[key] = value
