"""Family-surface exclusion codes survive onto the plan without changing describe()."""

from __future__ import annotations

from fx1.interactive.superpower import plan_goal
from tests.fx1.test_superpower_orchestration import _family, sample_catalog


def test_retired_and_no_runner_codes_are_kept() -> None:
    rows = [
        {"id": "volatility", "reason": "retired family", "code": "retired"},
        {"id": "hmm_likelihood", "reason": "no runner", "code": "no_runner"},
    ]
    plan = plan_goal(
        "run the full scorecard",
        catalog=sample_catalog(),
        family_selector=_family([], rows),
    )
    assert plan.family_exclusions == [
        {"id": "volatility", "reason": "retired family", "code": "retired"},
        {"id": "hmm_likelihood", "reason": "no runner", "code": "no_runner"},
    ]
    text = plan.describe()
    assert "  excluded volatility: retired family" in text
    assert "  excluded hmm_likelihood: no runner" in text
    assert "retired:" not in text
    assert "no_runner" not in text


def test_missing_code_is_an_empty_string_and_describe_matches_reason_only() -> None:
    plan = plan_goal(
        "run the full scorecard",
        catalog=sample_catalog(),
        family_selector=_family([], [{"id": "volatility", "reason": "not executable"}]),
    )
    assert plan.family_exclusions == [{"id": "volatility", "reason": "not executable", "code": ""}]
    assert "  excluded volatility: not executable" in plan.describe()
