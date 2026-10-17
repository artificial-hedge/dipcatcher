"""Read-only ``research-families`` catalog command."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from fx1.interactive.families import FamilyCatalog, goal_lookup_hashes
from quant_fund.cli.main import app
from quant_fund.research.catalog import registry
from quant_fund.research.catalog.optional_classification import justified_families
from quant_fund.research.catalog.registry import (
    BENCHMARK_CATALOG_VERSION,
    BENCHMARK_FAMILY_ORDER,
    FORBIDDEN_RESEARCH_METRIC_KEYS,
    LIVE_OPTIONAL_BENCHMARK_FAMILIES,
    REQUIRED_BENCHMARK_FAMILIES,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    RETIRED_BENCHMARK_FAMILIES,
)

_NOTE = (
    "the research runner has no per-family selector; automatic research is excluded "
    "when a goal names a family, while explicit research runs the configured scorecard."
)

_RUNNER = CliRunner()


def test_research_families_json_lists_the_catalog() -> None:
    result = _RUNNER.invoke(app, ["research-families"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 2
    assert payload["catalog_version"] == BENCHMARK_CATALOG_VERSION
    assert payload["receipt_schema_version"] == RESEARCH_RECEIPT_SCHEMA_VERSION
    assert payload["required"] == sorted(REQUIRED_BENCHMARK_FAMILIES)
    assert payload["live_optional"] == sorted(justified_families())
    assert len(payload["live_optional"]) == 6
    assert set(payload["live_optional"]).isdisjoint(RETIRED_BENCHMARK_FAMILIES)
    assert payload["live_optional_total"] == len(LIVE_OPTIONAL_BENCHMARK_FAMILIES) == 2693
    assert payload["exploratory_count"] == 2687
    assert "retired_sample" not in payload
    assert "lookup" not in payload
    assert payload["retired_count"] == len(RETIRED_BENCHMARK_FAMILIES)
    names = set(payload["required"]) | set(payload["live_optional"])
    assert names.isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)
    assert payload["order"] == list(BENCHMARK_FAMILY_ORDER)
    assert payload["forbidden_metric_keys"] == sorted(FORBIDDEN_RESEARCH_METRIC_KEYS)
    assert payload["runner"] == {
        "command": "research",
        "selection": "config",
        "execution_constraint": False,
        "note": _NOTE,
    }
    assert payload["runner"]["execution_constraint"] is False
    planner_catalog = FamilyCatalog.from_json(result.output)
    assert planner_catalog.eligible() == names
    assert planner_catalog.exploratory_count == 2687
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", payload["generated_at"])
    assert result.output == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_research_families_text_is_a_short_table() -> None:
    result = _RUNNER.invoke(app, ["research-families", "--text"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("benchmark family catalog v")
    assert f"required ({len(REQUIRED_BENCHMARK_FAMILIES)}):" in result.output
    assert "required (" in result.output
    assert "planner-selectable optional: 6" in result.output
    assert "exploratory optional: 2687" in result.output
    assert f"live optional total: {len(LIVE_OPTIONAL_BENCHMARK_FAMILIES)}" in result.output
    assert f"retired: {len(RETIRED_BENCHMARK_FAMILIES)}" in result.output
    assert "automatic research is excluded when a goal names a family" in result.output


def test_goal_hash_lookup_returns_only_matched_omitted_names() -> None:
    goal = "benchmark a1 homotopy"
    hashes = goal_lookup_hashes(goal)
    result = _RUNNER.invoke(app, ["research-families", "--lookup-hashes", hashes])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["lookup"]["omitted_matches"] == ["a1_homotopy"]
    catalog = FamilyCatalog.from_json(result.output, goal=goal)
    assert catalog.omitted_matches == frozenset({"a1_homotopy"})
    assert catalog.covers_goal(goal)
    assert not catalog.covers_goal("benchmark a1 homotopy again")


def test_goal_hash_lookup_rejects_noncanonical_or_oversize_request() -> None:
    digest = "0" * 64
    for bad in ("not-a-hash", f"{digest},{digest}", ",".join([digest] * 301)):
        result = _RUNNER.invoke(app, ["research-families", "--lookup-hashes", bad])
        assert result.exit_code != 0
    result = _RUNNER.invoke(app, ["research-families", "--text", "--lookup-hashes", digest])
    assert result.exit_code != 0


@pytest.mark.parametrize("invalid_name", ["BadFamily", "too_many_under_score_parts_here"])
def test_goal_hash_lookup_refuses_unrepresentable_retired_name(
    monkeypatch: pytest.MonkeyPatch, invalid_name: str
) -> None:
    monkeypatch.setattr(
        registry,
        "RETIRED_BENCHMARK_FAMILIES",
        {**registry.RETIRED_BENCHMARK_FAMILIES, invalid_name: {}},
    )
    result = _RUNNER.invoke(
        app, ["research-families", "--lookup-hashes", goal_lookup_hashes("research benchmark")]
    )
    assert result.exit_code != 0
    assert "lookup name contract" in result.output
