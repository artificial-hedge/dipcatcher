"""Read-only ``research-families`` catalog command."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from quant_fund.cli.main import app
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
    "dipcatcher research executes the REQUIRED scorecard plus live optional families "
    "per its config; per-family execution constraint is not supported yet — family "
    "selection narrows eligibility and reporting, not the benches executed."
)

_RUNNER = CliRunner()


def test_research_families_json_lists_the_catalog() -> None:
    result = _RUNNER.invoke(app, ["research-families"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["catalog_version"] == BENCHMARK_CATALOG_VERSION
    assert payload["receipt_schema_version"] == RESEARCH_RECEIPT_SCHEMA_VERSION
    assert payload["required"] == sorted(REQUIRED_BENCHMARK_FAMILIES)
    assert payload["live_optional"] == sorted(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    assert set(payload["live_optional"]).isdisjoint(RETIRED_BENCHMARK_FAMILIES)
    for name in payload["retired_sample"]:
        assert name not in payload["live_optional"]
    assert payload["retired_sample"] == sorted(RETIRED_BENCHMARK_FAMILIES)[:20]
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
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", payload["generated_at"])
    assert result.output == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_research_families_text_is_a_short_table() -> None:
    result = _RUNNER.invoke(app, ["research-families", "--text"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("benchmark family catalog v")
    assert f"required ({len(REQUIRED_BENCHMARK_FAMILIES)}):" in result.output
    assert "required (" in result.output
    assert f"live optional: {len(LIVE_OPTIONAL_BENCHMARK_FAMILIES)}" in result.output
    assert f"retired: {len(RETIRED_BENCHMARK_FAMILIES)}" in result.output
    assert "per-family execution constraint is not supported yet" in result.output
