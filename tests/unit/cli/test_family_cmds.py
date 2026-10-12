"""CLI wiring for the read-only family surface and selective research runs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quant_fund.cli.main import app

runner = CliRunner()


def test_families_status_json_for_required_family() -> None:
    result = runner.invoke(app, ["families", "--id", "volatility"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["id"] == "volatility"
    assert payload["executable"] is True
    assert payload["schema"] == "dipcatcher.family_surface.v1"


def test_families_refuses_unsafe_id() -> None:
    result = runner.invoke(app, ["families", "--id", "vol atility"])
    assert result.exit_code == 1
    assert "refused" in (result.stderr + result.output).lower()


def test_families_select_prints_exclusions() -> None:
    result = runner.invoke(app, ["families", "--select", "qlike volatility forecast", "--max", "2"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload["selected"]] == ["volatility"]


def test_research_only_refuses_unknown_family(tmp_path: Path) -> None:
    config = tmp_path / "research.yaml"
    config.write_text("data:\n  source: synthetic\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["research", "--config", str(config), "--only", "not_a_real_family_zzz"],
    )
    assert result.exit_code == 1
    assert "refused" in (result.stderr + result.output).lower()
    assert "PARTIAL_FAMILY_RUN" not in result.stdout
