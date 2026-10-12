"""``research-families`` — the read-only benchmark family catalog for planners.

The dipcatcher chat planner (``fx1.interactive.superpower``) must select only
known, qualified, non-retired research families, but ``fx1`` may not import
``quant_fund.research``. This command publishes the catalog state through the
registered harness surface instead: required and live optional families,
the retired count with a sample, the family order, the forbidden headline
metric keys, and an explicit statement that ``dipcatcher research`` still
executes its configured scorecard — family selection narrows eligibility and
reporting, not the benches executed.
"""

from __future__ import annotations

import json

import typer

from quant_fund.cli.app import app

FAMILY_CATALOG_SCHEMA_VERSION = 1

_FAMILY_RUNNER_NOTE = (
    "dipcatcher research executes the REQUIRED scorecard plus live optional families "
    "per its config; per-family execution constraint is not supported yet — family "
    "selection narrows eligibility and reporting, not the benches executed."
)


@app.command("research-families")
def research_families(
    json_out: bool = typer.Option(
        True,
        "--json/--text",
        help="JSON (default) or a short human table.",
    ),
) -> None:
    """Read-only benchmark family catalog: required / live optional / retired, for planners."""
    from datetime import UTC, datetime

    from quant_fund.research.catalog.registry import (
        BENCHMARK_CATALOG_VERSION,
        BENCHMARK_FAMILY_ORDER,
        FORBIDDEN_RESEARCH_METRIC_KEYS,
        LIVE_OPTIONAL_BENCHMARK_FAMILIES,
        REQUIRED_BENCHMARK_FAMILIES,
        RESEARCH_RECEIPT_SCHEMA_VERSION,
        RETIRED_BENCHMARK_FAMILIES,
    )

    required = sorted(REQUIRED_BENCHMARK_FAMILIES)
    live_optional = sorted(LIVE_OPTIONAL_BENCHMARK_FAMILIES)
    retired = sorted(RETIRED_BENCHMARK_FAMILIES)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not json_out:
        typer.echo(
            f"benchmark family catalog v{BENCHMARK_CATALOG_VERSION} "
            f"(receipt schema v{RESEARCH_RECEIPT_SCHEMA_VERSION})"
        )
        typer.echo(f"required ({len(required)}): {', '.join(required)}")
        typer.echo(f"live optional: {len(live_optional)}")
        typer.echo(f"retired: {len(retired)}")
        typer.echo(_FAMILY_RUNNER_NOTE)
        return
    document = {
        "schema_version": FAMILY_CATALOG_SCHEMA_VERSION,
        "catalog_version": BENCHMARK_CATALOG_VERSION,
        "receipt_schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "required": required,
        "live_optional": live_optional,
        "retired_count": len(retired),
        "retired_sample": retired[:20],
        "order": list(BENCHMARK_FAMILY_ORDER),
        "forbidden_metric_keys": sorted(FORBIDDEN_RESEARCH_METRIC_KEYS),
        "runner": {
            "command": "research",
            "selection": "config",
            "execution_constraint": False,
            "note": _FAMILY_RUNNER_NOTE,
        },
        "generated_at": generated_at,
    }
    typer.echo(json.dumps(document, indent=2, sort_keys=True))


__all__ = ["FAMILY_CATALOG_SCHEMA_VERSION", "research_families"]
