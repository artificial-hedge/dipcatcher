"""``research-families`` — the read-only benchmark family catalog for planners.

The dipcatcher chat planner (``fx1.interactive.superpower``) must select only
known, qualified, non-retired research families, but ``fx1`` may not import
``quant_fund.research``. This command publishes the catalog state through the
registered harness surface instead: required and audited justified optional
families, counts for exploratory and retired families, the family order, the forbidden headline
metric keys, and an explicit statement that the runner has no per-family
selector. Named-family goals cannot automatically launch the full scorecard;
explicit ``dipcatcher research`` still runs its configured scorecard.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256

import typer

from quant_fund.cli.app import app

FAMILY_CATALOG_SCHEMA_VERSION = 2
_LOOKUP_MAX_PARTS = 5
_LOOKUP_MAX_HASHES = 300
_LOOKUP_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LOOKUP_FAMILY_NAME = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")

_FAMILY_RUNNER_NOTE = (
    "the research runner has no per-family selector; automatic research is excluded "
    "when a goal names a family, while explicit research runs the configured scorecard."
)


@app.command("research-families")
def research_families(
    json_out: bool = typer.Option(
        True,
        "--json/--text",
        help="JSON (default) or a short human table.",
    ),
    lookup_hashes: str | None = typer.Option(
        None,
        "--lookup-hashes",
        help="Bounded SHA-256 name-candidate lookup for a single planner goal.",
    ),
) -> None:
    """Planner-selectable families and counts for exploratory and retired bulk."""
    from datetime import UTC, datetime

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

    required = sorted(REQUIRED_BENCHMARK_FAMILIES)
    # Schema v2 exposes only audited justified optional names to the planner.
    # Full accepted history remains in the registry for archived receipt checks.
    live_optional = sorted(justified_families())
    exploratory_count = len(LIVE_OPTIONAL_BENCHMARK_FAMILIES) - len(live_optional)
    retired_count = len(RETIRED_BENCHMARK_FAMILIES)
    lookup: dict[str, object] | None = None
    if lookup_hashes is not None:
        if not json_out:
            raise typer.BadParameter("--lookup-hashes requires JSON output")
        requested = lookup_hashes.split(",")
        if (
            not requested
            or len(requested) > _LOOKUP_MAX_HASHES
            or requested != sorted(set(requested))
            or any(_LOOKUP_HASH.fullmatch(value) is None for value in requested)
        ):
            raise typer.BadParameter("lookup hashes must be 1-300 sorted unique SHA-256 values")
        omitted = (LIVE_OPTIONAL_BENCHMARK_FAMILIES - set(live_optional)) | set(
            RETIRED_BENCHMARK_FAMILIES
        )
        if any(
            _LOOKUP_FAMILY_NAME.fullmatch(name) is None or len(name.split("_")) > _LOOKUP_MAX_PARTS
            for name in omitted
        ):
            raise typer.BadParameter("omitted family violates the lookup name contract")
        requested_set = set(requested)
        lookup = {
            "request_sha256": sha256(lookup_hashes.encode("ascii")).hexdigest(),
            "omitted_matches": sorted(
                name
                for name in omitted
                if sha256(name.encode("ascii")).hexdigest() in requested_set
            ),
        }
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not json_out:
        typer.echo(
            f"benchmark family catalog v{BENCHMARK_CATALOG_VERSION} "
            f"(receipt schema v{RESEARCH_RECEIPT_SCHEMA_VERSION})"
        )
        typer.echo(f"required ({len(required)}): {', '.join(required)}")
        typer.echo(f"planner-selectable optional: {len(live_optional)}")
        typer.echo(f"exploratory optional: {exploratory_count}")
        typer.echo(f"live optional total: {len(LIVE_OPTIONAL_BENCHMARK_FAMILIES)}")
        typer.echo(f"retired: {retired_count}")
        typer.echo(_FAMILY_RUNNER_NOTE)
        return
    document = {
        "schema_version": FAMILY_CATALOG_SCHEMA_VERSION,
        "catalog_version": BENCHMARK_CATALOG_VERSION,
        "receipt_schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "required": required,
        "live_optional": live_optional,
        "live_optional_total": len(LIVE_OPTIONAL_BENCHMARK_FAMILIES),
        "exploratory_count": exploratory_count,
        "retired_count": retired_count,
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
    if lookup is not None:
        document["lookup"] = lookup
    typer.echo(json.dumps(document, indent=2, sort_keys=True))


__all__ = ["FAMILY_CATALOG_SCHEMA_VERSION", "research_families"]
