"""Qualified family surface.

``families`` is read-only. ``research --only`` executes the named runners and
writes a partial, unverified record. Neither path invents market data.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.cli.app import app


def _dump(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command("families")
def families_cmd(
    select: str | None = typer.Option(
        None, "--select", help="Goal text. Prints eligible families and exclusions."
    ),
    family_id: str | None = typer.Option(None, "--id", help="Status for one family id."),
    max_families: int = typer.Option(4, "--max", min=0, max=23),
) -> None:
    """List or select qualified research families. Does not execute benches."""
    from quant_fund.research.family_surface import (
        SURFACE_SCHEMA,
        FamilySurfaceError,
        family_contract,
        list_executable,
        select_families,
    )

    if select is not None and family_id is not None:
        typer.echo("pass only one of --select and --id", err=True)
        raise typer.Exit(2)
    try:
        if family_id is not None:
            payload: object = family_contract(family_id)
        elif select is not None:
            payload = select_families(select, max_families=max_families)
        else:
            payload = {"schema": SURFACE_SCHEMA, "families": list_executable()}
    except FamilySurfaceError as exc:
        typer.echo(f"families refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    _dump(payload)


def emit_selected_run(config: Path, only: str) -> None:
    """Run the named families and print a partial, unverified record."""
    from quant_fund.cli.support import _cfg, format_data_label
    from quant_fund.research.family_surface import FamilySurfaceError, parse_only, run_families

    try:
        ids = parse_only(only)
        cfg = _cfg(config)
        payload = run_families(cfg, ids)
    except FamilySurfaceError as exc:
        typer.echo(f"family run refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    synthetic = payload["evidence_class"] == "SYNTHETIC"
    source = str(getattr(getattr(cfg, "data", None), "source", payload["data_source"]))
    typer.echo(format_data_label(synthetic=synthetic, data_source=source))
    if synthetic:
        typer.echo("SYNTHETIC")
    typer.echo("PARTIAL_FAMILY_RUN verified=false")
    typer.echo("families=" + ",".join(payload["family_ids"]))
    results = payload["results"]
    if isinstance(results, dict):
        for fam_id, row in results.items():
            if isinstance(row, dict):
                keys = ",".join(sorted(str(key) for key in row)[:8])
                typer.echo(f"{fam_id}: keys={keys}")
            else:
                typer.echo(f"{fam_id}: type={type(row).__name__}")
    typer.echo(f"json={payload.get('artifact', '')}")
    typer.echo(payload["note"])
