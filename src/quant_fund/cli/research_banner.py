"""Notebook banner lines split out of the research command."""

from __future__ import annotations

import typer


def _echo_synthetic_marker(synthetic: bool) -> None:
    """Emit the notebook's SYNTHETIC line when the run is synthetic."""
    if synthetic:
        typer.echo("SYNTHETIC")
