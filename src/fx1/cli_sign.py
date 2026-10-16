"""Checkpoint signing and attestation CLI handlers."""

from __future__ import annotations

import json
from pathlib import Path

import typer


def sign_checkpoint(
    checkpoint_dir: Path = typer.Argument(...),
    public: bool = typer.Option(False, "--public", help="Sign for public serving with Ed25519."),
) -> None:
    """Sign an fx-1 release (attestation ladder tier 1)."""
    from fx1.serve.signing import sign_release, sign_release_ed25519

    sig = sign_release_ed25519(checkpoint_dir) if public else sign_release(checkpoint_dir)
    typer.echo(f"signed: {sig}")


def attestation_status(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Report which attestation tiers a checkpoint satisfies."""
    from fx1.serve import attestation_ladder_status

    typer.echo(json.dumps(attestation_ladder_status(checkpoint_dir), indent=2))
