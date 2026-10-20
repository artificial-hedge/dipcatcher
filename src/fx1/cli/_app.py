"""Typer application root and subcommand groups for ``fx1``."""

from __future__ import annotations

import typer

from fx1.cli.support import _NO_WEIGHTS
from fx1.cli_harness import harness_app

app = typer.Typer(
    name="fx1",
    help=f"{_NO_WEIGHTS}. dipcatcher builds, evaluates, and verifies the program.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", help="Show the fx-1 package version and exit.", is_eager=True
    ),
) -> None:
    if version:
        from fx1 import __version__

        typer.echo(f"fx1 {__version__} ({_NO_WEIGHTS})")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


corpus_app = typer.Typer(help="Training-corpus construction.")
train_app = typer.Typer(help="Training-run manifests (LoRA/QLoRA on K3).")
sources_app = typer.Typer(help="Professional datasource registry, routing, fetch.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(train_app, name="train")
app.add_typer(harness_app, name="harness")
app.add_typer(sources_app, name="sources")
