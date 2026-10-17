"""Read-only quorum signer eligibility command and shared key parser."""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app


def _load_signer_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse 'PRIV:PUB' or two-line material; comma/newline separates signers."""
    fields = [
        f.strip()
        for line in raw.replace(",", "\n").splitlines()
        for f in line.replace(":", "\n").splitlines()
        if f.strip()
    ]
    if not fields or len(fields) % 2 != 0:
        raise ValueError("signer material must be '<priv_hex>:<pub_hex>' pairs")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields), 2)]


@app.command("quorum-check-key")
def quorum_check_key_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    key_file: list[Path] = typer.Option(
        ..., "--key-file", help="Registered signer pair file. Repeatable for M-of-N."
    ),
) -> None:
    """Check signer files against the committed quorum without writing pins."""
    from quant_fund.research.gate_signatures import validate_quorum_signers

    try:
        raws = [file.read_text().strip() for file in key_file]
        signers = [pair for raw in raws for pair in _load_signer_pairs(raw)]
        count, threshold = validate_quorum_signers(root, signers)
    except (OSError, UnicodeError) as exc:
        typer.echo("quorum-check-key: cannot read signer file")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(f"quorum-check-key: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"quorum-check-key: eligible {count}/{threshold}")
