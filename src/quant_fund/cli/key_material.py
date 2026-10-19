"""Key-material helpers for the signing CLI commands.

Private key material is accepted only from an owner-only file or the
environment, never as a command-line value, and a lone-key signature is
never written over a committed quorum registry. Shared by ``sign-pins``,
``quorum-sign`` and ``attest-release`` in :mod:`quant_fund.cli.research_cmds`.
"""

from __future__ import annotations

from pathlib import Path

import typer


def _read_key_file(path: Path, *, cmd: str, label: str | None = None) -> bytes:
    """Read private key material from ``path``; exit 2 unless owner-only.

    On POSIX a key file whose mode grants any group/other bit
    (``st_mode & 0o077``) is refused before its bytes are read — another
    local account may already hold a copy. The mode comes from the open
    descriptor, so the file checked is the file read. ``label`` replaces the
    path in messages when the value may itself be a secret (never echoed).
    """
    import os
    import stat

    shown = label or f"key file {path}"
    try:
        handle = path.open("rb")
    except OSError as exc:
        typer.echo(
            f"{cmd}: {shown} is not a readable key file ({exc.strerror}) — key "
            "material is accepted only from a file or the environment, never as "
            "a command-line value"
        )
        raise typer.Exit(code=2) from exc
    with handle:
        mode = os.fstat(handle.fileno()).st_mode
        if os.name == "posix" and mode & 0o077:
            typer.echo(
                f"{cmd}: refusing {shown}: mode {stat.S_IMODE(mode):04o} grants "
                "group/other access — `chmod 600` it (and rotate the key if anyone "
                "else could have read it)"
            )
            raise typer.Exit(code=2)
        return handle.read()


def _env_key_material(var: str, *, cmd: str) -> bytes | None:
    """Key material from env ``var``: inline value, or a permission-checked
    file when the value names one. ``None`` when unset or blank."""
    import os

    env = os.environ.get(var, "").strip()
    if not env:
        return None
    if Path(env).is_file():
        return _read_key_file(Path(env), cmd=cmd)
    return env.encode()


def _attest_signing_pair(key: Path | None, root: Path) -> tuple[str, str]:
    """(priv_hex, committed pub_hex) for ``attest-release``; fail closed."""
    if key is not None:
        material: bytes | None = _read_key_file(key, cmd="attest-release", label="--key")
    else:
        material = _env_key_material("GATE_SIGNING_KEY", cmd="attest-release")
    if material is None:
        raise typer.BadParameter(
            "no key material — pass --key <key file> or set GATE_SIGNING_KEY; "
            "cannot sign a release attestation"
        )
    raw = material.decode().strip()
    fields = [s.strip() for s in raw.replace(":", "\n").splitlines() if s.strip()]
    pub_path = root / "quality/gate_signing.pub"
    if not pub_path.is_file():
        raise typer.BadParameter(f"no committed pubkey at {pub_path}")
    pub = pub_path.read_text().strip()
    if len(fields) == 2 and fields[1] != pub:
        raise typer.BadParameter(
            "supplied pubkey != committed quality/gate_signing.pub — refusing to sign"
        )
    return fields[0], pub


def _attest_witness_pem(witness_key_file: Path | None) -> bytes:
    """Witness ECDSA PEM for ``attest-release --witness``; fail closed."""
    if witness_key_file is not None:
        return _read_key_file(witness_key_file, cmd="attest-release")
    pem = _env_key_material("WITNESS_SIGNING_KEY", cmd="attest-release")
    if pem is None:
        raise typer.BadParameter("--witness needs WITNESS_SIGNING_KEY env or --witness-key-file")
    return pem


def _refuse_lone_key_under_quorum(root: Path) -> None:
    """Exit 2 when a quorum registry is committed: ``verify_pin_signatures``
    rejects a lone-key ``gate_signatures.v1`` file there (``quorum_sig_missing``),
    so writing one would replace the quorum signature with a guaranteed fail.
    """
    from quant_fund.research.gate_signatures import DEFAULT_QUORUM_PATH

    registry = root / DEFAULT_QUORUM_PATH
    if registry.exists():
        typer.echo(
            f"sign-pins: {registry.as_posix()} is committed — a lone-key "
            "gate_signatures.v1 file fails verification there (quorum_sig_missing); "
            "refusing to overwrite the quorum signature. Use `dipcatcher quorum-sign` "
            "(--key-file per signer, or GATE_QUORUM_KEYS)."
        )
        raise typer.Exit(code=2)


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
