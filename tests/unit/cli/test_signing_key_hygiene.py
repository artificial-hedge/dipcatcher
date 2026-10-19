"""Signing keys never ride argv, and key files must be owner-only (D6-08).

Raw Ed25519 seeds on the command line leak through ``ps`` and shell history;
a key file with group/other mode bits may already have been copied. Every
key-reading CLI path refuses both, before anything is signed or written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli._app import app
from quant_fund.research.gate_signatures import (
    generate_keypair,
    init_quorum,
    sign_pins_quorum,
)

runner = CliRunner()
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits")
_KEY_ENVS = ("GATE_SIGNING_KEY", "GATE_QUORUM_KEYS", "WITNESS_SIGNING_KEY")
# Never a loadable key: should the refusal regress, PEM parsing fails before
# anything could be submitted to a transparency log.
_JUNK_PEM = b"-----BEGIN EC PRIVATE KEY-----\nbm90LWEta2V5\n-----END EC PRIVATE KEY-----\n"
_CLOSED_REKOR = "http://127.0.0.1:9"


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _KEY_ENVS:
        monkeypatch.delenv(var, raising=False)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _lone_tree(tmp_path: Path) -> tuple[Path, str, str]:
    """Pinned + checkpointed tree, lone gate key (no quorum registry)."""
    root = tmp_path / "repo"
    (root / "quality").mkdir(parents=True)
    (root / "quality" / "epoch_heads.json").write_text('{"heads": {}}\n')
    (root / "quality" / "crown_jewels.json").write_text('{"jewels": {}}\n')
    (root / "quality" / "checkpoint.json").write_text("{}\n")
    priv, pub = generate_keypair()
    (root / "quality" / "gate_signing.pub").write_text(pub + "\n")
    return root, priv, pub


def _quorum_tree(tmp_path: Path) -> tuple[Path, str, str]:
    """1-of-1 quorum registry, pins signed ``gate_signatures.v2``."""
    root, priv, pub = _lone_tree(tmp_path)
    init_quorum(root, [pub], threshold=1)
    sign_pins_quorum(root, [(priv, pub)])
    return root, priv, pub


def _key_file(path: Path, data: bytes, mode: int) -> Path:
    path.write_bytes(data)
    path.chmod(mode)
    return path


@pytest.mark.parametrize("shape", ["seed", "pair"])
def test_attest_release_refuses_raw_key_material_on_argv(tmp_path: Path, shape: str) -> None:
    root, priv, pub = _lone_tree(tmp_path)
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"wheel")
    out = tmp_path / "att.json"
    before = _snapshot(root)
    secret = priv if shape == "seed" else f"{priv}:{pub}"

    result = runner.invoke(
        app,
        ["attest-release", "--root", str(root), "--artifact", str(wheel)]
        + ["--out", str(out), "--key", secret],
    )

    assert result.exit_code == 2, result.output
    assert "never as a command-line value" in result.output
    assert priv not in result.output  # the seed is never echoed back
    assert not out.exists()
    assert _snapshot(root) == before


# (id, tree, key kind, argv template, env var naming the key file or None)
_CASES = [
    ("attest-release --key", "lone", "gate", ["attest-release", "--key", "{K}"], None),
    ("attest-release env", "lone", "gate", ["attest-release"], "GATE_SIGNING_KEY"),
    (
        "attest-release --witness-key-file",
        "lone",
        "witness",
        ["attest-release", "--key", "{GOOD}", "--witness", "--witness-key-file", "{K}"],
        None,
    ),
    (
        "attest-release witness env",
        "lone",
        "witness",
        ["attest-release", "--key", "{GOOD}", "--witness"],
        "WITNESS_SIGNING_KEY",
    ),
    ("sign-pins --key-file", "lone", "gate", ["sign-pins", "--key-file", "{K}"], None),
    ("sign-pins env", "lone", "gate", ["sign-pins"], "GATE_SIGNING_KEY"),
    ("checkpoint --key-file", "lone", "gate", ["checkpoint", "--key-file", "{K}"], None),
    ("checkpoint env", "lone", "gate", ["checkpoint"], "GATE_SIGNING_KEY"),
    ("quorum-sign --key-file", "quorum", "gate", ["quorum-sign", "--key-file", "{K}"], None),
    ("quorum-sign env", "quorum", "gate", ["quorum-sign"], "GATE_QUORUM_KEYS"),
    (
        "quorum-rotate --key-file",
        "quorum",
        "gate",
        ["quorum-rotate", "--registry", "{REG}", "--key-file", "{K}"],
        None,
    ),
    (
        "witness-checkpoint --key-file",
        "lone",
        "witness",
        ["witness-checkpoint", "--rekor-url", _CLOSED_REKOR, "--key-file", "{K}"],
        None,
    ),
    (
        "witness-checkpoint env",
        "lone",
        "witness",
        ["witness-checkpoint", "--rekor-url", _CLOSED_REKOR],
        "WITNESS_SIGNING_KEY",
    ),
]


@posix_only
@pytest.mark.parametrize("mode", [0o640, 0o604], ids=["group-read", "other-read"])
@pytest.mark.parametrize(
    ("case", "tree", "kind", "argv", "env_var"), _CASES, ids=[c[0] for c in _CASES]
)
def test_group_or_other_readable_key_file_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    tree: str,
    kind: str,
    argv: list[str],
    env_var: str | None,
    mode: int,
) -> None:
    root, priv, pub = (_quorum_tree if tree == "quorum" else _lone_tree)(tmp_path)
    data = f"{priv}\n{pub}\n".encode() if kind == "gate" else _JUNK_PEM
    loose = _key_file(tmp_path / "loose.key", data, mode)
    good = _key_file(tmp_path / "good.key", f"{priv}\n{pub}\n".encode(), 0o600)
    registry = tmp_path / "registry.json"
    registry.write_bytes(
        (root / "quality" / "gate_quorum.json").read_bytes() if tree == "quorum" else b"{}"
    )
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"wheel")
    out = tmp_path / "att.json"
    subs = {"{K}": str(loose), "{GOOD}": str(good), "{REG}": str(registry)}
    args = [subs.get(a, a) for a in argv] + ["--root", str(root)]
    if argv[0] == "attest-release":
        args += ["--artifact", str(wheel), "--out", str(out)]
    if env_var is not None:
        monkeypatch.setenv(env_var, str(loose))
    before = _snapshot(root)

    result = runner.invoke(app, args)

    assert result.exit_code == 2, f"{case}: {result.output}"
    assert "group/other access" in result.output
    assert f"mode {mode:04o}" in result.output
    assert "chmod 600" in result.output
    assert priv not in result.output
    assert not out.exists()
    assert _snapshot(root) == before  # refused before anything was signed or written


@posix_only
@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_owner_only_key_file_signs_and_verifies(tmp_path: Path, mode: int) -> None:
    root, priv, pub = _lone_tree(tmp_path)
    key = _key_file(tmp_path / "gate.key", f"{priv}:{pub}\n".encode(), mode)
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"wheel")
    out = tmp_path / "att.json"

    signed = runner.invoke(
        app,
        ["attest-release", "--root", str(root), "--artifact", str(wheel)]
        + ["--out", str(out), "--key", str(key)],
    )
    assert signed.exit_code == 0, signed.output
    assert json.loads(out.read_text())["schema"] == "release_attestation.v1"
    verified = runner.invoke(
        app, ["verify-release", str(out), "--artifact", str(wheel), "--root", str(root)]
    )
    assert verified.exit_code == 0, verified.output

    pins = runner.invoke(app, ["sign-pins", "--root", str(root), "--key-file", str(key)])
    assert pins.exit_code == 0, pins.output


def test_attest_release_env_inline_material_still_signs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment stays a supported channel for inline key material."""
    root, priv, pub = _lone_tree(tmp_path)
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"wheel")
    out = tmp_path / "att.json"
    monkeypatch.setenv("GATE_SIGNING_KEY", f"{priv}:{pub}")

    result = runner.invoke(
        app,
        ["attest-release", "--root", str(root), "--artifact", str(wheel), "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    verified = runner.invoke(
        app, ["verify-release", str(out), "--artifact", str(wheel), "--root", str(root)]
    )
    assert verified.exit_code == 0, verified.output
