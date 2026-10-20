"""sign-pins must never mint an ephemeral key over the committed trust root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli._app import app
from quant_fund.research.gate_signatures import generate_keypair, init_quorum

runner = CliRunner()


def test_sign_pins_refuses_missing_key(tmp_path: Path) -> None:
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "epoch_heads.json").write_text("{}\n")
    (tmp_path / "quality" / "crown_jewels.json").write_text("{}\n")
    (tmp_path / "quality" / "gate_signing.pub").write_text("a" * 64 + "\n")
    result = runner.invoke(app, ["sign-pins", "--root", str(tmp_path)])
    assert result.exit_code == 2
    assert "no key material" in result.output
    assert "refusing to mint an ephemeral key" in result.output
    # Trust root must be untouched.
    assert (tmp_path / "quality" / "gate_signing.pub").read_text().strip() == "a" * 64
    assert not (tmp_path / "gate_pins.sig").exists()


def test_sign_pins_refuses_pubkey_mismatch(tmp_path: Path) -> None:
    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "epoch_heads.json").write_text("{}\n")
    (tmp_path / "quality" / "crown_jewels.json").write_text("{}\n")
    committed = "b" * 64
    (tmp_path / "quality" / "gate_signing.pub").write_text(committed + "\n")
    priv, pub = generate_keypair()
    assert pub != committed
    key_file = tmp_path / "key.txt"
    key_file.write_text(f"{priv}\n{pub}\n")
    key_file.chmod(0o600)  # key files must be owner-only
    result = runner.invoke(app, ["sign-pins", "--root", str(tmp_path), "--key-file", str(key_file)])
    assert result.exit_code == 2
    assert "does not match" in result.output
    assert "rotate-key" in result.output
    assert (tmp_path / "quality" / "gate_signing.pub").read_text().strip() == committed


def _quorum_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Committed-shape trust root with a throwaway key: 1-of-1 registry whose
    key is also ``gate_signing.pub``, pins signed ``gate_signatures.v2``."""
    from quant_fund.research.gate_signatures import init_quorum, sign_pins_quorum

    root = tmp_path / "repo"
    (root / "quality").mkdir(parents=True)
    (root / "quality" / "epoch_heads.json").write_text('{"heads": {}}\n')
    (root / "quality" / "crown_jewels.json").write_text('{"jewels": {}}\n')
    priv, pub = generate_keypair()
    (root / "quality" / "gate_signing.pub").write_text(pub + "\n")
    init_quorum(root, [pub], threshold=1)
    sign_pins_quorum(root, [(priv, pub)])
    key_file = tmp_path / "key.txt"
    key_file.write_text(f"{priv}\n{pub}\n")
    key_file.chmod(0o600)  # key files must be owner-only
    return root, key_file


@pytest.mark.parametrize("key_source", ["key_file", "env", "none"])
def test_sign_pins_refuses_lone_key_under_quorum_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key_source: str
) -> None:
    from quant_fund.research.gate_signatures import verify_pin_signatures

    root, key_file = _quorum_tree(tmp_path)
    monkeypatch.delenv("GATE_SIGNING_KEY", raising=False)
    args = ["sign-pins", "--root", str(root)]
    if key_source == "key_file":
        args += ["--key-file", str(key_file)]
    elif key_source == "env":
        monkeypatch.setenv("GATE_SIGNING_KEY", str(key_file))
    sig = root / "gate_pins.sig"
    pub = root / "quality" / "gate_signing.pub"
    sig_before, pub_before = sig.read_bytes(), pub.read_bytes()
    assert verify_pin_signatures(root)["ok"] is True

    result = runner.invoke(app, args)

    assert result.exit_code == 2, result.output
    assert "quorum-sign" in result.output
    assert "quorum_sig_missing" in result.output
    # The v2 quorum signature is never replaced by a v1 the verifier rejects.
    assert sig.read_bytes() == sig_before
    assert pub.read_bytes() == pub_before
    assert verify_pin_signatures(root) == {
        "ok": True,
        "signed": True,
        "errors": [],
        "quorum": "1/1",
    }


def test_quorum_sign_is_the_working_resign_path(tmp_path: Path) -> None:
    """The path sign-pins points to re-signs drifted pins under the registry."""
    from quant_fund.research.gate_signatures import verify_pin_signatures

    root, key_file = _quorum_tree(tmp_path)
    (root / "quality" / "epoch_heads.json").write_text('{"heads": {"x": 1}}\n')
    assert "pin_drift:quality/epoch_heads.json" in verify_pin_signatures(root)["errors"]

    result = runner.invoke(app, ["quorum-sign", "--root", str(root), "--key-file", str(key_file)])

    assert result.exit_code == 0, result.output
    body = json.loads((root / "gate_pins.sig").read_text())
    assert body["schema"] == "gate_signatures.v2"
    assert verify_pin_signatures(root)["ok"] is True


def test_sign_pins_lone_key_path_still_signs_without_registry(tmp_path: Path) -> None:
    from quant_fund.research.gate_signatures import verify_pin_signatures

    (tmp_path / "quality").mkdir()
    (tmp_path / "quality" / "epoch_heads.json").write_text("{}\n")
    (tmp_path / "quality" / "crown_jewels.json").write_text("{}\n")
    priv, pub = generate_keypair()
    (tmp_path / "quality" / "gate_signing.pub").write_text(pub + "\n")
    key_file = tmp_path / "key.txt"
    key_file.write_text(f"{priv}:{pub}\n")
    key_file.chmod(0o600)  # key files must be owner-only

    result = runner.invoke(app, ["sign-pins", "--root", str(tmp_path), "--key-file", str(key_file)])

    assert result.exit_code == 0, result.output
    body = json.loads((tmp_path / "gate_pins.sig").read_text())
    assert body["schema"] == "gate_signatures.v1"
    assert verify_pin_signatures(tmp_path) == {"ok": True, "signed": True, "errors": []}
