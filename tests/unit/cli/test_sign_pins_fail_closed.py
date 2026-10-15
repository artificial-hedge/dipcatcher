"""sign-pins must never mint an ephemeral key over the committed trust root."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quant_fund.cli._app import app
from quant_fund.research.gate_signatures import generate_keypair

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
    result = runner.invoke(app, ["sign-pins", "--root", str(tmp_path), "--key-file", str(key_file)])
    assert result.exit_code == 2
    assert "does not match" in result.output
    assert "rotate-key" in result.output
    assert (tmp_path / "quality" / "gate_signing.pub").read_text().strip() == committed
