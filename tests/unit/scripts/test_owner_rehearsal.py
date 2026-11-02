"""The owner rehearsal must stay outside the real checkout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.owner_rehearsal import (
    RehearsalRefusal,
    assert_throwaway,
    public_hex,
    require_copy,
    retarget_data_manifest,
    unlink_under,
    write_quorum_registry,
)

from quant_fund.research.gate_signatures import generate_keypair, quorum_registry_errors

REPO = Path(__file__).resolve().parents[3]


def test_throwaway_refuses_the_source_and_its_children(tmp_path: Path) -> None:
    source = tmp_path / "real"
    source.mkdir()
    with pytest.raises(RehearsalRefusal):
        assert_throwaway(source, source)
    nested = source / "nested"
    nested.mkdir()
    with pytest.raises(RehearsalRefusal):
        assert_throwaway(source, nested)
    with pytest.raises(RehearsalRefusal):
        assert_throwaway(nested, source)


def test_throwaway_accepts_a_sibling_directory(tmp_path: Path) -> None:
    source = tmp_path / "real"
    dest = tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    assert_throwaway(source, dest)


def test_require_copy_needs_the_marker(tmp_path: Path) -> None:
    source = tmp_path / "real"
    dest = tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    with pytest.raises(RehearsalRefusal):
        require_copy(source, dest)
    (dest / ".owner-rehearsal").write_text("throwaway\n", encoding="utf-8")
    require_copy(source, dest)


def test_unlink_under_rejects_paths_outside_the_copy(tmp_path: Path) -> None:
    copy_root = tmp_path / "copy" / "quality"
    outside = tmp_path / "real" / "quality"
    copy_root.mkdir(parents=True)
    outside.mkdir(parents=True)
    victim = outside / "checkpoint.json"
    victim.write_text("keep\n", encoding="utf-8")
    with pytest.raises(RehearsalRefusal):
        unlink_under(copy_root, victim)
    assert victim.read_text(encoding="utf-8") == "keep\n"
    link = copy_root / "checkpoint.json"
    link.symlink_to(victim)
    with pytest.raises(RehearsalRefusal):
        unlink_under(copy_root, link)
    assert victim.is_file()


def test_public_hex_matches_the_seed(tmp_path: Path) -> None:
    seed, pub = generate_keypair()
    assert public_hex(seed) == pub
    registry = tmp_path / "quorum.json"
    write_quorum_registry(pub, registry)
    body = json.loads(registry.read_text(encoding="utf-8"))
    assert body["threshold"] == 1
    assert body["keys"][0]["pubkey"] == pub
    assert quorum_registry_errors(body) == []


def test_retarget_points_manifest_paths_at_the_copy(tmp_path: Path) -> None:
    source = tmp_path / "real"
    dest = tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    (dest / ".owner-rehearsal").write_text("throwaway\n", encoding="utf-8")
    manifest = dest / "data" / "metadata" / "data_manifest.json"
    manifest.parent.mkdir(parents=True)
    original = {
        "schema_version": 1,
        "source": "synthetic",
        "artifacts": {"bars": {"path": str(source / "data/raw/bars.parquet"), "rows": 1}},
    }
    manifest.write_text(json.dumps(original), encoding="utf-8")
    assert retarget_data_manifest(source, dest) is True
    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["artifacts"]["bars"]["path"] == str((dest / "data/raw/bars.parquet").resolve())
    assert "receipt_sha256" in body
    assert retarget_data_manifest(source, dest) is False


def test_owner_steps_sign_after_the_new_public_key_and_witness_the_one_checkpoint() -> None:
    text = (REPO / "OWNER_STEPS.md").read_text(encoding="utf-8")
    order = [
        "make rotate-key",
        "printf '%s\\n' \"$new_pub\" > quality/gate_signing.pub",
        "quorum-rotate",
        "crown-jewels --write",
        "make stamp-epochs",
        "make sign-pins",
        "make anchor-pins",
        "make checkpoint\n",
        "make anchor-checkpoint",
        "make witness-checkpoint",
        "git tag -a v0.4.0",
        "owner rehearsal: local tag",
    ]
    positions = [text.index(token) for token in order]
    assert positions == sorted(positions)
    assert "git push origin v0.4.0" in text


def test_sign_pins_uses_the_quorum_and_anchor_does_not_resign() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    sign = text.split("sign-pins:", 1)[1].split("\n\n", 1)[0]
    anchor = text.split("anchor-checkpoint:", 1)[1].split("\n\n", 1)[0]
    assert "quorum-sign" in sign
    assert "anchor-timestamp --file quality/checkpoint.json" in anchor
    assert "checkpoint --anchor" not in anchor
    assert "owner-rehearsal:" in text
