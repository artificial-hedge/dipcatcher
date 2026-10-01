"""tape_manifest.v1 — pin/verify round-trip and the receipt-binding ratchet.

The registry closes the self-attestation gap: a receipt binding
``dataset_sha256``/``tape_manifest_sha256`` under a non-synthetic
``data_label`` must resolve to a committed manifest. All tape frames here
are SYNTHETIC fixtures — the manifest machinery is what is under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.research.tape_registry import (
    TAPE_MANIFEST_SCHEMA,
    frame_csv_sha256,
    load_registry,
    lookup_digest,
    pin_tape,
    tape_binding_errors,
    verify_manifest,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]


def _tape_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["AAA", "AAA", "BBB"],
            "event_time": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "close": [101.0, 102.5, 7.0],
        }
    )


def _write_tape(tmp_path: Path, frame: pl.DataFrame | None = None) -> Path:
    tape = tmp_path / "bars.parquet"
    (frame if frame is not None else _tape_frame()).write_parquet(tape)
    return tape


def _pin(tmp_path: Path, tape: Path) -> dict[str, Any]:
    out_dir = tmp_path / "data" / "manifests"
    return pin_tape("unit_tape", tape, out_dir=out_dir, root=tmp_path)


def _sealed_v1(body: dict[str, Any]) -> dict[str, Any]:
    canonical = json.loads(canonical_json_bytes(body))
    return {**canonical, "receipt_sha256": hash_bytes(canonical_json_bytes(canonical))}


def test_pin_writes_sealed_manifest_and_verify_round_trip(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    result = _pin(tmp_path, tape)
    manifest_path = result["path"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == TAPE_MANIFEST_SCHEMA
    assert manifest["source_label"] == "unit_tape"
    assert manifest["tape_files"][0]["sha256"] == hash_bytes(tape.read_bytes())
    assert manifest["tape_files"][0]["n_bytes"] == tape.stat().st_size
    assert manifest["n_rows"] == 3
    assert manifest["n_names"] == 2
    assert manifest["window"]["start"] == "2024-01-02"
    assert manifest["receipt_sha256"] == hash_bytes(
        canonical_json_bytes({k: v for k, v in manifest.items() if k != "receipt_sha256"})
    )
    assert verify_manifest(manifest_path, tmp_path) == []


def test_verify_detects_tape_byte_tamper(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    result = _pin(tmp_path, tape)
    raw = bytearray(tape.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    tape.write_bytes(bytes(raw))
    errors = verify_manifest(result["path"], tmp_path)
    assert any(error.startswith("tape_sha256_mismatch") for error in errors)


def test_verify_detects_missing_tape_and_manifest_tamper(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    result = _pin(tmp_path, tape)
    tape.unlink()
    assert verify_manifest(result["path"], tmp_path) == ["tape_missing:bars.parquet"]
    tape = _write_tape(tmp_path)
    payload = json.loads(result["path"].read_text())
    payload["n_rows"] += 1
    result["path"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    errors = verify_manifest(result["path"], tmp_path)
    assert "receipt_sha256_mismatch" in errors


def test_registry_indexes_seal_tape_and_csv_digests(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    result = _pin(tmp_path, tape)
    manifest = json.loads(result["path"].read_text())
    registry = load_registry(tmp_path)
    for key in (
        manifest["receipt_sha256"],
        manifest["tape_files"][0]["sha256"],
        manifest["frame_csv_sha256"],
    ):
        assert lookup_digest(registry, key) is not None
    assert lookup_digest(registry, "f" * 64) is None


def test_registry_skips_unsealed_manifest(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    result = _pin(tmp_path, tape)
    payload = json.loads(result["path"].read_text())
    payload["n_rows"] += 1  # body drifts under the same seal
    result["path"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry = load_registry(tmp_path)
    assert lookup_digest(registry, payload["tape_files"][0]["sha256"]) is None


def test_csv_digest_matches_lane_convention(tmp_path: Path) -> None:
    """The manifest digest is the same bytes a lane seals as inputs_sha256."""
    tape = _write_tape(tmp_path)
    result = _pin(tmp_path, tape)
    manifest = json.loads(result["path"].read_text())
    lane_inputs = hash_bytes(pl.read_parquet(tape).write_csv().encode("utf-8"))
    assert manifest["frame_csv_sha256"] == lane_inputs == frame_csv_sha256(pl.read_parquet(tape))


def test_receipt_binding_unknown_digest_fails_closed(tmp_path: Path) -> None:
    receipt = _sealed_v1(
        {
            "schema": "unit_lane.v1",
            "kind": "unit_lane",
            "data_label": "yahoo_eod",
            "dataset_sha256": "f" * 64,
        }
    )
    result = verify_receipt_payload(receipt)
    assert "tape_manifest_unknown" in result["errors"]
    assert result["valid"] is False
    # Synthetic lanes are exempt from the tape-binding ratchet.
    synthetic = _sealed_v1(
        {
            "schema": "unit_lane.v1",
            "kind": "unit_lane",
            "data_label": "SYNTHETIC",
            "dataset_sha256": "f" * 64,
        }
    )
    assert "tape_manifest_unknown" not in verify_receipt_payload(synthetic)["errors"]


def test_receipt_binding_resolves_pinned_digests(tmp_path: Path) -> None:
    tape = _write_tape(tmp_path)
    manifest = _pin(tmp_path, tape)["manifest"]
    # Every digest a manifest attests resolves a receipt's declared binding.
    for binding_key, digest in (
        ("dataset_sha256", manifest["tape_files"][0]["sha256"]),
        ("dataset_sha256", manifest["frame_csv_sha256"]),
        ("tape_manifest_sha256", manifest["receipt_sha256"]),
    ):
        receipt = _sealed_v1(
            {
                "schema": "unit_lane.v1",
                "kind": "unit_lane",
                "data_label": "yahoo_eod",
                binding_key: digest,
            }
        )
        assert tape_binding_errors(receipt, root=tmp_path) == []


def test_committed_yahoo_manifest_attests_receipt_binding() -> None:
    """The pinned yahoo_eod manifest resolves a digest-bound receipt."""
    manifest_path = REPO_ROOT / "data" / "manifests" / "yahoo_eod.json"
    if not manifest_path.is_file():
        pytest.skip("committed yahoo_eod tape manifest absent")
    manifest = json.loads(manifest_path.read_text())
    registry = load_registry(REPO_ROOT)
    assert lookup_digest(registry, manifest["receipt_sha256"]) is not None
    assert lookup_digest(registry, manifest["tape_files"][0]["sha256"]) is not None
    receipt = _sealed_v1(
        {
            "schema": "unit_lane.v1",
            "kind": "unit_lane",
            "data_label": "yahoo_eod",
            "tape_manifest_sha256": manifest["receipt_sha256"],
        }
    )
    assert "tape_manifest_unknown" not in verify_receipt_payload(receipt)["errors"]


def test_pin_rejects_missing_and_outside_repo_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pin_tape("unit_tape", tmp_path / "absent.parquet", out_dir=tmp_path / "m")
    tape = _write_tape(tmp_path)
    with pytest.raises(ValueError, match="repo-relative"):
        pin_tape("unit_tape", tape, out_dir=tmp_path / "m", root=tmp_path / "elsewhere")


def test_cli_tape_pin_and_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    tape = _write_tape(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["tape-pin", "--source-label", "unit_tape", "--tape", str(tape)],
    )
    assert result.exit_code == 0, result.output
    manifest_path = tmp_path / "data" / "manifests" / "unit_tape.json"
    assert manifest_path.is_file()
    verified = runner.invoke(
        app, ["tape-verify", "--manifest", str(manifest_path), "--root", str(tmp_path)]
    )
    assert verified.exit_code == 0, verified.output
    assert '"valid": true' in verified.output
    missing = runner.invoke(
        app, ["tape-pin", "--source-label", "unit_tape", "--tape", "nope.parquet"]
    )
    assert missing.exit_code != 0
