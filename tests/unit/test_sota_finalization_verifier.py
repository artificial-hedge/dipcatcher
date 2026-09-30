"""Focused integrity tests for the independent finalization verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.verify_sota_finalization import verify_manifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(tmp_path: Path, rows: int = 3) -> Path:
    repo = tmp_path / "repo"
    inp = repo / "eval"
    run = inp / "runs" / "valid-run"
    inp.mkdir(parents=True)
    run.mkdir(parents=True)

    def make_npz(path: Path, count: int) -> None:
        np.savez(
            path,
            crps_matrix=np.ones((count, 2)),
            pinball_cube=np.ones((count, 2, 3)),
            asset_ids=np.array(["BTCUSDT"] * count),
            model_names=np.array(["model_a", "model_b"]),
            meta_json=np.array(json.dumps({"schema": "losses.v1"})),
        )

    shard = inp / "part.losses.npz"
    losses = run / "merged.losses.npz"
    make_npz(shard, rows)
    make_npz(losses, rows)
    receipt = run / "merged.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "sota_eval.v4",
                "research_only": True,
                "live_pnl_claim": False,
                "source_parts_sha256": {str(shard): sha(shard)},
                "losses_file": losses.name,
                "losses_sha256": sha(losses),
                "n_rows": rows,
                "n_complete": rows,
                "n_dropped_incomplete": 0,
                "scoring_contract": "verified.v1",
            }
        )
    )
    stdout, stderr, transcript = (
        run / n for n in ("group.stdout.txt", "group.stderr.txt", "group.transcript.txt")
    )
    command_record = run / "group.command.json"
    stdout.write_text("ok\n")
    stderr.write_text("")
    transcript.write_text("COMMAND: python evaluator\nEXIT_CODE: 0\n")
    group = {
        "group": "fixture",
        "command": ["python", "evaluator"],
        "command_text": "python evaluator",
        "cwd": str(repo),
        "shards": [
            {
                "path": str(shard),
                "relative": "eval/part.losses.npz",
                "bytes": shard.stat().st_size,
                "sha256": sha(shard),
            }
        ],
        "stdout": str(stdout),
        "stderr": str(stderr),
        "transcript": str(transcript),
        "stdout_sha256": sha(stdout),
        "stderr_sha256": sha(stderr),
        "transcript_sha256": sha(transcript),
        "receipt": str(receipt),
        "receipt_sha256": sha(receipt),
        "losses": str(losses),
        "losses_sha256": sha(losses),
        "exit_code": 0,
        "status": "succeeded",
        "command_record": str(command_record),
    }
    command_record.write_text(json.dumps(group))
    manifest = {
        "schema": "sota_finalization.v2",
        "status": "complete",
        "proof_status": "UNPROVEN",
        "run_id": "valid-run",
        "repo": str(repo),
        "input_dir": str(inp),
        "groups": [group],
    }
    path = run / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def test_valid_manifest_is_diagnostic_only(tmp_path: Path) -> None:
    result = verify_manifest(make_fixture(tmp_path))
    assert result["valid"] is True
    assert result["classification"] == "diagnostic_only"
    assert result["proof_eligible"] is False


def test_legacy_v1_manifest_without_output_hashes_remains_diagnostic(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    value["schema"] = "sota_finalization.v1"
    group = value["groups"][0]
    for key in ("stdout_sha256", "stderr_sha256", "transcript_sha256"):
        group.pop(key)
    Path(group["command_record"]).write_text(json.dumps(group))
    save(path, value)

    result = verify_manifest(path)

    assert result["valid"] is True
    assert result["classification"] == "diagnostic_only"
    assert result["proof_eligible"] is False


def test_v2_manifest_rejects_missing_execution_output_hash(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    group = value["groups"][0]
    group.pop("stdout_sha256")
    Path(group["command_record"]).write_text(json.dumps(group))
    save(path, value)

    result = verify_manifest(path)

    assert result["valid"] is False
    assert any("stdout_sha256" in error for error in result["errors"])


def test_proof_grade_rejects_unproven_manifest(tmp_path: Path) -> None:
    result = verify_manifest(make_fixture(tmp_path), proof_grade=True)
    assert result["valid"] is False
    assert any("proof-grade" in e for e in result["errors"])


def test_tampered_proof_status_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    value["proof_status"] = "PROVEN"
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("proof_status" in e for e in result["errors"])


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        (lambda m: m["groups"][0]["shards"][0].update(bytes=0), "byte-size mismatch"),
        (
            lambda m: m["groups"][0]["shards"].append(dict(m["groups"][0]["shards"][0])),
            "duplicate shard",
        ),
        (lambda m: m["groups"][0].update(exit_code=1), "nonzero exit"),
        (lambda m: m["groups"][0].update(command_text="python altered"), "command_text mismatch"),
    ],
)
def test_integrity_failures_are_rejected(tmp_path: Path, mutation, needle: str) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    mutation(value)
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any(needle in e for e in result["errors"])


def test_missing_shard_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    Path(value["groups"][0]["shards"][0]["path"]).unlink()
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("missing shard" in e for e in result["errors"])


def test_receipt_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    receipt = Path(value["groups"][0]["receipt"])
    receipt.write_text(receipt.read_text() + "\n")
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("receipt missing or hash mismatch" in e for e in result["errors"])


def test_partial_receipt_is_diagnostic_warning(tmp_path: Path) -> None:
    path = make_fixture(tmp_path, rows=4)
    value = load(path)
    losses = Path(value["groups"][0]["losses"])
    with np.load(losses, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["crps_matrix"][0, 0] = np.nan
    np.savez(losses, **arrays)
    value["groups"][0]["losses_sha256"] = sha(losses)
    receipt = Path(value["groups"][0]["receipt"])
    data = load(receipt)
    data["losses_sha256"] = sha(losses)
    data["n_complete"] = 3
    data["n_dropped_incomplete"] = 1
    receipt.write_text(json.dumps(data))
    value["groups"][0]["receipt_sha256"] = sha(receipt)
    value["groups"][0]["command_record"] = value["groups"][0]["command_record"]
    Path(value["groups"][0]["command_record"]).write_text(json.dumps(value["groups"][0]))
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is True
    assert any("incomplete rows" in w for w in result["warnings"])


def test_malformed_loss_arrays_are_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    losses = Path(value["groups"][0]["losses"])
    np.savez(
        losses, crps_matrix=np.ones((2, 2)), pinball_cube=np.ones((2, 2)), meta_json=np.array("{}")
    )
    value["groups"][0]["losses_sha256"] = sha(losses)
    receipt = Path(value["groups"][0]["receipt"])
    data = load(receipt)
    data["losses_sha256"] = sha(losses)
    receipt.write_text(json.dumps(data))
    value["groups"][0]["receipt_sha256"] = sha(receipt)
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("loss archive missing arrays" in e for e in result["errors"])


def test_command_record_tampering_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    record = Path(value["groups"][0]["command_record"])
    data = load(record)
    data["command_text"] = "python altered"
    record.write_text(json.dumps(data))
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("command record mismatch" in e for e in result["errors"])


@pytest.mark.parametrize("field", ["stdout", "stderr", "transcript"])
def test_execution_output_tampering_is_rejected(tmp_path: Path, field: str) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    output = Path(value["groups"][0][field])
    output.write_text(output.read_text() + "tampered after execution\n")
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any(f"{field} hash mismatch" in error for error in result["errors"])


def test_receipt_source_map_tampering_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    receipt = Path(value["groups"][0]["receipt"])
    data = load(receipt)
    key, digest = next(iter(data["source_parts_sha256"].items()))
    data["source_parts_sha256"] = {key + ".altered": digest}
    receipt.write_text(json.dumps(data))
    value["groups"][0]["receipt_sha256"] = sha(receipt)
    Path(value["groups"][0]["command_record"]).write_text(json.dumps(value["groups"][0]))
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("source-part paths or hashes" in e for e in result["errors"])


@pytest.mark.parametrize("field", ["receipt", "losses", "transcript"])
def test_run_artifact_escape_is_rejected(tmp_path: Path, field: str) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    outside = tmp_path / ("outside-" + field)
    original = Path(value["groups"][0][field])
    outside.write_bytes(original.read_bytes())
    value["groups"][0][field] = str(outside)
    Path(value["groups"][0]["command_record"]).write_text(json.dumps(value["groups"][0]))
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("escapes manifest run directory" in e for e in result["errors"])


def test_shard_symlink_alias_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    shard = Path(value["groups"][0]["shards"][0]["path"])
    target = tmp_path / "outside-shard.losses.npz"
    target.write_bytes(shard.read_bytes())
    shard.unlink()
    try:
        shard.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("symlink or reparse point" in e for e in result["errors"])


def test_manifest_input_root_escape_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    outside = tmp_path / "outside-input"
    outside.mkdir()
    value["input_dir"] = str(outside)
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("input directory escapes repository" in e for e in result["errors"])


def test_duplicate_group_name_is_rejected(tmp_path: Path) -> None:
    path = make_fixture(tmp_path)
    value = load(path)
    value["groups"].append(dict(value["groups"][0]))
    save(path, value)
    result = verify_manifest(path)
    assert result["valid"] is False
    assert any("duplicate group" in e for e in result["errors"])
