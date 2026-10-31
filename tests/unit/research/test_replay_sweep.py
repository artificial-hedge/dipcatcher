"""Contracts for the corpus-wide replay sweep (``replay_coverage.v1``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from quant_fund.research.replay_sweep import (
    REPLAY_COVERAGE_SCHEMA,
    replay_coverage_contract_errors,
    run_replay_sweep,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _write_self_receipt(tmp_path: Path) -> Path:
    """A lane receipt whose replay writes a deterministic artifact."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "artifact.json"
    artifact.write_text('{"fixed": true}')
    script = tmp_path / "emit.py"
    script.write_text(
        "import json, pathlib\n"
        "pathlib.Path('out/artifact.json').write_text(json.dumps({'fixed': True}))\n"
    )
    receipt = {
        "schema": "synthetic_lane.v1",
        "kind": "synthetic_lane",
        "replay": {
            "argv": [sys.executable, "emit.py"],
            "artifacts": [
                {
                    "path": "out/artifact.json",
                    "sha256": hash_bytes(artifact.read_bytes()),
                }
            ],
        },
    }
    receipt["receipt_sha256"] = hash_bytes(
        canonical_json_bytes({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    )
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def _carrier(tmp_path: Path, receipt: Path, name: str, claims_equal: bool = True) -> Path:
    carriers = tmp_path / "carriers"
    carriers.mkdir(parents=True, exist_ok=True)
    carrier = {
        "schema": "replay_manifest.v1",
        "kind": "replay_manifest",
        "data_label": "SYNTHETIC",
        "replay": json.loads(receipt.read_text())["replay"],
        "reproduces": [
            {
                "receipt": receipt.name,
                "committed_sha256": "ab" * 32,
                "claims_equal": claims_equal,
            }
        ],
    }
    carrier["receipt_sha256"] = hash_bytes(
        canonical_json_bytes({k: v for k, v in carrier.items() if k != "receipt_sha256"})
    )
    path = carriers / f"{name}.json"
    path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
    return path


def test_sweep_all_pass(tmp_path: Path) -> None:
    receipt = _write_self_receipt(tmp_path)
    carriers = tmp_path / "carriers"
    _carrier(tmp_path, receipt, "lane_a")
    _carrier(tmp_path, receipt, "lane_b", claims_equal=False)
    body = run_replay_sweep(carriers, root=tmp_path, timeout_s=60)
    assert body["schema"] == REPLAY_COVERAGE_SCHEMA
    assert body["n_carriers"] == 2
    assert body["n_pass"] == 2
    assert body["n_fail"] == 0
    assert body["verdict"] == "pass"
    assert body["coverage"] == 1.0
    assert body["n_claims_total"] == 2
    assert body["n_claims_byte_equal"] == 1
    assert replay_coverage_contract_errors(body) == []


def test_sweep_forged_artifact_fails(tmp_path: Path) -> None:
    receipt = _write_self_receipt(tmp_path)
    carriers = tmp_path / "carriers"
    carrier_path = _carrier(tmp_path, receipt, "lane_a")
    carrier = json.loads(carrier_path.read_text())
    carrier["replay"]["artifacts"][0]["sha256"] = "00" * 32
    carrier["receipt_sha256"] = hash_bytes(
        canonical_json_bytes({k: v for k, v in carrier.items() if k != "receipt_sha256"})
    )
    carrier_path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
    body = run_replay_sweep(carriers, root=tmp_path, timeout_s=60)
    assert body["n_fail"] == 1
    assert body["verdict"] == "fail"
    assert replay_coverage_contract_errors(body) == []


def test_sweep_empty_dir_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "carriers").mkdir()
    with pytest.raises(ValueError, match="no replay_manifest"):
        run_replay_sweep(tmp_path / "carriers", root=tmp_path)


def test_sweep_missing_dir_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        run_replay_sweep(tmp_path / "nope", root=tmp_path)


def test_contract_rejects_forged_counts(tmp_path: Path) -> None:
    receipt = _write_self_receipt(tmp_path)
    carriers = tmp_path / "carriers"
    _carrier(tmp_path, receipt, "lane_a")
    body = run_replay_sweep(carriers, root=tmp_path, timeout_s=60)
    assert replay_coverage_contract_errors(body) == []
    for field, bad in (
        ("n_pass", 0),
        ("n_fail", 1),
        ("coverage", 0.0),
        ("verdict", "fail"),
    ):
        forged = dict(body)
        forged[field] = bad
        assert replay_coverage_contract_errors(forged) != []
