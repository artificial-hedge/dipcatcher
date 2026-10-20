"""Headline claim counts ignore carriers that never ran."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from quant_fund.research.replay_sweep import replay_coverage_contract_errors, run_replay_sweep
from quant_fund.utils.receipt import seal_receipt


def _lane(root: Path) -> dict[str, Any]:
    return seal_receipt(
        {
            "schema": "headline_fixture.v1",
            "kind": "headline_fixture",
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "claim": "fixture",
        }
    )


def _write(path: Path, body: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def _git_track(root: Path, *paths: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "ledger@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Ledger"], cwd=root, check=True)
    for path in paths:
        subprocess.run(["git", "add", str(path.relative_to(root))], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)


def _carrier(
    root: Path,
    name: str,
    receipt_rel: str,
    digest: str,
    *,
    skip: bool,
) -> None:
    replay: dict[str, Any] = {
        "argv": ["emit.py"],
        "artifacts": [{"path": "out/artifact.json", "sha256": "ab" * 32}],
    }
    if skip:
        replay["input_tapes"] = [{"path": "data/raw/absent.parquet", "sha256": "cd" * 32}]
    else:
        artifact = b'{"fixed": true}\n'
        replay["artifacts"] = [
            {
                "path": "out/artifact.json",
                "sha256": hashlib.sha256(artifact).hexdigest(),
            }
        ]
        (root / "emit.py").write_text(
            "import pathlib\n"
            "path = pathlib.Path('out')\n"
            "path.mkdir(parents=True, exist_ok=True)\n"
            "(path / 'artifact.json').write_bytes(b'{\"fixed\": true}\\n')\n"
        )
    body = seal_receipt(
        {
            "schema": "replay_manifest.v1",
            "kind": "replay_manifest",
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "replay": replay,
            "reproduces": [
                {
                    "receipt": receipt_rel,
                    "committed_sha256": digest,
                    "claims_equal": True,
                }
            ],
        }
    )
    _write(root / "carriers" / f"{name}.json", body)


def test_fourteen_skipped_carriers_are_not_headline_equals(tmp_path: Path) -> None:
    receipt_rel = "receipts/lane.json"
    data = _write(tmp_path / receipt_rel, _lane(tmp_path))
    digest = hashlib.sha256(data).hexdigest()
    _carrier(tmp_path, "ran", receipt_rel, digest, skip=False)
    for index in range(14):
        _carrier(tmp_path, f"skip_{index:02d}", receipt_rel, digest, skip=True)
    _git_track(
        tmp_path,
        tmp_path / "emit.py",
        tmp_path / receipt_rel,
        *(tmp_path / "carriers").glob("*.json"),
    )
    body = run_replay_sweep(tmp_path / "carriers", root=tmp_path, timeout_s=60)
    assert body["n_skipped"] == 14
    assert body["n_pass"] == 1
    assert body["n_claims_byte_equal"] == 1
    assert body["n_claims_total"] == 1
    assert body["n_claims_skipped"] == 14
    assert body["claims_byte_equal_frac"] == 1.0
    assert replay_coverage_contract_errors(body) == []
    forged = dict(body)
    forged["n_claims_byte_equal"] = 15
    forged["n_claims_total"] = 15
    forged["n_claims_skipped"] = 0
    forged["claims_byte_equal_frac"] = 1.0
    assert "n_claims_byte_equal" in replay_coverage_contract_errors(forged)
    legacy = dict(body)
    legacy.pop("n_claims_skipped")
    legacy["n_claims_total"] = 15
    legacy["n_claims_byte_equal"] = 15
    legacy["claims_byte_equal_frac"] = 1.0
    assert replay_coverage_contract_errors(legacy) == []
