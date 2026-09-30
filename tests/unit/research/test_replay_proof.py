"""replay_proof — re-execute a lane's declared argv and verify artifact bytes."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.replay_proof import (
    REPLAY_PROOF_SCHEMA,
    replay_manifest,
    replay_proof_contract_errors,
    run_replay,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DECLARED_RECEIPT = REPO_ROOT / "receipts" / "serial_watch_78dd891261e2aae9.json"

# The argv under test writes exactly these bytes; the manifest pins their digest.
_ARTIFACT_CONTENT = '{"ok": true}'
_ARTIFACT_SHA256 = hashlib.sha256(_ARTIFACT_CONTENT.encode()).hexdigest()
_ARTIFACT_REL = "out/result.json"
_WRITE_ARTIFACT_ARGV = [
    sys.executable,
    "-c",
    (
        "from pathlib import Path\n"
        "p = Path('out')\n"
        "p.mkdir(parents=True, exist_ok=True)\n"
        f"(p / 'result.json').write_text({_ARTIFACT_CONTENT!r}, encoding='utf-8')\n"
    ),
]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(root: Path, body: dict[str, Any], *, name: str = "receipt.json") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def _receipt_with_manifest(
    root: Path,
    argv: list[str],
    artifacts: list[dict[str, str]],
    *,
    name: str = "receipt.json",
) -> Path:
    body = {
        "kind": "demo.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "replay": {"argv": argv, "artifacts": artifacts},
    }
    return _write_receipt(root, body, name=name)


def test_manifest_absent_is_neutral(tmp_path: Path) -> None:
    """No ``replay`` block → ``None`` — the lane is simply not replay-declared."""
    receipt = _write_receipt(tmp_path, {"kind": "demo.v1", "data_label": "SYNTHETIC"})
    assert replay_manifest(receipt) is None


def test_manifest_reads_v2_wrapped_payload(tmp_path: Path) -> None:
    """A manifest inside a ``receipt.v2`` envelope's ``payload`` is found."""
    inner = {
        "kind": "demo.v1",
        "replay": {
            "argv": ["dipcatcher", "demo"],
            "artifacts": [{"path": "out.json", "sha256": "a" * 64}],
        },
    }
    receipt = _write_receipt(
        tmp_path, {"schema": "receipt.v2", "payload": inner}, name="wrapped.json"
    )
    manifest = replay_manifest(receipt)
    assert manifest is not None
    assert manifest["argv"] == ["dipcatcher", "demo"]


@pytest.mark.parametrize(
    "bad",
    [
        "nope",
        {"argv": []},
        {"argv": ["dipcatcher", "demo"]},
        {"argv": ["dipcatcher", "demo"], "artifacts": []},
        {"argv": ["dipcatcher", "demo"], "artifacts": [{"path": "x", "sha256": "zz"}]},
        {
            "argv": ["dipcatcher", "demo"],
            "artifacts": [{"path": "x", "sha256": "a" * 64}],
            "cwd": 3,
        },
    ],
)
def test_manifest_malformed_raises(tmp_path: Path, bad: object) -> None:
    """A present-but-broken manifest is a broken declaration — fail closed."""
    receipt = _write_receipt(tmp_path, {"kind": "demo.v1", "replay": bad})
    with pytest.raises(ValueError, match="malformed"):
        replay_manifest(receipt)


def test_run_replay_passes_and_reexecutes(tmp_path: Path) -> None:
    """argv actually runs: the artifact does not exist until the replay."""
    artifact = tmp_path / _ARTIFACT_REL
    assert not artifact.exists()
    receipt = _receipt_with_manifest(
        tmp_path,
        _WRITE_ARTIFACT_ARGV,
        [{"path": _ARTIFACT_REL, "sha256": _ARTIFACT_SHA256}],
    )
    body = run_replay(receipt, root=tmp_path)
    assert body["schema"] == REPLAY_PROOF_SCHEMA
    assert artifact.is_file()  # argv produced the declared artifact
    assert body["exit_code"] == 0
    assert body["all_match"] is True
    assert body["verdict"] == "pass"
    row = body["artifacts"][0]
    assert row["match"] is True
    assert row["observed_sha256"] == _file_sha256(artifact)
    assert replay_proof_contract_errors(body) == []


def test_forged_artifact_digest_fails(tmp_path: Path) -> None:
    """Pinned digest ≠ produced bytes → fail, honestly reported."""
    receipt = _receipt_with_manifest(
        tmp_path,
        _WRITE_ARTIFACT_ARGV,
        [{"path": _ARTIFACT_REL, "sha256": "0" * 64}],
    )
    body = run_replay(receipt, root=tmp_path)
    assert body["artifacts"][0]["match"] is False
    assert body["artifacts"][0]["observed_sha256"] == _ARTIFACT_SHA256
    assert body["all_match"] is False
    assert body["verdict"] == "fail"
    assert replay_proof_contract_errors(body) == []


def test_missing_artifact_fails(tmp_path: Path) -> None:
    """A declared artifact the lane never writes → fail."""
    receipt = _receipt_with_manifest(
        tmp_path,
        _WRITE_ARTIFACT_ARGV,
        [{"path": "out/never_written.json", "sha256": "f" * 64}],
    )
    body = run_replay(receipt, root=tmp_path)
    row = body["artifacts"][0]
    assert row["observed_sha256"] is None
    assert row["note"] == "artifact_missing"
    assert row["match"] is False
    assert body["verdict"] == "fail"
    assert replay_proof_contract_errors(body) == []


def test_nonzero_exit_fails(tmp_path: Path) -> None:
    """The lane exits non-zero → fail even though artifact bytes match."""
    artifact = tmp_path / _ARTIFACT_REL
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(_ARTIFACT_CONTENT, encoding="utf-8")
    argv = [sys.executable, "-c", "import sys; sys.exit(3)"]
    receipt = _receipt_with_manifest(
        tmp_path, argv, [{"path": _ARTIFACT_REL, "sha256": _ARTIFACT_SHA256}]
    )
    body = run_replay(receipt, root=tmp_path)
    assert body["exit_code"] == 3
    assert body["artifacts"][0]["match"] is True  # bytes matched; the exit code still fails it
    assert body["all_match"] is False
    assert body["verdict"] == "fail"
    assert replay_proof_contract_errors(body) == []


def test_timeout_fails(tmp_path: Path) -> None:
    """A lane that outruns its bound fails closed rather than hanging."""
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    receipt = _receipt_with_manifest(tmp_path, argv, [{"path": "x", "sha256": "f" * 64}])
    started = time.monotonic()
    body = run_replay(receipt, root=tmp_path, timeout_s=0.5)
    assert time.monotonic() - started < 15
    assert body["timed_out"] is True
    assert body["exit_code"] is None
    assert body["verdict"] == "fail"


def test_artifact_escape_fails(tmp_path: Path) -> None:
    """An artifact path escaping ``root`` fails rather than reading outside it."""
    outside = (tmp_path / ".." / "escape_target.json").resolve()
    outside.write_text(_ARTIFACT_CONTENT, encoding="utf-8")
    receipt = _receipt_with_manifest(
        tmp_path,
        [sys.executable, "-c", "pass"],
        [{"path": str(outside), "sha256": _ARTIFACT_SHA256}],
    )
    body = run_replay(receipt, root=tmp_path)
    row = body["artifacts"][0]
    assert row["match"] is False
    assert row["note"] == "path_escapes_root"
    assert row["observed_sha256"] is None
    assert body["verdict"] == "fail"


def test_undeclared_receipt_raises(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path, {"kind": "demo.v1"})
    with pytest.raises(ValueError, match="not replay-declared"):
        run_replay(receipt, root=tmp_path)


def test_contract_round_trip_and_lies(tmp_path: Path) -> None:
    """The contract re-derives every claimed equivalence — forged flags fail."""
    receipt = _receipt_with_manifest(
        tmp_path,
        _WRITE_ARTIFACT_ARGV,
        [{"path": _ARTIFACT_REL, "sha256": _ARTIFACT_SHA256}],
    )
    body = run_replay(receipt, root=tmp_path)
    assert replay_proof_contract_errors(body) == []

    forged_verdict = dict(body, verdict="fail")
    assert "verdict" in replay_proof_contract_errors(forged_verdict)

    forged_all = dict(body, all_match=False)
    assert "all_match" in replay_proof_contract_errors(forged_all)

    forged_row = dict(body, artifacts=[dict(body["artifacts"][0], match=False)])
    assert "artifacts[0].match" in replay_proof_contract_errors(forged_row)

    forged_seal = dict(body, receipt_sha256="d" * 64)
    assert "receipt_sha256" in replay_proof_contract_errors(forged_seal)

    missing_exit = dict(body, exit_code=None)
    assert "all_match" in replay_proof_contract_errors(missing_exit)


def test_committed_serial_watch_lane_replays(tmp_path: Path) -> None:
    """End-to-end: the committed replay-declared serial_watch receipt passes.

    The receipt and its committed PIT fixture are mirrored into a scratch
    root so the replayed ``dipcatcher`` argv writes nowhere near the real
    receipts dir; argv resolves to ``sys.executable -m quant_fund.cli.main``.
    """
    if not DECLARED_RECEIPT.is_file():
        pytest.skip("declared serial_watch receipt not committed on this branch")
    source = json.loads(DECLARED_RECEIPT.read_text())
    manifest = replay_manifest(DECLARED_RECEIPT)
    assert manifest is not None

    fixture_src = REPO_ROOT / "tests" / "fixtures" / "replay" / "serial_pits.json"
    fixture_dst = tmp_path / "tests" / "fixtures" / "replay" / "serial_pits.json"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    fixture_dst.write_bytes(fixture_src.read_bytes())
    receipt = _write_receipt(tmp_path / "receipts", source, name=DECLARED_RECEIPT.name)

    body = run_replay(receipt, root=tmp_path, timeout_s=60.0)
    # The proof is bound to the source receipt's bytes and the seal it stamped.
    assert body["source_receipt_sha256"] == _file_sha256(receipt)
    assert body["source_receipt_sha256"] == _file_sha256(DECLARED_RECEIPT)
    assert body["source_receipt_seal"] == source["receipt_sha256"]
    assert body["argv"] == manifest["argv"]
    assert body["exit_code"] == 0
    assert body["all_match"] is True
    assert body["verdict"] == "pass"
    assert replay_proof_contract_errors(body) == []


# --- committed replay-declared lanes (beyond serial-watch) ----------------


def _committed_lane_receipts() -> list[Path]:
    """Committed receipts carrying a valid ``replay`` manifest, minus the
    serial-watch receipt already covered by ``test_committed_serial_watch_lane_replays``."""
    receipts_dir = REPO_ROOT / "receipts"
    if not receipts_dir.is_dir():
        return []
    found: list[Path] = []
    for candidate in sorted(receipts_dir.glob("*.json")):
        if candidate.name == DECLARED_RECEIPT.name:
            continue
        try:
            if replay_manifest(candidate) is not None:
                found.append(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return found


def _mirror_lane_run_inputs(root: Path) -> None:
    """Mirror the repo inputs every lane argv reads: ``configs/`` wholesale
    (research.yaml composes base.yaml; fleet/vol-bench/verdict/fleet-monitor
    resolve it relative to cwd) — lanes that do not read it are unaffected."""
    src_dir = REPO_ROOT / "configs"
    if not src_dir.is_dir():
        return
    dst_dir = root / "configs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            (dst_dir / src.name).write_bytes(src.read_bytes())


@pytest.mark.parametrize(
    "declared",
    _committed_lane_receipts(),
    ids=lambda path: path.name.removesuffix(".json"),
)
def test_committed_lane_replays(tmp_path: Path, declared: Path) -> None:
    """Each committed replay-declared lane re-executes argv and reproduces
    its pinned artifact bytes end-to-end."""
    source = json.loads(declared.read_text())
    manifest = replay_manifest(declared)
    assert manifest is not None

    _mirror_lane_run_inputs(tmp_path)
    receipt = _write_receipt(tmp_path / "receipts", source, name=declared.name)

    body = run_replay(receipt, root=tmp_path, timeout_s=180.0)
    assert body["source_receipt_sha256"] == _file_sha256(receipt)
    assert body["source_receipt_sha256"] == _file_sha256(declared)
    assert body["source_receipt_seal"] == source["receipt_sha256"]
    assert body["argv"] == manifest["argv"]
    assert body["exit_code"] == 0
    assert body["all_match"] is True
    assert body["verdict"] == "pass"
    assert replay_proof_contract_errors(body) == []


@pytest.mark.parametrize(
    "declared",
    _committed_lane_receipts(),
    ids=lambda path: path.name.removesuffix(".json"),
)
def test_committed_lane_forged_digest_fails(tmp_path: Path, declared: Path) -> None:
    """Pinning a wrong artifact digest in an otherwise valid manifest fails
    closed: argv still runs to completion but the comparison reports fail."""
    source = json.loads(declared.read_text())
    manifest = source.get("replay") or (source.get("payload") or {}).get("replay")
    assert isinstance(manifest, dict) and manifest["artifacts"]
    forged = json.loads(json.dumps(source))
    forged_manifest = forged.get("replay") or forged["payload"]["replay"]
    forged_manifest["artifacts"][0]["sha256"] = "0" * 64

    _mirror_lane_run_inputs(tmp_path)
    receipt = _write_receipt(tmp_path / "receipts", forged, name=declared.name)

    body = run_replay(receipt, root=tmp_path, timeout_s=180.0)
    assert body["exit_code"] == 0
    assert body["all_match"] is False
    assert body["verdict"] == "fail"
    assert replay_proof_contract_errors(body) == []


def _tape_manifest(root: Path, tapes: list[dict[str, Any]], name: str = "tape.json") -> Path:
    """Write a seal-valid tape_manifest.v1 into root/data/manifests."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    body = {
        "schema": "tape_manifest.v1",
        "source_label": name.removesuffix(".json"),
        "tape_files": tapes,
        "frame_csv_sha256": "f" * 64,
        "n_rows": 1,
        "n_names": 1,
        "window": None,
        "collected_at": "2026-01-01T00:00:00+00:00",
        "promotion_receipt_sha256": None,
    }
    manifest = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    return _write_receipt(root / "data" / "manifests", manifest, name=name)


def _with_tapes(root: Path, tapes: list[dict[str, Any]], *, name: str = "receipt.json") -> Path:
    body = {
        "kind": "demo.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "replay": {
            "argv": _WRITE_ARTIFACT_ARGV,
            "artifacts": [{"path": _ARTIFACT_REL, "sha256": _ARTIFACT_SHA256}],
            "input_tapes": tapes,
        },
    }
    return _write_receipt(root, body, name=name)


def test_input_tapes_verified_then_executes(tmp_path: Path) -> None:
    """A literal path pin that matches executes argv and passes."""
    tape = tmp_path / "data" / "raw" / "bars.parquet"
    tape.parent.mkdir(parents=True)
    tape.write_bytes(b"tape-bytes")
    receipt = _with_tapes(
        tmp_path, [{"path": "data/raw/bars.parquet", "sha256": _file_sha256(tape)}]
    )
    body = run_replay(receipt, root=tmp_path)
    assert body["inputs_ok"] is True
    assert body["input_tapes"][0]["match"] is True
    assert body["exit_code"] == 0 and body["verdict"] == "pass"
    assert replay_proof_contract_errors(body) == []


def test_input_tape_drift_blocks_execution(tmp_path: Path) -> None:
    """A tampered tape never reaches the lane — no subprocess spawns."""
    tape = tmp_path / "data" / "raw" / "bars.parquet"
    tape.parent.mkdir(parents=True)
    tape.write_bytes(b"tape-bytes")
    receipt = _with_tapes(tmp_path, [{"path": "data/raw/bars.parquet", "sha256": "0" * 64}])
    body = run_replay(receipt, root=tmp_path)
    assert body["inputs_ok"] is False
    assert body["input_tapes"][0]["note"] == "input_tape_drift"
    assert body["execution_skipped"] == "inputs_not_verified"
    assert body["exit_code"] is None and body["all_match"] is False
    assert body["verdict"] == "fail"
    assert not (tmp_path / _ARTIFACT_REL).exists()  # argv never ran
    assert replay_proof_contract_errors(body) == []


def test_input_tape_missing_blocks_execution(tmp_path: Path) -> None:
    receipt = _with_tapes(tmp_path, [{"path": "data/raw/gone.parquet", "sha256": "a" * 64}])
    body = run_replay(receipt, root=tmp_path)
    assert body["inputs_ok"] is False
    assert body["input_tapes"][0]["note"] == "input_tape_missing"
    assert body["verdict"] == "fail"
    assert replay_proof_contract_errors(body) == []


def test_input_tapes_manifest_expansion(tmp_path: Path) -> None:
    """A {manifest} entry expands to the committed manifest's tape_files."""
    tape = tmp_path / "data" / "raw" / "yahoo_eod.parquet"
    tape.parent.mkdir(parents=True)
    tape.write_bytes(b"bars")
    _tape_manifest(
        tmp_path,
        [
            {
                "path": "data/raw/yahoo_eod.parquet",
                "sha256": _file_sha256(tape),
                "n_bytes": 4,
            }
        ],
        name="yahoo_eod.json",
    )
    receipt = _with_tapes(tmp_path, [{"manifest": "data/manifests/yahoo_eod.json"}])
    body = run_replay(receipt, root=tmp_path)
    assert body["inputs_ok"] is True
    assert body["input_tapes"][0]["via_manifest"] == "data/manifests/yahoo_eod.json"
    assert body["exit_code"] == 0 and body["verdict"] == "pass"
    assert replay_proof_contract_errors(body) == []


def test_input_tapes_manifest_invalid_fails(tmp_path: Path) -> None:
    """A manifest reference whose seal does not verify fails closed."""
    tape = tmp_path / "data" / "raw" / "t.parquet"
    tape.parent.mkdir(parents=True)
    tape.write_bytes(b"x")
    path = _tape_manifest(
        tmp_path,
        [{"path": "data/raw/t.parquet", "sha256": _file_sha256(tape), "n_bytes": 1}],
        name="t.json",
    )
    # Corrupt the committed manifest AFTER sealing.
    doc = json.loads(path.read_text())
    doc["n_rows"] = 999
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    receipt = _with_tapes(tmp_path, [{"manifest": "data/manifests/t.json"}])
    body = run_replay(receipt, root=tmp_path)
    assert body["inputs_ok"] is False
    assert body["input_tapes"][0]["note"] == "manifest_invalid"
    assert body["verdict"] == "fail"
    assert replay_proof_contract_errors(body) == []


def test_input_tapes_manifest_missing_fails(tmp_path: Path) -> None:
    receipt = _with_tapes(tmp_path, [{"manifest": "data/manifests/nope.json"}])
    body = run_replay(receipt, root=tmp_path)
    assert body["input_tapes"][0]["note"] == "manifest_missing"
    assert body["verdict"] == "fail"


def test_input_tapes_malformed_rejected() -> None:
    from quant_fund.research.replay_proof import replay_manifest_errors

    base = {"argv": ["x"], "artifacts": [{"path": "p", "sha256": "a" * 64}]}
    for tapes in (
        "nope",
        [42],
        [{"path": "p"}],  # pin without sha
        [{"path": "p", "sha256": "a" * 64, "manifest": "m"}],  # both
        [{}],  # neither
    ):
        errors = replay_manifest_errors({**base, "input_tapes": tapes})
        assert errors and all("input_tapes" in e for e in errors), (tapes, errors)


def test_contract_catches_inputs_ok_lie(tmp_path: Path) -> None:
    """A body claiming inputs_ok while a tape row fails is caught."""
    tape = tmp_path / "data" / "raw" / "bars.parquet"
    tape.parent.mkdir(parents=True)
    tape.write_bytes(b"tape-bytes")
    receipt = _with_tapes(
        tmp_path, [{"path": "data/raw/bars.parquet", "sha256": _file_sha256(tape)}]
    )
    body = run_replay(receipt, root=tmp_path)
    assert body["verdict"] == "pass"
    forged = {k: v for k, v in body.items() if k != "receipt_sha256"}
    forged["input_tapes"] = [
        {**row, "match": False, "note": "input_tape_drift"} for row in body["input_tapes"]
    ]
    # inputs_ok still claims True — the re-derivation must catch it.
    errors = replay_proof_contract_errors(forged)
    assert "inputs_ok" in errors or any(e == "input_tapes[0].match" for e in errors)
