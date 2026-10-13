"""Replay must bind to the live corpus: a matched replay over drifted evidence is not a pass.

Every case runs the production functions end to end — ``run_replay_sweep``
executes a real subprocess (``emit.py`` resolved through ``_resolve_argv``)
in a scratch repo root, and receipts are checked with the production
verifier (``verify_receipt_file`` / ``verify_receipt_payload``). Nothing is
mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

import quant_fund.research.replay_sweep as replay_sweep_mod
from quant_fund.research.receipt_v2 import (
    seal_receipt as seal_v2,
)
from quant_fund.research.receipt_v2 import (
    verify_receipt_file,
    verify_receipt_payload,
    wrap_receipt_v2,
)
from quant_fund.research.replay_proof import replay_proof_contract_errors, run_replay
from quant_fund.research.replay_sweep import replay_coverage_contract_errors, run_replay_sweep
from quant_fund.utils.receipt import seal_receipt

REPO_ROOT = Path(__file__).resolve().parents[3]

_ARTIFACT_BYTES = b'{"fixed": true}'
_ARTIFACT_REL = "out/artifact.json"
_RECEIPT_REL = "receipts/lane.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(path: Path, body: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


def _lane_receipt(claim: str = "fixture") -> dict[str, Any]:
    """A minimal SYNTHETIC v1 receipt the production verifier accepts."""
    return seal_receipt(
        {
            "schema": "binding_fixture.v1",
            "kind": "binding_fixture",
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "claim": claim,
        }
    )


def _write_carrier(
    root: Path,
    *,
    reproduces: list[dict[str, Any]] | None,
    name: str = "lane",
    input_tapes: list[dict[str, Any]] | None = None,
    seal: bool = True,
) -> Path:
    replay: dict[str, Any] = {
        # A bare ``*.py`` argv resolves to this interpreter (``_resolve_argv``).
        "argv": ["emit.py"],
        "artifacts": [{"path": _ARTIFACT_REL, "sha256": _sha(_ARTIFACT_BYTES)}],
    }
    if input_tapes is not None:
        replay["input_tapes"] = input_tapes
    body: dict[str, Any] = {
        "schema": "replay_manifest.v1",
        "kind": "replay_manifest",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "replay": replay,
    }
    if reproduces is not None:
        body["reproduces"] = reproduces
    if seal:
        body = seal_receipt(body)
    path = root / "carriers" / f"{name}.json"
    _dump(path, body)
    return path


def _scratch_repo(root: Path) -> str:
    """Write the lane script and a sealed receipt; return the receipt's live digest."""
    (root / "emit.py").write_text(
        "import pathlib\n"
        "out = pathlib.Path('out')\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        f"(out / 'artifact.json').write_bytes({_ARTIFACT_BYTES!r})\n",
        encoding="utf-8",
    )
    data = _dump(root / _RECEIPT_REL, _lane_receipt())
    assert verify_receipt_file(root / _RECEIPT_REL)["valid"], "fixture receipt must verify"
    return _sha(data)


def _entry(receipt: str, committed_sha256: str) -> dict[str, Any]:
    return {"receipt": receipt, "committed_sha256": committed_sha256, "claims_equal": True}


def _sweep(root: Path) -> dict[str, Any]:
    return run_replay_sweep(root / "carriers", root=root, timeout_s=60)


def _sealed_coverage(body: dict[str, Any]) -> dict[str, Any]:
    """Seal a coverage body exactly as ``dipcatcher replay-all`` does."""
    return seal_v2(
        wrap_receipt_v2(
            body,
            code_files=(Path(replay_sweep_mod.__file__),),
            verdict=body["verdict"],
            dataset={"carriers_dir": body["carriers_dir"]},
            params={"timeout_s": body["timeout_s"]},
        )
    )


# --- (a) positive control ----------------------------------------------------


def test_bound_replay_passes(tmp_path: Path) -> None:
    live = _scratch_repo(tmp_path)
    _write_carrier(tmp_path, reproduces=[_entry(_RECEIPT_REL, live)])
    body = _sweep(tmp_path)
    row = body["rows"][0]
    assert row["verdict"] == "pass"
    assert row["execution_verdict"] == "pass"
    assert row["n_reproduces"] == 1
    assert row["binding_errors"] == []
    record = row["reproduces"][0]
    assert record["bound"] is True
    assert record["live_sha256"] == record["committed_sha256"] == live
    assert record["receipt_valid"] is True
    assert (tmp_path / _ARTIFACT_REL).read_bytes() == _ARTIFACT_BYTES  # argv really ran
    assert body["n_pass"] == 1 and body["verdict"] == "pass"
    assert body["n_binding_errors"] == 0
    assert body["n_reproduces_bound"] == 1
    assert replay_coverage_contract_errors(body) == []
    sealed = _sealed_coverage(body)
    assert verify_receipt_payload(sealed)["valid"], verify_receipt_payload(sealed)["errors"]


@pytest.mark.parametrize("reproduces", [[], None], ids=["empty", "absent"])
def test_carrier_without_reproduces_passes_promoting_nothing(
    tmp_path: Path, reproduces: list[dict[str, Any]] | None
) -> None:
    _scratch_repo(tmp_path)
    _write_carrier(tmp_path, reproduces=reproduces)
    body = _sweep(tmp_path)
    row = body["rows"][0]
    assert row["verdict"] == "pass"
    assert row["n_reproduces"] == 0 and row["reproduces"] == []
    assert body["n_reproduces_bound"] == 0
    assert replay_coverage_contract_errors(body) == []


# --- (b)-(e) execution matched, binding fails --------------------------------


def _assert_binding_fail(body: dict[str, Any], expected_error: str) -> dict[str, Any]:
    row = body["rows"][0]
    assert row["execution_verdict"] == "pass"  # the replay itself reproduced the artifact
    assert row["all_match"] is True and row["exit_code"] == 0
    assert row["verdict"] == "fail"
    assert expected_error in row["binding_errors"]
    assert expected_error in row["note"]
    assert body["n_pass"] == 0 and body["n_fail"] == 1 and body["verdict"] == "fail"
    assert body["n_binding_errors"] == len(row["binding_errors"]) >= 1
    assert replay_coverage_contract_errors(body) == []
    return row


def test_drifted_live_receipt_fails(tmp_path: Path) -> None:
    pinned = _scratch_repo(tmp_path)
    _write_carrier(tmp_path, reproduces=[_entry(_RECEIPT_REL, pinned)])
    # A later rewrite: still a valid, sealed receipt — just not the reproduced bytes.
    _dump(tmp_path / _RECEIPT_REL, _lane_receipt(claim="rewritten later"))
    assert verify_receipt_file(tmp_path / _RECEIPT_REL)["valid"]
    row = _assert_binding_fail(_sweep(tmp_path), f"reproduced_receipt_drift:{_RECEIPT_REL}")
    record = row["reproduces"][0]
    assert record["bound"] is False
    assert record["committed_sha256"] == pinned
    assert record["live_sha256"] != pinned
    assert record["receipt_valid"] is True


def test_missing_live_receipt_fails(tmp_path: Path) -> None:
    pinned = _scratch_repo(tmp_path)
    _write_carrier(tmp_path, reproduces=[_entry(_RECEIPT_REL, pinned)])
    (tmp_path / _RECEIPT_REL).unlink()
    row = _assert_binding_fail(_sweep(tmp_path), f"reproduced_receipt_missing:{_RECEIPT_REL}")
    assert row["reproduces"][0]["live_sha256"] is None


def test_digest_equal_but_invalid_receipt_fails(tmp_path: Path) -> None:
    _scratch_repo(tmp_path)
    tampered = _lane_receipt()
    tampered["claim"] = "edited after sealing"  # seal no longer binds the body
    data = _dump(tmp_path / _RECEIPT_REL, tampered)
    assert not verify_receipt_file(tmp_path / _RECEIPT_REL)["valid"]
    _write_carrier(tmp_path, reproduces=[_entry(_RECEIPT_REL, _sha(data))])
    row = _assert_binding_fail(_sweep(tmp_path), f"reproduced_receipt_invalid:{_RECEIPT_REL}")
    record = row["reproduces"][0]
    assert record["live_sha256"] == record["committed_sha256"]  # digest matched
    assert record["receipt_valid"] is False
    assert "receipt_sha256_mismatch" in record["receipt_errors"]
    assert not any(e.startswith("reproduced_receipt_drift") for e in row["binding_errors"])


@pytest.mark.parametrize("kind", ["dotdot", "absolute", "symlink"])
def test_reproduced_path_escape_fails(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _scratch_repo(root)
    # The escaped target exists, verifies, and its digest matches the pin —
    # only the path discipline can reject it.
    outside = tmp_path / "outside.json"
    data = _dump(outside, _lane_receipt())
    if kind == "dotdot":
        declared = "../outside.json"
    elif kind == "absolute":
        declared = str(outside)
    else:
        if not hasattr(os, "symlink"):
            pytest.skip("no symlink support")
        (root / "receipts" / "link.json").symlink_to(outside)
        declared = "receipts/link.json"
    _write_carrier(root, reproduces=[_entry(declared, _sha(data))])
    row = _assert_binding_fail(_sweep(root), f"reproduced_receipt_path_invalid:{declared}")
    assert row["reproduces"][0]["live_sha256"] is None


# --- (f) skipped rows stay skipped but show binding errors -------------------


def test_skipped_carrier_records_binding_errors(tmp_path: Path) -> None:
    pinned = _scratch_repo(tmp_path)
    _write_carrier(
        tmp_path,
        reproduces=[_entry(_RECEIPT_REL, pinned)],
        input_tapes=[{"path": "data/raw/absent.parquet", "sha256": "a" * 64}],
    )
    _dump(tmp_path / _RECEIPT_REL, _lane_receipt(claim="drifted"))
    body = _sweep(tmp_path)
    row = body["rows"][0]
    assert row["verdict"] == "skipped"
    assert row["skipped_reason"] == "inputs_not_verified"
    assert row["binding_errors"] == [f"reproduced_receipt_drift:{_RECEIPT_REL}"]
    assert not (tmp_path / _ARTIFACT_REL).exists()  # argv never ran
    assert body["n_skipped"] == 1 and body["n_pass"] == 0
    assert body["n_binding_errors"] == 1
    assert body["verdict"] == "fail"  # an all-skip sweep promotes nothing
    assert replay_coverage_contract_errors(body) == []


def test_unsealed_carrier_never_executes(tmp_path: Path) -> None:
    live = _scratch_repo(tmp_path)
    _write_carrier(tmp_path, reproduces=[_entry(_RECEIPT_REL, live)], seal=False)
    body = _sweep(tmp_path)
    row = body["rows"][0]
    assert row["verdict"] == "fail"
    assert row["note"].startswith("carrier_invalid:")
    assert not (tmp_path / _ARTIFACT_REL).exists()
    assert replay_coverage_contract_errors(body) == []


# --- (g) forged replay_proof.v1 bodies fail the production verifier ---------


def _failing_proof(root: Path) -> dict[str, Any]:
    lane = {
        "kind": "demo.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "replay": {
            "argv": [sys.executable, "-c", "import sys; sys.exit(3)"],
            "artifacts": [{"path": _ARTIFACT_REL, "sha256": _sha(_ARTIFACT_BYTES)}],
        },
    }
    _dump(root / "lane.json", lane)
    body = run_replay(root / "lane.json", root=root, timeout_s=60)
    assert body["verdict"] == "fail" and body["exit_code"] == 3
    return body


def _wrap_proof(body: dict[str, Any]) -> dict[str, Any]:
    from quant_fund.research import replay_proof

    return seal_v2(
        wrap_receipt_v2(
            body,
            code_files=(Path(replay_proof.__file__),),
            verdict=body["verdict"],
            dataset={
                "receipt": body["receipt"],
                "source_receipt_sha256": body["source_receipt_sha256"],
            },
            params={"argv": body["argv"], "timeout_s": body["timeout_s"]},
        )
    )


def test_honest_replay_proof_verifies(tmp_path: Path) -> None:
    body = _failing_proof(tmp_path)
    assert verify_receipt_payload(body)["valid"], verify_receipt_payload(body)["errors"]
    wrapped = _wrap_proof(body)
    assert verify_receipt_payload(wrapped)["valid"], verify_receipt_payload(wrapped)["errors"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"verdict": "pass", "all_match": True}, "all_match"),
        ({"verdict": "pass", "all_match": True, "exit_code": 0, "spawn_error": "x"}, "exit_code"),
    ],
    ids=["flip_verdict", "spawn_error_with_exit"],
)
def test_forged_replay_proof_rejected(
    tmp_path: Path, overrides: dict[str, Any], expected: str
) -> None:
    honest = _failing_proof(tmp_path)
    forged = seal_receipt({**honest, **overrides})  # fresh, honest seal over a lie
    assert expected in replay_proof_contract_errors(forged)
    v1 = verify_receipt_payload(forged)
    assert not v1["valid"] and expected in v1["errors"]
    path = tmp_path / "forged_proof.json"
    _dump(path, forged)
    assert not verify_receipt_file(path)["valid"]
    v2 = verify_receipt_payload(_wrap_proof(forged))
    assert not v2["valid"] and expected in v2["errors"]


def _committed(pattern: str) -> list[Path]:
    receipts = REPO_ROOT / "receipts"
    return sorted(receipts.glob(pattern)) if receipts.is_dir() else []


@pytest.mark.parametrize(
    "path",
    _committed("replay_proof_*.json") + _committed("replay_coverage_*.json"),
    ids=lambda p: p.name,
)
def test_committed_replay_receipts_survive_contract_dispatch(path: Path) -> None:
    verification = verify_receipt_file(path)
    assert verification["valid"], verification["errors"]


# --- (h) coverage contract re-derives binding arithmetic --------------------


def _drift_body(root: Path) -> dict[str, Any]:
    pinned = _scratch_repo(root)
    _write_carrier(root, reproduces=[_entry(_RECEIPT_REL, pinned)])
    _dump(root / _RECEIPT_REL, _lane_receipt(claim="rewritten later"))
    body = _sweep(root)
    assert body["rows"][0]["verdict"] == "fail"
    return body


def _promote_first_row(body: dict[str, Any]) -> dict[str, Any]:
    """Relabel row 0 as pass with every headline count re-derived consistently."""
    forged = json.loads(json.dumps(body))
    forged["rows"][0]["verdict"] = "pass"
    forged["rows"][0].pop("note", None)
    forged["n_pass"], forged["n_fail"] = 1, 0
    forged["coverage"], forged["verdict"] = 1.0, "pass"
    return forged


def test_contract_rejects_pass_row_with_binding_errors(tmp_path: Path) -> None:
    forged = _promote_first_row(_drift_body(tmp_path))
    errors = replay_coverage_contract_errors(forged)
    assert "rows[0].pass_with_binding_errors" in errors
    verification = verify_receipt_payload(_sealed_coverage(forged))
    assert not verification["valid"]
    assert "rows[0].pass_with_binding_errors" in verification["errors"]


def test_contract_rejects_pass_row_with_errors_scrubbed(tmp_path: Path) -> None:
    """Dropping the error strings but keeping the unbound record is still caught."""
    forged = _promote_first_row(_drift_body(tmp_path))
    forged["rows"][0]["binding_errors"] = []
    forged["n_binding_errors"] = 0
    errors = replay_coverage_contract_errors(forged)
    assert "rows[0].pass_with_binding_errors" in errors
    assert "rows[0].binding_errors" in errors


def test_contract_rejects_bound_record_with_unequal_digests(tmp_path: Path) -> None:
    forged = _promote_first_row(_drift_body(tmp_path))
    forged["rows"][0]["binding_errors"] = []
    forged["rows"][0]["reproduces"][0]["bound"] = True
    forged["n_binding_errors"], forged["n_reproduces_bound"] = 0, 1
    assert "rows[0].reproduces[0].digest" in replay_coverage_contract_errors(forged)


@pytest.mark.parametrize(
    ("field", "bad"),
    [("n_binding_errors", 0), ("n_binding_errors", 5), ("n_reproduces_bound", 1)],
)
def test_contract_rejects_mismatched_binding_counts(tmp_path: Path, field: str, bad: int) -> None:
    body = _drift_body(tmp_path)
    assert replay_coverage_contract_errors(body) == []
    assert body[field] != bad
    assert field in replay_coverage_contract_errors({**body, field: bad})


def test_contract_rejects_row_stripped_of_binding(tmp_path: Path) -> None:
    """A body declaring binding cannot shed it from one row."""
    body = _drift_body(tmp_path)
    forged = json.loads(json.dumps(body))
    for key in ("binding_errors", "n_reproduces", "reproduces"):
        forged["rows"][0].pop(key)
    errors = replay_coverage_contract_errors(forged)
    assert "rows[0].binding_errors" in errors
    assert "rows[0].n_reproduces" in errors


def test_contract_rejects_mismatched_n_reproduces(tmp_path: Path) -> None:
    body = _drift_body(tmp_path)
    forged = json.loads(json.dumps(body))
    forged["rows"][0]["n_reproduces"] = 2
    assert "rows[0].n_reproduces" in replay_coverage_contract_errors(forged)


def test_contract_still_accepts_legacy_bodies(tmp_path: Path) -> None:
    """Bodies sealed before binding existed (no binding fields) stay checkable."""
    live = _scratch_repo(tmp_path)
    _write_carrier(tmp_path, reproduces=[_entry(_RECEIPT_REL, live)])
    body = _sweep(tmp_path)
    legacy = json.loads(json.dumps(body))
    legacy.pop("n_binding_errors")
    legacy.pop("n_reproduces_bound")
    for row in legacy["rows"]:
        for key in ("binding_errors", "n_reproduces", "reproduces", "execution_verdict"):
            row.pop(key)
    assert replay_coverage_contract_errors(legacy) == []
    # ...but a legacy pass row must still report a clean execution.
    lying = json.loads(json.dumps(legacy))
    lying["rows"][0]["exit_code"] = 1
    assert "rows[0].exit_code" in replay_coverage_contract_errors(lying)
