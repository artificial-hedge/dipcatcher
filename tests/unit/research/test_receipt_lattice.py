"""Tests for the cross-receipt consistency lattice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.receipt_lattice import (
    LATTICE_SCHEMA,
    lattice_contract_errors,
    receipt_lattice,
)


def _write(root: Path, name: str, doc: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(doc))
    return path


def _receipt(inputs: str, score: float, extra: dict | None = None) -> dict:
    body = {
        "schema": "demo.v1",
        "inputs_sha256": inputs,
        "results": [{"head": "a", "pinball": score}],
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
    }
    if extra:
        body.update(extra)
    return body


def test_missing_dir_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        receipt_lattice(tmp_path / "nope")


def test_identical_inputs_identical_claims_are_consistent(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    _write(tmp_path, "r2.json", _receipt("in-a", 0.42))
    out = receipt_lattice(tmp_path)
    assert out["schema"] == LATTICE_SCHEMA
    assert out["verdict"] == "consistent"
    assert out["n_consistent_groups"] == 2  # head + pinball claim leaves
    assert out["n_inconsistent_groups"] == 0
    assert all(g["files"] == ["r1.json", "r2.json"] for g in out["groups"])


def test_identical_inputs_different_claims_are_inconsistent(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-b", 0.42))
    _write(tmp_path, "r2.json", _receipt("in-b", 0.43))
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "inconsistent"
    assert out["n_inconsistent_groups"] == 1
    group = next(g for g in out["groups"] if g["verdict"] == "inconsistent")
    assert group["disagreements"][0]["values"] == [0.42, 0.43]


def test_tiny_float_drift_is_drift_not_inconsistent(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-c", 1.0))
    _write(tmp_path, "r2.json", _receipt("in-c", 1.0 + 5e-10))
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "drift_or_stale"
    assert out["n_drift_groups"] == 1
    assert out["n_inconsistent_groups"] == 0


def test_different_inputs_never_compare(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-x", 0.42))
    _write(tmp_path, "r2.json", _receipt("in-y", 9.99))
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "consistent"
    assert out["n_claim_groups"] == 0


def test_unspecified_inputs_are_a_coverage_gap_not_a_flag(tmp_path: Path) -> None:
    doc = _receipt("in-z", 1.0)
    del doc["inputs_sha256"]
    _write(tmp_path, "r1.json", doc)
    doc2 = dict(doc)
    doc2["results"] = [{"head": "a", "pinball": 999.0}]
    _write(tmp_path, "r2.json", doc2)
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "consistent"
    assert out["n_inputs_unspecified"] == 2
    assert out["n_inconsistent_groups"] == 0


def test_parse_errors_downgrade_verdict(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-p", 0.5))
    (tmp_path / "bad.json").write_bytes(b"{not json")
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "partially_unreadable"
    assert out["n_parse_errors"] == 1
    assert out["parse_errors"][0]["file"] == "bad.json"


def test_stale_code_flagged_against_head(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-s", 0.5, {"git_revision": "abc123"}))
    _write(tmp_path, "r2.json", _receipt("in-t", 0.6, {"git_revision": "def456"}))
    out = receipt_lattice(tmp_path, head_sha="abc123")
    assert out["n_stale_code"] == 1
    assert out["stale_code"][0]["file"] == "r2.json"
    assert out["verdict"] == "drift_or_stale"


def test_v2_envelope_unwraps_to_payload(tmp_path: Path) -> None:
    inner = _receipt("in-v", 0.5)
    env = {
        "schema": "receipt.v2",
        "code_files": {"a.py": "x" * 64},
        "payload": inner,
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
    }
    _write(tmp_path, "e1.json", env)
    env2 = dict(env)
    env2["payload"] = _receipt("in-v", 0.5)
    _write(tmp_path, "e2.json", env2)
    out = receipt_lattice(tmp_path)
    assert out["n_consistent_groups"] == 2  # head + pinball


def test_dataset_tier_edges_across_different_inputs(tmp_path: Path) -> None:
    """Two lanes on the same tape with different run params still edge."""
    ds = "d" * 64
    r1 = _receipt("inputs-A-params", 0.42, {"dataset_sha256": ds})
    r2 = _receipt("inputs-B-params", 0.42, {"params": {"dataset_sha256": ds}})
    _write(tmp_path, "r1.json", r1)
    _write(tmp_path, "r2.json", r2)
    out = receipt_lattice(tmp_path)
    ds_groups = [g for g in out["groups"] if g["tier"] == "dataset"]
    assert len(ds_groups) == 2  # head + pinball
    assert all(g["verdict"] == "consistent" for g in ds_groups)


def test_dataset_tier_catches_disagreement_on_same_tape(tmp_path: Path) -> None:
    ds = "e" * 64
    _write(tmp_path, "r1.json", _receipt("inp-1", 0.42, {"dataset_sha256": ds}))
    _write(tmp_path, "r2.json", _receipt("inp-2", 0.99, {"dataset_sha256": ds}))
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "inconsistent"
    bad = next(g for g in out["groups"] if g["verdict"] == "inconsistent")
    assert bad["tier"] == "dataset"


def test_dataset_unspecified_counted(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-u", 0.5))
    out = receipt_lattice(tmp_path)
    assert out["n_dataset_unspecified"] == 1


def test_identity_fields_are_never_claims(tmp_path: Path) -> None:
    r1 = _receipt("in-i", 0.5)
    r2 = _receipt("in-i", 0.5)
    r2["results"][0]["row_sha256"] = "a" * 64
    r2["results"][0]["generated_at"] = "2099-01-01"
    _write(tmp_path, "r1.json", r1)
    _write(tmp_path, "r2.json", r2)
    out = receipt_lattice(tmp_path)
    # extra identity fields on r2 create singleton paths only — no inconsistency
    assert out["n_inconsistent_groups"] == 0


def test_contract_passes_on_own_output(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-k", 0.5))
    _write(tmp_path, "r2.json", _receipt("in-k", 0.55))
    out = receipt_lattice(tmp_path)
    assert lattice_contract_errors(out) == []


def test_contract_catches_forged_verdict(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-k", 0.5))
    _write(tmp_path, "r2.json", _receipt("in-k", 0.55))
    out = receipt_lattice(tmp_path)
    forged = dict(out)
    forged["verdict"] = "consistent"
    assert "verdict" in lattice_contract_errors(forged)


def test_contract_catches_forged_group_count(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-k", 0.5))
    _write(tmp_path, "r2.json", _receipt("in-k", 0.55))
    out = receipt_lattice(tmp_path)
    forged = dict(out)
    forged["n_inconsistent_groups"] = 0
    assert "n_inconsistent_groups" in lattice_contract_errors(forged)


def test_contract_catches_forged_inputs_digest(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-k", 0.5))
    out = receipt_lattice(tmp_path)
    forged = dict(out)
    forged["inputs_sha256"] = "0" * 64
    assert "inputs_sha256" in lattice_contract_errors(forged)


def test_contract_rejects_fake_inconsistency(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-k", 0.5))
    _write(tmp_path, "r2.json", _receipt("in-k", 0.5))
    out = receipt_lattice(tmp_path)
    forged = dict(out)
    forged["groups"] = [
        {
            "inputs_sha256": "in-k",
            "claim_path": "results[0].pinball",
            "n_receipts": 2,
            "files": ["r1.json", "r2.json"],
            "verdict": "inconsistent",
            "disagreements": [{"claim_path": "x", "files": ["r1", "r2"], "values": [1, 1]}],
        }
    ]
    forged["n_inconsistent_groups"] = 1
    forged["n_claim_groups"] = 1
    forged["n_consistent_groups"] = 0
    forged["verdict"] = "inconsistent"
    errors = lattice_contract_errors(forged)
    assert "groups[0].disagreements_not_real" in errors


def test_lattice_cli_emits_sealed_receipt(tmp_path: Path) -> None:
    """`dipcatcher lattice` writes a seal-verified receipt_lattice.v1 artifact."""

    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    src = tmp_path / "src"
    src.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write(src, "a.json", _receipt("aa" * 32, 0.5))
    runner = CliRunner()
    result = runner.invoke(app, ["lattice", "--receipts-dir", str(src), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    emitted = list(out_dir.glob("receipt_lattice_*.json"))
    assert len(emitted) == 1
    from quant_fund.research.receipt_v2 import verify_receipt_file

    verification = verify_receipt_file(emitted[0])
    assert verification["valid"], verification["errors"]


def test_lattice_receipt_v2_round_trip(tmp_path: Path) -> None:
    """receipt_version=2 seals the receipt_lattice.v1 body in the envelope."""
    from quant_fund.research.receipt_lattice import write_lattice_receipt
    from quant_fund.research.receipt_v2 import verify_receipt_file

    _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    _write(tmp_path, "r2.json", _receipt("in-a", 0.42))
    report = receipt_lattice(tmp_path)
    path = write_lattice_receipt(report, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "receipt_lattice.v1"
    assert payload["payload"]["inputs_sha256"] == report["inputs_sha256"]
    assert verify_receipt_file(path)["valid"] is True


def test_lattice_cli_strict_exits_on_inconsistent(tmp_path: Path) -> None:

    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    src = tmp_path / "src"
    src.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ds = "bb" * 32
    a = _receipt("aa" * 32, 0.5)
    a["dataset_sha256"] = ds
    b = _receipt("cc" * 32, 0.9)
    b["dataset_sha256"] = ds
    _write(src, "a.json", a)
    _write(src, "b.json", b)
    runner = CliRunner()
    ok = runner.invoke(
        app, ["lattice", "--strict", "--receipts-dir", str(src), "--out-dir", str(out_dir)]
    )
    assert ok.exit_code == 1
    plain = runner.invoke(app, ["lattice", "--receipts-dir", str(src), "--out-dir", str(out_dir)])
    assert plain.exit_code == 0


def test_known_inconsistent_pins_suppress_verdict(tmp_path: Path) -> None:
    """Byte-exact pinned files keep their groups 'inconsistent' + known:true
    but do not drive the top-level verdict."""
    import hashlib

    ds = "bb" * 32
    a = _receipt("aa" * 32, 0.5)
    a["dataset_sha256"] = ds
    b = _receipt("cc" * 32, 0.9)
    b["dataset_sha256"] = ds
    pa = _write(tmp_path, "a.json", a)
    pb = _write(tmp_path, "b.json", b)
    pins = {
        "a.json": hashlib.sha256(pa.read_bytes()).hexdigest(),
        "b.json": hashlib.sha256(pb.read_bytes()).hexdigest(),
    }
    receipt = receipt_lattice(tmp_path, known_inconsistent=pins)
    assert receipt["verdict"] == "consistent"
    assert receipt["n_inconsistent_groups"] == 0
    assert receipt["n_known_inconsistent_groups"] == 1
    inc = [g for g in receipt["groups"] if g["verdict"] == "inconsistent"]
    assert len(inc) == 1 and inc[0]["known"] is True
    assert all("known" not in g for g in receipt["groups"] if g["verdict"] != "inconsistent")
    assert lattice_contract_errors(receipt) == []


def test_drifted_pin_does_not_suppress(tmp_path: Path) -> None:
    """A pin whose digest does not match the file on disk applies to
    nothing — a tampered pinned receipt re-enters the gate."""
    ds = "bb" * 32
    a = _receipt("aa" * 32, 0.5)
    a["dataset_sha256"] = ds
    b = _receipt("cc" * 32, 0.9)
    b["dataset_sha256"] = ds
    _write(tmp_path, "a.json", a)
    _write(tmp_path, "b.json", b)
    pins = {"a.json": "00" * 32, "b.json": "00" * 32}
    receipt = receipt_lattice(tmp_path, known_inconsistent=pins)
    assert receipt["verdict"] == "inconsistent"
    assert receipt["n_inconsistent_groups"] == 1
    assert receipt["n_known_inconsistent_groups"] == 0


def test_mixed_group_membership_stays_inconsistent(tmp_path: Path) -> None:
    """A group is suppressed only when EVERY member is pinned — an unpinned
    third receipt rejoining the same claims keeps the gate armed."""
    import hashlib

    ds = "bb" * 32
    a = _receipt("aa" * 32, 0.5)
    a["dataset_sha256"] = ds
    b = _receipt("cc" * 32, 0.9)
    b["dataset_sha256"] = ds
    c = _receipt("dd" * 32, 0.9)
    c["dataset_sha256"] = ds
    pa = _write(tmp_path, "a.json", a)
    _write(tmp_path, "b.json", b)
    _write(tmp_path, "c.json", c)
    pins = {
        "a.json": hashlib.sha256(pa.read_bytes()).hexdigest(),
        "b.json": hashlib.sha256(b"not-the-file").hexdigest(),
    }
    receipt = receipt_lattice(tmp_path, known_inconsistent=pins)
    assert receipt["verdict"] == "inconsistent"
    assert receipt["n_known_inconsistent_groups"] == 0


def test_lattice_cli_known_inconsistent_flag(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    src = tmp_path / "src"
    src.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ds = "bb" * 32
    a = _receipt("aa" * 32, 0.5)
    a["dataset_sha256"] = ds
    b = _receipt("cc" * 32, 0.9)
    b["dataset_sha256"] = ds
    _write(src, "a.json", a)
    _write(src, "b.json", b)
    import hashlib

    pins = {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in ("a.json", "b.json")}
    pin_file = tmp_path / "pins.json"
    pin_file.write_text(json.dumps(pins))
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "lattice",
            "--strict",
            "--receipts-dir",
            str(src),
            "--out-dir",
            str(out_dir),
            "--known-inconsistent",
            str(pin_file),
        ],
    )
    assert res.exit_code == 0
    bad = runner.invoke(
        app,
        [
            "lattice",
            "--receipts-dir",
            str(src),
            "--out-dir",
            str(out_dir),
            "--known-inconsistent",
            str(tmp_path / "missing.json"),
        ],
    )
    assert bad.exit_code != 0


def test_meta_audit_kinds_carry_no_claims(tmp_path: Path) -> None:
    """corpus/process attestations are not measured claims — their verdict
    fields (chain position, admit/quarantine, prior audit verdicts) must not
    form claim groups."""
    _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    for i, kind in enumerate(
        ["corpus_epoch.v1", "receipt_admission.v1", "receipt_lattice.v1", "receipt_graph.v1"]
    ):
        _write(
            tmp_path,
            f"meta{i}.json",
            _receipt("in-a", 0.42, {"kind": kind, "verdict": f"verdict-{i}"}),
        )
    out = receipt_lattice(tmp_path)
    # Only the real member's claims form groups; the four meta receipts add none.
    assert out["verdict"] == "consistent"
    assert all(len(g["files"]) == 1 for g in out["groups"])
