"""Subprocess calls fail closed when the child hangs past its deadline."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from quant_fund.market_sim import native
from quant_fund.models import fbm, rbergomi
from quant_fund.proofcore import ci
from quant_fund.registry import schema_drift
from quant_fund.research import evidence_audit, timestamp_anchor, vine_dominance, vine_panel
from quant_fund.research.evidence_export import export_evidence_bundle
from quant_fund.utils.hashing import hash_bytes

from .research.test_evidence_export import _minimal_evidence_tree


def _expired(cmd: list[str], kwargs: dict[str, object]) -> subprocess.TimeoutExpired:
    timeout = kwargs["timeout"]
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    return subprocess.TimeoutExpired(cmd, float(timeout))


def test_cli_verifier_timeout_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = tmp_path / "r.json"
    receipt.write_text("{}")
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["timeout"] = kwargs["timeout"]
        assert cmd[-2:] == ["verify-receipt", str(receipt)]
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(ci.subprocess, "run", fake_run)
    assert ci._cli_verifier(receipt) is False
    assert seen["timeout"] == ci._VERIFY_TIMEOUT_S


def test_coverage_gate_records_a_hung_report() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == ci._COVERAGE_TIMEOUT_S
        raise _expired(cmd, kwargs)

    failures = ci.coverage_gate(pyproject, runner=fake_run)
    assert failures
    assert all("TimeoutExpired" in item for item in failures)


def test_git_revision_probe_is_bounded(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def runner(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["timeout"] = kwargs["timeout"]
        raise _expired(cmd, kwargs)

    assert ci._git_revision(tmp_path, runner=runner) is None
    assert seen["timeout"] == ci._GIT_TIMEOUT_S


def test_schema_drift_git_timeout_is_a_phantom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"kind": "t.v1", "claim": {"x": 1}, "git_revision": "abc1234"}
    (tmp_path / "a.json").write_text(json.dumps(payload))

    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == schema_drift._GIT_TIMEOUT_S
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(schema_drift.subprocess, "run", boom)
    audit = schema_drift.schema_drift(tmp_path, repo=tmp_path)
    assert audit["phantom_revisions"][0]["why"] == "not_a_commit"
    assert audit["n_flagged"] == 1


def test_matching_core_compile_timeout_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == native._COMPILE_TIMEOUT_S
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(native.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="compile exceeded"):
        native._compile(tmp_path / "lob.c", tmp_path / "lob.so")


def test_training_receipt_git_timeout_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.train import receipts

    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 15
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(receipts.subprocess, "run", boom)
    assert receipts._git_revision(tmp_path) == ("unknown", True)


def test_receipt_git_revision_matches_head() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout.strip()
    assert fbm._git_revision() == head
    assert rbergomi._git_revision() == head
    assert vine_panel._git_revision() == head
    assert vine_dominance._git_revision() == head


@pytest.mark.parametrize(
    "fn",
    [
        fbm._git_revision,
        rbergomi._git_revision,
        vine_panel._git_revision,
        vine_dominance._git_revision,
    ],
)
def test_receipt_git_timeout_stamps_unknown(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[[], str]
) -> None:
    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 15
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(subprocess, "run", boom)
    assert fn() == "unknown"


def test_evidence_export_does_not_swallow_a_git_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _minimal_evidence_tree(tmp_path)

    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 15
        raise _expired(cmd, kwargs)

    monkeypatch.setattr("quant_fund.research.evidence_export.subprocess.run", boom)
    with pytest.raises(subprocess.TimeoutExpired):
        export_evidence_bundle(root, tmp_path / "bundle")


def test_evidence_index_timeout_is_not_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "scripts" / "build_evidence_report.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('index')\n")
    index = tmp_path / "docs" / "evidence" / "index.md"
    index.parent.mkdir(parents=True)
    index.write_text("index\n")

    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == evidence_audit._INDEX_TIMEOUT_S
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(evidence_audit.subprocess, "run", boom)
    assert evidence_audit.check_evidence_index_fresh(tmp_path) is False


def test_openssl_probe_timeout_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 10
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(timestamp_anchor.subprocess, "run", boom)
    assert timestamp_anchor._openssl_available() is False


def test_chain_verify_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "quality" / "pin.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n")
    digest = hash_bytes(target.read_bytes())
    tdir = tmp_path / "quality" / "timestamps"
    tdir.mkdir()
    (tdir / timestamp_anchor.CACERT_NAME).write_text("CERT\n")
    (tdir / "pin.tsr").write_bytes(b"token")
    (tdir / timestamp_anchor.ANCHORS_MANIFEST).write_text(
        json.dumps(
            {
                "schema": timestamp_anchor.ANCHORS_SCHEMA,
                "anchors": {"pin.tsr": {"target": "quality/pin.json", "sha256": digest}},
            }
        )
    )
    monkeypatch.setattr(timestamp_anchor, "_openssl_available", lambda: True)
    monkeypatch.setattr(timestamp_anchor, "extract_imprint", lambda _data: bytes.fromhex(digest))

    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 30
        raise _expired(cmd, kwargs)

    monkeypatch.setattr(timestamp_anchor.subprocess, "run", boom)
    result = timestamp_anchor.verify_timestamps(tmp_path)
    assert result["ok"] is False
    assert result["chain"]["quality/pin.json"] == "chain_failed"
    assert "chain_verify_failed:quality/pin.json" in result["errors"]
