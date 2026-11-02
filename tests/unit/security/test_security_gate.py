"""Security-gate classification, the scorecard reader, and the CI wiring."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_LIB_PATH = _ROOT / "scripts" / "security_gate_lib.py"
_SPEC = importlib.util.spec_from_file_location("security_gate_lib", _LIB_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LIB = importlib.util.module_from_spec(_SPEC)
sys.modules["security_gate_lib"] = _LIB
_SPEC.loader.exec_module(_LIB)

_TODAY = date(2026, 11, 2)
_SECURITY_WORKFLOWS = (
    "codeql.yml",
    "dependency-audit.yml",
    "dependency-review.yml",
    "scorecard.yml",
    "secret-scan.yml",
)


def test_image_digest_must_match_the_inspect_id() -> None:
    digest = "a" * 64
    assert _LIB.digest_recorded(f"sha256:{digest}", digest) is True
    assert _LIB.digest_recorded(f"sha256:{digest}", f"sha256:{digest}") is True
    assert _LIB.digest_recorded(f"sha256:{'b' * 64}", digest) is False
    assert _LIB.digest_recorded("", digest) is False


def test_unreachable_service_is_status_zero() -> None:
    spec = importlib.util.spec_from_file_location(
        "security_gate_cli", _ROOT / "scripts" / "security_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["security_gate_cli"] = module
    spec.loader.exec_module(module)
    status, headers = module._request("http://127.0.0.1:1/health", None, {}, 2)
    assert status == 0
    assert headers == {}


def test_download_commit_rejects_a_bad_sha(tmp_path: Path) -> None:
    partial = tmp_path / "blob.partial"
    partial.write_bytes(b"not-the-bytes")
    dest = tmp_path / "blob"
    try:
        _LIB.commit_download(partial, dest, "0" * 64)
    except SystemExit as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("bad sha was accepted")
    assert not partial.exists()
    assert not dest.exists()


def test_trivy_fails_fixable_and_expired_and_accepts_current_unfixable() -> None:
    allow = {"CVE-UNFIXED": date(2027, 2, 2), "CVE-OLD": date(2026, 1, 1)}
    vulns = [
        _LIB.TrivyVuln("lab", "CVE-FIX", "HIGH", True),
        _LIB.TrivyVuln("lab", "CVE-UNFIXED", "CRITICAL", False),
        _LIB.TrivyVuln("research", "CVE-OLD", "HIGH", False),
        _LIB.TrivyVuln("research", "CVE-NEW", "HIGH", False),
        _LIB.TrivyVuln("lab", "CVE-FIX", "CRITICAL", False),
    ]
    collapsed = _LIB.parse_trivy(
        {
            "Results": [
                {
                    "Vulnerabilities": [
                        {"VulnerabilityID": "CVE-FIX", "Severity": "HIGH", "FixedVersion": "1.2.3"},
                        {"VulnerabilityID": "CVE-FIX", "Severity": "CRITICAL", "FixedVersion": ""},
                        {"VulnerabilityID": "CVE-LOW", "Severity": "LOW", "FixedVersion": "9"},
                    ]
                }
            ]
        },
        "lab",
    )
    assert len(collapsed) == 1
    assert collapsed[0].fixable is True
    failing = _LIB.trivy_failures(vulns, allow, _TODAY)
    assert "trivy:lab:fixable:CVE-FIX" in failing
    assert "trivy:research:expired:CVE-OLD" in failing
    assert "trivy:research:unlisted:CVE-NEW" in failing
    assert all("CVE-UNFIXED" not in item for item in failing)


def test_accepted_findings_expire() -> None:
    accepted = [{"id": "zap:10049:Non-Storable Content", "expiry": "2027-02-02"}]
    assert _LIB.finding_accepted(accepted[0]["id"], accepted, _TODAY) is True
    assert _LIB.finding_accepted(accepted[0]["id"], accepted, date(2027, 2, 3)) is False
    assert _LIB.new_ids(["zap:10049:Non-Storable Content", "zap:9:Other"], accepted, _TODAY) == [
        "zap:9:Other"
    ]


def test_parsers_keep_stable_ids(tmp_path: Path) -> None:
    bandit = _LIB.parse_bandit(
        {
            "results": [
                {"filename": str(tmp_path / "src" / "a.py"), "test_id": "B101", "line_number": 4}
            ]
        },
        tmp_path,
    )
    assert bandit == ["bandit:B101:src/a.py:4"]
    assert _LIB.parse_hadolint([{"code": "DL3066", "file": "Dockerfile", "line": 28}]) == [
        "hadolint:DL3066:Dockerfile:28"
    ]
    assert _LIB.parse_pip_audit(
        {"dependencies": [{"name": "virtualenv", "vulns": [{"id": "GHSA-cx6f-vjhg-g3cc"}]}]}
    ) == ["pip-audit:GHSA-cx6f-vjhg-g3cc:virtualenv"]
    assert _LIB.parse_osv(
        {
            "results": [
                {
                    "packages": [
                        {
                            "package": {"name": "virtualenv"},
                            "vulnerabilities": [{"id": "GHSA-cx6f-vjhg-g3cc"}],
                        }
                    ]
                }
            ]
        }
    ) == ["osv:GHSA-cx6f-vjhg-g3cc:virtualenv"]
    report = tmp_path / "leaks.json"
    report.write_text(
        json.dumps([{"RuleID": "generic", "File": "a.py", "StartLine": 3, "Secret": "hidden"}]),
        encoding="utf-8",
    )
    assert _LIB.scrub_gitleaks(report) == ["gitleaks:generic:a.py:3"]
    assert not report.exists()
    assert _LIB.parse_zap(
        {"site": [{"alerts": [{"pluginid": "10049", "name": "Non-Storable Content"}]}]}
    ) == ["zap:10049:Non-Storable Content"]


def test_summary_reader_requires_head_and_every_tool() -> None:
    summary = {
        "schema": _LIB.SCHEMA,
        "head": "abc",
        "ok": True,
        "dirty": False,
        "new_findings": [],
        "tools": {name: {"ok": True} for name in _LIB.REQUIRED_TOOLS},
    }
    assert _LIB.evaluate_summary(summary, "abc") == []
    summary["head"] = "other"
    assert "head" in _LIB.evaluate_summary(summary, "abc")


def test_lock_allowlist_and_scorecard_are_dated() -> None:
    lock = json.loads(
        (_ROOT / "deploy" / "security" / "tools.lock.json").read_text(encoding="utf-8")
    )
    assert lock["schema"] == "dipcatcher.security_tools/1"
    for name in ("bandit", "gitleaks", "hadolint", "osv-scanner", "pip-audit", "trivy", "zap"):
        spec = lock["tools"][name]
        assert spec["version"]
        assert len(spec["sha256"]) == 64
        assert spec["url"]
    allow = json.loads(
        (_ROOT / "deploy" / "security" / "trivy-allow.json").read_text(encoding="utf-8")
    )
    accepted = json.loads(
        (_ROOT / "deploy" / "security" / "accepted-findings.json").read_text(encoding="utf-8")
    )
    _LIB.dated_entries(allow, "entries", "dipcatcher.trivy_allow/1")
    _LIB.dated_entries(accepted, "findings", "dipcatcher.security_accepted/1")
    text = (_ROOT / "deploy" / "security.mk").read_text(encoding="utf-8")
    assert "security-gate:" in text
    assert "scripts/security_gate.py" in text
    scorecard = (_ROOT / "docs" / "ULTRA_PROD_READINESS.md").read_text(encoding="utf-8")
    assert "security_gate" in scorecard
    assert "check_security_gate_summary.py" in scorecard
    assert "docs/SECURITY_GATE.md" in scorecard
    gate_doc = (_ROOT / "docs" / "SECURITY_GATE.md").read_text(encoding="utf-8")
    assert ".tmp_security/summary.json" in gate_doc
    assert "100000" in gate_doc
    assert "QUANT_API_RATE_PER_SEC=off" in gate_doc


def test_cursor_ci_runs_the_gate_without_a_write_token() -> None:
    path = _ROOT / ".github" / "workflows" / "cursor-ci.yml"
    text = path.read_text(encoding="utf-8")
    assert "pull_request_target" not in text
    assert "make -f deploy/security.mk security-gate" in text
    loaded = yaml.load(text, Loader=yaml.BaseLoader)
    assert loaded["permissions"] == {"contents": "read"}
    job = loaded["jobs"]["security-gate"]
    assert job["runs-on"] == "instabox-large"
    runs = [step.get("run") for step in job["steps"] if isinstance(step, dict)]
    assert "make -f deploy/security.mk security-gate" in runs


def test_security_workflows_point_at_the_gate_and_keep_their_runners() -> None:
    for name in _SECURITY_WORKFLOWS:
        text = (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "pull_request_target" not in text
        assert "make -f deploy/security.mk security-gate" in text
        assert "runs-on: instabox-large" in text


def _load_gate() -> object:
    path = _ROOT / "scripts" / "security_gate.py"
    spec = importlib.util.spec_from_file_location("security_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_copies_the_ready_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _load_gate()
    work = tmp_path / "lab"
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], capture_output: bool = False) -> SimpleNamespace:
        del capture_output
        seen.append(cmd)
        (work / "lab-baseline.json").write_text('{"site": []}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert (
        gate.one_baseline("zap:test", "http://127.0.0.1:8000", work, "lab-baseline.json", 18091)
        == []
    )
    cmd = seen[0]
    assert cmd[cmd.index("-c") + 1] == "zap-rules.tsv"
    assert "--autooff" in cmd
    source = (_ROOT / "deploy" / "security" / "zap-rules.tsv").read_text(encoding="utf-8")
    assert (work / "zap-rules.tsv").read_text(encoding="utf-8") == source
