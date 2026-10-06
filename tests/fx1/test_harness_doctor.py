"""harness doctor — the structured deployment-diagnosis feature.

All four legs ship the same verdict shape: ``Fx1Harness.doctor`` (the
in-process builder), ``GET /harness/doctor`` (admin-scoped route),
``HarnessClient.doctor`` (proxy + wire-side checks), and
``fx1 harness doctor`` (exit 0 healthy / 1 broken / 2 degraded).
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import fx1.serve.backends as _be
from fx1.cli import app
from fx1.sdk import Fx1Harness
from fx1.serve.api import create_app
from fx1.serve.client import HarnessClient
from fx1.serve.doctor import DoctorReport, build_doctor_report, doctor_verdict
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore

_DOCTOR_ENV_VARS = (
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
    "FX1_CHECKPOINT_DIR",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_SIGNING_KEY",
    "MOONSHOT_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_doctor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor reads env — pin a deterministic baseline per test."""
    for name in _DOCTOR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _checks_by_name(report: DoctorReport) -> dict[str, Any]:
    return {c.name: c for c in report.checks}


def _byok_transport_ok(request: urllib.request.Request, *, timeout_s: float):
    @contextlib.contextmanager
    def _open():
        yield object()

    return _open()


# ---- verdict arithmetic ---------------------------------------------------


def test_report_model_and_verdict() -> None:
    from fx1.serve.doctor import DoctorCheck

    report = DoctorReport(mode="in_process", checked_at=0.0, verdict="healthy", checks=[])
    assert report.ok is True
    report.checks.append(DoctorCheck(name="w", ok=False, severity="warn"))
    report.verdict = doctor_verdict(report.checks)
    assert report.verdict == "degraded" and report.ok is False
    out = report.model_dump(mode="json")
    assert out["checks"][0]["name"] == "w" and out["verdict"] == "degraded"


def test_doctor_verdict_severity_rules() -> None:
    from fx1.serve.doctor import DoctorCheck

    assert doctor_verdict([DoctorCheck(name="a", ok=True)]) == "healthy"
    assert doctor_verdict([DoctorCheck(name="a", ok=False, severity="warn")]) == "degraded"
    assert doctor_verdict([DoctorCheck(name="a", ok=False, severity="error")]) == "broken"
    both = [DoctorCheck(name="w", ok=False, severity="warn"), DoctorCheck(name="e", ok=False)]
    assert doctor_verdict(both) == "broken"


# ---- builder: config checks ------------------------------------------------


def test_no_backends_is_broken() -> None:
    report = build_doctor_report(
        mode="in_process",
        key_store=ApiKeyStore(),
        registered_commands=1,
        probe_backends=False,
    )
    checks = _checks_by_name(report)
    assert checks["config.backends"].ok is False
    assert report.verdict == "broken"


def test_hosted_env_makes_backend_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    report = build_doctor_report(
        mode="in_process",
        key_store=ApiKeyStore(),
        registered_commands=1,
        probe_backends=False,
    )
    checks = _checks_by_name(report)
    assert checks["config.backends"].ok is True
    assert "hosted_k3" in checks["config.backends"].detail


def test_byok_partial_config_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_BYOK_BASE_URL", "https://provider.example/v1")
    report = build_doctor_report(
        mode="in_process",
        key_store=ApiKeyStore(),
        registered_commands=1,
        probe_backends=False,
    )
    check = _checks_by_name(report)["config.byok"]
    assert check.ok is False
    assert "missing" in check.detail


def test_byok_probe_ok_auth_refused_and_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, val in (
        ("FX1_BYOK_BASE_URL", "https://provider.example/v1"),
        ("FX1_BYOK_API_KEY", "sk-k"),
        ("FX1_BYOK_MODEL", "m"),
    ):
        monkeypatch.setenv(name, val)

    monkeypatch.setattr(
        _be,
        "_openai_urlopen",
        lambda req, *, timeout_s: _byok_transport_ok(req, timeout_s=timeout_s),
    )
    report = build_doctor_report(mode="in_process", key_store=ApiKeyStore(), registered_commands=1)
    check = _checks_by_name(report)["config.byok"]
    assert check.ok is True and "accepted" in check.detail

    def _auth_refused(req: Any, *, timeout_s: float):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(_be, "_openai_urlopen", _auth_refused)
    check = _checks_by_name(
        build_doctor_report(mode="in_process", key_store=ApiKeyStore(), registered_commands=1)
    )["config.byok"]
    assert check.ok is False and check.severity == "error"
    assert "credential refused" in check.detail

    def _unreachable(req: Any, *, timeout_s: float):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(_be, "_openai_urlopen", _unreachable)
    check = _checks_by_name(
        build_doctor_report(mode="in_process", key_store=ApiKeyStore(), registered_commands=1)
    )["config.byok"]
    assert check.ok is False and "unreachable" in check.detail


def test_byok_bad_url_fails_without_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, val in (
        ("FX1_BYOK_BASE_URL", "not-a-url"),
        ("FX1_BYOK_API_KEY", "sk-k"),
        ("FX1_BYOK_MODEL", "m"),
    ):
        monkeypatch.setenv(name, val)
    check = _checks_by_name(
        build_doctor_report(mode="in_process", key_store=ApiKeyStore(), registered_commands=1)
    )["config.byok"]
    assert check.ok is False


def test_local_engine_partial_config_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_LOCAL_SERVE_CMD", "vllm serve")
    check = _checks_by_name(
        build_doctor_report(mode="in_process", key_store=ApiKeyStore(), registered_commands=1)
    )["config.local_fx1"]
    assert check.ok is False
    assert "CHECKPOINT" in check.detail


# ---- builder: state checks -------------------------------------------------


def test_state_dir_missing_and_journals(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    journal = JobJournal(state / "jobs.jsonl")
    journal.append({"k": 1})
    journal.append({"k": 2})
    report = build_doctor_report(
        mode="in_process",
        key_store=ApiKeyStore(),
        state_path=state,
        registered_commands=1,
        probe_backends=False,
    )
    checks = _checks_by_name(report)
    assert checks["state.dir"].ok is True
    assert checks["state.journals"].ok is True
    assert "jobs.jsonl=2" in checks["state.journals"].detail


def test_torn_journal_degrades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")  # keep every error check green
    state = tmp_path / "state"
    state.mkdir()
    journal = JobJournal(state / "jobs.jsonl")
    journal.append({"k": 1})
    with (state / "jobs.jsonl").open("ab") as fh:
        fh.write(b'{"seq":9,"chain":"bad","payload":')  # torn tail
    report = build_doctor_report(
        mode="in_process",
        key_store=ApiKeyStore(),
        state_path=state,
        registered_commands=1,
        probe_backends=False,
    )
    check = _checks_by_name(report)["state.journals"]
    assert check.ok is False and check.severity == "warn"
    assert "chain broke" in check.detail
    assert report.verdict == "degraded"


def test_state_dir_not_a_directory(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    report = build_doctor_report(
        mode="in_process",
        key_store=ApiKeyStore(),
        state_path=blocker,
        registered_commands=1,
        probe_backends=False,
    )
    assert _checks_by_name(report)["state.dir"].ok is False


# ---- builder: key checks ---------------------------------------------------


def test_no_usable_credential_is_broken() -> None:
    store = ApiKeyStore()
    _raw, rec = store.mint(name="svc")
    store.revoke(rec["key_id"])
    report = build_doctor_report(
        mode="in_process", key_store=store, registered_commands=1, probe_backends=False
    )
    check = _checks_by_name(report)["keys.auth"]
    assert check.ok is False
    assert "no credential can authenticate" in check.detail


def test_quota_exhausted_key_degrades() -> None:
    store = ApiKeyStore()
    raw, _rec = store.mint(name="svc", max_requests=1)
    store.authenticate(raw)  # spend the single allowed request
    report = build_doctor_report(
        mode="in_process", key_store=store, registered_commands=1, probe_backends=False
    )
    check = _checks_by_name(report)["keys.quota"]
    assert check.ok is False and check.severity == "warn"
    assert "quota-exhausted" in check.detail


def test_env_key_auth_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_API_KEY", "root")
    report = build_doctor_report(
        mode="in_process", key_store=ApiKeyStore(), registered_commands=1, probe_backends=False
    )
    assert _checks_by_name(report)["keys.auth"].ok is True


# ---- SDK leg ----------------------------------------------------------------


def test_sdk_leg_returns_doctor_report(tmp_path: Path) -> None:
    harness = Fx1Harness(state_dir=tmp_path / "state")
    report = harness.doctor(probe_backends=False)
    assert isinstance(report, DoctorReport)
    assert report.mode == "in_process"
    checks = _checks_by_name(report)
    assert checks["capacity"].detail.startswith("in-process")
    assert checks["harness.commands"].ok is True


# ---- HTTP leg ----------------------------------------------------------------


def _doctor_client(**kw: Any) -> TestClient:
    return TestClient(create_app(**kw), raise_server_exceptions=False)


def test_route_shape_loopback_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _doctor_client()
    resp = client.get("/harness/doctor")
    assert resp.status_code == 200
    out = resp.json()
    assert out["mode"] == "server"
    assert out["verdict"] in ("healthy", "degraded", "broken")
    assert all(set(c) == {"name", "ok", "detail", "severity"} for c in out["checks"])
    names = {c["name"] for c in out["checks"]}
    assert "config.backends" in names and "capacity" in names


def test_route_probe_flag_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _doctor_client()
    resp = client.get("/harness/doctor?probe=false")
    assert resp.status_code == 200


def test_route_requires_admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_API_KEY", "root-secret")
    client = _doctor_client()
    minted = client.post("/harness/keys", json={"name": "ro"}, headers={"X-API-Key": "root-secret"})
    assert minted.status_code == 201
    raw = minted.json()["key"]
    # a default [read, write] minted key is refused the admin surface
    resp = client.get("/harness/doctor", headers={"X-API-Key": raw})
    assert resp.status_code == 403
    # while the env (admin) credential answers
    resp = client.get("/harness/doctor", headers={"X-API-Key": "root-secret"})
    assert resp.status_code == 200


# ---- client leg -------------------------------------------------------------


def _fake_transport(payloads: dict[str, Any]):
    def _t(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        for path, (status, body) in payloads.items():
            if url.endswith(path):
                return status, {"X-Fx1-Api-Version": "1"}, json.dumps(body).encode()
        return 404, {}, b"{}"

    return _t


def test_client_leg_appends_wire_checks() -> None:
    report_payload = {
        "mode": "server",
        "checked_at": 1.0,
        "verdict": "healthy",
        "checks": [{"name": "config.backends", "ok": True, "detail": "", "severity": "error"}],
    }
    transport = _fake_transport(
        {
            "/harness/doctor": (200, report_payload),
            "/harness/version": (200, {"api_version": "1", "fx1_version": "0.4.0"}),
            "/openapi.json": (200, {"openapi": "3.1.0", "paths": {}}),
        }
    )
    client = HarnessClient("http://harness.test", transport=transport)
    report = client.doctor()
    names = {c.name for c in report.checks}
    assert {"version_contract", "openapi_spec"} <= names
    assert report.verdict == "healthy"


def test_client_leg_version_mismatch_breaks() -> None:
    report_payload = {
        "mode": "server",
        "checked_at": 1.0,
        "verdict": "healthy",
        "checks": [{"name": "config.backends", "ok": True, "detail": "", "severity": "error"}],
    }
    transport = _fake_transport(
        {
            "/harness/doctor": (200, report_payload),
            "/harness/version": (200, {"api_version": "0", "fx1_version": "0.3.0"}),
            "/openapi.json": (200, {"openapi": "3.1.0", "paths": {}}),
        }
    )
    client = HarnessClient("http://harness.test", transport=transport)
    report = client.doctor()
    check = {c.name: c for c in report.checks}["version_contract"]
    assert check.ok is False and check.severity == "error"
    assert report.verdict == "broken"


def test_client_leg_probe_flag() -> None:
    seen: list[str] = []

    def _t(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        seen.append(url)
        if "doctor" in url:
            return (
                200,
                {},
                json.dumps(
                    {
                        "mode": "server",
                        "checked_at": 0.0,
                        "verdict": "healthy",
                        "checks": [],
                    }
                ).encode(),
            )
        if "version" in url:
            return 200, {}, json.dumps({"api_version": "1"}).encode()
        return 200, {}, json.dumps({"paths": {}}).encode()

    client = HarnessClient("http://harness.test", transport=_t)
    client.doctor(probe=False)
    assert any("probe=false" in u for u in seen)


# ---- CLI --------------------------------------------------------------------


def test_cli_doctor_json_local_healthy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    cr = CliRunner().invoke(
        app, ["harness", "doctor", "--json", "--no-probe", "--state-dir", str(tmp_path / "s")]
    )
    assert cr.exit_code == 0, cr.output
    payload = json.loads(cr.output)
    assert payload["verdict"] == "healthy"
    assert payload["mode"] == "in_process"


def test_cli_doctor_table_and_broken_exit() -> None:
    cr = CliRunner().invoke(app, ["harness", "doctor", "--no-probe"])
    assert cr.exit_code == 1, cr.output
    assert "verdict: broken" in cr.output
    assert "config.backends" in cr.output


def test_cli_doctor_state_dir_with_remote_is_arg_fault() -> None:
    cr = CliRunner().invoke(
        app,
        ["harness", "doctor", "--remote", "http://x.test", "--state-dir", "/tmp/x"],
    )
    assert cr.exit_code == 2


def test_cli_doctor_remote_unreachable_exits_broken() -> None:
    cr = CliRunner().invoke(
        app,
        ["harness", "doctor", "--remote", "http://127.0.0.1:1", "--timeout", "1"],
    )
    assert cr.exit_code == 1
    assert "error:" in cr.output
