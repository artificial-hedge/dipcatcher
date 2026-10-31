"""harness_doctor lane: the diagnostic battery over both legs.

The remote leg runs against a real ``create_app`` deployment (TestClient
transport), the local leg against ``Fx1Harness`` in-process — both must
report green with no secret material in any check detail.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.serve.api import create_app
from fx1.serve.client import HarnessClient, HarnessTransportError
from fx1.serve.harness_doctor import run_local_doctor, run_remote_doctor

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_RECEIPTS = _REPO_ROOT / "receipts"


def _checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in report["checks"]}


def _transport_for(tc: TestClient) -> Any:
    """Adapt TestClient to the HarnessClient Transport callable."""

    def transport(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        parsed = urllib.parse.urlparse(url)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        req_headers = dict(headers)
        data = None
        if isinstance(payload, bytes):
            data = payload
        elif payload is not None:
            data = json.dumps(payload).encode()
            req_headers["Content-Type"] = "application/json"
        resp = tc.request(method, target, content=data, headers=req_headers)
        return resp.status_code, dict(resp.headers), resp.content

    return transport


def test_local_doctor_all_green(tmp_path: Path) -> None:
    report = run_local_doctor(receipts_dir=_REPO_RECEIPTS)
    checks = _checks(report)
    assert report["object"] == "fx1_doctor_report.v1"
    assert report["mode"] == "local"
    assert report["ok"] is True, checks
    assert checks["honesty_gate"]["ok"] is True
    assert checks["completion_roundtrip"]["ok"] is True
    assert checks["key_lifecycle"]["ok"] is True


def test_local_doctor_empty_receipts_skips(tmp_path: Path) -> None:
    report = run_local_doctor(receipts_dir=tmp_path / "absent")
    checks = _checks(report)
    assert report["ok"] is True
    assert checks["receipt_roundtrip"].get("skipped") is True


def test_local_doctor_state_dir_replay(tmp_path: Path) -> None:
    report = run_local_doctor(receipts_dir=tmp_path, state_dir=tmp_path / "durable")
    checks = _checks(report)
    assert report["ok"] is True, checks
    assert checks["state_dir_replay"]["ok"] is True


def test_local_doctor_no_secret_material() -> None:
    report = run_local_doctor(receipts_dir=_REPO_RECEIPTS)
    blob = json.dumps(report)
    # raw probe keys are fx1k_* secrets — the report may carry key record
    # ids (key_…) but never a minted raw secret.
    assert "fx1k_" not in blob


def test_remote_doctor_all_green(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FX1_API_KEY", "doctor-admin")
    app = create_app(state_dir=tmp_path / "state", receipts_dir=_REPO_RECEIPTS)
    with TestClient(app) as tc:
        transport = _transport_for(tc)
        make = lambda k: HarnessClient(  # noqa: E731
            "https://probe.local", api_key=k, transport=transport
        )
        report = run_remote_doctor(make("doctor-admin"), make)
    checks = _checks(report)
    assert report["mode"] == "remote"
    assert report["ok"] is True, checks
    assert checks["unauth_refused"]["ok"] is True
    assert checks["key_lifecycle"]["ok"] is True


def test_remote_doctor_unreachable_fails_closed() -> None:
    def dead_transport(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        raise HarnessTransportError("refused")

    client = HarnessClient("https://probe.local", api_key="k", transport=dead_transport)
    report = run_remote_doctor(client, lambda k: client)
    assert report["ok"] is False
    assert all(c["ok"] is False for c in report["checks"])


def test_remote_doctor_open_mode_skips_auth_check(tmp_path: Path) -> None:
    """A deployment with no bootstrap key: gated routes accept anything —
    doctor reports open mode as skipped, never as a pass."""
    app = create_app(state_dir=tmp_path / "state", receipts_dir=_REPO_RECEIPTS)
    with TestClient(app) as tc:
        transport = _transport_for(tc)
        make = lambda k: HarnessClient(  # noqa: E731
            "https://probe.local", api_key=k, transport=transport
        )
        report = run_remote_doctor(make(None), make)
    checks = _checks(report)
    assert checks["unauth_refused"].get("skipped") is True
    assert checks["key_lifecycle"].get("skipped") is True
