"""Current and previous API keys from files, during a grace window."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.api.http_controls import key_file_env_names, load_key_slot
from quant_fund.api.research_api import ResearchApiSettings, create_app

_CURRENT = "current-admin-key"
_PREVIOUS = "previous-admin-key"
_READ = "current-read-key"
_READ_PREVIOUS = "previous-read-key"
_OPEN = "2099-01-01T00:00:00Z"
_CLOSED = "2000-01-01T00:00:00Z"


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in key_file_env_names():
        monkeypatch.delenv(name, raising=False)
    for name in (
        "QUANT_API_KEY",
        "QUANT_API_READ_KEY",
        "RESEARCH_API_KEY",
        "RESEARCH_API_READ_KEY",
        "QUANT_API_AUDIT_LOG",
        "RESEARCH_API_AUDIT_LOG",
    ):
        monkeypatch.delenv(name, raising=False)


def _line(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")
    path.chmod(0o600)


def _arm_admin(monkeypatch: pytest.MonkeyPatch, root: Path, *, until: str | None) -> None:
    current = root / "admin"
    previous = root / "admin.previous"
    _line(current, _CURRENT)
    _line(previous, _PREVIOUS)
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(current))
    monkeypatch.setenv("QUANT_API_KEY_PREVIOUS_FILE", str(previous))
    if until is not None:
        deadline = root / "admin.until"
        _line(deadline, until)
        monkeypatch.setenv("QUANT_API_KEY_GRACE_UNTIL_FILE", str(deadline))


def test_file_replaces_the_env_value_without_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    key_path = tmp_path / "admin"
    _line(key_path, _CURRENT)
    monkeypatch.setenv("QUANT_API_KEY", "retired-env-key")
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(key_path))
    client = TestClient(app)
    assert client.get("/models", headers={"X-API-Key": "retired-env-key"}).status_code == 401
    assert client.get("/models", headers={"X-API-Key": _CURRENT}).status_code == 200
    _line(key_path, "rotated-admin-key")
    assert client.get("/models", headers={"X-API-Key": _CURRENT}).status_code == 401
    assert client.get("/models", headers={"X-API-Key": "rotated-admin-key"}).status_code == 200


def test_previous_admin_key_works_only_inside_the_grace_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    _arm_admin(monkeypatch, tmp_path, until=_OPEN)
    client = TestClient(app)
    assert client.get("/models", headers={"X-API-Key": _CURRENT}).status_code == 200
    assert client.get("/models", headers={"X-API-Key": _PREVIOUS}).status_code == 200
    posted = client.post(
        "/portfolio/optimize",
        headers={"X-API-Key": _PREVIOUS},
        json={"config_path": "../pyproject.toml"},
    )
    assert posted.status_code == 400

    closed = tmp_path / "closed"
    _line(closed, _CLOSED)
    monkeypatch.setenv("QUANT_API_KEY_GRACE_UNTIL_FILE", str(closed))
    assert client.get("/models", headers={"X-API-Key": _PREVIOUS}).status_code == 401
    assert client.get("/models", headers={"X-API-Key": _CURRENT}).status_code == 200


def test_previous_key_without_a_deadline_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    _arm_admin(monkeypatch, tmp_path, until=None)
    client = TestClient(app)
    assert client.get("/models", headers={"X-API-Key": _PREVIOUS}).status_code == 401
    assert client.get("/models", headers={"X-API-Key": _CURRENT}).status_code == 200


def test_grace_boundary_is_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    _arm_admin(monkeypatch, tmp_path, until="2026-11-02T00:00:00Z")
    at_deadline = datetime(2026, 11, 2, tzinfo=UTC)
    before = datetime(2026, 11, 1, 23, 59, 59, tzinfo=UTC)
    assert load_key_slot("QUANT_API_KEY", now=at_deadline)[1] is None
    assert load_key_slot("QUANT_API_KEY", now=before)[1] == _PREVIOUS


def test_unreadable_key_file_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear(monkeypatch)
    missing = tmp_path / "missing"
    monkeypatch.setenv("QUANT_API_KEY", _CURRENT)
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(missing))
    client = TestClient(app)
    assert client.get("/models", headers={"X-API-Key": _CURRENT}).status_code == 401
    remote = TestClient(app, client=("203.0.113.50", 9))
    assert remote.get("/models").status_code == 401


def test_previous_read_key_is_safe_methods_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    _arm_admin(monkeypatch, tmp_path, until=_OPEN)
    current = tmp_path / "read"
    previous = tmp_path / "read.previous"
    until = tmp_path / "read.until"
    _line(current, _READ)
    _line(previous, _READ_PREVIOUS)
    _line(until, _OPEN)
    monkeypatch.setenv("QUANT_API_READ_KEY_FILE", str(current))
    monkeypatch.setenv("QUANT_API_READ_KEY_PREVIOUS_FILE", str(previous))
    monkeypatch.setenv("QUANT_API_READ_KEY_GRACE_UNTIL_FILE", str(until))
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/models", headers={"X-API-Key": _READ_PREVIOUS}).status_code == 200
    denied = client.post("/portfolio/optimize", headers={"X-API-Key": _READ_PREVIOUS}, json={})
    assert denied.status_code == 403
    assert denied.json() == {"detail": "read key cannot modify"}
    assert _READ_PREVIOUS not in denied.text
    assert _CURRENT not in denied.text


def test_research_file_rotation_leaves_an_explicit_key_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    for name in ("data", "receipts", "verifier", "artifacts"):
        (tmp_path / name).mkdir()
    key_path = tmp_path / "research"
    _line(key_path, "file-research-key")
    monkeypatch.setenv("RESEARCH_API_KEY_FILE", str(key_path))
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
        api_key="explicit-research-key",
    )
    client = TestClient(create_app(settings), client=("203.0.113.51", 9))
    assert client.get("/receipts", headers={"X-API-Key": "file-research-key"}).status_code == 401
    assert (
        client.get("/receipts", headers={"X-API-Key": "explicit-research-key"}).status_code == 200
    )

    env_app = TestClient(create_app(), client=("203.0.113.52", 9))
    assert env_app.get("/receipts", headers={"X-API-Key": "file-research-key"}).status_code == 200
    _line(key_path, "rotated-research-key")
    assert env_app.get("/receipts", headers={"X-API-Key": "file-research-key"}).status_code == 401
    assert (
        env_app.get("/receipts", headers={"X-API-Key": "rotated-research-key"}).status_code == 200
    )


def test_cli_bind_accepts_the_admin_key_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    _clear(monkeypatch)
    key_path = tmp_path / "admin"
    _line(key_path, _CURRENT)
    monkeypatch.setenv("QUANT_API_KEY_FILE", str(key_path))
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    result = CliRunner().invoke(cli, ["api", "--host", "0.0.0.0", "--port", "9"])
    assert result.exit_code == 0


def test_forwarded_for_does_not_become_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    remote = TestClient(app, client=("203.0.113.53", 9))
    refused = remote.get("/models", headers={"X-Forwarded-For": "127.0.0.1"})
    assert refused.status_code == 403
