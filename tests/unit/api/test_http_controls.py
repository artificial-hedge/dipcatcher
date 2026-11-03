"""Body cap, token buckets, closed CORS, and the generic 500 body."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.api.http_controls import (
    keys_match,
    loopback_rate_config,
    rate_backend_env_names,
    rate_config,
    reset_rate_gates,
)
from quant_fund.api.research_api import ResearchApiSettings, create_app


@pytest.fixture(autouse=True)
def _clear_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_rate_gates()
    for name in rate_backend_env_names():
        monkeypatch.delenv(name, raising=False)
    # Hold the clock so a slow handler cannot refill the bucket mid-test.
    monkeypatch.setattr("quant_fund.api.http_controls.time.monotonic", lambda: 1_000.0)


def _research(tmp_path: Path, api_key: str | None = None) -> TestClient:
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
        api_key=api_key,
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


def test_keys_match_is_equal_for_same_bytes_and_rejects_other_lengths() -> None:
    assert keys_match("alpha", "alpha")
    assert not keys_match("alpha", "alphabet")
    assert not keys_match("alpha", "beta")
    assert not keys_match("", "alpha")


def test_loopback_uses_its_own_higher_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    for name in (
        "QUANT_API_RATE_PER_SEC",
        "QUANT_API_RATE_BURST",
        "QUANT_API_LOOPBACK_RATE_PER_SEC",
        "QUANT_API_LOOPBACK_RATE_BURST",
        "RESEARCH_API_LOOPBACK_RATE_PER_SEC",
        "RESEARCH_API_LOOPBACK_RATE_BURST",
    ):
        monkeypatch.delenv(name, raising=False)
    assert rate_config("QUANT_API") == (10.0, 30.0)
    assert loopback_rate_config("QUANT_API") == (100.0, 1000.0)
    monkeypatch.setenv("QUANT_API_RATE_PER_SEC", "1")
    monkeypatch.setenv("QUANT_API_RATE_BURST", "1")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/models").status_code == 200
    assert client.get("/models").status_code == 200
    monkeypatch.setenv("QUANT_API_LOOPBACK_RATE_PER_SEC", "1")
    monkeypatch.setenv("QUANT_API_LOOPBACK_RATE_BURST", "1")
    reset_rate_gates()
    assert client.get("/models").status_code == 200
    limited = client.get("/models")
    assert limited.status_code == 429
    assert limited.json() == {"detail": "rate limit exceeded"}
    assert client.get("/health").status_code == 200
    remote = TestClient(app, client=("198.51.100.70", 9), raise_server_exceptions=False)
    assert remote.get("/models").status_code == 403
    assert remote.get("/models").status_code == 429
    monkeypatch.setenv("QUANT_API_LOOPBACK_RATE_PER_SEC", "off")
    reset_rate_gates()
    assert client.get("/models").status_code == 200
    assert client.get("/models").status_code == 200
    for name in ("data", "receipts", "verifier", "artifacts"):
        (tmp_path / name).mkdir()
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
    )
    monkeypatch.setenv("RESEARCH_API_LOOPBACK_RATE_PER_SEC", "1")
    monkeypatch.setenv("RESEARCH_API_LOOPBACK_RATE_BURST", "1")
    reset_rate_gates()
    research = TestClient(create_app(settings), raise_server_exceptions=False)
    assert research.get("/receipts").status_code == 200
    assert research.get("/receipts").status_code == 429


def test_remote_ip_and_key_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_API_KEY", "lab-key")
    monkeypatch.setenv("QUANT_API_RATE_PER_SEC", "1")
    monkeypatch.setenv("QUANT_API_RATE_BURST", "1")
    first = TestClient(app, client=("198.51.100.10", 9), raise_server_exceptions=False)
    second = TestClient(app, client=("198.51.100.11", 9), raise_server_exceptions=False)
    headers = {"X-API-Key": "lab-key"}
    assert first.get("/models", headers=headers).status_code == 200
    # Same key, different address: the key bucket is already empty.
    assert second.get("/models", headers=headers).status_code == 429
    # Same address, public liveness stays unlimited.
    assert first.get("/health").status_code == 200
    # A second models call from the first address is also limited.
    assert first.get("/models", headers=headers).status_code == 429


def test_unparsed_rate_config_keeps_the_default_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    monkeypatch.setenv("QUANT_API_RATE_PER_SEC", "nope")
    client = TestClient(app, client=("198.51.100.21", 9), raise_server_exceptions=False)
    statuses = [client.get("/models").status_code for _ in range(31)]
    assert statuses[:30] == [403] * 30
    assert statuses[30] == 429


def test_rate_limit_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_API_RATE_PER_SEC", "off")
    client = TestClient(app, client=("198.51.100.20", 9), raise_server_exceptions=False)
    assert client.get("/models").status_code == 403
    assert client.get("/models").status_code == 403


def test_read_key_cannot_modify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QUANT_API_KEY", "admin-key")
    monkeypatch.setenv("QUANT_API_READ_KEY", "read-key")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"X-API-Key": "wrong-key"}).status_code == 401
    assert client.get("/models", headers={"X-API-Key": "read-key"}).status_code == 200
    assert client.get("/models", headers={"X-API-Key": "admin-key"}).status_code == 200
    denied = client.post(
        "/portfolio/optimize",
        headers={"X-API-Key": "read-key"},
        json={},
    )
    assert denied.status_code == 403
    assert denied.json() == {"detail": "read key cannot modify"}
    assert "read-key" not in denied.text
    assert "admin-key" not in denied.text

    monkeypatch.setenv("RESEARCH_API_READ_KEY", "read-key")
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
        api_key="admin-key",
    )
    for path in (
        settings.data_root,
        settings.receipts_dir,
        settings.verifier_dir,
        settings.artifacts_dir,
    ):
        path.mkdir()
    research = TestClient(
        create_app(settings), client=("198.51.100.41", 9), raise_server_exceptions=False
    )
    assert research.get("/receipts").status_code == 401
    assert research.get("/receipts", headers={"X-API-Key": "read-key"}).status_code == 200
    read_post = research.post("/receipts", headers={"X-API-Key": "read-key"})
    assert read_post.status_code == 403
    assert read_post.json() == {"detail": "read key cannot modify"}
    assert "read-key" not in read_post.text
    admin_post = research.post("/receipts", headers={"X-API-Key": "admin-key"})
    assert admin_post.status_code == 405


def test_lab_cors_default_reflects_nothing() -> None:
    response = TestClient(app).get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_lab_internal_error_hides_the_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    # The package attribute ``quant_fund.api.app`` is the FastAPI instance.
    api = importlib.import_module("quant_fund.api.app")
    monkeypatch.delenv("QUANT_API_KEY", raising=False)

    def boom(_config_path: str) -> dict[str, str]:
        raise RuntimeError("token sk-test and path /tmp/lab-secret")

    monkeypatch.setattr(api, "doctor", boom)
    response = TestClient(app, raise_server_exceptions=False).get("/doctor")
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "access-control-allow-origin" not in response.headers
    assert "sk-test" not in response.text
    assert "/tmp/lab-secret" not in response.text
    assert "Traceback" not in response.text


def test_research_reads_the_env_key_on_each_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RESEARCH_API_KEY", "first-key")
    client = TestClient(create_app(), client=("198.51.100.30", 9), raise_server_exceptions=False)
    assert client.get("/receipts", headers={"X-API-Key": "first-key"}).status_code == 200
    monkeypatch.setenv("RESEARCH_API_KEY", "second-key")
    assert client.get("/receipts", headers={"X-API-Key": "first-key"}).status_code == 401
    assert client.get("/receipts", headers={"X-API-Key": "second-key"}).status_code == 200


def test_research_remote_body_cors_and_headers(tmp_path: Path) -> None:
    client = _research(tmp_path)
    health = client.get("/health", headers={"Origin": "https://evil.example"})
    assert health.status_code == 200
    assert health.headers["X-Frame-Options"] == "DENY"
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert health.headers["Cache-Control"] == "no-store"
    assert "access-control-allow-origin" not in health.headers
    oversized = client.post(
        "/health",
        content=b"x" * (64 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "request body exceeds the 64 KiB limit"


def test_research_rate_limit_and_plain_500(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_API_RATE_PER_SEC", "1")
    monkeypatch.setenv("RESEARCH_API_RATE_BURST", "1")
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
        api_key="research-key",
    )
    built = create_app(settings)

    @built.get("/boom")
    def boom() -> dict[str, str]:
        raise RuntimeError("secret-value /srv/research/private")

    remote = TestClient(built, client=("198.51.100.40", 9), raise_server_exceptions=False)
    headers = {"X-API-Key": "research-key"}
    assert remote.get("/receipts", headers=headers).status_code == 200
    assert remote.get("/receipts", headers=headers).status_code == 429
    broken = TestClient(built, raise_server_exceptions=False).get("/boom", headers=headers)
    assert broken.status_code == 500
    assert broken.json() == {"detail": "internal error"}
    assert "secret-value" not in broken.text
    assert "/srv/research" not in broken.text
    assert "Traceback" not in broken.text
