"""Invalid header bytes must take the ordinary authentication failure path."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.api.research_api import ResearchApiSettings, create_app


@pytest.fixture(params=["harness", "research"])
def client(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("QUANT_API_KEY", "test-api-key")
    target = app
    if request.param == "research":
        target = create_app(
            ResearchApiSettings(
                data_root=tmp_path / "data",
                receipts_dir=tmp_path / "receipts",
                verifier_dir=tmp_path / "verifier",
                artifacts_dir=tmp_path / "artifacts",
                api_key="test-api-key",
            )
        )
    with TestClient(target, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize("provided", [b"\xff", b"test-api-key\xe9", b"\xc3\xa9", b"wrong", b""])
def test_invalid_api_key_returns_401(client: TestClient, provided: bytes) -> None:
    response = client.get("/openapi.json", headers=[(b"X-API-Key", provided)])
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid or missing X-API-Key"}
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_valid_api_key_still_authenticates(client: TestClient) -> None:
    assert client.get("/openapi.json", headers={"X-API-Key": "test-api-key"}).status_code == 200
