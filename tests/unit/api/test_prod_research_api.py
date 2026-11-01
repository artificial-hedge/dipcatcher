from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.research_api import ResearchApiSettings, create_app


def test_evidence_read_failure_is_a_controlled_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    receipts_dir = tmp_path / "receipts"
    verifier_dir = tmp_path / "verifier"
    artifacts_dir = tmp_path / "artifacts"
    for directory in (data_root, receipts_dir, verifier_dir, artifacts_dir):
        directory.mkdir()
    receipt = receipts_dir / "clean.json"
    receipt.write_text('{"schema":"synthetic.test/v1"}', encoding="utf-8")
    settings = ResearchApiSettings(
        data_root=data_root,
        receipts_dir=receipts_dir,
        verifier_dir=verifier_dir,
        artifacts_dir=artifacts_dir,
    )
    original_read_bytes = Path.read_bytes

    def fail_for_receipt(path: Path) -> bytes:
        if path == receipt:
            raise OSError(f"private filesystem detail: {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_for_receipt)
    client = TestClient(create_app(settings), raise_server_exceptions=False)

    response = client.get("/receipts/clean")

    assert response.status_code == 503
    assert response.json() == {"detail": "evidence store temporarily unavailable"}
    assert "private filesystem detail" not in response.text
    assert response.headers["Cache-Control"] == "no-store"
