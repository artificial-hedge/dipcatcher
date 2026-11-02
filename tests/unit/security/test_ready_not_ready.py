"""Lab GET /ready 503 is the fail-closed doctor signal, and ZAP records that."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.pipeline.doctor import doctor

_ROOT = Path(__file__).resolve().parents[3]
_RULES = _ROOT / "deploy" / "security" / "zap-rules.tsv"
_READY_URLS = (
    "http://127.0.0.1:8000/ready",
    "http://127.0.0.1:8000/ready/",
    "http://127.0.0.1:8000/ready?config_path=configs/research.yaml",
    "http://127.0.0.1:8000/ready?config_path=http%3A%2F%2Fwww.google.com%2Fsearch%3Fq%3DZAP",
    "https://api.example:8443/ready/?config_path=configs/research.yaml",
)
_OTHER_URLS = (
    "http://127.0.0.1:8000/health",
    "http://127.0.0.1:8000/portfolio/optimize",
    "http://127.0.0.1:8000/readyz",
    "http://127.0.0.1:8000/ready/extra",
    "http://127.0.0.1:8010/receipts",
)


def _ready_patterns() -> list[re.Pattern[str]]:
    """Parse the committed rule file the way zap_common.load_config does."""
    patterns: list[re.Pattern[str]] = []
    for line in _RULES.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        assert line.count("\t") >= 2, line
        key, level, optional = line.rstrip().split("\t", 2)
        assert key == "100000"
        assert level == "OUTOFSCOPE"
        patterns.append(re.compile(optional))
    assert patterns
    return patterns


def test_empty_lake_ready_is_503_with_the_admin_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    config = root / "research.yaml"
    config.write_text(
        f"runtime:\n  mode: research\ndata:\n  root: {root.as_posix()}\n  source: synthetic\n",
        encoding="utf-8",
    )
    report = doctor(str(config))
    assert report["data_manifest"] == "missing"
    assert report["research_receipt"] == "missing"

    api = importlib.import_module("quant_fund.api.app")
    monkeypatch.setattr(api, "_CONFIGS_DIR", root)
    monkeypatch.setenv("QUANT_API_KEY", "ready-admin-key")
    response = TestClient(api.app).get(
        "/ready",
        params={"config_path": "research.yaml"},
        headers={"X-API-Key": "ready-admin-key"},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["data_manifest"] is False
    assert body["checks"]["research_receipt"] is False
    assert "version" not in body
    assert "python_package" not in body
    assert "version" not in body["report"]
    assert "python_package" not in body["report"]
    assert "Traceback" not in response.text
    assert "ready-admin-key" not in response.text


def test_zap_ready_rule_matches_only_the_ready_url() -> None:
    patterns = _ready_patterns()
    for url in _READY_URLS:
        assert any(pattern.match(url) for pattern in patterns), url
    for url in _OTHER_URLS:
        assert all(pattern.match(url) is None for pattern in patterns), url
